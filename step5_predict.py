import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import utils

if __name__ == '__main__':

    gpu_id = utils.get_avail_gpu()
    torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    model_path = './models'
    model_name = 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.tar'

    mesh_path = './'  # need to define
    sample_filenames = ['Example.stl'] # need to define
    output_path = './outputs'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    num_classes = 15
    num_channels = 15

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    #cudnn
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    # Predicting
    model.eval()
    with torch.no_grad():
        for i_sample in sample_filenames:

            print('Predicting Sample filename: {}'.format(i_sample))
            mesh = vedo.load(os.path.join(mesh_path, i_sample))

            # pre-processing: downsampling
            if mesh.ncells > 10000:
                print('\tDownsampling...')
                target_num = 10000
                ratio = target_num/mesh.ncells # calculate ratio
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio)
                predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)
            else:
                mesh_d = mesh.clone()
                predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            points = mesh_d.points()
            mean_cell_centers = mesh_d.center_of_mass()
            points[:, 0:3] -= mean_cell_centers[0:3]

            ids = np.array(mesh_d.faces())
            cells = points[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

            # customized normal calculation; the vtk/vedo build-in function will change number of points
            mesh_d.compute_normals()
            normals = mesh_d.celldata['Normals']

            # move mesh to origin
            barycenters = mesh_d.cell_centers() # don't need to copy
            barycenters -= mean_cell_centers[0:3]

            #normalized data
            maxs = points.max(axis=0)
            mins = points.min(axis=0)
            means = points.mean(axis=0)
            stds = points.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i] #point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i] #point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i] #point 3
                barycenters[:,i] = (barycenters[:,i] - mins[i]) / (maxs[i]-mins[i])
                normals[:,i] = (normals[:,i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            # computing A_S and A_L
            A_S = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            A_L = np.zeros([X.shape[0], X.shape[0]], dtype='float32')
            D = distance_matrix(X[:, 9:12], X[:, 9:12])
            A_S[D<0.1] = 1.0
            A_S = A_S / np.dot(np.sum(A_S, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            A_L[D<0.2] = 1.0
            A_L = A_L / np.dot(np.sum(A_L, axis=1, keepdims=True), np.ones((1, X.shape[0])))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)
            A_S = A_S.reshape([1, A_S.shape[0], A_S.shape[1]])
            A_L = A_L.reshape([1, A_L.shape[0], A_L.shape[1]])
            A_S = torch.from_numpy(A_S).to(device, dtype=torch.float)
            A_L = torch.from_numpy(A_L).to(device, dtype=torch.float)

            tensor_prob_output = model(X, A_S, A_L).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            mesh2 = mesh_d.clone()
            mesh2.celldata['Label'] = predicted_labels_d
            vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

            print('Sample filename: {} completed'.format(i_sample))
