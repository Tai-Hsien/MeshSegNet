import os
import numpy as np
import torch
import torch.nn as nn
from imeshsegnet import *
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
import utils

if __name__ == '__main__':

    gpu_id = utils.get_avail_gpu()
    torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    model_path = './models'
    model_name = 'iMeshSegNet_17_classes_best.tar'  # need to define

    mesh_path = './'  # need to define
    sample_filenames = ['Example.stl']  # need to define
    output_path = './outputs'
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    num_classes = 17
    num_channels = 15

    # set model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iMeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    # load trained model
    checkpoint = torch.load(os.path.join(model_path, model_name), map_location='cpu', weights_only=True)
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
            vertices = mesh_d.vertices
            mean_center = mesh_d.center_of_mass()
            vertices -= mean_center

            ids = np.array(mesh_d.cells)
            cells = vertices[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

            # compute cell normals; direction vectors are unaffected by the translation
            mesh_d.compute_normals()
            normals = mesh_d.celldata['Normals'].copy()

            # barycenters from shifted vertices
            barycenters = vertices[ids].mean(axis=1).astype('float32')

            #normalized data
            maxs = vertices.max(axis=0)
            mins = vertices.min(axis=0)
            means = vertices.mean(axis=0)
            stds = vertices.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i]      # point 1
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i]  # point 2
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i]  # point 3
                barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
                normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))

            # numpy -> torch.tensor
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)

            tensor_prob_output = model(X).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output downsampled predicted labels
            mesh2 = mesh_d.clone()
            mesh2.celldata['Label'] = predicted_labels_d
            vedo.write(mesh2, os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

            print('Sample filename: {} completed'.format(i_sample))
