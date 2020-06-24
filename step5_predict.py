import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import utils
from easy_mesh_vtk import *
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix

if __name__ == '__main__':
    
    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)
      
    model_path = './models'
    model_name = 'Mesh_Segementation_MeshSegNet_15_classes_60samples_best.tar'
    
    mesh_path = ''  # need to define
    sample_filenames = ['Sample_0101_d.stl'] # need to define
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
            # read image and label (annotation)
            mesh = Easy_Mesh(os.path.join(mesh_path, i_sample))
            predicted_labels = np.zeros([mesh.cells.shape[0], 1])
        
            # move mesh to origin
            cell_centers = (mesh.cells[:, 0:3] + mesh.cells[:, 3:6] + mesh.cells[:, 6:9])/3.0
            mean_cell_centers = np.mean(cell_centers, axis=0)
            mesh.cells[:, 0:3] -= mean_cell_centers[0:3]
            mesh.cells[:, 3:6] -= mean_cell_centers[0:3]
            mesh.cells[:, 6:9] -= mean_cell_centers[0:3]
            mesh.update_cell_ids_and_points() # update object when change cells
            mesh.get_cell_normals() # get cell normal

            # preprae input
            cells = mesh.cells[:]
            normals = mesh.cell_attributes['Normal'][:]
            cell_ids = mesh.cell_ids[:]
            points = mesh.points[:]
            barycenters = (cells[:,0:3]+cells[:,3:6]+cells[:,6:9]) / 3

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
            X = (X-np.ones((X.shape[0], 1))*np.mean(X, axis=0)) / (np.ones((X.shape[0], 1))*np.std(X, axis=0))

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
                predicted_labels[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output predicted labels
            mesh = Easy_Mesh(os.path.join(mesh_path, i_sample))
            mesh2 = Easy_Mesh()
            mesh2.cells = mesh.cells
            mesh2.update_cell_ids_and_points()
            mesh2.cell_attributes['Label'] = predicted_labels
            mesh2.to_vtp(os.path.join(output_path, '{}_deployed.vtp'.format(i_sample[:-4])))
            print('Sample filename: {} completed'.format(i_sample))
