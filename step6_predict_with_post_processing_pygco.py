import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import utils
from easy_mesh_vtk.easy_mesh_vtk import *
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix
import scipy.io as sio
import shutil
import time
#from sklearn.svm import SVC # uncomment this line if you don't install thudersvm
from thundersvm import SVC # comment this line if you don't install thudersvm
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph

if __name__ == '__main__':

    #gpu_id = utils.get_avail_gpu()
    gpu_id = 0
    torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    upsampling_method = 'SVM'
    #upsampling_method = 'KNN'

    model_path = './models'
    model_name = 'Mesh_Segementation_MeshSegNet_15_classes_60samples_best.tar'

    mesh_path = './inputs'
    sample_filenames = ['upper_T0.stl'] # need to modify
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

            start_time = time.time()
            # create tmp folder
            tmp_path = './.tmp/'
            if not os.path.exists(tmp_path):
                os.makedirs(tmp_path)

            print('Predicting Sample filename: {}'.format(i_sample))
            # read image and label (annotation)
            mesh = Easy_Mesh(os.path.join(mesh_path, i_sample))

            # pre-processing: downsampling
            print('\tDownsampling...')
            target_num = 10000
            ratio = 1 - target_num/mesh.cells.shape[0] # calculate ratio
            mesh_d = Easy_Mesh(os.path.join(mesh_path, i_sample))
            mesh_d.mesh_decimation(ratio)
            predicted_labels_d = np.zeros([mesh_d.cells.shape[0], 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            cell_centers = (mesh_d.cells[:, 0:3] + mesh_d.cells[:, 3:6] + mesh_d.cells[:, 6:9])/3.0
            mean_cell_centers = np.mean(cell_centers, axis=0)
            mesh_d.cells[:, 0:3] -= mean_cell_centers[0:3]
            mesh_d.cells[:, 3:6] -= mean_cell_centers[0:3]
            mesh_d.cells[:, 6:9] -= mean_cell_centers[0:3]
            mesh_d.update_cell_ids_and_points() # update object when change cells
            mesh_d.get_cell_normals() # get cell normal

            # preprae input
            cells = mesh_d.cells[:]
            normals = mesh_d.cell_attributes['Normal'][:]
            cell_ids = mesh_d.cell_ids[:]
            points = mesh_d.points[:]
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
            #X = (X-np.ones((X.shape[0], 1))*np.mean(X, axis=0)) / (np.ones((X.shape[0], 1))*np.std(X, axis=0))

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
            mesh_d = Easy_Mesh(os.path.join(mesh_path, i_sample))
            mesh_d.mesh_decimation(ratio)
            mesh2 = Easy_Mesh()
            mesh2.cells = mesh_d.cells
            mesh2.update_cell_ids_and_points()
            mesh2.cell_attributes['Label'] = predicted_labels_d
            mesh2.to_vtp(os.path.join(output_path, '{}_d_predicted.vtp'.format(i_sample[:-4])))

            # refinement
            print('\tRefining by pygco...')
            round_factor = 100
            patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6

            # unaries
            unaries = -round_factor * np.log10(patch_prob_output)
            unaries = unaries.astype(np.int32)
            unaries = unaries.reshape(-1, num_classes)

            # parawise
            pairwise = (1 - np.eye(num_classes, dtype=np.int32))

            #edges
            mesh_d.get_cell_normals()
            normals = mesh_d.cell_attributes['Normal'][:]
            cells = mesh_d.cells[:]
            barycenters = (cells[:, 0:3] + cells[:, 3:6] + cells[:, 6:9])/3.0

            lambda_c = 30
            edges = np.empty([1, 3], order='C')
            for i_node in range(cells.shape[0]):
                # Find neighbors
                nei = np.sum(np.isin(cell_ids, cell_ids[i_node, :]), axis=1)
                nei_id = np.where(nei==2)
                for i_nei in nei_id[0][:]:
                    if i_node < i_nei:
                        cos_theta = np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])/np.linalg.norm(normals[i_node, 0:3])/np.linalg.norm(normals[i_nei, 0:3])
                        if cos_theta >= 1.0:
                            cos_theta = 0.9999
                        theta = np.arccos(cos_theta)
                        phi = np.linalg.norm(barycenters[i_node, :] - barycenters[i_nei, :])
                        if theta > np.pi/2.0:
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -math.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
                        else:
                            beta = 1 + np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3]))
                            edges = np.concatenate((edges, np.array([i_node, i_nei, -beta*math.log10(theta/np.pi)*phi]).reshape(1, 3)), axis=0)
            edges = np.delete(edges, 0, 0)
            edges[:, 2] *= lambda_c*round_factor
            edges = edges.astype(np.int32)

            refine_labels = cut_from_graph(edges, unaries, pairwise)
            refine_labels = refine_labels.reshape([-1, 1])

            # output refined result
            mesh2 = Easy_Mesh()
            mesh2.cells = mesh_d.cells
            mesh2.update_cell_ids_and_points()
            mesh2.cell_attributes['Label'] = refine_labels
            mesh2.to_vtp(os.path.join(output_path, '{}_d_predicted_refined_pygco.vtp'.format(i_sample[:-4])))

            # upsampling
            print('\tUpsampling...')
            if mesh.cells.shape[0] > 100000:
                target_num = 100000 # set max number of cells
                ratio = 1 - target_num/mesh.cells.shape[0] # calculate ratio
                mesh.mesh_decimation(ratio)
                print('Original contains too many cells, simpify to {} cells'.format(mesh.cells.shape[0]))

            fine_cells = mesh.cells

            if upsampling_method == 'SVM':
                #clf = SVC(kernel='rbf', gamma='auto', probability=True, gpu_id=gpu_id)
                clf = SVC(kernel='rbf', gamma='auto', gpu_id=gpu_id)
                # train SVM
                clf.fit(cells, np.ravel(refine_labels))
                fine_labels = clf.predict(fine_cells)
                fine_labels = fine_labels.reshape([mesh.cells.shape[0], 1])
                #fine_prob = clf.predict_proba(fine_cells)
            elif upsampling_method == 'KNN':
                neigh = KNeighborsClassifier(n_neighbors=3)
                # train KNN
                neigh.fit(cells, np.ravel(refine_labels))
                fine_labels = neigh.predict(fine_cells)
                fine_labels = fine_labels.reshape([mesh.cells.shape[0], 1])

            mesh2 = Easy_Mesh()
            mesh2.cells = mesh.cells
            mesh2.update_cell_ids_and_points()
            mesh2.cell_attributes['Label'] = fine_labels
            mesh2.to_vtp(os.path.join(output_path, '{}_predicted_refined.vtp'.format(i_sample[:-4])))

            #remove tmp folder
            shutil.rmtree(tmp_path)

            end_time = time.time()
            print('Sample filename: {} completed'.format(i_sample))
            print('\tcomputing time: {0:.2f} sec'.format(end_time-start_time))
