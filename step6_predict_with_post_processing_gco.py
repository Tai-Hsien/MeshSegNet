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
import matlab.engine
import shutil
import time
#from sklearn.svm import SVC
from thundersvm import SVC
from sklearn.neighbors import KNeighborsClassifier

if __name__ == '__main__':
    
    gpu_id = utils.get_avail_gpu()
    torch.cuda.set_device(gpu_id) # assign which gpu will be used (only linux works)

    upsampling_method = 'SVM'
    #upsampling_method = 'KNN'
      
    model_path = './models'
    model_name = 'Mesh_Segementation_MeshSegNet_15_classes_60samples_best.tar'
    
    mesh_path = './inputs'
    sample_filenames = ['upper_T0.stl', 'upper_T1_aligned.stl']
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
            print('\tRefining...')
            patch_prob_output[patch_prob_output<1.0e-6] = 1.0e-6
            sio.savemat(os.path.join(tmp_path, 'prob_labels_{0}.mat'.format(i_sample)), {'prob_labels': patch_prob_output[0, :]})
            sio.savemat(os.path.join(tmp_path, 'predicted_labels_{0}.mat'.format(i_sample)), {'predicted_labels': predicted_labels_d})

            mesh_d.get_cell_normals()
            cells = mesh_d.cells[:]
            cell_ids = mesh_d.cell_ids[:]
            normals = mesh_d.cell_attributes['Normal'][:]
            points = mesh_d.points[:]
            barycenters = (cells[:, 0:3] + cells[:, 3:6] + cells[:, 6:9])/3.0
            neighbor_mat = np.zeros([cells.shape[0], cells.shape[0]])

            # Calculate the second term (connection between nodes)
            i_concave = 0
            i_convex = 0
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
                            i_concave += 1
                            neighbor_mat[i_node, i_nei] = -math.log10(theta/np.pi)*phi
                        else:
                            i_convex += 1
                            beta = 1 + np.absolute(np.linalg.norm(np.dot(normals[i_node, 0:3], normals[i_nei, 0:3])))
                            neighbor_mat[i_node, i_nei] = -beta*math.log10(theta/np.pi)*phi

            sio.savemat(os.path.join(tmp_path, 'neighbor_terms_{0}.mat'.format(i_sample)),
                       {'neighbor_terms': neighbor_mat}) #output neighbor_term.mat for MATLAB

            # MATLAB graph-cut algorithm
            eng = matlab.engine.start_matlab()
            eng.tooth_segmentation_refinement_mat(tmp_path, str(i_sample), 30, 100, nargout=0) #output refine_label.mat for python

            # back to python
            refine_label_mat = sio.loadmat(os.path.join(tmp_path, 'refine_label_{0}.mat'.format(i_sample)))
            refine_labels = refine_label_mat['refine_label']
            refine_labels -= 1 #change 1-based to 0-based

            # output refined result
            mesh2 = Easy_Mesh()
            mesh2.cells = mesh_d.cells
            mesh2.update_cell_ids_and_points()
            mesh2.cell_attributes['Label'] = refine_labels
            mesh2.to_vtp(os.path.join(output_path, '{}_d_predicted_refined.vtp'.format(i_sample[:-4])))

            # upsampling
            print('\tUpsampling...')
            if mesh.cells.shape[0] > 100000:
                target_num = 100000 # set max number of cells
                ratio = 1 - target_num/mesh.cells.shape[0] # calculate ratio
                mesh.mesh_decimation(ratio)
                print('Original contains too many cells, simpify to {} cells'.format(mesh.cells.shape[0]))

            fine_cells = mesh.cells

            if upsampling_method == 'SVM':
                clf = SVC(kernel='rbf', gamma='auto', gpu_id=gpu_id)
                # train SVM
                clf.fit(cells, np.ravel(refine_labels))
                fine_labels = clf.predict(fine_cells)
                fine_labels = fine_labels.reshape([mesh.cells.shape[0], 1])
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
