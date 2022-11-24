import os
import numpy as np
import torch
import torch.nn as nn
from meshsegnet import *
import utils
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *
from scipy.spatial import distance_matrix

if __name__ == '__main__':

    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)

    model_path = './models'
    model_name = 'MeshSegNet_Max_15_classes_72samples_lr1e-2_best.tar'

    mesh_path = '' # need to define
    test_list = pd.read_csv('test_list_1.csv')['Test ID'].values # need to change the test_list by users if you have your own test samples
    test_mesh_filename = 'Sample_0{0}_d.vtp'
    test_path = './test'
    if not os.path.exists(test_path):
        os.mkdir(test_path)

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

    # Testing
    dsc = []
    sen = []
    ppv = []

    print('Testing')
    model.eval()
    with torch.no_grad():
        for i_sample in test_list:

            print('Predicting Sample filename: {}'.format(test_mesh_filename.format(i_sample)))
            # read image and label (annotation)
            mesh = vedo.load(os.path.join(mesh_path, test_mesh_filename.format(i_sample)))
            labels = mesh.celldata['Label'].astype('int32').reshape(-1, 1)
            predicted_labels = np.zeros(labels.shape)

            # move mesh to origin
            points = mesh.points()
            mean_cell_centers = mesh.center_of_mass()
            points[:, 0:3] -= mean_cell_centers[0:3]

            ids = np.array(mesh.faces())
            cells = points[ids].reshape(mesh.ncells, 9).astype(dtype='float32')

            # calculate normals
            mesh.compute_normals()
            normals = mesh.celldata['Normals']

            # move mesh to origin
            barycenters = mesh.cell_centers() # don't need to copy
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
                barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i]-mins[i])
                normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

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
                predicted_labels[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output predicted labels
            mesh2 = mesh.clone()
            mesh2.celldata['Label'] = predicted_labels
            vedo.write(mesh2, os.path.join(test_path, 'Sample_{}_predicted.vtp'.format(i_sample)), binary=True)

            # convert predict result and label to one-hot maps
            tensor_predicted_labels = torch.from_numpy(predicted_labels)
            tensor_test_labels = torch.from_numpy(labels)
            tensor_predicted_labels = tensor_predicted_labels.long()
            tensor_test_labels = tensor_test_labels.long()
            one_hot_predicted_labels = nn.functional.one_hot(tensor_predicted_labels[:, 0], num_classes=num_classes)
            one_hot_labels = nn.functional.one_hot(tensor_test_labels[:, 0], num_classes=num_classes)

            # calculate DSC
            i_dsc = DSC(one_hot_predicted_labels, one_hot_labels)
            i_sen = SEN(one_hot_predicted_labels, one_hot_labels)
            i_ppv = PPV(one_hot_predicted_labels, one_hot_labels)
            dsc.append(i_dsc)
            sen.append(i_sen)
            ppv.append(i_ppv)
            #print('\tLabel 1: {}; Label 2: {}'.format(dsc[0], dsc[1]))

    dsc = np.asarray(dsc)
    sen = np.asarray(sen)
    ppv = np.asarray(ppv)

    # output all DSCs
    all_dsc = pd.DataFrame(data=dsc, index=test_list, columns=['label {}'.format(i) for i in range(1, num_classes)])
    all_sen = pd.DataFrame(data=sen, index=test_list, columns=['label {}'.format(i) for i in range(1, num_classes)])
    all_ppv = pd.DataFrame(data=ppv, index=test_list, columns=['label {}'.format(i) for i in range(1, num_classes)])
    print(all_dsc)
    print(all_dsc.describe())
    print(all_sen)
    print(all_sen.describe())
    print(all_ppv)
    print(all_ppv.describe())
    all_dsc.to_csv(os.path.join(test_path, 'test_DSC_report_fold_1.csv'), header=True, index=True)
    all_sen.to_csv(os.path.join(test_path, 'test_SEN_report_fold_1.csv'), header=True, index=True)
    all_ppv.to_csv(os.path.join(test_path, 'test_PPV_report_fold_1.csv'), header=True, index=True)
