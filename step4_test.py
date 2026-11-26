import os
import numpy as np
import torch
import torch.nn as nn
from imeshsegnet import *
import utils
import vedo
import pandas as pd
from losses_and_metrics_for_mesh import *

if __name__ == '__main__':

    torch.cuda.set_device(utils.get_avail_gpu()) # assign which gpu will be used (only linux works)

    model_path = './models'
    model_name = 'iMeshSegNet_17_classes_best.tar'  # need to define

    mesh_path = ''  # need to define
    test_list = pd.read_csv('test_list.csv')['Test ID'].values  # need to change per dataset
    test_mesh_filename = 'Sample_0{0}_d.vtp'
    test_path = './test'
    if not os.path.exists(test_path):
        os.mkdir(test_path)

    num_classes = 17
    num_channels = 15

    FDI2label = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
                 21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
                 31: 1, 32: 2, 33: 3, 34: 4, 35: 5, 36: 6, 37: 7, 38: 8,
                 41: 9, 42: 10, 43: 11, 44: 12, 45: 13, 46: 14, 47: 15, 48: 16}

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
            labels = np.zeros([mesh.ncells], dtype=np.int32)
            for fdi, cls in FDI2label.items():
                labels[mesh.celldata['FDI_label'] == fdi] = cls
            labels = labels.reshape(-1, 1)
            predicted_labels = np.zeros(labels.shape)

            # move mesh to origin
            vertices = mesh.vertices
            mean_center = mesh.center_of_mass()
            vertices -= mean_center

            ids = np.array(mesh.cells)
            cells = vertices[ids].reshape(mesh.ncells, 9).astype(dtype='float32')

            # calculate normals
            mesh.compute_normals()
            normals = mesh.celldata['Normals'].copy()

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
                predicted_labels[np.argmax(patch_prob_output[0, :], axis=-1)==i_label] = i_label

            # output predicted labels
            mesh2 = mesh.clone()
            mesh2.celldata['Label'] = predicted_labels
            vedo.write(mesh2, os.path.join(test_path, 'Sample_{}_predicted.vtp'.format(i_sample)))

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
