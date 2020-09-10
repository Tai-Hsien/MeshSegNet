from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from easy_mesh_vtk.easy_mesh_vtk import *
from scipy.spatial import distance_matrix

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=15, patch_size=7000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path, header=None)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        i_mesh = self.data_list.iloc[idx][0] #vtk file name
        
        # read vtk
        mesh = Easy_Mesh(i_mesh)
        labels = mesh.cell_attributes['Label'].astype('int32')
        
        #create one-hot map
#        label_map = np.zeros([mesh.cells.shape[0], self.num_classes], dtype='int32')
#        label_map = np.eye(self.num_classes)[labels]
#        label_map = label_map.reshape([len(labels), self.num_classes])
        
        # move mesh to origin
        cell_centers = (mesh.cells[:, 0:3] + mesh.cells[:, 3:6] + mesh.cells[:, 6:9])/3.0
        mean_cell_centers = np.mean(cell_centers, axis=0)
        mesh.cells[:, 0:3] -= mean_cell_centers[0:3]
        mesh.cells[:, 3:6] -= mean_cell_centers[0:3]
        mesh.cells[:, 6:9] -= mean_cell_centers[0:3]
        mesh.update_cell_ids_and_points() # update object when change cells
        mesh.get_cell_normals() # get cell normal
        
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
        Y = labels
        
        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')
        S1 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        S2 = np.zeros([self.patch_size, self.patch_size], dtype='float32')
        
        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx
        
        num_positive = len(positive_idx) # number of selected tooth cells
        num_negative = self.patch_size - num_positive # number of selected gingiva cells
        
        positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
        negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
        
        selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))
        selected_idx = np.sort(selected_idx, axis=None)
        
        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]
        
        # output to visualize
#        mesh2 = Easy_Mesh()
#        mesh2.cells = X_train[:, 0:9]
#        mesh2.update_cell_ids_and_points()
#        mesh2.cell_attributes['Normal'] = X_train[:, 12:15]
#        mesh2.cell_attributes['Label'] = Y_train
#        mesh2.to_vtp('tmp.vtp')
        
        D = distance_matrix(X_train[:, 9:12], X_train[:, 9:12])
        S1[D<0.1] = 1.0
        S1 = S1 / np.dot(np.sum(S1, axis=1, keepdims=True), np.ones((1, self.patch_size)))
        
        S2[D<0.2] = 1.0
        S2 = S2 / np.dot(np.sum(S2, axis=1, keepdims=True), np.ones((1, self.patch_size)))

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train),
                  'A_S': torch.from_numpy(S1), 'A_L': torch.from_numpy(S2)}
        
        return sample


if __name__ == '__main__':
    dataset = Mesh_Dataset('./train_list_1.csv')
    dataset.__getitem__(0)
    
#    dataset = Mesh_Dataset('./val_list_1.csv')
#    print(dataset.__getitem__(0))
