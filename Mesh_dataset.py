from torch.utils.data import Dataset
import pandas as pd
import torch
import numpy as np
from vedo import *

class Mesh_Dataset(Dataset):
    def __init__(self, data_list_path, num_classes=17, patch_size=9000):
        """
        Args:
            h5_path (string): Path to the txt file with h5 files.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_list = pd.read_csv(data_list_path)
        self.num_classes = num_classes
        self.patch_size = patch_size

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        i_mesh = self.data_list.iloc[idx, 0] #vtk file name

        # read vtk
        mesh = load(i_mesh)
        # FDI_label = mesh.celldata['FDI_label']#.astype('int32').reshape(-1, 1)
        labels = np.zeros([mesh.ncells], dtype=np.int32)
        FDI2label = {11: 1, 12: 2, 13: 3, 14: 4, 15: 5, 16: 6, 17: 7, 18: 8,
                     21: 9, 22: 10, 23: 11, 24: 12, 25: 13, 26: 14, 27: 15, 28: 16,
                     31: 1, 32: 2, 33: 3, 34: 4, 35: 5, 36: 6, 37: 7, 38: 8,
                     41: 9, 42: 10, 43: 11, 44: 12, 45: 13, 46: 14, 47: 15, 48: 16}
        for i_label in list(FDI2label.keys()):
            labels[mesh.celldata['FDI_label']==i_label] = FDI2label[i_label]

        X = np.column_stack((mesh.celldata['Input_Features'][:, 0:9], mesh.celldata['Normalized_Barycenters'], mesh.celldata['Normalized_Normals']))
        Y = labels.reshape(-1, 1)

        # initialize batch of input and label
        X_train = np.zeros([self.patch_size, X.shape[1]], dtype='float32')
        Y_train = np.zeros([self.patch_size, Y.shape[1]], dtype='int32')

        # calculate number of valid cells (tooth instead of gingiva)
        positive_idx = np.argwhere(labels>0)[:, 0] #tooth idx
        negative_idx = np.argwhere(labels==0)[:, 0] # gingiva idx

        num_positive = len(positive_idx) # number of selected tooth cells

        if num_positive > self.patch_size: # all positive_idx in this patch
            positive_selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
            selected_idx = positive_selected_idx
        else:   # patch contains all positive_idx and some negative_idx
            num_negative = self.patch_size - num_positive # number of selected gingiva cells
            positive_selected_idx = np.random.choice(positive_idx, size=num_positive, replace=False)
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative, replace=False)
            selected_idx = np.concatenate((positive_selected_idx, negative_selected_idx))

        selected_idx = np.sort(selected_idx, axis=None)

        X_train[:] = X[selected_idx, :]
        Y_train[:] = Y[selected_idx, :]

        X_train = X_train.transpose(1, 0)
        Y_train = Y_train.transpose(1, 0)

        sample = {'cells': torch.from_numpy(X_train), 'labels': torch.from_numpy(Y_train)}

        return sample

class Npz_Mesh_Dataset(Dataset):
    """
    Reads pre-processed meshes stored as .npz files. data_list_path is a csv file with a header
    line followed by one .npz path per line. Each .npz file holds one mesh (about 10,000 cells):
        points  (num_points, 3) float32  vertex coordinates
        faces   (num_cells, 3)  int32    vertex ids of each triangle
        normals (num_cells, 3)  float32  cell normals
        labels  (num_cells,)    uint8    class of each cell

    Labels (17 classes; upper and lower arches share the classes), from the FDI tooth numbers:
        0      gingiva
        1..8   FDI 11..18 (upper right) and 31..38 (lower left)
        9..16  FDI 21..28 (upper left) and 41..48 (lower right)
    i.e. the same mapping as FDI2label in Mesh_Dataset.

    Features are computed exactly like step5/step6 inference.
    augment=True adds a random rotation (about x, y and z, each with probability 0.5, +-180 degrees).
    """
    def __init__(self, data_list_path, patch_size=9000, augment=False):
        self.data_list = pd.read_csv(data_list_path)
        self.patch_size = patch_size
        self.augment = augment

    def __len__(self):
        return self.data_list.shape[0]

    def __getitem__(self, idx):
        data = np.load(self.data_list.iloc[idx, 0])
        vertices, ids, normals, labels = data['points'], data['faces'], data['normals'], data['labels'].astype('int32')

        if self.augment:
            T = np.eye(3, dtype='float32')
            for axis in range(3):
                if np.random.rand() < 0.5:
                    angle = np.random.uniform(-np.pi, np.pi)
                    c, s = np.cos(angle), np.sin(angle)
                    i, j = [a for a in range(3) if a != axis]
                    R = np.eye(3, dtype='float32')
                    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
                    T = R @ T
            vertices = vertices @ T.T
            normals = normals @ T.T

        # same feature computation as step5/step6
        vertices = vertices - vertices.mean(axis=0)
        cells = vertices[ids].reshape(-1, 9)
        barycenters = vertices[ids].mean(axis=1)
        cells = (cells - np.tile(vertices.mean(axis=0), 3)) / np.tile(vertices.std(axis=0), 3)
        barycenters = (barycenters - vertices.min(axis=0)) / (vertices.max(axis=0) - vertices.min(axis=0))
        normals = (normals - normals.mean(axis=0)) / normals.std(axis=0)
        X = np.column_stack((cells, barycenters, normals)).astype('float32')

        # patch: all tooth cells first, then fill with gingiva cells
        positive_idx = np.argwhere(labels > 0)[:, 0]
        negative_idx = np.argwhere(labels == 0)[:, 0]
        if len(positive_idx) > self.patch_size:
            selected_idx = np.random.choice(positive_idx, size=self.patch_size, replace=False)
        else:
            num_negative = self.patch_size - len(positive_idx)
            if len(negative_idx) == 0:  # partial scan without gingiva: fill with repeated tooth cells
                negative_idx = positive_idx
            negative_selected_idx = np.random.choice(negative_idx, size=num_negative,
                                                     replace=num_negative > len(negative_idx))
            selected_idx = np.concatenate((positive_idx, negative_selected_idx))
        selected_idx = np.sort(selected_idx, axis=None)

        return {'cells': torch.from_numpy(X[selected_idx].T.copy()),
                'labels': torch.from_numpy(labels[selected_idx].reshape(1, -1))}


if __name__ == '__main__':
    import time
    dataset = Mesh_Dataset('./train_list.csv')
    start_time = time.time()
    print(dataset.__getitem__(0))
    print(dataset.__getitem__(0)['cells'].shape)
    end_time = time.time()
    total_time = end_time - start_time
    print("---Label has been refined: {} seconds ---".format(total_time))
