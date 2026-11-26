import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from imeshsegnet import *
import vedo
from losses_and_metrics_for_mesh import *
import time
from sklearn.svm import SVC  # replace with: from thundersvm import SVC
from sklearn.neighbors import KNeighborsClassifier
from pygco import cut_from_graph
import utils

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='iMeshSegNet prediction with pygco post-processing')
    parser.add_argument('--input', '-i', nargs='+', required=True, help='Input mesh file(s) (e.g. *.stl *.vtp)')
    parser.add_argument('--model', default='models/iMeshSegNet_17_classes_best.tar',
                        help='Path to model checkpoint (default: models/iMeshSegNet_17_classes_best.tar)')
    parser.add_argument('--output', '-o', default='outputs',
                        help='Output folder for all products (default: outputs/). Must be a folder, not a file.')
    parser.add_argument('--num-classes', type=int, default=17)
    parser.add_argument('--num-channels', type=int, default=15)
    parser.add_argument('--upsampling', choices=['KNN', 'SVM'], default='KNN',
                        help='Upsampling method (default: KNN)')
    args = parser.parse_args()

    if os.path.exists(args.output) and not os.path.isdir(args.output):
        parser.error('--output must be a folder, not a file: {}'.format(args.output))
    os.makedirs(args.output, exist_ok=True)

    if torch.cuda.is_available():
        torch.cuda.set_device(utils.get_avail_gpu())

    upsampling_method = args.upsampling
    num_classes = args.num_classes
    num_channels = args.num_channels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = iMeshSegNet(num_classes=num_classes, num_channels=num_channels).to(device, dtype=torch.float)

    checkpoint = torch.load(args.model, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    del checkpoint
    model = model.to(device, dtype=torch.float)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    model.eval()
    with torch.no_grad():
        for input_path in args.input:

            start_time = time.time()
            sample_name = os.path.splitext(os.path.basename(input_path))[0]
            output_dir = args.output
            output_file = os.path.join(output_dir, '{}_predicted_refined.vtp'.format(sample_name))

            print('Predicting: {}'.format(input_path))
            mesh = vedo.load(input_path)

            # downsampling
            print('\tDownsampling...')
            target_num = 10000
            if mesh.ncells > target_num:
                ratio = target_num / mesh.ncells
                mesh_d = mesh.clone()
                mesh_d.decimate(fraction=ratio)
            else:
                mesh_d = mesh.clone()
            predicted_labels_d = np.zeros([mesh_d.ncells, 1], dtype=np.int32)

            # move mesh to origin
            print('\tPredicting...')
            vertices = mesh_d.vertices  # view; -= centers the mesh in-place
            mean_center = mesh_d.center_of_mass()
            vertices -= mean_center

            ids = np.array(mesh_d.cells)
            cells = vertices[ids].reshape(mesh_d.ncells, 9).astype(dtype='float32')

            mesh_d.compute_normals()
            # copy before normalization so raw unit normals are preserved for edge costs
            raw_normals = mesh_d.celldata['Normals'].copy()
            normals = raw_normals.copy()

            barycenters = vertices[ids].mean(axis=1).astype('float32')
            raw_barycenters = barycenters.copy()  # un-normalized; used for edge distances

            maxs = vertices.max(axis=0)
            mins = vertices.min(axis=0)
            means = vertices.mean(axis=0)
            stds = vertices.std(axis=0)
            nmeans = normals.mean(axis=0)
            nstds = normals.std(axis=0)

            for i in range(3):
                cells[:, i] = (cells[:, i] - means[i]) / stds[i]
                cells[:, i+3] = (cells[:, i+3] - means[i]) / stds[i]
                cells[:, i+6] = (cells[:, i+6] - means[i]) / stds[i]
                barycenters[:, i] = (barycenters[:, i] - mins[i]) / (maxs[i] - mins[i])
                normals[:, i] = (normals[:, i] - nmeans[i]) / nstds[i]

            X = np.column_stack((cells, barycenters, normals))
            X = X.transpose(1, 0)
            X = X.reshape([1, X.shape[0], X.shape[1]])
            X = torch.from_numpy(X).to(device, dtype=torch.float)

            tensor_prob_output = model(X).to(device, dtype=torch.float)
            patch_prob_output = tensor_prob_output.cpu().numpy()

            for i_label in range(num_classes):
                predicted_labels_d[np.argmax(patch_prob_output[0, :], axis=-1) == i_label] = i_label

            mesh2 = mesh_d.clone()
            mesh2.celldata['Label'] = predicted_labels_d
            vedo.write(mesh2, os.path.join(output_dir, '{}_d_predicted.vtp'.format(sample_name)))

            # pygco refinement
            print('\tRefining by pygco...')
            round_factor = 100
            patch_prob_output[patch_prob_output < 1.0e-6] = 1.0e-6

            unaries = -round_factor * np.log10(patch_prob_output)
            unaries = unaries.astype(np.int32)
            unaries = unaries.reshape(-1, num_classes)

            pairwise = (1 - np.eye(num_classes, dtype=np.int32))

            cell_ids = np.array(mesh_d.cells)

            lambda_c = 30
            # neighbors = cells sharing exactly one edge (two vertices)
            cell_edges = np.sort(cell_ids[:, [0, 1, 1, 2, 2, 0]].reshape(-1, 2), axis=1)
            order = np.lexsort((cell_edges[:, 1], cell_edges[:, 0]))
            cell_edges = cell_edges[order]
            owner = order // 3  # cell that each sorted edge belongs to
            pairs = []
            shift = 1  # >1 only matters for edges shared by more than two cells
            while True:
                same = (cell_edges[:-shift] == cell_edges[shift:]).all(axis=1)
                if not same.any():
                    break
                pairs.append(np.column_stack((owner[:-shift][same], owner[shift:][same])))
                shift += 1
            pairs = np.sort(np.concatenate(pairs), axis=1)
            pairs, num_shared = np.unique(pairs, axis=0, return_counts=True)
            pairs = pairs[(num_shared == 1) & (pairs[:, 0] != pairs[:, 1])]
            i_node, i_nei = pairs[:, 0], pairs[:, 1]

            n_dot = (raw_normals[i_node] * raw_normals[i_nei]).sum(axis=1)
            cos_theta = (n_dot / np.linalg.norm(raw_normals[i_node], axis=1) /
                         np.linalg.norm(raw_normals[i_nei], axis=1))
            cos_theta = np.clip(cos_theta, -1.0, 0.9999)
            theta = np.maximum(np.arccos(cos_theta), 1.0e-6)  # guard log10(0)
            phi = np.linalg.norm(raw_barycenters[i_node] - raw_barycenters[i_nei], axis=1)
            beta = np.where(theta > np.pi / 2.0, 1.0, 1 + np.abs(n_dot))
            w = -beta * np.log10(theta / np.pi) * phi
            edges = np.column_stack((i_node, i_nei, w)).astype(np.float64)
            edges[:, 2] *= lambda_c * round_factor
            edges = edges.astype(np.int32)

            refine_labels = cut_from_graph(edges, unaries, pairwise)
            refine_labels = refine_labels.reshape([-1, 1])

            mesh3 = mesh_d.clone()
            mesh3.celldata['Label'] = refine_labels
            vedo.write(mesh3, os.path.join(output_dir, '{}_d_predicted_refined.vtp'.format(sample_name)))

            # upsampling to original resolution
            print('\tUpsampling...')
            if mesh.ncells > 50000:
                ratio = 50000 / mesh.ncells
                mesh.decimate(fraction=ratio)
                print('\tOriginal contains too many cells, simplified to {} cells'.format(mesh.ncells))

            # both sets of barycenters must be in the same centered coordinate frame
            barycenters = raw_barycenters
            orig_ids = np.array(mesh.cells)
            fine_barycenters = (np.array(mesh.vertices) - mean_center)[orig_ids].mean(axis=1)

            if upsampling_method == 'SVM':
                clf = SVC(kernel='rbf', gamma='auto')
                clf.fit(barycenters, np.ravel(refine_labels))
                fine_labels = clf.predict(fine_barycenters).reshape(-1, 1)
            elif upsampling_method == 'KNN':
                neigh = KNeighborsClassifier(n_neighbors=3, weights='distance')
                neigh.fit(barycenters, np.ravel(refine_labels))
                fine_labels = neigh.predict(fine_barycenters).reshape(-1, 1)

            mesh.celldata['Label'] = fine_labels
            vedo.write(mesh, output_file)

            end_time = time.time()
            print('{} completed'.format(input_path))
            print('\tcomputing time: {0:.2f} sec'.format(end_time - start_time))
