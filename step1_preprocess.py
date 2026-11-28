'''
Labeled intraoral scans (.vtp) -> .npz files and csv lists for step3_training.py.

Expected raw dataset: one folder of .vtp files, one scan per file, with the class of each
cell (0 = gingiva, 1..16 = teeth, see README) stored as cell data, e.g. celldata['Label'].

    data/raw/
        scan_001.vtp
        scan_002.vtp
        ...

Each scan is downsampled to about --num-cells cells; the labels of the downsampled cells are
taken from the nearest original cells. The scans are split into a training and a validation
set, and the following files are written:

    data/
        train/scan_001.npz ...
        val/scan_002.npz ...
        train_list.csv
        val_list.csv

Usage:
    python step1_preprocess.py --input-dir data/raw --output-dir data
'''
import os
import glob
import argparse
import numpy as np
import vedo
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def preprocess(vtp_file, npz_file, label_name, num_cells, num_classes):
    mesh = vedo.load(vtp_file)
    if label_name not in mesh.celldata.keys():
        raise ValueError("{}: no cell data named '{}'".format(vtp_file, label_name))
    if any(len(c) != 3 for c in mesh.cells):
        raise ValueError('{}: not a triangle mesh'.format(vtp_file))
    labels = np.asarray(mesh.celldata[label_name]).reshape(-1)
    if labels.min() < 0 or labels.max() >= num_classes:
        raise ValueError('{}: labels must be in 0..{}, found {}..{}'.format(
            vtp_file, num_classes - 1, labels.min(), labels.max()))
    mesh_d = mesh.clone()
    if mesh.ncells > num_cells:
        mesh_d.decimate(fraction=num_cells / mesh.ncells)
        # labels of the downsampled cells: majority of the 3 nearest original cells
        neigh = KNeighborsClassifier(n_neighbors=3)
        neigh.fit(mesh.cell_centers().coordinates, labels)
        labels = neigh.predict(mesh_d.cell_centers().coordinates)
    mesh_d.compute_normals()
    np.savez(npz_file,
             points=mesh_d.vertices.astype('float32'),
             faces=np.array(mesh_d.cells, dtype='int32'),
             normals=mesh_d.celldata['Normals'].astype('float32'),
             labels=labels.astype('uint8'))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Labeled .vtp scans -> .npz files and csv lists for training')
    parser.add_argument('--input-dir', required=True, help='folder of labeled .vtp scans')
    parser.add_argument('--output-dir', default='./data', help='output folder (default: ./data)')
    parser.add_argument('--label-name', default='Label', help='name of the cell data that holds the classes (default: Label)')
    parser.add_argument('--num-classes', type=int, default=17, help='number of classes, to check the labels (default: 17)')
    parser.add_argument('--num-cells', type=int, default=10000, help='number of cells after downsampling (default: 10000)')
    parser.add_argument('--val-size', type=float, default=0.2, help='fraction of scans used for validation (default: 0.2)')
    parser.add_argument('--seed', type=int, default=42, help='random seed of the train/val split')
    args = parser.parse_args()

    vtp_files = sorted(glob.glob(os.path.join(args.input_dir, '*.vtp')))
    print('{} scans found in {}'.format(len(vtp_files), args.input_dir))
    if len(vtp_files) < 2:
        raise SystemExit('At least 2 scans are needed (one for training, one for validation).')
    train_files, val_files = train_test_split(vtp_files, test_size=args.val_size, random_state=args.seed)

    for split, files in (('train', train_files), ('val', val_files)):
        os.makedirs(os.path.join(args.output_dir, split), exist_ok=True)
        npz_files = []
        for i, vtp_file in enumerate(files):
            name = os.path.splitext(os.path.basename(vtp_file))[0]
            npz_file = os.path.join(args.output_dir, split, name + '.npz')
            preprocess(vtp_file, npz_file, args.label_name, args.num_cells, args.num_classes)
            npz_files.append('{}/{}.npz'.format(split, name))  # relative to the folder of the list
            print('[{}] {}/{} {}'.format(split, i + 1, len(files), name))
        with open(os.path.join(args.output_dir, '{}_list.csv'.format(split)), 'w') as f:
            f.write('path\n' + '\n'.join(npz_files) + '\n')
