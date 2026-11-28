**iMeshSegNet: Automated Tooth Labeling on 3D Intraoral Scans**
===============================================================

This repository started as the PyTorch implementation of **MeshSegNet**
(*IEEE Transactions on Medical Imaging*,
<https://ieeexplore.ieee.org/abstract/document/8984309>, and *MICCAI 2019*,
<https://link.springer.com/chapter/10.1007/978-3-030-32226-7_93>). It now also
holds **iMeshSegNet**, the improved version of MeshSegNet described in
<https://ieeexplore.ieee.org/abstract/document/9789163>
(open access: <https://pmc.ncbi.nlm.nih.gov/articles/PMC10547011/pdf/nihms-1846005.pdf>). Both networks label the teeth on
digitalized 3D dental surface models acquired by intraoral scanners (IOSs).

MeshSegNet is still available (*meshsegnet.py*), but we recommend using iMeshSegNet
(*imeshsegnet.py*), which is faster and performs better.

What changed
------------

* **Model**: iMeshSegNet (*imeshsegnet.py*) is added and is now the recommended model.
  It builds its neighbourhoods with a k-nearest-neighbour graph inside the network, so
  the large adjacency matrices (A<sub>S</sub> and A<sub>L</sub>) are no longer computed
  by the data loader. MeshSegNet (*meshsegnet.py*) is still available.
* **Classes**: 17 classes instead of 15 (see [Labels](#labels)). The upper and lower
  arches share the same classes, so one model handles both.
* **Data**: *step1_preprocess.py* converts labeled VTP scans into the *.npz* files and the
  lists used for training. It replaces *step1_augmentation.py* and *step2_get_list.py*.
* **Environment**: packages are managed with [uv](https://docs.astral.sh/uv/)
  (*pyproject.toml* and *uv.lock*). *requirements.txt* is removed, and all packages are
  upgraded to versions without known vulnerabilities.
* **Training**: multi-GPU training, mixed precision, and automatic resume from the
  latest checkpoint. Training curves go to TensorBoard instead of visdom.
  *step3_1_continous_training.py* is removed because *step3_training.py* resumes by itself.
* **Speed**: the losses, the metrics, and the graph-cut post-processing are vectorized.
* **Checkpoints**: the old 15-class MeshSegNet checkpoints are removed. No pre-trained
  iMeshSegNet checkpoint is included.

Installation
------------

Install [uv](https://docs.astral.sh/uv/), then run one of

<pre><code>uv sync --extra cuda   # GPU (CUDA 12.6 build of PyTorch)
uv sync --extra cpu    # CPU only</code></pre>

Run the scripts with `uv run --no-sync python <script>`, or activate *.venv* first.

Labels
------

The 17 classes come from the FDI tooth numbers. The upper and lower arches share the
classes.

| Class | Meaning |
|-------|---------|
| 0 | gingiva |
| 1 to 8 | FDI 11 to 18 (upper right) and FDI 31 to 38 (lower left) |
| 9 to 16 | FDI 21 to 28 (upper left) and FDI 41 to 48 (lower right) |

Training data
-------------

Unfortunately, we are unable to provide the data. To train with your own data, prepare one
folder of labeled scans in VTP (VTK polygonal data) format, one scan per file. The class of
each cell (see [Labels](#labels)) is stored as cell data named `Label`:

<pre><code>data/raw/
    scan_001.vtp        # celldata['Label']: 0 = gingiva, 1..16 = teeth
    scan_002.vtp
    ...</code></pre>

Upper and lower scans go into the same folder. To read, write, and manipulate VTP files
programmingly, we use [vedo](https://github.com/marcomusy/vedo). If you need a GUI tool to
annotate and save VTP files, please refer to <https://github.com/Tai-Hsien/Mesh_Labeler>.

Step 1 Pre-processing
---------------------

<pre><code>uv run --no-sync python step1_preprocess.py --input-dir data/raw --output-dir data</code></pre>

*step1_preprocess.py* downsamples each scan to about 10,000 cells (`--num-cells`), takes the
label of each downsampled cell from the nearest original cells, and splits the scans into a
training and a validation set (`--val-size`, 0.2 by default). Use `--label-name` if your
cell data has another name. The output is

<pre><code>data/
    train/scan_001.npz ...
    val/scan_002.npz ...
    train_list.csv
    val_list.csv</code></pre>

The lists are CSV files with a header line followed by one *.npz* path per line, relative to
the folder of the list. Scans whose labels are outside 0 to 16 (`--num-classes`) are
rejected. Each *.npz* file holds one mesh:

| Key | Shape | Type | Content |
|-----|-------|------|---------|
| points | (num_points, 3) | float32 | vertex coordinates |
| faces | (num_cells, 3) | int32 | vertex ids of each triangle |
| normals | (num_cells, 3) | float32 | cell normals |
| labels | (num_cells,) | uint8 | class of each cell (see [Labels](#labels)) |

The previous *step1_augmentation.py* and *step2_get_list.py* are removed: the random
rotation is now applied during training, and the lists are written by this step.

Step 3 Training
---------------

*step3_training.py* reads the lists written by step 1 (`train_list`, `val_list`) with
`Npz_Mesh_Dataset` in *Mesh_dataset.py*. Please define `model_name`. The
number of features is 15, corresponding to cell vertices (9 elements), cell normal vector
(3 elements), and the relative position (3 elements). During training, all tooth cells
and randomly selected gingival cells form the 9,000-cell inputs, and each mesh is randomly
rotated.

<pre><code># single GPU
uv run --no-sync python step3_training.py

# multiple GPUs (here 3)
uv run --no-sync torchrun --nproc_per_node=3 step3_training.py</code></pre>

The batch size and the number of data-loading workers are per GPU. The latest checkpoint
(*latest_checkpoint.tar*) and the best one (*&lt;model_name&gt;_best.tar*) are saved in
“*./models*”. Running the same command again resumes from the latest checkpoint.
The training curves can be viewed with

<pre><code>uv run --no-sync tensorboard --logdir models/runs</code></pre>

Step 4 Testing
--------------

*step4_test.py* tests a trained model on a folder of labeled VTP files and displays the
metrics (DSC, SEN, PPV). The VTP files are labeled in the same way as the raw data of step 1
(`celldata['Label']`) and are downsampled in the same way. Step 1 does not make a test set:
keep the test scans out of *data/raw* and put them in their own folder, e.g.

<pre><code>data/test/
    scan_101.vtp
    scan_102.vtp
    ...</code></pre>

Please define the checkpoint and the folder of the test scans (variable “*mesh_path*”). The
predictions and the reports are saved in “*./test*”.

<pre><code>uv run --no-sync python step4_test.py</code></pre>

Steps 5 and 6 Predicting unseen intraoral scans
-----------------------------------------------

*step5_predict.py* predicts the tooth labeling on unseen intraoral scans. Please define the
checkpoint, the data path, and the filenames in the script.

<pre><code>uv run --no-sync python step5_predict.py</code></pre>

*step6_predict_with_post_processing_pygco.py* also refines the prediction with the
multi-label graph-cut method described in our *IEEE Transactions on Medical Imaging*
publication, implemented by the python package [pygco](https://github.com/amueller/gco_python).

<pre><code>uv run --no-sync python step6_predict_with_post_processing_pygco.py \
    --input scan.stl --model models/iMeshSegNet_17_classes_best.tar --output outputs</code></pre>

Both scripts downsample the mesh if the number of cells is larger than 10,000.
*step5_predict.py* saves the labels of the downsampled mesh. Step 6 also maps the refined
labels back to the original mesh (`--upsampling KNN` or `SVM`); a mesh with more than
50,000 cells is first reduced to 50,000 cells.

Previous MeshSegNet model
-------------------------

MeshSegNet is still available in *meshsegnet.py*. It takes the adjacency matrices
A<sub>S</sub> and A<sub>L</sub> as extra inputs. The step 3 to 6 scripts in this version
use iMeshSegNet; the scripts and the trained 15-class models for MeshSegNet are
available in the git history of this repository.

License
--------

The code is released under MIT License (see LICENSE file for details).

Citation
--------

If you find our work useful in your research, please cite:

* T.-H. Wu et al., "Two-Stage Mesh Deep Learning for Automated Tooth Segmentation and Landmark Localization on 3D Intraoral Scans," in IEEE Transactions on Medical Imaging, vol. 41, no. 11, pp. 3158-3166, Nov. 2022, doi: 10.1109/TMI.2022.3180343.
* C. Lian et al., "Deep Multi-Scale Mesh Feature Learning for Automated Labeling of Raw Dental Surfaces From 3D Intraoral Scanners," in IEEE Transactions on Medical Imaging, vol. 39, no. 7, pp. 2440-2450, July 2020, doi: 10.1109/TMI.2020.2971730.
* Lian C. et al. (2019) MeshSNet: Deep Multi-scale Mesh Feature Learning for End-to-End Tooth Labeling on 3D Dental Surfaces. In: Shen D. et al. (eds) Medical Image Computing and Computer Assisted Intervention – MICCAI 2019. MICCAI 2019. Lecture Notes in Computer Science, vol 11769. Springer, Cham. https://doi.org/10.1007/978-3-030-32226-7_93
* Wu TH. et al. (2021) Machine (Deep) Learning for Orthodontic CAD/CAM Technologies. In: Ko CC., Shen D., Wang L. (eds) Machine Learning in Dentistry. Springer, Cham. https://doi.org/10.1007/978-3-030-71881-7_10
