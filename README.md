**iMeshSegNet: Automated Tooth Labeling on 3D Intraoral Scans**
===============================================================

This repository started as the PyTorch implementation of **MeshSegNet**
(*IEEE Transactions on Medical Imaging*,
<https://ieeexplore.ieee.org/abstract/document/8984309>, and *MICCAI 2019*,
<https://link.springer.com/chapter/10.1007/978-3-030-32226-7_93>). It now holds
**iMeshSegNet**, the improved version of MeshSegNet described in
<https://ieeexplore.ieee.org/document/9775592>. Both networks label the teeth on
digitalized 3D dental surface models acquired by intraoral scanners (IOSs).

What changed
------------

* **Model**: *meshsegnet.py* is replaced by *imeshsegnet.py*. iMeshSegNet builds its
  neighbourhoods with a k-nearest-neighbour graph inside the network, so the large
  adjacency matrices (A<sub>S</sub> and A<sub>L</sub>) are no longer computed by the
  data loader.
* **Classes**: 17 classes instead of 15 (see [Labels](#labels)). The upper and lower
  arches share the same classes, so one model handles both.
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

If you need a GUI tool to read, annotate, and save VTP files, please refer to
<https://github.com/Tai-Hsien/Mesh_Labeler>.

Training data
-------------

Unfortunately, we are unable to provide the data.

*step3_training.py* reads pre-processed meshes stored as *.npz* files, using
`Npz_Mesh_Dataset` in *Mesh_dataset.py*. The training and validation lists are CSV files
with a header line followed by one *.npz* path per line. Each *.npz* file holds one mesh
downsampled to about 10,000 cells:

| Key | Shape | Type | Content |
|-----|-------|------|---------|
| points | (num_points, 3) | float32 | vertex coordinates |
| faces | (num_cells, 3) | int32 | vertex ids of each triangle |
| normals | (num_cells, 3) | float32 | cell normals |
| labels | (num_cells,) | uint8 | class of each cell (see [Labels](#labels)) |

The script that creates these *.npz* files is not part of this repository yet.

*step1_augmentation.py* and *step2_get_list.py* still augment VTP files and write the
lists for the older VTP-based `Mesh_Dataset`. They do not create *.npz* files.

Training
--------

In *step3_training.py*, please define `train_list`, `val_list`, and `model_name`. The
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

Testing
-------

*step4_test.py* tests a trained model on labeled VTP files and displays the metrics (DSC,
SEN, PPV). Please define the checkpoint, the path of the test dataset (variable
“*mesh_path*”), and the filenames according to your data.

<pre><code>uv run --no-sync python step4_test.py</code></pre>

Predicting unseen intraoral scans
---------------------------------

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
labels back to the original resolution (`--upsampling KNN` or `SVM`).

Previous MeshSegNet model
-------------------------

The original network is kept in *meshsegnet.py*. It takes the adjacency matrices
A<sub>S</sub> and A<sub>L</sub> as extra inputs. The step 3 to 6 scripts in this version
only support iMeshSegNet; the scripts and the trained 15-class models for MeshSegNet are
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
