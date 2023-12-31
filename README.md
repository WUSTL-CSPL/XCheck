# XCheck - 3D Printed Patient-Specific Devices Verifier

This repository hosts the source code for paper "XCheck: Verifying Integrity of 3D Printed Patient-Specific Devices via Computing Tomography". The paper has been accepted by [32nd USENIX Security Symposium, 2023](https://www.usenix.org/conference/usenixsecurity23).

XCheck is a defense-in-depth approach that leverages medical imaging to verify the integerity of patient-specific devices (PSD) produced by medical printing. Different from side-channel based defenses, XCheck directly checks the computed tomography (CT) scan of the printed device to its original design. It verfies the geometry and material of the printed device with adapted geometry comparison techniques and statistical distribution analysis. To further enhance usability, XCheck also provides an adjustable visualization scheme and a quantitative risk score. Details regarding this work can be found in the [Sec'23 paper](https://www.usenix.org/system/files/sec23fall-prepub-483-yu-zhiyuan.pdf) and project website at [https://3dxcheck.github.io/](https://3dxcheck.github.io/) (in built).

## Dependencies
XCheck is implemented and tested on **Python 3.9** with 16GB of RAM. 
The following external packages are required: **vedo**, **matplotlib**, **numpy**, **scipy**, **point-cloud-utils**, **open3d**, **pydicom**, **seaborn**, and **vtkplotter**.
We recommend running XCheck in a virtual environment (e.g. conda) with at least of 16GB of RAM space. 

```bash
conda env create -f xcheck.yml 
```

XCheck is tested on machines with the following specs with OpenGL version of 4.1:

+ NVIDIA GeForce RTX 3070 Ti (8GB VRAM) + AMD Ryzen 9 3900X + 32GB RAM

+ AMD Radeon Pro 5500M (8GB VRAM) + Intel i9-9880H + 16GB RAM

# Execution Command Explained
XCheck takes 2 volume objects in the form of a mesh (.stl, .obj, .ply, etc.), dicom series (folder containing .dcm files), or volume image (.tiff, .vti, .slc etc.), displaying a visual comparison based on voxel and ray-based analysis.

Invoke with the following arguments:

`-f1`: File path to object: Accepts volume image (.tiff, .vti, .slc etc...), directory containing Dicom series, or mesh objects (.stl, .obj, .ply etc...)

`-f2`: File path to object: Accepts volume image (.tiff, .vti, .slc etc...), directory containing Dicom series, or mesh objects (.stl, .obj, .ply etc...)

`-o`: Output name. XCheck automatically generates a directory under ./Results/ matching this name, checkpoints are stored here. 

`-etdist`: Provide a maximum error tolerance for distance

`-ets`: Provide a maximum error tolerance for delta scale.

`-etg`: Provide a maximum error tolerance for delta group.

`-etm`: Provide a maximum error tolerance for delta material.

`-t1`: Dual threshold values for feature extraction for input object. Ex: -200, 100, -500, 200

`-t2`: Dual threshold values for feature extraction for input object. Ex: -200, 100, -500, 200

`-basic`: Run basic tests, visualize results without running the entire system. Can be added after a command.

## Folder Structure Explained
The *Results* folder contains multiple sub-directories, whose names match the `-o` argument in the execution command of *run.py*. Each of the sub-directories stored intermediary files and folders related to the analysis steps, which we explain in detail as follows:

- `RayMesh`. It contains mesh objects that highlight potentially malicious regions generated by ray-based analysis. They are rendered in red in the ray-based analysis in the visualizer. In the cases that ray-based analysis does not identify any malicious region, this folder will be empty.
- `shape_feature`. The content in this folder is used only for ray-based analysis. Specifically, the *origins* sub-folder stores the coordinates of origins where rays are cast; and the *twomesh_shape* sub-folder contains the intersection points of rays and models. Subsequently, these intersection points are used to generate alpha shapes to be visualized. For the detailed process of ray-based analysis, please see Section 5.5 in the paper.
- `CTValues.txt`. This file stores the CT values of all the voxels in the CT model. These values are subsequently used for statistical material analysis.
- `gamma.txt`. This is a log of the output of gamma analysis.
- `src.pcd` and `tgt.pcd`. These are the point cloud objects after registration. The Housdorff distances can be directly calculated from these two files without having to go through the registration process again. These distances are used for subsequent voxel analysis.
- `src_dist.npy` and `tgt_dist.npy`. These are the numpy objects storing the Housdorff distance between the voxels of the CT and design models. These values are subsequently used to visualize potentially malicious areas with the colormap, with red areas indicating larger discrepancies.
- `src_mesh.stl` and `tgt_mesh.stl`. These are the mesh objects used by the visualization.

The *Geometry* folder contains (1) the raw CT images and (2) the original design files in the “.stl” format of individual models. By comparing CT models and design models, XCheck verifies the geometry and material. 


## Basic Tests
The basic tests are designed to be simple functionality checks that do not necessitate running the complete system, but instead verify that all required software components are functioning properly. In our repository, the basic tests utilize the intermediate results obtained from the prior experiments, to provide both visualization and final determination (i.e., malicious or benign) of the corresponding model. As such, these basic tests verify the proper functioning of XCheck while not having to go through intermediate processes such as model registration and Housdorff distance calculation.

To run an existing analysis in basic test mode, run the following command. Please make sure the result subdirectory exists under the ./Results/ directory.
```
python3 run.py -basic <result_dir>
```
For example, the basic test for Bone_12 is:
```
python3 run.py -basic Bone_12
```

## Complete Execution 
To re-run the analysis in complete test mode, remove the `-basic` commands and enter the full command instead. Note that re-running analysis will overwrite the stored results of the respective model. 
```
python3 run.py -f1 <CT_scan_dir> -f2 <original_model_dir> -o <result_dir> -etdist <error_tolerance_distance> -ets <error_tolerance_scale> -etg <error_tolerance_group> -etm <error_tolerance_material>"
```

<!-- bone 12 -->
Compare CT scans of a 3D printed bone scaffold (manipulated by adding various internal solid regions) to its original model:
```
python3 run.py -f1 Geometry/Bone_12 -f2 Geometry/Bone.stl -o Bone_12 -etdist 1.7 -ets 0.05 -etg 0.005 -etm 1 
```
![Screenshot](images/bone_12.png)

<!-- bone 14 -->
Compare CT scans of a 3D printed bone scaffold (manipulated by adding various internal solid regions) to its original model:
```
python3 run.py -f1 Geometry/Bone_14 -f2 Geometry/Bone.stl -o Bone_14 -etdist 1.7 -ets 0.05 -etg 0.005 -etm 1 
```
![Screenshot](images/bone_14.png)

<!-- bone 24 solid region -->
Compare CT scans of a 3D printed bone scaffold (manipulated by adding various internal solid regions) to its original model:
```
python3 run.py -f1 Geometry/Bone_24 -f2 Geometry/Bone.stl -o Bone_24 -etdist 1.9 -ets 0.05 -etg 0.005 -etm 1 
```
![Screenshot](images/bone_24.png)

<!-- chip 42 solid region -->
Compare CT scans of a 3D printed lung-on-chip (manipulated with solidified airways) to its original model:
```
python3 run.py -f1 Geometry/Chip_42 -f2 Geometry/Chip.stl -o Chip_42 -etdist 1.7 -ets 0.05 -etg 0.005 -etm 1
```
![Screenshot](images/chip_42.png)

<!-- guide 2 -->
Compare CT scans of a 3D printed dental guide (swapped for a different material) to its original model:
```
python3 run.py -f1 Geometry/Guide_2 -f2 Geometry/Guide.stl -o Guide_2 -etdist 1.8 -ets 0.05 -etg 0.005 -etm 1
```
![Screenshot](images/guide_2.png)

<!-- guide ball -->
Compare CT scans of another 3D printed dental guide (manipulated by adding a solid sphere) to its original model:
```
python3 run.py -f1 Geometry/Guide_ball.stl -f2 Geometry/Guide.stl -o Guide_ball -etdist 1.6 -ets 0.05 -etg 0.005 -etm 1 
```
![Screenshot](images/guide_ball.png)

<!-- screw short -->
Compare CT scans of a 3D printed bone screw (shorten the screw stem) to its original model:
```
python3 run.py -f1 Geometry/Screw_short -f2 Geometry/Screw.STL -o Screw_short -etdist 1.6 -ets 0.05 -etg 0.005 -etm 1
```
![Screenshot](images/screw_short.png)

<!-- screw 5 -->
Compare CT scans of a 3D printed bone screw (significantly enlarged thread distance) to its original model:
```
python3 run.py -f1 Geometry/Screw_5/ -f2 Geometry/Screw.STL -o Screw_5 -etdist 1.6 -ets 0.05 -etg 0.005 -etm 1
```
![Screenshot](images/screw_5.png)

<!-- screw material -->
Compare CT scans of a 3D printed bone screw (swapped for a different material) to its original model:
```
python3 run.py -f1 Geometry/C-Screw -f2 Geometry/Screw.STL -o C_screw -etdist 1.6 -ets 0.05 -etg 0.005 -etm 1
```
![Screenshot](images/screw_mat.png)

# Citation

If you find the platform useful, please cite our work with the following reference:
```
@inproceedings {yu2023xcheck,
author = {Zhiyuan Yu and Yuanhaur Chang and Shixuan Zhai and Nicholas Deily and Tao Ju and XiaoFeng Wang and Uday Jammalamadaka and Ning Zhang},
title = {{XCheck}: Verifying Integrity of 3D Printed {Patient-Specific} Devices via Computing Tomography},
booktitle = {32nd USENIX Security Symposium (USENIX Security 23)},
year = {2023},
isbn = {978-1-939133-37-3},
address = {Anaheim, CA},
pages = {2815--2832},
url = {https://www.usenix.org/conference/usenixsecurity23/presentation/yu-zhiyuan-xcheck},
publisher = {USENIX Association},
month = aug,
}
```
