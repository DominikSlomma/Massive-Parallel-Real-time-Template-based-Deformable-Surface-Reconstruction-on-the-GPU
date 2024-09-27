<h1> Massive Parallel Real-time Template based Deformable Surface Reconstruction on the GPU</h1>

# 1. Overview 

![alternive textText](./doc/ColonCmp.gif)
# 2. Acknowledgements


The work has included analyses of the code from the following publications:

[Zhao, Liang & Huang, Shoudong & Sun, Yanbiao & Yan, Lei & Dissanayake, Gamini. (2015). ParallaxBA: Bundle adjustment using parallax angle feature parametrization. The International Journal of Robotics Research. 34. 493-516. 10.1177/0278364914551583.](https://www.researchgate.net/publication275260778_ParallaxBA_Bundle_adjustment_using_parallax_angle_feature_paraetrization)
```
@article{article,
author = {Zhao, Liang and Huang, Shoudong and Sun, Yanbiao and Yan, Lei and
Dissanayake, Gamini},
year = {2015},month = {04},
pages = {493-516},
title = {ParallaxBA: Bundle adjustment using parallax angle feature
parametrization},
volume = {34},
journal = {The International Journal of Robotics Research},
doi = {10.1177/0278364914551583}
}
```
[Lamarca, Jose, et al. "DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular Sequences." arXiv preprint arXiv:1908.08918 (2019).](https://arxiv.orgabs/1908.08918)
```
@article{lamarca2019defslam,
title={DefSLAM: Tracking and Mapping of Deforming Scenes from Monocular
Sequences},
author={Lamarca, Jose and Parashar, Shaifali and Bartoli, Adrien and Montiel,
JMM},
journal={arXiv preprint arXiv:1908.08918},
year={2019}
}
```
# 3. Dependencies
## 3.1 Overview
<ul>
<li>C++17</li>
<li>Cuda-11.8</li>
<li>OpenCV-4.5.0</li>
<li>Eigen-3.3</li>
<li>Open3D-0.18.0</li>
<li>Pangolin-0.9</li>
<li>Suitsparse</li>
</ul>

## 3.2 Pangolin
We use [Pangolin 0.9](https://github.com/stevenlovegrove/Pangolin) for
visualization and user interface. Dowload and install instructions can be found at:
https://github.com/stevenlovegrove/Pangolin.

## 3.3 OpenCV
We use [OpenCV 4.5.0](http://opencv.org) to manipulate images and track the
features. Dowload and install instructions can be found at: http://opencv.org.

## 3.4 Eigen3
We use Eigen 3.3 to make the code more readable. Download and install
instructions can be found at: http://eigen.tuxfamily.org.

## 3.5 Open3D
We use [Open3D 0.18.0](https://github.com/isl-org/Open3D) for loading
pointclouds, meshes and to compare the result with ground truth. Download and
install instructions can be found at: https://www.open3d.org/

## 3.6 Suitsparse
We use the Suitsparse v7.7.0 for the optimization process.

# 4. Installation and Building

Clone the repository:
```
git clone https://github.com/DominikSlomma/Massive-Parallel-Real-time-Template-based-Deformable-Surface-Reconstruction-on-the-GPU
```
We provide a script `build.sh`to build *Massive Parallel Real-time Template-based Deformable Surface Reconstruction on the GPU*.
Please make sure that all dependencies are installed (see section 3).
```
cd Massive-Parallel-Real-time-Template-based-Deformable-Surface-Reconstruction-on-the-GPU
chmod +x build.sh
./build.sh
```
This will create a folder *build* where the executables will be created.

# 5. Datasets 

You can download our dataset over the following link: [our dataset](https://studentutsedu-my.sharepoint.com/:f:/g/personal/dominik_slomma_student_uts_edu_au/EhRgrDQHanxPuQ0Z_k3ik5gBY6FViCGKqNdPxEUWx0DCHw?e=3rvjQh)

If you want the original blanket dataset please refere to the  [$\phi$-SfT](https://4dqv.mpi-inf.mpg.de/phi-SfT/) project website.

If you are interest on your own colonoscopy dataset use the folloging github: [Colonoscopy simulator](https://github.com/zsustc/colon_reconstruction_dataset) 

If you want to have the original Phantom dataset please refer to the [Hamlyn-Website](https://hamlyn.doc.ic.ac.uk/vision/)


# 6. Run
The executable files are created in the *build* folder. To be able to execute these executable files successfully, first switch to the *build* folder and execute the corresponding program. To be able to run the program successfully, please adjust the corresponding configuration file in the *App* folder. 

If you want to use the current configuration, please create a *data* folder and include *our* dataset. Also change into the *build* folder:

```
cd ./build/
```
After that you can run one out of 14 tests. The appications are shown below:

Blanket-Dataset:
```
./S1_real
./S2_real
./S3_real
./S4_real
./S5_real
./S6_real
./S7_real
./S8_real
./S9_real
```
Colonoscopy-Dataset:
```
./colonscopy_1
./colonscopy_2
./colonscopy_3
```
Hamlyn-Dataset:
```
./hamlyn_f5
./hamlyn_f7
```

# 7. Docker 

<h3> Currently working on a Docker file!</h3>

# 8. License
This project is licensed under the [GNU General Public License (GPL)](https:/
www.gnu.org/licenses/gpl-3.0.html).
# 9. Reference

<h3> Currently working on this part </h3>


# 10. Troubleshooting


<h3> This part will later include the visualisation problems when using docker. </h3>
