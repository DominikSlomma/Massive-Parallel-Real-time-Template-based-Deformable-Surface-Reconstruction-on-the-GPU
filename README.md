<h1> Massive Parallel Real time Template based Deformable Surface Reconstruction on the GPU</h1>

# 1. Overview !Todo
* Create a gif or two

![alternive textText](./data/image.png)
# 2. Acknowledgements !Todo
* Double checking and expand it!

The work has included analyses of the code from the following publications:
[Zhao, Liang & Huang, Shoudong & Sun, Yanbiao & Yan, Lei & Dissanayake, Gamini.
(2015). ParallaxBA: Bundle adjustment using parallax angle feature
parametrization. The International Journal of Robotics Research. 34. 493-516.
10.1177/0278364914551583.](https://www.researchgate.net/publication
275260778_ParallaxBA_Bundle_adjustment_using_parallax_angle_feature_para
etrization)
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
[Lamarca, Jose, et al. "DefSLAM: Tracking and Mapping of Deforming Scenes from
Monocular Sequences." arXiv preprint arXiv:1908.08918 (2019).](https://arxiv.org
abs/1908.08918)
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
# 3. Dependencies !Todo
* 3.1- 3.6 Updaten
* kann ich das ausführlicher machen?
## 3.1 Overview
<ul>
<li>C++17</li>
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

# 4. Installation and Building !Todo
* Mach es einfach aber verständlich für alle

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

# 5. Datasets !Todo
* completly neu machen mit Datensätzen! Ich werde die ich habe reduzieren
* ich muss auch erwähnen, wenn diese einen teil des datensatzes von phi oder hamlyn nehmen dann müssen diese den auch zitieren!

In our work we used the [&phi;-SfT](https://drive.google.com/drive/folders
1gpzp5k64S6TnDbl8ZW8lgSmDE_nzHdh9?usp=sharing) dataset, the [Phantom
(http://hamlyn.doc.ic.ac.uk/vision/) dataset and an own created laparoscopic
dataset which we created with [colon_reconstruction_dataset](https:/
github.com/zsustc/colon_reconstruction_dataset) 
You can download the dataset which we used here: [Download]()

# 6. Run !Todo
* Das einfacher machen!

The executable files are created in the *build* folder. To be able to execute these
executable files successfully, first switch to the *build* folder and execute the
corresponding program. To be able to run the program successfully, please adjust
the corresponding configuration file in the *App* folder.

# 7. Docker !Todo
* Auf docker eingehen und erklären!

# 8. License
This project is licensed under the [GNU General Public License (GPL)](https:/
www.gnu.org/licenses/gpl-3.0.html).
# 9. Reference

# 10. Troubleshooting