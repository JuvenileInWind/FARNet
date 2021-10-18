# Feature Aggregation and Refinement Network for 2D Anatomical Landmark Detection

## Overview
Localization of anatomical landmarks is essential for clinical diagnosis, treatment planning, and research. In this paper, we propose a novel deep network, named feature aggregation and refinement network (FARNet), for the automatic detection of anatomical landmarks. To alleviate the problem of limited training data in the medical domain, our network adopts a CNN pre-trained on natural images as the backbone network and several popular networks have been compared. Our FARNet also includes a multi-scale feature aggregation module for multiscale feature fusion and a feature refinement module for high-resolution heatmap regression. Coarse-to-fine supervisions are applied to the two modules to facilitate the endto-end training. We further propose a novel loss function named Exponential Weighted Center loss for more accurate heatmap regression, which focuses on the losses from the pixels near landmarks and suppresses the ones from far away. Our network has been evaluated on three publicly available anatomical landmark detection datasets, including cephalometric radiographs, hand radiographs, and spine radiographs, and achieves state-of-art performances on all three datasets.

![The architecture of the feature aggregation and refinement network (FARNet). FARNet includes a backbone network
(in the pink dashed box), a multi-scale feature aggregation (MSFA) module (in the blue dashed box) and a feature refinement
(FR) module (in the brown dashed box). We also give the feature level labels {L0, L1, L2, L3, L4, L5} at the left side of the
figure, and all feature maps at the same horizontal level have the same spatial resolution.](https://github.com/JuvenileInWind/Farnet/tree/master/image/FARNet_bold.pdf)

## Data
In this paper, we evaluate our landmark detection network
on three public benchmark data sets, a cephalometric X-rays
dataset [1], a hand X-rays dataset [2] and a Spinal AnteriorPosterior (AP) X-rays dataset [3].
## How to use
### Dependencies
This tutorial depends on the following libraries:
* pytorch = 1.0.1
* numpy = 1.18.5
* python >= 3.6
* xlwt

### config.py
You should set the image path in config by yourself

### Run main.py
Run main.py to train the model and test its performance

### Some results 
![ Illustration of landmark detection results by our proposed method on three public datasets. The first row is the task
of cephalometric landmark detetcion(19 landmarks), the second row is the task of hand radiographs landmark detection(37
landmarks) and the last row is the task of spinal anterior-posterior x-ray landmark detection(68 landmarks). The red points
denote our detected landmarks via our framework, while blue points represent the ground-truth landmarks.](https://github.com/JuvenileInWind/Farnet/tree/master/image/results.png)

## Reference
[1] C.-W. Wang, C.-T. Huang, J.-H. Lee, C.-H. Li, S.-W. Chang, M.-J.Siao, T.-M. Lai, B. Ibragimov, T. Vrtovec, O. Ronneberger, et al., “A benchmark for comparison of dental radiography analysis algorithms,” Medical image analysis, vol. 31, pp. 63–76, 2016.  
[2] C. Payer, D. ˇStern, H. Bischof, and M. Urschler, “Integrating spatial configuration into heatmap regression based cnns for landmark localization,” Medical Image Analysis, vol. 54, pp. 207–219, 2019.  
[3] H. Wu, C. Bailey, P. Rasoulinejad, and S. Li, “Automatic landmark estimation for adolescent idiopathic scoliosis assessment using boostnet,” in International Conference on Medical Image Computing and ComputerAssisted Intervention, 2017.  
