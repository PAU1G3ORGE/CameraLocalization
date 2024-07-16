# CameraLocalization

<br>

#### <summary>Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses
Authors: Eric Brachmann, Tommaso Cavallari, Victor Adrian Prisacariu
<details span>
<summary><b>Abstract</b></summary>
Learning-based visual relocalizers exhibit leading pose accuracy, but require hours or days of training. Since training needs to happen on each new scene again, long training times make learning-based relocalization impractical for most applications, despite its promise of high accuracy. In this paper we show how such a system can actually achieve the same accuracy in less than 5 minutes. We start from the obvious: a relocalization network can be split in a scene-agnostic feature backbone, and a scene-specific prediction head. Less obvious: using an MLP prediction head allows us to optimize across thousands of view points simultaneously in each single training iteration. This leads to stable and extremely fast convergence. Furthermore, we substitute effective but slow end-to-end training using a robust pose solver with a curriculum over a reprojection loss. Our approach does not require privileged knowledge, such a depth maps or a 3D model, for speedy training. Overall, our approach is up to 300x faster in mapping than state-of-the-art scene coordinate regression, while keeping accuracy on par.

![image](https://github.com/PAU1G3ORGE/-CameraLocalization/assets/167790336/b6bd7a6d-cdbc-4de4-8933-d4255069bf5f)


</details>

[📃 arXiv:2305](https://arxiv.org/pdf/2305.14059) | [⌨️ Code](https://github.com/nianticlabs/ace) | [🌐 Project Page](https://nianticlabs.github.io/ace)


#### <summary>PNeRFLoc: Visual Localization with Point-based Neural Radiance Fields
Authors: Boming Zhao, Luwei Yang, Mao Mao, Hujun Bao, Zhaopeng Cui
<details span>
<summary><b>Abstract</b></summary>
Due to the ability to synthesize high-quality novel views, Neural Radiance Fields (NeRF) have been recently exploited to improve visual localization in a known environment. However, the existing methods mostly utilize NeRFs for data augmentation to improve the regression model training, and the performance on novel viewpoints and appearances is still limited due to the lack of geometric constraints. In this paper, we propose a novel visual localization framework, \ie, PNeRFLoc, based on a unified point-based representation. On the one hand, PNeRFLoc supports the initial pose estimation by matching 2D and 3D feature points as traditional structure-based methods; on the other hand, it also enables pose refinement with novel view synthesis using rendering-based optimization. Specifically, we propose a novel feature adaption module to close the gaps between the features for visual localization and neural rendering. To improve the efficacy and efficiency of neural rendering-based optimization, we also develop an efficient rendering-based framework with a warping loss function. Furthermore, several robustness techniques are developed to handle illumination changes and dynamic objects for outdoor scenarios. Experiments demonstrate that PNeRFLoc performs the best on synthetic data when the NeRF model can be well learned and performs on par with the SOTA method on the visual localization benchmark datasets.

![image](https://github.com/user-attachments/assets/8802372f-3d0f-4997-8a59-3d4573042d91)


</details>

[📃 arXiv:2312](https://arxiv.org/pdf/2312.10649) | [⌨️ Code](https://github.com/BoMingZhao/PNeRFLoc?tab=readme-ov-file) | [🌐 Project Page](https://zju3dv.github.io/PNeRFLoc/)



#### <summary>Learning to Produce Semi-dense Correspondences for Visual Localization
Authors: Khang Truong Giang, Soohwan Song, Sungho Jo
<details span>
<summary><b>Abstract</b></summary>
This study addresses the challenge of performing visual localization in demanding conditions such as night-time scenarios, adverse weather, and seasonal changes. While many prior studies have focused on improving image-matching performance to facilitate reliable dense keypoint matching between images, existing methods often heavily rely on predefined feature points on a reconstructed 3D model. Consequently, they tend to overlook unobserved keypoints during the matching process. Therefore, dense keypoint matches are not fully exploited, leading to a notable reduction in accuracy, particularly in noisy scenes. To tackle this issue, we propose a novel localization method that extracts reliable semi-dense 2D-3D matching points based on dense keypoint matches. This approach involves regressing semi-dense 2D keypoints into 3D scene coordinates using a point inference network. The network utilizes both geometric and visual cues to effectively infer 3D coordinates for unobserved keypoints from the observed ones. The abundance of matching information significantly enhances the accuracy of camera pose estimation, even in scenarios involving noisy or sparse 3D models. Comprehensive evaluations demonstrate that the proposed method outperforms other methods in challenging scenes and achieves competitive results in large-scale visual localization benchmarks.

![image](https://github.com/user-attachments/assets/73307b7c-f612-4ab6-939a-fd5697d7139c)


</details>

[📃 arXiv:2402](https://arxiv.org/pdf/2402.08359) | [⌨️ Code](https://github.com/TruongKhang/DeViLoc?tab=readme-ov-file) | [🌐 Project Page]



#### <summary>Map-Relative Pose Regression for Visual Re-Localization
Authors: Shuai Chen, Tommaso Cavallari, Victor Adrian Prisacariu, Eric Brachmann
<details span>
<summary><b>Abstract</b></summary>
Pose regression networks predict the camera pose of a query image relative to a known environment. Within this family of methods, absolute pose regression (APR) has recently shown promising accuracy in the range of a few centimeters in position error. APR networks encode the scene geometry implicitly in their weights. To achieve high accuracy, they require vast amounts of training data that, realistically, can only be created using novel view synthesis in a days-long process. This process has to be repeated for each new scene again and again. We present a new approach to pose regression, map-relative pose regression (marepo), that satisfies the data hunger of the pose regression network in a scene-agnostic fashion. We condition the pose regressor on a scene-specific map representation such that its pose predictions are relative to the scene map. This allows us to train the pose regressor across hundreds of scenes to learn the generic relation between a scene-specific map representation and the camera pose. Our map-relative pose regressor can be applied to new map representations immediately or after mere minutes of fine-tuning for the highest accuracy. Our approach outperforms previous pose regression methods by far on two public datasets, indoor and outdoor.

![image](https://github.com/PAU1G3ORGE/-CameraLocalization/assets/167790336/f2d6ad7c-b782-482f-95ad-5f963bc1c3fa)


</details>

[📃 arXiv:2404](https://arxiv.org/pdf/2404.09884) | [⌨️ Code](https://github.com/nianticlabs/marepo) | [🌐 Project Page](https://nianticlabs.github.io/marepo/)


#### <summary>Hybrid Structure-from-Motion and Camera Relocalization for Enhanced Egocentric Localization
Authors: Jinjie Mai, Abdullah Hamdi, Silvio Giancola, Chen Zhao, Bernard Ghanem
<details span>
<summary><b>Abstract</b></summary>
We built our pipeline EgoLoc-v1, mainly inspired by EgoLoc. We propose a model ensemble strategy to improve the camera pose estimation part of the VQ3D task, which has been proven to be essential in previous work. The core idea is not only to do SfM for egocentric videos but also to do 2D-3D matching between existing 3D scans and 2D video frames. In this way, we have a hybrid SfM and camera relocalization pipeline, which can provide us with more camera poses, leading to higher QwP and overall success rate. Our method achieves the best performance regarding the most important metric, the overall success rate. We surpass previous state-of-the-art, the competitive EgoLoc, by 1.5%.

![image](https://github.com/user-attachments/assets/4a776328-b152-43c9-bcda-62daf19d821f)


</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.08023) | [⌨️ Code](https://github.com/Wayne-Mai/egoloc_v1) | [🌐 Project Page]


#### <summary>3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization
Authors: Peng Jiang, Gaurav Pandey, Srikanth Saripalli
<details span>
<summary><b>Abstract</b></summary>
This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized cross-correlation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.

![image](https://github.com/user-attachments/assets/69d9c355-caff-45ca-bb59-74f1eb72dcca)


</details>

[📃 arXiv:2403](https://arxiv.org/pdf/2403.11367v1) | [⌨️ Code] | [🌐 Project Page]




<br>
<br>


## RetrievalMatching


#### <summary>Improved Scene Landmark Detection for Camera Localization
Authors: Tien Do, Sudipta N. Sinha
<details span>
<summary><b>Abstract</b></summary>
Camera localization methods based on retrieval, local feature matching, and 3D structure-based pose estimation are accurate but require high storage, are slow, and are not privacy-preserving. A method based on scene landmark detection (SLD) was recently proposed to address these limitations. It involves training a convolutional neural network (CNN) to detect a few predetermined, salient, scene-specific 3D points or landmarks and computing camera pose from the associated 2D-3D correspondences. Although SLD outperformed existing learning-based approaches, it was notably less accurate than 3D structure-based methods. In this paper, we show that the accuracy gap was due to insufficient model capacity and noisy labels during training. To mitigate the capacity issue, we propose to split the landmarks into subgroups and train a separate network for each subgroup. To generate better training labels, we propose using dense reconstructions to estimate visibility of scene landmarks. Finally, we present a compact architecture to improve memory efficiency. Accuracy wise, our approach is on par with state of the art structure based methods on the INDOOR-6 dataset but runs significantly faster and uses less storage.

![image](https://github.com/PAU1G3ORGE/-CameraLocalization/assets/167790336/6d45ebce-ca91-4383-9af5-e2e732ab6e78)


</details>

[📃 arXiv:2401](https://arxiv.org/pdf/2401.18083) | [⌨️ Code](https://github.com/microsoft/SceneLandmarkLocalization) | [🌐 Project Page]



<br>
<br>

#### <summary>
Authors: 
<details span>
<summary><b>Abstract</b></summary>


![image]()

</details>

[📃 arXiv:2407] | [⌨️ Code] | [🌐 Project Page]
