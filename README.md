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



#### <summary>Map-Relative Pose Regression for Visual Re-Localization
Authors: Shuai Chen, Tommaso Cavallari, Victor Adrian Prisacariu, Eric Brachmann
<details span>
<summary><b>Abstract</b></summary>
Pose regression networks predict the camera pose of a query image relative to a known environment. Within this family of methods, absolute pose regression (APR) has recently shown promising accuracy in the range of a few centimeters in position error. APR networks encode the scene geometry implicitly in their weights. To achieve high accuracy, they require vast amounts of training data that, realistically, can only be created using novel view synthesis in a days-long process. This process has to be repeated for each new scene again and again. We present a new approach to pose regression, map-relative pose regression (marepo), that satisfies the data hunger of the pose regression network in a scene-agnostic fashion. We condition the pose regressor on a scene-specific map representation such that its pose predictions are relative to the scene map. This allows us to train the pose regressor across hundreds of scenes to learn the generic relation between a scene-specific map representation and the camera pose. Our map-relative pose regressor can be applied to new map representations immediately or after mere minutes of fine-tuning for the highest accuracy. Our approach outperforms previous pose regression methods by far on two public datasets, indoor and outdoor.

![image](https://github.com/PAU1G3ORGE/-CameraLocalization/assets/167790336/f2d6ad7c-b782-482f-95ad-5f963bc1c3fa)


</details>

[📃 arXiv:2404](https://arxiv.org/pdf/2404.09884) | [⌨️ Code](https://github.com/nianticlabs/marepo) | [🌐 Project Page](https://nianticlabs.github.io/marepo/)

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

[📃 arXiv:2406] | [⌨️ Code] | [🌐 Project Page]
