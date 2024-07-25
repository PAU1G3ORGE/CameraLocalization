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


#### <summary>Differentiable Product Quantization for Memory Efficient Camera Relocalization
Authors: Zakaria Laskar, Iaroslav Melekhov, Assia Benbihi, Shuzhe Wang, Juho Kannala
<details span>
<summary><b>Abstract</b></summary>
Camera relocalization relies on 3D models of the scene with a large memory footprint that is incompatible with the memory budget of several applications. One solution to reduce the scene memory size is map compression by removing certain 3D points and descriptor quantization. This achieves high compression but leads to performance drop due to information loss. To address the memory performance trade-off, we train a light-weight scene-specific auto-encoder network that performs descriptor quantization-dequantization in an end-to-end differentiable manner updating both product quantization centroids and network parameters through back-propagation. In addition to optimizing the network for descriptor reconstruction, we encourage it to preserve the descriptor-matching performance with margin-based metric loss functions. Results show that for a local descriptor memory of only 1MB, the synergistic combination of the proposed network and map compression achieves the best performance on the Aachen Day-Night compared to existing compression methods.

![image](https://github.com/user-attachments/assets/59eaaf18-39a7-40fd-83c8-0c9dc8a7fff4)


</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.15540) | [⌨️ Code] | [🌐 Project Page]


#### <summary>6DGS: 6D Pose Estimation from a Single Image and a 3D Gaussian Splatting Model
Authors: Matteo Bortolon, Theodore Tsesmelis, Stuart James, Fabio Poiesi, Alessio Del Bue
<details span>
<summary><b>Abstract</b></summary>
We propose 6DGS to estimate the camera pose of a target RGB image given a 3D Gaussian Splatting (3DGS) model representing the scene. 6DGS avoids the iterative process typical of analysis-by-synthesis methods (e.g. iNeRF) that also require an initialization of the camera pose in order to converge. Instead, our method estimates a 6DoF pose by inverting the 3DGS rendering process. Starting from the object surface, we define a radiant Ellicell that uniformly generates rays departing from each ellipsoid that parameterize the 3DGS model. Each Ellicell ray is associated with the rendering parameters of each ellipsoid, which in turn is used to obtain the best bindings between the target image pixels and the cast rays. These pixel-ray bindings are then ranked to select the best scoring bundle of rays, which their intersection provides the camera center and, in turn, the camera rotation. The proposed solution obviates the necessity of an "a priori" pose for initialization, and it solves 6DoF pose estimation in closed form, without the need for iterations. Moreover, compared to the existing Novel View Synthesis (NVS) baselines for pose estimation, 6DGS can improve the overall average rotational accuracy by 12% and translation accuracy by 22% on real scenes, despite not requiring any initialization pose. At the same time, our method operates near real-time, reaching 15fps on consumer hardware.

![image](https://github.com/user-attachments/assets/0a15b55d-0d93-430f-855c-5d319675b42c)

</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.15484) | [⌨️ Code](https://github.com/mbortolon97/6dgs) | [🌐 Project Page](https://mbortolon97.github.io/6dgs/)

#### <summary>Generative Lifting of Multiview to 3D from Unknown Pose: Wrapping NeRF inside Diffusion
Authors: Xin Yuan, Rana Hanocka, Michael Maire
<details span>
<summary><b>Abstract</b></summary>
We cast multiview reconstruction from unknown pose as a generative modeling problem. From a collection of unannotated 2D images of a scene, our approach simultaneously learns both a network to predict camera pose from 2D image input, as well as the parameters of a Neural Radiance Field (NeRF) for the 3D scene. To drive learning, we wrap both the pose prediction network and NeRF inside a Denoising Diffusion Probabilistic Model (DDPM) and train the system via the standard denoising objective. Our framework requires the system accomplish the task of denoising an input 2D image by predicting its pose and rendering the NeRF from that pose. Learning to denoise thus forces the system to concurrently learn the underlying 3D NeRF representation and a mapping from images to camera extrinsic parameters. To facilitate the latter, we design a custom network architecture to represent pose as a distribution, granting implicit capacity for discovering view correspondences when trained end-to-end for denoising alone. This technique allows our system to successfully build NeRFs, without pose knowledge, for challenging scenes where competing methods fail. At the conclusion of training, our learned NeRF can be extracted and used as a 3D scene model; our full system can be used to sample novel camera poses and generate novel-view images.

![image](https://github.com/PAU1G3ORGE/AwesomeGaussian/assets/167790336/dff892ae-98e8-456b-b9ea-554d05c1be3e)


</details>

[📃 arXiv:2406](https://arxiv.org/pdf/2406.06972) | [⌨️ Code] | [🌐 Project Page]


#### <summary>SRPose: Two-view Relative Pose Estimation with Sparse Keypoints
Authors: Rui Yin, Yulun Zhang, Zherong Pan, Jianjun Zhu, Cheng Wang, Biao Jia
<details span>
<summary><b>Abstract</b></summary>
Two-view pose estimation is essential for map-free visual relocalization and object pose tracking tasks. However, traditional matching methods suffer from time-consuming robust estimators, while deep learning-based pose regressors only cater to camera-to-world pose estimation, lacking generalizability to different image sizes and camera intrinsics. In this paper, we propose SRPose, a sparse keypoint-based framework for two-view relative pose estimation in camera-to-world and object-to-camera scenarios. SRPose consists of a sparse keypoint detector, an intrinsic-calibration position encoder, and promptable prior knowledge-guided attention layers. Given two RGB images of a fixed scene or a moving object, SRPose estimates the relative camera or 6D object pose transformation. Extensive experiments demonstrate that SRPose achieves competitive or superior performance compared to state-of-the-art methods in terms of accuracy and speed, showing generalizability to both scenarios. It is robust to different image sizes and camera intrinsics, and can be deployed with low computing resources.

![image](https://github.com/user-attachments/assets/a8ce5d31-0fa4-4150-92ef-8bc832ae6d8f)


</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.08199) | [⌨️ Code] | [🌐 Project Page]


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
