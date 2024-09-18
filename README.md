# CameraLocalization

<br>

#### <summary>On the Instability of Relative Pose Estimation and RANSAC's Role
Authors: Hongyi Fan, Joe Kileel, Benjamin Kimia
<details span>
<summary><b>Abstract</b></summary>
In this paper we study the numerical instabilities of the 5- and 7-point problems for essential and fundamental matrix estimation in multiview geometry. In both cases we characterize the ill-posed world scenes where the condition number for epipolar estimation is infinite. We also characterize the ill-posed instances in terms of the given image data. To arrive at these results, we present a general framework for analyzing the conditioning of minimal problems in multiview geometry, based on Riemannian manifolds. Experiments with synthetic and real-world data then reveal a striking conclusion: that Random Sample Consensus (RANSAC) in Structure-from-Motion (SfM) does not only serve to filter out outliers, but RANSAC also selects for well-conditioned image data, sufficiently separated from the ill-posed locus that our theory predicts. Our findings suggest that, in future work, one could try to accelerate and increase the success of RANSAC by testing only well-conditioned image data.


</details>

[📃 arXiv:2112](https://arxiv.org/pdf/2112.14651) | [⌨️ Code] | [🌐 Project Page]



#### <summary>DFNet: Enhance Absolute Pose Regression with Direct Feature Matching
Authors: Shuai Chen, Xinghui Li, Zirui Wang, Victor Adrian Prisacariu
<details span>
<summary><b>Abstract</b></summary>
We introduce a camera relocalization pipeline that combines absolute pose regression (APR) and direct feature matching. By incorporating exposure-adaptive novel view synthesis, our method successfully addresses photometric distortions in outdoor environments that existing photometric-based methods fail to handle. With domain-invariant feature matching, our solution improves pose regression accuracy using semi-supervised learning on unlabeled data. In particular, the pipeline consists of two components: Novel View Synthesizer and DFNet. The former synthesizes novel views compensating for changes in exposure and the latter regresses camera poses and extracts robust features that close the domain gap between real images and synthetic ones. Furthermore, we introduce an online synthetic data generation scheme. We show that these approaches effectively enhance camera pose estimation both in indoor and outdoor scenes. Hence, our method achieves a state-of-the-art accuracy by outperforming existing single-image APR methods by as much as 56%, comparable to 3D structure-based methods.

![image](https://github.com/user-attachments/assets/68f2c753-1aa6-4da8-9f50-bcea1a7f4f00)

</details>

[📃 arXiv:2204](https://arxiv.org/pdf/2204.00559) | [⌨️ Code](https://github.com/ActiveVisionLab/DFNet) | [🌐 Project Page](https://code.active.vision/)


#### <summary>CROSSFIRE: Camera Relocalization On Self-Supervised Features from an Implicit Representation
Authors: Arthur Moreau, Nathan Piasco, Moussab Bennehar, Dzmitry Tsishkou, Bogdan Stanciulescu, Arnaud de La Fortelle
<details span>
<summary><b>Abstract</b></summary>
Beyond novel view synthesis, Neural Radiance Fields are useful for applications that interact with the real world. In this paper, we use them as an implicit map of a given scene and propose a camera relocalization algorithm tailored for this representation. The proposed method enables to compute in real-time the precise position of a device using a single RGB camera, during its navigation. In contrast with previous work, we do not rely on pose regression or photometric alignment but rather use dense local features obtained through volumetric rendering which are specialized on the scene with a self-supervised objective. As a result, our algorithm is more accurate than competitors, able to operate in dynamic outdoor environments with changing lightning conditions and can be readily integrated in any volumetric neural renderer.

![image](https://github.com/user-attachments/assets/39ccd9a1-690a-492f-878a-2b6a9dd4f4dc)

</details>

[📃 arXiv:2303](https://arxiv.org/pdf/2303.04869) | [⌨️ Code] | [🌐 Project Page]

#### <summary>Neural Refinement for Absolute Pose Regression with Feature Synthesis
> *However, since our NeFeS outputs both colors and features simultaneously, we find this approach perturbs the feature output values and causes instability. ACT origial from NeRF-W*

Authors: Shuai Chen, Yash Bhalgat, Xinghui Li, Jiawang Bian, Kejie Li, Zirui Wang, Victor Adrian Prisacariu
<details span>
<summary><b>Abstract</b></summary>
Absolute Pose Regression (APR) methods use deep neural networks to directly regress camera poses from RGB images. However, the predominant APR architectures only rely on 2D operations during inference, resulting in limited accuracy of pose estimation due to the lack of 3D geometry constraints or priors. In this work, we propose a test-time refinement pipeline that leverages implicit geometric constraints using a robust feature field to enhance the ability of APR methods to use 3D information during inference. We also introduce a novel Neural Feature Synthesizer (NeFeS) model, which encodes 3D geometric features during training and directly renders dense novel view features at test time to refine APR methods. To enhance the robustness of our model, we introduce a feature fusion module and a progressive training strategy. Our proposed method achieves state-of-the-art single-image APR accuracy on indoor and outdoor datasets.

![image](https://github.com/user-attachments/assets/08a5c8f6-77c8-4df8-8fca-bf7cd4a7f0f3)

</details>

[📃 arXiv:2303](https://arxiv.org/pdf/2303.10087) | [⌨️ Code](https://github.com/ActiveVisionLab/NeFeS) | [🌐 Project Page]


#### <summary>NeRF-Loc: Visual Localization with Conditional Neural Radiance Field
Authors: Jianlin Liu, Qiang Nie, Yong Liu, Chengjie Wang
<details span>
<summary><b>Abstract</b></summary>
We propose a novel visual re-localization method based on direct matching between the implicit 3D descriptors and the 2D image with transformer. A conditional neural radiance field(NeRF) is chosen as the 3D scene representation in our pipeline, which supports continuous 3D descriptors generation and neural rendering. By unifying the feature matching and the scene coordinate regression to the same framework, our model learns both generalizable knowledge and scene prior respectively during two training stages. Furthermore, to improve the localization robustness when domain gap exists between training and testing phases, we propose an appearance adaptation layer to explicitly align styles between the 3D model and the query image. Experiments show that our method achieves higher localization accuracy than other learning-based approaches on multiple benchmarks.

![image](https://github.com/user-attachments/assets/f0b3a9e3-fa19-4d07-89fe-3208e1ee80bd)

</details>

[📃 arXiv:2304](https://arxiv.org/pdf/2304.07979) | [⌨️ Code](https://github.com/JenningsL/nerf-loc) | [🌐 Project Page]



#### <summary>Accelerated Coordinate Encoding: Learning to Relocalize in Minutes using RGB and Poses
Authors: Eric Brachmann, Tommaso Cavallari, Victor Adrian Prisacariu
<details span>
<summary><b>Abstract</b></summary>
Learning-based visual relocalizers exhibit leading pose accuracy, but require hours or days of training. Since training needs to happen on each new scene again, long training times make learning-based relocalization impractical for most applications, despite its promise of high accuracy. In this paper we show how such a system can actually achieve the same accuracy in less than 5 minutes. We start from the obvious: a relocalization network can be split in a scene-agnostic feature backbone, and a scene-specific prediction head. Less obvious: using an MLP prediction head allows us to optimize across thousands of view points simultaneously in each single training iteration. This leads to stable and extremely fast convergence. Furthermore, we substitute effective but slow end-to-end training using a robust pose solver with a curriculum over a reprojection loss. Our approach does not require privileged knowledge, such a depth maps or a 3D model, for speedy training. Overall, our approach is up to 300x faster in mapping than state-of-the-art scene coordinate regression, while keeping accuracy on par.

![image](https://github.com/user-attachments/assets/26decca7-d0b5-46db-be1e-b3bad615d7e5)

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


#### <summary>HR-APR: APR-agnostic Framework with Uncertainty Estimation and Hierarchical Refinement for Camera Relocalisation
> *all three APR models exhibit more accurate predictions for test queries with viewpoints similar to those in the training set, compared to queries that fall outside the coverage of the training data*

Authors: Changkun Liu, Shuai Chen, Yukun Zhao, Huajian Huang, Victor Prisacariu, Tristan Braud
<details span>
<summary><b>Abstract</b></summary>
Absolute Pose Regressors (APRs) directly estimate camera poses from monocular images, but their accuracy is unstable for different queries. Uncertainty-aware APRs provide uncertainty information on the estimated pose, alleviating the impact of these unreliable predictions. However, existing uncertainty modelling techniques are often coupled with a specific APR architecture, resulting in suboptimal performance compared to state-of-the-art (SOTA) APR methods. This work introduces a novel APR-agnostic framework, HR-APR, that formulates uncertainty estimation as cosine similarity estimation between the query and database features. It does not rely on or affect APR network architecture, which is flexible and computationally efficient. In addition, we take advantage of the uncertainty for pose refinement to enhance the performance of APR. The extensive experiments demonstrate the effectiveness of our framework, reducing 27.4\% and 15.2\% of computational overhead on the 7Scenes and Cambridge Landmarks datasets while maintaining the SOTA accuracy in single-image APRs.

![image](https://github.com/user-attachments/assets/f4d002cb-3794-4daa-bfbe-a462729f98e2)

</details>

[📃 arXiv:2402](https://arxiv.org/pdf/2402.14371) | [⌨️ Code](https://github.com/lck666666/HR-APR) | [🌐 Project Page](https://lck666666.github.io/research/HR-APR/index.html)


#### <summary>The NeRFect Match: Exploring NeRF Features for Visual Localization
Authors: Qunjie Zhou, Maxim Maximov, Or Litany, Laura Leal-Taixé
<details span>
<summary><b>Abstract</b></summary>
In this work, we propose the use of Neural Radiance Fields (NeRF) as a scene representation for visual localization. Recently, NeRF has been employed to enhance pose regression and scene coordinate regression models by augmenting the training database, providing auxiliary supervision through rendered images, or serving as an iterative refinement module. We extend its recognized advantages -- its ability to provide a compact scene representation with realistic appearances and accurate geometry -- by exploring the potential of NeRF's internal features in establishing precise 2D-3D matches for localization. To this end, we conduct a comprehensive examination of NeRF's implicit knowledge, acquired through view synthesis, for matching under various conditions. This includes exploring different matching network architectures, extracting encoder features at multiple layers, and varying training configurations. Significantly, we introduce NeRFMatch, an advanced 2D-3D matching function that capitalizes on the internal knowledge of NeRF learned via view synthesis. Our evaluation of NeRFMatch on standard localization benchmarks, within a structure-based pipeline, sets a new state-of-the-art for localization performance on Cambridge Landmarks.

![image](https://github.com/user-attachments/assets/1a12c49b-6f5a-40de-84f8-fa6523fb6a4e)


</details>

[📃 arXiv:2403](https://arxiv.org/pdf/2403.09577) | [⌨️ Code] | [🌐 Project Page](https://nerfmatch.github.io/)

#### <summary>3DGS-ReLoc: 3D Gaussian Splatting for Map Representation and Visual ReLocalization
Authors: Peng Jiang, Gaurav Pandey, Srikanth Saripalli
<details span>
<summary><b>Abstract</b></summary>
This paper presents a novel system designed for 3D mapping and visual relocalization using 3D Gaussian Splatting. Our proposed method uses LiDAR and camera data to create accurate and visually plausible representations of the environment. By leveraging LiDAR data to initiate the training of the 3D Gaussian Splatting map, our system constructs maps that are both detailed and geometrically accurate. To mitigate excessive GPU memory usage and facilitate rapid spatial queries, we employ a combination of a 2D voxel map and a KD-tree. This preparation makes our method well-suited for visual localization tasks, enabling efficient identification of correspondences between the query image and the rendered image from the Gaussian Splatting map via normalized cross-correlation (NCC). Additionally, we refine the camera pose of the query image using feature-based matching and the Perspective-n-Point (PnP) technique. The effectiveness, adaptability, and precision of our system are demonstrated through extensive evaluation on the KITTI360 dataset.

![image](https://github.com/user-attachments/assets/69d9c355-caff-45ca-bb59-74f1eb72dcca)


</details>

[📃 arXiv:2403](https://arxiv.org/pdf/2403.11367v1) | [⌨️ Code] | [🌐 Project Page]


#### <summary>Learning Neural Volumetric Pose Features for Camera Localization
Authors: Jingyu Lin, Jiaqi Gu, Bojian Wu, Lubin Fan, Renjie Chen, Ligang Liu, Jieping Ye
<details span>
<summary><b>Abstract</b></summary>
We introduce a novel neural volumetric pose feature, termed PoseMap, designed to enhance camera localization by encapsulating the information between images and the associated camera poses. Our framework leverages an Absolute Pose Regression (APR) architecture, together with an augmented NeRF module. This integration not only facilitates the generation of novel views to enrich the training dataset but also enables the learning of effective pose features. Additionally, we extend our architecture for self-supervised online alignment, allowing our method to be used and fine-tuned for unlabelled images within a unified framework. Experiments demonstrate that our method achieves 14.28% and 20.51% performance gain on average in indoor and outdoor benchmark scenes, outperforming existing APR methods with state-of-the-art accuracy.

![image](https://github.com/user-attachments/assets/634a0dcd-a127-4d5d-8107-a395199ca1d7)

</details>

[📃 arXiv:2403](https://arxiv.org/pdf/2403.12800) | [⌨️ Code] | [🌐 Project Page](https://gujiaqivadin.github.io/posemap/)


#### <summary>Map-Relative Pose Regression for Visual Re-Localization
Authors: Shuai Chen, Tommaso Cavallari, Victor Adrian Prisacariu, Eric Brachmann
<details span>
<summary><b>Abstract</b></summary>
Pose regression networks predict the camera pose of a query image relative to a known environment. Within this family of methods, absolute pose regression (APR) has recently shown promising accuracy in the range of a few centimeters in position error. APR networks encode the scene geometry implicitly in their weights. To achieve high accuracy, they require vast amounts of training data that, realistically, can only be created using novel view synthesis in a days-long process. This process has to be repeated for each new scene again and again. We present a new approach to pose regression, map-relative pose regression (marepo), that satisfies the data hunger of the pose regression network in a scene-agnostic fashion. We condition the pose regressor on a scene-specific map representation such that its pose predictions are relative to the scene map. This allows us to train the pose regressor across hundreds of scenes to learn the generic relation between a scene-specific map representation and the camera pose. Our map-relative pose regressor can be applied to new map representations immediately or after mere minutes of fine-tuning for the highest accuracy. Our approach outperforms previous pose regression methods by far on two public datasets, indoor and outdoor.

![image](https://github.com/user-attachments/assets/4e3b7b90-2573-438d-9a15-4526017e4816)

</details>

[📃 arXiv:2404](https://arxiv.org/pdf/2404.09884) | [⌨️ Code](https://github.com/nianticlabs/marepo) | [🌐 Project Page](https://nianticlabs.github.io/marepo/)

#### <summary>Scene Coordinate Reconstruction: Posing of Image Collections via Incremental Learning of a Relocalizer
Authors: Eric Brachmann, Jamie Wynn, Shuai Chen, Tommaso Cavallari, Áron Monszpart, Daniyar Turmukhambetov, Victor Adrian Prisacariu
<details span>
<summary><b>Abstract</b></summary>
We address the task of estimating camera parameters from a set of images depicting a scene. Popular feature-based structure-from-motion (SfM) tools solve this task by incremental reconstruction: they repeat triangulation of sparse 3D points and registration of more camera views to the sparse point cloud. We re-interpret incremental structure-from-motion as an iterated application and refinement of a visual relocalizer, that is, of a method that registers new views to the current state of the reconstruction. This perspective allows us to investigate alternative visual relocalizers that are not rooted in local feature matching. We show that scene coordinate regression, a learning-based relocalization approach, allows us to build implicit, neural scene representations from unposed images. Different from other learning-based reconstruction methods, we do not require pose priors nor sequential inputs, and we optimize efficiently over thousands of images. In many cases, our method, ACE0, estimates camera poses with an accuracy close to feature-based SfM, as demonstrated by novel view synthesis.

![image](https://github.com/user-attachments/assets/343a1852-73ec-46d1-a2db-adc94bde075d)

</details>

[📃 arXiv:2404](https://arxiv.org/pdf/2404.14351) | [⌨️ Code](https://github.com/nianticlabs/acezero?tab=readme-ov-file) | [🌐 Project Page](https://nianticlabs.github.io/acezero/)



#### <summary>GLACE: Global Local Accelerated Coordinate Encoding
> *Research [38] shows that the final layer has an important effect on the prior of CNNs that regress spatial positions, if the direct output of the last linear layer is a linear combination of bases in its weight*

Authors: Fangjinhua Wang, Xudong Jiang, Silvano Galliani, Christoph Vogel, Marc Pollefeys
<details span>
<summary><b>Abstract</b></summary>
Scene coordinate regression (SCR) methods are a family of visual localization methods that directly regress 2D-3D matches for camera pose estimation. They are effective in small-scale scenes but face significant challenges in large-scale scenes that are further amplified in the absence of ground truth 3D point clouds for supervision. Here, the model can only rely on reprojection constraints and needs to implicitly triangulate the points. The challenges stem from a fundamental dilemma: The network has to be invariant to observations of the same landmark at different viewpoints and lighting conditions, etc., but at the same time discriminate unrelated but similar observations. The latter becomes more relevant and severe in larger scenes. In this work, we tackle this problem by introducing the concept of co-visibility to the network. We propose GLACE, which integrates pre-trained global and local encodings and enables SCR to scale to large scenes with only a single small-sized network. Specifically, we propose a novel feature diffusion technique that implicitly groups the reprojection constraints with co-visibility and avoids overfitting to trivial solutions. Additionally, our position decoder parameterizes the output positions for large-scale scenes more effectively. Without using 3D models or depth maps for supervision, our method achieves state-of-the-art results on large-scale scenes with a low-map-size model. On Cambridge landmarks, with a single model, we achieve 17% lower median position error than Poker, the ensemble variant of the state-of-the-art SCR method ACE.

![image](https://github.com/user-attachments/assets/740abc52-a696-4957-8ff8-f3e99b32355b)

</details>

[📃 arXiv:2406](https://arxiv.org/pdf/2406.04340) | [⌨️ Code](https://github.com/cvg/glace?tab=readme-ov-file) | [🌐 Project Page](https://xjiangan.github.io/glace)




#### <summary>Hybrid Structure-from-Motion and Camera Relocalization for Enhanced Egocentric Localization
Authors: Jinjie Mai, Abdullah Hamdi, Silvio Giancola, Chen Zhao, Bernard Ghanem
<details span>
<summary><b>Abstract</b></summary>
We built our pipeline EgoLoc-v1, mainly inspired by EgoLoc. We propose a model ensemble strategy to improve the camera pose estimation part of the VQ3D task, which has been proven to be essential in previous work. The core idea is not only to do SfM for egocentric videos but also to do 2D-3D matching between existing 3D scans and 2D video frames. In this way, we have a hybrid SfM and camera relocalization pipeline, which can provide us with more camera poses, leading to higher QwP and overall success rate. Our method achieves the best performance regarding the most important metric, the overall success rate. We surpass previous state-of-the-art, the competitive EgoLoc, by 1.5%.

![image](https://github.com/user-attachments/assets/4a776328-b152-43c9-bcda-62daf19d821f)


</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.08023) | [⌨️ Code](https://github.com/Wayne-Mai/egoloc_v1) | [🌐 Project Page]




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


#### <summary>SRPose: Two-view Relative Pose Estimation with Sparse Keypoints
Authors: Rui Yin, Yulun Zhang, Zherong Pan, Jianjun Zhu, Cheng Wang, Biao Jia
<details span>
<summary><b>Abstract</b></summary>
Two-view pose estimation is essential for map-free visual relocalization and object pose tracking tasks. However, traditional matching methods suffer from time-consuming robust estimators, while deep learning-based pose regressors only cater to camera-to-world pose estimation, lacking generalizability to different image sizes and camera intrinsics. In this paper, we propose SRPose, a sparse keypoint-based framework for two-view relative pose estimation in camera-to-world and object-to-camera scenarios. SRPose consists of a sparse keypoint detector, an intrinsic-calibration position encoder, and promptable prior knowledge-guided attention layers. Given two RGB images of a fixed scene or a moving object, SRPose estimates the relative camera or 6D object pose transformation. Extensive experiments demonstrate that SRPose achieves competitive or superior performance compared to state-of-the-art methods in terms of accuracy and speed, showing generalizability to both scenarios. It is robust to different image sizes and camera intrinsics, and can be deployed with low computing resources.

![image](https://github.com/user-attachments/assets/a8ce5d31-0fa4-4150-92ef-8bc832ae6d8f)


</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.08199) | [⌨️ Code] | [🌐 Project Page]

#### <summary>From 2D to 3D: AISG-SLA Visual Localization Challenge
Authors: Jialin Gao, Bill Ong, Darld Lwi, Zhen Hao Ng, Xun Wei Yee, Mun-Thye Mak, Wee Siong Ng, See-Kiong Ng, Hui Ying Teo, Victor Khoo, Georg Bökman, Johan Edstedt, Kirill Brodt, Clémentin Boittiaux, Maxime Ferrera, Stepan Konev
<details span>
<summary><b>Abstract</b></summary>
Research in 3D mapping is crucial for smart city applications, yet the cost of acquiring 3D data often hinders progress. Visual localization, particularly monocular camera position estimation, offers a solution by determining the camera's pose solely through visual cues. However, this task is challenging due to limited data from a single camera. To tackle these challenges, we organized the AISG-SLA Visual Localization Challenge (VLC) at IJCAI 2023 to explore how AI can accurately extract camera pose data from 2D images in 3D space. The challenge attracted over 300 participants worldwide, forming 50+ teams. Winning teams achieved high accuracy in pose estimation using images from a car-mounted camera with low frame rates. The VLC dataset is available for research purposes upon request via vlc-dataset@aisingapore.org.



</details>

[📃 arXiv:2407](https://arxiv.org/pdf/2407.18590) | [⌨️ Code] | [🌐 Project Page]

#### <summary>Certifying Robustness of Learning-Based Keypoint Detection and Pose Estimation Methods
Authors: Xusheng Luo, Tianhao Wei, Simin Liu, Ziwei Wang, Luis Mattei-Mendez, Taylor Loper, Joshua Neighbor, Casidhe Hutchison, Changliu Liu
<details span>
<summary><b>Abstract</b></summary>
This work addresses the certification of the local robustness of vision-based two-stage 6D object pose estimation. The two-stage method for object pose estimation achieves superior accuracy by first employing deep neural network-driven keypoint regression and then applying a Perspective-n-Point (PnP) technique. Despite advancements, the certification of these methods' robustness remains scarce. This research aims to fill this gap with a focus on their local robustness on the system level--the capacity to maintain robust estimations amidst semantic input perturbations. The core idea is to transform the certification of local robustness into neural network verification for classification tasks. The challenge is to develop model, input, and output specifications that align with off-the-shelf verification tools. To facilitate verification, we modify the keypoint detection model by substituting nonlinear operations with those more amenable to the verification processes. Instead of injecting random noise into images, as is common, we employ a convex hull representation of images as input specifications to more accurately depict semantic perturbations. Furthermore, by conducting a sensitivity analysis, we propagate the robustness criteria from pose to keypoint accuracy, and then formulating an optimal error threshold allocation problem that allows for the setting of a maximally permissible keypoint deviation thresholds. Viewing each pixel as an individual class, these thresholds result in linear, classification-akin output specifications. Under certain conditions, we demonstrate that the main components of our certification framework are both sound and complete, and validate its effects through extensive evaluations on realistic perturbations. To our knowledge, this is the first study to certify the robustness of large-scale, keypoint-based pose estimation given images in real-world scenarios.

![image](https://github.com/user-attachments/assets/8b3818ea-16e0-4a69-8dbc-f564fa5d9e20)


</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.00117) | [⌨️ Code] | [🌐 Project Page]


#### <summary>Generalized Maximum Likelihood Estimation for Perspective-n-Point Problem
Authors: Tian Zhan, Chunfeng Xu, Cheng Zhang, Ke Zhu
<details span>
<summary><b>Abstract</b></summary>
The Perspective-n-Point (PnP) problem has been widely studied in the literature and applied in various vision-based pose estimation scenarios. However, existing methods ignore the anisotropy uncertainty of observations, as demonstrated in several real-world datasets in this paper. This oversight may lead to suboptimal and inaccurate estimation, particularly in the presence of noisy observations. To this end, we propose a generalized maximum likelihood PnP solver, named GMLPnP, that minimizes the determinant criterion by iterating the GLS procedure to estimate the pose and uncertainty simultaneously. Further, the proposed method is decoupled from the camera model. Results of synthetic and real experiments show that our method achieves better accuracy in common pose estimation scenarios, GMLPnP improves rotation/translation accuracy by 4.7%/2.0% on TUM-RGBD and 18.6%/18.4% on KITTI-360 dataset compared to the best baseline. It is more accurate under very noisy observations in a vision-based UAV localization task, outperforming the best baseline by 34.4% in translation estimation accuracy.

![image](https://github.com/user-attachments/assets/ddbc5d7f-dcf1-4d0b-9fbd-4b8a7559a397)


</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.01945) | [⌨️ Code] | [🌐 Project Page]

#### <summary>MambaLoc: Efficient Camera Localisation via State Space Model
Authors: Jialu Wang, Kaichen Zhou, Andrew Markham, Niki Trigoni
<details span>
<summary><b>Abstract</b></summary>
Location information is pivotal for the automation and intelligence of terminal devices and edge-cloud IoT systems, such as autonomous vehicles and augmented reality. However, achieving reliable positioning across diverse IoT applications remains challenging due to significant training costs and the necessity of densely collected data. To tackle these issues, we have innovatively applied the selective state space (SSM) model to visual localization, introducing a new model named MambaLoc. The proposed model demonstrates exceptional training efficiency by capitalizing on the SSM model's strengths in efficient feature extraction, rapid computation, and memory optimization, and it further ensures robustness in sparse data environments due to its parameter sparsity. Additionally, we propose the Global Information Selector (GIS), which leverages selective SSM to implicitly achieve the efficient global feature extraction capabilities of Non-local Neural Networks. This design leverages the computational efficiency of the SSM model alongside the Non-local Neural Networks' capacity to capture long-range dependencies with minimal layers. Consequently, the GIS enables effective global information capture while significantly accelerating convergence. Our extensive experimental validation using public indoor and outdoor datasets first demonstrates our model's effectiveness, followed by evidence of its versatility with various existing localization models. Our code and models are publicly available to support further research and development in this area.

![image](https://github.com/user-attachments/assets/c902f2b7-df6d-4cb2-bba8-4637824356dd)


</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.09680) | [⌨️ Code] | [🌐 Project Page]

#### <summary>ADen: Adaptive Density Representations for Sparse-view Camera Pose Estimation
> *ECCV24 oral*

Authors: Hao Tang, Weiyao Wang, Pierre Gleize, Matt Feiszli
<details span>
<summary><b>Abstract</b></summary>
Recovering camera poses from a set of images is a foundational task in 3D computer vision, which powers key applications such as 3D scene/object reconstructions. Classic methods often depend on feature correspondence, such as keypoints, which require the input images to have large overlap and small viewpoint changes. Such requirements present considerable challenges in scenarios with sparse views. Recent data-driven approaches aim to directly output camera poses, either through regressing the 6DoF camera poses or formulating rotation as a probability distribution. However, each approach has its limitations. On one hand, directly regressing the camera poses can be ill-posed, since it assumes a single mode, which is not true under symmetry and leads to sub-optimal solutions. On the other hand, probabilistic approaches are capable of modeling the symmetry ambiguity, yet they sample the entire space of rotation uniformly by brute-force. This leads to an inevitable trade-off between high sample density, which improves model precision, and sample efficiency that determines the runtime. In this paper, we propose ADen to unify the two frameworks by employing a generator and a discriminator: the generator is trained to output multiple hypotheses of 6DoF camera pose to represent a distribution and handle multi-mode ambiguity, and the discriminator is trained to identify the hypothesis that best explains the data. This allows ADen to combine the best of both worlds, achieving substantially higher precision as well as lower runtime than previous methods in empirical evaluations.

![image](https://github.com/user-attachments/assets/1dbe5361-507d-4c73-8306-64ca81257331)


</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.09042) | [⌨️ Code] | [🌐 Project Page]

#### <summary>GSLoc: Efficient Camera Pose Refinement via 3D Gaussian Splatting
Authors: Changkun Liu, Shuai Chen, Yash Bhalgat, Siyan Hu, Zirui Wang, Ming Cheng, Victor Adrian Prisacariu, Tristan Braud
<details span>
<summary><b>Abstract</b></summary>
We leverage 3D Gaussian Splatting (3DGS) as a scene representation and propose a novel test-time camera pose refinement framework, GSLoc. This framework enhances the localization accuracy of state-of-the-art absolute pose regression and scene coordinate regression methods. The 3DGS model renders high-quality synthetic images and depth maps to facilitate the establishment of 2D-3D correspondences. GSLoc obviates the need for training feature extractors or descriptors by operating directly on RGB images, utilizing the 3D vision foundation model, MASt3R, for precise 2D matching. To improve the robustness of our model in challenging outdoor environments, we incorporate an exposure-adaptive module within the 3DGS framework. Consequently, GSLoc enables efficient pose refinement given a single RGB query and a coarse initial pose estimation. Our proposed approach surpasses leading NeRF-based optimization methods in both accuracy and runtime across indoor and outdoor visual localization benchmarks, achieving state-of-the-art accuracy on two indoor datasets.

![image](https://github.com/user-attachments/assets/a503a7df-9db5-4c70-9776-7b6a1e75e0ee)

</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.11085) | [⌨️ Code] | [🌐 Project Page](https://gsloc.active.vision/)

#### <summary>FUSELOC: Fusing Global and Local Descriptors to Disambiguate 2D-3D Matching in Visual Localization
Authors: Son Tung Nguyen, Alejandro Fontan, Michael Milford, Tobias Fischer
<details span>
<summary><b>Abstract</b></summary>
Hierarchical methods represent state-of-the-art visual localization, optimizing search efficiency by using global descriptors to focus on relevant map regions. However, this state-of-the-art performance comes at the cost of substantial memory requirements, as all database images must be stored for feature matching. In contrast, direct 2D-3D matching algorithms require significantly less memory but suffer from lower accuracy due to the larger and more ambiguous search space. We address this ambiguity by fusing local and global descriptors using a weighted average operator within a 2D-3D search framework. This fusion rearranges the local descriptor space such that geographically nearby local descriptors are closer in the feature space according to the global descriptors. Therefore, the number of irrelevant competing descriptors decreases, specifically if they are geographically distant, thereby increasing the likelihood of correctly matching a query descriptor. We consistently improve the accuracy over local-only systems and achieve performance close to hierarchical methods while halving memory requirements. Extensive experiments using various state-of-the-art local and global descriptors across four different datasets demonstrate the effectiveness of our approach. For the first time, our approach enables direct matching algorithms to benefit from global descriptors while maintaining memory efficiency.

![image](https://github.com/user-attachments/assets/79325260-52eb-491c-adcf-e661b7394cce)

</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.12037) | [⌨️ Code] | [🌐 Project Page]

#### <summary>Visual Localization in 3D Maps: Comparing Point Cloud, Mesh, and NeRF Representations
Authors: intong Zhang, Yifu Tao, Jiarong Lin, Fu Zhang, Maurice Fallon
<details span>
<summary><b>Abstract</b></summary>
This paper introduces and assesses a cross-modal global visual localization system that can localize camera images within a color 3D map representation built using both visual and lidar sensing. We present three different state-of-the-art methods for creating the color 3D maps: point clouds, meshes, and neural radiance fields (NeRF). Our system constructs a database of synthetic RGB and depth image pairs from these representations. This database serves as the basis for global localization. We present an automatic approach that builds this database by synthesizing novel images of the scene and exploiting the 3D structure encoded in the different representations. Next, we present a global localization system that relies on the synthetic image database to accurately estimate the 6 DoF camera poses of monocular query images. Our localization approach relies on different learning-based global descriptors and feature detectors which enable robust image retrieval and matching despite the domain gap between (real) query camera images and the synthetic database images. We assess the system's performance through extensive real-world experiments in both indoor and outdoor settings, in order to evaluate the effectiveness of each map representation and the benefits against traditional structure-from-motion localization approaches. Our results show that all three map representations can achieve consistent localization success rates of 55% and higher across various environments. NeRF synthesized images show superior performance, localizing query images at an average success rate of 72%. Furthermore, we demonstrate that our synthesized database enables global localization even when the map creation data and the localization sequence are captured when travelling in opposite directions. Our system, operating in real-time on a mobile laptop equipped with a GPU, achieves a processing rate of 1Hz.

![image](https://github.com/user-attachments/assets/4fb70b48-fea4-4acd-981a-f991fd363ae9)

</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.11966) | [⌨️ Code] | [🌐 Project Page]


#### <summary>Reprojection Errors as Prompts for Efficient Scene Coordinate Regression
>*It is observed that regions with low reprojection errors tend to have relatively higher inlier ratios, indicating a greater impact on the estimated camera poses*

Authors: Ting-Ru Liu, Hsuan-Kung Yang, Jou-Min Liu, Chun-Wei Huang, Tsung-Chih Chiang, Quan Kong, Norimasa Kobori, Chun-Yi Lee
<details span>
<summary><b>Abstract</b></summary>
Scene coordinate regression (SCR) methods have emerged as a promising area of research due to their potential for accurate visual localization. However, many existing SCR approaches train on samples from all image regions, including dynamic objects and texture-less areas. Utilizing these areas for optimization during training can potentially hamper the overall performance and efficiency of the model. In this study, we first perform an in-depth analysis to validate the adverse impacts of these areas. Drawing inspiration from our analysis, we then introduce an error-guided feature selection (EGFS) mechanism, in tandem with the use of the Segment Anything Model (SAM). This mechanism seeds low reprojection areas as prompts and expands them into error-guided masks, and then utilizes these masks to sample points and filter out problematic areas in an iterative manner. The experiments demonstrate that our method outperforms existing SCR approaches that do not rely on 3D information on the Cambridge Landmarks and Indoor6 datasets.

![image](https://github.com/user-attachments/assets/8141b540-a458-47b0-8ef9-a00546e09e7c)

</details>

[📃 arXiv:2409](https://arxiv.org/pdf/2409.04178) | [⌨️ Code] | [🌐 Project Page](https://tingru0203.github.io/egfs/)

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

## Others


#### <summary>Guiding Local Feature Matching with Surface Curvature
> *Curvature Extractor*

Authors: Shuzhe Wang, Juho Kannala, Marc Pollefeys, Daniel Barath
<details span>
<summary><b>Abstract</b></summary>
We propose a new method, named curvature similarity extractor (CSE), for improving local feature matching across images. CSE calculates the curvature of the local 3D surface patch for each detected feature point in a viewpoint-invariant manner via fitting quadrics to predicted monocular depth maps. This curvature is then leveraged as an additional signal in feature matching with off-the-shelf matchers like SuperGlue and LoFTR. Additionally, CSE enables end-to-end joint training by connecting the matcher and depth predictor networks. Our experiments demonstrate on large-scale real-world datasets that CSE continuously improves the accuracy of state-of-the-art methods. Fine-tuning the depth prediction network further enhances the accuracy. The proposed approach achieves state-of-the-art results on the ScanNet dataset, showcasing the effectiveness of incorporating 3D geometric information into feature matching.

![image](https://github.com/user-attachments/assets/a83eaa42-d9b1-4162-9cc4-a73746dec12f)

</details>

[📃 arXiv:xxxx](https://openaccess.thecvf.com/content/ICCV2023/papers/Wang_Guiding_Local_Feature_Matching_with_Surface_Curvature_ICCV_2023_paper.pdf) | [⌨️ Code](https://github.com/AaltoVision/surface-curvature-estimator/) | [🌐 Project Page]

#### <summary>ConDense: Consistent 2D/3D Pre-training for Dense and Sparse Features from Multi-View Images
>*Due to the biased and scarcer nature of the existing multi-view image datasets, if we optimize the networks based only on this 2D-3D consistency loss, the feature quality may degrade due to trivial solutions and biased data distribution.*

Authors: Xiaoshuai Zhang, Zhicheng Wang, Howard Zhou, Soham Ghosh, Danushen Gnanapragasam, Varun Jampani, Hao Su, Leonidas Guibas
<details span>
<summary><b>Abstract</b></summary>
To advance the state of the art in the creation of 3D foundation models, this paper introduces the ConDense framework for 3D pre-training utilizing existing pre-trained 2D networks and large-scale multi-view datasets. We propose a novel 2D-3D joint training scheme to extract co-embedded 2D and 3D features in an end-to-end pipeline, where 2D-3D feature consistency is enforced through a volume rendering NeRF-like ray marching process. Using dense per pixel features we are able to 1) directly distill the learned priors from 2D models to 3D models and create useful 3D backbones, 2) extract more consistent and less noisy 2D features, 3) formulate a consistent embedding space where 2D, 3D, and other modalities of data (e.g., natural language prompts) can be jointly queried. Furthermore, besides dense features, ConDense can be trained to extract sparse features (e.g., key points), also with 2D-3D consistency -- condensing 3D NeRF representations into compact sets of decorated key points. We demonstrate that our pre-trained model provides good initialization for various 3D tasks including 3D classification and segmentation, outperforming other 3D pre-training methods by a significant margin. It also enables, by exploiting our sparse features, additional useful downstream tasks, such as matching 2D images to 3D scenes, detecting duplicate 3D scenes, and querying a repository of 3D scenes through natural language -- all quite efficiently and without any per-scene fine-tuning.

![image](https://github.com/user-attachments/assets/623bdf37-4aaf-42f7-ac78-014d7e525abc)


</details>

[📃 arXiv:2408](https://arxiv.org/pdf/2408.17027) | [⌨️ Code] | [🌐 Project Page]



<br>
<br>

#### <summary>
Authors: 
<details span>
<summary><b>Abstract</b></summary>


![image]()

</details>

[📃 arXiv:2408] | [⌨️ Code] | [🌐 Project Page]
