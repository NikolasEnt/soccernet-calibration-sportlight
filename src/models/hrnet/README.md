# HRnet

The model is based on the code of [HRNet semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR) with [GCCPM pose estimation](https://github.com/Daniil-Osokin/gccpm-look-into-person-cvpr19.pytorch/) model keypoints refinemen stage. The core of the model used in the challenge is HRNetV2-w48 backbone.

![HRNet-based keypoint detection](/readme_img/hrnet_architecture.jpg)
_The model architecture._

The model predicts 57 keypoints, depicted below. The target tensor consisted of a 2D heatmap for each point, where Gaussian peaks were positioned at the keypoint locations. Locations of the keypoints, including the approach to refine lines intersections localtions and ellipses tangent points are computed using code in [/src/datatools/](/src/datatools/).

[EDA notebook](/notebooks/EDA.ipynb) contains visualization of the raw annotations and the derived keypoints, utilized in the model training process.

[Ellipse notebook](/notebooks/ellipse.ipynb) contains visualization of the ellipse-line intersection and ellipse tangent point calculation algorithms used in construction of the targets.

An additional target channel was included, which represented the inverse of the maximal value among the other target feature maps. This ensured that the final target tensor summed up to 1.0 at each spatial point.

![HRNet-based keypoint detection](/readme_img/pitch_pattern.jpg)
_The pitch pattern with all the points depicted. Red - line-line intersection, blue - line-conic intersection, purple - conic tangent point, dark-green - other points projected by homography._

The details on the model, training procedure and transforms are available in the blog [post](https://nikolasent.github.io/deeplearning/computervision/2023/06/20/SoccerNet-Camera-Calibration-2023.html).

Papers:
* [Global Context for Convolutional Pose Machines](https://arxiv.org/pdf/1906.04104.pdf)
* [Deep High-Resolution Representation Learning for Visual Recognition](https://arxiv.org/pdf/1908.07919.pdf)
* [High-Resolution Representations for Labeling Pixels and Regions](https://arxiv.org/pdf/1904.04514.pdf)