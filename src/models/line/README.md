# EHM (Extreme Heatpoints Model) for lines

The model is based on the code of [HRNet semantic segmentation](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/HRNet-OCR) with [GCCPM pose estimation](https://github.com/Daniil-Osokin/gccpm-look-into-person-cvpr19.pytorch/) model keypoints refinemen stage. The core of the model used in the challenge is HRNetV2-w48 backbone.

The model predicts 23 heatpoint pairs for lines, from 23 heatmap layers.
semi-circles and circles are ignored.
The target tensor consisted of a 2D heatmap for each point pairs,
where Gaussian peaks were positioned at the keypoint locations. 

Since there are two point from each heatmap, we extract the first extreme point 
with the highest probability. After that a Gaussian kernel was used to mask the 
first extreme point area, followed by a similar process to find the second extreme 
point.

### Prepare the dataset
The dataset shall be organised in the following structure.

```bash
|-- soccernet-calibration-sportlight
    |-- data
        |-- dataset
        |   |   |-- valid
        |   |   |   |-- 00000.jpg
        |   |   |   |-- 00000.json
        |   |   |   | ...
        |   |   |-- train
        |   |   |   |-- 00000.jpg
        |   |   |   |-- 00000.json
        |   |   |   | ...
```

### Visualise extreme points
In some annotations, there are more than 2 points, an algorithm is designed to
find extreme points for lines. User can run the following demo code:

```bash
python src/datatools/line.py
```
The following plot will be generated for visual inspection.
![extreme point demo](/readme_img/line_example.png)


## Train the extreme heatpoint model (EHM)
- The training config file for EHM is `src/models/line/train_config.yaml`
- The EHM is based on HRNetV2-w48 backbone, it is possible to use a pretraind weight in `/src/models/line/model_config/hrnet_w48.yaml`
- Run `python src/models/line/train.py` to start training.

## Assemble line model result
The result of the line model could be saved for each image from a folder, in a pickle format.
run:

```bash
python src/utils/export_line_result.py \
--image-folder /workdir/data/dataset/test \
--result-file /workdir/data/result/line_model_result.pkl \
--model line_model_pytorch.pth \
--device cuda:0  \
--sigma 3.0 \
--prob-thre 0. \
--no-vis
```

Uncomment `no-vis` and increase `prob-thre` to see the visualisation on images. 

Here set `prob-thre` to save all result, a probability search might be conducted in 
later stage when combining model result with other model results.