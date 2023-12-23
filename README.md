## SoccerNet Camera Callibration 2023

The repository contains the 1st place solution for SoccerNet Camera Calibration 2023, one of the challenges held at CVPR 2023.

Technical details of the approach are available in [Top-1 solution of SoccerNet Camera Calibration Challenge 2023](https://nikolasent.github.io/deeplearning/computervision/2023/06/20/SoccerNet-Camera-Calibration-2023.html).

The solution was developed by Sportlight Technology team: [Nikolay Falaleev](https://github.com/NikolasEnt) and [Ruilong Chen](https://github.com/ruilongml).

<iframe width="560" height="315" src="https://www.youtube-nocookie.com/embed/bP72jfyecrw" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>

## Quick start guide

The environment is provided as a Docker image, built it with `make build`. Use `make run` commamd to start the container.



## Project structure

* `src` - The project's source directory.
* `notebooks` - Jupyter notebooks.
* `data` - The project's storage for required files.
  * `data/experiments/` - Folder with individual experiments results and artifacts (each experiment has its individual folder in this location).
  * `data/dataset/` - Folder with `challenge`, `test`, `train` and `valid` data from the challenge organizers. Use the official [development kit](https://github.com/SoccerNet/sn-calibration) to get the datasets.
* `baseline` - The code of the baseline, adopted from the official [development kit](https://github.com/SoccerNet/sn-calibration). It is used for data handling and evaluation metrics.

## Useful links

* [https://github.com/SoccerNet/sn-calibration](https://github.com/SoccerNet/sn-calibration) Challenge discription and the baseline.
* [https://www.soccer-net.org/tasks/camera-calibration](https://www.soccer-net.org/tasks/camera-calibration) Challenge homepage.
* [https://arxiv.org/abs/2309.06006](https://arxiv.org/abs/2309.06006) Soccernet 2023 results report.
* [Evaluation server](https://eval.ai/web/challenges/challenge-page/1946/overview)