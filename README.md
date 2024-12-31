# GM-PHD-Tracker
This code implements the paper [Occlusion-robust online multi-object visual tracking using a GM-PHD filter with CNN-based re-identification](https://www.sciencedirect.com/science/article/pii/S1047320321001814); 
however, we used different reid model for extracting deep appearance representations in this implementation. In addition 
to using MOT Challenge and HiEve public detections, an option to use YOLOv8 custom detections is also included.



## Overview
In this work, we propose an online multi-object visual tracker using a Gaussian mixture Probability Hypothesis Density 
(GM-PHD) filter and deep appearance learning. The GM-PHD filter has a linear complexity with the number of objects and 
observations while estimating the states and cardinality of time-varying number of objects, however, it is susceptible 
to miss-detections and does not include the identity of objects. We use visual-spatio-temporal information obtained from 
object bounding boxes and deeply learned appearance representations to perform estimates-to-tracks data association for 
target labeling as well as formulate an augmented likelihood and then integrate into the update step of the GM-PHD 
filter. We also employ additional unassigned tracks prediction after the data association step to overcome the 
susceptibility of the GM-PHD filter towards miss-detections caused by occlusion. Extensive evaluations on MOT16, MOT17, 
MOT20 and HiEve benchmark data sets show that our tracker significantly outperforms several state-of-the-art trackers in 
terms of tracking accuracy and identification.


The qualitative result (demo) of the GM-PHD-Tracker on MOT17-03 data with a SDP public detections is shown below. 

![](./assets/demo_MOT17_03_SDP.gif)



## Installation
Git clone this repo and install dependencies to have the same environment configuration as the one we used.

```shell
git clone https://github.com/nathanlem1/GM-PHD-Tracker.git
cd GM-PHD-Tracker/
pip install -r requirements.txt
```

The code was tested using torch 2.2.2+cu118 and torchvision 0.17.2+cu118. You can install torch and matched torchvision from [pytorch.org](https://pytorch.org/get-started/locally/).


## Data Preparation 
We used [MOT16](https://motchallenge.net/data/MOT16/), [MOT17](https://motchallenge.net/data/MOT17/), 
[MOT20](https://motchallenge.net/data/MOT20/), [HiEve](http://humaninevents.org/) and [DanceTrack](https://github.com/DanceTrack/DanceTrack) 
benchmark data sets. Download these datasets from their corresponding links and put them in `datasets` folder created under the 
`GM-PHD-Tracker` folder.


## Tracking
To perfrom tracking, you need to download the reid model from 
[here](https://drive.google.com/file/d/1XWXzfcSrE2ie9TSGlIqQEeFfXE2lMmDe/view?usp=drive_link) and put it in `pretrained` 
folder under the `GM-PHD-Tracker`. You also need to set the correct data type you need to run the tracker on in 
`config.yaml`. Then run the following code on terminal (using public detections):

```shell
python tracker.py --base_data ./datasets --base_result ./results/trackers --reid_path ./pretrained/reid_model.pth --detections_type " "
```

There are two detection types to use: using MOT Challenge and HiEve public detections OR YOLOv8 custom detections. 
Please look into the code for more details, particularly `config.yaml` for parameters setting. Run the following 
code on terminal (for using YOLOv8 detections):

```shell
python tracker.py --base_data ./datasets --base_result ./results/trackers --reid_path ./pretrained/reid_model.pth --detections_type "yolo"
```

## Evaluation
To evaluate on MOT16, MOT17, MOT20, HiEve or DanceTrack train datasets, you can run the following code on terminal (which 
uses [py-motmetrics](https://github.com/cheind/py-motmetrics)):

```shell
python evaluate.py --base_data ./datasets --base_result ./results/trackers
```

Please look into the code for more details, particularly `config.yaml` for parameters setting.

You can also use the official MOTChallenge evaluation code from [TrackEval](https://github.com/JonathonLuiten/TrackEval) 
to evaluate the MOT16, MOT17, MOT20, HiEve and DanceTrack `train` datasets. DanceTrack also has `val` dataset, in which 
case you need to use `val` for `--SPLIT_TO_EVAL`.

```shell
python TrackEval/scripts/run_mot_challenge.py --BENCHMARK MOT16 --SPLIT_TO_EVAL train --TRACKERS_TO_EVAL GMPHD1 --METRICS HOTA CLEAR Identity VACE --GT_FOLDER results/gt/ --TRACKERS_FOLDER results/trackers/ --USE_PARALLEL False --NUM_PARALLEL_CORES 1
```


## Citation

If you use this code for your research, please cite our paper.

```
@article{Nathanael_JVCI2021,
author = {Nathanael L. Baisa},
title = {Occlusion-robust online multi-object visual tracking using a {GM-PHD} filter with {CNN}-based re-identification},
journal = {Journal of Visual Communication and Image Representation},
volume = {80},
pages = {103279},
year = {2021},
issn = {1047-3203},
doi = {https://doi.org/10.1016/j.jvcir.2021.103279},
url = {https://www.sciencedirect.com/science/article/pii/S1047320321001814},
}
```