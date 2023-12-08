# Proactive-Object-Detection

# PrObeD:Proactive Object Detection Wrapper
Official Pytorch implementation of Neurips 2023 paper "PrObeD:Proactive Object Detection Wrapper".

[Vishal Asnani](https://github.com/vishal3477), [Abhinav Kumar](https://sites.google.com/view/abhinavkumar), Suya You, [Xiaoming Liu](https://www.cse.msu.edu/~liuxm/index2.html)

[Paper + Supplementary](https://arxiv.org/abs/2310.18788)


![alt text](https://github.com/vishal3477/Proactive-Object-Detection/blob/main/images/overview_4.png?raw=true)
## Prerequisites

Please look at the environment.yaml file for setting up the environment. 

## Getting Started

## Datasets 
- We use multiple datasets in our paper. For generic object detection (GOD), we use [MS-COCO 2017](https://cocodataset.org/#home). For camouflaged object detection (COD), we use [CAMO](https://sites.google.com/view/ltnghia/research/camo), [COD10K](https://dengpingfan.github.io/pages/COD.html), and [NC4K](https://github.com/JingZhang617/COD-Rank-Localize-and-Segment)
- Please download the datasets from the above links and pepare them according to the requirements of differennt object detector.


## Training
- We incorporate our wrapper on the official implementation of all the detectors. Please refer to the official repositories for all the detectors as mentioned in the papers.
- We show the training code for Faster R-CNN (GOD detector) and DGNet (COD detector) incorporating our detector.

### Faster R-CNN

- Please download the Faster R-CNN repository from [here](https://github.com/jwyang/faster-rcnn.pytorch).
- Setup the code and data according to the official repository instructions.
- We change the training code with train_faster_rcnn.py
```
python train_faster_rcnn.py --dataset coco --net res101 --bs 8 --nw 1 --lr 0.000001 --lr_decay_step 4 --cuda 
```

### DGNet
- Please download the DGNet repository pytorch version from [here](https://github.com/GewelsJI/DGNet/tree/main/lib_pytorch).
- Setup the code and data according to the official repository instructions.
- We change the training code with train_dgnet.py
```
python train_dgnet.py --gpu_id 0 --model DGNet
```

## Pre-trained model
The pre-trained model for Faster R-CNN and DGNet can be downloaded from below:

Model     | Link 
---------|--------
Faster-RCNN | [Model](https://drive.google.com/drive/folders/1y3oi3cVDT0cRz66Q97IFNGMoCaqRM80L?usp=sharing)    
DGNet | Coming soon!!  

## Testing using pre-trained models
- Download the pre-trained model using the above links.
- Download the evaluation toolbox for COD [here](https://github.com/GewelsJI/DGNet/tree/main/lib_pytorch).
- Provide the model path in the code
- Run the code as shown below:

### Faster R-CNN
```
python 14_testnet_2.py --dataset coco --net res101 --cuda --model_path "MODEL PATH" 
```

### DGNet
Run the below command to generate and save the loca;ization maps. 
```
python test_dgnet_loc.py --gpu_id 0 --model DGNet --model_path "MODEL PATH" 
```
After this, run the evaluation script eval_dgnet.py with the paths of validation data ground-truth and savd predcition to estimate and print all the metrics. 

```
python eval_dgnet.py
```
## Sample pseudo code on how to use our wrapper with any detector

Coming soon!!

If you would like to use our work, please cite:
```
@inproceedings{asnani2023probed,
  title={PrObeD: Proactive Object Detection Wrapper},
  author={Asnani, Vishal and Kumar, Abhinav and You, Suya and Liu, Xiaoming},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```
