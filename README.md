[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/jdgMly3P)
# Segmentation

This is our third lab, and it is about Semantic, Instance and Panoptic Segmentation. We will use a portion of the [COCO dataset](https://cocodataset.org/#home) to produce segmentations. We will use two architectures. The first one will be based on the well-known [Mask R-CNN](https://github.com/facebookresearch/maskrcnn-benchmark/) architecture. Recall that a couple of architectures before Mask R-CNN preceded this masterpiece. Secondly, we will use a Transformer-based architecture. In our case, we will use a [Swin Transformer](https://openaccess.thecvf.com/content/ICCV2021/papers/Liu_Swin_Transformer_Hierarchical_Vision_Transformer_Using_Shifted_Windows_ICCV_2021_paper.pdf). We want to learn how these methods work and how we can adapt them for our benefit. 

Please pay attention! In this lab, we will use three segmentation types. We aim to determine how training for an specific type of segmentation changes the performance for the others. That is, we would like to study how the Instance Segmentation performance changes if we train only for Instance Segmentation (without stuff) or Panoptic Segmentation (with stuff). 

## Semantic Segmentation
This task aims at classifying all pixel within an image with a semantic label according to the visual entity it belong to.

## Instance Segmentation
This task mixes the idea of detecting individual objects, brought from Object Detection, with the idea behind Semantic Segmentation. Instance Segmentation aims to assign a semantic category to every salient object within an image, differentiating between objects from the same category. That means we would like to know how many *instances* of an object we have in the image. 
<p align="center"><img src="https://miro.medium.com/v2/resize:fit:720/format:webp/1*CkbIVSDbiQFrwyGYfQwD9g.png"/></p>

## Panoptic Segmentation
The [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868.pdf) consists of assigning a number to each Instance of the same category, but it does not neglect non-salient objects in an image. Therefore, Panoptic Segmentation gives us a holistic way to understand a scene. 
![](images/panop.png)

## Dataset
You do not have to download it; the Data is already stored on the machine 
```
BCV002: /media/SSD1/vision/coco/ Train & Val
```
The images we will use in this laboratory were selected from the original COCO Dataset. However, the annotations are not ready to use, so part of your work should be focused on curating them for the segmentation setup you want to study. You will have access to the filename of the images, but you need to find the annotation for each segment. I HIGHLY recommend checking (in fact, it is not an option) [this explanation](https://youtu.be/h6s61a_pqfM) of the COCO annotation format. 

As you know, we are using a subset of the COCO dataset. As I mentioned, you will choose 30 categories for the original 133. Bear in mind that you need to have at least eight stuff categories. Now, you know how to manipulate the COCO dataset to reach a point where you can choose the categories you want. I recommend choosing first the set for panoptic segmentation and then filtering it for Instance Segmentation. 


## Your turn

You will run Mask R-CNN and Swin Transformer on the COCO segmentation dataset. Here you'll focus on the training schedule and parameters (lr, input image size, optimizer, etc.). 

Main results for **Instance Segmentation**:
- Comparison of Mask R-CNN using a backbone with and without pre-trained weights.
- At least 2 experiments based on Mask R-CNN after choosing the most appropriate backbone (the baseline does not count!).
- At least 2 experiments based on the Swin Transformer (the baseline does not count!).
- Compare CNN and Transformer!

Main results for **Panoptic Segmentation**:
For this experiments you should report results for the three tasks: Semantic, Instance, and Panoptic Segmentation:
- Compare the CNN and the Transformer architecture without moving any parameter. Does the performance decrease?
- Tune in the parameters to increase the performance of both architectures. 
- Report at least one Panoptic metric. It should be different from the normal mean AP. I recommend reading the original paper to find out how to calculate them.


## Deadline
**April 8th, 11:59pm** The report should be brief (no more than 5 pages, you can still use additional pages for figures and references). 
Don't forget to upload your report and scripts. Make sure you upload your *train.py*, *test.py*, and *demo.py* scripts. These scripts should perform the corresponding segmentation on training, testing, or demo, according to the input argument (use [argparse](https://docs.python.org/3/library/argparse.html)).
The demo script will load your models' trained weights and should segment a RANDOM image of the test dataset. In this case, this script will select the best model for the CNN and Transformer versions. The script should SAVE the results in a folder called "RESULTS/," and it should print the metrics on the terminal for both methods.


## References
https://hasty.ai/docs/mp-wiki/model-families/instance-segmentor

