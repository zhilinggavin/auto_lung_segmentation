# Automatic Lung segmentation
## Introduction
The automatic segmentation of the lung region in chest X-rays (CXR) aids doctors in diagnosing various lung diseases. However, severe lung deformities and indistinct lung boundaries caused by serious conditions can lead to errors in the segmentation model.

## Data and task description
![data-example](images/data-example.png)

Dataset consists of collected from public available chest X-Ray (CXR) images.
Overall amount of images is 800 meanwhile labeled only 704 of them.
Whole dataset was randomly divided into train (0.8 of total) validation (0.1 splited from train) and test parts.

The main task is to implement pixel-wise segmentation on the available data to detect lung area.

## Proposed solution
The most obvious solution for semantic segmentation problems is UNet - fully convolutional network with an encoder-decoder path. High-resolution features from the contracting path are combined with the upsampled output in order to predict more precise output based on this information, which is the main idea of this architecture.

Softmax function was applied to model output and negative log-likelihood loss was used to train network.
Optimization criterion - Adam with 0.0005 learning rate.

Some kinds of data augmentation were used: horizontal and vertical shift, minor zoom and padding.
All images and masks were resized to 512x512 size before passing the network.
To improve performance was decided to use pretrained on ImageNet encoder from vgg11 network.
This approach slightly improves performance and greatly accelerate network convergence.
Vanilla unet configuration doesn't have batch normalization. Nowadays it is used almost every time, so it was added to improve network convergence too.
Such network configuration outperforms other variations of unet without batch norm and pretrained weights on validation dataset so it was chosen for final evaluation

Networks were trained on a batch of 4 images during more than 50 epochs on average.

After 40 epoch network stops to improve validation score and network began to overfit.


- unet-2v: simple unet + augmentation

```
test loss: 0.0634, test jaccard: 0.9110, test dice: 0.9520
```

- unet-6v: pretrained vgg11 encoder + batch_norm + bilinear upscale + augmentation

```
test loss: 0.0530, test jaccard: 0.9268, test dice: 0.9611
```


## Evaluation
For evaluation of model output was Jaccard and Dice metrics, well known for such kind of computer vision tasks.

Evaluation was performed on test dataset, which was not used during training phase. 

There are the best-achived results: Jaccard score - **0.9268**, Dice score - **0.9611**.

Some you obtained results could see on the figure below.

![obtained-results](images/obtained-results.png)

## More Image Processing Samples for Future Task
![obtained-results](images/box-sample-Cardiomegaly.png)

## References
- https://arxiv.org/pdf/1505.04597.pdf - U-Net: Convolutional Networks for Biomedical Image Segmentation
- https://arxiv.org/pdf/1811.12638.pdf - Towards Robust Lung Segmentation in Chest Radiographs with Deep Learning
- https://arxiv.org/pdf/1801.05746.pdf - TernausNet: U-Net with VGG11 Encoder Pre-Trained on ImageNet for Image Segmentation
- https://arxiv.org/pdf/1708.00710.pdf - Accurate Lung Segmentation via Network-WiseTraining of Convolutional Networks