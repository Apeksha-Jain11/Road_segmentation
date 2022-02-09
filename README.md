# Road_Segmentation


Road segmentation.ipynb is a file for creating 256*256 patches of data and mask and creting train and test split.
Training.ipynb file is performing preprocesing ,generating batches and creating model using Resnet-34 as bacbone with Unet. It uses imagenet pretrained weights for initialization.
Testing.py is inferencing on testing data 
score.py is calculating Accuracy,Mean IOU ,Dice
