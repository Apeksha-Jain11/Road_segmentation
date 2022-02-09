# Road_Segmentation


Road segmentation.ipynb is a file for creating 256*256 patches of data and mask and creting train and test split.
Training.ipynb file is performing preprocesing ,generating batches and creating model using Resnet-34 as bacbone with Unet. It uses imagenet pretrained weights for initialization.
Testing.py is inferencing on testing data 
score.py is calculating Accuracy,Mean IOU ,Dice

Model Information:
Trained for 25 epochs using 19175 images of size 256*256*3
batch_size=8
optimizer = Adam
Model= U-net with Resnet-34 backbone
pretrained_weights = Imagenet

Performance numbers on 325 testing images of size 256*256*3
CLASS: 0: Accuracy: 0.9677574275090144, IOU: 0.9658876362010166, Dice: 0.9826478567895651
CLASS: 1: Accuracy: 0.9677574275090144, IOU: 0.6296310877744029, Dice: 0.7727283708539138
Mean Accuracy: 0.9677574275090144, Mean IOU: 0.7977593619877097, Mean Dice: 0.8776881138217394


