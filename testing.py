import cv2
import numpy as np
import os

from matplotlib import pyplot as plt
from patchify import patchify, unpatchify
from PIL import Image
import segmentation_models as sm

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()
from tensorflow.keras.models import load_model

model = load_model("C:/Users/ajape/Downloads/road_segmentation_25_epochs_RESNET_backbone_batch16.h5", compile=False)
testing_image_path = "C:/Users/ajape/Downloads/archive/road_segmentation_ideal/testing/input/"
testing_label_path =  "C:/Users/ajape/Downloads/archive/road_segmentation_ideal/testing/output/"
# size of patches
patch_size = 256
n_classes = 2



BACKBONE = 'resnet34'
preprocess_input = sm.get_preprocessing(BACKBONE)
testing_images = os.listdir(testing_image_path)

for image_name in testing_images:
    #######Reading input image and maskg#####
    img = cv2.imread(testing_image_path + "/" + image_name, 1 )
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    original_mask = cv2.imread(testing_label_path + "/" + image_name, 1)
    original_mask = original_mask[:, :, 0]  # Use only single channel...

    
    #######Converting the image to 256*256 patches for inferencing#####
    size_X = (img.shape[0] // patch_size) * patch_size
    size_Y = (img.shape[1] // patch_size) * patch_size
    image = Image.fromarray(img)
    image = image.crop((0, 0, size_X, size_Y)) # Cropping image to 1280*280( Nearest size divisible by 256)
    image = np.array(image)
    patches_image = patchify(image, (256, 256, 3), step=256)
    patches_output= np.zeros((1280,1280),dtype="uint8")
    # print(patches_image)
    patches_list = []

    for i in range(patches_image.shape[0]):
        for j in range(patches_image.shape[1]):
            single_patch_img = patches_image[i, j, :, :]
            single_patch_img = single_patch_img[0]  # Drop the extra unecessary dimension that patchify adds.
            single_patch_img = scaler.fit_transform(single_patch_img.reshape(-1, single_patch_img.shape[-1])).reshape(single_patch_img.shape)
            single_patch_img = preprocess_input(single_patch_img)
            output = model.predict(np.expand_dims(single_patch_img,axis=0))
            output = np.argmax(output, axis=3)
            patches_output[i*256:i*256+256,j*256:j*256+256] = output #Stiching the patches
    path2save= 'C:/Users/ajape/Downloads/archive/road_segmentation_ideal/testing/Model_output_test/'
    plt.imsave(path2save+image_name, image)
    plt.imsave(path2save+image_name.split('.')[0]+"_output_"+".png", patches_output)

    # plt.figure(figsize=(12, 12))
    # plt.subplot(131)
    # plt.title('input_img')
    # plt.imshow(img)
    # plt.subplot(132)
    # plt.title('Testing Label')
    # plt.imshow(original_mask)
    # plt.subplot(133)
    # plt.title('Prediction')
    # plt.imshow(patches_output)
    # plt.show()


