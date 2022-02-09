import numpy as np
import cv2
import sklearn.metrics as metrics
from pathlib import Path
import os
import matplotlib.pyplot as plt

# Macro Definition
# LABEL_PATH = 'X:\\Gargi\\JTC_Heart\\Parameters\\Vacuolation\\TrainData\\Vacuoles\\TrainingData\\20x\\rgbTile\\vallidationdata\\Target\\'
# OUTPUT_PATH = 'X:\\Gargi\\JTC_Heart\\Parameters\\Vacuolation\\TrainData\\Vacuoles\\TrainingData\\20x\\rgbTile\\vallidationdata\\Predicted\\'

LABEL_PATH = 'C:/Users/ajape/Downloads/archive/road_segmentation_ideal/testing/output/'

OUTPUT_PATH = 'C:/Users/ajape/Downloads/archive/road_segmentation_ideal/testing/Model_output/'

NO_OF_CLASSES = 2

LABEL_SUFFIX = '.png'
LABEL_SUFFIX = '.png'
OUTPUT_SUFFIX = '_modlabel.png'
OUTPUT_SUFFIX = '_modlabel.png'
''

confusion_matrix = np.zeros([NO_OF_CLASSES, NO_OF_CLASSES])

output_file_name_list = os.listdir(LABEL_PATH)

for output_file_name in output_file_name_list:
    print(output_file_name)
    ######Read ground truth and model_output#######
    img_name = os.path.basename(output_file_name)
    ground_truth = cv2.imread(LABEL_PATH +output_file_name, cv2.IMREAD_GRAYSCALE)
    ground_truth_crop = ground_truth[:1280,:1280]
    output_image = cv2.imread(OUTPUT_PATH+output_file_name.split('.')[0]+"_output_"+".png", cv2.IMREAD_GRAYSCALE)

    ######Convert the values 0 and 1#######
    ground_truth[ground_truth==255]=1
    output_image[output_image == 215] = 1
    output_image[output_image == 30] = 0
    ######Confusion_Matrix for all images#######
    confusion_matrix += metrics.confusion_matrix(ground_truth_crop.reshape(-1), output_image.reshape(-1),
                                                 range(NO_OF_CLASSES))
    print(confusion_matrix)
######Calculate performance metrics using confusion matrix#######
total_predictions = np.sum(confusion_matrix)
mean_accuracy = mean_iou = mean_dice = 0
for class_id in range(0, NO_OF_CLASSES):
    # tn, fp, fn, tp = confusion_matrix.ravel()
    tp = confusion_matrix[class_id, class_id]  # 0,0
    fp = np.sum(confusion_matrix[: class_id, class_id]) + np.sum(confusion_matrix[class_id + 1:, class_id])  # 1,0
    fn = np.sum(confusion_matrix[class_id, : class_id]) + np.sum(confusion_matrix[class_id, class_id + 1:])  # 0,1
    tn = total_predictions - tp - fp - fn

    accuracy = (tp + tn) / (tn + fn + tp + fp)
    mean_accuracy += accuracy

    if ((tp + fp + fn) != 0):
        iou = (tp) / (tp + fp + fn)
        dice = (2 * tp) / (2 * tp + fp + fn)
    else:
        # When there are no positive samples and model is not having any false positive, we can not judge IOU or Dice score
        # In this senario we assume worst case IOU or Dice score. This also avoids 0/0 condition
        iou = 0.0
        dice = 0.0

    mean_iou += iou
    mean_dice += dice

    print("CLASS: {}: Accuracy: {}, IOU: {}, Dice: {}".format(class_id, accuracy, iou, dice))

mean_accuracy = mean_accuracy / (NO_OF_CLASSES)
mean_iou = mean_iou / (NO_OF_CLASSES)
mean_dice = mean_dice / (NO_OF_CLASSES)
print("Mean Accuracy: {}, Mean IOU: {}, Mean Dice: {}".format(mean_accuracy, mean_iou, mean_dice))
