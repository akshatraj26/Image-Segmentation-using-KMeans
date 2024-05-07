# -*- coding: utf-8 -*-
"""
Created on Tue May  7 10:56:40 2024

@author: aksha
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


image = cv2.imread("Image Segmentation using KMeans/Friend image segmentation/IMG_20220827_155310.jpg")


image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.imshow(image)

# convert thevalue to float
shap = image.shape
pixels_val = image.reshape((-1, 3))

pixels_val = np.float32(pixels_val)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)

k = 3
retval, labels, centers = cv2.kmeans(pixels_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)


# convert the data into 8 bits values
centers = np.uint8(centers)
segmented_data = centers[labels.flatten()]

seg_data = segmented_data.shape

# reshape the data into original dimensions
segmented_image = segmented_data.reshape((image.shape))


plt.imshow(segmented_image)


def image_segmentation(image_path, criteria, k: int):
    # read the file
    image = cv2.imread(image_path)
    # conver bgr to rgb
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # converting 3 dims image to dimensional
    pixels_val = image.reshape((-1, 3))
    # convert the value to float
    pixels_val = np.float32(pixels_val)
    retval, labels, centers = cv2.kmeans(pixels_val, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert data into 8bit values
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    # reshape the data into original dimensions
    segmented_image = segmented_data.reshape((image.shape))
    
    return segmented_image


path = "Image Segmentation using KMeans/Friend image segmentation/IMG_20220827_153601.jpg"
seg = image_segmentation(path, criteria, k=3)
plt.imshow(seg)
    