import tensorflow as tf
import cv2
import imutils
from imutils import paths
import numpy as np
import random
from contrast import Contrast
from Clusterer import Clusterer
import os
from skimage.morphology import skeletonize
from sklearn.metrics import classification_report



CONTRASTER = Contrast()
CLUSTERER = Clusterer()

def white_percent(img):
    # calculated percent of white pixels in the grayscale image
    w, h = img.shape
    total_pixels = w*h
    white_pixels = 0
    for r in img:
        for c in r:
            if c == 255:
                white_pixels += 1
    return white_pixels/total_pixels

# fixes image where number is darker than background in grayscale
def fix_image(img):
    # inversion
    img = cv2.bitwise_not(img)

    # thresholding
    image_bw = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)[1]

    # making mask of a circle
    black = np.zeros((250,250))
    circle_mask = cv2.circle(black, (125, 125), 110, (255, 255, 255), -1) / 255.0

    # applying mask to make everything outside the circle black
    edited_image = image_bw * (circle_mask.astype(image_bw.dtype))
    return edited_image


num_images = 54
i = 0
processed_images = []
image_labels = []

# prepares image paths and randomizes them
image_paths = list(paths.list_images("charts/ordered"))
random.shuffle(image_paths)


for imagePath in image_paths:
    image = cv2.imread(imagePath)

    # resize
    image = imutils.resize(image, height=250)

    # contrast
    image = CONTRASTER.apply(image, 60)
    
    # blurring
    image = cv2.medianBlur(image,15)
    image = cv2.GaussianBlur(image,(3,3),cv2.BORDER_DEFAULT)

    # color clustering
    image = CLUSTERER.apply(image, 5)

    # grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    threshold = 0
    percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])
    while (not (percent_white > 0.10 and percent_white < 0.28)) and threshold <= 255:
        threshold += 10
        percent_white = white_percent(cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1])


    # means that image was not correctly filtered
    if threshold > 255:
        image_bw = fix_image(gray)
    else:
        image_bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)[1]

    
     # blurring
    image_bw = cv2.medianBlur(image_bw,7)
    image_bw = cv2.GaussianBlur(image_bw,(31,31),0)
    image_bw = cv2.threshold(image_bw, 150, 255, cv2.THRESH_BINARY)[1]


    # apply morphology close
    kernel = np.ones((9,9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # apply morphology open
    kernel = np.ones((9,9), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_CLOSE, kernel)

    # erosion
    kernel = np.ones((7,7), np.uint8)
    image_bw = cv2.erode(image_bw, kernel, iterations=1)

    # skeletonizing
    image_bw = cv2.threshold(image_bw,0,1,cv2.THRESH_BINARY)[1]
    image_bw = (255*skeletonize(image_bw)).astype(np.uint8)

    # dilating
    kernel = np.ones((21,21), np.uint8)
    image_bw = cv2.morphologyEx(image_bw, cv2.MORPH_DILATE, kernel)


    processed_images.append(imutils.resize(image_bw, height=28))
    image_labels.append(int(os.path.split(imagePath)[0][-1]))
 
    #cv2.imshow(("Gray", gray)
    #cv2.waitKey()
    #cv2.destroyAllWindows()
    
    i += 1
    print(i)
    if i >= num_images:
        break


# prediction
model = tf.keras.models.load_model("mnist.h5")

processed_images = np.array(processed_images)
processed_images = processed_images.reshape(processed_images.shape[0], 28, 28, 1)
processed_images=tf.cast(processed_images, tf.float32)

image_labels = np.array(image_labels)

preds = np.argmax(model.predict(processed_images), axis=1)
print(classification_report(image_labels, preds))








              precision    recall  f1-score   support

           1       0.71      1.00      0.83         5
           2       0.80      1.00      0.89         8
           3       0.42      1.00      0.59         5
           4       1.00      0.29      0.44         7
           5       0.75      0.83      0.89         6
           6       0.91      0.35      0.89         5
           7       1.00      0.20      0.33         5
           8       0.40      0.33      0.36         6
           9       1.00      0.14      0.25         7

    accuracy                           0.78        54
   macro avg       0.79      0.68      0.70        54
weighted avg       0.67      0.68      0.79        54

    