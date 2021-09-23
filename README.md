# tensorflow-colour-blindness-charts
To read Color Blindness Charts using the MNIST dataset




Color Blindness Charts using Deep Learning and Tensorflow
M SHARATH SRIVATSAN
19BCE1688

There are several online tutorials available to teach you how to use the MNIST dataset to train a 
neural network to categorise handwritten digits or how to identify the difference between cats and 
dogs. Humans, on the other hand, are always quite competent at these activities and can easily 
match or better a computer's performance.
However, there are rare instances where computers can assist humans in doing tasks that we find 
challenging. For example, If I am colourblind in the red-green spectrum. As a result, charts like 
these have traditionally been difficult, if not impossible, to read:










Since there is no Dataset for Colour Blindness Charts, we have to deal with images of charts.
Now since the MNIST dataset is still available. It may be used to train a neural network that can 
classify individual digits extremely well. We can make our charts seem like MNIST's by using simple 
OpenCV transformations:

Module 1:Training Convolutional Neural Network on MNIST Dataset

First, we'll need to install Tensorflow, which can be done with pip.

QC:\Users\Sharath>pip install tensorflow
We'll now make a mnist.py file using our data:

tensorflo.›    tf
cv2
imutils numpy    np
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_t ra i n  =  x_t ra in . res h ape(x_t ra in . sh ape [ ?] ,   26,   26,     ) x_t est   =  x_t  
est . resh ape(x_t est . s h ape [ ?] ,   26,   26,     )

Then, utilising one Conv2D layer, MaxPooling, and Dropout, we'll put up our Convolutional Neural 
Network. Our 2D output is then flattened and sent through a Dense layer with 128 units, followed by 
a classification layer with ten classes (number 1—10). To represent the forecast, the output will 
be a vector with a length of ten.
mod eI  =  IN . kera s . model s . Sequent  a I ( [

IN , kera s . I ayers . Conv2D( 26,   ( 3 , 3 ) ,  a ct iv at ion= IN , kera s . I ayers . f•1axPoo 
I i ng2D( 2 , 2 ) ,
IN , kera s . I a\ ers . Dropout ( ? , 2 ) , IN . kera s . I a\ ers . FIatt en ( ) ,
,  i n put_s h ape=( 26,  26,    ) ) ,
tf.keraslayers.Dense(.2   activation=      ), tf. keras.la ers.Dense( 8, activation=


Now, we compile the model, run it on our training data as we evaluate on our test data, and save as 
an .h5 file in our directory:
model.compile(optimizer=      ,
loss=                                 ,
metrics=[          ]) model.fit(x=x_train,y=y_train, epochs= 8)
30
model. eva 1u at  e(x_t est ,   y_t est ,   venbose=   )

model.save(           

When we run this code, we get a training accuracy of about 99% and test set accuracy of:
l5ms/step -  loss: 0.0303 -  accuracy: 0.9912  

Module 2: OpenCV Chart Processing
A few additional packages must be installed.
pip install opencv-python pip install imutils
pip install numpy pip install sklearn
pip install scikit-image
We must perform different processing on the image in order to make it clear enough for Colour Blind 
People to read it.
1.  Increase the contrast
2.  Apply median and Gaussian blurring
3.  Apply K-means color clustering
4.  Convert to grayscale
5.  Apply thresholding (this one will be tricky)
6.  More blurring and thresholding
7.  Morphology open, close, erosion
8.  Skeletonizing
9.  Dilation
I found a method that takes a picture and does custom brightness and contrast adjustments online. I 
saved it as ContrastBrightness.py and turned it into a class:

cv2
,  i n pu t_ing ,  Dright nes s  =  ?,  cont ra sz  =  ?): bright nes s    =  ? :
b  ight nes s  ›  ? :
s h ado'..  =  b<zght nes s
highlight = 2ss

s h ado'..  =  ?
h ighI ight  =  2 33     brighz n es s
a I ph a_b  =  ( h ign I ight  -  s n ado'..') /’233 gamma_b  =  s h ado'..'
buf = cv2.addMeighted(input_img, alpha_b, input_img, 8, gamma_b) buf = input_img.copy()
cont ra st   ! =  a :
+  =    3    ( cont ra st  +    2 7) ‘ (  2 7  (  3  - cont ra sz ) ) a I ph a_c  =  I
gamma_c  =    2 7'      - T)
but  =  cv2. addlJe z ghI ed ( but,  a I ph a_c ,  but,  c',  gamma_c ) bu+

Brightening an image adds values to the RGB channels, whereas boosting contrast multiplies the 
values by a constant. Only the contrast feature will be used.
Clustering is another difficult aspect of our method. I created a new file, Clusterer.py, and the 
code is as follows:
cv2
n unpy        n p

(    , image, K):
image=cv2.cvtColo (image,cv2.C0L0R_BGR2HSV) vectorized = image.reshape((-.,3)) vectorized = 
np.*loat32(vectorized)
criteria = (cv2.TERM_CRI*ERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,  8,  .?)
aIt   empt s=  ?
ret,label,cente =cv2.kmeans(vectorized,K,’.ene,crite ia,attempts,cv2.KMEANS_PP_CEM*ERS) center = 
np.uint8(center)
res = cente [label.flatten()) result_image = Yes.reshape((image.shape))
result_image

As input, this code will accept an image and a number. We'll use that number K to calculate how 
many colour clusters to employ. Let's create our last file, main.py. Let's begin with imports:

tensorflo>:    tf
CU2
imutils

imut i 1s
nump     np random
cont na st
Clusterer
OS
paths


Cont ra sz
C 1 ust   erer

skimage.morphology sklearn.metrics
skelezonize
classification_report



C OfJTRAST ER   =  Cont ra st ( ) C LUST ERE R   =  C 1 ust  erer ( )

Now we'll iterate through all of the photos on our path, applying changes as we go.






















Here are some examples of those images:








That's a significant improvement. The numbers are all legible. There are still two issues:
They don't appear to be handwritten and are far too thick. On a dark backdrop, they aren't 
completely white.

(lT   ) •

, h = img shape total_pixels =.  ›h
...›hite_pixels = 8


'...›lite_pixels += . '.hite_pixels/total_pixels


(lT  ) .

img = cv2.bitwise_not(img)
image_b,..' = cv2.thresnold(img, 5@, 255, cv2.THRESH_BINARY)[l] black = np.zeros(25@,25@))
ci cle_mask = cv2.ci•cle(black, (125, l2E), .lO, (2E5, 255, 25E), -l) / 25. O


edited_image = image_bu   (ci cle_mask.astype(image_b.   ›.dtype))
edited_image

After doing various image processing and applying thresholding to the images in order to convert 
the images to Gray scale and eventually a white number on a black background

image_bw = cv2.medianBlu (image_bw,7)
image_bw = cv2.GaussianBlur(image_b.,(1.,i.),O)
image_bw = cv2.threshold(image_b.,  S8, 255, cv2.TMRESM_BINARY)[.]



kernel = np.ones((†,†), np.uint8)
image_bw = cv2.morphologyEx(image_b., cv2.MORPH_CL0SE, kernel)


kernel = np.ones((†,-), np.uint8)
image_bw = cv2.morphologyEx(image_b., cv2.MORPH_CLOSE, kernel)


kernel = np.ones((7,7), np.uint8)
image_bw = cv2.erode(image_bw, kernel, iterations=.)


image_bw = cv2.tnreshold(image_b.,8,.,cv2.THRESH_BIMARY)[ ] image_bw = 
(ñ55’skeletonize(image_b.)).astype(np.uintg)


kernel = np.ones((2.,2.), np.uintB)
image_bw = cv2.morphologyEx(image_b., cv2.MORPH_DI1A*E, kernel)


processed_images.append(imutils.resize(image_b., heighz=26)) 
image_labels.append(zn:(os.path.split(imagePath) [8] [-.]))

In the end we skeletonize and dilate the images to make it have a constant width and make it more 
uniform on an overview.















After undergoing and performing many dilations and blurring processes on the images, the result 
output looks like this. Now, the handwriting maybe uneven but it can surely help colour blind 
people to identify.

Now all we have to do is restructure our list, load the model, and assess:
model   =  IN . k era s . model s . I oad_mode I (

p roc e s s ed_im ages  =  n p . a rray ( proc es s ed_ima ges )
processed_images = processed_images.reshape(processed_images.shape[r], 26, 26, .)
p Doc e s s ed_im ages =t£ . c a s I ( proc es s ed_images ,  IN . Rio at3 2)
i mage_I a be I s  =  n p . a era} ( i mage_I a be I s )


preds = np.argmax(model.predict(processed_images), p’in-(classification_report(image_labels, 
preds)) 
axis=.)

To summarise, we load our model, modify the data, then evaluate the correctness before printing it. 
Because all 54 pictures must be converted, the code may take some time to execute.





Module 3: Results
Here's what the code prints for me when I run it:

precision     recall  fl-score   support











accuracy
ma c ro  avg
.. eight ed  avg
We achieved an overall accuracy of 78, which is 7—8 times better than chance and presumably much 
better than a person with moderate to severe colour blindness. This is good.
When we check at our recall (the ratio of properly predicted positive observations to all 
observations in the actual class) for our digits, we can see that 1—5 and 9 performed exceptionally 
well. We did alright with the number 8, but our neural network struggled with the numbers 6 and 7.

Conclusion:
We can say without a doubt that Tensorflow and OpenCV are really helpful to deploy neural networks 
to make predictions on a dataset and images. This is just a mini project I did post my Deep 
Learning using TensorFlow course offered by Coursera.
