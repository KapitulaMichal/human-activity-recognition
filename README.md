# Human activity recognition

Program developed for the master thesis **Human Activity Recognition On Sequences Of Digital Images Using Convolutional Neural Networks**.

## TOOLS & TECHNOLOGIES
	1. Python
	2. Tensorflow
<br>


It proposes the method of human activity recognition using both spacial and temporal features. As an experimental basis used was the KTH human activity recognition dataset, available at http://www.nada.kth.se/cvap/actions/. This database contains 598 clips of six different human actions: walking, jogging, running, boxing, hand waving and hand clapping. They are performed several times by 25 subjects in different scenarios, recorded by a static camera on a
homogeneous background. Sequences have in average 4 seconds in length.

In order to increase the training efficiency of the network two additional features are extracted from the videos. Optical flow is the method to acquire motion information of objects, contours and areas
in the static image. Difference of Gaussian (DoG) allows for amplifying the edges visible in the image. Normal black and white frames of the video are combined with additional features and used as an input for the Neural Network.

In order to deal with video action recognition, with use of Tensorflow created is the 3D Convolutional Neural Network, which as an input takes 4 dimensional matrix consisting of spatial dimension, temporal dimension and different channels. The focus is put on creating less complex system for easier preparation and implementation in real life applications.
