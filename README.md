# Determining the  Height of Cloud Base with Machine Learning  
#### Work in progress, the repository still needs to be improved/reworked

### Problem description and proposed solution
The problem is to evaluate the Height of Cloud Base, using methods of Machine Learning.  
Data: pictures taken by two cameras to the zenith at the same time.  
Images:  
![Image1](https://github.com/Ars235/Determining_HOCB/blob/master/assets/data_examples/img-2016-03-21T08-52-45devID1.jpg)|![Image2](https://github.com/Ars235/Determining_HOCB/blob/master/assets/data_examples/img-2016-03-21T08-52-45devID2.jpg)  
If you look closely, you can see clouds from different angles.  
There is some systematic component in this displacement in angle: (1) the cameras are not absolutely parallel looking at the zenith; (2) the cameras are not absolutely precisely oriented in the same direction. But this systematic component is constant throughout the experiment.  
In parallel with these images (for which UTC time is known exactly), there are model data for the height of the cloud base (HOCB). This is global data (a grid of data around the globe, with a period of 1 hour), from which it is possible to take HOCB for the final target by coordinates and time.  

Masks:  
![Mask1](https://github.com/Ars235/Determining_HOCB/blob/master/data/masks/mask-id1.jpg) ![Mask2](https://github.com/Ars235/Determining_HOCB/blob/master/data/masks/mask-id2.jpg)  

There are several types of pictures we discovered:  
![DataTypes](https://github.com/Ars235/Determining_HOCB/blob/master/assets/data_types/data_types.png)  
As we see, some pictures don't have any target value. There are some pictures, when cameras are covered with raindrops.  
Also there are some types of clouds (top right corner), that are hard to extract some information from them.  

#### Proposed solution
We used pretrained [SuperGlue model](https://github.com/magicleap/SuperGluePretrainedNetwork) to extract dx, dy and *confidence* *p* between key points of two images (each training sample is a pair of images). The number of detected key points may vary from sample to sample, so we chose 15 key points with the highest *p* value, samples that had less than 15, we discarded. Having confidences between key points allowed us to plot the distrtibution of confidence, extract statistics from it (median value, percentiles, variance), and hence, allowed SuperGlue to compute some features, such as:  
1) cosines and sines of spherical angles (8 features)  
2) pixel coordinates  (4 features)
3) modules of pixel difference along the x, y axes (2 features) 
4) pixels intensity (2 features)
In summary: 16 features  

Extracted features:
![Extracted features](https://github.com/Ars235/Determining_HOCB/blob/master/assets/data_examples/angles.png)  


These 16 features are the input features for the Regression model, that will learn to predict the height.  
The came up with the next architecture:  
Layers dimensions: 16 -> 13 -> 10 -> 8 -> 5 -> 1, activation is *ReLU*, the Loss function and metric are both _MAPE_. As an otimizing algorithm we used *Adam*.  

The best result we achieved was MAPE ~ 44%.  

Some of the model's outputs:  

![ex1](https://github.com/Ars235/Determining_HOCB/blob/master/assets/outputs/good1.jpg)  
![ex2](https://github.com/Ars235/Determining_HOCB/blob/master/assets/outputs/good2.jpg)  
![ex3](https://github.com/Ars235/Determining_HOCB/blob/master/assets/outputs/good3.jpg)  
![ex4](https://github.com/Ars235/Determining_HOCB/blob/master/assets/outputs/good4.jpg)  
![bad](https://github.com/Ars235/Determining_HOCB/blob/master/assets/outputs/bad.png)
