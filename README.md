# Real-time road detection implementation of UNet architecture for autonomous driving
UHA Dataset, Road Segmentation, artificial intelligence

# Abstract 
This paper presents a real-time implementation workflow of neural networks for autonomous driving tasks. The UNet structure is chosen for a road segmentation task, providing good performance for low complexity. The model is trained and validated using two datasets, KITTI (validation of the model with respect to state of art) and a local highway dataset (UHA dataset), collected by the laboratory research team. The performance of the model for road detection is evaluated using the F1 score metric. After a simulation validation on both sets, the model is integrated into a real vehicle through the RTMaps platform. The application is tested in real-time conditions, around the city, under various weather and light. Finally, the proposed model proves low complexity and good performance for real-time road detection tasks.

# Architecture 
![alt text](https://github.com/vasigiurgi/Real-time-road-detection-with-Unet-for-autonomous-driving/blob/master/images/unet.png)

# Real-time implementation 
In the emebbeded system, RTMaps has been used. Within then platform, the user can create a Python Bridge where the model is used to make the predictions. Due to GPU limitations, the prediction time was quite slow, however the software works with GPU, which eventually could improve the prediction of the frames per second. 

The software is provided by INTEMPORA (dSpace) and it can be downloaded from the following link: 
!https://intempora.com/products/rtmaps/ 
In Python Bridge document file illustrates how to setup a working environment with Python Bridge connected to your Python. 
