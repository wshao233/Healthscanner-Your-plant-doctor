# Healthscanner- Your plant doctor

Healthscanner is a web application that can make image-based diagnosis of plant health and give personalized treatment recommendations to home gardeners by using computer vision and Convolutional Neural Networks (CNN) models. By simply uploading an image of leaves of sick plants, the webapp will predict the disease with confidence % shown together, as well as a description of the disease and also recommendated treatments including detailed treatment plans, like how to apply the treatment at what time, and what dosages.

Healthscan is based on transfer learning method with Inception architecutre and deep trained on a combined dataset of ~58K labelled RGB images of both healthy and diseased leaves from PlantVillage website and Plant Pathology 2020 challenge dataset (https://arxiv.org/abs/2004.11958). Since the data is highly imbalanced for the 38 different classes of diseases, data augmentation was performed which include flipping images vertically and horizontally, adding convolution which create a sunshine effect, as well as blurring image by adding more noise. This resulted in ~89K RGB images that are more balanced distributed compared to original data. After data augmentation, the data was split into training and validation datasets in a 80-20 ratio. The Inception architecutre was customized by adding a global average pooling layer and a dense layer (with softmax) to make classification for 38 diseases classes. All network layers were fine-tuned with backpropagation optimisation from the pre-trained network.  
 
## Webapp screenshot

![alt text](https://user-images.githubusercontent.com/34289565/95389316-7ecfaf00-08a8-11eb-883b-0ff95125e42a.png)

![alt text](https://user-images.githubusercontent.com/34289565/95389430-aa529980-08a8-11eb-9379-bf3f83bf769e.png)

## Demo
![alt text](https://user-images.githubusercontent.com/34289565/95389747-22b95a80-08a9-11eb-9195-e9e0d45e9140.gif)
