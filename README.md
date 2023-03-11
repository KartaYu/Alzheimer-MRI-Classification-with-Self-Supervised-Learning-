# Alzheimer MRI Classification with SimCLR(Self Supervised Learning)
### Brief Description
- A course project for Deep learning and biomedical applications
- Programming Language : python
- Deep Learning Framework : pytorch
- Dev tool : Jupyter Notebook
### Background Motivation
- With the rapid increase in the elderly population, dementia has become an increasingly serious global public health problem. In Taiwan, 1 in every 13 people over the age of 65 suffers from dementia, with the most common type being Alzheimer's disease. Early diagnosis and treatment are the main measures for delaying the onset of dementia. Therefore, the application of AI in medical image recognition has become the main research target

### Purpose
- Although supervised learning methods can achieve high accuracy rates, they require a large amount of data sets and high-quality annotated images. Obtaining medical images can be difficult. Self-supervised learning methods can achieve high accuracy rates with fewer data. Therefore, **this project chooses the SimCLR training ResNet50 model in self-supervised learning to compare the accuracy of supervised learning and unsupervised learning in MRI image classification.**

### Data info
- Modality : MRI
- Detail :  6,400 images in 128*128 JPG format
- Source : [Alzheimer MRI Preprocessed Dataset](https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset)
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Data%20info.png)


### Method & Workflow
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Method%20%26%20Workflow.png)

#### Pretraining protocol
- In this section, we train the network based on the SimCLR learning framework. During the training process, the all input data (6,400) consists only of images without any label information.
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Pretraining%20protocol.png)

#### Fine-Tuning protocol 
- In this section, we fine-tune the network trained using the SimCLR method. That means using the pre-trained weights as initial weights for training, where the input data includes label information.
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Fine-Tuning%20protocol.png)

### Parameter setting
#### Pretraining protocol
- SGD with SAM (Sharpness-Aware Minimization)
- loss function : the normalized temperature-scaled cross entropy loss
- learning rate : 1e-2
- momentum : 0.9
- ρ : 2

#### Pretraining protocol
- epoch : 100
- loss function : cross entropy
- optimizer : AdamW
- learning rate : 1e-4
- weight decay : 1e-8

### Experimental Results
#### Loss & Accuracy
- This experiment trains the network using 1%, 10%, 20%, 50%, and 100% of the labeled data separately.
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Loss%20and%20Accuracy.png)
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Loss%20and%20Accuracy%20line%20plot.png)
#### Confuse Matrix
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/confuse%20matrix%201%25.png)
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/confuse%20matrix%2010%25.png)
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/confuse%20matrix%2020%25.png)
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/confuse%20matrix%2050%25.png)
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/confuse%20matrix%20100%25.png)
#### Visualization of heatmaps
- To determine whether the ResNet50 with SimCLR model accurately identifies important areas, we used the GradCAM method to create heatmaps and visualize the model's detection results. When diagnosing the severity of Alzheimer's disease from MRI images, the degree of atrophy in the cortex and hippocampus, as well as the extent of ventricular growth, are the main factors considered. We can see that the heatmaps focus on areas such as the cortex, ventricles, and hippocampus of the brain. Therefore, ResNet50 with SimCLR not only has high accuracy but also identifies the appropriate regions for diagnosis.
![image](https://github.com/KartaYu/Alzheimer-MRI-Classification-with-Self-Supervised-Learning-/blob/main/pic/Visualization%20of%20heatmaps%20of%20ResNet50%20with%20SimCLR.png)

### Conclusions
- This project uses two learning methods, supervised learning, and self-supervised learning, to train a ResNet50 model for Alzheimer's disease MRI recognition. During model training, 1%, 10%, 20%, 50%, and 100% of the training data were used. In the testing phase, it was found that the accuracy of SimCLR was higher than that of ResNet50, indicating that the SimCLR method (self-supervised learning) performs well even with fewer data. Furthermore, **self-supervised pretraining uses unlabeled data, which can improve the utilization efficiency of medical images labeling, a task that is typically time-consuming and laborious.**
### Model weight 
- [ResNet50(SL)](https://tinyurl.com/ymjx39n7)
- [ResNet50(SSL)](https://tinyurl.com/mtvkppwr)
### Source code repo
- [SAM](https://github.com/davda54/sam)
