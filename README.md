# COVID-19 Pneumonias Automated Detection Based on CT Imaging using Deep Transfer Learning
[Google Colab](https://colab.research.google.com/drive/1LQZWt-OakH7JoK3bdEIgnqhUdyECMons?authuser=1)

## About the projects
The world is being hit by the COVID-19 pandemic which can cause pneumonia. Due to the high level of spread of the SARS-CoV-2 virus, rapid and massive tests must be done so that the infected can be isolated as soon as possible. The gold standard used to detect COVID-19 is RT-PCR. However, there are some drawbacks such as it only has 60-70% sensitivity and has relatively high cost. This study aims to develop an alternative detection method for COVID-19 based on Computed Tomography images using deep transfer learning techniques. Transfer learning is carried out using the pre-trained DenseNet-201 model. The process consists of several stages:
* Data preprocessing
* Data splitting
* Data augmentation, 
* Hyperparameter tuning
* Developing deep transfer learning classification model 
* Model evaluation

The model that has been built has a fairly good performance in identifying patients with pneumonia COVID-19 and the healthy ones with **accuracy 93,41%**, **precision 94,19%**, **sensitivity (recall) 93,10%**, and **specificity 93,75%**.
