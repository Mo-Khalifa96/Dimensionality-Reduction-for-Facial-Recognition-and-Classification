<div style="text-align: center;"> 
  <h2>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Dimensionality Reduction for Facial Recognition <br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; & <br>
    &emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp; Facilitated Model Implementation 
  </h2>
</div>
<br>

## About The Project 
**The aim of this project is to implement, evaluate and identify the best dimesnionality reduction technique for a large face images dataset and utilize the derived, reduced data to train a classification algorithm to perform classification on these images, identifying the face images belonding to each target individual in the dataset. Thus, the goal here is twofold: first, to represent the higher dimensional data in a lower dimensional space whilst retaining a reasonable capacity for facial recognition post-transformation, and, second, to facilitate the development and implementation of classification algorithms. As such, three data reduction techniques or models are considered: Principal Component Analysis (PCA), Multi-Dimensional Scaling (MDS), and Non-Negative Matrix Factorization (NMF). With each of these techniques, the necessary measures were taken to identify the optimal number of components or features post-reduction. The models were then evaluated further across each of the following dimensions: i) assessing the model's image reconstruction quality, comparing the face images before and after dimensionality reduction; ii) assessing model generalizability, i.e. the capacity of the obtained model to represent and deal with novel face images, previously unseen during model fitting; and iii) assessing the model's capacity for facial recognition, whether it can match the same faces to their corresponding target individual. As for the classification task, first a baseline classification model was trained and evaluated with the entire data intact, prior to any reduction, and on the basis of this baseline model the effectiveness of these different dimensionality reduction models for representing the original data were evaluated. Further, finally, having identified the dimensionality reduction model that best represented the data, different classification algorithms were developed, optimized, and evaluated to find the best performing one.** <br>
<br>

**Overall, this project is broken down into four parts: <br>
&emsp; 1) Reading and Inspecting the Data <br>
&emsp; 2) Data Preprocessing <br>
&emsp; 3) Dimensionality Reduction <br>
&emsp; 4) Classification** <br>

<br>
<br>


## About The Data  
**The dataset being considered here was taken from Kaggle.com, a popular website for finding and publishing datasets. You can quickly access it on Kaggle by clicking [here](https://www.kaggle.com/datasets/atulanandjha/lfwpeople/data). It is a large dataset comprised of more than 13,200 JPEG images of faces of mostly famous personas and popular figures gathered from the internet. Most individuals featured have at least two distinct photos of them. Each picture is centered on a single face with each pixel of each channel encoded by a float in range 0.0-1.0 which represent RBG color. Further, each face image is labeled with the person's name which enables the classification of faces and facial recognition or identification.** 
<br>

**Here is a sample of the faces featured in the data:**
<br>

<img src="faces sample.png" alt="ongoing projects/faces sample.png" width="650" height="450"/>

<br>
<br>

## Quick Access 
**To quickly access the project, I provided two links, both of which will direct you to a Jupyter Notebook with all the code and corresponding output, rendered, organized into sections and sub-sections, and supplied with thorough explanations and insights that gradually guide the unfolding of the project from one step to the next. The first link allows you to view the project, with its code and corresponding output rendered and organized into different sections and cells. The second link allows you to view the project as well as interact with it directly and reproduce the results if you prefer so. To execute the code, please make sure to run the first two cells first in order to install and import the Python packages necessary for the task. To run any given block of code, simply select the cell and click on the 'Run' icon on the notebook toolbar.**
<br>
<br>
<br>
***To view the project only, click on the following link:*** <br>
https://nbviewer.org/github/Mo-Khalifa96/Dimensionality-Reduction-for-Facial-Recognition-and-Classification/blob/main/Dimensionality%20Reduction%20for%20Facial%20Recognition%20and%20Facilitated%20Model%20Implementation.ipynb
<br>
<br>
***Alternatively, to view the project and interact with its code, click on the following link:*** <br>
https://mybinder.org/v2/gh/Mo-Khalifa96/Dimensionality-Reduction-for-Facial-Recognition-and-Classification/main?labpath=Dimensionality+Reduction+for+Facial+Recognition+and+Facilitated+Model+Implementation.ipynb
<br>
<br>


