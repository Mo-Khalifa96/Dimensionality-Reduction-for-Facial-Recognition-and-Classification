#DIMENSIONALITY REDUCTION FOR FACIAL RECOGNITION AND FACILITATED MODEL IMPLEMENTATION
#import necessary modules
import os
import shutil
import errno
import tarfile 
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.manifold import MDS
from sklearn.decomposition import PCA, NMF
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics.pairwise import cosine_distances
from sklearn.preprocessing import StandardScaler
from imblearn.pipeline import Pipeline
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from kaggle.api.kaggle_api_extended import KaggleApi

import warnings
warnings.simplefilter("ignore")


#Defining Custom Functions 
#Define function to copy downloaded files from source to correct directory
def copy_files(src, dest):
    try:
        shutil.copytree(src, dest)
    except OSError as e:
        # If the error was caused because the source wasn't a directory
        if e.errno == errno.ENOTDIR:
            shutil.copy(src, dest)
        else:
            print('Directory not copied. Error: %s' % e)

#Define custom function to plot the amount of explained variance by PCA model
def plot_PCA_explained_variance(pca):
    #This function graphs the accumulated explained variance ratio for a fitted PCA object.
    acc_variance = [*np.cumsum(pca.explained_variance_ratio_)]
    fig, ax = plt.subplots(1, figsize=(15,4))
    ax.stackplot(range(pca.n_components_), acc_variance)
    ax.scatter(range(pca.n_components_), acc_variance, color='black', s=10)
    ax.set_ylim(0, 1)
    ax.set_xlim(0, pca.n_components_+1)
    ax.tick_params(axis='both')
    ax.set_xlabel('N Components', fontsize=11)
    ax.set_ylabel('Accumulated explained variance', fontsize=11)
    plt.tight_layout()

#Define custom function to plot the confusion matrix using a heatmap
def plot_cm(cm, names):
    plt.figure(figsize=(10,7))
    hmap = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues',
            xticklabels=names, yticklabels=names)
    hmap.set_xlabel('Predicted Value', fontsize=13)
    hmap.set_ylabel('Truth Value', fontsize=13)
    plt.tight_layout()

#Define custom functions to compute and report classification error scores
def error_scores(ytest, ypred):
    error_metrics = {
        'Accuracy': accuracy_score(ytest, ypred),
        'Recall': recall_score(ytest, ypred, average='weighted'),
        'Precision': precision_score(ytest, ypred, average='weighted'),
        'F1 score': f1_score(ytest, ypred, average='weighted')
        }

    return pd.DataFrame(error_metrics, index=['Error score']).apply(lambda x:round(x,2)).T

#Define custom function to compute error and return results in dictionary form
def error_scores_dict(ytest, ypred, model):
    #create empty dict for storing results 
    error_dict = {
        'Model': model, 
        'Accuracy': round(accuracy_score(ytest,ypred),2),
        'Precision': round(precision_score(ytest,ypred, average='weighted'),2),
        'Recall': round(recall_score(ytest,ypred, average='weighted'),2),
        'F1 Score': round(f1_score(ytest,ypred, average='weighted'),2)
    }
    return error_dict

#Define function to evaluate the image reconstruction quality of a given dim reduction model 
def evaluate_reconstruction(X, estimator, face_indx):
    '''This function evaluates the reconstruction of a given image by index'''
   
    #get face image by index 
    X_face = X[face_indx]
    #get the PCA approximated image 
    X_trans = estimator.transform(X_face.reshape(1,-1))
    X_inv = estimator.inverse_transform(X_trans)

    #plot original image 
    plt.figure(figsize=(10,4))
    plt.subplots_adjust(top=.8)
    plt.subplot(1,2,1)
    plt.imshow(X_face.reshape(h, w), cmap=plt.cm.gray)  
    plt.title("Original Image")
    plt.axis('off')

    #plot image with PCA
    plt.subplot(1,2,2)
    plt.imshow(X_inv.reshape(h,w), cmap='gray')
    plt.title("Approximated Image")
    plt.axis('off')
    plt.show()
    print('\n\n_____________________________________________________________________________________________\n\n')

#Define function to compare images from the train and test sets
def plot_TrainVsTest(X_train, X_test, h, w, train_indx, test_indx, N_imgs):
    random_indx = np.random.randint(0, len(train_indx), N_imgs)
    for train_sample, test_sample in zip(train_indx[random_indx], test_indx[random_indx]):
        #plot figure
        plt.figure(figsize=(9,4.2))
        plt.subplots_adjust(top=.75)
        plt.suptitle(f'True Target: {names[y_train[train_sample]]}\nObtained Target: {names[y_test[test_sample]]}\n\n\n\n', fontsize=11)
        #plot image from training set
        plt.subplot(1,2,1)
        plt.imshow(X_train[train_sample].reshape(h,w), cmap='gray')
        plt.title(f'Train sample {train_sample}')
        plt.axis('off')

        #plot image from testing set 
        plt.subplot(1,2,2)
        plt.imshow(X_test[test_sample].reshape(h,w), cmap='gray')
        plt.title(f'Test sample {test_sample}')
        plt.axis('off')
        plt.show()
        print('\n\n___________________________________________________________________________________\n\n')

#Define custom function to specify threshold criteria for measuring image similarity
def get_threshold(dist_similarity, dist_sim_indices, max_cos=0.1, min_cos=0):
    X_train_indx = np.where(np.logical_and( (dist_similarity>min_cos), (dist_similarity<max_cos) ))[0]
    X_test_indx = dist_sim_indices[np.logical_and( (dist_similarity>min_cos), (dist_similarity<max_cos) )]
    return X_train_indx, X_test_indx

#Define function to compare results of multiple models
def visualize_models_results(results):
    print(results)
    print('\n\n')

    #identify data for plotting
    x = np.arange(4)
    LR_res = results.loc['LR']  
    SVC_res = results.loc['SVC']
    KNN_res = results.loc['KNN']

    #set figure characteristics 
    width = 0.2
    plt.figure(figsize=(14,7), dpi=80)
    
    #Plot bars
    bars1 = plt.bar(x-0.2, LR_res, width, alpha=.9)
    bars2 = plt.bar(x, SVC_res, width, alpha=.85)
    bars3 = plt.bar(x+0.2, KNN_res, width, alpha=.75)

    #set labels and ticks 
    plt.title('Evaluation Scores per Classifier', fontsize=15)
    plt.xticks(x, ['Accuracy', 'Recall', 'Precision', 'Fscore'])
    plt.yticks(np.linspace(0,1,11))
    plt.xlabel("Evaluation Metrics", fontsize=13)
    plt.ylabel("Score", fontsize=13)
    plt.legend(["Logistic Regression", "Support Vector Machine", "K-Nearest Neighbors"], loc='upper right', borderaxespad=0.)
    
    #Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2.0 + 0.02, height+0.01,
                     f'{height:.2f}', ha='center', va='bottom', rotation=45, fontsize=10)
    plt.show()



#Definign a random state for reproducible results 
rs = 222 


#DOWNLOADING AND ACCESSING THE DATASET
#Initialize the Kaggle API to access kaggle
api = KaggleApi()
api.authenticate()

#specifying the kaggle dataset identifier 
kaggle_dataset = 'atulanandjha/lfwpeople'

#specifying the path to save the data to
mypath = 'LFW people dataset/' 

#Download the dataset
api.dataset_download_files(kaggle_dataset, path=mypath, unzip=True)

#Extract the data from the tgz file
tgz_filepath = os.path.join(mypath, 'lfw-funneled.tgz')
with tarfile.open(tgz_filepath, "r:gz") as tar:
    tar.extractall(path=mypath)
os.remove(tgz_filepath)

#Copying files to lfw home for reading 
lfw_home = 'LFW people dataset/lfw_home'
copy_files(mypath, lfw_home)



#PART ONE: READING AND INSPECTING THE DATA 
#Loading the data downloaded 
lfw_dataset = fetch_lfw_people(data_home=mypath, 
                               min_faces_per_person=35,  #to fetch only target individuals with at least 35 face images featuring them
                               resize=1, 
                               download_if_missing=False)


#Inspecting the Data 
#report images shape
n_samples, h, w = lfw_dataset.images.shape
print('Number of face samples in the dataset:', n_samples)
print('Number of rows in the dataset:', h)
print('Number of coloumns in the dataset:', w)
print(f'Therefore we have {n_samples} face samples, each being {h} x {w} pixels')

#report number of unique targets
n_classes, names = len(lfw_dataset.target_names), lfw_dataset.target_names
print('Number of unique people in the data:', n_classes)


#Viewing the target individuals in the data 
n_row,n_col = 4,6    #specify number of columns and rows
plt.figure(figsize=(2.4 * n_col , 3.2 * n_row))   #create and set figure characteristics 
plt.subplots_adjust(bottom=0, top=.90, left=.01, right=.99, hspace=.35) #set subplots characteristics
for i,face in enumerate(np.unique(lfw_dataset.target)):   
    face_idx = np.argmax(lfw_dataset.target == face)   #get face unique identifier index
    plt.subplot(n_row, n_col, i + 1)   #set subplot position
    plt.imshow(lfw_dataset.data[face_idx].reshape(h, w), cmap=plt.cm.gray)   #plot face in given subplot
    plt.title(names[face], size=13)  #set image title (person's name)
    plt.axis('off')
plt.show()



#PART TWO: DATA PREPROCESSING 
#In this section, I will perform the necessary data preprocessing procedures to make the data ready for model development 
# and evaluation. First, I will select the predictors and target variable; examine the class distribution and deal with any 
# class imbalances; split the data into training and testing sets; and finally perform feature scaling to standardize the data.

#Data Selection 
#Identifying the predictors and target variable 
X = lfw_dataset.data
y = lfw_dataset.target


#Examining Data Shape 
#obtain and report shape of the raw data 
shape = X.shape 
print('Number of coloumns:', shape[1])
print('Number of rows:', shape[0])


#Examining Class Distribution 
#plot and examine the class distribution in the target variable
sns.set_context('paper')
class_freq = pd.Series(y).value_counts().sort_index()
ax = class_freq.plot(kind='bar', figsize=(8,4.5), color='#24799e', 
                width=.8, linewidth=.8, edgecolor='k', rot=90,
                xlabel='Class', ylabel='Count')
ax.set_xticklabels([name.split(' ')[-1] for name in names])
plt.show()


#Data Splitting  
#Performing stratified data splitting to obtain a training and testing set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.25, stratify=y, random_state=rs)


#Dealing with Imbalanced Classes: Resampling 
#Define the target number of samples per class
target_samples = 100

#Define sampling strategies
undersampling_strategy = {key: target_samples for key in range(n_classes) if Counter(y_train)[key] > target_samples}  #undersample if target has less than 100 images
oversampling_strategy = {key: target_samples for key in range(n_classes) if Counter(y_train)[key] < target_samples} #oversample if target has less than 100 images

#Create a pipeline combining over- and undersampling
sampling_pipeline = Pipeline([
    ('under', RandomUnderSampler(sampling_strategy=undersampling_strategy, random_state=rs)),
    ('over', RandomOverSampler(sampling_strategy=oversampling_strategy, random_state=rs))
])

#Resampling the data to get equal sized classes 
X_train, y_train = sampling_pipeline.fit_resample(X_train, y_train)

#We can check the class distribution once again
class_freq = pd.Series(y_train).value_counts().sort_index()
ax = class_freq.plot(kind='bar', figsize=(8,4.5), color='#24799e', 
                width=.7, linewidth=.8, edgecolor='k', rot=90,
                xlabel='Class', ylabel='Count', ylim=(0,np.max(class_freq)+10))
ax.set_xticklabels([name.split(' ')[-1] for name in names])
plt.text(7.5, 115, f'All classes have an equal size of {int(class_freq[0])}', ha='center', va='bottom', fontsize=14)
plt.subplots_adjust(top=.9)
plt.show()


#Feature Scaling 
#Perform feature standardization 
std_scaler = StandardScaler()
X_train_std = std_scaler.fit_transform(X_train)
X_test_std = std_scaler.transform(X_test)



#PART THREE: DIMENSIONALITY REDUCTION 
#In this section, I will test out different dimensionality reduction models and use them to transform the data and perform 
# classification with the new, reduced data. This should help significantly decrease the number of features to be used in 
# training while also preserving most of the original data's properties which also means that the classification algorithm 
# to be used will be faster and more computationally efficient. As such, I will test three different dimensionality reduction 
# techniques: Principal Component Analysis, Multi-Dimensional Scaling and Non-Negative Matrix Factorization. For each technique, 
# I will train a logistic regression classifier to identify and classify the faces in the data.

#Baseline Classification Model 
#Create logistic regression object
LR = LogisticRegression(solver='lbfgs', random_state=rs, n_jobs=-1)

#fit the LR classifier with the original data
LR_model = LR.fit(X_train_std, y_train)

#get class peridctions 
y_pred = LR_model.predict(X_test_std)

#Report overall classification error results 
print('Classification error scores (weighted average):')
print(error_scores(y_test, y_pred))

#Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_cm(cm, names=[name.split(' ')[-1] for name in names])


#Dimensionality Reduction - Model Development and Optimization 
#Model One: Principal Component Analysis 
#Create PCA object
pca = PCA(random_state=rs)
#fit the PCA model 
pca.fit(X_train_std)
#transform the train and test data
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)

#Visualize first 10 PCA components
fig, axes = plt.subplots(2, 5, figsize=(15,7))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(h, w), cmap='gray')
    ax.set_title(f'PCA Component {i+1}')
plt.show()

#Plotting the accumulation of explained variance across PCA components
plot_PCA_explained_variance(pca)


#Get number of PCA dimensions necessary for 99% explained variance
acc_variance = np.cumsum(pca.explained_variance_ratio_) <= 0.99
n_components = acc_variance.sum() + 1
print(f'n_components={n_components}')


#Final Model Selection 
#Create PCA object with obtained number of components
pca = PCA(n_components=n_components, random_state=rs)
#fit the PCA model 
pca.fit(X_train_std)
#transform the train and test data
X_train_pca = pca.transform(X_train_std)
X_test_pca = pca.transform(X_test_std)


#Model Evaluation
#Testing the Model: Assessing image reconstruction quality 
#Create lambda function to generate random indices for 5 images 
random_face_generator = lambda X: np.random.randint(0,len(X),5)

#evaluate PCA reconstruction quality of selected images 
for face_indx in random_face_generator(X):
    evaluate_reconstruction(X, pca, face_indx)


#Testing the Model: Testing Model Generalizability 
#Get pairwise cosine distances
cos_distances_mtrx = cosine_distances(X_train_pca, X_test_pca)

#Get column indices with most distance similarity and their corresponding cosine distance values (sorted in ascending order)
min_dist_indices = np.argmin(cos_distances_mtrx, axis=1)
min_cosine_dist = np.min(cos_distances_mtrx, axis=1)

#Now visualize the distribution of the distance values using a histogram
plt.hist(min_cosine_dist, bins=100)
plt.title('Cosine Distance Values')
plt.annotate(text=f'mean={np.mean(min_cosine_dist):.1f}', xy=(0.45,60), fontsize=11)
plt.show()


#Testing the Model: Facial Recognition Performance
#Get indices for images that are most similar with their pairwise distances falling between 0 and 0.1
train_indx, test_indx = get_threshold(min_cosine_dist, min_dist_indices, min_cos=0, max_cos=0.1)

#check the resulting shape
print(train_indx.shape, test_indx.shape)

#plot and compare 10 of the top most similar images
plot_TrainVsTest(X_train, X_test, h, w, train_indx, test_indx, 10)


#Classification with PCA 
#Now we can use the obtained principal components to train a logistic regression classifier to identify and classify the face images
#Create logistic regression object
LR = LogisticRegression(solver='lbfgs', random_state=rs, n_jobs=-1)

#fit the LR classifier with the PCA data
LR_model = LR.fit(X_train_pca, y_train)

#get class peridctions 
y_pred = LR_model.predict(X_test_pca)

#Report overall classification error results 
print('Classification error scores (weighted average):')
print(error_scores(y_test, y_pred))

#Plot the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_cm(cm, names=[name.split(' ')[-1] for name in names])



#Model Two: Multi-Dimensional Scaling 
#Compute the cosine distance matrices
cosine_dist_train = cosine_distances(X_train_std)
cosine_dist_test = cosine_distances(X_test_std)

#In order to identify the most optimal number of dimensions, I will test out different n_components values and measure 
# the reduction in reconstruction error using the stress metric
#Define number of components to test out 
n_components_lst = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]

#create empty list to store results
stress_scores = []

#iterate over n components and get stress score to find the better one
for i,n_comp in enumerate(n_components_lst):
        #Create mds object and set dim reduction characteristics
        mds = MDS(n_components=n_comp, metric=False, normalized_stress=False, dissimilarity='precomputed', random_state=rs, n_jobs=-1)
        
        #Fit cosine matrix data 
        mds.fit(cosine_dist_train)

        #compute and store stress score 
        stress_scores.append(round(mds.stress_,3))
        print(f'{i+1}/{len(n_components_lst)} runs completed')

#Report results
stress_scores_df = pd.DataFrame({'n_components': n_components_lst, 'stress score': stress_scores}).set_index('n_components')
print(stress_scores_df)

#Plotting out the stress scores across the different dimensions 
stress_scores_df.plot(figsize=(8,5), marker='o', markersize=4, 
                    title='Stress Scores and Number of Components',
                    xlabel='n_components', ylabel='stress score', legend=False)

#Final Model Selection 
#Create MDS model with the best number of components obtained
n_components = 200
mds = MDS(n_components=n_components, metric=False, dissimilarity='precomputed', random_state=rs, n_jobs=-1)

#Fit and transform the data
X_train_mds = mds.fit_transform(cosine_dist_train)
X_test_mds = mds.fit_transform(cosine_dist_test)


#Model Evaluation
#Testing the Model: Testing Model Generalizability 
#Get pairwise cosine distances
cos_distances_mtrx = cosine_distances(X_train_mds, X_test_mds)

#Get column indices with most distance similarity and their corresponding cosine distance values (sorted in ascending order)
min_dist_indices = np.argmin(cos_distances_mtrx, axis=1)
min_cosine_dist = np.min(cos_distances_mtrx, axis=1)

#Now visualize the distribution of the distance values using a histogram
plt.hist(min_cosine_dist, bins=100)
plt.title('Cosine Distance Values')
plt.annotate(text=f'mean={np.mean(min_cosine_dist):.1f}', xy=(0.7,75), fontsize=11)
plt.show()


#Testing the Model: Facial Recognition Performance
#Get indices for images that are most similar with their pairwise distances falling between 0.6 and 0.7
train_indx, test_indx = get_threshold(min_cosine_dist, min_dist_indices, min_cos=0.6, max_cos=0.7)
#check the resulting shape
print(train_indx.shape, test_indx.shape)

#plot and compare the 10 matches obtained
plot_TrainVsTest(X_train, X_test, h, w, train_indx, test_indx, 10)


#Classification with MDS 
#Create logistic regression object 
LR = LogisticRegression(solver='lbfgs', random_state=rs, n_jobs=-1)

#fit the LR classifier with the MDS data 
LR_model = LR.fit(X_train_mds, y_train)

#get class peridctions 
y_pred = LR_model.predict(X_test_mds)

#Report overall classification error results 
print('Classification error scores (weighted average):')
print(error_scores(y_test, y_pred))

#visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_cm(cm, names=[name.split(' ')[-1] for name in names])



#Model Three: Non-Negative Matrix Factorization
#Rhe task here once again is to identify the most appropriate number of basis components by which to represent the data with a lower 
# number of features overall
#Define the parameter values to traverse through
n_components_lst = [10, 25, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

#Create empty lists to store results 
reconstruction_errors = []

#loop over n_components_lst and return frobenius error score
for i,n_component in enumerate(n_components_lst):
    #Create NMF object and set parameters
    nmf = NMF(n_components=n_component, init='nndsvd', max_iter=500, random_state=rs)

    #Fit the data
    nmf.fit(X_train)

    #compute and store reconstruction error and corresponding parameters
    reconstruction_errors.append(round(nmf.reconstruction_err_,3))
    print(f"{i+1}/{len(n_components_lst)} runs completed")

#organize and report results
results_table = pd.DataFrame({'n_components': n_components_lst, 'error score': reconstruction_errors}).set_index('n_components')
print(results_table)

#visualize the results
results_table['error score'].plot(figsize=(7,5), marker='o', markersize=4, 
                        title='Reconstruction Error and Number of Components',
                        xlabel='Number of Components', ylabel='Reconstruction Error (Frobenius Norm)')


#Final Model Selection 
#Developing an NMF model with the most optimal n_components obtained
#assign the number of components
n_components = 400
#create NMF object and set relevant parameters
NMF_best = NMF(n_components=n_components,init='nndsvd', max_iter=500, random_state=rs)
#transform the train and test data to get the factor matrices
W_train = NMF_best.fit_transform(X_train)
W_test = NMF_best.transform(X_test)
H_mtrx = NMF_best.components_


#Model Evaluation
#Testing the Model: Assessing image reconstruction quality
#Get the data approximate using the tuned NMF model
X_test_inv = np.dot(W_test,H_mtrx)

#get cosine distances between original data and NMF-approximated data
cos_distances = cosine_distances(X_test, X_test_inv)

#get diagonal pairwise distances for direct one-to-one comparisons
diagonal_mtrx = np.diagonal(cos_distances)

#compute and report the mean cosine distance
mean_cos_distance = np.mean(diagonal_mtrx)
print('Mean cosine distance score (X vs. X_inv):', round(mean_cos_distance,3))

#We can assess reconstruction quality by looking at some of the images before and after NMF approximation
#evaluate NMF reconstruction quality of selected images 
for face_indx in random_face_generator(X):  
    evaluate_reconstruction(X, NMF_best, face_indx)


#Testing the Model: Testing Model Generalizability
#Get cosine distance between the two coefficient matrices
W_cos_distances = cosine_distances(W_train, W_test)

#get diagonal pairwise distances for one-to-one comparisons
W_diagonal_mtrx = np.diagonal(W_cos_distances)

#compute and report the mean cosine distance
W_mean_cos_distance = np.mean(W_diagonal_mtrx)
print('Mean cosine distance score (W_train vs. W_test):', round(W_mean_cos_distance,3))
print()

#We can also look directly at the distribution of cosine distances capturing the distance between the
# coefficient matrices of the training and testing data
#Get column indices with most distance similarity and their corresponding cosine distance values (sorted in ascending order)
min_dist_indices, min_cosine_dist = np.argmin(W_cos_distances, axis=1), np.min(W_cos_distances, axis=1)

#Now visualize the distribution of the distance values using a histogram
plt.hist(min_cosine_dist, bins=100)
plt.title('Cosine Distance Values')
plt.annotate(text=f'mean={np.mean(min_cosine_dist):.2f}', xy=(0.4,90), fontsize=11)
plt.show()


#Testing the Model: Facial Recognition Performance
#Get indices for images that are most identical, with their pairwise distances falling between 0 and 0.005
train_indx, test_indx = get_threshold(min_cosine_dist, min_dist_indices, min_cos=0, max_cos=0.15)
#check the resulting shape
print(train_indx.shape, test_indx.shape)

#Plot and compare 10 of the top most similar images 
plot_TrainVsTest(X_train, X_test, h, w, train_indx, test_indx, 10)


#Classification with NMF
#Create logistic regression object 
LR = LogisticRegression(solver='lbfgs', random_state=rs, n_jobs=-1)

#fit the LR classifier with the MDS data 
LR_model = LR.fit(W_train, y_train)

#get class peridctions 
y_pred = LR_model.predict(W_test)

#Report overall classification error results 
print('Classification error scores (weighted average):')
error_scores(y_test, y_pred)

#visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plot_cm(cm, names=[name.split(' ')[-1] for name in names])



#PART FOUR: CLASSIFICATION - MODEL COMPARISON AND SELECTION 
#In this section, I will consider each of three classification models: Logistic Regression, Support Vector Machine, and K-Nearest Neighbors. I will once again 
# use the data derived from the PCA model and pass it to each of the three classifiers, perform hyperparameter tuning on each, and then compare and contrast 
# their performances in order to identify the best performing model.
#Define estimators to test 
estimators = [('LR', LogisticRegression(solver='lbfgs', random_state=rs, n_jobs=-1)), 
              ('SVC', SVC(kernel='rbf', random_state=rs)), 
              ('KNN', KNeighborsClassifier(metric='cosine', n_jobs=-1)) ]


#Define parameters for grid search with each 
#parameters for the logistic regression model 
params_LR = { 'penalty': [None, 'l2'], 'C': np.geomspace(0.00001,100,8) }
#parameters for the SVC model 
params_SVC = {'C': np.geomspace(0.0001,10,6) } 
#parameters for the KNN model 
params_KNN = { 'n_neighbors': np.arange(1,11), 'weights': ['uniform', 'distance'] }

#Create a list with the parameters 
estimators_params = [params_LR,  params_SVC, params_KNN]

#Create empty lists to store result 
error_results = []
best_estimators = []
best_params = []

#Now iterating over the different models and applying different resampling techniques to find the best combinations
for estimator, params in zip(estimators, estimators_params):
    #create grid search object 
    grid = GridSearchCV(estimator=estimator[1], param_grid=params, scoring='accuracy', cv=3, n_jobs=-1, verbose=1)
    #fitting and tuning classifier 
    grid.fit(X_train_pca, y_train)
    #get best estimator 
    best_estimator = grid.best_estimator_
    #generate class predictions
    y_pred = best_estimator.predict(X_test_pca)
    #get error scores and store them
    error_dict = error_scores_dict(y_test, y_pred, model=estimator[0])
    error_results.append(error_dict)
    best_estimators.append((estimator[0], best_estimator))

#Report error scores 
error_table = pd.DataFrame(error_results).set_index('Model')
print(error_table)

#Report the best estimator 
print('Best estimator (and parameters):\n\n', best_estimators[np.argmax(error_table['Accuracy'])][1])

#Finally, I will plot the grid search results to compare the models' performances
#Visualize the evaluation results 
visualize_models_results(error_table)

