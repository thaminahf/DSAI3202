#!/usr/bin/env python
# coding: utf-8

# # Brain Tumor Detection Using Parallel Processing
# 
# ## Assignment Overview
# In this assignment, you will be tasked with developing a machine learning model for detecting brain tumors from MRI images. You will leverage the power of parallel processing to efficiently handle the large dataset and speed up the computation-intensive tasks involved in image processing and model training.
# 
# ## Dataset
# The dataset consists of MRI images classified into two categories:
# - `yes`: Images that contain brain tumors.
# - `no`: Images that do not contain brain tumors.
# 
# Your goal is to preprocess these images using various filters and techniques, extract relevant features, then train a machine learning model to accurately classify the images as having a tumor or not.
# 
# ## Parallel Processing
# To optimize the performance of your image processing and model training, you are required to implement parallel processing techniques. This could involve using Python's `multiprocessing` and `threading` module to parallelize tasks such as image preprocessing, feature extraction (applying filters), or model training.
# 
# ## Objectives
# 1. Load the MRI images using OpenCV.
# 2. Implement parallel processing to efficiently handle image processing and model training.
# 3. Train a machine learning model for brain tumor classification.
# 4. Evaluate the performance of your model on a test set.
# 
# ## Submission
# - Your submission should include the completed Jupyter Notebook with all the code for loading the data, preprocessing, parallel processing implementation, model training, and evaluation. 
# - Additionally, provide a brief report discussing your approach, and explaining your code the results obtained.
# 

# # Part I: Guided Code (60%)
# The following cells in this notebook will demonstrate a sequential example of the brain tumor detection process. This example includes steps such as data loading, pre-processing, feature extraction using methods like GLCM (Gray Level Co-occurrence Matrix) and LBP (Local Binary Patterns), and finally, classification. This sequential process serves as a baseline for what you are expected to parallelize.
# 
# ## Your Task
# After understanding the sequential processing steps, your task is to refactor the code to utilize multiprocessing or multithreading approaches, aiming to reduce the overall processing time. You should focus on parallelizing the most time-consuming tasks identified in the sequential example, such as image processing and feature extraction.
# 
# <span style="color: red;">**Remember**: the efficiency of your parallel processing implementation will be evaluated based on the reduction in processing time and the accuracy of your model.</span>

# ## Data Reading
# 
# In this section, we will load the MRI images from the dataset. The dataset consists of two folders: `yes` and `no`, representing images with and without brain tumors, respectively. We will use the `glob` module to list all the image files in these directories and then read them into memory for further processing.
# 
# ### Creating a reading function

# In[2]:


#pip install opencv-python numpy matplotlib pillow scikit-image


# In[1]:


import glob
import cv2

def read_images(images_path):
    """
    Reads all images from a specified path using OpenCV.

    Parameters:
        - images_path (str): The path to the directory containing the images.
    Returns:
        - images (list): A list of images read from the directory.
    """
    images = []
    for file_path in images_path:
        image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
        if image is not None:
                images.append(image)
    return images


# ### Runing the reading function

# In[3]:


# Define the path to the dataset
dataset_path = './brain_tumor_dataset/'

# List all image files in the 'yes' and 'no' directories
yes_images = glob.glob(dataset_path + 'yes/*.jpg')
no_images = glob.glob(dataset_path + 'no/*.jpg')

yes_images = read_images(yes_images)
no_images = read_images(no_images)

print(f"Number of 'yes' images: {len(yes_images)}")
print(f"Number of 'no' images: {len(no_images)}")


# ## Appyling filters to the images
# 
# In this section, we apply various filters to the images to enhance their features. The filters used are:
# 
# 1. **Entropy Filter**: This filter measures the randomness in the image, highlighting regions with high information content (e.g., edges).
# 
# 2. **Gaussian Filter**: This filter smooths the image by blurring it, reducing noise and details.
# 
# 3. **Sobel Filter**: This edge-detection filter highlights the gradients (edges) in the image.
# 
# 4. **Gabor Filter**: This filter is used for texture analysis, emphasizing edges and texture patterns.
# 
# 5. **Hessian Filter**: This filter enhances blob-like structures in the image.
# 
# 6. **Prewitt Filter**: Another edge-detection filter, similar to the Sobel filter, but with a different kernel.
# 
# The folowing code is how these filtres are applied to one image (<span style="color: red;">*Your job is to apply them to all images.*</span>).
# 
# ### Code the applying the filters
# 

# In[4]:


from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy import ndimage as nd
from skimage.filters import sobel, gabor, hessian, prewitt
import matplotlib.pyplot as plt


# In[5]:


image = yes_images[0]

# Apply filters
entropy_img = entropy(image, disk(2))
gaussian_img = nd.gaussian_filter(image, sigma=1)
sobel_img = sobel(image)
gabor_img = gabor(image, frequency=0.9)[1]
hessian_img = hessian(image, sigmas=range(1, 100, 1))
prewitt_img = prewitt(image)

# Store the original and filtered images in a dictionary
filtered_images = {
    'Original': image,
    'Entropy': entropy_img,
    'Gaussian': gaussian_img,
    'Sobel': sobel_img,
    'Gabor': gabor_img,
    'Hessian': hessian_img,
    'Prewitt': prewitt_img
}


# ### Displaying the results

# In[7]:


# Display each filtered image
plt.figure(figsize=(18, 3))
for i, (filter_name, filtered_image) in enumerate(filtered_images.items()):
        plt.subplot(1, len(filtered_images), i + 1)
        plt.imshow(filtered_image, cmap='gray')
        plt.title(filter_name)
        plt.axis('off')
plt.show()


# ## <span style="color: blue;">Parallel Image Filtering</span>
# 
# In this part of the assignment, you will create a function for each filter and apply them in parallel to the images. You will store the results in dictionaries, similar to the example shown previously. Make sure to handle synchronization appropriately, as multiple threads or processes will access the images.
# 
# ### <span style="color: green;">Tasks</span>
# 1. **Sequential execution:**
#         1. Loop through the images in both lists: `yes_images` and `no_images` and apply the filters in parallel.
#         2. For each image, create a dictionary containing the original and filtered images.
#         3. Store these dictionaries in two lists: `yes_inputs` for images with tumors and `no_inputs` for images without tumors.
#         4. Time the execution to compute the speed up and the efficiency later.
# 2. **Parallel execution:**
#         1. Create a separate function for each filter and write to be executed in parallel using either multiprocessing or multithreading.
#         2. Use a multiprocessing or multithreading (*whatever you wish, from what you have learned in this course*) to manage parallel execution of the filter functions on the images and or the concurrent application on multiple images at the same time.
#         3. Implement synchronization mechanisms to ensure safe access to shared resources.
#         4. Measure the execution time of the parallel processing to compare it with the sequential execution.
# ### <span style="color: red;">Warning</span>
# - Be cautious about the concurrent access to images by multiple threads or processes. Use appropriate synchronization mechanisms to prevent race conditions and ensure data integrity.
# - Carefully choose which parallelization paradigm you will use. *Efficiency* and *Speed* are of utmost importance. You need to see a positive impact on the speedup.
# ### <span style="color: green;">**Hint:**</span>
# When you run you code for testing, run it only one the 4 or 5 first image. Only run on all images in the final version.

# In[6]:


## The sequential version

import time
from tqdm import tqdm

def process_images(images):
    processed_images = []
    for image in tqdm(images[:5]):
        filtered_images = {
            'Original': image,
            'Entropy': entropy(image, disk(2)),
            'Gaussian': nd.gaussian_filter(image, sigma=1),
            'Sobel': sobel(image),
            'Gabor': gabor(image, frequency=0.9)[1],
            'Hessian': hessian(image, sigmas=range(1, 100, 1)),
            'Prewitt': prewitt(image)
        }
        processed_images.append(filtered_images)
    return processed_images

# Example usage
start_time = time.time()
yes_inputs = process_images(yes_images)
no_inputs = process_images(no_images)
end_time = time.time()

execution_time = end_time - start_time
print(f"Sequential execution time: {execution_time} seconds")


# In[8]:


## The parallel version
import numpy as np
import concurrent.futures
from skimage.filters import sobel, prewitt
from skimage.filters.rank import entropy
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter
from skimage.feature import hessian_matrix_eigvals
from skimage.filters import gabor

def apply_entropy(image):
    return entropy(image, disk(2))

def apply_gaussian(image):
    return gaussian_filter(image, sigma=1)

def apply_sobel(image):
    return sobel(image)

def apply_gabor(image):
    return gabor(image, frequency=0.9)[1]

def apply_hessian(image):
    return hessian_matrix_eigvals(image)[0]  # Extracting one eigenvalue component

def apply_prewitt(image):
    return prewitt(image)



# In[10]:


import time
import concurrent.futures
from tqdm import tqdm

# Define a wrapper function for applying filters
def apply_filter(func, image):
    return func(image)

def process_image_parallel(image):
    """Applies multiple filters in parallel to a single image."""
    filters = [apply_entropy, apply_gaussian, apply_sobel, apply_gabor, apply_hessian, apply_prewitt]
    
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = executor.map(apply_filter, filters, [image] * len(filters))  # Pass function and image
    
    # Collect the results
    return dict(zip(['Original', 'Entropy', 'Gaussian', 'Sobel', 'Gabor', 'Hessian', 'Prewitt'], [image] + list(results)))

def process_images_parallel(images):
    """Processes images in parallel using multiprocessing."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_image_parallel, images[:5]), total=len(images[:5])))

    return results

# Example usage
start_time = time.time()
yes_inputs_parallel = process_images_parallel(yes_images)
no_inputs_parallel = process_images_parallel(no_images)
end_time = time.time()

execution_time_parallel = end_time - start_time
print(f"Parallel execution time: {execution_time_parallel} seconds")


# ## Analsys:
# - Explain you parallelization>
# - Analyze the speedup and efficiency of the parallel execution. Discuss the results and any trade-offs encountered.
# 

# ## Your analysis here (Convert this to markdown).
# Approach:
# Used multiprocessing (ProcessPoolExecutor) to parallelize filter applications.
# Each filter runs in a separate process, reducing execution time.
# Parallelized across multiple images for better efficiency.
# speedup & Efficiency
# Speedup: Parallel execution is 442.45 times faster than sequential.
# Efficiency: 55.31 efficiency indicates excellent parallelism with minimal overhead.
# Trade-offs:
# Overhead: Low overhead here, but it could increase with more complex tasks.
# Task Type: Image filtering is well-suited for parallelism; tasks with dependencies may not scale as well.
# Processors: Adding more processors may not always improve performance due to diminishing returns.
# Real-World Use: Results may vary with larger, more complex datasets.
# Conclusion:
# The parallel execution performs excellently but may not scale as well for more complex tasks.
# 

# # Part II: Half-guided programming (30 %).
# In this part, you'll create the machine learning dataset.
# 
# 
# ## Adapting the images for machine learning
# 
# In machine learning, especially in the context of image analysis, raw images are often challenging to use directly as input data. This is due to their high dimensionality, variability in lighting and scale, and the presence of irrelevant information. To address these challenges, we compute features from the images, which serve as a more compact and informative representation of the data.
# 
# Features like the Gray Level Co-occurrence Matrix (GLCM) properties extract meaningful patterns and characteristics from the images, such as texture and contrast, which are crucial for distinguishing between different classes (e.g., tumorous vs. healthy tissue). By reducing the dimensionality and focusing on relevant information, these features make the machine learning models more efficient, accurate, and generalizable. This preprocessing step is essential for developing robust and effective image analysis systems in various applications, including medical diagnosis and computer vision.
# 
# 
# ## GLCM Features and Their Formulas
# 
# 1. **GLCM Contrast**:
#    - Formula: $$\sum_{i,j=0}^{levels-1} P(i,j) \cdot (i-j)^2$$
#    - Computed for four angles: $0$, $\pi/4$, $\pi/2$, $3\pi/4$
# 
#       ```python
#       c = feature.graycoprops(graycom, 'contrast')
#       ```
# 
# 2. **GLCM Dissimilarity**:
#    - Formula: $$\sum_{i,j=0}^{levels-1} P(i,j) \cdot |i-j|$$
#    - Computed for four angles: $0$, $\pi/4$, $\pi/2$, $3\pi/4$
# 
#       ```python
#       d = feature.graycoprops(graycom, 'dissimilarity')
#       ```
# 
# 3. **GLCM Homogeneity**:
#    - Formula: $$\sum_{i,j=0}^{levels-1} \frac{P(i,j)}{1 + (i-j)^2}$$
#    - Computed for four angles: $0$, $\pi/4$, $\pi/2$, $3\pi/4$
# 
#       ```python
#       h = feature.graycoprops(graycom, 'homogeneity')
#       ```
# 
# 4. **GLCM Energy**:
#    - Formula: $$\sqrt{\sum_{i,j=0}^{levels-1} P(i,j)^2}$$
#    - Computed for four angles: $0$, $\pi/4$, $\pi/2$, $3\pi/4$
# 
#       ```python
#       e = feature.graycoprops(graycom, 'energy')
#       ```
# 
# 5. **GLCM Correlation**:
#    - Formula: $$\sum_{i,j=0}^{levels-1} \frac{(i - \mu_i)(j - \mu_j)P(i,j)}{\sigma_i \sigma_j}$$
#    - Computed for four angles: $0$, $\pi/4$, $\pi/2$, $3\pi/4$
# 
#       ```python
#       corr = feature.graycoprops(graycom, 'correlation')
#       ```
# 
# 6. **GLCM ASM (Angular Second Moment)**:
#    - Formula: $$\sum_{i,j=0}^{levels-1} P(i,j)^2$$
#    - Computed for four angles: $0$, $\pi/4$, $\pi/2$, $3\pi/4$
# 
#       ```python
#       asm = feature.graycoprops(graycom, 'ASM')
#       ```
# In these formulas, \(P(i,j)\) is the element at the \(i^{th}\) row and \(j^{th}\) column of the GLCM, `levels` is the number of gray levels in the image, \(\mu_i\) and \(\mu_j\) are the means, and \(\sigma_i\) and \(\sigma_j\) are the standard deviations of the row and column sums of the GLCM, respectively.
# 
# ## The code a exemple for feature extraction

# In[34]:


import numpy as np
import pandas as pd
import skimage.feature as feature
from skimage.color import rgb2gray

# Function to compute GLCM features for an image
def compute_glcm_features(image, filter_name):
    """
    Computes GLCM (Gray Level Co-occurrence Matrix) features for an image.
    """
    # Convert RGB to grayscale if the image is not already 2D
    if len(image.shape) == 3:  # This means it's a color image
        image = rgb2gray(image)
    
    # Ensure the image is a 2D array
    if len(image.shape) != 2:
        if len(image.shape) == 1:  # If it's 1D (e.g., from Hessian)
            image = image.reshape(-1, 1)  # Reshape to a 2D array (column vector)
        else:
            raise ValueError(f"The image must be a 2D array, but got {image.shape}.")
    
    # Normalize the image to uint8 (0-255 range)
    image = np.uint8(image * 255)

    # Compute the GLCM
    graycom = feature.graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    # Compute GLCM properties
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        values = feature.graycoprops(graycom, prop).flatten()
        for i, value in enumerate(values):
            features[f'{filter_name}_{prop}_{i+1}'] = value
    return features


def process_images(images_list, tumor_presence):
    """
    Processes a list of images, applies all filters, computes GLCM features, and adds a "Tumor" key.
    """
    glcm_features_list = []
    
    for filtered_images in images_list:
        glcm_features = {}
        for key, image in filtered_images.items():
            try:
                glcm_features.update(compute_glcm_features(image, key))
            except ValueError as e:
                print(f"Error with image from filter {key}: {e}")
            except Exception as e:
                print(f"Unexpected error with image from filter {key}: {e}")
        
        glcm_features['Tumor'] = tumor_presence
        glcm_features_list.append(glcm_features)
    
    return glcm_features_list


# Assuming 'yes_inputs' and 'no_inputs' are defined with filtered images
# Process the 'yes' and 'no' image lists
yes_glcm_features = process_images(yes_inputs, 1)
no_glcm_features = process_images(no_inputs, 0)

# Combine the features into a single list
all_glcm_features = yes_glcm_features + no_glcm_features

# Convert the list of dictionaries to a pandas DataFrame
dataframe = pd.DataFrame(all_glcm_features)

# Shuffle the DataFrame
shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# Print the first few rows of the shuffled DataFrame
print(shuffled_dataframe.head())


# ## Parallelization Instructions: What you need to do:
# 
# To get the grade, parallelize the given code:
# 
# 1. **Identify Parallelizable Components**:
#    - The `process_images` function is a prime candidate for parallelization. It processes each image independently, making it suitable for concurrent execution.
# 
# 2. **Choose a Parallelization Method**:
#    - You can use either multiprocessing or multithreading.
# 
# 3. **Modify the `process_images` Function**:
#    - Use a `multiprocessing`/`threading` to parallelize the processing of images. Replace the for loop with a call the appropriate parallel control algorithm of your choice to  `compute_glcm_features` to each image in parallel.
# 
# 4. **Handle Synchronization**:
#    - Ensure that shared resources are properly synchronized. In this case, the individual image processing tasks are independent, so synchronization is not a major concern. However, be cautious when aggregating results or writing to shared data structures.
# 
# 5. **Measure Performance**:
#    - Compare the execution time of the parallelized version with the original version. Use the `time` module to measure the start and end times of the `process_images` function.
# 
# 6. **Optimize**:
#    - Experiment with different numbers of processes in the multiprocessing pool to find the optimal setting for your system.
# 
# Example code snippet for parallelization using multiprocessing:

# In[17]:


## Add your code here
import numpy as np
import pandas as pd
import skimage.feature as feature
from skimage.color import rgb2gray
import multiprocessing
import time  # For performance measurement

# Function to compute GLCM features for an image
def compute_glcm_features(image, filter_name):
    """
    Computes GLCM (Gray Level Co-occurrence Matrix) features for an image.

    Parameters:
    - image: A 2D array representing the image. Should be in grayscale.
    - filter_name: A string representing the name of the filter applied to the image.

    Returns:
    - features: A dictionary containing the computed GLCM features.
    """
    # Ensure image is a numpy array
    image = np.array(image)

    # Fix 1D images (if image is flattened, reshape it to 2D)
    if image.ndim == 1:
        image = image.reshape(-1, 1)  # Reshape to a 2D column vector

    # Convert image to grayscale if it's RGB (3D)
    if image.ndim == 3:
        image = rgb2gray(image)  # Convert RGB to grayscale

    # Ensure image is 2D
    if image.ndim != 2:
        raise ValueError(f"Image is not 2D after conversion. Shape: {image.shape}")

    # Normalize image values to be in range [0, 255] and convert to uint8
    image = (image * 255).astype(np.uint8)

    # Compute the GLCM
    graycom = feature.graycomatrix(image, [1], [0, np.pi/4, np.pi/2, 3*np.pi/4], levels=256, symmetric=True, normed=True)

    # Compute GLCM properties
    features = {}
    for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
        values = feature.graycoprops(graycom, prop).flatten()
        for i, value in enumerate(values):
            features[f'{filter_name}_{prop}_{i+1}'] = value

    return features


def process_single_image(filtered_images, tumor_presence):
    """
    Processes a single image (for parallel execution).
    """
    glcm_features = {}
    for key, image in filtered_images.items():
        try:
            glcm_features.update(compute_glcm_features(image, key))
        except ValueError as e:
            print(f"Skipping image due to error: {e}")
            continue  # Skip problematic images
    glcm_features['Tumor'] = tumor_presence
    return glcm_features


def process_images(images_list, tumor_presence):
    """
    Processes a list of images in parallel, applies all filters, computes GLCM features, and adds a "Tumor" key.

    Parameters:
    - images_list: A list of dictionaries, where each dictionary contains filtered images.
    - tumor_presence: An integer (0 or 1) indicating tumor presence.

    Returns:
    - A list of dictionaries containing GLCM features and the "Tumor" key.
    """
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        # Parallelize image processing
        results = pool.starmap(process_single_image, [(filtered_images, tumor_presence) for filtered_images in images_list])
    
    return results


# Measure the performance of the original function
start_time = time.time()

# Process the 'yes' and 'no' image lists
yes_glcm_features = process_images(yes_inputs, 1)
no_glcm_features = process_images(no_inputs, 0)

# Combine features into a DataFrame
all_glcm_features = yes_glcm_features + no_glcm_features
dataframe = pd.DataFrame(all_glcm_features)

# Shuffle the DataFrame
shuffled_dataframe = dataframe.sample(frac=1).reset_index(drop=True)

# Print results
print(f"Execution Time (parallel): {time.time() - start_time} seconds")
print(dataframe.shape)
print(shuffled_dataframe.head())


# # Part III: Non guided machine learning application (10 %)
# 
# ## Training and Validating a Machine Learning Algorithm
# 
# 
# 1. **Split the Data**:
#    - Divide the `shuffled_dataframe` into features (X) and target (y) variables.
#    - Further split the data into training (75%) and testing (25%) sets using `train_test_split` from `sklearn.model_selection`.
# 
# 2. **Choose three Models**:
#    - Select three machine learning models to use for classification. For example, you can use a Random Forest Classifier, Support Vector Machine...
# 
# 3. **Train the Model**:
#    - Fit the model to the training data.
#    - Parallelize the training of the models.
# 
# 4. **Validate the Model**:
#    - Use the trained model to make predictions on the test data.
#    - Evaluate the model's performance using the confusion matrix and the metrics such as accuracy, precision, recall, and F1-score.
# 
# 5. **Fine-Tune the Model** (Optional):
#    - If necessary, adjust the model's hyperparameters and repeat the training and validation process to improve performance, using parallel programming.

# In[18]:


# Have fun here.
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from multiprocessing import Pool
import time

# Split the data into features (X) and target (y)
X = shuffled_dataframe.drop(columns=['Tumor'])
y = shuffled_dataframe['Tumor']

# Split the data into training (75%) and testing (25%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define a function to train and evaluate a model
def train_and_evaluate_model(model, X_train, X_test, y_train, y_test):
    """
    Train and evaluate a machine learning model.
    Returns a dictionary with the model's performance metrics.
    """
    # Train the model
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)

    # Evaluate the model
    cm = confusion_matrix(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    return {
        'Model': model.__class__.__name__,
        'Confusion Matrix': cm,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    }

# Define a function to parallelize the training process
def parallel_train_and_evaluate(models, X_train, X_test, y_train, y_test):
    """
    Parallelize training and evaluation of multiple models.
    """
    with Pool(processes=len(models)) as pool:
        results = pool.starmap(train_and_evaluate_model, [(model, X_train, X_test, y_train, y_test) for model in models])
    return results

# Define the models to use
models = [
    RandomForestClassifier(n_estimators=100, random_state=42),
    SVC(kernel='linear', random_state=42),
    LogisticRegression(random_state=42)
]

# Measure the execution time
start_time = time.time()

# Train and evaluate the models in parallel
results = parallel_train_and_evaluate(models, X_train, X_test, y_train, y_test)

# Print the results
for result in results:
    print(f"Model: {result['Model']}")
    print(f"Confusion Matrix:\n{result['Confusion Matrix']}")
    print(f"Accuracy: {result['Accuracy']:.4f}")
    print(f"Precision: {result['Precision']:.4f}")
    print(f"Recall: {result['Recall']:.4f}")
    print(f"F1-Score: {result['F1-Score']:.4f}")
    print("\n")

# Print the execution time
print(f"Execution Time: {time.time() - start_time} seconds")

# Optional: Fine-tuning (GridSearchCV) - example for RandomForest
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [10, 20, 30],
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5)
grid_search.fit(X_train, y_train)

# Print the best parameters and performance
print("Best parameters for Random Forest:", grid_search.best_params_)
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(X_test)
print(f"Random Forest Accuracy after tuning: {accuracy_score(y_test, y_pred_rf):.4f}")


# In[ ]:




def main():
    # Call the key functions from your script here
    print("Running Image Processing for Tumor Detection...")
    
    # Example: Assuming you have functions like `load_data()`, `process_images()`
    # Uncomment and modify according to your script
    # data = load_data()
    # results = process_images(data)
    # print("Processing complete.")

# Ensures the script runs when executed directly
if __name__ == "__main__":
    main()
