# DSAI3202

-Grayscale conversion reduces computational complexity by removing color information, focusing on structural details.
-The function ensures that only valid images are processed, preventing runtime errors.
-The dataset categorization is essential for training a classification model.
-GLCM is great for detecting smooth vs. rough textures, which is helpful in identifying tumor regions.
-LBP is useful for capturing small textural variations within an image, which helps in distinguishing between normal and abnormal tissues.
-These features are more informative than raw pixel values and are crucial for machine learning models.
-Without parallelization, feature extraction would run sequentially, leading to slow processing.
-By using multiprocessing, each image is processed in parallel, making execution significantly faster.
-Combines extracted features into a dataset (X) and assigns labels (y → 1 for tumor, 0 for non-tumor).
-Splits data into training (80%) and testing (20%).
-Uses Support Vector Machine (SVM) with a linear kernel to classify images.
-Evaluates model performance using accuracy score.
-SVM is effective for small datasets, making it a good choice for initial tumor classification.
-The linear kernel assumes that tumor and non-tumor features are linearly separable. If performance is poor, a non-linear kernel (e.g., RBF) may be needed.
-The accuracy metric helps gauge model effectiveness, but additional metrics (precision, recall, F1-score) should be used for medical applications.



