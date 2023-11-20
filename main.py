#!/usr/bin/env python
# coding: utf-8

# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tqdm import tqdm

# Loading and Pre-Processing the Fashion MNIST dataset
pic_class = keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = pic_class.load_data()

# Reshaping and normalizing the data
x_train = x_train.reshape((x_train.shape[0], -1)).astype('float32') / 255.0
x_test = x_test.reshape((x_test.shape[0], -1)).astype('float32') / 255.0

# Split data into training, validation, and testing sets
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

# Function to calculate Euclidean distance
def euclidean_distance(x1, x2):
    return np.linalg.norm(x1 - x2)

# Calculate the Euclidean distance for each class
def euclidean_calculator(data, labels):
    class_dfs_ED = []
    class_mean_dictionaries = []
    for i in range(10): 
        class_data = data[labels == i]
        class_mean = np.mean(class_data, axis=0)
        class_mean_dictionaries.append(class_mean)
        class_dfs_ED.append(np.linalg.norm(class_data - class_mean, axis=1))
    return class_dfs_ED, class_mean_dictionaries

class_dfs_ED, class_mean_dictionaries = euclidean_calculator(x_train, y_train)

# Plotting mean representation and samples
# Set plot style
sns.set_style("whitegrid", {'axes.grid': False})

# Create a figure with subplots
fig, axes = plt.subplots(10, 2, figsize=(10, 20))

for i in range(10):
    class_mean_image = class_mean_dictionaries[i].reshape(28, 28)
    closest_sample_idx = np.argmin(class_dfs_ED[i])
    furthest_sample_idx = np.argmax(class_dfs_ED[i])

    # Plotting the mean image for each class
    axes[i, 0].imshow(class_mean_image, cmap='gray_r')
    axes[i, 0].set_title(f"Class {i} Mean Image")
    axes[i, 0].axis('off')

    # Plotting the closest and furthest samples for each class
    closest_sample_image = x_train[closest_sample_idx].reshape(28, 28)
    axes[i, 1].imshow(closest_sample_image, cmap='gray_r')
    axes[i, 1].set_title(f"Class {i} Sample Image")
    axes[i, 1].axis('off')

plt.tight_layout()
plt.show()

# Dimensionality Reduction with PCA
components = [10, 20, 50, 84, 100, 200, 500, 784]
accuracies = []
best_n_components = None
best_accuracy = 0.0
for n_components in tqdm(components, desc='Computing...'):
    pca = PCA(n_components=n_components, random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    x_valid_pca = pca.transform(x_valid)
    
    # Training Linear SVM classifier
    classifier = LinearSVC(random_state=42, max_iter=1000)
    classifier.fit(x_train_pca, y_train)
    
    # Predicting and calculating accuracy
    y_pred = classifier.predict(x_valid_pca)
    accuracy = accuracy_score(y_valid, y_pred)
    accuracies.append(accuracy)
    
    # Updating best accuracy and components
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_n_components = n_components

print(f"Best number of components: {best_n_components}, with accuracy: {best_accuracy * 100:.2f}%")

# Testing the model with the best number of components
pca_test = PCA(n_components=best_n_components, random_state=42)
x_test_pca = pca_test.fit_transform(x_test)

classifier_test = LinearSVC(random_state=42, max_iter=1000)
classifier_test.fit(x_test_pca, y_test)  
y_pred_test = classifier_test.predict(x_test_pca)
accuracy_test = accuracy_score(y_test, y_pred_test)

print(f'Test Accuracy: {accuracy_test * 100:.2f}%')

# RMSE Analysis
k_RMSE_dict = {}
for n_components in components:
    pca = PCA(n_components=n_components, random_state=42)
    x_train_pca = pca.fit_transform(x_train)
    x_valid_pca = pca.transform(x_valid)
    rmse_list = []
    for i in range(10):  # Assuming 10 classes
        x_class = x_valid[y_valid == i]
        x_class_pca = pca.transform(x_class)
        x_class_reconstructed = pca.inverse_transform(x_class_pca)
        rmse = np.sqrt(mean_squared_error(x_class, x_class_reconstructed))
        rmse_list.append(rmse)
    k_RMSE_dict[n_components] = np.mean(rmse_list)

# Print or plot RMSE results
plt.figure(figsize=(10, 6))
plt.plot(list(k_RMSE_dict.keys()), list(k_RMSE_dict.values()), marker='o')
plt.xlabel('Number of PCA Components')
plt.ylabel('Average RMSE')
plt.title('RMSE vs Number of PCA Components')
plt.grid(True)
plt.show()
