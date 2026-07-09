"""
PAMAP2 Physical Activity Monitoring: SVM vs KNN Classifier comparison
Paramita: July 7, 2026
----------------------------------------------------------
This script trains and evaluates a Support Vector Machine (SVM) against a 
Baseline K-Nearest Neighbors (KNN) model to classify human physical activities.


This script loads `features.csv`, handles missing sensor data (NaNs), scales the features, 
and evaluates the models using 5-Fold Stratified Cross-Validation.
"""

import numpy as np
import pandas as pd
from sklearn.svm import SVC 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score


# STEP 1: LOAD DATA AND MAP ACTIVITY LABELS

# Load the extracted features dataset

print("Loading features.csv...")
full_df = pd.read_csv('features.csv') #

# Dictionary to map numerical PAMAP2 IDs to readable labels
activity_names = {
    1: 'Lying', 2: 'Sitting', 3: 'Standing', 4: 'Walking', 
    5: 'Running', 6: 'Cycling', 7: 'Nordic Walking', 
    12: 'Ascending Stairs', 13: 'Descending Stairs', 
    16: 'Vacuuming', 17: 'Ironing', 24: 'Rope Jumping'
}


# STEP 2: RECONSTRUCT THE FEATURE LIST

# dynamically rebuilding the exact 325 feature column names generated in the extraction script
device_list = ['hand', 'chest', 'ankle']
axes_list = ['x', 'y', 'z', 'amp']
sensor_list = ['acc16', 'gyro']
stat_list = ['mean', 'hmean', 'std', 'max', 'min', 'median', 'p2p', 
             'mad', 'iqr', 'sum_abs', 'mean_energy', 'skew', 'kurtosis']

feature_list = []

# Adding Heart Rate Features
for stat in stat_list:
    feature_list.append('heart_rate_'+stat)

# 2. Adding IMU Statistical and Correlation Features
for device in device_list:
    for sensor in sensor_list:
        # Stats
        for axis in axes_list:
            for stat in stat_list:
                label = '_'.join([device, sensor, axis, stat])
                feature_list.append(label)
        
        # Correlations
        for pair in ['xy', 'xz', 'yz']:
            label = '_'.join([device, sensor, 'corr', pair])
            feature_list.append(label)

# Defining X (325 predictive features) and y (target activity IDs)
X = full_df[feature_list]
y = full_df['activity_id']


# STEP 3: DEFINE CROSS-VALIDATION STRATEGY

# StratifiedKFold ensures that each fold contains roughly the same proportion 
# of each activity class as the original dataset. This prevents scenarios where 
# a rare activity (like Rope Jumping) is entirely missing from a training fold.

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)


# STEP 4: BUILD MACHINE LEARNING PIPELINES


#  Support Vector Machine (SVM)
# SVM attempts to find a mathematical hyperplane (boundary) that maximizes the 
# margin between different classes in high-dimensional space.

svm_pipeline = Pipeline([
    # Imputer: Wireless sensors drop packets. This replaces any NaN values with the column's mean.
    ("imputer", SimpleImputer(strategy="mean")),
    
    # Scaler: SVM relies on geometric distances. If 'Heart Rate' is in the 100s and 
    # 'Acceleration' is in decimals, the SVM fails. Scaler normalizes all features to mean=0, std=1.
    ("scaler", StandardScaler()),
    
    # The Model: Linear kernel is used because we have more features (325) than samples (105).
    # A linear boundary is highly effective and less prone to overfitting in this scenario.
    ("SVM", SVC(C=1.0, kernel='linear', random_state=42))
])

# Optional: Can be skipped: I wanted to compare with the Baseline K-Nearest Neighbors (KNN) 
# KNN classifies a sample based on the "majority vote" of its k nearest neighbors in the training data.

knn_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler()),
    ("KNN", KNeighborsClassifier(n_neighbors=5)) 
])


# STEP 5: TRAIN & PREDICT VIA CROSS-VALIDATION

print("Running 5-Fold Cross-Validation for SVM...")
# cross_val_predict trains the model 5 times (on 4/5ths of the data each time) 
# and returns predictions for the 1/5th test chunk, combining them into a full prediction array.
y_pred_svm = cross_val_predict(svm_pipeline, X, y, cv=cv)

print("Running 5-Fold Cross-Validation for KNN (Baseline)...")
y_pred_knn = cross_val_predict(knn_pipeline, X, y, cv=cv)


# STEP 6: EVALUATE AND COMPARE MODELS

# Generate detailed classification metrics (precision, recall, f1-score)

report_svm = classification_report(y, y_pred_svm, zero_division=0, output_dict=True)
report_knn = classification_report(y, y_pred_knn, zero_division=0, output_dict=True)

comparison_data = []

# Iterate through every unique activity present in the dataset
for act_id in np.unique(y):
    str_id = str(act_id)
    name = activity_names.get(act_id, f"Activity {act_id}")
    
    # Extract the F1-Score (balances False Positives and False Negatives)
    svm_f1 = report_svm[str_id]['f1-score']
    knn_f1 = report_knn[str_id]['f1-score']
    
    # Crown a winner
    if svm_f1 > knn_f1:
        winner = 'SVM 🏆'
    elif knn_f1 > svm_f1:
        winner = 'KNN 🏆'
    else:
        winner = 'Tie'
        
    comparison_data.append({
        'ID': act_id,
        'Activity Name': name,
        'SVM F1': f"{svm_f1:.3f}",
        'KNN F1': f"{knn_f1:.3f}",
        'Winner': winner
    })

# Convert results to a pandas DataFrame for a clean console printout

comp_df = pd.DataFrame(comparison_data)

print("\n" + "="*55)
print("--- SVM vs. KNN: CLASS-BY-CLASS COMPARISON ---")
print("="*55)
print(comp_df.to_string(index=False))

# Print final overarching metrics
print("\n" + "="*55)
print(f"Overall SVM Accuracy: {accuracy_score(y, y_pred_svm):.3f}")
print(f"Overall KNN Accuracy: {accuracy_score(y, y_pred_knn):.3f}")
print("="*55)
print("Pipeline execution complete.")