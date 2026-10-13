# T9: Data Preprocessing and Support Vector Machine (SVM)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report


# 1. Load Dataset
iris = load_iris()

X = iris.data
y = iris.target

print("Dataset loaded successfully")
print("Dataset shape:", X.shape)


# 2. Create DataFrame
df = pd.DataFrame(X, columns=iris.feature_names)
df["target"] = y

print("\nFirst 5 rows:")
print(df.head())


# 3. Check Dataset Information
print("\nDataset Information:")
print(df.info())


# 4. Check Missing Values
print("\nMissing Values:")
print(df.isnull().sum())


# 5. Statistical Description
print("\nStatistical Description:")
print(df.describe())


# 6. Separate Features and Target
X = df.drop("target", axis=1)
y = df["target"]

print("\nFeatures:")
print(X.head())

print("\nTarget:")
print(y.head())


# 7. Split Dataset into Training and Testing
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)


# 8. Data Preprocessing - Feature Scaling
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nFeature scaling completed")


# 9. Create SVM Classifier
svm_model = SVC(kernel="linear")

print("SVM model created")


# 10. Train SVM Model
svm_model.fit(X_train_scaled, y_train)

print("SVM model trained successfully")


# 11. Make Predictions
y_pred = svm_model.predict(X_test_scaled)

print("\nPredicted Values:")
print(y_pred)


# 12. Compare Actual and Predicted Values
comparison = pd.DataFrame({
    "Actual": y_test.values,
    "Predicted": y_pred
})

print("\nActual vs Predicted:")
print(comparison)


# 13. Calculate Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
print("Accuracy Percentage:", accuracy * 100, "%")


# 14. Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)


# 15. Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 16. Plot Confusion Matrix
plt.figure(figsize=(6, 5))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Class")
plt.ylabel("Actual Class")
plt.colorbar()

plt.xticks([0, 1, 2])
plt.yticks([0, 1, 2])

plt.show()


# 17. Final Result
print("\nT9 completed successfully.")
print("Data preprocessing and SVM classification performed successfully.")
