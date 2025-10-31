import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

print("🔹 Loading Breast Cancer dataset...")
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

print(f"Dataset shape: {X.shape}")
print("Target classes:", data.target_names, "\n")

print("🔹 Splitting dataset (80% train, 20% test)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🔹 Scaling features using StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("🔹 Training Support Vector Machine (SVM)...")
svm = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm.fit(X_train_scaled, y_train)

print("🔹 Making predictions...")
y_pred = svm.predict(X_test_scaled)

print("🔹 Evaluating model performance...")
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)

print(f"""
Model Performance:
Accuracy : {acc:.3f}
Precision: {prec:.3f}
Recall   : {rec:.3f}
F1 Score : {f1:.3f}
""")
print("Confusion Matrix:\n", cm)

print("\n🔹 Saving confusion matrix as 'cancer_confusion_matrix.png'...")
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=data.target_names, yticklabels=data.target_names)
plt.title("Breast Cancer Classification - Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("cancer_confusion_matrix.png", dpi=150)
plt.close()
print("✅ Saved successfully!\n")

print("🎯 Task complete! Check your confusion_matrix.png in the folder.")