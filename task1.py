import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset
url = "EEG data - Sheet1.csv"
data = pd.read_csv(url)
print(data.head())
if 'Unnamed: 0' in data.columns:
    data = data.drop('Unnamed: 0', axis=1)
# Debug: Inspect data
print("Missing values:", data.isnull().sum().sum())
print("Unique labels:", data.iloc[:, -1].unique())
print("Data shape:", data.shape)
print("Class distribution:", data.iloc[:, -1].value_counts(normalize=True))

# Separate features and labels
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split data (stratify for balance)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --- Task 1: Classification ---
# Define models with tuned parameters
models = {
    "Logistic Regression": LogisticRegression(max_iter=2000, C=0.01, class_weight='balanced'),
    "SVM (Linear)": SVC(kernel='linear', C=0.1, class_weight='balanced'),
    "SVM (RBF)": SVC(kernel='rbf', C=0.01, gamma=0.001, class_weight='balanced'),  # From grid search
    "kNN (k=7)": KNeighborsClassifier(n_neighbors=7)  # Optimized for small data
}

# Train and evaluate
results = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='binary', zero_division=0)
    results[name] = {"Accuracy": accuracy, "Precision": precision}
    print(f"\n{name} Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    if name == "Logistic Regression":
        print(f"{name} iterations: {model.n_iter_[0]} (max: 2000)")

# Print Task 1 results
print("\nTask 1 - Classification Results:")
for name, metrics in results.items():
    print(f"{name}: Accuracy = {metrics['Accuracy']:.3f}, Precision = {metrics['Precision']:.3f}")

# Visualization: Grouped Bar Chart with CV Error Bars
models_list = list(results.keys())
accuracy = [results[name]['Accuracy'] for name in models_list]
precision = [results[name]['Precision'] for name in models_list]
cv_accuracy_std = [cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='accuracy').std()
                   for model in models.values()]
cv_precision_std = [cross_val_score(model, X_train_scaled, y_train, cv=3, scoring='precision').std()
                    for model in models.values()]

fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
r1 = np.arange(len(models_list))
r2 = [x + bar_width for x in r1]

bars1 = ax.bar(r1, accuracy, bar_width, label='Accuracy', color='#1f77b4', yerr=cv_accuracy_std, capsize=5)
bars2 = ax.bar(r2, precision, bar_width, label='Precision', color='#ff7f0e', yerr=cv_precision_std, capsize=5)

for bar in bars1 + bars2:
    yval = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, yval + 0.02, f'{yval:.2f}', ha='center', va='bottom')

ax.set_xlabel('Models', fontweight='bold')
ax.set_ylabel('Score', fontweight='bold')
ax.set_title('Task 1: Classification Performance with CV Std', fontweight='bold')
ax.set_xticks([r + bar_width/2 for r in range(len(models_list))])
ax.set_xticklabels(models_list, rotation=15)
ax.legend()
plt.tight_layout()
plt.show()

# --- Task 2: Feature Selection ---
# 1. Univariate Feature Selection (UFS)
ufs = SelectKBest(score_func=f_classif, k=5)
ufs.fit(X_train_scaled, y_train)
ufs_scores = ufs.scores_
ufs_indices = ufs.get_support(indices=True)
ufs_top_features = X.columns[ufs_indices].tolist()

# 2. Recursive Feature Elimination (RFE) with SVM (Linear)
rfe = RFE(estimator=SVC(kernel='linear', C=0.1), n_features_to_select=5)
rfe.fit(X_train_scaled, y_train)
rfe_indices = rfe.get_support(indices=True)
rfe_top_features = X.columns[rfe_indices].tolist()

# 3. PCA
pca = PCA(n_components=5)
X_train_pca = pca.fit_transform(X_train_scaled)
explained_variance = pca.explained_variance_ratio_
pca_components = pca.components_
pca_top_features = [X.columns[np.argmax(np.abs(comp))] for comp in pca_components]

# Print Task 2 results
print("\nTask 2 - Top 5 Features:")
print("UFS:", ufs_top_features)
print("RFE:", rfe_top_features)
print("PCA:", pca_top_features)

# Visualization: UFS Top Features
top_n = 20
top_indices = np.argsort(ufs_scores)[-top_n:]
top_scores = ufs_scores[top_indices]
top_features = X.columns[top_indices]

plt.figure(figsize=(12, 8))
plt.barh(top_features[::-1], top_scores[::-1], color='#2ca02c')
plt.title(f"Task 2: Top {top_n} UFS Feature Scores", fontweight='bold')
plt.xlabel("F-Score", fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# Visualization: PCA Cumulative Variance
cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 4))
plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o', color='#9467bd')
plt.title("Task 2: PCA Cumulative Explained Variance", fontweight='bold')
plt.xlabel("Number of Components", fontweight='bold')
plt.ylabel("Cumulative Variance Ratio", fontweight='bold')
plt.grid(True)
plt.tight_layout()
plt.show()
