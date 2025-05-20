# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier

# Load Dataset
df = pd.read_csv("heart_disease system_uci.csv")  # Change to your correct filename

# Basic info
print(df.head())
print(df.info())

# Drop unnecessary columns if any (like 'id')
if 'id' in df.columns:
    df = df.drop('id', axis=1)

# Handle missing values
df = df.dropna()  # or df.fillna(method='ffill')

# Convert categorical columns to numeric
for col in df.select_dtypes(include='object').columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# Identify target column
print(df.columns)
target_col = 'num'  # OR 'target' as per dataset
df[target_col] = df[target_col].apply(lambda x: 1 if x > 0 else 0)

# Pie Chart - Target distribution
plt.figure(figsize=(6, 6))
df[target_col].value_counts().plot.pie(labels=["No Disease", "Disease"], autopct='%1.1f%%', colors=['skyblue', 'salmon'])
plt.title("Heart Disease Distribution")
plt.ylabel("")
plt.show()

# Bar Graph - Count of each chest pain type (replace 'cp' if not present)
if 'cp' in df.columns:
    plt.figure(figsize=(6, 4))
    sns.countplot(x='cp', data=df, palette='Set2')
    plt.title("Count of Chest Pain Types")
    plt.xlabel("Chest Pain Type")
    plt.ylabel("Count")
    plt.show()

# Histogram - Age distribution
if 'age' in df.columns:
    plt.figure(figsize=(6, 4))
    plt.hist(df['age'], bins=20, color='teal', edgecolor='black')
    plt.title("Age Distribution")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

# Prepare data
X = df.drop(target_col, axis=1)
y = df[target_col]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)

# Models to compare
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(),
    "SVM": SVC(probability=True),
    "KNN": KNeighborsClassifier()
}

# Track model accuracies
model_accuracies = {}

# Training and evaluating models
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    model_accuracies[name] = acc
    print(f"\n{name} Accuracy: {acc:.2f}")
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

# Line graph - Model Accuracies
plt.figure(figsize=(8, 5))
plt.plot(list(model_accuracies.keys()), list(model_accuracies.values()), marker='o', linestyle='-', color='purple')
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(True)
plt.show()

# Heatmap - Feature correlation
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation")
plt.show()

# ROC Curve - Random Forest
rf_model = RandomForestClassifier()
rf_model.fit(X_train, y_train)
y_prob = rf_model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.title('ROC Curve - Random Forest (Binary Classification)')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend(loc="lower right")
plt.show()
