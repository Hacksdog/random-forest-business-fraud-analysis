import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("Company_Data.csv")
print(df.head())

# EDA
print(df.info())
print(df.describe())

# Convert Sales to categorical
df['Sales_cat'] = pd.qcut(df['Sales'], q=2, labels=['Low', 'High'])
df.drop('Sales', axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in ['ShelveLoc', 'Urban', 'US', 'Sales_cat']:
    df[col] = le.fit_transform(df[col])

# Feature & target
X = df.drop('Sales_cat', axis=1)
y = df['Sales_cat']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    random_state=42
)
rf.fit(X_train, y_train)

# Prediction
y_pred = rf.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Feature Importance
importance = pd.Series(rf.feature_importances_, index=X.columns)
print("\nFeature Importance:\n", importance.sort_values(ascending=False))

# Plot Feature Importance
importance.sort_values().plot(kind='barh')
plt.title("Feature Importance - Company Data")
plt.show()
