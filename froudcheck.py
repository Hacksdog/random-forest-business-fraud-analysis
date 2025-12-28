import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
df = pd.read_csv("Fraud_check.csv")
print(df.head())

# Create target variable
df['Risk_Status'] = np.where(
    df['Taxable.Income'] <= 30000, 'Risky', 'Good'
)

df.drop('Taxable.Income', axis=1, inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# EDA
sns.countplot(x='Risk_Status', data=df)
plt.title("Risk Status Distribution")
plt.show()

# Feature & target
X = df.drop('Risk_Status', axis=1)
y = df['Risk_Status']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Random Forest Model
rf = RandomForestClassifier(
    n_estimators=250,
    max_depth=8,
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
plt.title("Feature Importance - Fraud Data")
plt.show()
