import pandas as pd

data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print(data.head())
# Remove customerID column
data.drop("customerID", axis=1, inplace=True)

print("\nAfter Removing customerID:")
print(data.head())
# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

print("\nData Types:")
print(data.dtypes)
print("\nMissing Values:")
print(data.isnull().sum())
# Fill missing values with median
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

print("\nMissing Values After Fix:")
print(data.isnull().sum())
from sklearn.preprocessing import LabelEncoder

for col in data.select_dtypes(include=['object', 'string']).columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

print("\nEncoded Data:")
print(data.head())
# Features and Target
X = data.drop("Churn", axis=1)
y = data["Churn"]

print("\nX Shape:", X.shape)
print("y Shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=5000)

model.fit(X_train, y_train)

print("\nModel Trained Successfully")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("\nAccuracy:", accuracy)
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)
from sklearn.metrics import classification_report

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
import joblib

joblib.dump(model, "churn_model.pkl")

print("Model Saved Successfully!")