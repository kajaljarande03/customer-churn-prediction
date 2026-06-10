import pandas as pd

# Load dataset
data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")

print("Dataset Loaded Successfully")
print()

print("First 5 Records:")
print(data.head())

print()
print("Dataset Shape:")
print(data.shape)
print("\nDataset Information:")
print(data.info())

print("\nMissing Values:")
print(data.isnull().sum())
# Remove customerID column
data.drop("customerID", axis=1, inplace=True)

print("\nAfter Removing customerID:")
print(data.head())
# Convert TotalCharges to numeric
data["TotalCharges"] = pd.to_numeric(
    data["TotalCharges"],
    errors="coerce"
)

print("\nData Types After Conversion:")
print(data.dtypes)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()

for col in data.columns:
    if data[col].dtype == "object":
        data[col] = le.fit_transform(data[col])

print("\nEncoded Dataset:")
print(data.head())
from sklearn.preprocessing import LabelEncoder

for col in data.select_dtypes(include=['object', 'string']).columns:
    encoder = LabelEncoder()
    data[col] = encoder.fit_transform(data[col])

print(data.head())
print("\nMissing Values:")
print(data.isnull().sum())
# Fill missing values in TotalCharges with median
data["TotalCharges"] = data["TotalCharges"].fillna(
    data["TotalCharges"].median()
)

print("\nAfter Fixing Missing Values:")
print(data.isnull().sum())
X = data.drop("Churn", axis=1)
y = data["Churn"]

print("X Shape:", X.shape)
print("y Shape:", y.shape)
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("Training Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

print("Model Trained Successfully")
from sklearn.metrics import accuracy_score

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)