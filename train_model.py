import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle

data = pd.read_csv("data/StudentsPerformance.csv")

data = pd.get_dummies(data, drop_first=True)

X = data.drop("math score", axis=1)
y = data["math score"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

# Save column names
pickle.dump(X.columns, open("columns.pkl", "wb"))

print("Model and columns saved")