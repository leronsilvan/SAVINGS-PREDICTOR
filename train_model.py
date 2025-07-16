import pandas as pd


df = pd.read_csv("data.csv")

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# 🎯 Target variable
y = df['Desired_Savings']

# ✅ Feature variables
X = df.drop(columns=[
    'Desired_Savings', 
    'Desired_Savings_Percentage', 
    'Potential_Savings_Groceries', 
    'Potential_Savings_Transport', 
    'Potential_Savings_Eating_Out',
    'Potential_Savings_Entertainment',
    'Potential_Savings_Utilities',
    'Potential_Savings_Healthcare',
    'Potential_Savings_Education',
    'Potential_Savings_Miscellaneous'
])

# 🔁 Categorical columns to encode
categorical_cols = ['Occupation', 'City_Tier']

# 💡 Use ColumnTransformer to apply OneHotEncoding only to categoricals
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # Keep other columns as is
)

# 📦 ML Pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
])

# 📊 Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧠 Train the model
model.fit(X_train, y_train)

# 📈 Predict and evaluate
y_pred = model.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

import joblib
joblib.dump(model, "savings_predictor.pkl")
print("✅ Model trained and saved!")