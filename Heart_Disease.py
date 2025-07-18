
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# 2. Load dataset
df = pd.read_csv('heart_disease_dataset.csv')

# 3. Fill missing categorical Alcohol Intake with mode
df['Alcohol Intake'] = df['Alcohol Intake'].fillna(df['Alcohol Intake'].mode(dropna=True).iloc[0])

# Fill numeric columns with mean
numeric_cols = df.select_dtypes(include=[np.number]).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].mean())

# Fill remaining categorical columns with mode
categorical_cols = df.select_dtypes(include='object').columns
for col in categorical_cols:
    mode_val = df[col].mode(dropna=True)
    if not mode_val.empty:
        df[col] = df[col].fillna(mode_val.iloc[0])
    else:
        df[col] = df[col].fillna('Unknown')

# Encode categorical variables
label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# 4. Visualizations

# Ensure Alcohol Intake column is numeric after encoding

plt.figure(figsize=(12, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix")
plt.show()

# Scatter plots for key pairs
column_pairs = [
    ('Age', 'Cholesterol'),
    ('Blood Pressure', 'Stress Level'),
    ('Alcohol Intake', 'Cholesterol'),
    ('Age', 'Blood Pressure')
]

for x_col, y_col in column_pairs:
    plt.figure(figsize=(6, 6))
    sns.scatterplot(data=df, x=x_col, y=y_col, hue='Heart Disease', palette='coolwarm')
    plt.title(f'{x_col} vs {y_col} (colored by Heart Disease)')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend(title='Heart Disease')
    plt.tight_layout()
    plt.show()

# Pairplot of important features
selected_features = ['Age', 'Cholesterol', 'Blood Pressure', 'Alcohol Intake', 'Stress Level', 'Heart Disease']
df_pairplot = df[selected_features].copy()
if isinstance(df_pairplot, pd.DataFrame):
    sns.pairplot(df_pairplot, hue='Heart Disease', palette='coolwarm')
    plt.suptitle("Pairwise Relationships", y=1.02)
    plt.show()

scaler = StandardScaler()
X = df.drop('Heart Disease', axis=1)
y = df['Heart Disease']
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# 6. Model Training
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Evaluation
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# 8. Prediction Function
def predict_heart_disease(input_dict):
    input_df = pd.DataFrame([input_dict])

    for col, le in label_encoders.items():
        if col in input_df.columns:
            input_df[col] = le.transform(input_df[col])

    input_scaled = scaler.transform(input_df)
    prediction = model.predict(input_scaled)
    return prediction[0]

# 9. Sample Prediction
new_patient = {
    'Age': 60,
    'Gender': 'Male',
    'Cholesterol': 210,
    'Blood Pressure': 130,
    'Heart Rate': 70,
    'Smoking': 'Never',
    'Alcohol Intake': 'Moderate',
    'Exercise Hours': 3,
    'Family History': 'Yes',
    'Diabetes': 'No',
    'Obesity': 'No',
    'Stress Level': 5,
    'Blood Sugar': 140,
    'Exercise Induced Angina': 'Yes',
    'Chest Pain Type': 'Atypical Angina'
}

result = predict_heart_disease(new_patient)
print("Prediction:", "Heart Disease" if result == 1 else "No Heart Disease")

# 10. Save Model and Encoders
joblib.dump(model, 'heart_disease_model.pkl')
joblib.dump(scaler, 'heart_disease_scaler.pkl')
joblib.dump(label_encoders, 'heart_disease_label_encoders.pkl')
