# Heart Disease Prediction Model

A machine learning project that predicts the likelihood of heart disease based on various health and lifestyle factors using a Random Forest classifier.

## Overview

This project implements a comprehensive heart disease prediction system that analyzes patient data including demographics, medical history, lifestyle factors, and clinical measurements to predict the probability of heart disease. The model achieves high accuracy through careful data preprocessing, feature engineering, and the use of ensemble learning techniques.

## Features

- **Data Preprocessing**: Handles missing values, encodes categorical variables, and scales numerical features
- **Exploratory Data Analysis**: Comprehensive visualizations including correlation matrices, scatter plots, and pairwise relationships
- **Machine Learning Model**: Random Forest classifier for robust predictions
- **Model Persistence**: Saves trained model and preprocessing components for future use
- **Prediction Interface**: Easy-to-use function for making predictions on new patient data

## Dataset Features

The model uses the following features to make predictions:

- **Demographics**: Age, Gender
- **Clinical Measurements**: Cholesterol levels, Blood Pressure, Heart Rate, Blood Sugar
- **Lifestyle Factors**: Smoking habits, Alcohol Intake, Exercise Hours, Stress Level
- **Medical History**: Family History, Diabetes, Obesity
- **Symptoms**: Exercise Induced Angina, Chest Pain Type

## Requirements


pandas
numpy
matplotlib
seaborn
scikit-learn
joblib


## Installation

1. Clone the repository:

git clone https://github.com/yourusername/Heart-Disease-Prediction-Model.git
cd Heart-Disease-Prediction-Model


2. Install required packages:

pip install pandas numpy matplotlib seaborn scikit-learn joblib


3. Ensure you have the dataset file `heart_disease_dataset.csv` in the project directory

## Usage

### Training the Model

Run the main script to train the model:


python Heart_Disease.py


This will:
- Load and preprocess the dataset
- Generate visualizations for data exploration
- Train the Random Forest model
- Evaluate model performance
- Save the trained model and preprocessing components

### Making Predictions

After training, you can use the saved model to make predictions:


import joblib
import pandas as pd

# Load the saved components
model = joblib.load('heart_disease_model.pkl')
scaler = joblib.load('heart_disease_scaler.pkl')
label_encoders = joblib.load('heart_disease_label_encoders.pkl')

# Example prediction
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

# Make prediction using the predict_heart_disease function
result = predict_heart_disease(new_patient)
print("Prediction:", "Heart Disease" if result == 1 else "No Heart Disease")


## Model Performance

The Random Forest classifier provides:
- High accuracy on test data
- Detailed classification report with precision, recall, and F1-scores
- Confusion matrix for performance analysis
- Feature importance rankings

## Visualizations

The project generates several informative visualizations:

1. **Correlation Matrix**: Shows relationships between all features
2. **Scatter Plots**: Key feature pairs colored by heart disease status
3. **Pairplot**: Comprehensive pairwise relationships of important features

## File Structure


Heart-Disease-Prediction-Model/
│
├── Heart_Disease.py              # Main script with model training and evaluation
├── heart_disease_dataset.csv     # Dataset file
├── heart_disease_model.pkl       # Saved trained model
├── heart_disease_scaler.pkl      # Saved feature scaler
├── heart_disease_label_encoders.pkl # Saved label encoders
├── README.md                     # Project documentation
└── visualization plots/          # Generated visualization files
    ├── Plot 1.png
    ├── Plot 2.png
    └── ...


## Data Preprocessing Steps

1. **Missing Value Handling**:
   - Categorical variables: Filled with mode
   - Numerical variables: Filled with mean

2. **Feature Encoding**:
   - Label encoding for categorical variables
   - Standard scaling for numerical features

3. **Data Splitting**:
   - 80% training, 20% testing
   - Stratified sampling to maintain class balance

## Model Details

- **Algorithm**: Random Forest Classifier
- **Preprocessing**: StandardScaler for feature scaling
- **Evaluation Metrics**: Accuracy, Precision, Recall, F1-Score
- **Cross-validation**: Built-in bootstrap sampling in Random Forest

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add new feature'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Create a Pull Request


## Acknowledgments

- Dataset contributors and medical research community
- Scikit-learn library for machine learning tools
- Matplotlib and Seaborn for visualization capabilities



## Contact

For questions or suggestions, please open an issue in this repository or simpply email me on syeddanishhussain230@gmail.com
Thank you
