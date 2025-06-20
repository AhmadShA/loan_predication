import joblib
import numpy as np
import pandas as pd
from pathlib import Path

def load_model():
    try:
        # Load the trained model using joblib
        model_path = Path('loan_manager/ml/model.joblib')
        model = joblib.load(model_path)
        return model
    except FileNotFoundError:
        raise FileNotFoundError("Model file not found. Please ensure the model is trained.")

def test_specific_features():
    # Test features
    test_features = {
        'Gender': 'Male',
        'Married': 'Yes',
        'Dependents': '0',
        'Education': 'Not Graduate',
        'Self_Employed': 'No',
        'ApplicantIncome': 7660.0,
        'CoapplicantIncome': 0.0,
        'LoanAmount': 104.0,
        'Loan_Amount_Term': 360.0,
        'Credit_History': 0.0,
        'Property_Area': 'Urban'
    }

    try:
        # Load model
        model = load_model()
        
        # Convert features to DataFrame
        X = pd.DataFrame([test_features])
        
        # Get prediction using the pipeline
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]  # Probability of positive class
        
        # Print results
        print("\nTest Features:")
        print("-" * 50)
        for key, value in test_features.items():
            print(f"{key}: {value}")
        
        print("\nPrediction Results:")
        print("-" * 50)
        print(f"Prediction: {'Approved' if prediction == 1 else 'Rejected'}")
        print(f"Probability: {probability:.2%}")
        
    except FileNotFoundError as e:
        print(f"Error: {str(e)}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")

if __name__ == "__main__":
    test_specific_features() 