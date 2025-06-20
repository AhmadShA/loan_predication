# Credit ML System

A full-stack Django web application for managing and assessing individual loan/credit requests using machine learning.

---

## Features
- Upload and view loan/credit requests
- Model-based approval/rejection with manual override
- Add/remove requests (persistent in database)
- Exploratory Data Analysis (EDA) and model performance pages
- Modern, responsive Bootstrap UI

---

## Requirements
- Python 3.9+
- pip
- (Recommended) Virtual environment tool: `venv` or `virtualenv`

---

## Setup Instructions

### 1. **Clone the repository**
```
git clone <your-repo-url>
cd <project-folder>
```

### 2. **Create and activate a virtual environment**
```
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. **Install dependencies**
```
pip install -r requirements.txt
```

### 4. **Apply database migrations**
```
python manage.py makemigrations
python manage.py migrate
```

### 5. **Create a superuser (optional, for admin access)**
```
python manage.py createsuperuser
```

### 6. **Run the development server**
```
python manage.py runserver
```
Visit [http://127.0.0.1:8000/](http://127.0.0.1:8000/) in your browser.

---

## Machine Learning Model: Training & Integration

### 1. **Prepare your dataset**
- Place your CSV dataset (e.g., `train.csv`) somewhere accessible.
- The target column should be named `Loan_Status` (Y/N).

### 2. **Train the model**
```
python -m loan_manager.ml.train_model --data-file path/to/train.csv
```
- This will create `loan_manager/ml/model.joblib` and `metrics.json`.

### 3. **Using the model in the app**
- When you add a new request, the app will use the trained model to predict approval/rejection.
- Model performance metrics will appear on the Model Performance page.

---

## EDA & Visualization
- EDA and performance pages are ready for integration with Plotly or other Python visualization libraries.
- To add custom charts, update the corresponding Django views and templates.

---

## Troubleshooting
- If you see errors about missing model files, make sure you have trained the model as above.
- For static file issues, run `python manage.py collectstatic` (for production).
- For database issues, ensure migrations are applied.

---

## License
MIT (or specify your license)
