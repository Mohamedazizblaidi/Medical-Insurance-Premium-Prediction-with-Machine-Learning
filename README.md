# 🏥 Medical Insurance Premium Prediction with Machine Learning
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/Numpy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-013243?style=for-the-badge&logo=plotly&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-4C8CBF?style=for-the-badge&logo=plotly&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Joblib](https://img.shields.io/badge/Joblib-00A98F?style=for-the-badge)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)

## 📌 Overview
This project predicts **medical insurance premiums** based on personal and health-related data such as age, gender, BMI, smoking status, number of children, and region.  
It uses **Python, Pandas, Scikit-learn, and Machine Learning models** to train, evaluate, and make predictions.

---

## 📂 Project Structure
### Medical Insurance Premium Prediction Project Structure

```
├── Medical_Insurance_Premium_Prediction_with_Machine_Learning.ipynb  # Main notebook
├── insurance.csv                                                     # Dataset
├── insurance_model.h5                                               # Saved model (Keras/TensorFlow)
├── linear_regression_model.joblib                                   # Saved Linear Regression model
├── scaler_x.joblib                                                  # Feature scaler
├── scaler_y.joblib                                                  # Target scaler
└── README.md                                                        # Project documentation in markdown file
```

## File Descriptions

### 📊 **Data & Analysis**
- **`Medical_Insurance_Premium_Prediction_with_Machine_Learning.ipynb`**
  - Main Jupyter notebook containing data exploration, preprocessing, model training, and evaluation
  - Core analysis and machine learning pipeline

- **`insurance.csv`**
  - Primary dataset containing insurance premium data
  - Likely includes features such as age, BMI, smoking status, number of children, region, etc.

### 🤖 **Trained Models**
- **`insurance_model.h5`**
  - Deep learning model saved in Keras/TensorFlow format
  - Neural network architecture for premium prediction

- **`linear_regression_model.joblib`**
  - Traditional linear regression model
  - Baseline model for comparison with deep learning approach

### ⚙️ **Preprocessing Components**
- **`scaler_x.joblib`**
  - Feature scaler (StandardScaler/MinMaxScaler) for input variables
  - Essential for model deployment and consistent predictions

- **`scaler_y.joblib`**
  - Target variable scaler for premium amounts
  - Ensures proper scaling of prediction outputs

### 📝 **Documentation**
- **`README.md`**
  - Project documentation in markdown format
  - Setup instructions, usage guidelines, and project overview

## Project Highlights

✅ **Multiple ML Approaches**: Both traditional (Linear Regression) and modern (Deep Learning) methods  
✅ **Proper Preprocessing**: Saved scalers for consistent data transformation  
✅ **Model Persistence**: Trained models saved for future use and deployment  
✅ **Documentation**: README file for project explanation  
✅ **Clear Structure**: Well-organized file naming and project layout

## 🚀 Usage

### 1️⃣ Run the Notebook
Open `Medical_Insurance_Premium_Prediction_with_Machine_Learning.ipynb` in Jupyter Notebook or JupyterLab and execute all cells.

### 2️⃣ Save the Model
The model can be saved using:

```python
import joblib
joblib.dump(model, "linear_regression_model.joblib")
```

### 3️⃣ Load the Model

```python
import joblib
model = joblib.load("linear_regression_model.joblib")
```

### 4️⃣ Predict for New Samples

```python
import pandas as pd

sample_data = pd.DataFrame([
    {'age': 30, 'sex': 'male', 'bmi': 25.5, 'children': 1, 'smoker': 'no', 'region': 'southeast'}
])

sample_data_processed = preprocessor.transform(sample_data)
pred = model.predict(sample_data_processed)
print(f"Predicted Premium: ${pred[0]:.2f}")
```

## 📈 Models Used

* **Linear Regression** - Baseline model for premium prediction
* **TensorFlow/Keras Model** - Deep learning approach for complex pattern recognition
* **Feature scaling** with `StandardScaler` - Normalizes numerical features
* **Encoding categorical variables** with `OneHotEncoder` - Converts categorical data to numerical format

## 🛠 Future Improvements

* Implement more advanced ML models (Random Forest, XGBoost, LightGBM)
* Add hyperparameter tuning with GridSearchCV or RandomizedSearchCV
* Deploy as a web app with Flask or Streamlit
* Add model interpretability with SHAP or LIME
* Implement cross-validation for better model evaluation
* Create automated data pipeline for real-time predictions
---

## 📊 Dataset
- **Source**: `insurance.csv`
- **Features**:
  - `age` → Age of the individual
  - `sex` → Gender (male/female)
  - `bmi` → Body Mass Index
  - `children` → Number of dependents
  - `smoker` → Smoking status (yes/no)
  - `region` → Residential region (northeast, northwest, southeast, southwest)
  - `charges` → Medical insurance premium (target variable)

---

## ⚙️ Installation
Clone the repository:
```bash
git clone https://github.com/Mohamedazizblaidi/Medical_Insurance_Premium_Prediction.git
cd Medical_Insurance_Premium_Prediction
