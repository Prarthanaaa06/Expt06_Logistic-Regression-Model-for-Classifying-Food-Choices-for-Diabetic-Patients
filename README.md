# BLENDED_LEARNING
# Implementation of Logistic Regression Model for Classifying Food Choices for Diabetic Patients

## AIM:
To implement a logistic regression model to classify food items for diabetic patients based on nutrition information.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1. **Start**

2. **Import Libraries**

   * Import required libraries:

     * `pandas`, `numpy`
     * `train_test_split`
     * `LogisticRegression`
     * `LabelEncoder`, `MinMaxScaler`
     * Evaluation metrics (`accuracy`, `precision`, `recall`, `F1-score`, `confusion_matrix`, `classification_report`)
     * `seaborn`, `matplotlib`

3. **Load Dataset**

   * Read dataset `food_items (1).csv` into a DataFrame
   * Display:

     * First few rows (`head()`)
     * Dataset information (`info()`)

4. **Separate Features and Target**

   * Define input features `X_raw` (all columns except last column)
   * Define target variable `y_raw` (last column)

5. **Feature Scaling**

   * Initialize `MinMaxScaler`
   * Scale feature values to range [0,1]
   * Store result in `X`

6. **Encode Target Variable**

   * Initialize `LabelEncoder`
   * Convert categorical target values into numerical form
   * Store result in `y`

7. **Split Dataset**

   * Split data into:

     * Training set (80%)
     * Testing set (20%)
   * Use `stratify=y` to maintain class distribution
   * Set `random_state = 123`

---

### **Model Training**

8. **Set Model Parameters**

   * Penalty: `l2`
   * Multi-class: `multinomial`
   * Solver: `lbfgs`
   * Maximum iterations: `1000`

9. **Create Logistic Regression Model**

   * Initialize `LogisticRegression` with above parameters

10. **Train Model**

* Fit model using training data (`X_train`, `y_train`)

---

### **Prediction**

11. **Make Predictions**

* Predict class labels for test data (`X_test`)
* Store predictions in `y_pred`

---

### **Model Evaluation**

12. **Calculate Accuracy**

* Compute accuracy score

13. **Generate Classification Report**

* Display:

  * Precision
  * Recall
  * F1-score

14. **Compute Confusion Matrix**

* Generate confusion matrix for predicted vs actual values
* Display matrix

15. **End**

---


## Program:
```
/*
Program to implement Logistic Regression for classifying food choices based on nutritional information.
Developed by: PRARTHANA D
RegisterNumber:  21225230213

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix,classification_report
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('food_items (1).csv')

print("Name: PRARTHANA D")
print("Reg. No: 21225230213")
print("Dataset Overview:")
print(df.head())
print("\nDataset Info:")
print(df.info())

X_raw = df.iloc[:,:-1]
y_raw = df.iloc[:,-1:]

scaler = MinMaxScaler()
X = scaler.fit_transform(X_raw)

laben = LabelEncoder()
y = laben.fit_transform(y_raw.values.ravel())

X_train,X_test,y_train,y_test= train_test_split(X,y,test_size=0.2,stratify=y, random_state=123)

penalty = 'l2'

multi_class = 'multinomial'

solver = 'lbfgs'

max_iter = 1000

l2_model = LogisticRegression(random_state=123,penalty=penalty, multi_class=multi_class,solver=solver,max_iter=max_iter)
l2_model.fit(X_train,y_train)
y_pred = l2_model.predict(X_test)

print("Name: PRARTHANA D")
print("Reg. No: 212225230213")
print("\nModel Evaluvation:")
print("Accuracy:", accuracy_score(y_test,y_pred))
print("\nClassification Report:")
print(classification_report(y_test,y_pred))

conf_matrix = confusion_matrix(y_test,y_pred)
print(conf_matrix)
*/
```

## Output:

<img width="803" height="833" alt="image" src="https://github.com/user-attachments/assets/eb80a610-b3ee-4178-82d6-23f2c39d97a7" />

<img width="1313" height="446" alt="image" src="https://github.com/user-attachments/assets/13318a48-520f-4f89-ac06-56be92f72c2a" />


## Result:
Thus, the logistic regression model was successfully implemented to classify food items for diabetic patients based on nutritional information, and the model's performance was evaluated using various performance metrics such as accuracy, precision, and recall.
