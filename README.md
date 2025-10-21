# Logistic Regression using Newton's Method

This project implements **Logistic Regression** from scratch using **Newton's Method** for optimization — without relying on machine learning libraries like Scikit-learn.  
It demonstrates how gradient and Hessian-based optimization can be used to find model parameters efficiently for classification problems.

---

## Overview

Logistic Regression is a fundamental supervised learning algorithm used for binary classification.  
Instead of using standard gradient descent, this project uses **Newton-Raphson optimization**, which leverages second-order derivatives (the Hessian) to converge faster.

---

## Implementation Details

The implementation is divided into modular files:

- **`log_regress_newton.py`** → Main script that runs training and evaluation.
- **`linear_model.py`** → Contains the `LinearModel` and logistic regression model class definitions.
- **`utility.py`** → Helper functions for loading datasets, adding intercepts, and visualization.
- **`data/`** → Contains the training and validation CSV files.

---

## Dataset

The project uses two datasets:

- `ds1_train.csv` — Training dataset  
- `ds1_valid.csv` — Validation dataset  

Each dataset consists of input features and binary labels (0 or 1).  
You can easily replace them with your own dataset in CSV format.

---

This will:

* Load and preprocess the dataset
* Train the logistic regression model using Newton’s Method
* Display the decision boundary plot and training results

---

## Core Formula

Newton’s Method updates the parameters as:

[
\theta^{(t+1)} = \theta^{(t)} - H^{-1} \nabla_\theta J(\theta)
]

Where:

* ( \nabla_\theta J(\theta) ) is the gradient
* ( H ) is the Hessian matrix

---

## 📈 Visualization

The project includes visualization of:

* Data points and class boundaries
* Convergence behavior (if implemented)
* Decision boundary separating classes

---

## 🧰 Technologies Used

* **Python 3.x**
* **NumPy**
* **Matplotlib**
* (No ML libraries used — all math done manually)

---

## 💡 Future Improvements

* Extend to **multiclass logistic regression**
* Implement **regularization (L2 penalty)**
* Add **performance metrics** (accuracy, precision, recall)
* Implement **stochastic gradient descent** as comparison

---

## 🧑‍💻 Author

**Aviral Utkarsh**
Metallurgical Engineering 
Exploring AI/ML through mathematical foundations

---

## Acknowledgment

Inspired by **Andrew Ng’s ML lectures** and **CS229 materials**.
Developed to strengthen mathematical intuition and implementation skills.


