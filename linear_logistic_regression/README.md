# Linear & Logistic Regression From Scratch (Python + NumPy + Matplotlib)

This project demonstrates **Linear Regression** and **Logistic Regression** built **completely from scratch** using `numpy` and `matplotlib` (without using sklearn’s regression models).  
Both models are trained, evaluated, and visualized with **animated plots, metrics, and loss curves**.

---

## Quick Features to Note

- **Implemented from scratch**: Manual forward pass, gradient descent, weight updates.  
- **Rich visualizations**: Training animations, error metric plots, and logistic decision boundaries.  
- **Clean architecture & documentation**: Classes, methods, type hints, and docstrings.  
- **Strong performance**: Logistic regression reaches ~99% accuracy.  
- **Engineer’s polish**: CLI options, reproducible results, organized repo structure.  

---

## Features

✅ **Linear Regression**  
- Gradient Descent implementation  
- Forward & Backpropagation  
- Animated training with regression line updates (`.gif`)  
- Error metrics: **MSE, RMSE, MAE, R²**  

✅ **Logistic Regression**  
- Binary classification using sigmoid activation  
- Gradient Descent optimization  
- Accuracy & Precision metrics  
- Visual comparison of train vs test predictions  

---

## Method Inventory & Technical Showcase

Below is a comprehensive breakdown of the implemented classes, methods, and their mathematical underpinnings.

---

## 🔹 Linear Regression (Gradient Descent)

### **Mathematical Formulation**

- **Hypothesis (Prediction Function):**

  $$
  \hat{y} = Xw + b
  $$

- **Cost Function (Mean Squared Error):**

  $$
  J(w, b) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
  $$

- **Gradient Updates:**

  $$
  w := w - \eta \cdot \frac{1}{n} X^T (Xw - y), \quad 
  b := b - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} (Xw - y)
  $$

---

### **Method Inventory – LinearRegressionGD**

| Method | Signature | Description | Returns | Notes |
|--------|-----------|-------------|---------|-------|
| `__init__` | `(lr: float = 0.01, n_iters: int = 1000)` | Initialize learning rate and iterations | None | Hyperparameters only |
| `fit` | `(X: np.ndarray, y: np.ndarray)` | Perform gradient descent updates on weights and bias | None | Core training loop |
| `predict` | `(X: np.ndarray) -> np.ndarray` | Predict regression outputs | `np.ndarray` | Uses learned weights |
| `mse` | `(y_true, y_pred) -> float` | Mean Squared Error | `float` | Cost function |
| `mae` | `(y_true, y_pred) -> float` | Mean Absolute Error | `float` | Robust to outliers |
| `rmse` | `(y_true, y_pred) -> float` | Root Mean Squared Error | `float` | Penalizes large errors |
| `r2_score` | `(y_true, y_pred) -> float` | Coefficient of Determination (R²) | `float` | Goodness of fit |

---

### **Training Results – Linear Regression**

| Metric | Training | Testing |
|--------|----------|---------|
| **Loss (MSE)** | 8.716 | 6.614 |
| **RMSE** | 2.95 | 2.57 |
| **MAE** | 2.70 | 2.35 |
| **R² Score** | 0.89 | 0.91 |

---

### **Linear Regression Outputs**

- **Animated Fit Progression (Gradient Descent Updates):**

  ![Linear_Regression](https://github.com/user-attachments/assets/331db847-bcc8-4d49-a85c-244010cf56ee)

- **Training vs Testing Predictions:**

  <img width="1000" height="600" alt="Linear_Regression_Errors" src="https://github.com/user-attachments/assets/5a6edc6f-7be7-4a5c-819e-c01bc6330850" />
  
- **Training Loss Curve:**

  <img width="1920" height="967" alt="Linear_Regression_Losses" src="https://github.com/user-attachments/assets/f3038a98-575a-44e5-b71b-4d6a840fa415" />

---

## Logistic Regression (Gradient Descent)

### **Mathematical Formulation**

- **Hypothesis (Sigmoid Function):**

  $$
  \hat{y} = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad z = Xw + b
  $$

- **Cost Function (Binary Cross-Entropy):**

  $$
  J(w, b) = -\frac{1}{n} \sum_{i=1}^{n} \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
  $$

- **Gradient Updates:**

  $$
  w := w - \eta \cdot \frac{1}{n} X^T (\hat{y} - y), \quad 
  b := b - \eta \cdot \frac{1}{n} \sum_{i=1}^{n} (\hat{y} - y)
  $$

---

### **Method Inventory – LogisticRegressionGD**

| Method | Signature | Description | Returns | Notes |
|--------|-----------|-------------|---------|-------|
| `__init__` | `(lr: float = 0.01, n_iters: int = 1000)` | Initialize logistic regression | None | Hyperparameters only |
| `fit` | `(X: np.ndarray, y: np.ndarray)` | Perform gradient descent with BCE loss | None | Core training loop |
| `predict_proba` | `(X: np.ndarray) -> np.ndarray` | Predict probabilities (sigmoid outputs) | `np.ndarray` | Returns [0,1] values |
| `predict` | `(X: np.ndarray) -> np.ndarray` | Convert probabilities to 0/1 labels | `np.ndarray` | Threshold = 0.5 |
| `accuracy` | `(y_true, y_pred) -> float` | Accuracy score | `float` | % of correct predictions |
| `precision` | `(y_true, y_pred) -> float` | Precision score | `float` | Handles imbalanced data |
| `_sigmoid` | `(z: np.ndarray) -> np.ndarray` | Apply sigmoid function | `np.ndarray` | Internal helper |

---

### **Training Results – Logistic Regression**

| Metric | Training | Testing |
|--------|----------|---------|
| **Loss (BCE)** | 0.056 | 0.062 |
| **Accuracy** | 99.0% | 98.5% |
| **Precision** | 98.3% | 97.9% |

---

### **Logistic Regression Outputs**

- **Train Set Predictions:**

  <img width="1000" alt="Logistic_Regression_Train" src="https://github.com/user-attachments/assets/fab8d835-97a2-42be-82dd-20cf10ff9bdc" />

- **Training Loss Curve:**

  <img width="1000" alt="Logistic_Loss_Curve" src="https://github.com/user-attachments/assets/69868fc6-4a32-4691-8f44-f76c649e1a58" />

- **Test Set Predictions:**

  <img width="1000" alt="Logistic_Regression_Test" src="https://github.com/user-attachments/assets/dba493ad-2c49-4393-8733-6fdb72c36c07" />

---

## 📊 Summary Comparison

| Model | Loss Function | Final Loss | Accuracy | R² Score | Complexity | Notes |
|-------|--------------|------------|----------|-----------|------------|-------|
| **Linear Regression** | MSE | 6.614 (test) | – | 0.91 (test) | $O(n \cdot d \cdot iter)$ | Excellent regression fit |
| **Logistic Regression** | BCE | 0.062 (test) | 98.5% | – | $O(n \cdot d \cdot iter)$ | Near-perfect classification |

---

## Technical Summary

This project illustrates the **end-to-end workflow of building learning algorithms from first principles**:

- **Mathematical Foundations → Implementation → Evaluation → Visualization**  
  - Each model directly encodes its cost function, gradients, and parameter updates using `numpy`.  
  - No reliance on external ML libraries for optimization, ensuring transparency and correctness.  

- **Evaluation Framework**  
  - Multiple error metrics for regression (MSE, MAE, RMSE, R²) provide a holistic view of fit quality.  
  - Classification metrics (Accuracy, Precision) quantify logistic regression’s predictive power.  

- **Computational Complexity**  
  - Both models train in $O(n \cdot d \cdot iter)$ where *n* = samples, *d* = features, *iter* = iterations.  
  - Clear trade-offs between accuracy, convergence speed, and learning rate are demonstrated.  

- **Visualization & Diagnostics**  
  - Training dynamics captured via animated regression lines and loss curves.  
  - Clear diagnostic plots help verify convergence and generalization behavior.  

- **Extensibility**  
  - Current design allows straightforward extension to:  
    - Regularization terms (L1/L2 penalty in cost function)  
    - Multi-class logistic regression (softmax, cross-entropy)  
    - Mini-batch or stochastic gradient descent variants  
    - Unit testing and CI for reproducibility  

---
