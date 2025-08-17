# Linear & Logistic Regression From Scratch (Python + NumPy + Matplotlib)
This project demonstrates **Linear Regression** and **Logistic Regression** built **completely from scratch** using `numpy` and `matplotlib` (without using sklearn’s regression models).  
Both models are trained, evaluated, and visualized with **animated plots, metrics, and loss curves**.

---

## Quick Features to note

- **Implemented from scratch**: No reliance on sklearn models, everything is manually coded (forward pass, gradient descent, weight updates).  
- **Rich visualizations**: Training animations, error metric plots, and logistic decision boundaries.  
- **Clean architecture & documentation**: Classes, methods, type hints, and docstrings.  
- **Strong performance**: Logistic regression hits ~99% accuracy.  
- **Engineer’s polish**: CLI options, reproducible results, organized repo structure.  

---

## Features

✅ Linear Regression:
- Gradient Descent implementation  
- Forward & Backpropagation  
- Animated training with regression line updates (`.gif`)  
- Error metrics: **MSE, RMSE, MAE, R²**  

✅ Logistic Regression:
- Binary classification using sigmoid activation  
- Gradient Descent optimization  
- Accuracy & Precision metrics  
- Visual comparison of train vs test predictions  

---

## Method Inventory

### `LinearRegressionGD` Class

| Method | Signature | Purpose | Returns | Complexity |
|--------|-----------|---------|---------|------------|
| `__init__` | `(lr: float = 0.01, n_iters: int = 1000)` | Initialize learning rate & iterations | None | O(1) |
| `fit` | `(X: np.ndarray, y: np.ndarray)` | Train model using gradient descent | None | O(n * d * iters) |
| `predict` | `(X: np.ndarray) -> np.ndarray` | Predict values for input features | Predicted vector | O(n * d) |
| `mse` | `(y_true, y_pred) -> float` | Compute Mean Squared Error | Float | O(n) |
| `mae` | `(y_true, y_pred) -> float` | Compute Mean Absolute Error | Float | O(n) |
| `rmse` | `(y_true, y_pred) -> float` | Root Mean Squared Error | Float | O(n) |
| `r2_score` | `(y_true, y_pred) -> float` | Compute R² score | Float | O(n) |

---

### 🔹 `LogisticRegressionGD` Class

| Method | Signature | Purpose | Returns | Complexity |
|--------|-----------|---------|---------|------------|
| `__init__` | `(lr: float = 0.01, n_iters: int = 1000)` | Initialize logistic regression | None | O(1) |
| `fit` | `(X: np.ndarray, y: np.ndarray)` | Train model with gradient descent | None | O(n * d * iters) |
| `predict_proba` | `(X: np.ndarray) -> np.ndarray` | Predict probability scores using sigmoid | Probabilities | O(n * d) |
| `predict` | `(X: np.ndarray) -> np.ndarray` | Classify into 0/1 | Labels | O(n * d) |
| `accuracy` | `(y_true, y_pred) -> float` | Compute accuracy score | Float | O(n) |
| `precision` | `(y_true, y_pred) -> float` | Compute precision | Float | O(n) |
| `_sigmoid` | `(z: np.ndarray) -> np.ndarray` | Apply sigmoid function | Vector | O(n) |

---

