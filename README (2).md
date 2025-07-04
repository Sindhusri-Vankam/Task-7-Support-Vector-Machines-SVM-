# Task 7: Support Vector Machines (SVM)

## ✅ Objective
Use Support Vector Machines (SVMs) for linear and non-linear classification using the Breast Cancer dataset.

## 📦 Tools & Libraries Used
- Python
- Scikit-learn
- NumPy
- Matplotlib
- PCA (for visualization)

## 🧠 Concepts Covered
- Margin maximization
- Kernel trick
- Hyperparameter tuning
- Cross-validation

## 📊 Dataset
**Breast Cancer Wisconsin Dataset** — available directly via `sklearn.datasets`.

## 🧪 Steps Performed

### 1. Data Preparation
- Loaded the breast cancer dataset
- Standardized features using `StandardScaler`
- Split into training and testing sets

### 2. SVM Models
- Trained two models:
  - SVM with **Linear Kernel**
  - SVM with **RBF Kernel**
- Evaluated performance using test accuracy

### 3. Visualization
- Used **PCA** to reduce features to 2D for visualization
- Plotted class separation visually

### 4. Hyperparameter Tuning
- Performed **GridSearchCV** for tuning:
  - `C` (regularization parameter)
  - `gamma` (for RBF kernel)
- Found best parameters using 5-fold cross-validation

### 5. Evaluation
- Final evaluation using cross-validation on best model
- Reported average cross-validation accuracy

## 📈 Results

| Model         | Accuracy |
|---------------|----------|
| SVM (Linear)  | 95% |
| SVM (RBF)     | 97% |
| Grid Search CV Best Score | 97% |

---

## 💬 Interview Questions

1. **What is a support vector?**  
   A data point closest to the decision boundary (hyperplane), which helps define the margin.

2. **What does the C parameter do?**  
   It controls the trade-off between achieving a low training error and a low testing error (margin width vs misclassification).

3. **What are kernels in SVM?**  
   Kernels are functions that project input features into higher-dimensional space to make them linearly separable.

4. **Difference between Linear and RBF kernel?**  
   - **Linear**: Straight-line decision boundary.
   - **RBF**: Curved decision boundary using Gaussian function.

5. **Advantages of SVM?**  
   - Works well in high-dimensional spaces  
   - Effective when number of features > number of samples  
   - Memory efficient (uses support vectors only)

6. **Can SVMs be used for regression?**  
   Yes, using **SVR** (Support Vector Regression).

7. **What happens when data is not linearly separable?**  
   Use non-linear kernels (e.g., RBF) or apply soft margin with regularization.

8. **How is overfitting handled in SVM?**  
   Through tuning of hyperparameters like `C` and `gamma`, and using cross-validation.

---

## 📁 Files Included
- `svm_task.py`: Python script for entire workflow
- `README.md`: This file

