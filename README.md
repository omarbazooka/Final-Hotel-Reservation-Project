# Hotel Booking Cancellation Prediction

## 📌 Project Overview

This project predicts whether a hotel reservation will be **Canceled** or **Not Canceled** using Machine Learning.

The main goal is to build a complete ML pipeline that can process hotel reservation data, train multiple models, compare their performance, and evaluate the effect of Feature Selection on prediction quality.

The target column is:

```text
booking_status
```

Target mapping:

```text
Canceled      → 1
Not_Canceled  → 0
```

---

## 🎯 Project Objective

Hotel booking cancellations can affect revenue planning, room availability, and operational decisions.  
This project helps predict cancellation risk based on reservation details such as lead time, room type, meal plan, previous cancellations, number of guests, average room price, and special requests.

The project compares four ML pipelines:

1. **XGBoost with Feature Selection**
2. **LightGBM with Feature Selection**
3. **XGBoost without Feature Selection**
4. **LightGBM without Feature Selection**

This comparison helps answer an important question:

> Does Feature Selection improve the model performance?

---

## 📂 Dataset

The dataset used in this project is:

```text
Hotel Reservations.csv
```

In the notebook, the dataset is loaded from Google Drive:

```python
df = pd.read_csv("/content/drive/MyDrive/Hotel Reservations.csv")
```

### Main Columns Used

The project drops the following columns:

```text
Booking_ID
booking_status
```

`Booking_ID` is removed because it is only an identifier and does not help prediction.

`booking_status` is used as the target variable.

### Numerical Features

The notebook detects numerical columns automatically. In the used dataset, they include:

```text
no_of_adults
no_of_children
no_of_weekend_nights
no_of_week_nights
required_car_parking_space
lead_time
arrival_year
arrival_month
arrival_date
repeated_guest
no_of_previous_cancellations
no_of_previous_bookings_not_canceled
avg_price_per_room
no_of_special_requests
```

### Categorical Features

The categorical columns used are:

```text
type_of_meal_plan
room_type_reserved
market_segment_type
```

---

## 🛠️ Technologies Used

The project is implemented in Python using the following libraries:

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
lightgbm
google.colab
```

---

## ⚙️ Machine Learning Workflow

The notebook follows a structured ML workflow:

### 1. Import Libraries

The required libraries are imported, including:

- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- XGBoost
- LightGBM

A fixed random seed is used for reproducibility:

```python
SEED = 42
```

---

### 2. Load Dataset

The dataset is loaded from Google Drive:

```python
df = pd.read_csv("/content/drive/MyDrive/Hotel Reservations.csv")
```

The input features and target are prepared:

```python
X = df.drop(columns=["Booking_ID", "booking_status"], errors="ignore")

y = df["booking_status"].map({
    "Canceled": 1,
    "Not_Canceled": 0
}).astype("int64")
```

---

### 3. Train/Test Split

The dataset is split into training and testing sets:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=SEED
)
```

### Split Details

- 80% training data
- 20% testing data
- Stratified split to preserve the class distribution

---

## 🧹 Data Preprocessing

The project uses `ColumnTransformer` to handle numerical and categorical features separately.

### Numerical Pipeline

Numerical columns are processed using:

1. Missing value handling using median
2. Feature scaling using StandardScaler

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])
```

### Categorical Pipeline

Categorical columns are processed using:

1. Missing value handling using most frequent value
2. One-Hot Encoding

```python
Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
])
```

### Full Preprocessor

```python
preprocessor = ColumnTransformer([
    (
        "num",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]),
        num_cols,
    ),
    (
        "cat",
        Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]),
        cat_cols,
    ),
])
```

This prevents data leakage because preprocessing is fitted only on the training data inside the pipeline.

---

## 🎯 Feature Selection

Feature Selection is applied using `SelectFromModel`.

The selector is based on Logistic Regression with L1 regularization behavior:

```python
l1_selector = SelectFromModel(
    LogisticRegression(
        solver="saga",
        penalty="elasticnet",
        l1_ratio=1.0,
        C=0.5,
        random_state=SEED,
        max_iter=3000,
    ),
    threshold=-np.inf,
    max_features=28,
)
```

### Why Feature Selection?

Feature Selection is used to:

- Remove less important features
- Reduce noise
- Improve generalization
- Reduce overfitting
- Make training faster
- Test whether fewer features can still produce strong results

---

## 🤖 Models Used

Two gradient boosting models are used in this project.

---

### 1. XGBoost

XGBoost is a powerful boosting algorithm commonly used for structured/tabular data.

```python
xgb_best = XGBRegressor(
    n_estimators=350,
    max_depth=8,
    learning_rate=0.12028665880905137,
    subsample=0.8795975452591109,
    colsample_bytree=0.7468055921327309,
    random_state=SEED,
    n_jobs=-1,
    tree_method="hist",
    objective="binary:logistic",
)
```

Although the model object is `XGBRegressor`, the objective is set to `binary:logistic`, so the model outputs probability-like scores for binary classification.

---

### 2. LightGBM

LightGBM is a fast and efficient gradient boosting model that performs well on tabular datasets.

```python
lgbm_best = LGBMClassifier(
    n_estimators=501,
    max_depth=12,
    num_leaves=73,
    learning_rate=0.031194713369389154,
    random_state=SEED,
    n_jobs=-1,
)
```

---

## 🔗 Pipelines

The project creates four complete pipelines.

### 1. XGBoost with Feature Selection

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("selector", l1_selector),
    ("model", xgb_best)
])
```

### 2. LightGBM with Feature Selection

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("selector", l1_selector),
    ("model", lgbm_best)
])
```

### 3. XGBoost without Feature Selection

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("model", xgb_best)
])
```

### 4. LightGBM without Feature Selection

```python
Pipeline([
    ("preprocessor", preprocessor),
    ("model", lgbm_best)
])
```

---

## 📊 Evaluation Metrics

The models are evaluated using the following metrics:

| Metric | Meaning |
|---|---|
| Accuracy | Overall percentage of correct predictions |
| Precision | How many predicted cancellations were actually cancellations |
| Recall | How many actual cancellations were correctly detected |
| F1-Score | Balance between Precision and Recall |
| ROC-AUC | Ability of the model to distinguish between classes |

Predictions are converted into binary classes using a threshold of `0.5`:

```python
y_pred = (y_score >= 0.5).astype(int)
```

---

## ✅ Results

The following results were produced from the notebook:

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| XGB_with_FS | 0.9031 | 0.8737 | 0.8233 | 0.8477 | 0.9587 |
| LGBM_with_FS | 0.9023 | 0.8750 | 0.8187 | 0.8459 | 0.8808 |
| XGB | 0.9019 | 0.8698 | 0.8237 | 0.8462 | 0.9593 |
| LGBM | 0.9023 | 0.8740 | 0.8199 | 0.8461 | 0.8812 |

---

## 🏆 Best Model

Based on the results:

### Best Accuracy

```text
XGB_with_FS → 0.9031
```

### Best F1-Score

```text
XGB_with_FS → 0.8477
```

### Best ROC-AUC

```text
XGB → 0.9593
```

### Best Precision

```text
LGBM_with_FS → 0.8750
```

### Best Recall

```text
XGB → 0.8237
```

Overall, **XGB_with_FS** is selected in the notebook for visual evaluation because it achieved the highest Accuracy and F1-Score.

---

## 📈 Visualizations

The notebook includes three types of visualizations:

### 1. Accuracy Comparison

A bar chart comparing the accuracy of all models.

Purpose:

- Identify the model with the highest accuracy
- Compare models with and without Feature Selection

---

### 2. Metrics Comparison

A grouped bar chart comparing:

- Accuracy
- Precision
- Recall
- F1-Score

Purpose:

- Evaluate the overall balance of each model
- Understand trade-offs between Precision and Recall

---

### 3. Confusion Matrix and ROC Curve

The notebook visualizes the selected best model:

```python
best_model = "XGB_with_FS"
```

The visual evaluation includes:

- Confusion Matrix
- ROC Curve

Purpose:

- Understand the model’s classification errors
- Evaluate how well the model separates canceled from non-canceled bookings

---

## ▶️ How to Run the Project

### Option 1: Run on Google Colab

1. Upload the notebook to Google Colab.
2. Upload the dataset to your Google Drive.
3. Make sure the dataset path matches:

```text
/content/drive/MyDrive/Hotel Reservations.csv
```

4. Run the notebook cells from top to bottom.

---

### Option 2: Run Locally

1. Clone the repository:

```bash
git clone <your-repository-link>
cd <your-project-folder>
```

2. Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn xgboost lightgbm
```

3. Update the dataset path in the notebook:

```python
df = pd.read_csv("Hotel Reservations.csv")
```

4. Run the notebook using Jupyter:

```bash
jupyter notebook
```

---

## 📦 Suggested Project Structure

```text
Hotel-Booking-Cancellation-Prediction/
│
├── Final_Hotel_Reservation_Project.ipynb
├── Hotel Reservations.csv
├── README.md
└── requirements.txt
```

---

## 📄 Suggested requirements.txt

```text
numpy
pandas
matplotlib
scikit-learn
xgboost
lightgbm
jupyter
```

---

## 🔍 Key Insights

- XGBoost achieved the strongest overall performance.
- Feature Selection slightly improved Accuracy and F1-Score for XGBoost.
- XGBoost without Feature Selection achieved the highest ROC-AUC.
- LightGBM produced competitive Accuracy and Precision.
- The difference between models is small, which means both XGBoost and LightGBM are suitable for this dataset.
- Feature Selection can reduce model complexity while maintaining strong performance.

---

## 🚀 Future Improvements

Possible improvements for the project:

1. Use `XGBClassifier` instead of `XGBRegressor` for clearer classification implementation.
2. Add cross-validation for more reliable evaluation.
3. Save the best trained model using `joblib` or `pickle`.
4. Add feature importance visualization.
5. Add SHAP explainability to explain model predictions.
6. Build a simple web app using Streamlit or Flask.
7. Handle class imbalance using class weights or resampling techniques.
8. Add Optuna tuning code inside the notebook for full reproducibility.
9. Export predictions to a CSV file.
10. Add a user input form to predict cancellation for a new booking.

---

## 🧠 Conclusion

This project successfully builds and compares multiple Machine Learning pipelines for hotel booking cancellation prediction.

The best overall model in the notebook is **XGB_with_FS**, which achieved:

```text
Accuracy: 0.9031
F1-Score: 0.8477
ROC-AUC: 0.9587
```

The results show that gradient boosting models are effective for hotel reservation cancellation prediction, and Feature Selection can slightly improve model performance while reducing the number of features.

---

## 👨‍💻 Author

This project was developed as a Machine L
