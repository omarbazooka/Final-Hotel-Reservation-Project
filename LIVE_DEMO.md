# StaySure — Live Demo

**Production demo:** https://hotel-booking-cancellation-demo.vercel.app

This repository now includes a lightweight presentation frontend for the Hotel Booking Cancellation Prediction project.

## Demo flow

1. Open the live demo.
2. Use **Load low-risk example** or **Load high-risk example** for a quick presentation.
3. Change reservation inputs and click **Predict cancellation risk**.
4. Review the risk gauge and the main risk drivers.
5. Scroll to **Original model performance** to show the notebook results and compare the four ML pipelines.

## Important implementation note

The interactive browser estimator is intentionally lightweight and runs without a Python backend so the presentation is fast and reliable on Vercel. It uses the same booking input schema and directional risk factors for demonstration. The validated research metrics shown on the page come directly from the original notebook, where the selected pipeline is **XGBoost + Feature Selection**.

## Original notebook highlights

- Best Accuracy: **90.31%** — XGBoost + Feature Selection
- Best F1-score: **84.77%** — XGBoost + Feature Selection
- XGB + FS ROC-AUC: **95.87%**
- Feature selection: L1-based Logistic Regression, top 28 transformed features
- Train/Test split: 80/20 with stratification and random seed 42
