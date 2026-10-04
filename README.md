# 📦 Sales Forecasting for E-commerce Inventory Optimization

<img width="1600" height="1067" alt="image" src="https://github.com/user-attachments/assets/fd07dac7-fe39-41eb-b506-9a715337e3db" />


**Author:** Nguyễn Thị Thanh Vân  
**Tools:** Machine Learning, Python

---

# Table of Contents

- Background & Overview
- Objective
- Dataset Description & Data Structure
- Analysis Process
    + EDA & Data Processing
    + Feature Engineering
    + Train & Applying Model
    + Explaining Model
- Final Conclusion and Recommendations

---

# Background & Overview

Inventory management is one of the most critical challenges in e-commerce operations. Accurate demand forecasting enables businesses to maintain optimal stock levels, reduce operational costs, and improve customer satisfaction.

Currently, inventory decisions are heavily dependent on manual estimation and operational experience, which often leads to:

- Stock shortages for high-demand SKUs, resulting in lost revenue and poor customer experience.
- Excess inventory for slow-moving products, increasing warehousing and holding costs.
- Difficulties in procurement planning and product allocation across warehouses and sales channels.

To address these challenges, this project develops a **Machine Learning-based Sales Forecasting System** capable of predicting the daily sales quantity (**Qty Sold**) for each SKU.

Additionally, **SHAP** is used to explain model predictions and provide transparent business insights for stakeholders.
---

# Objective

The primary objective of this project is to build a forecasting model that predicts the daily quantity sold (**Qty**) for each SKU.

### Business Goals

- Forecast daily sales quantity at SKU level.
- Improve inventory planning and replenishment decisions.
- Reduce stockout risks for fast-moving products.
- Minimize excess inventory and storage costs.
- Provide explainable predictions through SHAP.

---

# What Business Questions Will This Project Solve?

### Demand Forecasting

- How many units of a SKU are expected to be sold tomorrow?
- What is the expected demand trend for each product?

### Inventory Optimization

- Which products require replenishment?
- Which products are likely to experience stock shortages?
- Which products have excessive inventory compared to expected demand?

### Procurement Planning

- How much inventory should be ordered?
- What reorder quantity is appropriate for each SKU?

### Business Insights

- Which factors drive product demand?
- How do holidays, weekdays, seasonality, and sales channels affect sales performance?

---

# Who Is This Project For?

| Stakeholder | Business Value |
|------------|---------------|
| Inventory Planner | Optimize stock levels and replenishment planning |
| Supply Chain Team | Improve procurement decisions |
| Operation Team | Reduce stockout and overstock situations |
| Category Managers | Monitor demand trends at SKU level |
| Business Analysts | Understand demand drivers through SHAP |
| Management Team | Support data-driven decision making |

---

# Dataset Description & Data Structure

The dataset contains transactional sales information from an e-commerce business during 2021.

Each row represents sales activity of a SKU on a specific date and sales channel.

## Data_Processed

| Feature | Description |
|----------|-------------|
| shipped_date | Transaction date |
| sku | Product ID |
| channel | Sales channel |
| qty | Quantity sold |
| revenue | Revenue generated |
| COGS | Cost of Goods Sold |
| MOQ_orders | Minimum Order Quantity |

## Target Variable

| Variable | Description |
|----------|-------------|
| qty | Daily quantity sold per SKU |

---

# Analysis Process

---

# 1. EDA & Data Processing

The first step focused on understanding data quality, identifying patterns, and preparing the dataset for modeling.

### Main Activities

## Import Necessary Libraries

<img width="1348" height="1288" alt="image" src="https://github.com/user-attachments/assets/954295f2-daa5-4877-9997-2952c48b7a77" />

## Overview Data


## Missing value analysis

- Missing value imputation
- Outlier correction
- Correlation analysis
- Demand distribution analysis
- Holiday impact analysis
- Seasonality and trend analysis
- Pareto analysis to identify top revenue-generating SKUs

---

# 2. Feature Engineering

To improve prediction performance, multiple time-based and business-driven features were created.

## Temporal Features

- Month
- Day
- Day of Week
- Week of Year
- Quarter
- Weekend Indicator
- Even/Odd Day Indicator

## Holiday Features

- New Year
- Lunar New Year (Tet Holiday)
- National Holidays
- Christmas

## SKU Behavioral Features

- Average sales by weekday
- Average sales by month

## Time-Series Features

### Lag Features

- Lag 1
- Lag 7
- Lag 30

### Rolling Statistics

- Rolling Mean (7 Days)
- Rolling Mean (30 Days)
- Rolling Standard Deviation (7 Days)
- Rolling Standard Deviation (30 Days)

## Target Transformation

To reduce skewness and stabilize variance:

```python
qty_log = np.log1p(qty)
```

---

# 3. Train & Applying Model

Several machine learning approaches were evaluated.

## Baseline Model

### Random Forest Regressor

Performance:

| Metric | Result |
|----------|----------|
| R² | 0.607 |
| MAE | 107.85 |
| RMSE | 208.78 |
| WMAPE | 46.42% |

---

## Final Model

### LightGBM Regressor

Reasons for selection:

- Strong performance on tabular data
- Handles non-linear relationships effectively
- Supports categorical variables
- Fast training and inference
- Highly scalable

## Validation Strategy

A **TimeSeriesSplit** approach was applied to preserve temporal order and prevent data leakage.

## Hyperparameter Optimization

Optuna was used to optimize:

- num_leaves
- learning_rate
- feature_fraction
- bagging_fraction
- bagging_freq
- min_child_samples

## Final Model Performance

| Metric | Result |
|----------|----------|
| MAE | 0.54 |
| RMSE | 0.76 |
| WMAPE | 11.55% |

The optimized LightGBM model significantly outperformed the Random Forest baseline.

---

# 4. Explaining Model

Model interpretability is critical for business adoption.

To explain the forecasting results, SHAP (SHapley Additive Explanations) was applied.

## Global Explanation

Top features impacting sales forecasts:

| Rank | Feature |
|--------|---------|
| 1 | rolling_mean_7 |
| 2 | channel |
| 3 | mean_qty_sku_month |
| 4 | rolling_std_7 |
| 5 | rolling_mean_30 |
| 6 | lag1 |
| 7 | dayofweek |
| 8 | weekofyear |
| 9 | MOQ_orders |
| 10 | day |

### Insights

- Historical demand patterns are the strongest predictors.
- Sales channels influence SKU demand.
- Seasonality and calendar effects contribute to prediction performance.
- Recent sales trends significantly improve forecasting accuracy.

## Local Explanation

SHAP was also used to explain individual predictions:

- Why demand is expected to increase.
- Why demand is expected to decrease.
- Which features contribute most to each forecast.

This provides transparency and improves business trust in the forecasting system.

---

# Final Conclusion

This project successfully developed an end-to-end Machine Learning solution for forecasting daily SKU demand in an e-commerce environment.

By combining data preprocessing, feature engineering, LightGBM modeling, and SHAP explainability, the forecasting system achieved strong predictive performance while remaining interpretable for business stakeholders.

The findings indicate that historical demand behavior, sales channels, and seasonality are the primary drivers of future sales performance. The resulting model can support inventory optimization, procurement planning, and supply chain decision-making.

### Key Achievements

- ✅ Built a daily SKU-level sales forecasting system.
- ✅ Reduced forecasting error significantly using LightGBM.
- ✅ Identified key business drivers using SHAP.
- ✅ Developed an explainable and scalable prediction framework.
- ✅ Generated actionable recommendations for inventory planning.

---

# Business Recommendations

| Aspect | Insight | Recommendation |
|----------|----------|----------|
| Forecasting Performance | The optimized LightGBM model achieved approximately **11.55% WMAPE**, substantially outperforming the Random Forest baseline. | Deploy LightGBM as the production forecasting model for SKU-level demand prediction. |
| Demand Drivers | Historical demand indicators (rolling averages, lag features) contribute the most to forecasting accuracy. | Continuously maintain and monitor historical sales data quality. |
| Inventory Management | A small percentage of SKUs contributes the majority of revenue (Pareto Principle). | Prioritize replenishment planning and safety stock for top-performing SKUs. |
| Seasonality | Demand varies significantly by weekday, month, and holiday periods. | Incorporate seasonal inventory strategies and prepare stock before peak demand periods. |
| Stockout Prevention | High-demand products are vulnerable to stock shortages if demand is underestimated. | Use forecasts to establish reorder thresholds and stockout warning systems. |
| Overstock Reduction | Slow-moving products increase inventory holding costs. | Align purchasing decisions with forecasted demand to reduce excess inventory. |
| Explainability | SHAP provides transparent explanations behind every prediction. | Integrate SHAP visualizations into business dashboards to increase model trust. |
| Data Quality | Missing values and abnormal transactions can negatively impact model performance. | Build automated validation and preprocessing pipelines for incoming data. |
| Model Monitoring | Customer behavior and demand patterns evolve over time. | Retrain and evaluate the model periodically using the latest available data. |
| Future Improvements | Current forecasts rely mainly on historical sales and calendar-related features. | Add promotion data, pricing information, marketing campaigns, competitor data, and inventory availability to further improve forecasting accuracy. |

---

## Key Takeaways

- Forecasting demand at the SKU level can significantly improve inventory planning.
- Historical sales trends remain the strongest predictor of future demand.
- Explainable AI (SHAP) bridges the gap between machine learning and business decision-making.
- Data-driven replenishment strategies can reduce stockouts and excess inventory simultaneously.

**Business Impact:**  
This solution enables better procurement planning, inventory optimization, and operational efficiency while supporting data-driven decision making across the e-commerce supply chain.
