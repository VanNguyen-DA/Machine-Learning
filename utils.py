import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
import lightgbm as lgbm
import math


def fill_missing_values(data):
    data_filled = data.copy()
    data_filled['qty'] = data_filled.groupby('sku')['qty'].transform(lambda x: x.fillna(x.mean()))
    data_filled['revenue'] = data_filled.groupby('sku')['revenue'].transform(lambda x: x.fillna(x.mean()))
    data_filled['COGS'] = data_filled.groupby('sku')['qty'].transform(lambda x: x.fillna(x.mean()))
    return data_filled

def correct_outliers(df, factor=3):
    """Identify and correct outliers in the 'sales' column by reducing them to the mean"""
    df_corrected = df.copy()

    # Identify outliers using z-score
    z_scores = (df_corrected["qty"] - df_corrected["qty"].mean()) / df_corrected[
        "qty"
    ].std()
    outlier_indices = np.abs(z_scores) > factor  # Adjust the threshold as needed
    # Correct outliers by reducing them to the mean
    df_corrected.loc[outlier_indices, "qty"] = df_corrected["qty"].mean()

    return df_corrected

def fill_date_template(data,min_date, max_date):
    date_range = pd.date_range(start=min_date, end=max_date)
    date_range['key'] = 1
    sku_template = pd.DataFrame({'sku': data['sku'].unique()})
    sku_template['key'] = 1
    template = pd.merge(sku_template, date_range, on='key').drop('key', axis=1)

    data_merge = pd.merge(template, data, on=['sku', 'shipped_date'], how='left')
    data_merge['qty'] = data_merge['qty'].fillna(0)
    data_merge['revenue'] = data_merge['revenue'].fillna(0)
    data_merge['COGS'] = data_merge['COGS'].fillna(0)
    
    return data_merge

def add_template(data, min_date, max_date, skus = None):
    data['shipped_date'] = pd.to_datetime(data['shipped_date'])
    date_template = pd.DataFrame({'shipped_date': pd.date_range(start=min_date, end=max_date)})
    date_template['key'] = 1
    sku_template = pd.DataFrame({'sku': skus})
    sku_template['key'] = 1
    template = pd.merge(date_template, sku_template, on='key').drop('key', axis=1)
    data_merge = pd.merge(template, data, on=['shipped_date', 'sku'], how='left')
    data_merge['qty'] = data_merge['qty'].fillna(0)
    data_merge['revenue'] = data_merge['revenue'].fillna(0)
    data_merge['COGS'] = data_merge['COGS'].fillna(0)
    return data_merge

def engineer_sku_features(data):
    data_test = data.copy()
    data_test["shipped_date"] = pd.to_datetime(data_test["shipped_date"])
    data_test = data_test.sort_values(["sku", "shipped_date"])

    # the average sales of each SKU on the same day of the week (e.g., average sales of SKU A on Mondays, Tuesdays, etc.)
    data_test["dow"] = data_test["shipped_date"].dt.weekday  # 0 = Monday

    data_test["mean_qty_sku_dow"] = (
        data_test.groupby(["sku", "dow"])["qty"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )

    #the average sales of each SKU in the same month (e.g., average sales of SKU A in January, February, etc.)
    data_test["month"] = data_test["shipped_date"].dt.month

    data_test["mean_qty_sku_month"] = (
        data_test.groupby(["sku", "month"])["qty"]
        .transform(lambda x: x.shift(1).expanding().mean())
    )  

    data_test['mean_qty_sku_dow'].fillna(data_test.groupby('sku')['qty'].mean(), inplace=True)
    data_test['mean_qty_sku_month'].fillna(data_test.groupby('sku')['qty'].transform(lambda x: x.shift(1).expanding().mean()), inplace=True)

    data_test['mean_qty_sku_dow'].fillna(0, inplace=True)
    data_test['mean_qty_sku_month'].fillna(0, inplace=True) 

    return data_test

def cal_ma_lag_features(data):

    data_final = data.copy()

    data_final['lag1'] = data_final.groupby('sku')['qty'].shift(1)
    data_final['lag7'] = data_final.groupby('sku')['qty'].shift(7)
    data_final['lag30'] = data_final.groupby('sku')['qty'].shift(30)
    data_final['rolling_mean_7'] = data_final.groupby('sku')['qty'].transform(lambda x: x.rolling(window=7,min_periods=1).mean())
    data_final['rolling_mean_30'] = data_final.groupby('sku')['qty'].transform(lambda x: x.rolling(window=30,min_periods=1).mean())   
    data_final['rolling_std_7'] = data_final.groupby('sku')['qty'].transform(lambda x: x.rolling(window=7,min_periods=1).std())
    data_final['rolling_std_30'] = data_final.groupby('sku')['qty'].transform(lambda x: x.rolling(window=30,min_periods=1).std())

    return data_final

# Function to calculate WAPE (Weighted Absolute Percentage Error)
def weighted_absolute_percentage_error(y_true, y_pred):
    """
    Calculate Weighted Absolute Percentage Error

    Args:
        y_true: Actual values
        y_pred: Predicted values

    Returns:
        WAPE value (percentage)
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return 100 * np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true))

def load_model(file_path):
    """
    Load a machine learning model from a file.

    Parameters:
    - file_path: The file path from where the model will be loaded.

    Returns:
    - The loaded model.
    """
    try:
        with open(file_path, "rb") as file:
            model = pickle.load(file)
            print(f"Sklearn model loaded from {file_path}")

    except (pickle.UnpicklingError, FileNotFoundError):
        # If loading as scikit-learn model fails or the file is not found,
        # assume it is a LightGBM model (scikit-learn API)
        model = lgbm.Booster(model_file=file_path)
        print(f"LightGBM (scikit-learn API) model loaded from {file_path}")

    return model

def plot_sku_forecast(
    data,
    sku_list,
    date_col="shipped_date",
    actual_col="qty",
    pred_col="prediction",
    sku_col="sku",
    n_cols=3,
    figsize=(18, 10),
):
    """
    Plot actual vs prediction for multiple SKUs.

    Args:
        data (pd.DataFrame): Input dataframe
        sku_list (list): List of SKUs to plot
        date_col (str): Date column name
        actual_col (str): Actual quantity column
        pred_col (str): Prediction column
        sku_col (str): SKU column
        n_cols (int): Number of subplot columns
        figsize (tuple): Figure size
    """

    # Filter selected SKUs
    plot_df = data[data[sku_col].isin(sku_list)].copy()

    # Convert date column
    plot_df[date_col] = pd.to_datetime(plot_df[date_col])

    n_skus = len(sku_list)
    n_rows = math.ceil(n_skus / n_cols)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=figsize,
        squeeze=False
    )

    axes = axes.flatten()

    for i, sku in enumerate(sku_list):

        sku_data = (
            plot_df[plot_df[sku_col] == sku]
            .sort_values(date_col)
        )

        ax = axes[i]

        # Actual line
        ax.plot(
            sku_data[date_col],
            sku_data[actual_col],
            label="Actual",
            linewidth=2
        )

        # Prediction line
        ax.plot(
            sku_data[date_col],
            sku_data[pred_col],
            linestyle="--",
            marker="o",
            markersize=3,
            label="Forecast"
        )

        ax.set_title(f"SKU: {sku}")
        ax.set_xlabel("Date")
        ax.set_ylabel("Qty")
        ax.legend()
        ax.grid(True)

        # Rotate x-axis labels
        ax.tick_params(axis="x", rotation=45)

    # Remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

