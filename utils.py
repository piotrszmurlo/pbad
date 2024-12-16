from math import ceil

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.nonparametric.smoothers_lowess import lowess

columns = [
    "engine_id", "time",
    "operational_setting_1",
    "operational_setting_2",
    "operational_setting_3",
    "Fan_inlet_temperature",
    "LPC_outlet_temperature",
    "HPC_outlet_temperature",
    "LPT_outlet_temperature",
    "Fan_inlet_Pressure",
    "bypass-duct_pressure",
    "HPC_outlet_pressure",
    "Physical_fan_speed",
    "Physical_core_speed",
    "Engine_pressure_ratio(",
    "HPC_outlet_Static_pressure",
    "Ratio_of_fuel_flow_to_Ps30",
    "Corrected_fan_speed",
    "Corrected_core_speed",
    "Bypass_Ratio",
    "Burner_fuel-air_ratio",
    "Bleed_Enthalpy",
    "Required_fan_speed",
    "Required_fan_conversion_speed",
    "High-pressure_turbines_Cool_air_flow",
    "Low-pressure_turbines_Cool_air_flow"
]


def load_data(csvname):
    df = pd.read_csv(csvname, sep=" ", header=None)
    df.drop(axis=1, columns=[26, 27], inplace=True)  # źle wczytane kolumny
    df.columns = columns
    return df


def draw_corr(df):
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)


def lowess_features(df, engine_id):
    df_copy = df.copy()
    filtered_df = df_copy[df_copy['engine_id'] == engine_id]
    filtered_df = filtered_df.copy()
    columns_to_smooth = [col for col in filtered_df.columns if col not in ['time', 'engine_id']]
    frac = 1.
    for col in columns_to_smooth:
        smoothed_values = []
        for _, group in filtered_df.groupby('engine_id'):
            smoothed = lowess(group[col], group['time'], frac=frac)[:, 1]
            smoothed_values.extend(smoothed)
        filtered_df[col + "_lowess"] = smoothed_values
    return filtered_df


def lowess_features_overwrite(df):
    df_copy = df.copy()
    columns_to_smooth = [col for col in df_copy.columns if col not in ['time', 'engine_id']]
    frac = 1.
    for col in columns_to_smooth:
        smoothed_values = []
        for _, group in df_copy.groupby('engine_id'):
            smoothed = lowess(group[col], group['time'], frac=frac)[:, 1]
            smoothed_values.extend(smoothed)
        df_copy[col] = smoothed_values
    return df_copy


def draw_time_series(df, engine_id):
    df = df.copy()
    columns_to_plot = [col for col in df.columns if col not in ['engine_id', 'time'] and "lowess" not in col]

    num_columns = len(columns_to_plot)
    fig, axes = plt.subplots(nrows=ceil(num_columns / 2), ncols=2, figsize=(24, 16), sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(columns_to_plot):
        axes[i].plot(df['time'], df[column], label=column)
        axes[i].plot(df['time'], df[column + "_lowess"], label=column + "_lowess")
        axes[i].grid(True)
        axes[i].legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Time series for engine_id {engine_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def draw_time_series_for_all_engines(df):
    columns_to_plot = [col for col in df.columns if col not in ['engine_id', 'time']]

    averaged_features = df.groupby('time')[columns_to_plot].mean()
    median_features = df.groupby('time')[columns_to_plot].median()

    fig, axes = plt.subplots(nrows=ceil(len(columns_to_plot) // 2), ncols=2, figsize=(24, 16), sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(columns_to_plot):
        axes[i].plot(averaged_features.index, averaged_features[column], label=f'Mean of smoothed data: {column}')
        axes[i].plot(median_features.index, median_features[column], label=f'Median of smoothed data: {column}')
        axes[i].set_title(f'Mean/media for smoothed data: {column}')
        axes[i].grid(True)
        axes[i].legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Mean/average of smoothed data for all engines", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def draw_pct_change(df_with_pctchange, engine_id):
    df_with_pctchange = df_with_pctchange[df_with_pctchange['engine_id'] == engine_id]
    columns_to_plot = [col for col in df_with_pctchange.columns if "pct_change" in col]
    num_columns = len(columns_to_plot)
    fig, axes = plt.subplots(nrows=num_columns // 2, ncols=2, figsize=(24, num_columns * 2), sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(columns_to_plot):
        axes[i].plot(df_with_pctchange['time'], df_with_pctchange[column], label=column)
        axes[i].grid(True)
        axes[i].legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"pct change for engine_id {engine_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def pct_change(df, periods):
    feature_cols = [col for col in df.columns if col not in ['engine_id', 'time']]
    df_with_pctchange = df.copy()
    for col in feature_cols:
        df_with_pctchange[f'{col}_pct_change'] = df_with_pctchange.groupby('engine_id')[col].pct_change(periods)
    df_with_pctchange.dropna(inplace=True)
    return df_with_pctchange


def evaluate(y_true, y_hat, label='test'):
    mse = mean_squared_error(y_true, y_hat)
    rmse = np.sqrt(mse)
    variance = r2_score(y_true, y_hat)
    print('{} set RMSE:{}, R2:{}'.format(label, rmse, variance))


def evaluate_mapie_output(result, y_true, max_rul, draw_size=100):
    plt.figure(figsize=(30, 5))
    y_pred, y_intervals = result
    y_pred_min = y_intervals[:, 0, :].clip(min=0, max=max_rul)
    y_pred_max = y_intervals[:, 1, :].clip(min=0, max=max_rul)
    x_vals = range(1, draw_size + 1)
    plt.plot(x_vals, y_true[:draw_size], color='green', label='RUL (Actual)', alpha=1.)
    plt.plot(x_vals, y_pred_min[:draw_size], color='blue', label='RUL min (Predicted)', alpha=1.)
    plt.plot(x_vals, y_pred_max[:draw_size], color='red', label='RUL max (Predicted)', alpha=0.3)
    plt.fill_between(x_vals, y_pred_min[:draw_size].flatten(), y_pred_max[:draw_size].flatten(), alpha=0.5)
    plt.xlabel("Sample")
    plt.ylabel("RUL")
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
