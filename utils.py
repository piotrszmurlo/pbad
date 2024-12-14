from math import ceil

import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

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


def rolling_average_features(df, engine_id, window_size):
    df_copy = df.copy()
    columns_to_smooth = [col for col in df_copy.columns if col not in ['time', 'engine_id']]  # Exclude 'time' column
    filtered_df = df_copy[df_copy['engine_id'] == engine_id]
    df_averaged = filtered_df.copy()
    for col in columns_to_smooth:
        df_averaged[col + " rolling"] = filtered_df[col].rolling(window=window_size, min_periods=1).mean()
    return df_averaged


def draw_time_series(df, engine_id, window_size):
    df_averaged = rolling_average_features(df, engine_id, window_size)
    print(df_averaged.columns)
    columns_to_plot = [col for col in df_averaged.columns if col not in ['engine_id', 'time'] and "rolling" not in col]

    num_columns = len(columns_to_plot)
    fig, axes = plt.subplots(nrows=ceil(num_columns / 2), ncols=2, figsize=(24, 16), sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(columns_to_plot):
        axes[i].plot(df_averaged['time'], df_averaged[column], label=column)
        axes[i].plot(df_averaged['time'], df_averaged[column + " rolling"], label=column + " rolling average")
        axes[i].grid(True)
        axes[i].legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Time series for engine_id {engine_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def rolling_average_features_all(df, window_size):
    df_copy = df.copy()
    columns_to_smooth = [col for col in df_copy.columns if col not in ['time', 'engine_id']]

    for col in columns_to_smooth:
        df_copy[col] = (df_copy.groupby('engine_id')[col]
                        .rolling(window=window_size, min_periods=1)
                        .mean()
                        .reset_index(level=0, drop=True))
    return df_copy, columns_to_smooth


def draw_time_series_for_all_engines(df, window_size):
    df_copy = df.copy()
    df_copy, columns_to_smooth = rolling_average_features_all(df_copy, window_size)
    rolling_average_features_all(df_copy, window_size)

    averaged_rolling = df_copy.groupby('time')[columns_to_smooth].mean()
    median_rolling = df_copy.groupby('time')[columns_to_smooth].median()

    fig, axes = plt.subplots(nrows=ceil(len(columns_to_smooth) / 2), ncols=2, figsize=(24, 16), sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(columns_to_smooth):
        axes[i].plot(averaged_rolling.index, averaged_rolling[column], label=f'Average Rolling mean {column}')
        axes[i].plot(median_rolling.index, median_rolling[column], label=f'Average Rolling median {column}')
        axes[i].set_title(f'Average Rolling mean/median for {column}')
        axes[i].grid(True)
        axes[i].legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"Average Rolling mean/average for all Engines", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def draw_pct_change(df_with_pctchange, engine_id):
    df_with_pctchange = df_with_pctchange[df_with_pctchange['engine_id'] == engine_id]
    columns_to_plot = [col for col in df_with_pctchange.columns if "pct_change" in col]
    num_columns = len(columns_to_plot)
    fig, axes = plt.subplots(nrows=num_columns // 2 + 1, ncols=2, figsize=(24, num_columns * 2), sharex=True)
    axes = axes.flatten()
    fig.subplots_adjust(hspace=0.5)

    for i, column in enumerate(columns_to_plot):
        axes[i].plot(df_with_pctchange['rul'], df_with_pctchange[column], label=column)
        axes[i].grid(True)
        axes[i].legend(loc="upper left", fontsize=8)

    axes[-1].set_xlabel("Time")
    fig.suptitle(f"pct change for engine_id {engine_id}", fontsize=14)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def pct_change(df, feature_columns):
    df_with_pctchange = df.copy()
    for col in feature_columns:
        df_with_pctchange[f'{col}_pct_change'] = df_with_pctchange.groupby('engine_id')[col].pct_change(5)
    df_with_pctchange.fillna(0, inplace=True)
    return df_with_pctchange
