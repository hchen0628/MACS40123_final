import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns
from scipy import stats

# Load data
df = pd.read_csv(r"C:\Users\A1157\Downloads\cleaned2_expression_patterns.csv")
df['date'] = pd.to_datetime(df['date'])

# Define key dates
nsl_date = pd.Timestamp('2020-06-30')
movement_date = pd.Timestamp('2019-03-15')
training_start_date = nsl_date - pd.Timedelta(days=180)  # Use only 6 months data before NSL

# Aggregate daily statistics
daily_stats = df.groupby('date').agg({
    'criticism': 'count',  # Total posts
    'vague': lambda x: (x == 1).sum()  # Vague expression count
}).reset_index()

daily_stats['vague_ratio'] = daily_stats['vague'] / daily_stats['criticism']

# Prepare Prophet data
prophet_df_total = daily_stats[['date', 'criticism']].rename(
    columns={'date': 'ds', 'criticism': 'y'})
prophet_df_vague = daily_stats[['date', 'vague_ratio']].rename(
    columns={'date': 'ds', 'vague_ratio': 'y'})

# Train model (using only 6 months data before NSL)
def train_prophet(df, changepoint_prior_scale=0.05):
    train_df = df[
        (df['ds'] >= training_start_date) & 
        (df['ds'] < nsl_date)
    ].copy()
    model = Prophet(
        changepoint_prior_scale=changepoint_prior_scale,
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df)
    return model

# Train model and generate predictions
def generate_predictions(df, model):
    future = model.make_future_dataframe(
        periods=(df['ds'].max() - nsl_date).days
    )
    forecast = model.predict(future)
    return forecast

# Train models
model_total = train_prophet(prophet_df_total)
model_vague = train_prophet(prophet_df_vague)

# Generate predictions
forecast_total = generate_predictions(prophet_df_total, model_total)
forecast_vague = generate_predictions(prophet_df_vague, model_vague)

# Calculate differences between predicted and actual values
def calculate_differences(actual_df, forecast_df, value_column='y'):
    merged_df = pd.merge(
        actual_df,
        forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']],
        left_on='ds',
        right_on='ds'
    )
    merged_df['difference'] = merged_df[value_column] - merged_df['yhat']
    merged_df['significant_difference'] = (
        (merged_df[value_column] < merged_df['yhat_lower']) |
        (merged_df[value_column] > merged_df['yhat_upper'])
    )
    return merged_df

# Calculate differences
diff_total = calculate_differences(prophet_df_total, forecast_total)
diff_vague = calculate_differences(prophet_df_vague, forecast_vague)

# Analyze differences across periods
def analyze_period_differences(diff_df, period_start, period_end, period_name):
    period_data = diff_df[
        (diff_df['ds'] >= period_start) &
        (diff_df['ds'] < period_end)
    ]
    
    t_stat, p_value = stats.ttest_1samp(period_data['difference'], 0)
    d = np.mean(period_data['difference']) / np.std(period_data['difference'])
    
    return {
        'period': period_name,
        'mean_difference': period_data['difference'].mean(),
        'std_difference': period_data['difference'].std(),
        'p_value': p_value,
        'effect_size': d,
        'significant_days': period_data['significant_difference'].sum(),
        'total_days': len(period_data)
    }

# Define analysis periods
periods = [
    (nsl_date, nsl_date + pd.Timedelta(days=180), 'First 6 months'),
    (nsl_date + pd.Timedelta(days=180), nsl_date + pd.Timedelta(days=365), 'Second 6 months'),
    (nsl_date + pd.Timedelta(days=365), df['date'].max(), 'After first year')
]

# Analyze each period
results_total = []
results_vague = []
for start, end, name in periods:
    results_total.append(analyze_period_differences(diff_total, start, end, name))
    results_vague.append(analyze_period_differences(diff_vague, start, end, name))

# Visualize results
plt.style.use('seaborn-v0_8-paper')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12))

# Plot total posts comparison
ax1.plot(prophet_df_total['ds'], prophet_df_total['y'], 
         label='Actual', alpha=0.6)
ax1.plot(forecast_total['ds'], forecast_total['yhat'], 
         label='Predicted', color='red')
ax1.fill_between(forecast_total['ds'], 
                 forecast_total['yhat_lower'], 
                 forecast_total['yhat_upper'],
                 color='red', alpha=0.1)
ax1.axvline(x=nsl_date, color='black', linestyle='--', 
            label='National Security Law')
ax1.axvline(x=movement_date, color='blue', linestyle='--', 
            label='Anti-ELAB Movement')
ax1.axvline(x=training_start_date, color='green', linestyle=':', 
            label='Training Start')
ax1.set_title('Total Critical Posts: Actual vs Predicted')
ax1.legend()

# Plot vague expression ratio comparison
ax2.plot(prophet_df_vague['ds'], prophet_df_vague['y'], 
         label='Actual', alpha=0.6)
ax2.plot(forecast_vague['ds'], forecast_vague['yhat'], 
         label='Predicted', color='red')
ax2.fill_between(forecast_vague['ds'], 
                 forecast_vague['yhat_lower'], 
                 forecast_vague['yhat_upper'],
                 color='red', alpha=0.1)
ax2.axvline(x=nsl_date, color='black', linestyle='--', 
            label='National Security Law')
ax2.axvline(x=movement_date, color='blue', linestyle='--', 
            label='Anti-ELAB Movement')
ax2.axvline(x=training_start_date, color='green', linestyle=':', 
            label='Training Start')
ax2.set_title('Vague Expression Ratio in Criticise: Actual vs Predicted')
ax2.legend()

plt.tight_layout()
plt.show()

# Print analysis results
print("\n=== Total Posts Analysis ===")
for result in results_total:
    print(f"\nPeriod: {result['period']}")
    print(f"Mean difference: {result['mean_difference']:.2f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size']:.2f}")
    print(f"Significant days: {result['significant_days']}/{result['total_days']}")

print("\n=== Vague Expression Ratio Analysis ===")
for result in results_vague:
    print(f"\nPeriod: {result['period']}")
    print(f"Mean difference: {result['mean_difference']:.2f}")
    print(f"P-value: {result['p_value']:.4f}")
    print(f"Effect size (Cohen's d): {result['effect_size']:.2f}")