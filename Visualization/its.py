import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import statsmodels.api as sm
from prophet import Prophet

# Read data
df = pd.read_csv(r"C:\Users\A1157\Downloads\cleaned2_expression_patterns.csv")
df['date'] = pd.to_datetime(df['date'])

# Aggregate daily data
daily_stats = df.groupby('date').agg({
    'criticism': 'sum',
    'vague': 'sum'
}).reset_index()
daily_stats['vague_ratio'] = daily_stats['vague'] / daily_stats['criticism']

# Define key dates
nsl_date = pd.Timestamp('2020-06-30')
training_start_date = nsl_date - pd.Timedelta(days=180)

# Prepare data for Prophet
prophet_df_total = daily_stats[['date', 'criticism']].rename(
    columns={'date': 'ds', 'criticism': 'y'})
prophet_df_vague = daily_stats[['date', 'vague_ratio']].rename(
    columns={'date': 'ds', 'vague_ratio': 'y'})

# Train the Prophet model (using only pre-NSL data)
def train_prophet(df):
    train_df = df[df['ds'] < nsl_date].copy()
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False
    )
    model.fit(train_df)
    return model

# Generate predictions
def generate_predictions(df, model):
    future = model.make_future_dataframe(periods=(df['ds'].max() - nsl_date).days)
    forecast = model.predict(future)
    return forecast

# Train models and generate predictions
model_total = train_prophet(prophet_df_total)
model_vague = train_prophet(prophet_df_vague)

forecast_total = generate_predictions(prophet_df_total, model_total)
forecast_vague = generate_predictions(prophet_df_vague, model_vague)

# Compute differences and prepare data for segmented regression
def prepare_regression_data(actual_df, forecast_df, value_name):
    # Merge actual values and predicted values
    regression_data = pd.merge(
        actual_df.rename(columns={'y': f'actual_{value_name}'}),
        forecast_df[['ds', 'yhat']].rename(columns={'yhat': f'predicted_{value_name}'}),
        on='ds'
    )
    
    # Compute the difference
    regression_data[f'{value_name}_difference'] = (
        regression_data[f'actual_{value_name}'] - 
        regression_data[f'predicted_{value_name}']
    )
    
    # Add variables required for regression
    regression_data['Time'] = (regression_data['ds'] - regression_data['ds'].min()).dt.days
    regression_data['Post_NSL'] = (regression_data['ds'] >= nsl_date).astype(int)
    regression_data['Time_Post_NSL'] = regression_data['Time'] * regression_data['Post_NSL']
    
    return regression_data

# Perform segmented regression analysis
def perform_segmented_regression(data, y_var, title):
    # Prepare regression variables
    X = sm.add_constant(data[['Time', 'Post_NSL', 'Time_Post_NSL']])
    y = data[y_var]
    
    # Run the regression
    model = sm.OLS(y, X).fit()
    
    # Create predictions
    data['regression_predicted'] = model.predict(X)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(15, 10))
    
    # Plot observed differences
    ax.scatter(data['ds'], data[y_var], alpha=0.5, color='gray', label='Observed Difference')
    
    # Plot the regression line
    ax.plot(data['ds'], data['regression_predicted'], color='red', label='Regression Line')
    
    # Add a reference line at zero
    ax.axhline(y=0, color='blue', linestyle='--', alpha=0.3, label='No Effect Reference')
    
    # Add NSL implementation vertical line
    ax.axvline(x=nsl_date, color='black', linestyle='--', label='NSL Implementation')
    
    # Set chart formatting
    ax.set_title(f'Segmented Regression Analysis of {title}\n(Difference between Actual and Predicted Values)', pad=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Difference (Actual - Predicted)')
    ax.legend()
    
    # Add regression equation and statistics
    equation = f'Difference = {model.params["const"]:.3f}'
    equation += f' + {model.params["Time"]:.3f}×Time'
    equation += f' + {model.params["Post_NSL"]:.3f}×Post_NSL'
    equation += f' + {model.params["Time_Post_NSL"]:.3f}×Time×Post_NSL'
    
    plt.text(0.02, 0.95, f'Regression Equation:\n{equation}', 
             transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.text(0.02, 0.85, 
             f'R² = {model.rsquared:.3f}\n'
             f'Adj R² = {model.rsquared_adj:.3f}\n'
             f'Long-term Effect (β₂) = {model.params["Post_NSL"]:.3f} (p={model.pvalues["Post_NSL"]:.3f})\n'
             f'Immediate Effect (β₃) = {model.params["Time_Post_NSL"]:.3f} (p={model.pvalues["Time_Post_NSL"]:.3f})',
             transform=ax.transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    return model, fig

# Prepare regression data
regression_data_total = prepare_regression_data(prophet_df_total, forecast_total, 'posts')
regression_data_vague = prepare_regression_data(prophet_df_vague, forecast_vague, 'vague')

# Perform segmented regression analysis
model_total, fig_total = perform_segmented_regression(
    regression_data_total, 'posts_difference', 'Critical Posts'
)

model_vague, fig_vague = perform_segmented_regression(
    regression_data_vague, 'vague_difference', 'Vague Expression Ratio'
)

# Save figures
fig_total.savefig('segmented_regression_criticism_difference.png')
fig_vague.savefig('segmented_regression_vague_ratio_difference.png')

# Print detailed statistical results
print("\n=== Segmented Regression Results for Critical Posts ===")
print(model_total.summary())
print("\n=== Segmented Regression Results for Vague Expression Ratio ===")
print(model_vague.summary())

plt.show()
