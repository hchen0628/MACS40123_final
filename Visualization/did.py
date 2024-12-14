import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
import statsmodels.api as sm
from datetime import datetime

# Load data
df = pd.read_csv(r"C:\Users\A1157\Downloads\cleaned2_expression_patterns.csv")
df['date'] = pd.to_datetime(df['date'])

# Aggregate daily statistics
daily_stats = df.groupby('date').agg({
    'criticism': 'sum',
    'vague': 'sum'
}).reset_index()
daily_stats['vague_ratio'] = daily_stats['vague'] / daily_stats['criticism']

# Define key dates
nsl_date = pd.Timestamp('2020-06-30')

# Prepare Prophet data and train model
def prepare_prophet_data(data, target_col):
    prophet_df = data[['date', target_col]].rename(
        columns={'date': 'ds', target_col: 'y'})
    
    # Train model using pre-NSL data
    model = Prophet(yearly_seasonality=True,
                   weekly_seasonality=True,
                   daily_seasonality=False)
    model.fit(prophet_df[prophet_df['ds'] < nsl_date])
    
    # Generate predictions
    future = model.make_future_dataframe(periods=(data['date'].max() - nsl_date).days)
    forecast = model.predict(future)
    
    return prophet_df, forecast

# Prepare DID data
def prepare_did_data(actual_data, forecast_data, value_name):
    # Actual data (treatment group)
    treatment_data = actual_data[['ds', 'y']].copy()
    treatment_data['treatment'] = 1
    treatment_data['post'] = (treatment_data['ds'] >= nsl_date).astype(int)
    treatment_data = treatment_data.rename(columns={'y': value_name})
    
    # Predicted data (control group)
    control_data = forecast_data[['ds', 'yhat']].copy()
    control_data['treatment'] = 0
    control_data['post'] = (control_data['ds'] >= nsl_date).astype(int)
    control_data = control_data.rename(columns={'yhat': value_name})
    
    # Merge data
    did_data = pd.concat([treatment_data, control_data], ignore_index=True)
    
    # Add time variable
    did_data['time'] = (did_data['ds'] - did_data['ds'].min()).dt.days
    
    return did_data

# Perform DID regression
def perform_did_regression(data, value_name):
    # Create interaction term
    data['did_interaction'] = data['treatment'] * data['post']
    
    # Prepare regression variables
    X = sm.add_constant(data[['treatment', 'post', 'time', 'did_interaction']])
    y = data[value_name]
    
    # Run regression
    model = sm.OLS(y, X).fit()
    
    return model

# Visualize DID results
def plot_did_results(data, value_name, model, title):
    plt.figure(figsize=(12, 8))
    
    # Create scatter plots
    sns.scatterplot(data=data[data['treatment']==1], 
                    x='ds', y=value_name, 
                    color='blue', alpha=0.5, 
                    label='Treatment (Actual)')
    
    sns.scatterplot(data=data[data['treatment']==0], 
                    x='ds', y=value_name, 
                    color='red', alpha=0.5, 
                    label='Control (Predicted)')
    
    # Add regression lines
    for treatment in [0, 1]:
        group_data = data[data['treatment'] == treatment]
        pre_nsl = group_data[group_data['ds'] < nsl_date]
        post_nsl = group_data[group_data['ds'] >= nsl_date]
        
        # Fit pre and post period trend lines
        if len(pre_nsl) > 0:
            z = np.polyfit(pre_nsl['time'], pre_nsl[value_name], 1)
            p = np.poly1d(z)
            plt.plot(pre_nsl['ds'], p(pre_nsl['time']), 
                    '--', color='blue' if treatment else 'red')
        
        if len(post_nsl) > 0:
            z = np.polyfit(post_nsl['time'], post_nsl[value_name], 1)
            p = np.poly1d(z)
            plt.plot(post_nsl['ds'], p(post_nsl['time']), 
                    '-', color='blue' if treatment else 'red')
    
    # Add NSL implementation line
    plt.axvline(x=nsl_date, color='black', linestyle='--', label='NSL Implementation')
    
    # Set chart format
    plt.title(f'DID Analysis: {title}')
    plt.xlabel('Date')
    plt.ylabel(value_name)
    plt.legend()
    
    # Add regression results
    text = f'DID Estimate (δ): {model.params["did_interaction"]:.3f}\n'
    text += f'p-value: {model.pvalues["did_interaction"]:.3f}\n'
    text += f'R²: {model.rsquared:.3f}'
    
    plt.text(0.02, 0.98, text, transform=plt.gca().transAxes, 
             verticalalignment='top', bbox=dict(facecolor='white', alpha=0.8))
    
    plt.grid(True, alpha=0.3)
    return plt.gcf()

# Analyze critical posts
prophet_df_criticism, forecast_criticism = prepare_prophet_data(daily_stats, 'criticism')
did_data_criticism = prepare_did_data(prophet_df_criticism, forecast_criticism, 'posts')
did_model_criticism = perform_did_regression(did_data_criticism, 'posts')

# Analyze vague expression ratio
prophet_df_vague, forecast_vague = prepare_prophet_data(daily_stats, 'vague_ratio')
did_data_vague = prepare_did_data(prophet_df_vague, forecast_vague, 'vague_ratio')
did_model_vague = perform_did_regression(did_data_vague, 'vague_ratio')

# Generate and save figures
fig_criticism = plot_did_results(did_data_criticism, 'posts', 
                                did_model_criticism, 'Critical Posts')
fig_vague = plot_did_results(did_data_vague, 'vague_ratio', 
                            did_model_vague, 'Vague Expression Ratio')

# Save figures
fig_criticism.savefig('did_analysis_criticism.png')
fig_vague.savefig('did_analysis_vague_ratio.png')

# Print detailed results
print("\n=== DID Results for Critical Posts ===")
print(did_model_criticism.summary())

print("\n=== DID Results for Vague Expression Ratio ===")
print(did_model_vague.summary())

plt.show()