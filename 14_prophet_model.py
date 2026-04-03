import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("="*80)
print("PROPHET DEEP LEARNING FORECASTING MODEL")
print("Built by Facebook/Meta")
print("FYP - FMCG Demand Forecasting")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading prepared data...")

df = pd.read_csv(r'C:\Users\User\Documents\FPY\prepared_forecast_data.csv')
df['Date'] = pd.to_datetime(df['Date'])
df         = df.sort_values(['Route_Id','Date'])

print(f"Data loaded  : {len(df):,} rows")
print(f"Routes       : {df['Route_Id'].nunique():,}")
print(f"Regions      : {df['Region'].nunique():,}")

# ============================================================================
# STEP 2: Aggregate to Regional Level
# ============================================================================
print("\nSTEP 2: Aggregating to Regional Level...")

regional_df = df.groupby(['Region','Date']).agg(
    Sales_Quantity  = ('Sales_Quantity', 'sum'),
    Eid_Indicator   = ('Eid_Indicator',  'max'),
    Avg_Temperature = ('Avg_Temperature','mean'),
).reset_index().sort_values(['Region','Date'])

print(f"Regional records : {len(regional_df):,}")
print(f"Regions          : {regional_df['Region'].nunique()}")

# ============================================================================
# STEP 3: Create Eid Holidays DataFrame
# ============================================================================
print("\nSTEP 3: Setting up Eid holidays...")

# Eid dates for Pakistan
eid_holidays = pd.DataFrame({
    'holiday'   : 'Eid',
    'ds'        : pd.to_datetime([
        '2023-04-21', '2023-06-28',
        '2024-04-10', '2024-06-17',
        '2025-03-30', '2025-06-06',
        '2026-03-20', '2026-05-27',
    ]),
    'lower_window': -1,
    'upper_window':  2,
})

print(f"Eid holidays added: {len(eid_holidays)}")

# ============================================================================
# STEP 4: Train Prophet Per Region
# ============================================================================
print("\nSTEP 4: Training Prophet per Region...")
print("Please wait...")

region_results = []
all_mape       = []
all_mae        = []
all_rmse       = []

for region in regional_df['Region'].unique():
    rdf = regional_df[
        regional_df['Region']==region
    ].copy().sort_values('Date')

    if len(rdf) < 24:
        print(f"   {region:<20} : Skipped ({len(rdf)} months)")
        continue

    # Prophet needs ds and y
    prophet_df = rdf.rename(columns={
        'Date'          : 'ds',
        'Sales_Quantity': 'y'
    })

    train_df = prophet_df.iloc[:-6].copy()
    test_df  = prophet_df.iloc[-6:].copy()

    try:
        # Build Prophet model
        m = Prophet(
            yearly_seasonality  = True,
            weekly_seasonality  = False,
            daily_seasonality   = False,
            seasonality_mode    = 'additive',
            holidays            = eid_holidays,
            changepoint_prior_scale = 0.05,
        )

        # Add temperature as regressor
        m.add_regressor('Avg_Temperature')

        # Fit
        m.fit(train_df[['ds','y','Avg_Temperature']])

        # Make future dataframe
        future          = m.make_future_dataframe(
                            periods=6, freq='MS')
        future          = future.merge(
                            prophet_df[['ds','Avg_Temperature']],
                            on='ds', how='left')
        future['Avg_Temperature'] = future[
            'Avg_Temperature'].fillna(
            future['Avg_Temperature'].mean())

        # Predict
        forecast = m.predict(future)

        # Test accuracy
        test_preds = forecast.tail(6)['yhat'].values
        test_preds = np.maximum(0, test_preds)
        actuals    = test_df['y'].values

        mask   = actuals > 0
        act_f  = actuals[mask]
        pred_f = test_preds[mask]

        if len(act_f) == 0:
            continue

        mape_r = np.mean(np.abs(
            (act_f - pred_f) / act_f) * 100)
        mae_r  = mean_absolute_error(act_f, pred_f)
        rmse_r = np.sqrt(mean_squared_error(act_f, pred_f))

        all_mape.append(mape_r)
        all_mae.append(mae_r)
        all_rmse.append(rmse_r)

        jan_forecast = max(0, forecast['yhat'].iloc[-1])

        region_results.append({
            'Region'            : region,
            'MAPE'              : round(mape_r, 2),
            'MAE'               : round(mae_r, 2),
            'RMSE'              : round(rmse_r, 2),
            'Jan2026_Forecast'  : round(jan_forecast),
            'Jan2026_Growth_25' : round(jan_forecast * 1.25),
            'Rating'            : 'EXCELLENT' if mape_r<=20 else
                                  'GOOD'      if mape_r<=40 else
                                  'ACCEPTABLE'if mape_r<=60 else 'POOR'
        })

        print(f"   {region:<20} : MAPE={mape_r:.1f}%  "
              f"Forecast={jan_forecast:,.0f}")

    except Exception as e:
        print(f"   {region:<20} : Failed - {str(e)[:60]}")
        continue

# ============================================================================
# STEP 5: Overall Accuracy
# ============================================================================
print("\nSTEP 5: Overall Accuracy...")

med_mape = np.median(all_mape) if len(all_mape) > 0 else 0
med_mae  = np.median(all_mae)  if len(all_mae)  > 0 else 0
med_rmse = np.median(all_rmse) if len(all_rmse) > 0 else 0

print(f"\n{'='*60}")
print(f"PROPHET ACCURACY METRICS")
print(f"{'='*60}")
print(f"MAPE : {med_mape:.2f}%   (lower is better)")
print(f"MAE  : {med_mae:,.2f} units")
print(f"RMSE : {med_rmse:,.2f} units")
print(f"{'='*60}")

# ============================================================================
# STEP 6: Results Table
# ============================================================================
print("\nSTEP 6: Regional Results...")

results_df      = pd.DataFrame(region_results)
national_base   = results_df['Jan2026_Forecast'].sum() \
                  if len(results_df) > 0 else 0
national_growth = results_df['Jan2026_Growth_25'].sum() \
                  if len(results_df) > 0 else 0

if len(results_df) > 0:
    print(f"\n{'REGION':<20} {'MAPE':>8} {'FORECAST':>14} {'RATING':>12}")
    print("-"*57)
    for _, row in results_df.sort_values('MAPE').iterrows():
        print(f"{row['Region']:<20} {row['MAPE']:>7.1f}% "
              f"{row['Jan2026_Forecast']:>14,.0f} "
              f"{row['Rating']:>12}")
    print(f"\nNational Base   : {national_base:,.0f} units")
    print(f"National +25%   : {national_growth:,.0f} units")

# ============================================================================
# STEP 7: Visualization
# ============================================================================
print("\nSTEP 7: Creating Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16,12))
fig.suptitle('Prophet Forecasting Results\n'
             'Facebook/Meta - Handles Eid + Seasonality',
             fontsize=14, fontweight='bold')

# Plot 1: Regional MAPE
if len(results_df) > 0:
    colors = ['green' if m<=20 else 'blue' if m<=40 else
              'orange' if m<=60 else 'red'
              for m in results_df.sort_values('MAPE')['MAPE']]
    axes[0,0].barh(
        results_df.sort_values('MAPE')['Region'],
        results_df.sort_values('MAPE')['MAPE'],
        color=colors, alpha=0.8)
    axes[0,0].axvline(x=med_mape, color='red',
                      linestyle='--', linewidth=2,
                      label=f'Median: {med_mape:.1f}%')
    axes[0,0].set_title('Prophet MAPE by Region',
                        fontweight='bold')
    axes[0,0].set_xlabel('MAPE %')
    axes[0,0].legend()
    axes[0,0].grid(True, alpha=0.3, axis='x')

# Plot 2: Regional Forecasts
if len(results_df) > 0:
    axes[0,1].barh(
        results_df.sort_values(
            'Jan2026_Forecast')['Region'],
        results_df.sort_values(
            'Jan2026_Forecast')['Jan2026_Forecast']/1000,
        color='steelblue', alpha=0.8)
    axes[0,1].set_title('Jan 2026 Regional Forecasts',
                        fontweight='bold')
    axes[0,1].set_xlabel('Forecast (000 units)')
    axes[0,1].grid(True, alpha=0.3, axis='x')

# Plot 3: All Models Comparison
models     = ['Moving\nAverage','HW\nRoute',
              'GLM\nPoisson','LSTM',
              'HW\nRegional','Bottom-Up\nHW',
              'XGBoost','Random\nForest',
              'Prophet']
mapes      = [101.06, 64.26, 59.49, 45.24,
              26.08,  12.91,  4.55,  1.55,
              med_mape]
bar_colors = ['red','orange','brown','purple',
              'green','darkgreen','coral',
              'steelblue','gold']
bars = axes[1,0].bar(models, mapes,
                      color=bar_colors,
                      alpha=0.8, edgecolor='white')
for bar, val in zip(bars, mapes):
    axes[1,0].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{val:.1f}%', ha='center',
        fontweight='bold', fontsize=7)
axes[1,0].set_title('All 9 Models Comparison (MAPE)',
                    fontweight='bold')
axes[1,0].set_ylabel('MAPE % (lower is better)')
axes[1,0].grid(True, alpha=0.3, axis='y')

# Plot 4: Summary
axes[1,1].axis('off')
summary_txt = f"""
PROPHET MODEL SUMMARY

Built by     : Facebook / Meta
Type         : Bayesian Forecasting
Handles      : Eid Holidays + Seasonality
               + Temperature + Trend

ACCURACY:
Median MAPE  : {med_mape:.2f}%
Median MAE   : {med_mae:,.0f} units
Median RMSE  : {med_rmse:,.0f} units

NATIONAL FORECAST:
Base         : {national_base:>12,.0f} units
+25% Growth  : {national_growth:>12,.0f} units

Regions Done : {len(results_df)}
"""
axes[1,1].text(0.5, 0.5, summary_txt,
               transform=axes[1,1].transAxes,
               fontsize=10,
               verticalalignment='center',
               horizontalalignment='center',
               bbox=dict(boxstyle='round',
                         facecolor='lightyellow',
                         alpha=0.8),
               fontfamily='monospace')

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\prophet_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 8: Save Results
# ============================================================================
print("\nSTEP 8: Saving Results...")

output = r'C:\Users\User\Documents\FPY\prophet_results.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    if len(results_df) > 0:
        results_df.to_excel(
            writer, sheet_name='Regional Results',
            index=False)
    pd.DataFrame([
        {'Model':'Moving Average',
         'Type':'Statistical',      'MAPE':101.06},
        {'Model':'HW Route',
         'Type':'Statistical',      'MAPE': 64.26},
        {'Model':'GLM Poisson',
         'Type':'Statistical ML',   'MAPE': 59.49},
        {'Model':'LSTM',
         'Type':'Deep Learning',    'MAPE': 45.24},
        {'Model':'HW Regional',
         'Type':'Statistical',      'MAPE': 26.08},
        {'Model':'Bottom-Up HW',
         'Type':'Statistical',      'MAPE': 12.91},
        {'Model':'XGBoost',
         'Type':'Machine Learning', 'MAPE':  4.55},
        {'Model':'Random Forest',
         'Type':'Machine Learning', 'MAPE':  1.55},
        {'Model':'Prophet',
         'Type':'Bayesian DL',
         'MAPE': round(med_mape,2)},
    ]).to_excel(writer,
                sheet_name='All Models',
                index=False)

print(f"Saved: {output}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PROPHET MODEL COMPLETE")
print("="*80)
print(f"\nProphet Accuracy : {med_mape:.2f}% MAPE")
print(f"\nAll 9 Models:")
print(f"  1. Moving Average  : 101.06%  Statistical")
print(f"  2. HW Route        :  64.26%  Statistical")
print(f"  3. GLM Poisson     :  59.49%  Statistical ML")
print(f"  4. LSTM            :  45.24%  Deep Learning")
print(f"  5. HW Regional     :  26.08%  Statistical")
print(f"  6. Bottom-Up HW    :  12.91%  Statistical")
print(f"  7. XGBoost         :   4.55%  Machine Learning")
print(f"  8. Random Forest   :   1.55%  Machine Learning")
print(f"  9. Prophet         : {med_mape:>7.2f}%  Bayesian DL")
print(f"\nNational Base   : {national_base:,.0f} units")
print(f"National +25%   : {national_growth:,.0f} units")
print(f"\nFiles saved:")
print(f"  1. prophet_results.xlsx")
print(f"  2. prophet_visualization.png")
print("="*80)
