import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("="*80)
print("REGIONAL HOLT-WINTERS FORECASTING MODEL")
print("FYP - FMCG Demand Forecasting - Regional Level")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

df = pd.read_csv(r'C:\Users\User\Documents\FPY\prepared_forecast_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Data loaded     : {len(df):,} rows")
print(f"Date range      : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Unique routes   : {df['Route_Id'].nunique():,}")
print(f"Unique regions  : {df['Region'].nunique():,}")

# ============================================================================
# STEP 2: Aggregate to Regional Level
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Aggregating Routes to Regional Level")
print("="*80)

regional_df = df.groupby(['Region', 'Date']).agg(
    Sales_Quantity  = ('Sales_Quantity', 'sum'),
    Eid_Indicator   = ('Eid_Indicator',  'max'),
    Avg_Temperature = ('Avg_Temperature','mean'),
    Route_Count     = ('Route_Id',       'nunique')
).reset_index()

regional_df = regional_df.sort_values(['Region', 'Date'])

print(f"Regional records : {len(regional_df):,}")
print(f"Regions          : {regional_df['Region'].nunique():,}")
print(f"\nRegion Summary:")
summary = regional_df.groupby('Region').agg(
    Months      = ('Date',           'count'),
    Avg_Sales   = ('Sales_Quantity', 'mean'),
    Total_Routes= ('Route_Count',    'mean')
).round(0)
print(summary.to_string())

# ============================================================================
# STEP 3: Holt-Winters on Each Region
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Holt-Winters Forecast per Region")
print("="*80)

hw_mape    = []
hw_mae     = []
hw_rmse    = []
hw_mase    = []
hw_r2      = []
ma_mape    = []

region_forecasts = []

for region in regional_df['Region'].unique():
    rdf   = regional_df[regional_df['Region'] == region].sort_values('Date')
    sales = rdf['Sales_Quantity'].values

    if len(sales) < 24:
        print(f"   {region:<20} : Skipped (only {len(sales)} months)")
        continue

    tr = sales[:-6]
    te = sales[-6:]

    # Holt-Winters predictions
    hw_preds = []
    for i in range(6):
        history = list(tr) + list(te[:i])
        try:
            m    = ExponentialSmoothing(
                     history,
                     trend='add',
                     seasonal='add',
                     seasonal_periods=12
                   ).fit(optimized=True)
            pred = max(0, m.forecast(steps=1)[0])
        except:
            pred = np.mean(history[-3:])
        hw_preds.append(pred)

    # Moving Average baseline
    ma_preds = []
    for i in range(6):
        if i < 3:
            p = np.mean(tr[-(3-i):])
        else:
            p = np.mean(te[i-3:i])
        ma_preds.append(p)

    a    = np.array(te)
    a_hw = np.array(hw_preds)
    a_ma = np.array(ma_preds)

    if np.any(a == 0):
        continue

    # HW metrics
    e_hw     = a - a_hw
    mape_hw  = np.mean(np.abs(e_hw / a)) * 100
    mae_hw   = np.mean(np.abs(e_hw))
    rmse_hw  = np.sqrt(np.mean(e_hw**2))
    nb       = np.mean(np.abs(np.diff(sales)))
    mase_hw  = mae_hw / nb if nb != 0 else np.nan
    ss_r     = np.sum(e_hw**2)
    ss_t     = np.sum((a - np.mean(a))**2)
    r2_hw    = 1 - (ss_r / ss_t) if ss_t != 0 else np.nan

    # MA MAPE
    e_ma    = a - a_ma
    mape_ma = np.mean(np.abs(e_ma / a)) * 100

    hw_mape.append(mape_hw)
    hw_mae.append(mae_hw)
    hw_rmse.append(rmse_hw)
    ma_mape.append(mape_ma)
    if not np.isnan(mase_hw): hw_mase.append(mase_hw)
    if not np.isnan(r2_hw):   hw_r2.append(r2_hw)

    # Forecast next month
    try:
        full_model    = ExponentialSmoothing(
                          sales,
                          trend='add',
                          seasonal='add',
                          seasonal_periods=12
                        ).fit(optimized=True)
        next_forecast = max(0, full_model.forecast(steps=1)[0])

        # Eid adjustment
        last_eid = rdf['Eid_Indicator'].iloc[-1]
        if last_eid == 1:
            next_forecast = next_forecast * 1.25
    except:
        next_forecast = np.mean(sales[-3:])

    region_forecasts.append({
        'Region'           : region,
        'MAPE'             : round(mape_hw, 2),
        'MA_MAPE'          : round(mape_ma, 2),
        'MAE'              : round(mae_hw, 2),
        'RMSE'             : round(rmse_hw, 2),
        'MASE'             : round(mase_hw, 4) if not np.isnan(mase_hw) else None,
        'R2'               : round(r2_hw, 4)   if not np.isnan(r2_hw)   else None,
        'Next_Month_Forecast': round(next_forecast),
        'Rating'           : 'EXCELLENT' if mape_hw<=30 else
                             'GOOD'      if mape_hw<=50 else
                             'ACCEPTABLE'if mape_hw<=80 else 'POOR'
    })

    print(f"   {region:<20} : MAPE={mape_hw:.1f}%  MA={mape_ma:.1f}%  Forecast={next_forecast:,.0f}")

# ============================================================================
# STEP 4: Overall Results
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Overall Regional Accuracy")
print("="*80)

med_hw_mape = np.median(hw_mape)
med_hw_mae  = np.median(hw_mae)
med_hw_rmse = np.median(hw_rmse)
med_hw_mase = np.median(hw_mase)
med_hw_r2   = np.median(hw_r2)
med_ma_mape = np.median(ma_mape)
improvement = med_ma_mape - med_hw_mape

print(f"\n{'='*65}")
print(f"{'METRIC':<12} {'MOVING AVG':>15} {'HOLT-WINTERS':>15} {'BETTER?':>10}")
print(f"{'='*65}")
print(f"{'MAPE':<12} {med_ma_mape:>14.2f}% {med_hw_mape:>14.2f}% {'YES' if improvement>0 else 'NO':>10}")
print(f"{'MAE':<12} {'---':>15} {med_hw_mae:>15,.2f}")
print(f"{'RMSE':<12} {'---':>15} {med_hw_rmse:>15,.2f}")
print(f"{'MASE':<12} {'---':>15} {med_hw_mase:>15.4f}  {'GOOD<1.0' if med_hw_mase<1 else 'BAD>1.0'}")
print(f"{'R2':<12} {'---':>15} {med_hw_r2:>15.4f}  {'GOOD>0.7' if med_hw_r2>0.7 else 'BAD<0.7'}")
print(f"{'='*65}")
print(f"Improvement vs Moving Average: {improvement:+.2f}%")

# Region details table
print(f"\n{'REGION':<20} {'MAPE':>8} {'MA MAPE':>10} {'FORECAST':>12} {'RATING':>12}")
print("-"*65)
rf_df = pd.DataFrame(region_forecasts).sort_values('MAPE')
for _, row in rf_df.iterrows():
    print(f"{row['Region']:<20} {row['MAPE']:>7.1f}% {row['MA_MAPE']:>9.1f}% "
          f"{row['Next_Month_Forecast']:>12,} {row['Rating']:>12}")

# ============================================================================
# STEP 5: Compare Route vs Regional Accuracy
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Route Level vs Regional Level Comparison")
print("="*80)

print(f"\n{'MODEL':<35} {'MAPE':>10} {'IMPROVEMENT':>15}")
print("-"*62)
print(f"{'Moving Average (Baseline)':<35} {'101.06%':>10} {'---':>15}")
print(f"{'Holt-Winters Route Level':<35} {'64.26%':>10} {'+36.80%':>15}")
print(f"{'Holt-Winters Regional Level':<35} {med_hw_mape:>9.2f}% {med_ma_mape-med_hw_mape:>+14.2f}%")

# ============================================================================
# STEP 6: Visualization
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Creating Visualization")
print("="*80)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: MAPE by Region
regions_sorted = rf_df.sort_values('MAPE')
colors = ['green' if m<=30 else 'blue' if m<=50 else
          'orange' if m<=80 else 'red'
          for m in regions_sorted['MAPE']]

axes[0].barh(regions_sorted['Region'],
             regions_sorted['MAPE'],
             color=colors, alpha=0.8, edgecolor='white')
axes[0].axvline(x=med_hw_mape, color='red', linewidth=2,
                linestyle='--',
                label=f'Median MAPE: {med_hw_mape:.1f}%')
axes[0].set_title('Holt-Winters MAPE by Region',
                  fontsize=13, fontweight='bold')
axes[0].set_xlabel('MAPE (%)', fontsize=11)
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3, axis='x')

# Plot 2: HW vs MA comparison
x     = np.arange(len(rf_df))
width = 0.35
axes[1].bar(x - width/2, rf_df['MA_MAPE'],
            width, label='Moving Average', color='coral', alpha=0.8)
axes[1].bar(x + width/2, rf_df['MAPE'],
            width, label='Holt-Winters', color='steelblue', alpha=0.8)
axes[1].set_xticks(x)
axes[1].set_xticklabels(rf_df['Region'], rotation=45, ha='right')
axes[1].set_title('Holt-Winters vs Moving Average by Region',
                  fontsize=13, fontweight='bold')
axes[1].set_ylabel('MAPE (%)', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\regional_forecast_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 7: Save Report
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Saving Report")
print("="*80)

report = f"""
REGIONAL HOLT-WINTERS FORECAST REPORT
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

MODEL: Holt-Winters Exponential Smoothing - Regional Aggregation
Level  : Regional (14 regions aggregated from 2,645 routes)
Method : Additive Trend + Additive Seasonality
Period : Monthly (seasonal_periods=12)

OVERALL ACCURACY (Median):
Moving Average MAPE  : {med_ma_mape:.2f}%
Holt-Winters MAPE    : {med_hw_mape:.2f}%
Improvement          : {improvement:+.2f}%
MAE                  : {med_hw_mae:,.2f} units
RMSE                 : {med_hw_rmse:,.2f} units
MASE                 : {med_hw_mase:.4f}
R2                   : {med_hw_r2:.4f}

3-LEVEL COMPARISON:
Moving Average (Baseline)    : 101.06%
Holt-Winters Route Level     :  64.26%
Holt-Winters Regional Level  :  {med_hw_mape:.2f}%

REGIONAL FORECASTS (Next Month):
"""

for _, row in rf_df.sort_values('Region').iterrows():
    report += f"  {row['Region']:<20}: {row['Next_Month_Forecast']:>10,} units  (MAPE={row['MAPE']}%  {row['Rating']})\n"

report += "="*80

report_file = r'C:\Users\User\Documents\FPY\regional_forecast_report.txt'
with open(report_file, 'w') as f:
    f.write(report)

print(report)
print(f"Report saved: {report_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("REGIONAL FORECASTING COMPLETE")
print("="*80)
print(f"\nRegional MAPE     : {med_hw_mape:.2f}%")
print(f"Route Level MAPE  : 64.26%")
print(f"Moving Avg MAPE   : 101.06%")
print(f"\nRegional improved by: {med_ma_mape - med_hw_mape:+.2f}% vs Moving Average")
print(f"\nFiles saved:")
print(f"  1. regional_forecast_visualization.png")
print(f"  2. regional_forecast_report.txt")
print("="*80)