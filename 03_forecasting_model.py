import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("="*80)
print("HOLT-WINTERS SEASONAL FORECASTING MODEL")
print("FYP - FMCG Demand Forecasting System")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading prepared data...")

df = pd.read_csv(r'C:\Users\User\Documents\FPY\prepared_forecast_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Data loaded     : {len(df):,} rows")
print(f"Date range      : {df['Date'].min().date()} to {df['Date'].max().date()}")
print(f"Unique routes   : {df['Route_Id'].nunique():,}")
print(f"Unique regions  : {df['Region'].nunique():,}")

# ============================================================================
# STEP 2: Filter Eligible Routes
# ============================================================================
print("\n" + "="*80)
print("STEP 2: Filtering Eligible Routes")
print("="*80)

# Holt-Winters needs at least 24 months (2 full seasons)
eligible   = []
ineligible = []

for route_id in df['Route_Id'].unique():
    rdf   = df[df['Route_Id'] == route_id]
    sales = rdf['Sales_Quantity'].values
    mean_s = np.mean(sales)
    if len(sales) >= 24 and mean_s > 0:
        eligible.append(route_id)
    else:
        ineligible.append(route_id)

print(f"Eligible routes   (24+ months) : {len(eligible):,}")
print(f"Ineligible routes (< 24 months): {len(ineligible):,}")
print(f"Total routes                   : {len(eligible)+len(ineligible):,}")

# ============================================================================
# STEP 3: Holt-Winters on Sample Route
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Holt-Winters Forecast - Sample Route")
print("="*80)

route_counts = df[df['Route_Id'].isin(eligible)].groupby('Route_Id').size()
sample_route = route_counts.idxmax()
route_data   = df[df['Route_Id'] == sample_route].sort_values('Date').copy()

print(f"Route ID    : {sample_route}")
print(f"Route Name  : {route_data['Route_Name'].iloc[0]}")
print(f"Region      : {route_data['Region'].iloc[0]}")
print(f"Territory   : {route_data['Territory'].iloc[0]}")
print(f"Data points : {len(route_data)}")

sales_series = route_data['Sales_Quantity'].values

# Fit Holt-Winters with Additive Trend + Additive Seasonality
try:
    hw_model    = ExponentialSmoothing(
                    sales_series,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=12
                  ).fit(optimized=True)
    hw_forecast = hw_model.forecast(steps=1)[0]
    hw_forecast = max(0, hw_forecast)   # no negative forecast
    print(f"\nHolt-Winters Forecast next month: {hw_forecast:,.0f} units")
    print(f"Alpha (level)   : {hw_model.params['smoothing_level']:.4f}")
    print(f"Beta  (trend)   : {hw_model.params['smoothing_trend']:.4f}")
    print(f"Gamma (seasonal): {hw_model.params['smoothing_seasonal']:.4f}")
except Exception as e:
    hw_forecast = np.mean(sales_series[-3:])
    print(f"HW fallback to MA: {hw_forecast:,.0f} units | Error: {e}")

# Eid Adjustment
last_eid = route_data['Eid_Indicator'].iloc[-1] if 'Eid_Indicator' in route_data.columns else 0
next_month_eid = 0   # set to 1 if next month has Eid
if next_month_eid == 1:
    hw_forecast = hw_forecast * 1.25   # 25% Eid boost
    print(f"After Eid adjustment: {hw_forecast:,.0f} units")

# ============================================================================
# STEP 4: Accuracy on ALL Eligible Routes
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Accuracy Across ALL Eligible Routes")
print("="*80)
print("Processing routes - please wait...")

hw_mape  = []
hw_mae   = []
hw_rmse  = []
hw_mase  = []
hw_r2    = []
ma_mape  = []   # Moving Average baseline for comparison

done = 0
errors_count = 0

for route_id in eligible:
    rdf   = df[df['Route_Id'] == route_id].sort_values('Date')
    sales = rdf['Sales_Quantity'].values

    if len(sales) < 24:
        continue

    tr = sales[:-6]   # train: all except last 6
    te = sales[-6:]   # test:  last 6 months

    # ── Holt-Winters predictions ─────────────────────────────────────────────
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
            pred = m.forecast(steps=1)[0]
            pred = max(0, pred)
        except:
            pred = np.mean(history[-3:])
            errors_count += 1
        hw_preds.append(pred)

    # ── Moving Average baseline ───────────────────────────────────────────────
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
    e_hw    = a - a_hw
    mape_hw = np.mean(np.abs(e_hw / a)) * 100
    mae_hw  = np.mean(np.abs(e_hw))
    rmse_hw = np.sqrt(np.mean(e_hw**2))
    nb      = np.mean(np.abs(np.diff(sales)))
    mase_hw = mae_hw / nb if nb != 0 else np.nan
    ss_r    = np.sum(e_hw**2)
    ss_t    = np.sum((a - np.mean(a))**2)
    r2_hw   = 1 - (ss_r / ss_t) if ss_t != 0 else np.nan

    # MA baseline MAPE
    e_ma    = a - a_ma
    mape_ma = np.mean(np.abs(e_ma / a)) * 100

    hw_mape.append(mape_hw)
    hw_mae.append(mae_hw)
    hw_rmse.append(rmse_hw)
    ma_mape.append(mape_ma)
    if not np.isnan(mase_hw): hw_mase.append(mase_hw)
    if not np.isnan(r2_hw):   hw_r2.append(r2_hw)

    done += 1
    if done % 50 == 0:
        print(f"   Progress: {done}/{len(eligible)} routes done...")

# Final metrics
med_hw_mape  = np.median(hw_mape)
med_hw_mae   = np.median(hw_mae)
med_hw_rmse  = np.median(hw_rmse)
med_hw_mase  = np.median(hw_mase)
med_hw_r2    = np.median(hw_r2)
med_ma_mape  = np.median(ma_mape)
improvement  = med_ma_mape - med_hw_mape

print(f"\nRoutes Evaluated  : {len(hw_mape):,}")
print(f"Fallback errors   : {errors_count:,}")

print(f"\n{'='*65}")
print(f"{'METRIC':<12} {'MOVING AVG':>15} {'HOLT-WINTERS':>15} {'BETTER?':>10}")
print(f"{'='*65}")
print(f"{'MAPE':<12} {med_ma_mape:>14.2f}% {med_hw_mape:>14.2f}% {'YES' if improvement>0 else 'NO':>10}")
print(f"{'MAE':<12} {'---':>15} {med_hw_mae:>15,.2f}")
print(f"{'RMSE':<12} {'---':>15} {med_hw_rmse:>15,.2f}")
print(f"{'MASE':<12} {'---':>15} {med_hw_mase:>15.4f}  {'GOOD < 1.0' if med_hw_mase<1 else 'BAD > 1.0'}")
print(f"{'R2':<12} {'---':>15} {med_hw_r2:>15.4f}  {'GOOD > 0.7' if med_hw_r2>0.7 else 'BAD < 0.7'}")
print(f"{'='*65}")
print(f"MAPE Improvement  : {improvement:+.2f}% vs Moving Average baseline")

# ============================================================================
# STEP 5: Accuracy by Region
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Accuracy by Region")
print("="*80)

region_results = {}
for route_id in eligible:
    rdf    = df[df['Route_Id'] == route_id].sort_values('Date')
    region = rdf['Region'].iloc[0]
    sales  = rdf['Sales_Quantity'].values

    if len(sales) < 24:
        continue

    tr = sales[:-6]
    te = sales[-6:]

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

    a    = np.array(te)
    a_hw = np.array(hw_preds)

    if np.any(a == 0):
        continue

    e_hw    = a - a_hw
    mape_hw = np.mean(np.abs(e_hw / a)) * 100

    if region not in region_results:
        region_results[region] = []
    region_results[region].append(mape_hw)

print(f"\n{'REGION':<20} {'ROUTES':>8} {'MEDIAN MAPE':>14} {'RATING':>10}")
print("-"*55)
for region, mapes in sorted(region_results.items()):
    med   = np.median(mapes)
    count = len(mapes)
    if med <= 30:
        rating = "EXCELLENT"
    elif med <= 50:
        rating = "GOOD"
    elif med <= 80:
        rating = "ACCEPTABLE"
    else:
        rating = "POOR"
    print(f"{region:<20} {count:>8} {med:>13.2f}% {rating:>10}")

# ============================================================================
# STEP 6: Visualization
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Creating Visualization")
print("="*80)

sales_s  = route_data['Sales_Quantity'].values
dates_s  = route_data['Date'].values
tr_s     = sales_s[:-6]
te_s     = sales_s[-6:]
dates_te = dates_s[-6:]

preds_plot = []
for i in range(6):
    history = list(tr_s) + list(te_s[:i])
    try:
        m  = ExponentialSmoothing(
               history,
               trend='add',
               seasonal='add',
               seasonal_periods=12
             ).fit(optimized=True)
        p  = max(0, m.forecast(steps=1)[0])
    except:
        p = np.mean(history[-3:])
    preds_plot.append(p)

next_date = pd.Timestamp(dates_s[-1]) + pd.DateOffset(months=1)

fig, axes = plt.subplots(2, 1, figsize=(14, 10))

# Plot 1: Forecast
axes[0].plot(dates_s, sales_s, 'o-', label='Actual Sales',
             linewidth=2, markersize=4, color='blue')
axes[0].plot(dates_te, preds_plot, 's--',
             label='Holt-Winters Fitted (Test)',
             linewidth=2, markersize=6, color='orange')
axes[0].plot([next_date], [hw_forecast], 'r*',
             markersize=20,
             label=f'HW Forecast: {hw_forecast:,.0f} units')
axes[0].axvline(x=dates_te[0], color='gray',
                linestyle=':', alpha=0.7, label='Train/Test Split')
axes[0].set_title(f'Holt-Winters Forecast - {route_data["Route_Name"].iloc[0]}',
                  fontsize=13, fontweight='bold')
axes[0].set_ylabel('Sales Quantity', fontsize=11)
axes[0].legend(fontsize=9)
axes[0].grid(True, alpha=0.3)
plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45)

# Plot 2: MAPE distribution
axes[1].hist(hw_mape, bins=30, color='steelblue',
             edgecolor='white', alpha=0.8)
axes[1].axvline(x=med_hw_mape, color='red', linewidth=2,
                linestyle='--', label=f'Median MAPE: {med_hw_mape:.1f}%')
axes[1].set_title('MAPE Distribution Across All Routes',
                  fontsize=13, fontweight='bold')
axes[1].set_xlabel('MAPE (%)', fontsize=11)
axes[1].set_ylabel('Number of Routes', fontsize=11)
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\holtwinters_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 7: Save Full Report
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Saving Report")
print("="*80)

report = f"""
HOLT-WINTERS DEMAND FORECAST REPORT
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

MODEL DESCRIPTION:
Method     : Holt-Winters Exponential Smoothing
Components : Additive Trend + Additive Seasonality
Period     : Monthly (seasonal_periods = 12)
Eid Adjust : 25% boost when Eid falls in forecast month

DATASET:
Total Rows       : {len(df):,}
Date Range       : {df['Date'].min().date()} to {df['Date'].max().date()}
Total Routes     : {df['Route_Id'].nunique():,}
Eligible Routes  : {len(eligible):,} (24+ months data)
Routes Evaluated : {len(hw_mape):,}

SAMPLE ROUTE FORECAST:
Route ID   : {sample_route}
Route Name : {route_data['Route_Name'].iloc[0]}
Region     : {route_data['Region'].iloc[0]}
Territory  : {route_data['Territory'].iloc[0]}
Forecast   : {hw_forecast:,.0f} units next month

ACCURACY METRICS (Median across {len(hw_mape):,} routes):
Metric       Holt-Winters    Moving Avg    Improvement
MAPE       : {med_hw_mape:>10.2f}%   {med_ma_mape:>10.2f}%   {improvement:>+10.2f}%
MAE        : {med_hw_mae:>10,.2f} units
RMSE       : {med_hw_rmse:>10,.2f} units
MASE       : {med_hw_mase:>10.4f}      (< 1.0 = better than naive)
R2         : {med_hw_r2:>10.4f}      (> 0.7 = strong model)

PRODUCTION RECOMMENDATIONS (Sample Route):
Production Target : {hw_forecast*1.10:,.0f} units (10% safety buffer)
Reorder Point     : {hw_forecast*0.80:,.0f} units (80% threshold)
Safety Stock      : {hw_forecast*0.20:,.0f} units (20% buffer)

================================================================================
"""

report_file = r'C:\Users\User\Documents\FPY\holtwinters_report.txt'
with open(report_file, 'w') as f:
    f.write(report)

print(report)
print(f"Report saved: {report_file}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("HOLT-WINTERS MODEL COMPLETE")
print("="*80)
print(f"\nNext Month Forecast : {hw_forecast:,.0f} units")
print(f"\nAccuracy (Median - {len(hw_mape):,} routes):")
print(f"  MAPE : {med_hw_mape:.2f}%")
print(f"  MAE  : {med_hw_mae:,.2f} units")
print(f"  RMSE : {med_hw_rmse:,.2f} units")
print(f"  MASE : {med_hw_mase:.4f}")
print(f"  R2   : {med_hw_r2:.4f}")
print(f"\nHolt-Winters vs Moving Average:")
print(f"  HW MAPE  : {med_hw_mape:.2f}%")
print(f"  MA MAPE  : {med_ma_mape:.2f}%")
print(f"  Better by: {improvement:+.2f}%")
print(f"\nFiles saved:")
print(f"  1. holtwinters_visualization.png")
print(f"  2. holtwinters_report.txt")
print("="*80)