import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("="*80)
print("BOTTOM-UP HIERARCHICAL FORECASTING MODEL (FAST VERSION)")
print("FYP - FMCG Demand Forecasting System")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

db_path = r'C:\Users\User\Documents\FPY\sales_data.db'
conn    = sqlite3.connect(db_path)
df      = pd.read_sql_query("SELECT * FROM sales_data", conn)
conn.close()

month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
df['Month_Num'] = df['Month'].map(month_map)
df['Date']      = pd.to_datetime({'year' :df['Year'],
                                   'month':df['Month_Num'],
                                   'day'  :1})

SKU_COLS = [c for c in df.columns if any(x in c for x in
                                          ['CSD','JUICE','WATER'])]

print(f"Data loaded : {len(df):,} rows")
print(f"SKUs        : {len(SKU_COLS)}")
print(f"Routes      : {df['Route_Id'].nunique():,}")
print(f"Regions     : {df['Region'].nunique():,}")

# ============================================================================
# STEP 2: SKU Level — Simple Moving Average (FAST)
# ============================================================================
print("\n" + "="*80)
print("STEP 2: SKU Level Forecast (Moving Average - Fast)")
print("="*80)

sku_forecasts = []

for sku in SKU_COLS:
    for route_id in df['Route_Id'].unique():
        rdf   = df[df['Route_Id'] == route_id].sort_values('Date')
        sales = pd.to_numeric(rdf[sku], errors='coerce').fillna(0).values

        # Simple 3-month moving average
        if len(sales) >= 3:
            forecast_val = np.mean(sales[-3:])
        elif len(sales) > 0:
            forecast_val = np.mean(sales)
        else:
            forecast_val = 0

        sku_forecasts.append({
            'Route_Id'   : route_id,
            'Route_Name' : rdf['Route_Name'].iloc[0],
            'Region'     : rdf['Region'].iloc[0],
            'SKU'        : sku,
            'Forecast'   : max(0, round(forecast_val))
        })

sku_df = pd.DataFrame(sku_forecasts)
print(f"SKU-Route forecasts : {len(sku_df):,}")

# ============================================================================
# STEP 3: Route Level — Holt-Winters (on total sales)
# ============================================================================
print("\n" + "="*80)
print("STEP 3: Route Level Holt-Winters Forecast")
print("="*80)
print("Processing routes...")

# Total sales per route per month
df['Total_Sales'] = df[SKU_COLS].sum(axis=1)

route_hw_forecasts = []
route_actuals      = []
done               = 0

for route_id in df['Route_Id'].unique():
    rdf   = df[df['Route_Id'] == route_id].sort_values('Date')
    sales = rdf['Total_Sales'].values

    if len(sales) < 24:
        # Use MA for short series
        forecast_val = np.mean(sales[-3:]) if len(sales) >= 3 else np.mean(sales)
        route_hw_forecasts.append({
            'Route_Id'      : route_id,
            'Route_Name'    : rdf['Route_Name'].iloc[0],
            'Region'        : rdf['Region'].iloc[0],
            'HW_Forecast'   : max(0, round(forecast_val)),
            'Method'        : 'Moving Average'
        })
        continue

    tr = sales[:-6]
    te = sales[-6:]

    # Holt-Winters
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
            pred = max(0, np.mean(history[-3:]))
        hw_preds.append(pred)

    # Store for accuracy
    for i in range(6):
        if te[i] > 0:
            route_actuals.append({
                'Route_Id' : route_id,
                'Region'   : rdf['Region'].iloc[0],
                'Actual'   : te[i],
                'Predicted': hw_preds[i]
            })

    # Final forecast
    try:
        full_m  = ExponentialSmoothing(
                    sales,
                    trend='add',
                    seasonal='add',
                    seasonal_periods=12
                  ).fit(optimized=True)
        next_f  = max(0, full_m.forecast(steps=1)[0])
    except:
        next_f  = max(0, np.mean(sales[-3:]))

    route_hw_forecasts.append({
        'Route_Id'   : route_id,
        'Route_Name' : rdf['Route_Name'].iloc[0],
        'Region'     : rdf['Region'].iloc[0],
        'HW_Forecast': round(next_f),
        'Method'     : 'Holt-Winters'
    })

    done += 1
    if done % 100 == 0:
        print(f"   Progress: {done}/{df['Route_Id'].nunique()} routes done...")

route_hw_df  = pd.DataFrame(route_hw_forecasts)
actuals_df   = pd.DataFrame(route_actuals)
print(f"Route forecasts done: {len(route_hw_df):,}")

# ============================================================================
# STEP 4: Bottom-Up Aggregation
# ============================================================================
print("\n" + "="*80)
print("STEP 4: Bottom-Up Aggregation")
print("="*80)

# SKU Bottom-Up Route totals
sku_route_totals = sku_df.groupby(
    ['Route_Id','Route_Name','Region']
)['Forecast'].sum().reset_index()
sku_route_totals.columns = ['Route_Id','Route_Name',
                             'Region','SKU_BU_Forecast']

# Region level from SKU bottom-up
region_bu_df = sku_route_totals.groupby(
    'Region'
)['SKU_BU_Forecast'].sum().reset_index()

# Region level from HW route
region_hw_df = route_hw_df.groupby(
    'Region'
)['HW_Forecast'].sum().reset_index()

# Merge
region_df = region_bu_df.merge(region_hw_df, on='Region')

# National
national_bu = sku_route_totals['SKU_BU_Forecast'].sum()
national_hw = route_hw_df['HW_Forecast'].sum()

print(f"SKU Bottom-Up Route   : {len(sku_route_totals):,} routes")
print(f"HW Route Forecasts    : {len(route_hw_df):,} routes")
print(f"Region Forecasts      : {len(region_df):,} regions")
print(f"National (SKU BU)     : {national_bu:,.0f} units")
print(f"National (HW Route)   : {national_hw:,.0f} units")

# ============================================================================
# STEP 5: Accuracy Metrics
# ============================================================================
print("\n" + "="*80)
print("STEP 5: Accuracy Metrics")
print("="*80)

if len(actuals_df) > 0:
    actuals_df['Error']    = actuals_df['Actual'] - actuals_df['Predicted']
    actuals_df['Abs_Error']= np.abs(actuals_df['Error'])
    actuals_df['Pct_Error']= (actuals_df['Abs_Error'] /
                               actuals_df['Actual'] * 100)

    route_mape = actuals_df.groupby('Route_Id').apply(
        lambda x: np.mean(np.abs(
            (x['Actual']-x['Predicted'])/x['Actual']
        )*100)
    ).median()

    region_acc = actuals_df.groupby('Region').agg(
        Actual    = ('Actual',    'sum'),
        Predicted = ('Predicted', 'sum')
    ).reset_index()
    region_acc['Pct_Error'] = (np.abs(
        region_acc['Actual'] - region_acc['Predicted']
    ) / region_acc['Actual'] * 100)
    region_mape = region_acc['Pct_Error'].median()

    print(f"\n{'='*65}")
    print(f"{'MODEL':<35} {'MAPE':>10}")
    print(f"{'='*65}")
    print(f"{'Moving Average (Baseline)':<35} {'101.06%':>10}")
    print(f"{'Holt-Winters Route Level':<35} {'64.26%':>10}")
    print(f"{'Holt-Winters Regional Level':<35} {'26.08%':>10}")
    print(f"{'Bottom-Up HW Route Level':<35} {route_mape:>9.2f}%")
    print(f"{'Bottom-Up HW Region Level':<35} {region_mape:>9.2f}%")
    print(f"{'='*65}")

    print(f"\nRegion Detail:")
    prev_hw = {
        'BAHAWALPUR':14.6,'FAISALABAD':29.9,'GUJRANWALA':15.9,
        'KASUR':26.1,'LAHORE':21.1,'MULTAN':21.9,'PESHAWAR':39.0,
        'RAWALPINDI':29.8,'SAHIWAL':39.1,'SARGODHA':20.2,'SINDH':86.7
    }
    print(f"{'REGION':<20} {'BOTTOM-UP':>12} {'PREV HW':>10} {'RESULT':>10}")
    print("-"*55)
    for _, row in region_acc.sort_values('Pct_Error').iterrows():
        prev   = prev_hw.get(row['Region'], 0)
        better = "BETTER" if row['Pct_Error'] < prev else "WORSE"
        print(f"{row['Region']:<20} {row['Pct_Error']:>11.2f}% "
              f"{prev:>9.1f}%  {better:>10}")
else:
    route_mape  = 0
    region_mape = 0

# ============================================================================
# STEP 6: Save to Excel
# ============================================================================
print("\n" + "="*80)
print("STEP 6: Saving to Excel")
print("="*80)

output_excel = r'C:\Users\User\Documents\FPY\bottomup_forecasts.xlsx'
with pd.ExcelWriter(output_excel, engine='openpyxl') as writer:

    sku_df.to_excel(
        writer, sheet_name='SKU Forecasts', index=False)

    sku_route_totals.sort_values(
        ['Region','Route_Id']
    ).to_excel(writer, sheet_name='Route BU Forecasts', index=False)

    route_hw_df.sort_values(
        ['Region','Route_Id']
    ).to_excel(writer, sheet_name='Route HW Forecasts', index=False)

    region_df.to_excel(
        writer, sheet_name='Region Forecasts', index=False)

    pd.DataFrame([{
        'SKU_BottomUp_National' : national_bu,
        'HW_Route_National'     : national_hw
    }]).to_excel(writer, sheet_name='National', index=False)

    if len(actuals_df) > 0:
        pd.DataFrame([
            {'Level':'Route',  'MAPE':round(route_mape,2)},
            {'Level':'Region', 'MAPE':round(region_mape,2)},
        ]).to_excel(writer, sheet_name='Accuracy', index=False)

print(f"Excel saved: {output_excel}")

# ============================================================================
# STEP 7: Visualization
# ============================================================================
print("\n" + "="*80)
print("STEP 7: Creating Visualization")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Bottom-Up Hierarchical Forecasting Results',
             fontsize=16, fontweight='bold')

# Plot 1: Region HW forecasts
axes[0,0].barh(region_df['Region'],
               region_df['HW_Forecast']/1000,
               color='steelblue', alpha=0.8)
axes[0,0].set_title('HW Regional Forecasts (Next Month)',
                    fontweight='bold')
axes[0,0].set_xlabel('Forecast (000 units)')
axes[0,0].grid(True, alpha=0.3, axis='x')

# Plot 2: Top 10 SKUs
top_skus = sku_df.groupby('SKU')['Forecast'].sum().nlargest(10)
labels   = [s.replace('CSD ','').replace(
             'JUICE ','J-').replace(
             'WATER ','W-') for s in top_skus.index]
axes[0,1].barh(range(len(top_skus)),
               top_skus.values/1000,
               color='coral', alpha=0.8)
axes[0,1].set_yticks(range(len(top_skus)))
axes[0,1].set_yticklabels(labels, fontsize=8)
axes[0,1].set_title('Top 10 SKUs by Forecast Volume',
                    fontweight='bold')
axes[0,1].set_xlabel('Forecast (000 units)')
axes[0,1].grid(True, alpha=0.3, axis='x')

# Plot 3: Model comparison
models = ['Moving\nAverage',
          'HW Route\nLevel',
          'HW Regional\nLevel',
          'BU HW\nRoute',
          'BU HW\nRegion']
mapes  = [101.06, 64.26, 26.08, route_mape, region_mape]
colors = ['red','orange','green','steelblue','darkblue']
bars   = axes[1,0].bar(models, mapes,
                        color=colors, alpha=0.8,
                        edgecolor='white')
for bar, val in zip(bars, mapes):
    axes[1,0].text(bar.get_x() + bar.get_width()/2,
                   bar.get_height() + 1,
                   f'{val:.1f}%', ha='center',
                   fontweight='bold', fontsize=9)
axes[1,0].set_title('All Models Comparison (MAPE)',
                    fontweight='bold')
axes[1,0].set_ylabel('MAPE % (lower is better)')
axes[1,0].grid(True, alpha=0.3, axis='y')

# Plot 4: Hierarchy info
axes[1,1].axis('off')
txt = f"""
BOTTOM-UP HIERARCHY SUMMARY

National Level
SKU BU : {national_bu:>15,.0f} units
HW     : {national_hw:>15,.0f} units

Regional Level ({len(region_df)} regions)
Median MAPE : {region_mape:.1f}%

Route Level ({len(route_hw_df):,} routes)
Median MAPE : {route_mape:.1f}%

SKU Level ({len(sku_df):,} combinations)
35 SKUs x Routes
Method: Moving Average
"""
axes[1,1].text(0.5, 0.5, txt,
               transform=axes[1,1].transAxes,
               fontsize=10,
               verticalalignment='center',
               horizontalalignment='center',
               bbox=dict(boxstyle='round',
                         facecolor='lightblue',
                         alpha=0.5),
               fontfamily='monospace')

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\bottomup_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 8: Report
# ============================================================================
print("\n" + "="*80)
print("STEP 8: Saving Report")
print("="*80)

report = f"""
BOTTOM-UP HIERARCHICAL FORECAST REPORT
Generated : {datetime.now().strftime('%Y-%m-%d %H:%M')}
================================================================================

METHOD: Bottom-Up Hierarchical Forecasting
SKU Level    : Moving Average (fast, 35 SKUs x routes)
Route Level  : Holt-Winters Exponential Smoothing
Region Level : Aggregated from Route Level
National     : Aggregated from Region Level

HIERARCHY:
National SKU BU  : {national_bu:>15,.0f} units
National HW      : {national_hw:>15,.0f} units
Regions          : {len(region_df):>15,}
Routes           : {len(route_hw_df):>15,}
SKU-Routes       : {len(sku_df):>15,}

FULL MODEL COMPARISON (Median MAPE):
Moving Average Baseline  : 101.06%
Holt-Winters Route       :  64.26%
Holt-Winters Regional    :  26.08%
Bottom-Up HW Route       : {route_mape:>7.2f}%
Bottom-Up HW Region      : {region_mape:>7.2f}%

REGIONAL FORECASTS (Next Month - HW):
"""
for _, row in region_df.sort_values(
        'HW_Forecast', ascending=False).iterrows():
    report += (f"  {row['Region']:<20}: "
               f"SKU_BU={row['SKU_BU_Forecast']:>12,.0f}  "
               f"HW={row['HW_Forecast']:>12,.0f}\n")

report += "="*80

report_file = r'C:\Users\User\Documents\FPY\bottomup_report.txt'
with open(report_file, 'w') as f:
    f.write(report)

print(report)

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("="*80)
print("BOTTOM-UP MODEL COMPLETE")
print("="*80)
print(f"\nAll Models MAPE Comparison:")
print(f"  Moving Average     : 101.06%")
print(f"  HW Route Level     :  64.26%")
print(f"  HW Regional        :  26.08%")
print(f"  BU HW Route        : {route_mape:>7.2f}%")
print(f"  BU HW Region       : {region_mape:>7.2f}%")
print(f"\nNational Forecast   : {national_hw:,.0f} units")
print(f"\nFiles saved:")
print(f"  1. bottomup_forecasts.xlsx")
print(f"  2. bottomup_visualization.png")
print(f"  3. bottomup_report.txt")
print("="*80)
