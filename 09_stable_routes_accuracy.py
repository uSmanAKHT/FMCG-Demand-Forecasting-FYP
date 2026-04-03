import pandas as pd
import numpy as np
import sqlite3
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.holtwinters import ExponentialSmoothing

print("="*80)
print("STABLE ROUTES ACCURACY ANALYSIS")
print("Only Routes Active in ALL 3 Years (2023-2024-2025)")
print("="*80)

GROWTH_RATE = 0.25

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading data...")

conn = sqlite3.connect(r'C:\Users\User\Documents\FPY\sales_data.db')
df   = pd.read_sql_query("SELECT * FROM sales_data", conn)
conn.close()

month_map = {'Jan':1,'Feb':2,'Mar':3,'Apr':4,'May':5,'Jun':6,
             'Jul':7,'Aug':8,'Sep':9,'Oct':10,'Nov':11,'Dec':12}
df['Month_Num']   = df['Month'].map(month_map)
df['Date']        = pd.to_datetime({'year' :df['Year'],
                                     'month':df['Month_Num'],
                                     'day'  :1})
SKU_COLS          = [c for c in df.columns if any(
                     x in c for x in ['CSD','JUICE','WATER'])]
df['Total_Sales'] = df[SKU_COLS].sum(axis=1)

print(f"Data loaded: {len(df):,} rows")

# ============================================================================
# STEP 2: Classify Routes
# ============================================================================
print("\nSTEP 2: Classifying routes...")

r2023 = set(df[(df['Year']==2023) &
               (df['Total_Sales']>0)]['Route_Id'])
r2024 = set(df[(df['Year']==2024) &
               (df['Total_Sales']>0)]['Route_Id'])
r2025 = set(df[(df['Year']==2025) &
               (df['Total_Sales']>0)]['Route_Id'])

stable_routes  = r2023 & r2024 & r2025
new_routes     = r2025 - r2024
growing_routes = (r2024 - r2023) | (r2025 - r2024 - r2023)

print(f"Stable Routes  (all 3 years) : {len(stable_routes):,}")
print(f"Growing Routes               : {len(growing_routes):,}")
print(f"New Routes     (2025 only)   : {len(new_routes):,}")

# ============================================================================
# STEP 3: Forecast ONLY Stable Routes
# ============================================================================
print("\nSTEP 3: Forecasting stable routes...")

actuals_df = pd.read_excel(
    r'C:\Users\User\Documents\FPY\Holt_forcast_data.xlsx'
)
actuals_df.columns = actuals_df.columns.str.strip()
print(f"Actuals columns: {list(actuals_df.columns)}")

stable_df = df[df['Route_Id'].isin(stable_routes)].copy()
stable_df = stable_df.sort_values(['Route_Id','Date'])
print(f"Stable route records : {len(stable_df):,}")

results  = []
done     = 0
hw_count = 0
ma_count = 0

for route_id in stable_routes:
    rdf   = stable_df[stable_df['Route_Id']==route_id].sort_values('Date')
    sales = rdf['Total_Sales'].values

    if len(sales) < 24:
        base   = np.mean(sales[-3:]) if len(sales)>=3 else np.mean(sales)
        method = 'MA_Fallback'
        ma_count += 1
    else:
        try:
            m      = ExponentialSmoothing(
                       sales,
                       trend='add',
                       seasonal='add',
                       seasonal_periods=12
                     ).fit(optimized=True)
            base   = max(0, m.forecast(1)[0])
            method = 'Holt_Winters'
            hw_count += 1
        except:
            base   = max(0, np.mean(sales[-3:]))
            method = 'MA_Fallback'
            ma_count += 1

    jan_data   = rdf[rdf['Date'].dt.month==1]
    jan_avg    = jan_data['Total_Sales'].mean() \
                 if len(jan_data)>0 else base
    overall    = rdf['Total_Sales'].mean()
    jan_factor = (jan_avg/overall) if overall>0 else 1.0
    forecast   = max(0, base * jan_factor)
    growth_f   = forecast * (1 + GROWTH_RATE)

    results.append({
        'Route_Id'       : route_id,
        'Region'         : rdf['Region'].iloc[0],
        'Route_Status'   : 'Stable',
        'Base_Forecast'  : round(forecast),
        'Growth_Forecast': round(growth_f),
        'Method'         : method
    })

    done += 1
    if done % 200 == 0:
        print(f"   Progress: {done}/{len(stable_routes)} done...")

results_df = pd.DataFrame(results)
print(f"\nForecasts done : {len(results_df):,}")
print(f"Holt-Winters   : {hw_count:,}")
print(f"MA Fallback    : {ma_count:,}")

# ============================================================================
# STEP 4: Merge with Actuals and Calculate Accuracy
# ============================================================================
print("\nSTEP 4: Calculating accuracy on stable routes...")


sys_target_col = [c for c in actuals_df.columns
                  if 'target' in c.lower()][0]
sys_sales_col  = [c for c in actuals_df.columns
                  if 'sales' in c.lower()][0]

print(f"Target column : {sys_target_col}")
print(f"Sales column  : {sys_sales_col}")

merged = results_df.merge(
    actuals_df[['Route_Id', sys_target_col, sys_sales_col]],
    on='Route_Id', how='inner'
)
merged = merged.rename(columns={
    sys_target_col: 'System_Target',
    sys_sales_col : 'System_Sales'
})
merged = merged[merged['System_Sales']  > 0]
merged = merged[merged['System_Target'] > 0]

print(f"Stable routes with actuals : {len(merged):,}")

merged['err_vs_sales']  = np.abs(
    merged['Growth_Forecast'] - merged['System_Sales'])
merged['pct_vs_sales']  = (
    merged['err_vs_sales'] / merged['System_Sales'] * 100)

merged['err_vs_target'] = np.abs(
    merged['Growth_Forecast'] - merged['System_Target'])
merged['pct_vs_target'] = (
    merged['err_vs_target'] / merged['System_Target'] * 100)

merged['err_tgt_sales'] = np.abs(
    merged['System_Target'] - merged['System_Sales'])
merged['pct_tgt_sales'] = (
    merged['err_tgt_sales'] / merged['System_Sales'] * 100)

total = len(merged)

print(f"\n{'='*65}")
print(f"{'COMPARISON':<40} {'MAPE':>10} {'ACCURACY':>12}")
print(f"{'='*65}")
for label, col in [
    ('HW Forecast vs Actual Sales',   'pct_vs_sales'),
    ('HW Forecast vs System Target',  'pct_vs_target'),
    ('System Target vs Actual Sales', 'pct_tgt_sales'),
]:
    med = merged[col].median()
    print(f"{label:<40} {med:>9.2f}% {100-med:>11.2f}%")
print(f"{'='*65}")

print(f"\nWithin Range (HW Forecast vs Actual Sales):")
for pct in [10, 20, 30, 50]:
    w = (merged['pct_vs_sales'] <= pct).sum()
    print(f"  Within {pct}% : {w}/{total} = {w/total*100:.1f}%")

print(f"\n=== REGIONAL ACCURACY - STABLE ROUTES ONLY ===")
reg = merged.groupby('Region').agg(
    Routes        = ('Route_Id',      'count'),
    MAPE_vs_Sales = ('pct_vs_sales',  'median'),
    MAPE_vs_Tgt   = ('pct_vs_target', 'median'),
).round(1).reset_index()
reg['Accuracy'] = (100 - reg['MAPE_vs_Sales']).round(1)
reg['Rating']   = reg['MAPE_vs_Sales'].apply(
    lambda x: 'EXCELLENT' if x<=20 else
              'GOOD'      if x<=40 else
              'ACCEPTABLE'if x<=60 else 'POOR'
)
print(reg.sort_values('MAPE_vs_Sales').to_string(index=False))

over  = (merged['Growth_Forecast'] > merged['System_Sales']).sum()
under = (merged['Growth_Forecast'] < merged['System_Sales']).sum()
print(f"\nOver-forecast  : {over:,} routes ({over/total*100:.1f}%)")
print(f"Under-forecast : {under:,} routes ({under/total*100:.1f}%)")

# ============================================================================
# STEP 5: Save Results
# ============================================================================
print("\nSTEP 5: Saving results...")

output = r'C:\Users\User\Documents\FPY\stable_routes_accuracy.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:
    merged.to_excel(
        writer, sheet_name='Stable Routes', index=False)
    reg.to_excel(
        writer, sheet_name='Regional Accuracy', index=False)
    pd.DataFrame([
        {'Category':'Stable (all 3 years)', 'Count':len(stable_routes)},
        {'Category':'Growing Routes',        'Count':len(growing_routes)},
        {'Category':'New Routes (2025)',      'Count':len(new_routes)},
    ]).to_excel(
        writer, sheet_name='Route Classification', index=False)

print(f"Saved: {output}")

print("\n" + "="*80)
print("STABLE ROUTES ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey Finding:")
print(f"Stable Routes Only : {len(merged):,} routes")
print(f"HW vs Actual Sales : {merged['pct_vs_sales'].median():.2f}% MAPE")
print(f"Accuracy           : {100-merged['pct_vs_sales'].median():.2f}%")
print("="*80)
