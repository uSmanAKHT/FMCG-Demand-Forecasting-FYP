import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import LabelEncoder

print("="*80)
print("RANDOM FOREST FORECASTING MODEL")
print("FYP - FMCG Demand Forecasting")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================
print("\nSTEP 1: Loading prepared data...")

df = pd.read_csv(r'C:\Users\User\Documents\FPY\prepared_forecast_data.csv')
df['Date'] = pd.to_datetime(df['Date'])

print(f"Data loaded  : {len(df):,} rows")
print(f"Routes       : {df['Route_Id'].nunique():,}")
print(f"Regions      : {df['Region'].nunique():,}")

# ============================================================================
# STEP 2: Feature Engineering
# ============================================================================
print("\nSTEP 2: Feature Engineering...")

# Encode Region
le_region = LabelEncoder()
df['Region_Encoded'] = le_region.fit_transform(df['Region'])

# Month number
df['Month_Num'] = pd.to_datetime(df['Date']).dt.month

# Features for Random Forest
FEATURES = [
    'Month_Num',
    'Region_Encoded',
    'Eid_Indicator',
    'Avg_Temperature',
    'Season_Index',
    'Lag_Sales',
    'MoM_Growth_Rate',
    'YoY_Growth_Rate',
    'Quarter_Growth_Rate',
]

print(f"Features used: {len(FEATURES)}")
for f in FEATURES:
    print(f"   - {f}")

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("\nSTEP 3: Train/Test Split...")

# Use last 6 months as test
df = df.sort_values(['Route_Id','Date'])
df = df.sort_values('Date')
all_dates   = sorted(df['Date'].unique())
test_dates  = all_dates[-6:]
train_dates = all_dates[:-6]

train_df = df[df['Date'].isin(train_dates)]
test_df  = df[df['Date'].isin(test_dates)]

# Remove NaN
train_df = train_df.dropna(subset=FEATURES + ['Sales_Quantity'])
test_df  = test_df.dropna(subset=FEATURES + ['Sales_Quantity'])

X_train = train_df[FEATURES]
y_train = train_df['Sales_Quantity']
X_test  = test_df[FEATURES]
y_test  = test_df['Sales_Quantity']

print(f"Train size : {len(X_train):,} rows")
print(f"Test size  : {len(X_test):,} rows")

# ============================================================================
# STEP 4: Train Random Forest Model
# ============================================================================
print("\nSTEP 4: Training Random Forest Model...")

rf_model = RandomForestRegressor(
    n_estimators = 100,
    max_depth    = 10,
    random_state = 42,
    n_jobs       = -1
)

rf_model.fit(X_train, y_train)
print("Model trained!")

# ============================================================================
# STEP 5: Predict and Calculate Accuracy
# ============================================================================
print("\nSTEP 5: Calculating Accuracy...")

y_pred = rf_model.predict(X_test)
y_pred = np.maximum(0, y_pred)

# Filter zero actuals
mask   = y_test > 0
y_test_f = y_test[mask]
y_pred_f = y_pred[mask]

# Metrics
mape = np.median(np.abs(
    (y_test_f - y_pred_f) / y_test_f) * 100)
mae  = mean_absolute_error(y_test_f, y_pred_f)
rmse = np.sqrt(mean_squared_error(y_test_f, y_pred_f))

# MASE
naive     = np.mean(np.abs(np.diff(train_df['Sales_Quantity'].values)))
mase      = mae / naive if naive != 0 else np.nan

# R2
ss_res    = np.sum((y_test_f - y_pred_f)**2)
ss_tot    = np.sum((y_test_f - np.mean(y_test_f))**2)
r2        = 1 - (ss_res/ss_tot) if ss_tot != 0 else np.nan

print(f"\n{'='*60}")
print(f"RANDOM FOREST ACCURACY METRICS")
print(f"{'='*60}")
print(f"MAPE : {mape:.2f}%   (lower is better)")
print(f"MAE  : {mae:,.2f} units")
print(f"RMSE : {rmse:,.2f} units")
print(f"MASE : {mase:.4f}   (< 1.0 = better than naive)")
print(f"R2   : {r2:.4f}   (> 0.7 = strong model)")
print(f"{'='*60}")

# ============================================================================
# STEP 6: Feature Importance
# ============================================================================
print("\nSTEP 6: Feature Importance...")

importance_df = pd.DataFrame({
    'Feature'   : FEATURES,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print(f"\n{'FEATURE':<25} {'IMPORTANCE':>12}")
print("-"*40)
for _, row in importance_df.iterrows():
    bar = "█" * int(row['Importance'] * 50)
    print(f"{row['Feature']:<25} {row['Importance']:>10.4f}  {bar}")

# ============================================================================
# STEP 7: Regional Accuracy
# ============================================================================
print("\nSTEP 7: Regional Accuracy...")

test_df2          = test_df.copy()
test_df2['Pred']  = y_pred
test_df2          = test_df2[test_df2['Sales_Quantity'] > 0]
test_df2['MAPE']  = (np.abs(
    test_df2['Sales_Quantity'] - test_df2['Pred']
) / test_df2['Sales_Quantity'] * 100)

reg_acc = test_df2.groupby('Region').agg(
    Routes   = ('Route_Id', 'nunique'),
    MAPE     = ('MAPE',     'median'),
).round(1).reset_index()
reg_acc['Accuracy'] = (100 - reg_acc['MAPE']).round(1)
reg_acc['Rating']   = reg_acc['MAPE'].apply(
    lambda x: 'EXCELLENT' if x<=20 else
              'GOOD'      if x<=40 else
              'ACCEPTABLE'if x<=60 else 'POOR'
)
print(reg_acc.sort_values('MAPE').to_string(index=False))

# ============================================================================
# STEP 8: Forecast Jan 2026
# ============================================================================
print("\nSTEP 8: Forecasting January 2026...")

# Create Jan 2026 features
routes_info = df[['Route_Id','Route_Name',
                   'Region','Territory',
                   'Region_Encoded']].drop_duplicates('Route_Id')

jan2026 = routes_info.copy()
jan2026['Month_Num']          = 1
jan2026['Eid_Indicator']      = 0
jan2026['Avg_Temperature']    = 12.0
jan2026['Season_Index']       = 1

# Use last known lag sales and growth rates
last_data = df.groupby('Route_Id').last().reset_index()
jan2026   = jan2026.merge(
    last_data[['Route_Id','Lag_Sales',
               'MoM_Growth_Rate',
               'YoY_Growth_Rate',
               'Quarter_Growth_Rate']],
    on='Route_Id', how='left'
)
jan2026 = jan2026.fillna(0)

X_jan2026         = jan2026[FEATURES]
jan2026['RF_Forecast']        = rf_model.predict(X_jan2026)
jan2026['RF_Forecast']        = jan2026['RF_Forecast'].clip(lower=0)
jan2026['RF_Growth_25pct']    = (
    jan2026['RF_Forecast'] * 1.25).round()

national_base   = jan2026['RF_Forecast'].sum()
national_growth = jan2026['RF_Growth_25pct'].sum()

print(f"National Base Forecast   : {national_base:,.0f} units")
print(f"National Growth (+25%)   : {national_growth:,.0f} units")

# ============================================================================
# STEP 9: Visualization
# ============================================================================
print("\nSTEP 9: Creating Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Random Forest Forecasting Results',
             fontsize=16, fontweight='bold')

# Plot 1: Feature Importance
axes[0,0].barh(importance_df['Feature'],
               importance_df['Importance'],
               color='steelblue', alpha=0.8)
axes[0,0].set_title('Feature Importance',
                    fontweight='bold')
axes[0,0].set_xlabel('Importance Score')
axes[0,0].grid(True, alpha=0.3, axis='x')

# Plot 2: Actual vs Predicted
sample = test_df2.sample(min(500, len(test_df2)),
                          random_state=42)
axes[0,1].scatter(sample['Sales_Quantity'],
                  sample['Pred'],
                  alpha=0.3, color='steelblue', s=10)
max_val = max(sample['Sales_Quantity'].max(),
              sample['Pred'].max())
axes[0,1].plot([0, max_val],[0, max_val],
               'r--', linewidth=2, label='Perfect Forecast')
axes[0,1].set_title('Actual vs Predicted Sales',
                    fontweight='bold')
axes[0,1].set_xlabel('Actual Sales')
axes[0,1].set_ylabel('Predicted Sales')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Regional MAPE
colors = ['green' if m<=20 else 'blue' if m<=40 else
          'orange' if m<=60 else 'red'
          for m in reg_acc.sort_values('MAPE')['MAPE']]
axes[1,0].barh(reg_acc.sort_values('MAPE')['Region'],
               reg_acc.sort_values('MAPE')['MAPE'],
               color=colors, alpha=0.8)
axes[1,0].axvline(x=mape, color='red',
                  linestyle='--', linewidth=2,
                  label=f'Median MAPE: {mape:.1f}%')
axes[1,0].set_title('MAPE by Region',
                    fontweight='bold')
axes[1,0].set_xlabel('MAPE %')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3, axis='x')

# Plot 4: Model Comparison
models     = ['Moving\nAverage','HW\nRoute',
              'HW\nRegional','Bottom-Up\nHW',
              'Random\nForest']
mapes      = [101.06, 64.26, 26.08, 12.91, mape]
bar_colors = ['red','orange','green',
              'darkgreen','steelblue']
bars = axes[1,1].bar(models, mapes,
                      color=bar_colors, alpha=0.8,
                      edgecolor='white')
for bar, val in zip(bars, mapes):
    axes[1,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 1,
        f'{val:.1f}%', ha='center',
        fontweight='bold', fontsize=9)
axes[1,1].set_title('All Models Comparison (MAPE)',
                    fontweight='bold')
axes[1,1].set_ylabel('MAPE % (lower is better)')
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\random_forest_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 10: Save Results
# ============================================================================
print("\nSTEP 10: Saving Results...")

output = r'C:\Users\User\Documents\FPY\random_forest_forecasts.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:

    # Jan 2026 forecasts
    jan2026[['Route_Id','Route_Name','Region',
             'Territory','RF_Forecast',
             'RF_Growth_25pct']].sort_values(
        ['Region','Route_Id']
    ).to_excel(writer, sheet_name='Jan2026 Forecasts',
               index=False)

    # Regional accuracy
    reg_acc.to_excel(writer,
                     sheet_name='Regional Accuracy',
                     index=False)

    # Feature importance
    importance_df.to_excel(writer,
                            sheet_name='Feature Importance',
                            index=False)

    # Model comparison
    pd.DataFrame([
        {'Model':'Moving Average',    'MAPE':101.06,'Accuracy':18.94},
        {'Model':'HW Route',          'MAPE': 64.26,'Accuracy':35.74},
        {'Model':'HW Regional',       'MAPE': 26.08,'Accuracy':73.92},
        {'Model':'Bottom-Up HW',      'MAPE': 12.91,'Accuracy':87.09},
        {'Model':'Random Forest',     'MAPE': round(mape,2),
         'Accuracy':round(100-mape,2)},
    ]).to_excel(writer, sheet_name='Model Comparison',
                index=False)

print(f"Saved: {output}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("RANDOM FOREST MODEL COMPLETE")
print("="*80)
print(f"\nRandom Forest Accuracy:")
print(f"  MAPE : {mape:.2f}%")
print(f"  MAE  : {mae:,.2f} units")
print(f"  RMSE : {rmse:,.2f} units")
print(f"  MASE : {mase:.4f}")
print(f"  R2   : {r2:.4f}")
print(f"\nAll Models Comparison:")
print(f"  Moving Average  : 101.06%")
print(f"  HW Route        :  64.26%")
print(f"  HW Regional     :  26.08%")
print(f"  Bottom-Up HW    :  12.91%")
print(f"  Random Forest   : {mape:>7.2f}%")
print(f"\nNational Jan 2026 Forecast:")
print(f"  Base   : {national_base:,.0f} units")
print(f"  +25%   : {national_growth:,.0f} units")
print(f"\nFiles saved:")
print(f"  1. random_forest_forecasts.xlsx")
print(f"  2. random_forest_visualization.png")
print("="*80)