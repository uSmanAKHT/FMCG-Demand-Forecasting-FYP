import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import statsmodels.api as sm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

print("="*80)
print("GLM POISSON REGRESSION FORECASTING MODEL")
print("Based on Rostami-Tabar & Hyndman (2024)")
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
# STEP 2: Feature Engineering
# ============================================================================
print("\nSTEP 2: Feature Engineering...")

le                   = LabelEncoder()
df['Region_Encoded'] = le.fit_transform(df['Region'])
df['Month_Num']      = df['Date'].dt.month

# Fourier terms for seasonality (like reference paper)
df['sin_month']      = np.sin(2 * np.pi * df['Month_Num'] / 12)
df['cos_month']      = np.cos(2 * np.pi * df['Month_Num'] / 12)

# Time trend
df['Time_Index']     = df.groupby('Route_Id').cumcount()

FEATURES = [
    'Time_Index',
    'sin_month',
    'cos_month',
    'Region_Encoded',
    'Eid_Indicator',
    'Avg_Temperature',
    'Season_Index',
    'Lag_Sales',
    'MoM_Growth_Rate',
    'YoY_Growth_Rate',
    'Quarter_Growth_Rate',
]

TARGET = 'Sales_Quantity'

df = df.dropna(subset=FEATURES + [TARGET])
df = df[df[TARGET] >= 0]

print(f"Features used : {len(FEATURES)}")
for f in FEATURES:
    print(f"   - {f}")
print(f"Clean rows    : {len(df):,}")

# ============================================================================
# STEP 3: Train/Test Split
# ============================================================================
print("\nSTEP 3: Train/Test Split...")

all_dates  = sorted(df['Date'].unique())
test_dates = all_dates[-6:]
train_dates= all_dates[:-6]

train_df   = df[df['Date'].isin(train_dates)].copy()
test_df    = df[df['Date'].isin(test_dates)].copy()

print(f"Train size : {len(train_df):,} rows")
print(f"Test size  : {len(test_df):,} rows")

# ============================================================================
# STEP 4: Train GLM Poisson Model
# ============================================================================
print("\nSTEP 4: Training GLM Poisson Model...")
print("This aligns with Rostami-Tabar & Hyndman (2024)")

X_train = sm.add_constant(train_df[FEATURES])
y_train = train_df[TARGET]

X_test  = sm.add_constant(test_df[FEATURES])
y_test  = test_df[TARGET]

# Fit Poisson GLM
glm_model = sm.GLM(
    y_train,
    X_train,
    family=sm.families.Poisson()
).fit()

print(f"Model fitted!")
print(f"AIC  : {glm_model.aic:,.2f}")
print(f"Deviance : {glm_model.deviance:,.2f}")

# ============================================================================
# STEP 5: Predict and Calculate Accuracy
# ============================================================================
print("\nSTEP 5: Calculating Accuracy...")

y_pred   = glm_model.predict(X_test)
y_pred   = np.maximum(0, y_pred)

mask     = y_test > 0
y_test_f = y_test[mask]
y_pred_f = y_pred[mask]

mape = np.median(np.abs(
    (y_test_f - y_pred_f) / y_test_f) * 100)
mae  = mean_absolute_error(y_test_f, y_pred_f)
rmse = np.sqrt(mean_squared_error(y_test_f, y_pred_f))
naive= np.mean(np.abs(np.diff(train_df[TARGET].values)))
mase = mae / naive if naive != 0 else np.nan
ss_r = np.sum((y_test_f - y_pred_f)**2)
ss_t = np.sum((y_test_f - np.mean(y_test_f))**2)
r2   = 1 - (ss_r/ss_t) if ss_t != 0 else np.nan

print(f"\n{'='*60}")
print(f"GLM POISSON ACCURACY METRICS")
print(f"{'='*60}")
print(f"MAPE : {mape:.2f}%   (lower is better)")
print(f"MAE  : {mae:,.2f} units")
print(f"RMSE : {rmse:,.2f} units")
print(f"MASE : {mase:.4f}   (< 1.0 = better than naive)")
print(f"R2   : {r2:.4f}   (> 0.7 = strong model)")
print(f"{'='*60}")

# ============================================================================
# STEP 6: Regional Accuracy
# ============================================================================
print("\nSTEP 6: Regional Accuracy...")

test_df2          = test_df.copy()
test_df2['Pred']  = y_pred.values
test_df2          = test_df2[test_df2[TARGET] > 0]
test_df2['MAPE']  = (np.abs(
    test_df2[TARGET] - test_df2['Pred']
) / test_df2[TARGET] * 100)

reg_acc = test_df2.groupby('Region').agg(
    Routes = ('Route_Id', 'nunique'),
    MAPE   = ('MAPE',     'median'),
).round(1).reset_index()
reg_acc['Accuracy'] = (100 - reg_acc['MAPE']).round(1)
reg_acc['Rating']   = reg_acc['MAPE'].apply(
    lambda x: 'EXCELLENT' if x<=20 else
              'GOOD'      if x<=40 else
              'ACCEPTABLE'if x<=60 else 'POOR'
)
print(reg_acc.sort_values('MAPE').to_string(index=False))

# ============================================================================
# STEP 7: Feature Coefficients
# ============================================================================
print("\nSTEP 7: Model Coefficients (Top Features)...")

coef_df = pd.DataFrame({
    'Feature'    : glm_model.params.index,
    'Coefficient': glm_model.params.values,
    'P_Value'    : glm_model.pvalues.values
}).sort_values('P_Value')

print(f"\n{'FEATURE':<25} {'COEFFICIENT':>14} {'P_VALUE':>12} {'SIGNIFICANT':>12}")
print("-"*65)
for _, row in coef_df.head(10).iterrows():
    sig = 'YES ✅' if row['P_Value'] < 0.05 else 'NO'
    print(f"{row['Feature']:<25} {row['Coefficient']:>14.4f} "
          f"{row['P_Value']:>12.4f} {sig:>12}")

# ============================================================================
# STEP 8: Forecast Jan 2026
# ============================================================================
print("\nSTEP 8: Forecasting January 2026...")

routes_info = df[['Route_Id','Route_Name',
                   'Region','Territory',
                   'Region_Encoded']].drop_duplicates('Route_Id')

jan2026 = routes_info.copy()
jan2026['Month_Num']       = 1
jan2026['sin_month']       = np.sin(2 * np.pi * 1 / 12)
jan2026['cos_month']       = np.cos(2 * np.pi * 1 / 12)
jan2026['Eid_Indicator']   = 0
jan2026['Avg_Temperature'] = 12.0
jan2026['Season_Index']    = 1

last_data = df.groupby('Route_Id').last().reset_index()
jan2026   = jan2026.merge(
    last_data[['Route_Id','Time_Index',
               'Lag_Sales','MoM_Growth_Rate',
               'YoY_Growth_Rate',
               'Quarter_Growth_Rate']],
    on='Route_Id', how='left'
)
jan2026['Time_Index'] = jan2026['Time_Index'] + 1
jan2026 = jan2026.fillna(0)

X_jan        = sm.add_constant(jan2026[FEATURES], has_constant='add')
jan2026['GLM_Forecast']     = glm_model.predict(X_jan)
jan2026['GLM_Forecast']     = jan2026['GLM_Forecast'].clip(lower=0)
jan2026['GLM_Growth_25pct'] = (
    jan2026['GLM_Forecast'] * 1.25).round()

national_base   = jan2026['GLM_Forecast'].sum()
national_growth = jan2026['GLM_Growth_25pct'].sum()

print(f"National Base Forecast : {national_base:,.0f} units")
print(f"National Growth (+25%) : {national_growth:,.0f} units")

# ============================================================================
# STEP 9: Visualization
# ============================================================================
print("\nSTEP 9: Creating Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('GLM Poisson Forecasting Results\n'
             '(Aligned with Rostami-Tabar & Hyndman 2024)',
             fontsize=14, fontweight='bold')

# Plot 1: Actual vs Predicted scatter
sample = test_df2.sample(min(500, len(test_df2)),
                          random_state=42)
axes[0,0].scatter(sample[TARGET], sample['Pred'],
                  alpha=0.3, color='green', s=10)
max_val = max(sample[TARGET].max(),
              sample['Pred'].max())
axes[0,0].plot([0,max_val],[0,max_val],
               'r--', linewidth=2,
               label='Perfect Forecast')
axes[0,0].set_title('Actual vs Predicted Sales',
                    fontweight='bold')
axes[0,0].set_xlabel('Actual Sales')
axes[0,0].set_ylabel('Predicted Sales')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Regional MAPE
colors = ['green' if m<=20 else 'blue' if m<=40 else
          'orange' if m<=60 else 'red'
          for m in reg_acc.sort_values('MAPE')['MAPE']]
axes[0,1].barh(reg_acc.sort_values('MAPE')['Region'],
               reg_acc.sort_values('MAPE')['MAPE'],
               color=colors, alpha=0.8)
axes[0,1].axvline(x=mape, color='red',
                  linestyle='--', linewidth=2,
                  label=f'Median: {mape:.1f}%')
axes[0,1].set_title('Regional MAPE',
                    fontweight='bold')
axes[0,1].set_xlabel('MAPE %')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3, axis='x')

# Plot 3: Top coefficients
top_coef = coef_df[coef_df['Feature']!='const'].head(8)
colors3  = ['green' if v > 0 else 'red'
             for v in top_coef['Coefficient']]
axes[1,0].barh(top_coef['Feature'],
               top_coef['Coefficient'],
               color=colors3, alpha=0.8)
axes[1,0].set_title('Top Feature Coefficients',
                    fontweight='bold')
axes[1,0].set_xlabel('Coefficient Value')
axes[1,0].axvline(x=0, color='black',
                  linewidth=1)
axes[1,0].grid(True, alpha=0.3, axis='x')

# Plot 4: All Models Comparison
models     = ['Moving\nAverage','HW\nRoute',
              'HW\nRegional','Bottom-Up\nHW',
              'Random\nForest','XGBoost',
              'LSTM','GLM\nPoisson']
mapes      = [101.06, 64.26, 26.08, 12.91,
              1.55,   4.55,  45.24, mape]
bar_colors = ['red','orange','green','darkgreen',
              'steelblue','coral','purple','teal']
bars = axes[1,1].bar(models, mapes,
                      color=bar_colors,
                      alpha=0.8, edgecolor='white')
for bar, val in zip(bars, mapes):
    axes[1,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{val:.1f}%', ha='center',
        fontweight='bold', fontsize=7)
axes[1,1].set_title('All 8 Models Comparison (MAPE)',
                    fontweight='bold')
axes[1,1].set_ylabel('MAPE % (lower is better)')
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\glm_poisson_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 10: Save Results
# ============================================================================
print("\nSTEP 10: Saving Results...")

output = r'C:\Users\User\Documents\FPY\glm_poisson_results.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:

    jan2026[['Route_Id','Route_Name','Region',
             'Territory','GLM_Forecast',
             'GLM_Growth_25pct']].sort_values(
        ['Region','Route_Id']
    ).to_excel(writer,
               sheet_name='Jan2026 Forecasts',
               index=False)

    reg_acc.to_excel(writer,
                     sheet_name='Regional Accuracy',
                     index=False)

    coef_df.to_excel(writer,
                     sheet_name='Coefficients',
                     index=False)

    pd.DataFrame([
        {'Model':'Moving Average',
         'Type':'Statistical',
         'MAPE':101.06,'Accuracy':18.94,'R2':'---'},
        {'Model':'HW Route',
         'Type':'Statistical',
         'MAPE':64.26,'Accuracy':35.74,'R2':'---'},
        {'Model':'HW Regional',
         'Type':'Statistical',
         'MAPE':26.08,'Accuracy':73.92,'R2':'---'},
        {'Model':'Bottom-Up HW',
         'Type':'Statistical',
         'MAPE':12.91,'Accuracy':87.09,'R2':'---'},
        {'Model':'Random Forest',
         'Type':'Machine Learning',
         'MAPE':1.55,'Accuracy':98.45,'R2':0.9765},
        {'Model':'XGBoost',
         'Type':'Machine Learning',
         'MAPE':4.55,'Accuracy':95.45,'R2':0.9366},
        {'Model':'LSTM',
         'Type':'Deep Learning',
         'MAPE':45.24,'Accuracy':54.76,'R2':0.5610},
        {'Model':'GLM Poisson',
         'Type':'Statistical ML',
         'MAPE':round(mape,2),
         'Accuracy':round(100-mape,2),
         'R2':round(r2,4)},
    ]).to_excel(writer,
                sheet_name='All Models Comparison',
                index=False)

print(f"Saved: {output}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("GLM POISSON MODEL COMPLETE")
print("="*80)
print(f"\nGLM Poisson Accuracy:")
print(f"  MAPE : {mape:.2f}%")
print(f"  MAE  : {mae:,.2f} units")
print(f"  RMSE : {rmse:,.2f} units")
print(f"  MASE : {mase:.4f}")
print(f"  R2   : {r2:.4f}")
print(f"\nAll 8 Models Final Comparison:")
print(f"  1. Moving Average  : 101.06%  Statistical")
print(f"  2. HW Route        :  64.26%  Statistical")
print(f"  3. HW Regional     :  26.08%  Statistical")
print(f"  4. Bottom-Up HW    :  12.91%  Statistical")
print(f"  5. Random Forest   :   1.55%  Machine Learning")
print(f"  6. XGBoost         :   4.55%  Machine Learning")
print(f"  7. LSTM            :  45.24%  Deep Learning")
print(f"  8. GLM Poisson     : {mape:>7.2f}%  Statistical ML")
print(f"\nNational Jan 2026 Forecast:")
print(f"  Base   : {national_base:,.0f} units")
print(f"  +25%   : {national_growth:,.0f} units")
print(f"\nReference: Rostami-Tabar & Hyndman (2024)")
print(f"Files saved:")
print(f"  1. glm_poisson_results.xlsx")
print(f"  2. glm_poisson_visualization.png")
print("="*80)
