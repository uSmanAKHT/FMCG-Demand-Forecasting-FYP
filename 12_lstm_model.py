import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error

print("="*80)
print("LSTM DEEP LEARNING FORECASTING MODEL")
print("FYP - FMCG Demand Forecasting")
print(f"TensorFlow Version: {tf.__version__}")
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

TARGET = 'Sales_Quantity'

df = df.dropna(subset=FEATURES + [TARGET])
print(f"Features     : {len(FEATURES)}")
print(f"Clean rows   : {len(df):,}")

# ============================================================================
# STEP 3: Scale Data
# ============================================================================
print("\nSTEP 3: Scaling Data...")

scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(df[FEATURES])
y_scaled = scaler_y.fit_transform(
               df[[TARGET]]).flatten()

print("Data scaled to [0,1] range")

# ============================================================================
# STEP 4: Create Sequences for LSTM
# ============================================================================
print("\nSTEP 4: Creating LSTM Sequences...")

SEQ_LEN = 6  # Use 6 months to predict next month

def create_sequences(X, y, seq_len):
    Xs, ys = [], []
    for i in range(len(X) - seq_len):
        Xs.append(X[i:i+seq_len])
        ys.append(y[i+seq_len])
    return np.array(Xs), np.array(ys)

X_seq, y_seq = create_sequences(X_scaled, y_scaled, SEQ_LEN)

print(f"Sequences created : {len(X_seq):,}")
print(f"Sequence shape    : {X_seq.shape}")

# ============================================================================
# STEP 5: Train/Test Split
# ============================================================================
print("\nSTEP 5: Train/Test Split...")

split      = int(len(X_seq) * 0.8)
X_train    = X_seq[:split]
X_test     = X_seq[split:]
y_train    = y_seq[:split]
y_test     = y_seq[split:]

print(f"Train size : {len(X_train):,} sequences")
print(f"Test size  : {len(X_test):,} sequences")

# ============================================================================
# STEP 6: Build LSTM Model
# ============================================================================
print("\nSTEP 6: Building LSTM Model...")

model = Sequential([
    LSTM(64, return_sequences=True,
         input_shape=(SEQ_LEN, len(FEATURES))),
    Dropout(0.2),
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(
    optimizer = 'adam',
    loss      = 'mse'
)

model.summary()

# ============================================================================
# STEP 7: Train Model
# ============================================================================
print("\nSTEP 7: Training LSTM Model...")

early_stop = EarlyStopping(
    monitor  = 'val_loss',
    patience = 5,
    restore_best_weights = True
)

history = model.fit(
    X_train, y_train,
    epochs          = 50,
    batch_size      = 32,
    validation_split= 0.1,
    callbacks       = [early_stop],
    verbose         = 1
)

print("LSTM Model trained!")

# ============================================================================
# STEP 8: Predict and Calculate Accuracy
# ============================================================================
print("\nSTEP 8: Calculating Accuracy...")

y_pred_scaled = model.predict(X_test, verbose=0)
y_pred        = scaler_y.inverse_transform(
                    y_pred_scaled).flatten()
y_actual      = scaler_y.inverse_transform(
                    y_test.reshape(-1,1)).flatten()

y_pred   = np.maximum(0, y_pred)
mask     = y_actual > 0
y_act_f  = y_actual[mask]
y_pred_f = y_pred[mask]

mape = np.median(np.abs(
    (y_act_f - y_pred_f) / y_act_f) * 100)
mae  = mean_absolute_error(y_act_f, y_pred_f)
rmse = np.sqrt(mean_squared_error(y_act_f, y_pred_f))
naive= np.mean(np.abs(np.diff(
    scaler_y.inverse_transform(
        y_seq.reshape(-1,1)).flatten())))
mase = mae / naive if naive != 0 else np.nan
ss_r = np.sum((y_act_f - y_pred_f)**2)
ss_t = np.sum((y_act_f - np.mean(y_act_f))**2)
r2   = 1 - (ss_r/ss_t) if ss_t != 0 else np.nan

print(f"\n{'='*60}")
print(f"LSTM ACCURACY METRICS")
print(f"{'='*60}")
print(f"MAPE : {mape:.2f}%   (lower is better)")
print(f"MAE  : {mae:,.2f} units")
print(f"RMSE : {rmse:,.2f} units")
print(f"MASE : {mase:.4f}   (< 1.0 = better than naive)")
print(f"R2   : {r2:.4f}   (> 0.7 = strong model)")
print(f"{'='*60}")

# ============================================================================
# STEP 9: Visualization
# ============================================================================
print("\nSTEP 9: Creating Visualization...")

fig, axes = plt.subplots(2, 2, figsize=(16,12))
fig.suptitle('LSTM Deep Learning Forecasting Results',
             fontsize=16, fontweight='bold')

# Plot 1: Training Loss
axes[0,0].plot(history.history['loss'],
               label='Training Loss',
               color='steelblue', linewidth=2)
axes[0,0].plot(history.history['val_loss'],
               label='Validation Loss',
               color='coral', linewidth=2)
axes[0,0].set_title('LSTM Training Loss',
                    fontweight='bold')
axes[0,0].set_xlabel('Epoch')
axes[0,0].set_ylabel('Loss (MSE)')
axes[0,0].legend()
axes[0,0].grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted
sample_size = min(200, len(y_act_f))
axes[0,1].scatter(y_act_f[:sample_size],
                  y_pred_f[:sample_size],
                  alpha=0.3, color='purple', s=10)
max_val = max(y_act_f[:sample_size].max(),
              y_pred_f[:sample_size].max())
axes[0,1].plot([0,max_val],[0,max_val],
               'r--', linewidth=2,
               label='Perfect Forecast')
axes[0,1].set_title('Actual vs Predicted',
                    fontweight='bold')
axes[0,1].set_xlabel('Actual Sales')
axes[0,1].set_ylabel('Predicted Sales')
axes[0,1].legend()
axes[0,1].grid(True, alpha=0.3)

# Plot 3: Forecast vs Actual over time
axes[1,0].plot(y_act_f[:100],
               label='Actual', color='blue',
               linewidth=1.5)
axes[1,0].plot(y_pred_f[:100],
               label='LSTM Forecast',
               color='red', linewidth=1.5,
               linestyle='--')
axes[1,0].set_title('Actual vs LSTM Forecast (Sample)',
                    fontweight='bold')
axes[1,0].set_xlabel('Time Steps')
axes[1,0].set_ylabel('Sales Quantity')
axes[1,0].legend()
axes[1,0].grid(True, alpha=0.3)

# Plot 4: All Models Comparison
models     = ['Moving\nAverage','HW\nRoute',
              'HW\nRegional','Bottom-Up\nHW',
              'Random\nForest','XGBoost','LSTM']
mapes      = [101.06, 64.26, 26.08,
              12.91,  1.55,  4.55, mape]
bar_colors = ['red','orange','green',
              'darkgreen','steelblue',
              'coral','purple']
bars = axes[1,1].bar(models, mapes,
                      color=bar_colors,
                      alpha=0.8, edgecolor='white')
for bar, val in zip(bars, mapes):
    axes[1,1].text(
        bar.get_x() + bar.get_width()/2,
        bar.get_height() + 0.5,
        f'{val:.1f}%', ha='center',
        fontweight='bold', fontsize=8)
axes[1,1].set_title('All 7 Models Comparison (MAPE)',
                    fontweight='bold')
axes[1,1].set_ylabel('MAPE % (lower is better)')
axes[1,1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plot_file = r'C:\Users\User\Documents\FPY\lstm_visualization.png'
plt.savefig(plot_file, dpi=300, bbox_inches='tight')
print(f"Visualization saved: {plot_file}")

# ============================================================================
# STEP 10: Save Results
# ============================================================================
print("\nSTEP 10: Saving Results...")

output = r'C:\Users\User\Documents\FPY\lstm_results.xlsx'
with pd.ExcelWriter(output, engine='openpyxl') as writer:

    pd.DataFrame([
        {'Model':'Moving Average',  'Type':'Statistical',
         'MAPE':101.06,'Accuracy':18.94,'R2':'---'},
        {'Model':'HW Route',        'Type':'Statistical',
         'MAPE': 64.26,'Accuracy':35.74,'R2':'---'},
        {'Model':'HW Regional',     'Type':'Statistical',
         'MAPE': 26.08,'Accuracy':73.92,'R2':'---'},
        {'Model':'Bottom-Up HW',    'Type':'Statistical',
         'MAPE': 12.91,'Accuracy':87.09,'R2':'---'},
        {'Model':'Random Forest',   'Type':'Machine Learning',
         'MAPE':  1.55,'Accuracy':98.45,'R2':0.9765},
        {'Model':'XGBoost',         'Type':'Machine Learning',
         'MAPE':  4.55,'Accuracy':95.45,'R2':0.9366},
        {'Model':'LSTM',            'Type':'Deep Learning',
         'MAPE': round(mape,2),
         'Accuracy':round(100-mape,2),
         'R2':round(r2,4)},
    ]).to_excel(writer,
                sheet_name='All Models Comparison',
                index=False)

    pd.DataFrame({
        'Epoch'     : range(1, len(history.history['loss'])+1),
        'Train_Loss': history.history['loss'],
        'Val_Loss'  : history.history['val_loss']
    }).to_excel(writer,
                sheet_name='Training History',
                index=False)

print(f"Saved: {output}")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("LSTM DEEP LEARNING MODEL COMPLETE")
print("="*80)
print(f"\nLSTM Accuracy:")
print(f"  MAPE : {mape:.2f}%")
print(f"  MAE  : {mae:,.2f} units")
print(f"  RMSE : {rmse:,.2f} units")
print(f"  MASE : {mase:.4f}")
print(f"  R2   : {r2:.4f}")
print(f"\nAll 7 Models Final Comparison:")
print(f"  1. Moving Average  : 101.06%  Statistical")
print(f"  2. HW Route        :  64.26%  Statistical")
print(f"  3. HW Regional     :  26.08%  Statistical")
print(f"  4. Bottom-Up HW    :  12.91%  Statistical")
print(f"  5. Random Forest   :   1.55%  Machine Learning")
print(f"  6. XGBoost         :   4.55%  Machine Learning")
print(f"  7. LSTM            : {mape:>7.2f}%  Deep Learning")
print(f"\nFiles saved:")
print(f"  1. lstm_results.xlsx")
print(f"  2. lstm_visualization.png")
print("="*80)