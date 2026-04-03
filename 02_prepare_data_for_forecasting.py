import pandas as pd
import sqlite3
import numpy as np

print("="*80)
print("PREPARING DATA FOR FORECASTING - ENHANCED WITH GROWTH METRICS")
print("="*80)

# Connect to database
db_path = r'C:\Users\User\Documents\FPY\sales_data.db'
conn = sqlite3.connect(db_path)

# Load data
df = pd.read_sql_query("SELECT * FROM sales_data", conn)
conn.close()

print(f"✅ Loaded {len(df):,} rows from database")

# ============================================================================
# STEP 1: Create Date Column
# ============================================================================
print("\n" + "="*80)
print("STEP 1: CREATING DATE COLUMN")
print("="*80)

month_map = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
             'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}

df['Month_Num'] = df['Month'].map(month_map)
df['Date'] = pd.to_datetime({'year': df['Year'], 'month': df['Month_Num'], 'day': 1})

print(f"✅ Date column created")
print(f"   Date range: {df['Date'].min()} to {df['Date'].max()}")

# ============================================================================
# STEP 2: Add External Variables - Eid Indicator
# ============================================================================
print("\n" + "="*80)
print("STEP 2: ADDING EID INDICATOR")
print("="*80)

eid_months = [
    ('2023', 'Apr'),
    ('2024', 'Apr'),
    ('2025', 'Mar')
]

df['Eid_Indicator'] = 0
for year, month in eid_months:
    mask = (df['Year'] == int(year)) & (df['Month'] == month)
    df.loc[mask, 'Eid_Indicator'] = 1

eid_count = df['Eid_Indicator'].sum()
print(f"✅ Eid_Indicator created")
print(f"   {eid_count:,} rows marked as Eid months")

# ============================================================================
# STEP 3: Add External Variables - Average Temperature
# ============================================================================
print("\n" + "="*80)
print("STEP 3: ADDING AVERAGE TEMPERATURE")
print("="*80)

temp_data = {
    1: 12.0, 2: 15.0, 3: 21.0, 4: 27.0, 5: 32.0, 6: 35.5,
    7: 34.0, 8: 32.5, 9: 30.0, 10: 24.5, 11: 18.0, 12: 13.0
}

df['Avg_Temperature'] = df['Month_Num'].map(temp_data)

print(f"✅ Avg_Temperature created")
print(f"   Range: {df['Avg_Temperature'].min():.1f}°C to {df['Avg_Temperature'].max():.1f}°C")

# ============================================================================
# STEP 4: Calculate Total Sales per Route per Month
# ============================================================================
print("\n" + "="*80)
print("STEP 4: CALCULATING TOTAL SALES")
print("="*80)

sku_cols = [col for col in df.columns if any(x in col for x in ['CSD', 'JUICE', 'WATER'])]
df['Sales_Quantity'] = df[sku_cols].sum(axis=1)

print(f"✅ Sales_Quantity created (sum of {len(sku_cols)} SKU columns)")
print(f"   Average sales per route: {df['Sales_Quantity'].mean():,.0f} units")

# ============================================================================
# STEP 5: Create Growth Variables (3 Types)
# ============================================================================
print("\n" + "="*80)
print("STEP 5: CREATING GROWTH VARIABLES")
print("="*80)

# Sort by Route and Date
df = df.sort_values(['Route_Id', 'Date'])

# 5A: Lag Sales (Previous Month)
df['Lag_Sales'] = df.groupby('Route_Id')['Sales_Quantity'].shift(1)

# 5B: Month-over-Month Growth Rate
df['MoM_Growth_Rate'] = ((df['Sales_Quantity'] - df['Lag_Sales']) / df['Lag_Sales'] * 100)
df['MoM_Growth_Rate'] = df['MoM_Growth_Rate'].replace([np.inf, -np.inf], 0)

# 5C: Year-over-Year Growth Rate (12 months lag)
df['Sales_12M_Ago'] = df.groupby('Route_Id')['Sales_Quantity'].shift(12)
df['YoY_Growth_Rate'] = ((df['Sales_Quantity'] - df['Sales_12M_Ago']) / df['Sales_12M_Ago'] * 100)
df['YoY_Growth_Rate'] = df['YoY_Growth_Rate'].replace([np.inf, -np.inf], 0)

# 5D: 3-Month Rolling Average Growth
df['Sales_3M_Avg'] = df.groupby('Route_Id')['Sales_Quantity'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)
df['Quarter_Growth_Rate'] = ((df['Sales_Quantity'] - df['Sales_3M_Avg']) / df['Sales_3M_Avg'] * 100)
df['Quarter_Growth_Rate'] = df['Quarter_Growth_Rate'].replace([np.inf, -np.inf], 0)

print(f"✅ Growth variables created:")
print(f"   1. Lag_Sales (previous month)")
print(f"   2. MoM_Growth_Rate (month-over-month %)")
print(f"   3. YoY_Growth_Rate (year-over-year %)")
print(f"   4. Quarter_Growth_Rate (3-month rolling %)")

# ============================================================================
# STEP 6: Create Season Index
# ============================================================================
print("\n" + "="*80)
print("STEP 6: CREATING SEASON INDEX")
print("="*80)

season_map = {
    1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3,
    7: 3, 8: 3, 9: 4, 10: 4, 11: 1, 12: 1
}

df['Season_Index'] = df['Month_Num'].map(season_map)

print(f"✅ Season_Index created")
print(f"   1=Winter, 2=Spring, 3=Summer, 4=Fall")

# ============================================================================
# STEP 7: Select Final Variables
# ============================================================================
print("\n" + "="*80)
print("STEP 7: SELECTING FINAL VARIABLES")
print("="*80)

final_cols = [
    # Core Time Variables
    'Year',
    'Month',
    'Date',
    
    # Hierarchical Sales Structure
    'Region',
    'Territory',
    'Route_Id',
    'Route_Name',
    
    # Target Variable
    'Sales_Quantity',
    
    # External Variables
    'Eid_Indicator',
    'Avg_Temperature',
    'Season_Index',
    
    # Derived Variables - Growth Metrics
    'Lag_Sales',
    'MoM_Growth_Rate',
    'YoY_Growth_Rate',
    'Quarter_Growth_Rate'
]

df_final = df[final_cols].copy()

print(f"✅ Final dataset created")
print(f"   Variables: {len(final_cols)}")
print(f"\nVariable list:")
for i, col in enumerate(final_cols, 1):
    print(f"   {i:2}. {col}")

# ============================================================================
# STEP 8: Clean Data
# ============================================================================
print("\n" + "="*80)
print("STEP 8: CLEANING DATA")
print("="*80)

before_clean = len(df_final)
df_clean = df_final.dropna(subset=['Lag_Sales', 'MoM_Growth_Rate'])
after_clean = len(df_clean)

print(f"✅ Removed {before_clean - after_clean:,} rows with missing values")
print(f"   Final clean dataset: {after_clean:,} rows")

# ============================================================================
# STEP 9: Save to CSV
# ============================================================================
print("\n" + "="*80)
print("STEP 9: SAVING PREPARED DATA")
print("="*80)

output_file = r'C:\Users\User\Documents\FPY\prepared_forecast_data.csv'
df_clean.to_csv(output_file, index=False)

print(f"✅ Data saved to: {output_file}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("DATA PREPARATION COMPLETE - ENHANCED VERSION")
print("="*80)

print(f"\n📊 Summary:")
print(f"   Original rows: {len(df):,}")
print(f"   Final clean rows: {len(df_clean):,}")
print(f"   Total variables: {len(final_cols)}")
print(f"   Date range: {df_clean['Date'].min()} to {df_clean['Date'].max()}")
print(f"   Unique routes: {df_clean['Route_Id'].nunique():,}")

print(f"\n📈 Growth Variables Included:")
print(f"   • MoM_Growth_Rate (monthly momentum)")
print(f"   • YoY_Growth_Rate (annual trend)")
print(f"   • Quarter_Growth_Rate (smoothed 3-month trend)")

print("\n✅ Your enhanced dataset is ready for forecasting!")
print("="*80)