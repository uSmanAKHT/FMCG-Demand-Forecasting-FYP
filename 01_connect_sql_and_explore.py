import pandas as pd
import sqlite3

print("="*80)
print("CONNECTING TO SQL DATABASE")
print("="*80)

# Database connection
db_path = r'C:\Users\User\Documents\FPY\sales_data.db'
conn = sqlite3.connect(db_path)

print(f"✅ Connected to database: {db_path}")

# Load all data
df = pd.read_sql_query("SELECT * FROM sales_data", conn)

print(f"✅ Data loaded successfully")
print(f"   Total rows: {len(df):,}")
print(f"   Total columns: {len(df.columns)}")

# Show first 5 rows
print("\nFirst 5 rows of data:")
print(df[['Year', 'Month', 'Region', 'Zone', 'Route_Name']].head())

# Close connection
conn.close()
print("\n✅ DATABASE CONNECTION CLOSED")