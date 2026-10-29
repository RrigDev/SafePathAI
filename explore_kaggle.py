"""
explore_kaggle_data.py

Helps you understand the structure of the Kaggle Indian Crimes Dataset
and shows you how to map it to our app's expected format.

Usage:
    python explore_kaggle_data.py
"""

import pandas as pd
from pathlib import Path
import os

# Find the CSV file in the extracted folder
csv_files = list(Path('.').glob('crimes_indian_dataset/**/*.csv'))

if not csv_files:
    print("❌ No CSV files found in crimes_indian_dataset/")
    print("   Make sure you extracted the Kaggle ZIP file correctly.")
    exit(1)

print(f"Found CSV files: {[str(f) for f in csv_files]}\n")

for csv_path in csv_files:
    print(f"\n{'='*60}")
    print(f"📄 File: {csv_path}")
    print(f"{'='*60}")
    
    df = pd.read_csv(csv_path)
    
    print(f"\nShape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\nColumns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    print(f"\nData types:")
    print(df.dtypes)
    
    print(f"\nMissing values:")
    print(df.isnull().sum())
    
    # Try to identify useful columns
    print(f"\n🔍 Potential useful columns:")
    for col in df.columns:
        if 'district' in col.lower() or 'location' in col.lower() or 'city' in col.lower():
            print(f"   - {col} (location): {df[col].nunique()} unique values")
        elif 'state' in col.lower():
            print(f"   - {col} (state): {df[col].nunique()} unique values")
        elif any(crime in col.lower() for crime in ['theft', 'robbery', 'vehicle', 'harassment', 'crime', 'count', 'total']):
            print(f"   - {col} (crime): {df[col].dtype}")
    
    print(f"\n💡 Next step: Use this info to map columns to our format:")
    print("   Our app needs: district, state, theft, robbery, vehicle_theft, harassment")
    print("   Create a mapping in clean_kaggle_data.py based on actual column names above.")

print("\n" + "="*60)
print("✅ Review output above, then run: python clean_kaggle_data.py")
print("="*60)