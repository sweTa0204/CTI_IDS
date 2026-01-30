#!/usr/bin/env python3
"""
Dataset Analysis Script
Analyzes current dataset structure to understand what needs encoding and processing
"""

import pandas as pd
import numpy as np
import os

def main():
    # Load dataset
    data_path = '../data/dos_detection_dataset.csv'
    df = pd.read_csv(data_path)
    
    print('=== CURRENT DATASET ANALYSIS ===')
    print(f'Shape: {df.shape}')
    print(f'Total columns: {len(df.columns)}')
    print(f'Columns: {list(df.columns)}')
    print()

    # Check data types
    print('=== DATA TYPES SUMMARY ===')
    print(df.dtypes.value_counts())
    print()

    # Check text features specifically
    print('=== TEXT FEATURES TO ENCODE ===')
    text_columns = df.select_dtypes(include=['object']).columns
    print(f'Text columns found: {len(text_columns)}')
    
    for col in text_columns:
        unique_vals = df[col].unique()
        print(f'{col}: {len(unique_vals)} unique values')
        print(f'  Values: {list(unique_vals)}')
        print(f'  Value counts:')
        print(df[col].value_counts())
        print()

    # Check numeric features
    print('=== NUMERIC FEATURES ===')
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f'Numeric columns count: {len(numeric_cols)}')
    print(f'First 10 numeric columns: {list(numeric_cols)[:10]}')
    
    # Check for missing values
    print()
    print('=== MISSING VALUES ===')
    missing_total = df.isnull().sum().sum()
    print(f'Total missing values: {missing_total}')
    
    if missing_total > 0:
        print('Missing values by column:')
        missing_cols = df.isnull().sum()
        print(missing_cols[missing_cols > 0])

    # Check target variable
    print()
    print('=== TARGET VARIABLES ===')
    if 'label' in df.columns:
        print('Label distribution:')
        print(df['label'].value_counts())
    
    if 'attack_cat' in df.columns:
        print('Attack category distribution:')
        print(df['attack_cat'].value_counts())

if __name__ == "__main__":
    main()
