"""
Data processing for bank loan default analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path


def load_loan_data(filepath=None):
    """Load loan default dataset."""
    if filepath is None:
        project_root = Path(__file__).parent.parent
        filepath = project_root / "data" / "Loan_Default.csv"
    
    df = pd.read_csv(filepath)
    print(f"Loaded loan data: {len(df)} rows, {len(df.columns)} columns")
    return df


def clean_loan_data(df):
    """Clean and prepare loan data for analysis."""
    df_clean = df.copy()
    
    # Handle missing values in numeric columns
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Handle missing values in categorical columns
    cat_cols = df_clean.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna('Unknown')
    
    print(f"Cleaned data: {len(df_clean)} rows")
    return df_clean


def get_default_stats(df, target_col='Status'):
    """Calculate default statistics."""
    if target_col not in df.columns:
        # Try common alternatives
        for col in ['Status', 'Default', 'default', 'loan_status', 'TARGET']:
            if col in df.columns:
                target_col = col
                break
    
    if target_col not in df.columns:
        return {"error": "Target column not found"}
    
    total = len(df)
    defaults = df[target_col].sum() if df[target_col].dtype in ['int64', 'float64'] else (df[target_col] == 1).sum()
    
    return {
        'total_loans': total,
        'defaults': int(defaults),
        'non_defaults': int(total - defaults),
        'default_rate': round(defaults / total * 100, 2)
    }


def get_demographic_summary(df):
    """Get summary statistics by demographics."""
    summary = {}
    
    # Check for income column
    income_cols = [c for c in df.columns if 'income' in c.lower()]
    if income_cols:
        summary['avg_income'] = df[income_cols[0]].mean()
        summary['median_income'] = df[income_cols[0]].median()
    
    # Check for age column
    age_cols = [c for c in df.columns if 'age' in c.lower()]
    if age_cols:
        summary['avg_age'] = df[age_cols[0]].mean()
    
    # Check for loan amount column
    loan_cols = [c for c in df.columns if 'loan' in c.lower() and 'amount' in c.lower()]
    if loan_cols:
        summary['avg_loan_amount'] = df[loan_cols[0]].mean()
    
    return summary


def create_age_groups(df, age_col='age'):
    """Create age group categories."""
    if age_col not in df.columns:
        for col in df.columns:
            if 'age' in col.lower():
                age_col = col
                break
    
    if age_col not in df.columns:
        return df
    
    df_copy = df.copy()
    bins = [0, 25, 35, 45, 55, 65, 100]
    labels = ['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
    df_copy['Age_Group'] = pd.cut(df_copy[age_col], bins=bins, labels=labels)
    
    return df_copy


def create_income_groups(df, income_col='Income'):
    """Create income group categories."""
    if income_col not in df.columns:
        for col in df.columns:
            if 'income' in col.lower():
                income_col = col
                break
    
    if income_col not in df.columns:
        return df
    
    df_copy = df.copy()
    income_percentiles = df_copy[income_col].quantile([0.25, 0.5, 0.75])
    
    def categorize_income(x):
        if x <= income_percentiles[0.25]:
            return 'Low'
        elif x <= income_percentiles[0.5]:
            return 'Medium-Low'
        elif x <= income_percentiles[0.75]:
            return 'Medium-High'
        else:
            return 'High'
    
    df_copy['Income_Group'] = df_copy[income_col].apply(categorize_income)
    
    return df_copy


def export_for_tableau(df, output_dir=None, target_col='Status'):
    """Export processed data for Tableau."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "tableau"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    
    # Main data export (sampled for large datasets)
    if len(df) > 50000:
        df_sample = df.sample(n=50000, random_state=42)
    else:
        df_sample = df
    df_sample.to_csv(output_dir / "loan_data_tableau.csv", index=False)
    
    # Default rate by income group
    if 'Income_Group' in df.columns:
        income_summary = df.groupby('Income_Group').agg({
            target_col: ['count', 'sum', 'mean']
        }).round(4)
        income_summary.columns = ['Total_Loans', 'Defaults', 'Default_Rate']
        income_summary = income_summary.reset_index()
        income_summary.to_csv(output_dir / "default_by_income.csv", index=False)
    
    # Default rate by age group
    if 'Age_Group' in df.columns:
        age_summary = df.groupby('Age_Group').agg({
            target_col: ['count', 'sum', 'mean']
        }).round(4)
        age_summary.columns = ['Total_Loans', 'Defaults', 'Default_Rate']
        age_summary = age_summary.reset_index()
        age_summary.to_csv(output_dir / "default_by_age.csv", index=False)
    
    print(f"Exported data to {output_dir}")


if __name__ == "__main__":
    df = load_loan_data()
    df_clean = clean_loan_data(df)
    
    stats = get_default_stats(df_clean)
    print(f"\nDefault Statistics: {stats}")
