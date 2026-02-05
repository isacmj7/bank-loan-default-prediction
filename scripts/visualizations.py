"""
Visualizations for bank loan default analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

COLORS = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22', '#34495e']


def save_fig(fig, filename, output_dir=None):
    """Save figure to file."""
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "visualizations"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(exist_ok=True)
    fig.savefig(output_dir / filename, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plot_default_distribution(df, target_col='Status', output_dir=None):
    """Plot loan default distribution."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Count plot
    counts = df[target_col].value_counts()
    labels = ['No Default', 'Default']
    colors = [COLORS[2], COLORS[1]]
    
    axes[0].bar(labels, counts.values, color=colors)
    axes[0].set_xlabel('Loan Status')
    axes[0].set_ylabel('Number of Loans')
    axes[0].set_title('Loan Default Distribution')
    
    for i, v in enumerate(counts.values):
        axes[0].text(i, v + 500, f'{v:,}', ha='center', fontsize=11, fontweight='bold')
    
    # Pie chart
    axes[1].pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors, 
                explode=[0, 0.05], startangle=90)
    axes[1].set_title('Default Rate Distribution')
    
    plt.tight_layout()
    save_fig(fig, '01_default_distribution.png', output_dir)


def plot_income_analysis(df, income_col, target_col='Status', output_dir=None):
    """Plot income vs default analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Box plot: Income by default status
    df_plot = df[[income_col, target_col]].dropna()
    df_plot['Status_Label'] = df_plot[target_col].map({0: 'No Default', 1: 'Default'})
    
    colors_box = [COLORS[2], COLORS[1]]
    box = axes[0].boxplot([df_plot[df_plot[target_col] == 0][income_col],
                           df_plot[df_plot[target_col] == 1][income_col]],
                          labels=['No Default', 'Default'],
                          patch_artist=True)
    
    for patch, color in zip(box['boxes'], colors_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    axes[0].set_ylabel('Income')
    axes[0].set_title('Income Distribution by Loan Status')
    
    # Income group analysis
    if 'Income_Group' in df.columns:
        income_default = df.groupby('Income_Group')[target_col].mean() * 100
        income_order = ['Low', 'Medium-Low', 'Medium-High', 'High']
        income_default = income_default.reindex([g for g in income_order if g in income_default.index])
        
        bars = axes[1].bar(income_default.index, income_default.values, color=COLORS[0])
        axes[1].set_xlabel('Income Group')
        axes[1].set_ylabel('Default Rate (%)')
        axes[1].set_title('Default Rate by Income Group')
        
        for bar, val in zip(bars, income_default.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.5, f'{val:.1f}%', 
                        ha='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, '02_income_analysis.png', output_dir)


def plot_loan_analysis(df, loan_col, target_col='Status', output_dir=None):
    """Plot loan amount analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Loan amount distribution
    df_no_default = df[df[target_col] == 0][loan_col]
    df_default = df[df[target_col] == 1][loan_col]
    
    axes[0].hist(df_no_default, bins=50, alpha=0.7, label='No Default', color=COLORS[2])
    axes[0].hist(df_default, bins=50, alpha=0.7, label='Default', color=COLORS[1])
    axes[0].set_xlabel('Loan Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title('Loan Amount Distribution by Status')
    axes[0].legend()
    
    # Average loan by status
    avg_by_status = df.groupby(target_col)[loan_col].mean()
    labels = ['No Default', 'Default']
    colors = [COLORS[2], COLORS[1]]
    
    bars = axes[1].bar(labels, avg_by_status.values, color=colors)
    axes[1].set_ylabel('Average Loan Amount')
    axes[1].set_title('Average Loan Amount by Status')
    
    for bar, val in zip(bars, avg_by_status.values):
        axes[1].text(bar.get_x() + bar.get_width()/2, val + 1000, f'{val:,.0f}', 
                    ha='center', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, '03_loan_analysis.png', output_dir)


def plot_demographic_analysis(df, target_col='Status', output_dir=None):
    """Plot demographic analysis."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Age group analysis
    if 'Age_Group' in df.columns:
        age_default = df.groupby('Age_Group')[target_col].agg(['count', 'mean'])
        age_default['default_rate'] = age_default['mean'] * 100
        
        bars = axes[0, 0].bar(age_default.index.astype(str), age_default['default_rate'], color=COLORS[0])
        axes[0, 0].set_xlabel('Age Group')
        axes[0, 0].set_ylabel('Default Rate (%)')
        axes[0, 0].set_title('Default Rate by Age Group')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        for bar, val in zip(bars, age_default['default_rate']):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, val + 0.3, f'{val:.1f}%', 
                           ha='center', fontsize=9, fontweight='bold')
    
    # Loan count by age group
    if 'Age_Group' in df.columns:
        age_count = df['Age_Group'].value_counts().sort_index()
        colors_age = [COLORS[i % len(COLORS)] for i in range(len(age_count))]
        
        axes[0, 1].bar(age_count.index.astype(str), age_count.values, color=colors_age)
        axes[0, 1].set_xlabel('Age Group')
        axes[0, 1].set_ylabel('Number of Loans')
        axes[0, 1].set_title('Loan Distribution by Age Group')
        axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Employment type analysis (if available)
    emp_cols = [c for c in df.columns if 'employ' in c.lower() or 'job' in c.lower() or 'occupation' in c.lower()]
    if emp_cols:
        emp_col = emp_cols[0]
        emp_default = df.groupby(emp_col)[target_col].mean() * 100
        emp_default = emp_default.sort_values(ascending=False).head(10)
        
        axes[1, 0].barh(range(len(emp_default)), emp_default.values, color=COLORS[3])
        axes[1, 0].set_yticks(range(len(emp_default)))
        axes[1, 0].set_yticklabels(emp_default.index)
        axes[1, 0].set_xlabel('Default Rate (%)')
        axes[1, 0].set_title(f'Default Rate by {emp_col}')
        axes[1, 0].invert_yaxis()
    else:
        axes[1, 0].text(0.5, 0.5, 'Employment data not available', ha='center', va='center')
        axes[1, 0].axis('off')
    
    # Property/Region analysis (if available)
    region_cols = [c for c in df.columns if 'region' in c.lower() or 'state' in c.lower() or 'property' in c.lower()]
    if region_cols:
        region_col = region_cols[0]
        region_default = df.groupby(region_col)[target_col].mean() * 100
        region_default = region_default.sort_values(ascending=False).head(10)
        
        axes[1, 1].barh(range(len(region_default)), region_default.values, color=COLORS[4])
        axes[1, 1].set_yticks(range(len(region_default)))
        axes[1, 1].set_yticklabels(region_default.index)
        axes[1, 1].set_xlabel('Default Rate (%)')
        axes[1, 1].set_title(f'Default Rate by {region_col}')
        axes[1, 1].invert_yaxis()
    else:
        axes[1, 1].text(0.5, 0.5, 'Region data not available', ha='center', va='center')
        axes[1, 1].axis('off')
    
    plt.tight_layout()
    save_fig(fig, '04_demographic_analysis.png', output_dir)


def plot_credit_analysis(df, target_col='Status', output_dir=None):
    """Plot credit score and financial behavior analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Credit score analysis
    credit_cols = [c for c in df.columns if 'credit' in c.lower() and 'score' in c.lower()]
    if credit_cols:
        credit_col = credit_cols[0]
        
        df_no_default = df[df[target_col] == 0][credit_col].dropna()
        df_default = df[df[target_col] == 1][credit_col].dropna()
        
        axes[0].hist(df_no_default, bins=30, alpha=0.7, label='No Default', color=COLORS[2])
        axes[0].hist(df_default, bins=30, alpha=0.7, label='Default', color=COLORS[1])
        axes[0].set_xlabel('Credit Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Credit Score Distribution by Loan Status')
        axes[0].legend()
    else:
        axes[0].text(0.5, 0.5, 'Credit score data not available', ha='center', va='center')
        axes[0].axis('off')
    
    # Interest rate analysis
    rate_cols = [c for c in df.columns if 'rate' in c.lower() or 'interest' in c.lower()]
    if rate_cols:
        rate_col = rate_cols[0]
        
        rate_by_status = df.groupby(target_col)[rate_col].mean()
        labels = ['No Default', 'Default']
        colors = [COLORS[2], COLORS[1]]
        
        bars = axes[1].bar(labels, rate_by_status.values, color=colors)
        axes[1].set_ylabel('Average Interest Rate (%)')
        axes[1].set_title('Average Interest Rate by Loan Status')
        
        for bar, val in zip(bars, rate_by_status.values):
            axes[1].text(bar.get_x() + bar.get_width()/2, val + 0.1, f'{val:.2f}%', 
                        ha='center', fontsize=11, fontweight='bold')
    else:
        axes[1].text(0.5, 0.5, 'Interest rate data not available', ha='center', va='center')
        axes[1].axis('off')
    
    plt.tight_layout()
    save_fig(fig, '05_credit_analysis.png', output_dir)


def plot_correlation_matrix(df, target_col='Status', output_dir=None):
    """Plot correlation matrix for numeric features."""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Select numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Limit to top correlated features with target
    if len(numeric_cols) > 15:
        correlations = df[numeric_cols].corr()[target_col].abs().sort_values(ascending=False)
        top_features = correlations.head(15).index.tolist()
        numeric_cols = top_features
    
    corr_matrix = df[numeric_cols].corr()
    
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                fmt='.2f', linewidths=0.5, ax=ax, annot_kws={'size': 8})
    
    ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    save_fig(fig, '06_correlation_matrix.png', output_dir)


def create_all_visualizations(df, target_col='Status', output_dir=None):
    """Create all analysis charts."""
    print("Creating visualizations...")
    
    # Find columns
    income_cols = [c for c in df.columns if 'income' in c.lower()]
    loan_cols = [c for c in df.columns if 'loan' in c.lower() and ('amount' in c.lower() or 'amt' in c.lower())]
    
    income_col = income_cols[0] if income_cols else None
    loan_col = loan_cols[0] if loan_cols else None
    
    plot_default_distribution(df, target_col, output_dir)
    
    if income_col:
        plot_income_analysis(df, income_col, target_col, output_dir)
    
    if loan_col:
        plot_loan_analysis(df, loan_col, target_col, output_dir)
    
    plot_demographic_analysis(df, target_col, output_dir)
    plot_credit_analysis(df, target_col, output_dir)
    plot_correlation_matrix(df, target_col, output_dir)
    
    print("Visualizations complete!")


if __name__ == "__main__":
    from data_processing import load_loan_data, clean_loan_data, create_age_groups, create_income_groups
    
    df = load_loan_data()
    df_clean = clean_loan_data(df)
    df_clean = create_age_groups(df_clean)
    df_clean = create_income_groups(df_clean)
    
    create_all_visualizations(df_clean)
