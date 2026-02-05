# Bank Loan Default Prediction (India)

**Ishak Islam** | UMID28072552431 | Unified Mentor Internship

## About

Analysis of bank loan default patterns using borrower data to understand factors that contribute to loan defaults. This project examines demographic, financial, and loan characteristics to identify risk factors and help financial institutions make better lending decisions.

## How to Run

```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_loan_default_analysis.ipynb
```

Run all cells to see the analysis.

## Dataset

Download from: https://www.kaggle.com/datasets/yasserh/loan-default-dataset

Place the downloaded CSV file in the `data/` folder:
- `Loan_Default.csv` - Main dataset with borrower and loan information

## Files

```
├── data/           # Put dataset files here
├── notebooks/      # Analysis notebook
├── scripts/        # Helper functions
├── visualizations/ # Charts
├── tableau/        # Tableau exports
└── docs/           # Documentation
```

## Results

- Loan default rate analysis by borrower demographics
- Income and employment impact on default risk
- Loan amount and term analysis
- Credit score and financial behavior patterns
- Regional default distribution
- Data exports ready for Tableau dashboards

## Tableau Dashboard

**Live Interactive Dashboard:** [View on Tableau Public](https://public.tableau.com/app/profile/ishak.islam/viz/BankLoanDefaultAnalysisIndia/Dashboard)

## Tech Stack

Python, Pandas, NumPy, Matplotlib, Seaborn, Tableau

## GitHub Repository

**Source Code:** [https://github.com/isacmj7/bank-loan-default-prediction](https://github.com/isacmj7/bank-loan-default-prediction)
