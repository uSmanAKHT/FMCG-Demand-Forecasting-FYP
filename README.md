 Hierarchical Demand Forecasting for FMCG

 Project Overview
Final Year Project — MS Business Analytics, FAST-NUCES (2024–2025)
Real FMCG sales data from Gourmet Pakistan national distribution network.

 Dataset
- Records: 83,450 rows
- Routes: 2,645
- Regions: 14
- SKUs: 35 products
- Date Range: 2023–2025

 Models Built & Results
| Model | MAPE | Accuracy |
|-------|------|----------|
| Moving Average (baseline) | 101.06% | 18.94% |
| Holt-Winters Route | 64.26% | 35.74% |
| GLM Poisson | 59.49% | 40.51% |
| LSTM | 45.24% | 54.76% |
| Holt-Winters Regional | 26.08% | 73.92% |
| Bottom-Up Hierarchical | 12.91% | 87.09% |
| XGBoost | 4.55% | 95.45% |
| Random Forest (best) | 1.55% | 98.45% |

Key Achievement
98.45% accuracy — 88% improvement over baseline

 Tools & Technologies
Python | pandas | NumPy | scikit-learn | SQLite | SQL | matplotlib

## Reference
Rostami-Tabar & Hyndman (2024) — Journal of Service Research
