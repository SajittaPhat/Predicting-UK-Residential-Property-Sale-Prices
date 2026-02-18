# Predicting-UK-Residential-Property-Sale-Prices

**A Machine Learning Analysis of HM Land Registry Price Paid Data (1995–2025)**

This repository contains the complete code and analysis for predicting UK residential property sale prices using ensemble machine learning methods. The project compares four models — Linear Regression, Decision Tree, Random Forest, and XGBoost — on 30.9 million historical transactions.

---

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Key Findings](#key-findings)
- [Repository Structure](#repository-structure)
- [Installation & Setup](#installation--setup)
- [Data Download](#data-download)
- [Reproduction Instructions](#reproduction-instructions)
- [Model Performance](#model-performance)
- [Dependencies](#dependencies)
- [Citation](#citation)
- [License](#license)

---

## 🎯 Project Overview

**Research Question:**  
Can we predict UK residential property sale prices using property characteristics (type, new-build status), location factors (district, county, town/city), and transaction timing?

**Dataset:**  
HM Land Registry Price Paid Data — all completed residential property transactions in England and Wales from 1995 to 2025 (30,906,560 raw records).

**Outcome Variable:**  
Natural log-transformed sale price (£)

**Best Model:**  
Random Forest Regressor (R² = 0.6945, MAPE = 2.54%)

---

## 🏆 Key Findings

1. **Location dominates price** — District, County, and Town/City combined account for ~35.8% of feature importance
2. **Temporal trend is strongest** — Year accounts for 52.9% of importance (30-year house price inflation 1995–2025)
3. **Property type matters** — Flats and detached houses command different premiums
4. **New-build premium exists** — Old/New indicator contributes meaningful signal
5. **Seasonality is minor** — Month has minimal predictive power

**Random Forest outperformed XGBoost** by +2.58 percentage points on R² (log-price), demonstrating that variance reduction (bagging) is more effective than bias reduction (boosting) for this dataset.

---

## 🛠️ Installation & Setup

### Prerequisites

- Python 3.8 or higher
- Jupyter Notebook or JupyterLab
- 16GB+ RAM recommended (dataset is large)
- 5GB+ free disk space

### Step 1: Clone Repository

```bash
git clone https://github.com/SajittaPhat/Predicting-UK-Residential-Property-Sale-Prices.git
cd Predicting-UK-Residential-Property-Sale-Prices
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

**What gets installed:**
- `numpy` — Array operations
- `pandas` — Data manipulation
- `scikit-learn` — ML models (Random Forest, Linear Regression, Decision Tree)
- `xgboost` — Gradient boosting
- `matplotlib` — Visualization
- `seaborn` — Statistical plotting
- `jupyter` — Notebook environment

---

## 📊 Data Download

The dataset is **NOT included** in this repository due to its size (~2GB CSV file). You must download it separately.

### Download Instructions

1. **Visit the official data source:**  
   [HM Land Registry Price Paid Data](https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads)
   (http://prod.publicdata.landregistry.gov.uk.s3-website-eu-west-1.amazonaws.com/pp-complete.csv)

2. **Data Description**
  These include standard and additional price paid data transactions received at HM Land Registry from 1 January 1995 to the most current monthly data (end of 2025).


3.**Download the complete dataset:**
   - Click on **"Price Paid Data (complete file)"**
   - Download: `pp-complete.csv` (approximately 5GB)

5. **Place the file in the data folder:**
   ```bash
   mkdir -p data
   mv ~/Downloads/pp-complete.csv data/
   ```

6. **Verify the file:**
   ```bash
   # Should show approximately 30.9M rows
   wc -l data/pp-complete.csv
   ```
```

Then modify **Cell 2** in the notebook:
```python
# Change this line:
df_all = pd.read_csv("pp-complete.csv", header=None, names=columns)

# To this:
df_all = pd.read_csv("pp-sample.csv", header=None, names=columns)
```

⚠️ **Note:** Results will differ with sample data. The full dataset is required to reproduce the exact results in the report.

---

## 🔄 Reproduction Instructions

Follow these steps to reproduce all results from the analysis:

### Step 1: Launch Jupyter Notebook

```bash
jupyter notebook data_preprocessing_eda_v2.ipynb
```

Your browser will open with the notebook.

### Step 2: Run All Cells

**Option A: Run all at once**
- Click **Kernel → Restart & Run All**
- Wait approximately 30-45 minutes (depends on your CPU)

**Option B: Run cell by cell**
- Click on the first cell
- Press **Shift + Enter** to run each cell sequentially
- Recommended for understanding each step

### Step 3: Verify Outputs

After running all cells, check that these files were created in the `outputs/` folder:

```bash
ls outputs/
# Expected output:
# model_comparison.png
# feature_importance.png
# residuals.png
```

### Step 4: Check Final Results

The final model performance should match these values (allow ±0.001 variation due to random state):

| Model | CV R² | Test R² (log) | MAPE | Selected |
|-------|-------|---------------|------|----------|
| Linear Regression | 0.3371 | 0.3380 | 3.91% | ✗ |
| Decision Tree | 0.6228 | 0.6711 | 2.59% | ✗ |
| XGBoost | 0.6663 | 0.6687 | 2.67% | ✗ |
| **Random Forest** | **0.6935** | **0.6945** | **2.54%** | **✓** |

These values appear in **Cell 62** of the notebook.

---

## 📈 Model Performance

### Cross-Validation Results (5-Fold on Training Set)

```python
Random Forest:
  Mean CV R²: 0.6935  ±  0.0011
  Folds: [0.6945, 0.6927, 0.6945, 0.6917, 0.6940]

XGBoost:
  Mean CV R²: 0.6663  ±  0.0014
  Folds: [0.6690, 0.6654, 0.6664, 0.6649, 0.6660]
```

### Test Set Metrics (Random Forest)

- **R² (log-price):** 0.6945 — Explains 69.45% of price variance
- **R² (price scale):** 0.6899
- **MAPE:** 2.54% — Mean absolute percentage error
- **RMSE (log):** 0.0309

### Feature Importance (Top 5)

1. **Year:** 52.9% — 30-year price inflation trend
2. **Town/City:** 12.2% — Urban/rural gradient
3. **County:** 12.1% — Regional premium
4. **District:** 11.5% — Local micro-market
5. **Property_Type_is__F:** 3.4% — Flat indicator

---

## 📦 Dependencies

All dependencies are listed in `requirements.txt`:

```txt
numpy>=1.21.0
pandas>=1.3.0
scikit-learn>=1.0.0
xgboost>=1.5.0
matplotlib>=3.4.0
seaborn>=0.11.0
jupyter>=1.0.0
notebook>=6.4.0
```

Install with:
```bash
pip install -r requirements.txt
```

---

## 🔍 Notebook Structure

The main notebook (`data_preprocessing_eda_v2.ipynb`) contains **104 cells** organized as follows:

### Section 1: Environment Setup (Cells 1)
- Import libraries

### Section 2: Data Loading & Cleaning (Cells 2-17)
- Load 30.9M records
- Check null values (drop incomplete records)
- Drop irrelevant features
- Transform date fields
- Remove 'Other' property type
- **Result:** 1,585,860 clean records

### Section 3: Exploratory Data Analysis (Cells 18-27)
- Geographic coverage (1,118 unique cities)
- Transaction volume analysis (London: 364,855; York: 6,157; Burnley: 408)
- Price heatmaps by year
- City comparison plots
- Top 50 towns by value

### Section 4: Feature Engineering (Cells 28-46)
- One-hot encoding (Property Type)
- Label encoding (Old/New)
- Log transformation (skewness 9.92 → -0.09)
- Outlier capping (26,176 low + 21,211 high removed)
- Correlation analysis
- Factorization of location features
- Train/test split (75/25)

### Section 5: Baseline Models (Cells 47-50)
- Linear Regression (standardized features)
- Decision Tree (max_depth=20)

### Section 6: Enhanced Models (Cells 51-62) — **Added by AI Assistant**
- **Cell 51:** XGBoost installation
- **Cell 52:** Log1p transformation prep
- **Cell 53:** 7-metric evaluation function
- **Cell 54:** Random Forest + 5-fold CV
- **Cell 55:** Random Forest test evaluation
- **Cell 56:** XGBoost + 5-fold CV
- **Cell 57:** XGBoost test evaluation
- **Cell 58:** Baseline re-evaluation
- **Cell 59:** 6-panel visualization dashboard
- **Cell 60:** Feature importance comparison
- **Cell 61:** Residual analysis
- **Cell 62:** Best model selection + conclusion

---

## ⏱️ Expected Runtime

Running the complete notebook on a modern laptop:

| Section | Cells | Approximate Time |
|---------|-------|------------------|
| Data Loading | 2-17 | 2-3 minutes |
| EDA | 18-27 | 1-2 minutes |
| Feature Engineering | 28-46 | 2-3 minutes |
| Baseline Models | 47-50 | 1-2 minutes |
| Random Forest (300 trees) | 54-55 | 15-20 minutes |
| XGBoost (500 trees) | 56-57 | 5-10 minutes |
| Visualization | 59-61 | 1-2 minutes |
| **Total** | **All cells** | **30-45 minutes** |

💡 **Tip:** Use `n_jobs=-1` in all models to utilize all CPU cores and speed up training.

---

## 🐛 Troubleshooting

### Issue 1: "File not found: pp-complete.csv"

**Solution:**
```bash
# Ensure the data file is in the data/ folder
ls data/pp-complete.csv

# If not present, download from HM Land Registry (see Data Download section)
```

### Issue 2: "MemoryError: Unable to allocate array"

**Solution:**
- Ensure you have at least 16GB RAM
- Close other memory-intensive applications
- Or use the sample data approach (see Alternative: Use Sample Data)

### Issue 3: "ModuleNotFoundError: No module named 'xgboost'"

**Solution:**
```bash
pip install xgboost
# Or reinstall all dependencies:
pip install -r requirements.txt
```

### Issue 4: Models training very slowly

**Solution:**
- Check that `n_jobs=-1` is set in model parameters (uses all CPU cores)
- Reduce `n_estimators` for testing (e.g., 100 instead of 300)
- Use sample data for initial testing

### Issue 5: Different results than report

**Causes:**
- Using sample data instead of full dataset
- Different `random_state` values
- Different scikit-learn/xgboost versions

**Solution:**
- Use full `pp-complete.csv` dataset
- Verify `random_state=42` in all models
- Check dependency versions: `pip list | grep -E 'scikit|xgboost'`

---

## 📝 Citation

If you use this code or analysis in your research, please cite:

```bibtex
@misc{uk_house_price_prediction_2026,
  author = {Sajitta Phat},
  title = {Predicting UK Residential Property Sale Prices: A Machine Learning Analysis},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/SajittaPhat/Predicting-UK-Residential-Property-Sale-Prices}
}
```

### Data Source Citation

```bibtex
@misc{hm_land_registry_2025,
  author = {{HM Land Registry}},
  title = {Price Paid Data},
  year = {2025},
  url = {https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads},
  note = {Contains HM Land Registry data © Crown copyright and database right 2025}
}
```

---

## 🤝 Contributing

This repository is part of an academic assignment and is not actively maintained. However, if you find issues or have suggestions:

1. Fork the repository
2. Create a new branch (`git checkout -b feature/improvement`)
3. Make your changes
4. Submit a pull request

---

## 📧 Contact

**Author:** Sajitta Phat  
**GitHub:** [@SajittaPhat](https://github.com/SajittaPhat)  
**Repository:** [Predicting-UK-Residential-Property-Sale-Prices](https://github.com/SajittaPhat/Predicting-UK-Residential-Property-Sale-Prices)

For questions about the analysis or code, please open an issue in the repository.

---

## 📄 License

This project is licensed under the MIT License - see below for details:

```
MIT License

Copyright (c) 2026 Sajitta Phat

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

---

## 🙏 Acknowledgments

- **HM Land Registry** for providing the Price Paid Data
- **Claude AI (Anthropic)** for debugging assistance and code enhancement
- **scikit-learn** and **XGBoost** development teams for excellent ML libraries


**Last Updated:** February 2026  
**Status:** ✅ Complete — Ready for reproduction

---

## Quick Start Checklist

- [ ] Clone repository
- [ ] Create virtual environment
- [ ] Install dependencies (`pip install -r requirements.txt`)
- [ ] Download `pp-complete.csv` from HM Land Registry
- [ ] Place data file in `data/` folder
- [ ] Launch Jupyter (`jupyter notebook data_preprocessing_eda_v2.ipynb`)
- [ ] Run all cells (Kernel → Restart & Run All)
- [ ] Verify outputs in `outputs/` folder
- [ ] Check final results match report values

**Estimated total time:** 45-60 minutes (including download)

---

*For the complete academic report and detailed analysis, see `report/UK_House_Price_Final.docx`*
