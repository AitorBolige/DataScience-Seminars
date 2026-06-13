# Using Machine Learning for Site of Origin Prediction in Ventricular Arrhythmias from ECGs and Clinical Data

**CompBioMed 2026 Seminars, Data Science Seminars submission**
**Group D:** Aitor Bolige, Núria Margarit, Núria Torrijos

## Description

This project predicts the Site of Origin (SOO) of ventricular arrhythmias from
the QRS complex of the ECG together with clinical data. Localizing the SOO
before an ablation procedure helps plan the intervention, so here we test
whether machine learning models can do it automatically.

The pipeline works as follows:

1. Data loading from several ECG/QRS sources (`.mat` databases, simulation data,
   the clinical Teknon dataset, and a corrected full dataset in `.pkl`).
2. SOO standardization, where chamber and sub location labels are normalized
   into a consistent taxonomy (see `soo_standarization/`).
3. Feature extraction from the QRS complexes.
4. Modeling with six classifiers (SVM RBF, Random Forest, Extra Trees, MLP,
   Logistic Regression, XGBoost), evaluated with cross validation on three
   tasks: binary (RV vs LV outflow), 3 class (RVOT, Cusp, LVOT), and a
   hierarchical pipeline. Models are trained and validated across multiple data
   sources, including the clinical Teknon cohort.
5. Evaluation and interpretation using balanced accuracy, F1, AUC, per class
   recall, confusion matrices, PCA separability, and SHAP feature importance
   (figures saved in `output/`).

## Repository structure

```
.
├── data/                       Raw and processed datasets (.mat, .pkl)
├── models/
│   ├── model.1 ... model.5     Trained model files
│   ├── best_model_binary_classification_Extra Trees_China_Teknon.pkl
│   └── best_model_three_class_classification_SVM (RBF)_China_Teknon.pkl
├── notebooks/
│   ├── project_code.ipynb      Main analysis and modeling notebook
│   └── qrs_signals_function/   Per patient QRS signal arrays (.npy)
├── output/                     Generated figures and result plots
├── soo_standarization/         Standardized Site of Origin (SOO) tables (.csv)
├── requirements.txt            Python dependencies
└── README.md
```

The two `best_model_*.pkl` files inside `models/` are the final fitted
classifiers, one for the binary task (Extra Trees) and one for the three class
task (SVM RBF), both trained on the China and Teknon combination. They can be
loaded directly with `pickle` to run predictions without retraining.

## Requirements

The project was developed with Python 3.8 (course kernel `CompBioMed26`).
All dependencies are listed in [`requirements.txt`](requirements.txt).

Main libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `imbalanced-learn`,
`xgboost`, `torch`, `scikit-image`, `shap`, `matplotlib`, `seaborn`, `tqdm`.

The notebook also imports `sak`, a signal processing library provided by the
CompBioMed course environment. It is not a standard PyPI package and must be
installed from the course provided source.

## Reproducing the results

```bash
# 1. Clone the repository
git clone https://github.com/AitorBolige/DataScience-Seminars.git
cd DataScience-Seminars

# 2. Create and activate a Python 3.8 environment (conda recommended)
conda create -n soo python=3.8 -y
conda activate soo

# 3. Install dependencies
pip install -r requirements.txt
# plus the course provided 'sak' library (see note above)

# 4a. Reproduce everything by running the notebook top to bottom
jupyter nbconvert --to notebook --execute --inplace notebooks/project_code.ipynb

# 4b. Or open it interactively and run all cells in order
jupyter notebook notebooks/project_code.ipynb
```

`project_code.ipynb` covers the full analysis, trains the final classifiers,
regenerates the metrics reported below, and saves the figures in `output/` and
the `best_model_*.pkl` files in `models/`. The `data/` directory must be present
alongside the notebook (it is included in this repository).

## Results

Models are trained and validated across multiple data sources and tested on the
clinical Teknon cohort. All scores are stratified cross validation. BA is
balanced accuracy and F1 is macro F1.

| Task         | Selected model | Data combination    | CV BA         | AUC   | F1    |
|--------------|----------------|---------------------|:-------------:|:-----:|:-----:|
| Binary       | Extra Trees    | China + Teknon      | 0.745 ± 0.094 | 0.836 | 0.748 |
| 3 class      | SVM (RBF)      | China + Teknon      | 0.613 ± 0.077 | 0.789 | 0.593 |
| Hierarchical | Extra Trees    | China/Clinic+Teknon | 0.567         |       | 0.552 |

The binary RV vs LV task reaches the best and most stable performance. The
harder 3 class and hierarchical settings drop in balanced accuracy, with the
minority Cusp class being the main source of error.
