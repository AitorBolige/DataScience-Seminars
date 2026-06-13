# Using Machine Learning for Site of Origin Prediction in Ventricular Arrhythmias from ECGs and Clinical Data

**CompBioMed 2026 Seminars, Data Science Seminars submission**
**Group D:** Aitor Bolige, Núria Margarit, Núria Torrijos

## Description

This project predicts the Site of Origin (SOO) of ventricular arrhythmias from
the QRS complex of the ECG together with clinical data. Localizing the SOO
before an ablation procedure helps plan the intervention, so here we test
whether machine learning models can do it automatically.

The pipeline works as follows:

1. **Data loading** from several ECG/QRS sources: three `.mat` databases (the
   Clinic/CARTO cohort, the China cohort, and a simulation dataset) and the
   clinical Teknon cohort, provided as a corrected full dataset in `.pkl`.
2. **SOO standardization**, where chamber and sub location labels are normalized
   into a consistent taxonomy. The chamber is mapped to `{RV, LV}` and the
   sublocation to `{RVOT Septum, RFW, LCC, RCC, COMMISSURE, LVOT Subvalvular,
   LVOT Summit}`. Inconsistencies (e.g. an LV chamber carrying an RV
   sublocation) are corrected by trusting the more specific sublocation (tables
   in `soo_standarization/`).
3. **ECG segmentation and feature extraction** from the QRS complexes. For each
   patient the last beat (the premature ventricular contraction) is used; every
   lead is resampled and L1 normalized to build the feature vector.
4. **Modeling** with six classifiers (Logistic Regression, SVM RBF, Random
   Forest, Extra Trees, MLP, XGBoost), evaluated with stratified cross
   validation on three tasks: binary (RV vs LV), 3 class (RVOT, Cusp, LVOT),
   and a hierarchical pipeline built from two binary stages. Models are trained
   and validated across multiple dataset combinations, all evaluated on the
   held-out clinical Teknon cohort. The notebook also explores unsupervised
   methods (K-Means, and a CNN autoencoder latent space) on the Teknon data.
5. **Evaluation and interpretation** using balanced accuracy, F1, AUC, per class
   recall, confusion matrices, PCA separability, and SHAP feature importance
   (figures saved in `output/`).

## Repository structure

```
.
├── data/                                   Raw and processed datasets
│   ├── QRS_CARTO2.mat                       Clinic / CARTO cohort
│   ├── QRS_Database2.mat                    China cohort
│   ├── QRS_Sims2.mat                        Simulation dataset
│   └── full_data_corrected_2024.pkl         Clinical Teknon cohort
├── models/
│   ├── model.1 ... model.5                  ECG segmentation model files
│   ├── best_model_binary_classification_Extra Trees_China_Teknon.pkl
│   └── best_model_three_class_classification_SVM (RBF)_China_Teknon.pkl
├── notebooks/
│   ├── project_code.ipynb                   Main analysis and modeling notebook
│   ├── run_demo.ipynb                       Demo on the held-out Teknon test set
│   └── qrs_signals_function/                Per patient QRS signal arrays (.npy)
├── output/                                  Generated figures and result plots
├── soo_standarization/                      Standardized SOO tables (.csv)
├── requirements.txt                         Python dependencies
└── README.md
```

The two `best_model_*.pkl` files inside `models/` are the final fitted
classifiers, one for the binary task (Extra Trees) and one for the three class
task (SVM RBF), both trained on the China and Teknon combination. They can be
loaded directly with `pickle` to run predictions without retraining. The
`model.1` … `model.5` files are the pretrained ECG segmentation models used by
`project_code.ipynb` to extract the QRS complexes.

## Requirements

The project was developed with Python 3.8 (course kernel `CompBioMed26`).
All dependencies are listed in [`requirements.txt`](requirements.txt).

Main libraries: `numpy`, `pandas`, `scipy`, `scikit-learn`, `imbalanced-learn`,
`xgboost`, `torch`, `scikit-image`, `shap`, `matplotlib`, `seaborn`, `tqdm`.

`project_code.ipynb` additionally imports `sak`, a signal processing library
provided by the CompBioMed course environment. It is not a standard PyPI package
and must be installed from the course provided source. `run_demo.ipynb` does
**not** use `sak`.

## Reproducing the results

The commands below run on a clean machine and reproduce the held-out Teknon
evaluation reported below using the two saved models. No `sak` library and no
training step are required: the datasets and the fitted `best_model_*.pkl` files
are already included in the repository.

```bash
# 1. Clone the repository
git clone https://github.com/AitorBolige/CompBioMed26_GroupD_Submission.git
cd CompBioMed26_GroupD_Submission

# 2. Create and activate a Python 3.8 environment (conda recommended)
conda create -n soo python=3.8 -y
conda activate soo

# 3. Install dependencies
pip install -r requirements.txt

# 4. Reproduce the results: execute the demo notebook top to bottom
jupyter nbconvert --to notebook --execute --inplace notebooks/run_demo.ipynb
```

`run_demo.ipynb` loads the corrected Teknon dataset
(`data/full_data_corrected_2024.pkl`), replaces the raw labels with the
standardized SOO and chamber taxonomy from `soo_standarization/`, builds the
12-lead feature matrix (each QRS resampled to 10 samples per lead and L1
normalized), holds out the last 20% of patients as the test set, and runs the
two saved classifiers from `models/` on it. It prints the per class
`classification_report` and plots the confusion matrices for the binary
(LV vs RV) and three class (RVOT, Cusp, LVOT) tasks.

To open it interactively instead of executing it in batch:

```bash
jupyter notebook notebooks/run_demo.ipynb
```

### Full analysis (optional)

`project_code.ipynb` covers the complete pipeline: data loading and SOO
standardization, exploratory data analysis, ECG segmentation and feature
extraction, supervised and unsupervised modeling, the dataset combination
search, and the visual/interpretability analysis. It trains the six classifiers,
regenerates every figure in `output/`, and rewrites the `best_model_*.pkl` files
in `models/`. It additionally requires the course provided `sak` signal
processing library (see the note in [Requirements](#requirements)), which is not
on PyPI, so it cannot be installed on a clean machine without that source.

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/project_code.ipynb
```

## Results

Models are trained and validated across multiple dataset combinations and
evaluated on the clinical Teknon cohort. All scores are stratified cross
validation. BA is balanced accuracy and F1 is macro F1.

| Task         | Selected model | Data combination      | CV BA         | AUC   | F1    |
|--------------|----------------|-----------------------|:-------------:|:-----:|:-----:|
| Binary       | Extra Trees    | China + Teknon        | 0.745 ± 0.094 | 0.836 | 0.748 |
| 3 class      | SVM (RBF)      | China + Teknon        | 0.613 ± 0.077 | 0.789 | 0.593 |
| Hierarchical | Extra Trees    | China/Clinic + Teknon | 0.567         |   —   | 0.552 |

The binary RV vs LV task reaches the best and most stable performance. The
harder 3 class and hierarchical settings drop in balanced accuracy, with the
minority Cusp class being the main source of error.
