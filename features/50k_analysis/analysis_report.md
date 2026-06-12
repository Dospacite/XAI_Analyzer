# Feature Dataset Analysis

- Rows: `147524`
- Feature columns: `518`
- Model feature columns after zero-variance removal: `517`
- Label counts: `{'benign': np.int64(87924), 'phishing': np.int64(59600)}`
- Source-label NMI: `0.5750`
- Random Forest accuracy: `0.9345`
- Random Forest ROC AUC: `0.9795`
- SHAP: `completed`
- LIME: `completed`
- EBM: `completed`

## Bias Signals

High `source_label_nmi` means the collection source strongly predicts the label. That is expected when phishing_db is all phishing and Tranco is all benign, but it must not be exposed as a model feature.

The `feature_mutual_information.csv` table ranks features by association with both label and source. Features with high `source_mi` relative to `label_mi` deserve manual review for collection artifacts.

## Key Outputs

- `plots/source_label_crosstab.png`
- `plots/feature_source_mi.png`
- `plots/random_forest_importance.png`
- `plots/permutation_importance.png`
- `shap/shap_global_importance.png` and `shap/shap_summary.png`
- `lime/lime_local_explanations.csv` and per-sample HTML files
- `ebm/ebm_global_importance.png` when `interpret` is installed
