# Employee Attrition Modelling

This project builds classification models to predict employee attrition using the IBM HR attrition dataset.

## Methodology

- Start from the reduced-column dataset logic in `EDA_rm_col.ipynb`
- Remove constant, duplicated-meaning, and high-VIF features
- Keep the no-PCA path for modelling
- Use leakage-safe sklearn pipelines for preprocessing and training
- Compare multiple classifiers with stratified cross-validation
- Evaluate with imbalance-aware metrics, especially PR-AUC
- Select `SVM (RBF)` as the blackbox handoff model for later XAI work

## Main Deliverable

- `Attrition_Modelling_Pipeline.ipynb`: end-to-end modelling notebook with preprocessing, model comparison, tuning, evaluation, and recommendation

## Poetry Setup

Install dependencies:

```bash
poetry install --no-root
```

Run the notebook environment:

```bash
poetry shell
poetry run jupyter notebook
```

Or execute the modelling notebook directly:

```bash
poetry run jupyter nbconvert --to notebook --execute Attrition_Modelling_Pipeline.ipynb --output Attrition_Modelling_Pipeline.ipynb
```

