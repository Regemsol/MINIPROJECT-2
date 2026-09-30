# CKD Demo Steps

This file shows the easiest way to present the project to a teacher or evaluator.

## 1. Open the project folder

```bash
cd /workspaces/5024123-AI-CIA-P5
```

## 2. Reuse the saved model

The trained model is already saved in `models/knn_ckd.pkl`, so you do not need to preprocess or retrain for each demonstration. To print the evaluation metrics using the saved model, run:

```bash
python src/train.py
```

This command reuses the saved model. To intentionally train it again, run:

```bash
python src/train.py --retrain
```

Preprocessing and feature selection are only needed when rebuilding the dataset.

The script prints:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

This shows that the model is working on the dataset.

## 3. Show the notebook demo

```bash
jupyter notebook notebooks/demo.ipynb
```

Then run the cells from top to bottom.

The final cell predicts labels for sample patients and prints outputs like:

```python
Patient 1: Not CKD
Patient 2: CKD
Patient 3: CKD
```

## 4. Show a single-patient prediction script

```bash
python demo_single_patient.py
```

This script loads the saved KNN model and predicts one patient class without training.

Example output:

```text
Patient prediction: CKD
```

## 5. Explain the result

Say:

> This project uses a saved KNN classifier to predict whether a patient is likely to have CKD or not CKD based on five clinical features. The model is a screening tool and should not be treated as a medical diagnosis.

## 6. Short demo script for presentation

```text
I built a chronic kidney disease risk prediction system using a KNN classifier.
The trained model is saved, so I can demonstrate predictions without retraining it.
I evaluated it using accuracy, precision, recall, and F1-score, then predicted sample patients as CKD or Not CKD.
```

## Optional: Retrain the model

Retraining fits a new KNN model using the prepared dataset and replaces `models/knn_ckd.pkl`. You only need this if you want to train with changed data or intentionally rebuild the model. For the current prepared dataset, start at step 3.

1. Open a terminal in the project folder.
2. If you changed or replaced the raw dataset, rebuild the prepared data:

	```bash
	python src/preprocess.py
	python src/feature_selection.py
	```

	These commands clean the raw data and select the features used by the CKD model. Skip this step if `data/ckd_selected.csv` is already the dataset you want to use.

3. Train and save the new model:

	```bash
	python src/train.py --retrain
	```

	The script trains on the training portion of the data, prints evaluation metrics, saves the new model to `models/knn_ckd.pkl`, and updates `data/ckd_test.csv`.

4. Check the saved model by running `python src/train.py` without `--retrain`. It should load the model and print the metrics without fitting it again.
5. Run `python demo_single_patient.py` or open the notebook to demonstrate predictions from the new saved model.
