# CKD Demo Steps

This file shows the easiest way to present the project to a teacher or evaluator.

## 1. Open the project folder

```bash
cd /workspaces/5024123-AI-CIA-P5
```

## 2. Run the preprocessing pipeline

```bash
python src/preprocess.py
python src/feature_selection.py
```

These steps clean the raw CKD dataset and select the five most important features.

## 3. Train and evaluate the model

```bash
python src/train.py
```

The script prints:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

This shows that the model is working on the dataset.

## 4. Show the notebook demo

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

## 5. Show a single-patient prediction script

```bash
python demo_single_patient.py
```

This script loads the saved KNN model and predicts one patient class.

Example output:

```text
Patient prediction: CKD
```

## 6. Explain the result

Say:

> This project trains a KNN classifier to predict whether a patient is likely to have CKD or not CKD based on five clinical features. The model is a screening tool and should not be treated as a medical diagnosis.

## 7. Short demo script for presentation

```text
I built a chronic kidney disease risk prediction system using a KNN classifier.
First, I cleaned the dataset and selected the most relevant features.
Then I trained the model and evaluated it using accuracy, precision, recall, and F1-score.
Finally, I used the saved model to predict sample patients as CKD or Not CKD.
```
