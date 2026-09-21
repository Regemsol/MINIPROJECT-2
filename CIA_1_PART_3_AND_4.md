# CIA 1 Parts 3 and 4: CKD Risk Prediction System

## Project Context

This project is an academic screening demonstration for chronic kidney disease (CKD) risk prediction. It uses the UCI Chronic Kidney Disease dataset, which contains 400 patient records. The system predicts one of two classes: `CKD` or `Not CKD`.

The implemented pipeline is:

```text
Raw clinical data -> cleaning and encoding -> feature selection -> scaling
-> KNN prediction -> evaluation and sample-patient demonstration
```

The system is not a medical diagnosis tool. A qualified healthcare professional must make the final clinical decision.

# CIA 1 - Part 3

## 1. Knowledge Representation and Inference by the Agent

### Knowledge representation model

The system represents each patient as a structured collection of clinical facts and a target class. The selected facts are:

- `hemo`: hemoglobin level
- `pcv`: packed cell volume
- `sg`: urine specific gravity
- `htn`: hypertension status after encoding
- `rbcc`: red blood cell count

The target fact is `class`, represented as `1` for CKD and `0` for Not CKD. Missing values are first handled by replacing numeric values with the column median and categorical values with the column mode. Categorical values are then encoded into numbers.

This is a **data-driven knowledge representation** rather than a large hand-written medical rule base. The cleaned training records store examples of patient states and their known outcomes. The learned KNN model acts as the agent's decision mechanism for comparing a new patient with those stored examples.

### Inference process

When a new patient is entered, the knowledge-based prediction process is:

1. Validate that the input contains the same five features used during training.
2. Apply the same preprocessing and feature representation used for the training data.
3. Standardize the feature values so that a feature's measurement scale does not dominate the distance calculation.
4. Calculate the distance between the new patient and every patient in the training set.
5. Select the five nearest training patients because the model uses `n_neighbors=5`.
6. Count the class labels among those five neighbors.
7. Infer `CKD` when the CKD neighbors are the majority; otherwise infer `Not CKD`.
8. Return the predicted class as a screening result for review by a healthcare professional.

For example, if four of the five nearest patients have class `CKD` and one has class `Not CKD`, the KNN inference is `CKD`. The inference is based on similarity to observed labelled cases, not on a claim that any one feature causes CKD. This is why the system is useful for pattern-based screening but cannot replace clinical examination, additional tests, or expert judgement.

The implementation uses a scikit-learn `Pipeline` containing `StandardScaler` and `KNeighborsClassifier`. This keeps preprocessing and inference in the same saved model and reduces the risk of applying inconsistent transformations to new patient data.

## 2. Handling Uncertainty

Medical data is uncertain for several reasons: values may be missing or incorrectly recorded, measurements may vary between laboratories, and two patients with similar values may have different clinical outcomes. The system handles these sources of uncertainty as follows:

- **Missing values:** rows with excessive missing information are removed. Remaining numeric gaps are filled using the median, while categorical gaps use the most frequent value. This allows the demonstration to run, but imputation can introduce bias and should be monitored in a clinical system.
- **Different measurement scales:** `StandardScaler` converts the selected features to comparable scales before distance calculation.
- **Conflicting neighbours:** KNN uses a majority vote. If the five nearest patients are divided 3-to-2, the majority class is selected, but this should be treated as a lower-confidence case than a 5-to-0 vote.
- **Model uncertainty:** the system should expose the proportion of neighbouring votes, or a probability estimate, as an uncertainty indicator. A result close to 50% should be flagged for human review rather than treated as a strong prediction.
- **Data and population uncertainty:** the dataset is small and may not represent every hospital, age group, ethnicity, or population. The model should therefore be externally validated and periodically monitored before any real deployment.
- **Decision uncertainty:** a false negative could delay CKD assessment and a false positive could cause unnecessary anxiety or testing. The system should be used as a triage aid, with an escalation route for uncertain or high-risk cases.

The reported accuracy of 0.9375 on the fixed test split is evidence about this dataset and split only. It is not a guarantee of clinical accuracy.

## 3. Sustainable Solution

A sustainable version of this system would be a low-cost clinical screening support service that runs on existing hospital computers or a secure local server. It would use routinely collected laboratory and patient-history values to identify patients who may need further CKD assessment.

The sustainability design would include:

- **Efficient computation:** KNN with five selected features requires modest storage and processing power. It can run on ordinary hardware without expensive GPU resources.
- **Use of existing data:** the system can use measurements already collected during routine care, reducing the need for additional tests and unnecessary travel.
- **Human-in-the-loop operation:** the model produces a screening flag and explanation of the similar cases or vote balance. A clinician remains responsible for confirmation and action.
- **Regular monitoring:** performance should be checked on new, labelled data. The feature-selection and training process should be repeated when the population, measurement equipment, or clinical practice changes.
- **Secure and privacy-preserving deployment:** patient identifiers should be removed or protected, access should be role-based, and data should be encrypted in storage and transit.
- **Inclusive access:** the system should support clinics with limited technical resources and should not assume that every patient has internet access or expensive monitoring equipment.
- **Responsible model lifecycle:** model versions, training data dates, validation results, and changes to preprocessing should be recorded so that results can be audited.

This approach is more sustainable than replacing clinicians with an opaque automated system. It focuses computing resources on early screening while preserving professional oversight and the possibility of correcting a model error.

## 4. Societal and Environmental Impact

### Positive societal impact

- Earlier identification of patients who may require confirmatory CKD tests can support timely clinical follow-up.
- A low-cost screening aid can help smaller clinics prioritise limited healthcare resources.
- Automated analysis can reduce repetitive manual comparison of multiple measurements.
- A consistent first-pass screening process may support more uniform access to risk assessment.

### Societal risks and controls

- **Bias and unequal performance:** the UCI dataset is small and may not represent the local population. Performance must be reported separately across relevant demographic and clinical groups where legally and ethically appropriate.
- **False reassurance:** a Not CKD prediction could delay care if users treat it as a diagnosis. The interface must clearly state that it is only a screening result and that symptoms or clinical judgement override it.
- **Over-referral and anxiety:** false positives can create unnecessary tests and stress. Predictions should be combined with clinical protocols rather than used as the only referral criterion.
- **Privacy:** health data is sensitive. The system should minimise collected data, use access controls, and follow applicable health-data regulations.
- **Accountability and explainability:** clinicians should be able to see the input features, model version, evaluation evidence, and uncertainty indicator before acting on an output.

### Environmental impact

The model has a relatively small environmental footprint because it uses five features, a lightweight KNN classifier, and ordinary CPU hardware. Training and prediction require far less energy than repeatedly training a large deep-learning model. Using existing clinical measurements can also avoid some additional travel and duplicate testing.

However, digital systems still consume electricity and require hardware. A responsible deployment should use efficient local or shared infrastructure, retrain only when new data justifies it, retain only necessary records, and dispose of or recycle hardware according to organisational policy. The environmental benefit should not be overstated: it depends on whether the system actually reduces unnecessary tests or travel without increasing referrals and repeat measurements.

# CIA 1 - Part 4

## 5. Applicable Learning Method

The applicable learning method is **supervised machine learning**, specifically **binary classification using K-nearest neighbors (KNN)**.

Supervised learning is appropriate because each training record has input features and a known outcome label: `CKD` or `Not CKD`. The algorithm learns the relationship between the five selected clinical features and the labelled class. For a new unlabeled patient, KNN finds similar labelled patients and predicts the majority class.

KNN is suitable for this academic system because:

- it is simple to explain to non-specialists;
- it works naturally with labelled examples and similarity between patients;
- it is lightweight and does not require expensive training hardware;
- its predictions can be explained using the nearest patient examples and their class votes.

The project is not currently an unsupervised, reinforcement-learning, or deep-learning system. It also does not implement the complete hybrid KNN + SVM + EBT approach described in the reference paper; it implements one KNN classifier for a clear CIA-1 demonstration.

## 6. Training Process

The training process is implemented in `src/preprocess.py`, `src/feature_selection.py`, and `src/train.py`.

### Step 1: Prepare the data

1. Load `data/ckd_raw.csv`.
2. Convert `?` and blank entries into missing values.
3. Remove rows with excessive missing information.
4. Convert values that are mostly numeric into numeric columns.
5. Fill missing numeric values with the column median.
6. Fill missing categorical values with the column mode.
7. Encode categorical values and map `ckd` to `1` and `notckd` to `0`.
8. Save the cleaned data to `data/ckd_clean.csv`.

### Step 2: Select useful features

The system calculates the Pearson correlation between each cleaned predictor and the encoded target. It ranks predictors by the absolute value of their correlation and selects the top five:

| Feature | Absolute correlation |
| --- | ---: |
| `hemo` | 0.726368 |
| `pcv` | 0.673129 |
| `sg` | 0.659504 |
| `htn` | 0.590438 |
| `rbcc` | 0.566163 |

The reduced data is saved to `data/ckd_selected.csv`. Correlation is used for feature selection, but it does not prove that a feature causes CKD.

### Step 3: Split the labelled data

The selected dataset is divided into:

- **80% training data:** used to store examples from which KNN can learn similarity patterns.
- **20% test data:** kept separate until evaluation to estimate performance on unseen records.

The split is stratified, so the CKD and Not CKD proportions remain similar in both sets. `random_state=42` makes the experiment reproducible.

### Step 4: Build and fit the model

The training pipeline is:

```python
Pipeline([
    ("scale", StandardScaler()),
    ("knn", KNeighborsClassifier(n_neighbors=5)),
])
```

`StandardScaler` is fitted on the training data and transforms the feature values. KNN then stores the scaled training examples and their labels. There is no conventional parameter-fitting stage like there is in a neural network; KNN mainly learns by retaining the labelled examples and using them during prediction.

### Step 5: Evaluate and save the model

The trained model predicts the labels of the held-out test set. The project reports accuracy, precision, recall, F1-score, and a confusion matrix. The reproducible run gives:

```text
Accuracy: 0.9375
Precision: 0.9787
Recall: 0.9200
F1-score: 0.9485
Confusion matrix:
[[29  1]
 [ 4 46]]
```

Finally, the pipeline is saved as `models/knn_ckd.pkl`, and the test records are saved as `data/ckd_test.csv`. The saved pipeline can then be loaded by the notebook for confusion-matrix visualisation, classification reporting, and manual sample-patient predictions.

## Conclusion

The proposed agent combines structured clinical facts with supervised similarity-based inference. It provides a practical, low-resource CKD screening demonstration, while uncertainty indicators, human review, privacy controls, validation, and fairness monitoring are necessary before any real-world clinical use.
