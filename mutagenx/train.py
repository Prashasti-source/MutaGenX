import pandas as pd
import joblib

from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    train_test_split
)

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    classification_report
)

feature_df = pd.read_csv(
    "feature_matrix.csv"
)

labels_df = pd.read_csv(
    "labels.csv"
)

ml_df = pd.merge(
    feature_df,
    labels_df,
    on="Seq_ID"
)

X = ml_df.drop(
    columns=[
        "Seq_ID",
        "Label",
        "VariantName",
        "Gene"
    ],
    errors="ignore"
)

y = ml_df["Label"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=5,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

print("Training RandomForest...")

model.fit(
    X_train,
    y_train
)

y_pred = model.predict(X_test)

y_prob = model.predict_proba(X_test)[:,1]

acc = accuracy_score(
    y_test,
    y_pred
)

roc = roc_auc_score(
    y_test,
    y_prob
)

print(f"\nAccuracy: {acc:.4f}")

print(f"ROC-AUC: {roc:.4f}")

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

joblib.dump(
    model,
    "trained_model.pkl"
)

joblib.dump(
    X.columns.tolist(),
    "feature_columns.pkl"
)

print(
    "\nSaved:"
)

print(
    "trained_model.pkl"
)

print(
    "feature_columns.pkl"
)
