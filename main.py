# file: main.py

import warnings
warnings.filterwarnings('ignore')

from data_prep import load_and_prep_data
from models import (train_architecture_a, train_architecture_b,
                     predict_architecture_b,
                     save_models, load_models, models_exist)
from evaluation import (evaluate_predictions, print_metrics,
                         print_comparison, generate_all_confusion_plots)

# False -> load saved models if they exist, skip training
# True  -> always retrain from scratch and overwrite saved files
FORCE_RETRAIN = False

# grid-search C for both architectures (cheap -- runs on extracted features, not raw text)
TUNE_C = True


def main():
    filepath = r'C:\Users\jashw\Desktop\Future Interns\FUTURE_ML_02\customer_support_tickets.csv'

    print("=== 1. Data Preparation ===")
    (X_train_raw, X_test_raw, X_train_clean, X_test_clean,
     y_train, y_test, target_names) = load_and_prep_data(filepath)
    print(f"Train samples : {len(X_train_raw)} | Test samples : {len(X_test_raw)}")

    if not FORCE_RETRAIN and models_exist():
        print("\n=== Saved models found — loading from disk ===")
        model_a, model_b = load_models()
    else:
        print("\n=== 2. Training Architecture A (TF-IDF + SVD + Scaler + LinearSVC) ===")
        model_a = train_architecture_a(X_train_clean, y_train, tune_C=TUNE_C)

        print("\n=== 3. Training Architecture B (MiniLM Embeddings + Scaler + LinearSVC) ===")
        print("Encoding sentences with all-MiniLM-L6-v2 (this may take a while)...")
        model_b = train_architecture_b(X_train_raw, y_train, tune_C=TUNE_C)

        save_models(model_a, model_b)

    print("\n=== 4. Generating Predictions ===")
    preds_a = model_a.predict(X_test_clean)
    preds_b = predict_architecture_b(model_b, X_test_raw)

    print("\n=== 5. Evaluation ===")
    results_a = evaluate_predictions(y_test, preds_a, target_names)
    results_b = evaluate_predictions(y_test, preds_b, target_names)

    print_metrics(results_a, "Architecture A — TF-IDF + SVD + LinearSVC")
    print_metrics(results_b, "Architecture B — BERT (MiniLM) + LinearSVC")
    print_comparison(results_a, results_b)

    print("=== 6. Generating 3D Confusion Volume Plots ===")
    generate_all_confusion_plots(y_test, preds_a, preds_b, target_names)

    print("\nA/B Testing Complete. Check 'plots/' and 'saved_models/'.")


if __name__ == "__main__":
    main()