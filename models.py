# file: models.py

import os
import joblib
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, make_scorer
from sentence_transformers import SentenceTransformer

RANDOM_STATE = 42
MODELS_DIR = 'saved_models'
DEFAULT_C_GRID = (0.01, 0.1, 1, 10, 100)


class BertEmbedder(BaseEstimator, TransformerMixin):
    """Wraps a SentenceTransformer model as a scikit-learn transformer.

    fit() lazily loads the pre-trained model (nothing is actually trained).
    transform() returns dense sentence embeddings for the input text.
    """

    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model_name = model_name

    def fit(self, X, y=None):
        self.model_ = SentenceTransformer(self.model_name)
        return self

    def transform(self, X):
        texts = list(X)
        embeddings = self.model_.encode(texts, batch_size=32, show_progress_bar=True)
        return np.asarray(embeddings)


# --- Multi-output scoring for GridSearchCV ----------------------------------

def _multioutput_macro_f1(y_true, y_pred):
    scores = [
        f1_score(y_true[:, i], y_pred[:, i], average='macro', zero_division=0)
        for i in range(y_true.shape[1])
    ]
    return np.mean(scores)


multioutput_f1_scorer = make_scorer(_multioutput_macro_f1)


def tune_linear_svc_C(X_features, y_train, C_grid=DEFAULT_C_GRID):
    """Grid-search C on an already-extracted feature matrix.

    Cheap by design: the feature pipeline (TF-IDF+SVD or BERT) runs once;
    only the LinearSVC head is refit per candidate C.
    """
    base_clf = MultiOutputClassifier(
        LinearSVC(class_weight='balanced', random_state=RANDOM_STATE)
    )
    grid = GridSearchCV(
        base_clf,
        param_grid={'estimator__C': list(C_grid)},
        scoring=multioutput_f1_scorer,
        cv=3,
        n_jobs=-1
    )
    grid.fit(X_features, y_train)
    best_C = grid.best_params_['estimator__C']
    print(f"  Best C: {best_C}  (CV macro-F1: {grid.best_score_:.4f})")
    return best_C


# --- Architecture A: TF-IDF + SVD + Scaler + LinearSVC ----------------------

def train_architecture_a(X_train_clean, y_train, tune_C=True, C_grid=DEFAULT_C_GRID):
    feature_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(ngram_range=(1, 2), max_features=5000)),
        ('svd', TruncatedSVD(n_components=100, random_state=RANDOM_STATE)),
        ('scaler', StandardScaler())
    ])
    X_features = feature_pipeline.fit_transform(X_train_clean, y_train)

    C = tune_linear_svc_C(X_features, y_train, C_grid) if tune_C else 1.0

    clf = MultiOutputClassifier(
        LinearSVC(C=C, class_weight='balanced', random_state=RANDOM_STATE), n_jobs=-1
    )
    clf.fit(X_features, y_train)

    # all steps are already fitted -> Pipeline.predict() works without re-fitting
    return Pipeline(feature_pipeline.steps + [('clf', clf)])


# --- Architecture B: BERT embeddings + Scaler + LinearSVC -------------------

def train_architecture_b(X_train_raw, y_train, model_name='all-MiniLM-L6-v2',
                          tune_C=True, C_grid=DEFAULT_C_GRID):
    embedder = BertEmbedder(model_name=model_name)
    embedder.fit(X_train_raw)
    X_embeddings = embedder.transform(X_train_raw)

    scaler = StandardScaler()
    X_features = scaler.fit_transform(X_embeddings)

    C = tune_linear_svc_C(X_features, y_train, C_grid) if tune_C else 1.0

    clf = MultiOutputClassifier(
        LinearSVC(C=C, class_weight='balanced', random_state=RANDOM_STATE), n_jobs=-1
    )
    clf.fit(X_features, y_train)

    return embedder, scaler, clf


def predict_architecture_b(model_b, X_raw):
    embedder, scaler, clf = model_b
    X_embeddings = embedder.transform(X_raw)
    X_features = scaler.transform(X_embeddings)
    return clf.predict(X_features)


# --- Persistence --------------------------------------------------------------

def save_models(model_a, model_b):
    os.makedirs(MODELS_DIR, exist_ok=True)

    joblib.dump(model_a, os.path.join(MODELS_DIR, 'architecture_a.joblib'))

    embedder, scaler, clf = model_b
    embedder.model_.save(os.path.join(MODELS_DIR, 'bert_encoder'))   # native sentence-transformers save
    joblib.dump(
        (scaler, clf, embedder.model_name),
        os.path.join(MODELS_DIR, 'architecture_b_head.joblib')
    )

    print(f"Models saved to '{MODELS_DIR}/'")


def load_models():
    model_a = joblib.load(os.path.join(MODELS_DIR, 'architecture_a.joblib'))

    scaler, clf, model_name = joblib.load(
        os.path.join(MODELS_DIR, 'architecture_b_head.joblib')
    )
    embedder = BertEmbedder(model_name=model_name)
    embedder.model_ = SentenceTransformer(os.path.join(MODELS_DIR, 'bert_encoder'))
    model_b = (embedder, scaler, clf)

    print("Models loaded from disk — skipping training.\n")
    return model_a, model_b


def models_exist():
    return (
        os.path.exists(os.path.join(MODELS_DIR, 'architecture_a.joblib')) and
        os.path.exists(os.path.join(MODELS_DIR, 'architecture_b_head.joblib')) and
        os.path.isdir(os.path.join(MODELS_DIR, 'bert_encoder'))
    )