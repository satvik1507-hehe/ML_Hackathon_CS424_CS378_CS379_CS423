"""Initial candidate scorers for the required five-model comparison.

These estimators score observation-arc feature rows. Training must use trip-grouped folds and
training-only preprocessing; do not fit a model until candidate labels are aligned and validated.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def build_candidates(seed: int = 99) -> dict[str, object]:
    """Return the five pre-registered candidate-scoring estimators."""
    return {
        "logistic_regression": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", LogisticRegression(max_iter=600, class_weight="balanced", random_state=seed)),
        ]),
        "random_forest": RandomForestClassifier(
            n_estimators=120, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=100, learning_rate=0.08, l2_regularization=1.0, class_weight="balanced",
            random_state=seed,
        ),
        "calibrated_linear_svm": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", CalibratedClassifierCV(
                estimator=LinearSVC(C=1.0, class_weight="balanced", random_state=seed), cv=3,
            )),
        ]),
        "small_mlp": Pipeline([
            ("scale", StandardScaler()),
            ("classifier", MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3, max_iter=100,
                                          early_stopping=True, random_state=seed)),
        ]),
    }

