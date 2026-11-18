"""Initial candidate scorers for the required five-model comparison.

These estimators score observation-arc feature rows. Training must use trip-grouped folds and
training-only preprocessing; do not fit a model until candidate labels are aligned and validated.
"""

from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def build_candidates(seed: int = 99) -> dict[str, object]:
    """Return six distinct estimators; at least five must be trained and compared."""
    return {
        "logistic_regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
            n_jobs=-1, random_state=seed,
        ),
        "extra_trees": ExtraTreesClassifier(
            n_estimators=300, min_samples_leaf=2, class_weight="balanced",
            n_jobs=-1, random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.08, l2_regularization=1.0, random_state=seed,
        ),
        "calibrated_rbf_svm": make_pipeline(
            StandardScaler(),
            CalibratedClassifierCV(
                estimator=SVC(C=2.0, kernel="rbf", class_weight="balanced"), cv=3,
            ),
        ),
        "small_mlp": make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64, 32), alpha=1e-3, max_iter=300,
                          early_stopping=True, random_state=seed),
        ),
    }

