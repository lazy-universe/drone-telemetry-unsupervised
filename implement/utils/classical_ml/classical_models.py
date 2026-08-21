import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM

try:
    from pyod.models.knn import KNN
    HAS_PYOD = True
except ImportError:
    HAS_PYOD = False


class GMMWrapper:
    """Wraps GaussianMixture to have sklearn-compatible anomaly detection interface."""

    def __init__(self, n_components=3, covariance_type='full', random_state=42):
        self.n_components = n_components
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = GaussianMixture(
            n_components=n_components, covariance_type=covariance_type, random_state=random_state
        )

    def fit(self, X, y=None):
        self.model.fit(X)
        return self

    def decision_function(self, X):
        # Returns negative log-likelihood: higher = more anomalous
        return -self.model.score_samples(X)

    def predict(self, X, threshold=None):
        scores = self.decision_function(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)


class MahalanobisWrapper:
    """Computes Mahalanobis distance from mean of normal training data."""

    def __init__(self):
        self.mean = None
        self.inv_cov = None

    def fit(self, X, y=None):
        self.mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False) + 1e-6 * np.eye(X.shape[1])
        self.inv_cov = np.linalg.inv(cov)
        return self

    def decision_function(self, X):
        diff = X - self.mean
        dist = np.sqrt(np.sum(np.dot(diff, self.inv_cov) * diff, axis=1))
        return dist

    def predict(self, X, threshold=None):
        scores = self.decision_function(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)


def get_unsupervised_point_models(random_state=42):
    """
    Returns a dictionary of point-wise unsupervised anomaly detection models.
    All models are trained on normal data only (class 0).
    """
    models = {
        'Isolation Forest': IsolationForest(
            n_estimators=100, contamination='auto', random_state=random_state
        ),
        'GMM': GMMWrapper(
            n_components=3, covariance_type='full', random_state=random_state
        ),
        'Mahalanobis': MahalanobisWrapper(),
        'One-Class SVM': OneClassSVM(
            kernel='rbf', gamma='scale', nu=0.05
        ),
    }
    if HAS_PYOD:
        models['KNN'] = KNN(n_neighbors=20, contamination=0.05)
    else:
        print("pyod not installed — skipping KNN. Run: pip install pyod")
    return models
