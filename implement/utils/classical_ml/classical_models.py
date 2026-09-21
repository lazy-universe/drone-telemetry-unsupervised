import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.mixture import GaussianMixture
from sklearn.svm import OneClassSVM
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor

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


class PCAWrapper:
    """
    PCA-based reconstruction error (Q-statistic / SPE) and Hotelling's T^2 anomaly detector.
    Computes both Squared Prediction Error (Q) and Hotelling's T^2 statistic.
    """

    def __init__(self, n_components=0.95, random_state=42):
        self.n_components = n_components
        self.random_state = random_state
        self.pca = None
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        n_feats = X.shape[1]

        # Determine exact integer component count to avoid float shape casting in Python 3.13 / CuPy
        if isinstance(self.n_components, (float, np.floating)):
            full_pca = PCA(n_components=int(n_feats), random_state=self.random_state)
            full_pca.fit(X)
            cum_var = np.cumsum(full_pca.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cum_var, float(self.n_components)) + 1)
            n_comp = max(1, min(n_feats - 1, n_comp))
        elif isinstance(self.n_components, (int, np.integer)):
            n_comp = max(1, min(n_feats - 1, int(self.n_components)))
        else:
            n_comp = max(1, n_feats - 1)

        self.pca = PCA(n_components=int(n_comp), random_state=self.random_state)
        self.pca.fit(X)
        self.mean_ = self.pca.mean_
        self.components_ = self.pca.components_
        self.explained_variance_ = np.maximum(self.pca.explained_variance_, 1e-6)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        X_centered = X - self.mean_
        scores = np.dot(X_centered, self.components_.T)
        X_reconstructed = np.dot(scores, self.components_)

        # Q-statistic: Squared Prediction Error (Reconstruction Error)
        q_stat = np.sum((X_centered - X_reconstructed) ** 2, axis=1)

        # Hotelling's T^2 statistic: sum_i (score_i^2 / lambda_i)
        t2_stat = np.sum((scores ** 2) / self.explained_variance_, axis=1)

        # Combined normalized anomaly metric
        n_pcs = max(1, int(self.components_.shape[0]))
        n_residual = max(1, int(X.shape[1] - n_pcs))
        return (q_stat / n_residual) + (t2_stat / n_pcs)

    def predict(self, X, threshold=None):
        scores = self.decision_function(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)


class KMeansWrapper:
    """
    K-Means clustering distance anomaly detector.
    Anomaly score is the Euclidean distance to the nearest normal cluster centroid.
    """

    def __init__(self, n_clusters=5, random_state=42):
        self.n_clusters = int(n_clusters)
        self.random_state = random_state
        self.kmeans = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        k = max(1, min(int(self.n_clusters), int(len(X))))
        self.kmeans = KMeans(n_clusters=int(k), random_state=self.random_state, n_init='auto')
        self.kmeans.fit(X)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        distances = self.kmeans.transform(X)
        return np.min(distances, axis=1)

    def predict(self, X, threshold=None):
        scores = self.decision_function(X)
        if threshold is None:
            threshold = np.percentile(scores, 95)
        return (scores > threshold).astype(int)


class DBSCANWrapper:
    """
    DBSCAN core-point proximity anomaly detector.
    Fits DBSCAN on clean training data and uses the distance to the nearest core point
    as the anomaly score for incoming samples.
    """

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.dbscan = None
        self.core_points_nn_ = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.dbscan = DBSCAN(eps=float(self.eps), min_samples=int(self.min_samples))
        self.dbscan.fit(X)

        core_indices = self.dbscan.core_sample_indices_
        if len(core_indices) > 0:
            core_points = X[core_indices]
        else:
            core_points = X

        if len(core_points) > 5000:
            rng = np.random.RandomState(42)
            sub_idx = rng.choice(len(core_points), size=5000, replace=False)
            core_points = core_points[sub_idx]

        self.core_points_nn_ = NearestNeighbors(n_neighbors=1, algorithm='auto')
        self.core_points_nn_.fit(core_points)
        return self

    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        distances, _ = self.core_points_nn_.kneighbors(X)
        return distances.ravel()

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
        'PCA': PCAWrapper(n_components=0.95, random_state=random_state),
        'K-Means': KMeansWrapper(n_clusters=5, random_state=random_state),
        'DBSCAN': DBSCANWrapper(eps=0.5, min_samples=5),
    }
    if HAS_PYOD:
        models['KNN'] = KNN(n_neighbors=20, contamination=0.05)
    else:
        print("pyod not installed — skipping KNN. Run: pip install pyod")
    return models

