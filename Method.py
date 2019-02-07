"""
Dumb Delay Pool.
"""

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils.multiclass import _check_partial_fit_first_call
from sklearn import base
from sklearn import neighbors
from sklearn.metrics import f1_score
import numpy as np


class Method(BaseEstimator, ClassifierMixin):
    """
    DumbDelayPool.

    Opis niezwykle istotnego klasyfikatora

    References
    ----------
    .. [1] A. Kowalski, B. Nowak, "Bardzo ważna praca o klasyfikatorze
    niezwykle istotnym dla przetrwania gatunku ludzkiego."

    """

    def __init__(self, ensemble_size=5):
        """Initialization."""
        self.ensemble_size = ensemble_size

    def set_base_clf(self, base_clf=neighbors.KNeighborsClassifier()):
        """Establish base classifier."""
        self._base_clf = base_clf

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()

        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        candidate_clf = base.clone(self._base_clf)
        candidate_clf.fit(X, y)

        self.ensemble_ = [candidate_clf]

        # Return the classifier
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting."""
        if not hasattr(self, "_base_clf"):
            self.set_base_clf()
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes
            self.ensemble_ = []
            self.previous_X = self.X_
            self.previous_y = self.y_

        # Do przemyślenia
        # if len(self.ensemble_) > 1:
        #     test = self.region_of_competence_predict(X, n_neighbors=5)


        # Copy the old chunk
        self.previous_X = self.X_
        self.previous_y = self.y_

        # Preparing and training new candidate
        self.classes_ = classes
        candidate_clf = base.clone(self._base_clf)

        # Remove outliers
        # X_wo_outliers, y_wo_outliers = self.remove_outliers(X, y)

        candidate_clf.fit(X, y)
        self.ensemble_.append(candidate_clf)

        # Score base models
        base_scores = self.f1_score_base_classifiers(X, y)

        # Prune the worst classifer if ensemble size exceeded
        self.prune_worst_classifier(base_scores)

        # Prune all classifiers below f1 threshold
        self.prune_threshold(base_scores, threshold=0.94)

    def remove_outliers(self, X, y):
        y_processed = y.copy()
        X_processed = X.copy()
        # Calculate distance from each instance to the rest
        manhattan_distance_matrix = self.manhattan_distance(X_processed, X_processed)
        # Find 5 nearest neighbors based on distance
        neighbors = self.region_of_competence(manhattan_distance_matrix, n_neighbors=6)
        neighbors = neighbors[:, 1:]
        # Get neighbors classes
        neighbors_classes = y_processed[neighbors]
        # Find outliers
        outliers = np.where(np.sum(neighbors_classes-(1-y_processed).reshape(500,1), axis=1) == 0)[0]
        # Remove outliers
        for index in sorted(outliers, reverse=True):
            X_processed = np.delete(X_processed, index, axis=0)
            y_processed = np.delete(y_processed, index, axis=0)
        return X_processed, y_processed

    def previous_decision_matrix(self):
        """Ensemble decision matrix for the previous chunk"""
        return np.array(
            [member_clf.predict(self.previous_X) for member_clf in self.ensemble_]
        )

    def manhattan_distance(self, X1, X2):
        """Manhattan distance from each new instance in X1 to the X2 instances"""
        return np.array(
            [np.sum(np.absolute(X2 - instance), axis=1) for instance in X1]
        )

    def region_of_competence(self, manhattan_distance_matrix, n_neighbors=5):
        """ Region of competence based on Manhattan
        distance from each new instance to the previous chunk"""
        return np.argsort(manhattan_distance_matrix)[:, :n_neighbors]

    def region_of_competence_predict(self, X, n_neighbors=5):
        # Each clf prediction for previous chunk
        prev_decision_matrix = self.previous_decision_matrix()

        # Manhattan distance from each test instance to the previous chunk
        manhattan_distance_matrix = self.manhattan_distance(X, self.previous_X)

        # Region of competence for each test instance
        competence_region = self.region_of_competence(manhattan_distance_matrix, n_neighbors=n_neighbors)

        # Ni mom pojęcia co robie
        matrix = competence_region[prev_decision_matrix, :]
        # print(matrix.shape)
        # print(matrix.argmin(axis=0))

    def f1_score_base_classifiers(self, X, y):
        return np.array(
            [f1_score(y, member_clf.predict(X)) for member_clf in self.ensemble_]
        )

    def prune_worst_classifier(self, base_models_scores):
        """Prune the worst classifer if ensemble size exceeded"""
        if len(self.ensemble_) > self.ensemble_size:
            del self.ensemble_[base_models_scores.argmin()]

    def prune_threshold(self, base_models_scores, threshold=0.55):
        """Prune all classifiers below f1 threshold,
        always leaves one clf"""
        indices = np.argwhere(base_models_scores < threshold)
        for index in sorted(indices.ravel(), reverse=True):
            if len(self.ensemble_) > 1:
                del self.ensemble_[index]

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_])

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Acumulate supports
        esm = self.ensemble_support_matrix(X)
        acumulated_support = np.sum(esm, axis=0)
        return acumulated_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, "classes_")

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]

    def score(self, X, y):
        return f1_score(y, self.predict(X))
