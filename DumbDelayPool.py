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

class DumbDelayPool(BaseEstimator, ClassifierMixin):
    """
    DumbDelayPool.

    Opis niezwykle istotnego klasyfikatora

    References
    ----------
    .. [1] A. Kowalski, B. Nowak, "Bardzo ważna praca o klasyfikatorze
    niezwykle istotnym dla przetrwania gatunku ludzkiego."

    """

    def __init__(self, ensemble_size=20):
        """Initialization."""
        self.ensemble_size = ensemble_size

    def set_base_clf(self, base_clf=neighbors.KNeighborsClassifier()):
        """Establish base classifier."""
        self._base_clf = base_clf

    # Fitting
    def fit(self, X, y):
        """Fitting."""
        if not hasattr(self, '_base_clf'):
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
        if not hasattr(self, '_base_clf'):
            self.set_base_clf()
        X, y = check_X_y(X, y)
        self.X_ = X
        self.y_ = y

        if _check_partial_fit_first_call(self, classes):
            self.classes_ = classes
            self.ensemble_ = []


        # Preparing and training new candidate
        self.classes_ = classes
        candidate_clf = base.clone(self._base_clf)
        candidate_clf.fit(X, y)
        self.ensemble_.append(candidate_clf)

        # Prune oldest
        if len(self.ensemble_) > self.ensemble_size:
            del self.ensemble_[0]

    def ensemble_support_matrix(self, X):
        """ESM."""
        return np.array([member_clf.predict_proba(X)
                         for member_clf in self.ensemble_])

    def predict_proba(self, X):
        """Aposteriori probabilities."""
        # Check is fit had been called
        check_is_fitted(self, 'classes_')


        # Acumulate supports
        esm = self.ensemble_support_matrix(X)
        acumulated_support = np.sum(esm, axis=0)
        return acumulated_support

    def predict(self, X):
        """Hard decision."""
        # Check is fit had been called
        check_is_fitted(self, 'classes_')

        # Input validation
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError('number of features does not match')

        supports = self.predict_proba(X)
        prediction = np.argmax(supports, axis=1)

        return self.classes_[prediction]

    def score(self, X, y):
        return f1_score(y, self.predict(X))
