import numpy as np

from joblib import Parallel
from sklearn.base import clone
from sklearn.ensemble._base import _fit_single_estimator
from sklearn.utils.fixes import delayed
from sklearn.utils import Bunch
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.multiclass import check_classification_targets
from sklearn.utils.validation import column_or_1d
from sklearn.preprocessing import LabelEncoder
from sklearn.utils._encode import _unique

from sklearn.ensemble import VotingClassifier,VotingRegressor


def fit(self, X, y, sample_weight=None):
    """
    A modified version of _BaseVoting.fit that accounts for both
    fitted and unfitted estimators. It only fits unfitted estimators
    and saves both types in estimators_ and in their respective
    unfitted_estimators_ and fitted_estimators attributes.

    Used by both ModifiedVotingClassifier and ModifiedVotingRegressor.
    """
    
    """Get common fit operations."""
    names, clfs = self._validate_estimators()

    # Get just the models
    unfitted_clfs=[est for _,est in self.unfitted_estimators]
    fitted_clfs=[est for _,est in self.fitted_estimators]

    # Check that the number of weights matches the number of estimators
    if self.weights is not None and len(self.weights) != len(self.estimators):
        raise ValueError(
            "Number of `estimators` and weights must be equal"
            "; got %d weights, %d estimators"
            % (len(self.weights), len(self.estimators))
        )

    # Fit only unfitted estimators (that are also not dropped)
    self.estimators_ = Parallel(n_jobs=self.n_jobs)(
        delayed(_fit_single_estimator)(
            clone(clf),
            X,
            y,
            sample_weight=sample_weight,
            message_clsname="Voting",
            message=self._log_message(names[idx], idx + 1, len(unfitted_clfs)),
        )
        for idx, clf in enumerate(clfs)
        if clf != "drop" and clf in unfitted_clfs
    )
    # Save unfitted estimators
    self.unfitted_estimators_=self.estimators_[:]

    # Add fitted estimators (that are not dropped)
    self.estimators_.extend([clf for clf in clfs if clf != "drop" and clf in fitted_clfs])

    # Save fitted estimators
    self.fitted_estimators_=[clf for clf in clfs if clf != "drop" and clf in fitted_clfs]

    # Save a dictionary name -> estimator (containing both fitted and unfitted estimators).
    self.named_estimators_ = Bunch()

    # Uses 'drop' as placeholder for dropped estimators
    est_iter = iter(self.estimators_)
    for name, est in self.estimators:
        current_est = est if est == "drop" else next(est_iter)
        self.named_estimators_[name] = current_est

        if hasattr(current_est, "feature_names_in_"):
            self.feature_names_in_ = current_est.feature_names_in_

    return self

class VotingRegressor(VotingRegressor):
    """Prediction voting regressor for unfitted and/or fitted estimators.

    This is a modification of sklearn's VotingRegressor that takes both
    unfitted and fitted estimators as input instead of only unfitted estimators.

    To perform predictions it averages the individual regressor outputs.

    Parameters
    ----------
    unfitted_estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.unifitted_estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.

    fitted_estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingRegressor`` will *not* fit clones
        of these estimators.

    weights : array-like of shape (n_regressors,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted values before averaging. Uses uniform weights if `None`.
        When inputting both unfitted and fitted estimators, input the unfitted
        estimators' weights before the fitted ones.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    unfitted_estimators : list of (str, estimator) tuples
        The collection of fitted sub-estimators as defined in the constructor's ``unfitted_estimators``.

    fitted_estimators : list of (str, estimator) tuples
        The collection of fitted sub-estimators as defined in the constructor's ``fitted_estimators``.

    estimators_ : list of regressors
        The collection of fitted sub-estimators as defined in ``estimators``
        that are not 'drop'.

    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying regressor exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0

    See Also
    --------
    VotingClassifier : Soft Voting/Majority Rule classifier.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LinearRegression
    >>> from sklearn.ensemble import RandomForestRegressor
    >>> from sklearn.neighbors import KNeighborsRegressor
    >>> from modified_voting_estimators import ModifiedVotingRegressor
    >>> r1 = LinearRegression()
    >>> r2 = RandomForestRegressor(n_estimators=10, random_state=1)
    >>> r3 = KNeighborsRegressor()
    >>> X = np.array([[1, 1], [2, 4], [3, 9], [4, 16], [5, 25], [6, 36]])
    >>> y = np.array([2, 6, 12, 20, 30, 42])
    >>> er = ModifiedVotingRegressor([('lr', r1), ('rf', r2), ('r3', r3)])
    >>> print(er.fit(X, y).predict(X))
    [ 6.8...  8.4... 12.5... 17.8... 26...  34...]
    """
    def __init__(
        self,
        unfitted_estimators=[],
        fitted_estimators=[],
        *,
        weights=None,
        n_jobs=None,
        verbose=False
    ):
        super().__init__(
            estimators=unfitted_estimators+fitted_estimators,
            weights=weights,
            n_jobs=n_jobs,
            verbose=verbose
        )
        self.unfitted_estimators=unfitted_estimators
        self.fitted_estimators=fitted_estimators
    
    def fit(self,X=None,y=None,sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None.
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features.Can be None if only fitted_estimators
            are present.

        y : array-like of shape (n_samples,), default=None.
            Target values.Can be None if only fitted_estimators are present.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        if X is None or y is None:
            assert len(self.unfitted_estimators)==0,'X and y can only be None there are no unfitted_estimators'
            
        # TODO test with only fitted and X,y as None
        if X is not None and y is not None:
            self._validate_params()
            y = column_or_1d(y, warn=True)
        return fit(self,X, y, sample_weight)


class VotingClassifier(VotingClassifier):
    """Soft Voting/Majority Rule classifier for unfitted and/or fitted estimators.

    This is a modified implementation of sklearn's VotingClassifier that
    takes fitted estimators in addition to unfitted estimators as input. 

    Parameters
    ----------
    unfitted_estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will fit clones
        of those original estimators that will be stored in the class attribute
        ``self.unfitted_estimators_``. An estimator can be set to ``'drop'`` using
        :meth:`set_params`.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    fitted_estimators : list of (str, estimator) tuples
        Invoking the ``fit`` method on the ``VotingClassifier`` will *not* fit clones
        of these estimators. They will simply be saved in ``self.fitted_estimators_``.
        An estimator can be set to ``'drop'`` using :meth:`set_params`.

        .. versionchanged:: 0.21
            ``'drop'`` is accepted. Using None was deprecated in 0.22 and
            support was removed in 0.24.

    voting : {'hard', 'soft'}, default='hard'
        If 'hard', uses predicted class labels for majority rule voting.
        Else if 'soft', predicts the class label based on the argmax of
        the sums of the predicted probabilities, which is recommended for
        an ensemble of well-calibrated classifiers.

    weights : array-like of shape (n_classifiers,), default=None
        Sequence of weights (`float` or `int`) to weight the occurrences of
        predicted class labels (`hard` voting) or class probabilities
        before averaging (`soft` voting). Uses uniform weights if `None`.
        When both unfitted_estimators and fitted_estimators are present, 
        the weights for unfitted estimators shuould be provided *before* the
        fitted ones.

    n_jobs : int, default=None
        The number of jobs to run in parallel for ``fit``.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

        .. versionadded:: 0.18

    flatten_transform : bool, default=True
        Affects shape of transform output only when voting='soft'
        If voting='soft' and flatten_transform=True, transform method returns
        matrix with shape (n_samples, n_classifiers * n_classes). If
        flatten_transform=False, it returns
        (n_classifiers, n_samples, n_classes).

    verbose : bool, default=False
        If True, the time elapsed while fitting will be printed as it
        is completed.

        .. versionadded:: 0.23

    Attributes
    ----------
    unfitted_estimators : list of (str, estimator) tuples
        The collection of fitted sub-estimators as defined in the constructor's ``unfitted_estimators``.

    fitted_estimators : list of (str, estimator) tuples
        The collection of fitted sub-estimators as defined in the constructor's ``fitted_estimators``.


    estimators_ : list of classifiers
        The collection of fitted sub-estimators as defined in ``unfitted_estimators``
        and ``fitted_estimators`` that are not 'drop' in that order.


    named_estimators_ : :class:`~sklearn.utils.Bunch`
        Attribute to access any fitted sub-estimators by name.

        .. versionadded:: 0.20

    le_ : :class:`~sklearn.preprocessing.LabelEncoder`
        Transformer used to encode the labels during fit and decode during
        prediction.

    classes_ : ndarray of shape (n_classes,)
        The classes labels.

    n_features_in_ : int
        Number of features seen during :term:`fit`. Only defined if the
        underlying classifier exposes such an attribute when fit.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Only defined if the
        underlying estimators expose such an attribute when fit.
        .. versionadded:: 1.0

    See Also
    --------
    VotingRegressor : Prediction voting regressor.

    Examples
    --------
    >>> import numpy as np
    >>> from sklearn.linear_model import LogisticRegression
    >>> from sklearn.naive_bayes import GaussianNB
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> from modified_voting_estimators import ModifiedVotingClassifier
    >>> clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
    >>> clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
    >>> clf3 = GaussianNB()
    >>> X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
    >>> y = np.array([1, 1, 1, 2, 2, 2])
    >>> eclf1 = ModifiedVotingClassifier(unfitted_estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
    >>> eclf1 = eclf1.fit(X, y)
    >>> print(eclf1.predict(X))
    [1 1 1 2 2 2]
    >>> np.array_equal(eclf1.named_estimators_.lr.predict(X),
    ...                eclf1.named_estimators_['lr'].predict(X))
    True
    >>> eclf2 = ModifiedVotingClassifier(estimators=[
    ...         ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...         voting='soft')
    >>> eclf2 = eclf2.fit(X, y)
    >>> print(eclf2.predict(X))
    [1 1 1 2 2 2]

    To drop an estimator, :meth:`set_params` can be used to remove it. Here we
    dropped one of the estimators, resulting in 2 fitted estimators:

    >>> eclf2 = eclf2.set_params(lr='drop')
    >>> eclf2 = eclf2.fit(X, y)
    >>> len(eclf2.estimators_)
    2

    Setting `flatten_transform=True` with `voting='soft'` flattens output shape of
    `transform`:

    >>> eclf3 = ModifiedVotingClassifier(estimators=[
    ...        ('lr', clf1), ('rf', clf2), ('gnb', clf3)],
    ...        voting='soft', weights=[2,1,1],
    ...        flatten_transform=True)
    >>> eclf3 = eclf3.fit(X, y)
    >>> print(eclf3.predict(X))
    [1 1 1 2 2 2]
    >>> print(eclf3.transform(X).shape)
    (6, 6)
    """
    def __init__(
        self,
        unfitted_estimators=[],
        fitted_estimators=[],
        *,
        voting="hard",
        weights=None,
        n_jobs=None,
        flatten_transform=True,
        verbose=False,
    ):
        super().__init__(
            estimators=unfitted_estimators+fitted_estimators,
            voting=voting,
            weights=weights,
            n_jobs=n_jobs,
            flatten_transform=flatten_transform,
            verbose=verbose
        )
        self.unfitted_estimators=unfitted_estimators
        self.fitted_estimators=fitted_estimators
    
    def fit(self, X=None, y=None, sample_weight=None):
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features), default=None
            Training vectors, where `n_samples` is the number of samples and
            `n_features` is the number of features. Can be None if only fitted_estimators
            are present.

        y : array-like of shape (n_samples,), default=None
            Target values. Can be None if only fitted_estimators are present.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionadded:: 0.18

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        if X is None or y is None:
            assert len(self.unfitted_estimators)==0,'X and y can only be None there are no unfitted_estimators'

        self._validate_params()
        if y is not None:
            check_classification_targets(y)
            if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
                raise NotImplementedError(
                    "Multilabel and multi-output classification is not supported."
                )

        # To deal with the case where estimator's training data contained a different set of
        # labels we fit a LabelEncoder with a union of all of the fitted_estimators's classes
        # (labels they saw during training) and the labels present in y. We'll use this
        # encoder during prediction.
        
        # The classes all the models saw/will see during training
        all_classes=_unique(y) if isinstance(y,np.ndarray) else None
        for _,est in self.fitted_estimators:
            all_classes=np.union1d(all_classes,est.classes_) if all_classes is not None else est.classes_

        self.le_ = LabelEncoder().fit(all_classes)
        self.classes_ = self.le_.classes_

        return fit(self,X, y, sample_weight)

    def transform_probas(self,probas,est):
        """
        Given the predicted probabilities from an estimator, it outputs
        a numpy array that accounts for labels not present in the
        estimator's training data. 
        """
        out=np.zeros((probas.shape[0],self.le_.classes_.size))
        for local_col,class_i in enumerate(est.classes_):
            real_col=np.where(self.le_.classes_==class_i)[0][0]
            out[:,real_col]=probas[:,local_col]

        return out

    def _collect_probas(self, X):
        """
        Collect results from clf.predict calls.
        Modified to call transform_probas on every model's prediction output
        before assemblying.
        """
        return np.asarray([self.transform_probas(clf.predict_proba(X),clf) for clf in self.estimators_])

    
    def predict(self, X):
        """Predict class labels for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        maj : array-like of shape (n_samples,)
            Predicted class labels.
        """
        check_is_fitted(self)
        if self.voting == "soft":
            maj = np.argmax(self.predict_proba(X), axis=1)

        else:  # 'hard' voting
            # Transform the predictions using our LabelEncoder to have a universal mapping.
            predictions = np.asarray(
                [self.le_.transform(est.predict(X)) for est in self.estimators_ if est in self.unfitted_estimators_] +
                [self.le_.transform(est.predict(X)) for est in self.estimators_ if est in self.fitted_estimators_]
                ).T

            maj = np.apply_along_axis(
                lambda x: np.argmax(np.bincount(x, weights=self._weights_not_none)),
                axis=1,
                arr=predictions,
            )
        maj = self.le_.inverse_transform(maj)
        return maj
