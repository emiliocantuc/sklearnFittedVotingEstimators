# VotingEstimators with for fitted and not fitted models

Modified versions of Sklearn's VotingClassifier and VotingRegressor that support as input already fitted estimators in addition to not-yet-fitted estimators (the default). Calling the fit function only fits the 'unfitted' estimators and predictions are
made using both fitted and unfitted estimators.

In addition, the modified VotingClassifier supports scenarios where the voting models have been trained on a different set of labels.

# Usage
Both the modified VotingClassifier and VotingRegressor act like sklearn's VotingClassifier and VotingRegressor respectively, the only exception being that their constructors take in both unfitted and fitted estimators:

```
from betterVotingEstimators import VotingClassifier,VotingRegressor

# Classification
eclf1 = VotingClassifier(
    fitted_estimators=[('lr', clf1), ('rf', clf2)],
    unfitted_estimators=[('gnb', clf3)],
    weights=[0.5,0.3,0.2], # gnb:0.5, lr:0.3, rf:0.2
    voting='soft',
    n_jobs=-1
    )

# The unfitted estimators are fitted with X and y
eclf1 = eclf1.fit(X,y)
preds=eclf1.predict(X)

# Regression
er = VotingRegressor(
    fitted_estimators=[('lr', r1), ('rf', r2)],
    unfitted_estimators=[('r3', r3)],
    weights=[0.6,0.2,0.2],
    n_jobs=-1
    )
preds=er.fit(X,y).predict(X)
```

# Example

```
# Example: Modified VotingClassifier with mix of fitted and unfitted estimators with different data

# Define our three classifiers
clf1 = LogisticRegression(multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=1)
clf3 = GaussianNB()
# They all train on the same X
X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])

# But on different ys
y1 = np.array(['A','A','A','B','B','B'])
y2 = np.array(['B','A','C','C','B','Z'])
y3 = np.array(['D','E','A','C','B','Z'])

# We fit two estimators
for est,y in zip([clf1,clf2],[y1,y2]):
    est.fit(X,y)

# Instantiate our VotingClassifier
eclf1 = VotingClassifier(
    fitted_estimators=[('lr', clf1), ('rf', clf2)],
    unfitted_estimators=[('gnb', clf3)],
    voting='hard'
)
# Call fit - which will fit the unfitted_estimators with X and y3
eclf1 = eclf1.fit(X, y3)

# Make our predictions
mod_preds=eclf1.predict(X)

```

More examples can be found in the `examples.ipynb` notebook.

# Note

To support soft voting the and scenarios where fitted estimators have seen different labels in their training set, 
the VotingClassifier's `_collect_probas` function had to be overridden. The function now calls `transform_probas` on the
output of each model's `predict_proba` to account for classes the model might have not seen. Making classification
predictions with soft voting may therefore be slower but it was sufficient for my purposes.  
