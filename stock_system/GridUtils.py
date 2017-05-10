from sklearn.model_selection import GridSearchCV
# from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.model_selection import TimeSeriesSplit
import timeit


class GridSearcher(object):
    '''
    Init with a list of (model, dict) tuples.
    The dictionary for each model has parameters specific to that model.
    '''
    def __init__(self):
        gd_boost = {
            'learning_rate': [1, 0.05, 0.02, 0.01],
            'max_depth': [2, 4, 6],
            'max_features': ['sqrt', 'log2'],
            'n_estimators': [50, 100, 1000]
        }

        ada_boost = {
            'learning_rate': [1, 0.05, 0.02, 0.01],
            'base_estimator__max_depth': [2, 4, 6],
            'base_estimator__max_features': ['sqrt', 'log2'],
            'n_estimators': [50, 100, 1000]
        }

        decision_tree = {
            'max_depth': [2, 4, 6, 10],
            'min_samples_split': [5, 10, 20],
            'min_samples_leaf': [3, 5, 9, 17]
        }

        random_forest_grid = {
            'n_estimators': [50, 100, 1000],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 10, 50],
        }

        knn_grid = {
            'n_neighbors': [5, 10, 15],
            'weights': ['uniform', 'distance'],
        }

        svc_grid = {
            'C': [0.1, 1.0, 5],
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'degree': [2, 3],
            'shrinking': [True, False],
            'gamma': [.02, .08, 1.5, 5]
        }

        self.models = [
            (GradientBoostingClassifier(), gd_boost),
            (AdaBoostClassifier(DecisionTreeClassifier()), ada_boost),
            (DecisionTreeClassifier(), decision_tree),
            (RandomForestClassifier(), random_forest_grid),
            (KNeighborsClassifier(), knn_grid),
            (SVC(), svc_grid)
        ]

    def grid_search_reporter(self, X_train, y_train):
        searches = []

        for model, feature_dict in self.models:
            print "Running grid search for {}".format(model.__class__.__name__)
            start_time = timeit.default_timer()
            gs = self.grid_search(model, feature_dict, X_train, y_train)

            print "====={}=====".format(gs.best_estimator_.__class__.__name__)
            print "  processing time: {} ".format(timeit.default_timer() - start_time)
            print "  f1 score: {}".format(gs.best_score_)
            print "  best params: {}".format(gs.best_params_)

            searches.append(gs)

        return searches

    def grid_search(self, model, feature_dict, X_train, y_train):
        gscv = GridSearchCV(
            model,
            feature_dict,
            n_jobs=-1,
            verbose=True,
            scoring='f1',
            iid=False,
            cv=TimeSeriesSplit(n_splits=5).split(X_train)
        )

        gscv.fit(X_train, y_train)
        return gscv
