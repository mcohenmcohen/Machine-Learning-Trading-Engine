# ModelUtils
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, recall_score, accuracy_score
from sklearn.metrics import roc_auc_score, mean_squared_error, r2_score, f1_score


class ModelUtils(object):

    def __init__(self):
        self.models = []

    def grid_search(self, model, X_train, y_train):
        random_forest_grid = {
            'n_estimators': [50, 100, 1000],
            'max_features': ['sqrt', 'log2'],
            'min_samples_leaf': [1, 2, 10, 50],
        }
        gscv = GridSearchCV(
            model,
            random_forest_grid,
            n_jobs=-1,
            verbose=True,
            scoring='f1'
        )
        gscv.fit(X_train, y_train)
        print "  f1 score: {}".format(gscv.best_score_)
        print "  params: {}".format(gscv.best_params_)

    def print_scores(self, y_test, y_hat):
        relevant_metrics = [precision_score, recall_score, accuracy_score, roc_auc_score,
                            mean_squared_error, r2_score, f1_score]
        for metric in relevant_metrics:
            m = metric(y_test, y_hat)
            # scores[metric.__name__] = m
            print metric.__name__, ' = ', m
