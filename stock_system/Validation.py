# Validation
import numpy as np


def walk_forward_train(model, X, y, start_index=100, window_range=10):
    '''
    Perform walk forward testing on the input data
    '''
    # X = series.values
    n_records = len(X)
    for i in range(start_index, n_records):
        X_train, X_test = X[0:i], X[i:i+1]
        y_train, y_test = y[0:i], y[i:i+1]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        score = model.score(y_pred, y_test)
        print('train=%d, test=%d' % (len(train), len(test)))
        print '- score: ', score
