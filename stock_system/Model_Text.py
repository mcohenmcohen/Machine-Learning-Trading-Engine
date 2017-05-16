# TF-IDF
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split



v = TfidfVectorizer(stop_words='english',max_features=300)
X = v.fit_transform(df.text).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, y)
