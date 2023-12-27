#!/usr/bin/env python3

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import json
import numpy as np

# Factory methods
class CombineColumns(BaseEstimator, TransformerMixin):
    '''
    This class is used in an sklearn pipeline to combine several columns
    into one from a pandas dataframe
    '''

    def __init__(self, columns, name):
        self.columns = columns
        self.name = name

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)

        try:
            X[self.name] = X[self.columns].apply(
                lambda row: ' '.join(row.values.astype(str)), axis=1)
            return X
        except KeyError:
            cols_error = list(set(self.columns) - set(X.columns))
            raise KeyError(
                "The DataFrame does not include the columns: %s" % cols_error)


class RemoveMissing(BaseEstimator, TransformerMixin):
    '''
    This class drops na from a pandas dataframe and resets the index
    '''

    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        try:
            X.dropna(subset=self.columns, inplace=True)
            # # We check if the index needs to be reset
            # by checking if any numbers are missing
            # # Only do this if we have an index set
            if type(X.index) != pd.core.indexes.base.Index:
                myIterable = range(len(X))
                notInIndex = [x for x in myIterable if x not in X.index]

                if notInIndex:
                    X.reset_index(inplace=True)
            return X
        except KeyError:
            raise KeyError("Remove Missing failed", KeyError)


class ColumnSelector(BaseEstimator, TransformerMixin):
    '''
    Transformer to select a single column from the data
    frame to perform additional transformations on
    Use on text columns in the data
    '''

    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key]


class NumpyEncoder(json.JSONEncoder):
    '''
    Use this when dealing with ndarrays inside
    your dictionary
    that needs to be sent out as JSON objects.
    '''

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


# Yield successive n-sized chunks from a list


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def running_mean(x, N):
    '''Returns the running average given a list of values x'''
    cumsum = np.cumsum(np.insert(x, 0, 0))
    if N > 0:
        return (cumsum[N:] - cumsum[:-N]) / float(N)
    else:
        return 0

# for "pairs" of any length


def chunkwise(t, size=2):
    it = iter(t)
    return zip(*[it]*size)
