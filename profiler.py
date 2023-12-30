#!/usr/bin/env python3
import re
from ast import literal_eval
import pandas as pd
import numpy as np
from sklearn import preprocessing
from gensim.parsing import preprocessing as gpp
from pandas_eda.external_tools import CombineColumns


class DataReader(object):
    """
    A class to read in data to a pandas dataframe
    """

    def __init__(self,
                 file_path=None,
                 combine_cols=None,
                 out_col=None,
                 drop_cols=None,
                 stop_words=False,
                 minsize=3,
                 date_cols=None):

        self.file_path = file_path
        self.combine_cols = combine_cols
        self.out_col = out_col
        self.drop_cols = drop_cols
        self.stop_words = stop_words
        self.minsize = minsize
        self.date_cols = date_cols
        self.df = None

    def combine_columns(self):
        '''
        Combines the contents of "combine_cols" into one column, "out_col"
        '''
        cc = CombineColumns(columns=[self._normalize_column_names(
            i) for i in self.combine_cols], name=self.out_col)
        self.df = cc.fit_transform(self.df)

    def normalize_column_names(self):
        '''
        Normalizes column names to fit the format of "column_name".
        '''
        # we may have an "unnamed" column. This is the index from the csv
        for i in self.df.columns:
            if "Unnamed: 0" in i:
                self.df.drop(i, axis=1, inplace=True)

        self.df.rename(
            columns=lambda x: self._normalize_column_names(x), inplace=True)

    def _normalize_column_names(self, col):
        text = re.sub(r'(^[_])', r'', col)
        text = re.sub(r'(^[ ])', r'', text)
        text = re.sub(r'(^[0-9])', r'c/\1', text)
        text = re.sub(r'([/.$ "])', r'_', text)
        text = re.sub(r'([.])', r'_', text)
        text = text.lower()
        return text

    def clean_text(self, text, rs=True, minsize=4):
        '''
        Preprocesses the text of a given column.
        Removes newline markers, makes all text lowercase,
        strips numeric, multiple whitespace
        and alphanumeric text.

        Removes stopwords if requested and strips
        short words with a minimum size
        specified by minsize.
        '''
        # order matters!
        text = re.sub(r"\\n", ' ', text)
        text = re.sub(r"\n", ' ', text)
        # text = re.sub(r"/", 'x', text)
        text = str(text).lower()
        text = gpp.strip_numeric(text)
        text = gpp.strip_multiple_whitespaces(text)
        text = gpp.strip_non_alphanum(text)
        if rs:
            text = gpp.remove_stopwords(text)
        if minsize > 0:
            text = gpp.strip_short(text, minsize=minsize)
        return text

    def get_data(self):
        '''
        Handles loading data from different
        sources without having to specify the source.

        Currently supports .csv, .xlsx, .xlsm, .xls, .pkl.bz2, .pkl
        '''
        if isinstance(self.file_path, str):
            if self.file_path.endswith('.pkl.bz2') \
                    or self.file_path.endswith('.pkl'):
                self.df = pd.read_pickle(self.file_path)
            elif self.file_path.endswith('.csv'):
                self.df = pd.read_csv(self.file_path,
                                      error_bad_lines=False,
                                      skipinitialspace=True,
                                      encoding="utf-8",
                                      parse_dates=self.date_cols,
                                      infer_datetime_format=True,
                                      float_precision=None,
                                      low_memory=False)

            elif self.file_path.endswith(('.xls', '.xlsm', '.xlsx')):
                self.df = pd.read_excel(self.file_path)
            elif self.file_path.endswith(('.parq')):
                self.df = pd.read_parquet(self.file_path)
            else:
                print(f"Unsupported file type {self.file_path}")

        elif isinstance(self.file_path, list):
            files = self.file_path
            temp_df = pd.DataFrame()
            for i in files:
                self.file_path = i
                self.get_data()
                temp_df = pd.concat([temp_df, self.df],
                                    sort=False, ignore_index=True)
            self.df = temp_df


class DataProfiler(DataReader):
    '''
    Profiler is used to get a better view of the data
    Inherits from DataReader
    Returns a dataframe of useful information
    Requires pandas

    Usage example
    __________

    from utils import DataProfiler

    filename = "somefile.csv"
    cpd = DataProfiler()
    cpd.file_path = filename
    cpd.get_data()
    cpd.normalize_columns()
    cpd.get_profile()


    Parameters
    __________
    max_first_to_rest_ratio = 97/3,        # max threshold of the frequency of the top value as compared to the rest of the values (e.g. 97% vs 3%)
    manual_overrides = list()              # optional list of comments and actions that can be manually provided to override the automatic propositions of the data profiler):
    '''

    def __init__(self,
                 df=None,
                 max_first_to_rest_ratio=95,
                 manual_overrides=None,
                 **kwargs):
        super().__init__()
        self.max_first_to_rest_ratio = max_first_to_rest_ratio

        if manual_overrides is None:
            manual_overrides = []

        if isinstance(df, pd.DataFrame):
            self.df = df

        self.manual_overrides = manual_overrides
        self.profile = pd.DataFrame()
        

    def get_profile(self):
        '''
        Main method used to get a better view of the data

        Returns
        ---------
        a dataframe with useful information/statistics
        of each feature in the dataframe.
        '''
        # always reset when run
        self.profile = pd.DataFrame()

        try:
            nr = self.df.shape[0]
            for i in self.df.columns.tolist():
                if i not in self.manual_overrides:
                    if "Unnamed" in i:
                        self.df.drop(i, inplace=True, axis=1)
                        continue
                    vector = self.df[i]

                    # original data type(s) of the vector
                    data_type_at_source = vector.dtype.name

                    # absolute and relative number of unique
                    # values in the vector originally
                    unique_values = vector.nunique()

                    # absolute and ratio of null (NA) values in the vector
                    missing_values = vector.isnull().sum()

                    # get a sample of the data
                    sample_data = f"{list(vector[0:3])}"
                    counts = vector.value_counts(normalize=True, dropna=False)*100
                    most_frequent_value = str(list(counts.items())[0])
                    if (counts >= self.max_first_to_rest_ratio).any():
                        summary = "Low variance feature"
                        proposition = "Consider Dropping"
                    else:
                        summary = ""
                        proposition = ""

                    # Logic to deduce necessary pre-processing
                    # Assumes first entry is not empty
                    if (self._valid_date(str(vector[0]))):
                        summary = 'Date'
                        proposition = "Make datetime"
                    elif (self._is_categorical(i)):
                        summary = "Categorical"
                        if unique_values > 20:
                            proposition = "Label Encode"
                        else:
                            proposition = "One Hot Encode"
                    if missing_values == nr:
                        summary = "Empty column"
                        proposition = "Remove"
                    temp_df = pd.DataFrame(
                        {"Column Name": i,
                         "Number of Rows": nr,
                         "Source Data Type": data_type_at_source,
                         "Unique Values": unique_values,
                         "Ratio of Unique Values": unique_values/nr,
                         "Missing Values": missing_values,
                         "Ratio of missing values": missing_values/nr,
                         "Completeness": (1-(missing_values/nr)),
                         "Most Frequent Value": most_frequent_value,
                         "Summary": summary,
                         "Sample data": f"{sample_data}",
                         "Proposition": proposition}, index=[0])
                    self.profile = pd.concat([self.profile, temp_df],
                        ignore_index=True)

        except Exception as e:
            print(e)
            print("Could not profile the dataframe")
            pass

        return self.profile.reset_index(drop=True)

    def _batmaNaN(self, vec):
        '''
        Tests if nan is repeated more than 5 times. Just for fun.

        Parameters
        ----------
        vec  :list or series

        Returns
        ---------
        a (boolean) True if the given vector
        contains nan repeated more than
        5 times consecutively and False otherwise
        '''
        temp_vector = vec.isnull().astype(int).groupby(
            vec.notnull().astype(int).cumsum()).sum()
        return bool(any(temp_vector >= 5))

    def _is_categorical(self, column):
        '''
        Tests whether a given column name is a categorical column. To be used
        internally.

        Parameters
        ----------
        column  :string
            column name.

        Returns
        ---------
        a (boolean) True if the given column
        name is a categorical and False
        otherwise.
        '''
        ratio = self.df[column].nunique()/self.df.shape[0]
        if ratio > 0.001 and (self.df[column].dtype.name == 'object'
                              or self.df[column].dtype.name == 'category'):
            return True

        return False

    def _is_numeric(self, column):
        '''
        Tests whether a given column name is a numerical column. To be used
        internally.

        Parameters
        ----------
        column  :string
            column name.

        Returns
        ---------
        a (boolean) True if the given column
        name is a numerical and False
        otherwise.
        '''
        return column.dtype.kind in 'if'  # (i)nt or (f)loat

    def fix_types(self, col_types=None):
        '''
        Changes datatypes of the columns in the current df.

        Parameters
        ----------
        col_types : A dictionary of column names followed by their respective
                    types to be used. It accapts all typse supported in
                    Pandas (e.g. 'int', 'float', 'category', 'object', etc)
        '''
        for col, ntype in col_types.items():
            self.df[col] = self.df[col].astype(ntype)

    def remove_feature(self, col=None):
        '''
        Drops a given column from the dataframe (df)

        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). 
        '''
        if isinstance(col, str):
            col = [col]
        self.df.drop(col, axis=1, inplace=True)

    def transform_feature(self,
                          col=None,
                          func_str=None,
                          new_col_name=None,
                          addtional_params=None,
                          **kwargs):
        '''
        Uses an given function to apply a transformation on a specific column.
        This only works for columns with numerical values. If new_col_name
        is given, then the new transformed column will be saved in new column.
        Otherwise, it will override the same column.

        In the background, this uses df.apply() which applies a given funciton.
        This supports any arbitary transform function (lambda function) or scaling
        function from sklearn package (see scale_feature()).

        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all numerical columns will
                be used.
        func_str  : string (of function)
            Function to apply to each column/row. Accepts both lambda functions
            (enclosed within a string, e.g. 'lambda x: x**2') or a user-defined
            function (e.g. 'my_transformer').
        new_col_name : string
            If given, a new column will be created for the transformed column.
        addtional_params : dictionary
            any additional parameters to be used for external functions (like
            sklearn's scaling functions -- see scale_feature).
        **kwargs: additional arguments to be passed to panda's apply
        '''

        func = literal_eval(func_str)
        if func.__name__ != '<lambda>' \
                and func.__module__ != 'sklearn.preprocessing.data':
            raise TypeError('func is not recognized')
        if col is None:
            print("Please specify a column name or list of columns")
        elif isinstance(col, str):
            col = [col]

        if new_col_name is None:  # inplace
            if addtional_params is not None:
                self.df[col] = self.df[col].apply(
                    func, **addtional_params, **kwargs)
            else:
                self.df[col] = self.df[col].apply(func, **kwargs)
        else:
            # if type(new_col_name) is str:
            #    new_col_name = [new_col_name]
            if addtional_params is not None:
                self.df[new_col_name] = self.df[col].apply(
                    func, **addtional_params, **kwargs)
            else:
                self.df[new_col_name] = self.df[col].apply(func, **kwargs)

    def cc_split(self, var, new_col_name, num_split, outcome_col):
        """
        Split a continuous variable 'var' into
        discrete groups based on distribution of 'controls'
        - assumes 'controls' are `0`; cases' are `1` in 'outcome_col'
        """
        controls = self.df[self.df[str(outcome_col)] == 0]
        bins = pd.qcut(controls[str(var)],
                       int(num_split),
                       retbins=True,
                       labels=False)[1]
        new = []
        for i in self.df[str(var)].values:
            if np.isnan(i):
                new.append(1)
            elif i == min(bins):
                new.append(1)
            elif i < min(bins):
                new.append(1)
            elif i > max(bins):
                new.append(num_split - 1)
            else:
                new.append(np.searchsorted(bins, i, 'left'))

        self.df[new_col_name] = new

    def scale_feature(self, col=None, scaling=None, scaling_parms=None):
        '''
        Scales a given set  of numerical columns. This only works for columns
        with numerical values.

        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all numerical columns will
                be used.
        scaling  : {'zscore', 'minmax_scale' (default), 'scale', 'maxabs_scale',
                    'robust_scale'}
            User-defined scaling functions can also be used through self.transform_feature
        scaling_parms : dictionary
            any additional parameters to be used for sklearn's scaling functions.

        '''

        if scaling is None:
            scaling = 'minmax_scale'

        if scaling == 'zscore':
            scaling = 'lambda x: (x - x.mean()) / x.std()'
        elif scaling == 'minmax_scale' and scaling_parms is None:
            scaling_parms = {'feature_range': (0, 1), 'axis': 0}
        elif scaling == 'scale' and scaling_parms is None:
            scaling_parms = {'with_mean': True, 'with_std': True, 'axis': 0}
        elif scaling == 'maxabs_scale' and scaling_parms is None:
            scaling_parms = {'axis': 0}
        elif scaling == 'robust_scale' and scaling_parms is None:
            # 'quantile_range':(25.0, 75.0),
            scaling_parms = {'with_centering': True,
                             'with_scaling': True, 'axis': 0}
        else:
            raise TypeError('UNSUPPORTED scaling TYPE')

        self.transform_feature(col=col, func_str=scaling,
                               addtional_params=scaling_parms)

    def encode_categorical_feature(self, col=None):
        '''
        Convert categorical columns into dummy/indicator columns.

        Parameters
        ----------
        col : a string of a column name, or a list of many columns names or
                None (default). If col is None, all categorical columns will
                be used.
        '''
        # Create a label (category) encoder object
        le = preprocessing.LabelEncoder()

        if col is None:
            print("Please specify a column name or list of columns")
        elif isinstance(col, str):
            col = [col]

        for x in col:
            # Fit the encoder to the pandas column
            le.fit(self.df[x])
            # Apply the fitted encoder to the pandas column
            self.df[f"{x}_le"] = le.transform(self.df[x])

    def _valid_date(self, datestring):
        '''
        Tests if the given string is a datetime using regex.
        This may not cover all possible date patterns

        Parameters
        ----------
        datestring: string

        Returns
        ----------
        Returns boolean True if the pattern matches a date pattern and False otherwise

        '''
        try:
            # Expect the
            # Matches a date in yyyy-mm-dd format from 1900-01-01 through 2099-12-31, with a choice of four separators.
            match01 = re.search(
                r'^(19|20)\d\d[- /.]([1-9]|0[1-9]|1[012])[- /.]([1-9]|0[1-9]|[12][0-9]|3[01])', datestring)
            # To match a date in mm/dd/yyyy format
            match02 = re.search(
                r'^([1-9]|0[1-9]|1[012])[- /.]([1-9]|0[1-9]|[12][0-9]|3[01])[- /.](19|20)\d\d', datestring)
            # For dd-mm-yyyy format
            match03 = re.search(
                r'^([1-9]|0[1-9]|[12][0-9]|3[01])[- /.]([1-9]|0[1-9]|1[012])[- /.](19|20)\d\d', datestring)
            # match just the year
            match04 = re.search(r'^(19|20)\d\d', datestring)

            if match01 or match02 or match03 or match04 is not None:
                return True
        except ValueError:
            pass

        return False
