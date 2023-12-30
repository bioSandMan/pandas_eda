
from sklearn.feature_selection import RFECV
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from statsmodels.stats.outliers_influence import variance_inflation_factor    

class MultiCollinearity:
    """
    A class to test for and correct collinearity
    """
    def __init__(
        self,
        df,
        y_col,
        corr_plot=False,
        verbose=True,
    ):
        if y_col not in df.columns:
            raise ValueError(f"{y_col} must be present in the dataframe")

        self.df = df
        self.verbose = verbose
        self.corr_plot = corr_plot

        self.data_quality_checked = False
        self.data_cleaned = False

        self.X = self.df.drop(columns=[y_col])
        self.y = self.df[y_col]
        self.n_col = self.X.shape[1]
        self.all_features = self.X.columns

    def _data_quality_checks(self):
        """Check if the input dataframe is valid"""
        assert self.X.shape[0] > 0, "Empty dataframe (0 rows)"
        assert self.X.shape[1] > 0, "Empty dataframe (0 columns)"

        if self.y.dtype == "object":
            raise ValueError("y must be numeric")

        self.data_quality_checked = True

    def _data_cleaning(self):
        """Clean the input dataframe"""
        # discard rows with missing values
        self.X = self.X.dropna()
        if self.verbose:
            print(
                f"INFO: Discarded {self.n_col - self.X.shape[1]} rows with missing values"
            )
        # update the number of columns
        self.n_col = self.X.shape[1]

        # discard columns with constant values
        self.X = self.X.loc[:, self.X.apply(pd.Series.nunique) != 1]
        if self.verbose:
            print(
                f"INFO: Discarded {self.n_col - self.X.shape[1]} columns with constant values"
            )
        # update the number of columns
        self.n_col = self.X.shape[1]

        # discard categorical columns
        self.X = self.X.select_dtypes(exclude=["object"])
        if self.verbose:
            print(f"INFO: Discarded {self.n_col - self.X.shape[1]} categorical columns")
        # update the number of columns
        self.n_col = self.X.shape[1]

        # update the list of all columns
        self.all_features = self.X.columns

        if self.n_col <= self.min_vars_to_keep:
            raise Exception(
                f"Fewer than {self.min_vars_to_keep} valid variables found. Lower min_vars_to_keep and rerun."
            )

        self.data_cleaned = True

    def _corr_with_y(self):
        """Calculate the correlation between each feature and the output vector"""
        corr_with_y = {}
        for c in self.all_features:
            corr_with_y[c] = abs(self.y.corr(self.X[c]))
        return corr_with_y

    def _get_corr_matrix(self, modified=False):
        """Calculate the correlation matrix"""
        # create a correlation matrix with absolute values
        corr_matrix = abs(self.X.corr())
        if modified:
            # we will replace the diagonal values with -1 (to facilitate later operations)
            arr = corr_matrix.values
            np.fill_diagonal(arr, -1)
            corr_matrix = pd.DataFrame(
                arr, columns=corr_matrix.columns, index=corr_matrix.index
            )
        return corr_matrix

    def _plot_corr_matrix(self, corr_matrix, end=False):
        """Plot the correlation matrix"""
        import seaborn as sns
        import matplotlib.pyplot as plt

        # plot the correlation matrix
        sns.set(style="white")
        f, ax = plt.subplots(figsize=(8, 6), dpi=150)

        # Generate a mask for the upper triangle
        corr_mask = np.zeros_like(corr_matrix, dtype=bool)
        corr_mask[np.triu_indices_from(corr_mask)] = True

        ttl = "Before" if end == False else "After"

        ax.set_title(
            f"Correlation Matrix ({ttl} Feature Reduction)",
            fontsize=14,
            fontweight="bold",
        )

        sns.heatmap(
            corr_matrix,
            mask=corr_mask,
            cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True,
            ax=ax,
            vmin=-1,
            vmax=1,
        )
        plt.show()

    def pairwise_corr(
        self,
        min_vars_to_keep=10,
        corr_tol=0.9,
    ):
        """Drop features that are highly correlated with each other"""

        self.min_vars_to_keep = min_vars_to_keep
        self.corr_tol = corr_tol

        if self.data_quality_checked is False:
            self._data_quality_checks()
        if self.data_cleaned is False:
            self._data_cleaning()

        if self.verbose:
            print(f"INFO: Input number of features: {self.n_col:,}")

        # correlation with the output vector
        self.corr_with_y = self._corr_with_y()

        # clean correlation matrix
        corr_matrix = self._get_corr_matrix(True)

        if self.corr_plot:
            self._plot_corr_matrix(corr_matrix)

        # create a counter to keep track of the number of kept features
        surv_ct = len(corr_matrix)

        # we will keep dropping features until we reach the minimum number of features
        # or the highest (absolute) pairwise correlation is below the threshold
        # the corr_matrix will shrink as we drop features
        while surv_ct >= self.min_vars_to_keep:
            # identify the location of the highest correlation
            idx = np.argmax(corr_matrix)
            i, j = np.unravel_index(idx, corr_matrix.shape)

            # grab the value of that highest correlation
            row = corr_matrix.columns[i]
            col = corr_matrix.columns[j]
            highly_corr_pair = [row, col]

            # correlation between these two features
            highest_corr_val = corr_matrix.loc[row, col]

            if highest_corr_val > self.corr_tol:
                # get their corr with y (we need those to break the tie)
                corr_with_y_vals = [self.corr_with_y[row], self.corr_with_y[col]]
                # the feature that has lower corr with y is the loser
                loser_pos = np.argmin(corr_with_y_vals)
                loser = highly_corr_pair[loser_pos]
                winner = highly_corr_pair[~loser_pos]
                if self.verbose:
                    print(
                        f"DROP: {loser}\t[Highly correlated with {winner} ({highest_corr_val:.2f})]"
                    )
                corr_matrix = corr_matrix.drop(loser)
                corr_matrix = corr_matrix.drop(columns=loser)
                # another one bites the dust
                surv_ct -= 1
            else:
                if self.verbose:
                    print(
                        f"STOP: The max pairwise correlation ({highest_corr_val:.2f}) is below threshold ({self.corr_tol:.2f})"
                    )
                break

        # drop the discarded features from the original dataframe
        self.X = self.X[corr_matrix.columns]
        # update the number of columns
        self.n_col = self.X.shape[1]
        # update the list of all columns
        self.all_features = self.X.columns

        # return the reduced dataframe
        return pd.concat([self.X, self.y], axis=1)

    def multi_collin(
        self,
        cond_index_tol=30,
        min_vars_to_keep=10,
    ):
        """Drop features that contributes the most to multicollinearity"""

        self.min_vars_to_keep = min_vars_to_keep
        self.cond_ind_tol = cond_index_tol

        if self.data_quality_checked == False:
            self._data_quality_checks()
        if self.data_cleaned == False:
            self._data_cleaning()

        # recalculate the correlation matrix
        corr_matrix = self._get_corr_matrix()
        self.all_features = corr_matrix.keys()

        # create a counter to keep track of the number of kept features
        surv_ct = len(corr_matrix)

        # we will keep dropping features until we reach the minimum number of features
        # or the condition index if below the threshold
        # the corr_matrix will shrink as we drop features
        while True:
            self.all_features = corr_matrix.keys()

            # calculate the eigen values and vectors
            eigen_vals, eigen_vectors = np.linalg.eig(corr_matrix)
            # calculate the max of all conditinon indices

            # sometimes, the eigen values are negative
            c_ind = (max(eigen_vals) / abs(min(eigen_vals))) ** 0.5

            # if the condition index is lower than the threshold then multicolin is not an issue
            if c_ind <= self.cond_ind_tol:
                print(
                    f"STOP: The condition index ({c_ind:.2f}) is below threshold ({self.cond_ind_tol:.2f})"
                )
                break
            # if we have reached the minimum number of features, stop
            if surv_ct <= self.min_vars_to_keep:
                print(
                    f"STOP: Minimum number of features ({self.min_vars_to_keep}) has been reached"
                )
                break

            # find the eigenvector associated with with the min eigenvalue
            eigen_vector_w_lowest_eigen_val = eigen_vectors[np.argmin(eigen_vals)]
            # identify the feature that has the max weight on this eigen vector
            redundant_feature = self.all_features[
                np.argmax(eigen_vector_w_lowest_eigen_val)
            ]

            # drop it like it's hot
            if self.verbose:
                print(
                    f"DROP: {redundant_feature} [It leans the most on the eigen vector with the lowest eigen value ({min(eigen_vals):.2f})]"
                )
            corr_matrix = corr_matrix.drop(redundant_feature)
            corr_matrix = corr_matrix.drop(columns=redundant_feature)

        # drop the discarded features from the original dataframe
        self.X = self.X[corr_matrix.columns]
        # update the number of columns
        self.n_col = self.X.shape[1]
        # update the list of all columns
        self.all_features = self.X.columns

        if self.corr_plot:
            self._plot_corr_matrix(corr_matrix, end=True)

        # return the reduced dataframe
        return pd.concat([self.X, self.y], axis=1)


def calculate_vif_(X, thresh=5.0):
    X = X.assign(const=1)  # faster than add_constant from statsmodels
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables[:-1]])
    return X.iloc[:, variables[:-1]]

def get_low_variance_columns(dframe=None,
                             skip_columns=[],
                             thresh=0.16,
                             autoremove=False):
    """
    Wrapper for sklearn VarianceThreshold for use on pandas dataframes.
    """
    print("Finding low-variance features.")
    try:
        # get list of all the original df columns
        all_columns = dframe.columns

        # remove `skip_columns`
        remaining_columns = all_columns.drop(skip_columns)

        # get length of new index
        max_index = len(remaining_columns) - 1

        # get indices for `skip_columns`
        skipped_idx = [all_columns.get_loc(column)
                       for column
                       in skip_columns]

        # adjust insert location by the number of columns removed
        # (for non-zero insertion locations) to keep relative
        # locations intact
        for idx, item in enumerate(skipped_idx):
            if item > max_index:
                diff = item - max_index
                skipped_idx[idx] -= diff
            if item == max_index:
                diff = item - len(skip_columns)
                skipped_idx[idx] -= diff
            if idx == 0:
                skipped_idx[idx] = item

        # get values of `skip_columns`
        skipped_values = dframe.iloc[:, skipped_idx].values

        # get dataframe values
        X = dframe.loc[:, remaining_columns].values

        # instantiate VarianceThreshold object
        vt = VarianceThreshold(threshold=thresh)

        # fit vt to data
        vt.fit(X)

        # get the indices of the features that are being kept
        feature_indices = vt.get_support(indices=True)

        # remove low-variance columns from index
        feature_names = [remaining_columns[idx]
                         for idx, _
                         in enumerate(remaining_columns)
                         if idx
                         in feature_indices]

        # get the columns to be removed
        removed_features = list(np.setdiff1d(remaining_columns,
                                             feature_names))
        print("Found {0} low-variance columns."
              .format(len(removed_features)))

        # remove the columns
        if autoremove:
            print("Removing low-variance features.")
            # remove the low-variance columns
            X_removed = vt.transform(X)

            print("Reassembling the dataframe (with low-variance "
                  "features removed).")
            # re-assemble the dataframe
            dframe = pd.DataFrame(data=X_removed,
                                  columns=feature_names)

            # add back the `skip_columns`
            for idx, index in enumerate(skipped_idx):
                dframe.insert(loc=index,
                              column=skip_columns[idx],
                              value=skipped_values[:, idx])
            print("Succesfully removed low-variance columns.")

        # do not remove columns
        else:
            print("No changes have been made to the dataframe.")

    except Exception as e:
        print(e)
        print("Could not remove low-variance features. Something "
              "went wrong.")
        pass

    return dframe, removed_features

def recursive_feature_elimination(model, X, Y, min_features):
    rfecv = RFECV(model,
                  cv=5,
                  n_jobs=-1,
                  scoring='roc_auc',
                  min_features_to_select=min_features)

    rfecv = rfecv.fit(X, Y)

    # Best features
    cols = rfecv.get_support(indices=True)
    return X.columns[cols]