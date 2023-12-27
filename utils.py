import pandas as pd
import numpy as np
import scipy.stats as ss
import scipy.cluster.hierarchy as sch
import warnings
from collections import Counter
from pyitlib import discrete_random_variable as drv
import scipy
from scipy.stats import chi2
warnings.filterwarnings('ignore')


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def theils_u(x, y):
    """
    Calculates Theil's U statistic (Uncertainty coefficient) for categorical-
    categorical association. This is the uncertainty of x given y: value is
    on the range of [0,1] - where 0 means y provides no information about
    x, and 1 means y provides full information about x.
    This is an asymmetric coefficient: U(x,y) != U(y,x)
    Wikipedia: https://en.wikipedia.org/wiki/Uncertainty_coefficient
    Parameters:
    -----------
    x : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    y : list / NumPy ndarray / Pandas Series
        A sequence of categorical measurements
    Returns:
    --------
    float in the range of [0,1]
    """
    s_xy = drv.entropy_conditional(x, y)
    x_counter = Counter(x)
    total_occurrences = sum(x_counter.values())
    p_x = list(map(lambda n: n / total_occurrences, x_counter.values()))
    s_x = ss.entropy(p_x)
    if s_x == 0:
        return 1
    else:
        return (s_x - s_xy) / s_x


def comp_assoc(df):
    columns = df.columns
    corr = pd.DataFrame(index=columns, columns=columns)
    for i in range(0, len(columns)):
        for j in range(i, len(columns)):
            if i == j:
                corr.loc[columns[i], columns[j]] = 1.0
            else:
                ji = theils_u(
                    df[columns[i]],
                    df[columns[j]])
                ij = theils_u(
                    df[columns[j]],
                    df[columns[i]])
                corr.loc[columns[i], columns[j]] = ij \
                    if not np.isnan(ij) and abs(ij) < np.inf else 0.0
                corr.loc[columns[j], columns[i]] = ji \
                    if not np.isnan(ji) and abs(ji) < np.inf else 0.0
    corr.fillna(value=np.nan, inplace=True)
    return corr


def cluster_correlations(corr_mat, indices=None):
    """
    Apply agglomerative clustering in order to sort
    a correlation matrix.

    Parameters:
    -----------
    - corr_mat : a square correlation matrix (pandas DataFrame)
    - indices : cluster labels [None]; if not provided we'll do
        an aglomerative clustering to get cluster labels.

    Returns:
    --------
    - corr : a sorted correlation matrix
    - indices : cluster indexes based on the original dataset

    Example:
    --------
    >> assoc = associations(
        customers,
        plot=False
    )
    >> correlations = assoc['corr']
    >> correlations, _ = cluster_correlations(correlations)
    """
    if indices is None:
        X = corr_mat.values
        d = sch.distance.pdist(X)
        L = sch.linkage(d, method='complete')
        indices = sch.fcluster(L, 0.5 * d.max(), 'distance')
    columns = [corr_mat.columns.tolist()[i]
               for i in list((np.argsort(indices)))]
    corr_mat = corr_mat.reindex(columns=columns).reindex(index=columns)
    return corr_mat, indices


def chi_square_contingency_test(data,
                                groupby,
                                columns,
                                alpha=0.05,
                                verbose=False,
                                print_p_value=False):
    """
    https://pygot.wordpress.com/2018/06/28/hypothesis-testing-in-python/

    P-Value – this is the estimated probability of
              achieving this value by chance from the sample set.

    Generally, if we receive a p-value of less than 0.05,
        we can reject the null hypothesis
        and state that there is a significant difference.

    P-values give us an idea of how confident we can be in a result.

    Just because we don’t have enough data
     to detect a difference doesn’t mean that there isn’t one.
    """
    # create contingency table of validity findings
    # ct = pd.crosstab(data[alpha], data[beta])
    ct = pd.crosstab(index=data[groupby], columns=data[columns])
    Observed_Values = ct.values
    chi_square_statistic, p_value, df, ex = scipy.stats.chi2_contingency(ct)
    Expected_Values = ex
    # critical_value
    critical_value = chi2.ppf(q=1-alpha, df=df)
    # print out the collective
    if print_p_value:
        print('p-value:', p_value)
    if verbose:
        print('contingency_table:\n', ct)
        print("Observed Values:\n", Observed_Values)
        print("Expected Values:\n", Expected_Values)
        print('Significance level: ', alpha)
        print('Degree of Freedom: ', df)
        print('chi-square statistic:', chi_square_statistic)
        print('critical_value:', critical_value)
        print('p-value:', p_value)
    if chi_square_statistic >= critical_value:
        print("Reject H0."
              "There is a relationship between 2 categorical variables")
    else:
        print("Retain H0."
              "There is no relationship between 2 categorical variables")
    if p_value <= alpha:
        print("Reject H0."
              "There is a relationship between 2 categorical variables")
    else:
        print("Retain H0"
              "There is no relationship between 2 categorical variables")
