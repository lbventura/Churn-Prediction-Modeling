import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats
from scipy.stats import chi2_contingency
from scipy.stats import chi2

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance

from sklearn.model_selection import GridSearchCV

def feature_importance_indicator(X_df, y_df, target_var, classifier = 'ExtraTrees'):
    """Computes feature importance using a tree classifier. The user can choose between
    RandomTrees or ExtraTrees - Designed by Pedro Leal

    Args:
        X_df (pd.DataFrame): Contains predictive features
        y_df (pd.Series): Contains target feature
        target_var (str): Name of the target variable
        classifier (str, optional): Defaults to 'ExtraTrees'.

    Returns:
        tuple: Tuple containing variables, their respective column indices and
        feature importances
    """
    ## Defining the extra trees classifer - using the default with more estimators
    
    if classifier == 'ExtraTrees':
        spec_clf = ExtraTreesClassifier(n_estimators=250,
                              random_state=42)
    elif classifier == 'RandomTrees':
        spec_clf = RandomForestClassifier(n_estimators=250,
                              random_state=42)
    
    spec_clf.fit(X_df, y_df)
    
    vars_ = X_df.columns
    importances = spec_clf.feature_importances_
    clf_std = np.std([ele.feature_importances_ for ele in spec_clf.estimators_], axis = 0)
    
    indices = np.argsort(importances)[::-1]
    
    print('-'*20)
    print(f"Feature ranking ({classifier}):")

    for f in range(X_df.shape[1]):
        print("%d. %s. %d (%f)" % (f + 1, vars_[f], indices[f] , importances[indices[f]]))
        
    print('\n')
    
    plt.figure()
    plt.title(f"Feature importances ({classifier})")
    plt.bar(range(X_df.shape[1]), importances[indices], color="y", yerr=clf_std[indices], align="center")
    plt.xticks(range(X_df.shape[1]), X_df.iloc[:,indices].columns, rotation = 90)
    plt.xlim([-1, X_df.shape[1]])
    plt.tight_layout()
    plt.show()
    
    return vars_ , indices, importances

def feature_importance_permutation(X_df,y_df,target_var):
    """Computes feature importance using a permutation importance of a
    RandomForestClassifier - Designed by Pedro Leal

    Args:
        X_df (pd.DataFrame): Contains predictive features
        y_df (pd.Series): Contains target feature
        target_var (str): Name of the target variable
    """
    forest_2 = RandomForestClassifier(n_estimators=250,
                              random_state=42)
    forest_2.fit(X_df, y_df)
    
    result = permutation_importance(forest_2, X_df, y_df, n_repeats=10,
                                random_state=42)  
    ##Procedure is similar to above, but we are using permutations now
    
    perm_sorted_idx = np.argsort(result.importances_mean)[::-1]

    sorted_vars = np.array(X_df.columns.tolist())[perm_sorted_idx]
    
    print('-'*20)
    for f in range(len(sorted_vars)):
        print("%d. %s. (%f)" % (f + 1, sorted_vars[f], result.importances_mean[perm_sorted_idx][f]))
    
    print('\n')
    
    plt.figure()
    plt.title("Feature importances (Random Forest)")
    plt.bar(range(X_df.shape[1]), result.importances_mean[perm_sorted_idx], color="y", align="center")
    plt.xticks(range(X_df.shape[1]), X_df.iloc[:,perm_sorted_idx].columns, rotation = 90)    
    plt.xticks(rotation=90)
    plt.show()

def KS_test(df,target_var,target_var_values=[0,1]):
    """Computes the Komolgorov-Smirnov test - Designed by Pedro Leal

    Args:
        df (pd.DataFrame): Contains both feature and target data
        target_var (str): Name of the target variable
        target_var_values (list, optional): Labels of test result. Defaults to [0,1],
        with 0 being no stat significance and 1 being stat significance.
    """
    df_zero = df[df[target_var]==target_var_values[0]]
    df_one = df[df[target_var]==target_var_values[1]]
    
    ks_dict = {}
    
    for var in df.columns:
        ks_aux = stats.ks_2samp(df_zero[var], df_one[var])
        ks_dict[var] = ks_aux
    
    ks_vars = list(ks_dict.keys())
    ks_vars.remove(target_var)
    
    ks_results = [ks_dict[key][0] for key in ks_dict.keys() if key != target_var]
    ks_p_value = [ks_dict[key][1] for key in ks_dict.keys() if key != target_var]
    

    ks_results_idx_sorted = np.argsort(ks_results)[::-1]
    

    ks_significant = np.array([1 if x < 0.05 else 0 for x in ks_p_value])[ks_results_idx_sorted]
    ks_results_sorted = np.array(ks_results)[ks_results_idx_sorted]
    ks_vars_sorted = np.array(ks_vars)[ks_results_idx_sorted]
    

    plt.title("KS Test results")
    sns.barplot(x=ks_vars_sorted,y=ks_results_sorted, hue = ks_significant)
    plt.xticks(rotation=90)
    plt.show()
    
    #print(ks_significant)

def chi_squared_test(input_df, cat_var_1, cat_var_2, prob = 0.95):
    """Computes the chi-squared test for two categorical variables

    Args:
        input_df (pd.DataFrame): Contains both feature and target data
        cat_var_1 (str): Name of the first categorical variable
        cat_var_2 (str): Name of the second categorical variable
        prob (float, optional): Confidence interval. Defaults to 0.95.
    """
    print('\n')
    print(f'Cat var 1: {cat_var_1} , Cat var 2: {cat_var_2}')
    print('\n')
    
    input_array = pd.crosstab(input_df[cat_var_1], input_df[cat_var_2])
    
    stat, p, dof, expected = chi2_contingency(input_array)
    
    ## Interpret test statistic
    
    critical = chi2.ppf(prob, dof)
    
    print('-'*20)
    print('Probability = %.3f, critical stat = %3.f, statistic = %3.f' % (prob, critical, stat))
    
    if abs(stat) >= critical:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    print('-'*20)
    
    ## Interpret p-value
    
    alpha = 1 - prob
    
    print('-'*20)
    print('Significance (alpha) = %.3f, p = %3.f' % (alpha, p))
    
    if p <= alpha:
        print('Dependent (reject H0)')
    else:
        print('Independent (fail to reject H0)')
    print('-'*20)

def EstimatorGridSearch(X,y,model_dict,params_dict,score_method=None):
    """Computes a grid search across several models and respective parameters.
    - Designed by Pedro Leal

    Args:
        X (pd.DataFrame): Contains predictive feature data
        y (pd.Series): Contains target variable
        model_dict (dict): Dictionary with model selection
        params_dict (dict): Dictionary with parameter choices for each model
        score_method (str, optional): Choice of score method. Defaults to None,
        in which case it uses the pre-defined scoring function of Sklearn.

    Returns:
        pd.DataFrame: Contains a summary of model performance
    """
    grid_searches = {}

    for key in model_dict.keys():
        params = params_dict[key]
        gs = GridSearchCV(model_dict[key], params, cv=5, n_jobs=-1,
                              verbose=1, scoring=score_method, refit=True,
                              return_train_score=True) ##Doing cross validation
        gs.fit(X,np.ravel(y))
        grid_searches[key] = gs

    def row(key, scores, params):
        d = {
            'Estimator': key,
            'Min_score': min(scores),
            'Max_score': max(scores),
            'Mean_score': np.mean(scores),
            'Std_score': np.std(scores),
                }
        return pd.Series({**params,**d})


    rows = []
    for key in model_dict.keys():
        params = grid_searches[key].cv_results_['params']
        scores=[]
        for i in range(grid_searches[key].cv):
            key2 = "split{}_test_score".format(i)
            r = grid_searches[key].cv_results_[key2]
            scores.append(r.reshape(len(params),1))
        
        all_scores = np.hstack(scores)
    
        for p, s in zip(params,all_scores):
            rows.append((row(key, s, p)))
        
    df = pd.concat(rows, axis=1).T.sort_values(['Mean_score'], ascending=False)
    
    columns = ['Estimator', 'Min_score', 'Mean_score', 'Max_score', 'Std_score']
    columns = columns + [c for c in df.columns if c not in columns]
    
    return df[columns]

def naive_classifier(y_test, p_exit):
    """Generates a naïve classification based on the training sample frequencies 

    Args:
        y_test (pd.Series): Contains the target variable
        p_exit (float64): In-sample frequency for the probability of having exited
    """
    return np.random.choice([1,0], size = y_test.shape[0], p=[p_exit, 1- p_exit])