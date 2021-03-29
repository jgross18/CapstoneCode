#Set up enviornment
import re
import dask
import dask.dataframe as ddf
import numpy as np
import pandas as pd
import sklearn.metrics as metrics
from matplotlib import pyplot as plt
from sklearn import preprocessing
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (accuracy_score, auc, classification_report,
                             confusion_matrix, mean_squared_error,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

plt.style.use('ggplot')
%matplotlib inline

import re

import seaborn as sb
import statsmodels.api as sm
import statsmodels.formula.api as smf
from imblearn.over_sampling import RandomOverSampler
from category_encoders import TargetEncoder



#Utility Functions
def get_elasticnet(X, y, column_names, max_iter=1000):
    """
    Select significant variables according to Elastic Net.
    @param X <Pandas Dataframe>
    @param y <list>
    @columns <list>
    @max_iter <int>
    @returns <list>
    """ 
    def get_mse(alpha):
        model = ElasticNet(alpha=alpha, max_iter=max_iter).fit(X, y)   
        score = model.score(X, y)
        pred_y = model.predict(X)
        mse = mean_squared_error(y, pred_y)
        print()
        print('MSE after alpha ' + str(alpha) + ': ' + str(mse))
        print()
        return mse
        
    lowest_mse = 1.0
    best_alpha = 0.0
    alphas = [0.0001, 0.001, 0.01, 0.1, 0.3, 0.5, 0.7, 1]
    
    for a in alphas:
        mse = get_mse(a)
        if mse < lowest_mse:
            lowest_mse = mse
            best_alpha = a
        
    clf = ElasticNet(alpha=best_alpha, max_iter=max_iter)
    sfm = SelectFromModel(clf)
    sfm.fit(X, y)
    feature_indices = sfm.get_support()
    significant_features = []
    for c, b in zip(column_names, feature_indices):
        if b:
            significant_features.append(c)

    return significant_features

def get_balanced_accuracy(tpr, fpr):
    """
    Return average of Sensitivity and Specificity.
    """
    return (tpr + (1-fpr)) / 2

def get_tpr_fpr(cm):
    """
    Sensitivity: TruePos / (True Pos + False Neg) 
    Specificity: True Neg / (False Pos + True Neg)
    TN | FP
    -------
    FN | TP
    @param 2D array <list<list>>
    @returns <list<float>>
    """
    tn = float(cm[0][0])
    fp = float(cm[0][1])
    fn = float(cm[1][0])
    tp = float(cm[1][1])

    tpr = tp / (tp + fn)
    fpr = 1-(tn / (fp + tn))

    return [tpr, fpr]

def get_best_cutoff(actual, prob):  
    """
    Get the best cutoff according to Balanced Accuracy
    'Brute-force' technique - try all cutoffs from 0.01 to 0.99 in increments of 0.01

    @param actual <list<float>>
    @param prob <list<tuple<float, float>>>
    @returns <list<float>>
    """
    best_tpr = 0.0; best_fpr = 0.0; best_cutoff = 0.0; best_ba = 0.0; 
    cutoff = 0.0
    cm = [[0,0],[0,0]]
    while cutoff < 1.0:
        pred = list(map(lambda p: 1 if p >= cutoff else 0, prob))
        _cm = confusion_matrix(actual, pred)
        _tpr, _fpr = get_tpr_fpr(_cm)

        if(_tpr < 1.0):    
            ba = get_balanced_accuracy(tpr=_tpr, fpr=_fpr)

            if(ba > best_ba):
                best_ba = ba
                best_cutoff = cutoff
                best_tpr = _tpr
                best_fpr = _fpr
                cm = _cm

        cutoff += 0.01

    tn = cm[0][0]; fp = cm[0][1]; fn = cm[1][0]; tp = cm[1][1];
    return [best_tpr, best_fpr, best_cutoff, tn, fp, fn, tp]
    
# create confusion matrix
def get_predict_frame(actual, prob, model_name='Logit'):
    """
    Compute predicted based on estimated probabilities and best threshold. 
    Output predictions and confusion matrix.
    """
    # calculate TPR, FPR, best probability threshold
    tpr, fpr, cutoff, tn, fp, fn, tp = get_best_cutoff(actual, prob)
    accuracy = get_balanced_accuracy(tpr, fpr)
    auc = roc_auc_score(actual, prob)
    
    #print("Optimal prob. threshold is %0.3f: " % cutoff)
    yhat = list(map(lambda p: 1 if p >= cutoff else 0, prob))
    stats = pd.DataFrame(columns=['Model', 'TP', 'FP', 'TN', 'FN', 'Sensitivity', 'Specificity', 'Cutoff', 'Accuracy', 'AUC'],
                data=[[model_name, tp, fp, tn, fn, tpr, (1-fpr), cutoff, accuracy, auc]])

    print("Sensitivity: {0:.3f}%, Specificity: {1:.3f}%, Threshold: {2:.3f}".format(tpr*100, (1-fpr)*100, cutoff))
    return yhat, stats

def plot_roc(actual, prob):
    # calculate ROC curve
    fpr, tpr, thresholds = roc_curve(actual, prob)

    # plot ROC curve
    fig = plt.figure(figsize=(10, 10))
    # Plot the diagonal 50% line
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('Specificity')
    plt.ylabel('Sensitivity')
    plt.title('ROC Curve')
    plt.show()


"""
Split into training and validation sets
"""
df = pd.read_pickle("./PreSplitPreTargetEncodePreImputeDf.pkl")

## Modify
TARGET = 'isFraud'  
column_names = df.drop(['isFraud'], 1).columns

x_train, x_val, y_train, y_val = train_test_split(df.drop(['isFraud'], 1).values, df['isFraud'].values, test_size=0.1, random_state=123)

#create validation set
df_validate = pd.DataFrame(columns=column_names, data=x_val)
df_validate[TARGET] = y_val

# Oversample minority class in training set, create training df
ros = RandomOverSampler(sampling_strategy=0.1, random_state=123)
X_over, y_over = ros.fit_resample(x_train, y_train)

#Fraud ration on oversampled training set
fraud = [y for y in y_over if y == 1]
nonfraud = [y for y in y_over if y == 0]
ratio = len(fraud) / len(y_over)*100
print(str.format("The percentage of fraud in the oversampled training set is {0:.3f}%.", ratio))

#Create training set
df_over_train = pd.DataFrame(columns=column_names, data=X_over)
df_over_train['isFraud'] = y_over

#Encode columns for target endcoding on training set. Map to validation set
##Target Encode
enc = TargetEncoder(cols=columnsForTargetEncoding)
enc.fit(df_over_train.drop(['isFraud'], 1), df_over_train['isFraud'])
df_over_train = enc.transform(df_over_train.drop(['isFraud'], 1))
df_over_train[TARGET] = y_over
df_validate = enc.transform(df_validate.drop(['isFraud'], 1))
df_validate[TARGET] = y_val

#Impute
# Impute Rows
imp = SimpleImputer(missing_values=np.NaN)

idf = pd.DataFrame(imp.fit_transform(df_over_train))
idf.columns=df_over_train.columns
idf.index=df_over_train.index

df_over_train = idf
del idf 

imp = SimpleImputer(missing_values=np.NaN)

idf = pd.DataFrame(imp.fit_transform(df_validate))
idf.columns=df_validate.columns
idf.index=df_validate.index

df_validate = idf
del idf 

df_validate.to_pickle('./ValidationSet.pkl')
df_over_train.to_pickle('./TrainingSetOversample.pkl')

"""
DF Set up complete
"""