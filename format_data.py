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
Original DF Setup
"""

#Sample
df = ddf.read_csv('./csv_data/train_transaction.csv').compute()
dfIdentity = ddf.read_csv('./csv_data/train_identity.csv').compute()
df = df.set_index('TransactionID').join(dfIdentity.set_index('TransactionID'))
del dfIdentity
df = df.drop(['TransactionDT'], 1)

df['isFraud'] = df['isFraud'].astype(bool)
#Encode ProductCD
df['ProductCD'] = df['ProductCD'].astype('category')
df = pd.concat([df,pd.get_dummies(df['ProductCD'], prefix='ProductCD',dummy_na=False)],axis=1).drop(['ProductCD'],axis=1)

#Encode M Variables
df['M1'] = df['M1'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M1'], prefix='M1',dummy_na=True)],axis=1).drop(['M1'],axis=1)
df['M2'] = df['M2'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M2'], prefix='M2',dummy_na=True)],axis=1).drop(['M2'],axis=1)
df['M3'] = df['M3'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M3'], prefix='M3',dummy_na=True)],axis=1).drop(['M3'],axis=1)
df['M4'] = df['M4'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M4'], prefix='M4',dummy_na=True)],axis=1).drop(['M4'],axis=1)
df['M5'] = df['M5'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M5'], prefix='M5',dummy_na=True)],axis=1).drop(['M5'],axis=1)
df['M6'] = df['M6'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M6'], prefix='M6',dummy_na=True)],axis=1).drop(['M6'],axis=1)
df['M7'] = df['M7'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M7'], prefix='M7',dummy_na=True)],axis=1).drop(['M7'],axis=1)
df['M8'] = df['M8'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M8'], prefix='M8',dummy_na=True)],axis=1).drop(['M8'],axis=1)
df['M9'] = df['M9'].astype('category')
df = pd.concat([df,pd.get_dummies(df['M9'], prefix='M9',dummy_na=True)],axis=1).drop(['M9'],axis=1)

#One hot Encode Card Variables with few categories
df['card4'] = df['card4'].astype('category')
df = pd.concat([df,pd.get_dummies(df['card4'], prefix='card4',dummy_na=True)],axis=1).drop(['card4'],axis=1)
df['card6'] = df['card6'].astype('category')
df = pd.concat([df,pd.get_dummies(df['card6'], prefix='card6',dummy_na=True)],axis=1).drop(['card6'],axis=1)

#Make browser type usable seperate of it's version
df['id_31'] = df['id_31'].fillna('NotProvided')
df['id_31'] = df.id_31.str.replace(r'(^.*[Cc]hrome.*$)', 'Chrome')
df['id_31'] = df.id_31.str.replace(r'(^.*[Ff]irefox.*$)', 'Firefox')
df['id_31'] = df.id_31.str.replace(r'(^.*[Ee]dge.*$)', 'Edge')
df['id_31'] = df.id_31.str.replace(r'(^.*[Ii]e.*$)', 'Ie')
df['id_31'] = df.id_31.str.replace(r'(^.*[Ss]afari.*$)', 'Safari')
df['id_31'] = df.id_31.str.replace(r'(^.*[Ss]amsung.*$)', 'Samsung')
df['id_31'] = df.id_31.str.replace(r'(^.*[Aa]ol.*$)', 'Aol')
df['id_31'] = df.id_31.str.replace(r'(^.*[Oo]pera.*$)', 'Opera')
df['id_31'] = df.id_31.str.replace(r'(^.*[Gg]oogle.*$)', 'Google')
df['id_31'] = df.id_31.str.replace(r'(^.*[Aa]ndroid.*$)', 'Android')


#Target Encode Card Variables with many categories
##Set as categorical type
df['card1'] = df['card1'].astype("category")
df['card2'] = df['card2'].astype("category")
df['card3'] = df['card3'].astype("category")
df['card5'] = df['card5'].astype("category")
df['id_31'] = df['id_31'].astype("category")
df['R_emaildomain'] = df['R_emaildomain'].fillna('NotProvided')
df['R_emaildomain'] = df['R_emaildomain'].astype('category')
df['P_emaildomain'] = df['P_emaildomain'].fillna('NotProvided')
df['P_emaildomain'] = df['P_emaildomain'].astype('category')

###TARGET ENCODING WILL BE DONE ON THE TRAINING SET TO PREVENT DATA LEAKAGE


#Create a field for the transactions cents amount, treat as categorical (association with certain values more)
df['transactionCents'] = df['TransactionAmt'].sub(df['TransactionAmt'].astype(int)).mul(100).astype('category')

#Log transform the Transaction AMT Column
log_amt = list(map(lambda x: 0 if x <= 1 else np.log(x), df.TransactionAmt))

df['LogAmount'] = log_amt
box_amount = sb.boxplot(x=log_amt)

#Transformation Slide
hist_amount = sb.histplot(df.TransactionAmt, bins=25, kde=True)
hist_log_amount = sb.histplot(log_amt, bins=25, kde=True)
del log_amt

#Remove Outliers 
df = df[df.LogAmount < 9]

#Drop original TransactionAmt Column
df = df.drop(['TransactionAmt'], 1)

# Replace columns with > 5% (25,000) with a field for whether or not it had a value
columnsForTargetEncoding = ['card1', 'card2', 'card3', 'card5', 'P_emaildomain', 'R_emaildomain', 'transactionCents', 'id_31']
for col in df.columns:
    temp_df = df[col].isnull().sum()
    temp = temp_df > 25000
    if temp:
        #how many uniques?
        uniqueCount = df[col].nunique()
        if uniqueCount < 5: 
            df[col] = df[col].astype('category')
            df = pd.concat([df,pd.get_dummies(df[col], prefix=col,dummy_na=True)],axis=1).drop([col],axis=1)
        elif uniqueCount < 20:
            df[col].fillna('NotProvided')
            df[col] = df[col].astype('category')
            columnsForTargetEncoding.append(col)
        else:
            #If missing value is NaN
            df[col + '_na'] = np.where(df[col].isnull(), 1, 0)
            df = df.drop([col], 1)
del temp
del temp_df
del col

#Impute After target encoding

#Ensure column names are formated properly, set fraud to int bool 
df.columns = list(map(lambda c: re.sub(r'[^a-zA-Z0-9_]', '',  c), df.columns))

df['isFraud'] = df['isFraud'].astype('int')

#Save pre split/encoded df
df.to_pickle("./PreSplitPreTargetEncodePreImputeDf.pkl")

# Is this a rare event problem where the minority class < 2%.
fraud = df.query('isFraud == True')
nonfraud = df.query('isFraud == False')
ratio = len(fraud) / len(df)*100
print(str.format("The percentage of fraud transactions is {0:.3f}%", ratio))

