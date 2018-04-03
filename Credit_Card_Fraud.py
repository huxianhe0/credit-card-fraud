import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_validate
from sklearn.metrics import roc_auc_score, roc_curve, classification_report
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression

# XGBoost
import xgboost as xgb
    
# Microsoft Light GBM
import lightgbm as lgb

# Synthetic Minority Over-Sampling Technique (SMOTE)
from imblearn.over_sampling import SMOTE

# SMOTE resampling function

def resample_and_balance(X,y, num=1000):
    
    # Start with original data
    
    """
    Resample the data so that the fraudulent and legitimate purchases are 
    balanced and contain 'num' examples of each.
    
    Returns a dataframe containing 'num' randomly selected original legitimate 
    transactions and  'num' fraudulent transactions consiting of the original 
    fraudulent transactions plus transactions simulated usng SMOTE.
    
    SMOTE (Synthetic Minority Over-Sampling Technique) is a way to create
    a balanced dta set, using the imblearn python package
    """
    # Select original fraudulent transactions
    df_all = pd.concat([X,y], axis=1)
    df_fraud_original = df_all.loc[df_all.Class == 1]
    num_original = len(df_fraud_original)

    # SMOTE Resampling
    X_ = X.as_matrix()
    y_ = np.ravel(y.as_matrix())
    ratio_dict = {0: sum(y.Class == 0), 1: num}
    
    smote = SMOTE(ratio=ratio_dict, random_state=42)
    
    X_resampled, y_resampled = smote.fit_sample(X_, y_)
    
    resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), 
                           pd.DataFrame(y_resampled, columns=['Class'])], 
                           axis=1)
    
    # Select simulated fraudulent transactions created using SMOTE                      
    df_fraud_resample = resampled.loc[resampled.Class == 1]
    df_fraud_resample = df_fraud_resample.sample(n = num - num_original, replace=False)
    
    df_fraud = pd.concat([df_fraud_original, df_fraud_resample], axis=0)

    # Sample original legitimate transactions to match 
    # length of resampled original trnsactions
    df_legit = df_all.loc[df_all.Class == 0]
    df_legit = df_legit.sample(n=num, replace=False, random_state=42)

    return(df_legit, df_fraud)

def dist_compare(var):
    '''
    compare  two histograms of the legitimate and fraudulent transactions
    '''
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(1,2,1, yticks=[])
    ax.set_xlim(min(df_legit[var]) -1 , max(df_legit[var]) + 1)
    ax.set_title('Legitimate Purchases, %s' %(var))
    sns.distplot(df_legit[var], bins=100)
    ax = fig.add_subplot(1,2,2, yticks = [])
    ax.set_xlim(min(df_legit[var]) -1 , max(df_legit[var]) + 1)
    ax.set_title('Fraudulent Purchases, %s' %(var))
    sns.distplot(df_fraud[var], bins=100)
    plt.show()

# Read Data
df = pd.read_csv('creditcard.csv')

X = df.drop(['Class'], axis=1)
X = X.drop(['Time'], axis=1)
y = pd.DataFrame()
y['Class'] = df.Class

# Normalize 'Amount' feature

X['Amount'] = StandardScaler().fit_transform(X.Amount.values.reshape(-1,1))
X.describe()

print('There are %i instances of fraud from %i transactions' %(np.sum(y), len(y)))

# Set aside 25% of data to test on later

X, X_holdout, y, y_holdout = train_test_split(X, y, test_size = 0.25, random_state=42)

# Resampled dataset to augment the fraudulent transaction data

num = 4000 #int(213605 / 2) # Match Training Set

df_legit, df_fraud = resample_and_balance(X, y, num = num) 

df_all = pd.concat([df_legit, df_fraud], axis=0)

X_r = df_all.drop(['Class'], axis=1)
y_r = df_all.Class

# Convert y to 1-D
y = np.ravel(y)

# Dataframe to hold feature coeficients

FeatureCoefs = pd.DataFrame(index=X.columns)

# EDA

# Visualize Variables

dist_compare('V4')

# Visualize Relationships Between Variables

var1 = 'V1' # x axis
var2 = 'V2' # y axis

joinplot_1 = sns.jointplot(x=var1, y=var2, data=df_legit, kind='kde')
joinplot_2 = sns.jointplot(x=var1, y=var2, data=df_fraud, kind='kde')

fig = plt.figure(figsize=(12,12))
for plot in [joinplot_1, joinplot_2]:
    for ax in plot.fig.axes:
        fig._axstack.add(fig._make_key(ax), ax)

#subplots size adjustment
fig.axes[0].set_position([0.05, 0.05, 0.4,  0.4])
fig.axes[1].set_position([0.05, 0.45, 0.4,  0.05])
fig.axes[2].set_position([0.45, 0.05, 0.05, 0.4])
fig.axes[3].set_position([0.55, 0.05, 0.4,  0.4])
fig.axes[4].set_position([0.55, 0.45, 0.4,  0.05])
fig.axes[5].set_position([0.95, 0.05, 0.05, 0.4])

### Dimensionality Reduction

# Since there are only 29 features being used, no need
# to reduce the dimnsionality of the data before perfoming
# tSNE

tsne = TSNE(n_components=2, learning_rate=200, random_state=42)

X_new = tsne.fit_transform(X_r)

XY = pd.DataFrame(np.column_stack((X_new, y_r)), columns=['x','y','label'])

fraud = XY.loc[XY['label'] == 1].values
legit = XY.loc[XY['label'] == 0].values

ax, fig = plt.subplots()

ax = sns.regplot(fraud[:,0], fraud[:,1], color='r', scatter=True, fit_reg=False, scatter_kws={'s': 4})
ax = sns.regplot(legit[:,0], legit[:,1], color='b', scatter=True, fit_reg=False, scatter_kws={'s': 4})
ax.legend(labels = ['Fraudulent','Legitimate'], loc='best')
plt.show()


### Logistic Regression

# Tune parameters
params = [1, 5, 10, 15, 20]

for param in params:
    logreg = LogisticRegression(penalty='l1', # Default 
                                tol=0.0001,   # Default
                                C = param)    # 5
    
    scoring = ['precision', 'recall', 'f1', 'roc_auc']
    
    scores = cross_validate(logreg, X, y, scoring=scoring, cv=5, return_train_score=False)
    print('Tol: %f' %(param))
    print('Logistic Regression CV Scores, Original Data:\nprecision:\t%f \nreacll:\t\t%f\nf1:\t\t%f\nroc-auc:\t%f' 
            %(np.mean(scores.get('test_precision')), 
            np.mean(scores.get('test_recall')),
            np.mean(scores.get('test_f1')),
            np.mean(scores.get('test_roc_auc'))
            )
         )

logreg = LogisticRegression(penalty='l1', # Default 
                            tol=0.0001,   # Default
                            C = 5)        #5
    
# Original Data

logreg.fit(X,y)
preds = logreg.predict(X_holdout)
preds_proba = logreg.predict_proba(X_holdout)[:,1]

fpr, tpr, thresholds = roc_curve(y_holdout, preds_proba)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logisitc Regression ROC Curve on Holdout Set, Original Data')
plt.show()

auc = roc_auc_score(y_holdout, preds_proba)

# Scale and save feature weights

logreg_coefs_o = logreg.coef_
logreg_coefs_o = StandardScaler().fit_transform(logreg_coefs_o[0].reshape(-1,1))
FeatureCoefs['logreg_original'] = logreg_coefs_o

# Resampled Dataset

logreg.fit(X_r.as_matrix(), y_r)
preds_r = logreg.predict(X_holdout)
preds_proba_r = logreg.predict_proba(X_holdout)[:,1]

fpr, tpr, thresholds = roc_curve(y_holdout, preds_proba_r)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Logistic Regression')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logisitc Regression ROC Curve on Holdout Set, Resampled Data')
plt.show()

auc_r = roc_auc_score(y_holdout, preds_proba_r)

# Scale and save feature weights

logreg_coefs_r = logreg.coef_
logreg_coefs_r = StandardScaler().fit_transform(logreg_coefs_r[0].reshape(-1,1))
FeatureCoefs['logreg_resample'] = logreg_coefs_r

# Logreg Summaries
print('Original Logreg AUC Score: %f' %(auc))
print('Resampled Logreg AUC Score: %f' %(auc_r))
print('Original Data')
print(classification_report(y_holdout, preds))
print('Resampled Data')
print(classification_report(y_holdout, preds_r))

### XGBoost

# Tune parameters
xgb_model = xgb.XGBClassifier()

parameters = {'max_depth': [7], #5, Re: 7
              'learning_rate': [0.001],
              'n_estimators': [1000],
              'objective': ['binary:logistic'],
              'gamma': [0], #0, Re: 0
              'min_child_weight': [6], #6, Re:6
              'subsample': [0.7], #0.7, Re 0.7 
              'colsample_bytree': [0.5]} #0.5, Re 0.5

clf = GridSearchCV(xgb_model, 
                   param_grid=parameters, 
                   scoring='roc_auc', 
                   n_jobs=-1, 
                   cv=3
                  )
# Compare original data to resampled data

clf.fit(X,y)

# Scale and save feature weights
xgb_coefs_o = clf.best_estimator_.feature_importances_
xgb_coefs_o = StandardScaler().fit_transform(xgb_coefs_o.reshape(-1,1))
FeatureCoefs['xgb_original'] = xgb_coefs_o

best_parameters, score, _ = max(clf.grid_scores_, key=lambda x: x[1])
print('AUC score:', score)

for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

preds = clf.predict(X_holdout)
preds_proba = clf.predict_proba(X_holdout)[:,1]

fpr, tpr, thresholds = roc_curve(y_holdout, preds_proba)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='XGB Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGB ROC Curve on Holdout Set, Original Data')
plt.show()

auc = roc_auc_score(y_holdout, preds_proba)

# Resampled Dataset

clf.fit(X_r, y_r)
preds_r = clf.predict(X_holdout)
preds_proba_r = clf.predict_proba(X_holdout)[:,1]
X_holdout.shape
X_r.shape
fpr, tpr, thresholds = roc_curve(y_holdout, preds_proba_r)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='XGB Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGB ROC Curve on Holdout Set, Resampled Data')
plt.show()

auc_r = roc_auc_score(y_holdout, preds_proba_r)

# Scale and save feature weights
xgb_coefs_r = clf.best_estimator_.feature_importances_
xgb_coefs_r = StandardScaler().fit_transform(xgb_coefs_r.reshape(-1,1))
FeatureCoefs['xgb_resample'] = xgb_coefs_r

# XGB Summaries
print('Original XGB AUC Score: %f' %(auc))
print('Resampled XGB AUC Score: %f' %(auc_r))
print('Original Data')
print(classification_report(y_holdout, preds))
print('Resampled Data')
print(classification_report(y_holdout, preds_r))

### Light GBM

parameters = {'num_leaves': [70],
              'max_depth' : [7],
              'n_estimators' : [100],
              'min_child_weight' : [0.0001],
              'min_child_samples' : [50, 75, 100],
              'subsample ' : [1],
              'colsample_bytree': [0.5] ,
              'reg_alpha' : [0],
              'reg_lambda' : [0],
              'random_state': [42]}
              

gbm_model = lgb.LGBMClassifier(boosting_type = 'gbdt',
                               objective = 'binary',
                               learning_rate = 0.01,
                               #num_boost_round = 1000,
                               n_jobs = -1,
                               silent = False)

clf = GridSearchCV(gbm_model, 
                   param_grid=parameters,
                   scoring='roc_auc', 
                   n_jobs=-1, 
                   cv=3
                  )


clf.fit(X, y)

best_parameters = clf.best_params_
score = clf.best_score_
print('Raw AUC score:', score)

for param_name in sorted(best_parameters.keys()):
    print("%s: %r" % (param_name, best_parameters[param_name]))

# Scale and save feature weights
lgb_coefs_o = clf.best_estimator_.feature_importances_
lgb_coefs_o = StandardScaler().fit_transform(lgb_coefs_o.reshape(-1,1))
FeatureCoefs['lgb_original'] = lgb_coefs_o

# Original Data
preds_proba = clf.predict(X_holdout)

fpr, tpr, thresholds = roc_curve(y_holdout, preds_proba)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='Light GBM Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Light GBM ROC Curve on Holdout Set, Original Data')
plt.show()

auc = roc_auc_score(y_holdout, preds_proba)

# Resampled Dataset
clf.fit(X_r, y_r)
preds_r = clf.predict(X_holdout)

fpr, tpr, thresholds = roc_curve(y_holdout, preds_proba_r)

plt.plot([0,1], [0,1], 'k--')
plt.plot(fpr, tpr, label='XGB Model')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGB ROC Curve on Holdout Set, Resampled Data')
plt.show()

auc_r = roc_auc_score(y_holdout, preds_proba_r)

# Scale and save feature weights
lgb_coefs_r = clf.best_estimator_.feature_importances_
lgb_coefs_r = StandardScaler().fit_transform(lgb_coefs_r.reshape(-1,1))
FeatureCoefs['lgb_resample'] = lgb_coefs_r

# Light GBM Summaries
print('Original LGBM AUC Score: %f' %(auc))
print('Resampled LGBM AUC Score: %f' %(auc_r))
print('Original Data')
print(classification_report(y_holdout, preds))
print('Resampled Data')
print(classification_report(y_holdout, preds_r))

# Plot Feature Importance

# Take absolute value of feature importance, sort by model with hichest recall 
FeatureCoefs = FeatureCoefs.abs()

FeatureCoefs = FeatureCoefs.sort_values(by=['logreg_resample'], ascending=True)

FeatureCoefs.head()

fig, ax = plt.subplots(figsize=(12,12))
y_pos = np.arange(0,29,1)
ax.barh(y=y_pos + 0.2, width=FeatureCoefs.logreg_resample, height=0.2)
ax.barh(y=y_pos, width=FeatureCoefs.xgb_resample, height=0.2)
ax.barh(y=y_pos - 0.2, width=FeatureCoefs.lgb_resample, height=0.2)
ax.legend(['Logreg Feature Weights', 'XGBoost Feature Weights', 'Light GB Feature Weights'], loc=(0.6, 0.08), fontsize='large')
plt.yticks(y_pos, FeatureCoefs.index)
plt.ylabel('Feature', fontsize='large')
plt.xlabel('ABS of Feature Weight', fontsize='large')
plt.title('Comparison of Feature Importance by Model')
plt.show()


