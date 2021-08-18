# -*- coding: utf-8 -*-
"""
Created on Fri May 28 13:53:46 2021

@author: Peter
"""
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 29 21:12:19 2020

@author: Peter
"""
#                     PART1: nested-cv elasticnet
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegressionCV
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_roc_curve
from sklearn import metrics
from sklearn.feature_selection import SelectFromModel
import dill
import statistics
from sklearn.metrics import roc_curve
import scipy

szempont="roc_auc"
random_state=1
cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=random_state)
cv=5
test_size=0.2
l1_ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

os.chdir("/data10/working_groups/balint_group/gargya.peter/R/uterus")

features = pd.read_table('uterus_rnaseq_VST.txt', sep="\t", index_col=0)
features = features[features['label']!="G2"]
features=features.replace("G1",0)
features=features.replace("G3",1)
print(features["label"].value_counts())

labels = np.array(features['label'])

features= features.drop('label', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(features, labels, 
                                                    test_size = test_size, random_state = random_state)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
print([X_train.shape,y_train.shape,X_test.shape,y_test.shape])

best_c=[]
best_l1=[]
max_auc=[]
valid_auc=[]
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
i=1
genes=[]

for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):
    train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
    train_target, val_target = y_train[train_idx], y_train[val_idx]
    
    best_model = LogisticRegressionCV(random_state=random_state, cv=cv, scoring=szempont, penalty="elasticnet", 
                          fit_intercept=True, solver="saga", Cs=10, n_jobs=15, max_iter=20000, 
                          l1_ratios=l1_ratios)
    best_model.fit(train_data, train_target)
    
    model = SelectFromModel(best_model, prefit=True, threshold=None)
    mask=model.get_support()
    train_data_genes = train_data.loc[:, mask]
    [genes.append(x) for x in train_data_genes.columns.values]
    coefs_array=+ best_model.coef_
    
    y_pred_prob = best_model.predict_proba(val_data)[:,1]
    valid_auc.append(roc_auc_score(val_target, y_pred_prob))
    best_c.append(best_model.C_)
    best_l1.append(best_model.l1_ratio_)
    max_auc.append(best_model.scores_[1].mean(axis=0).max())
    
    viz = plot_roc_curve(best_model, val_data, val_target,
                         name='ROC fold {}'.format(i), alpha=0.5, lw=1, ax=ax)
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
    i=i+1


dill.dump_session('globalsave_part1.pkl')

ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = metrics.auc(mean_fpr, mean_tpr)
print("Mean AUC: ", mean_auc)
std_auc = np.std(aucs)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/crossval_roc.pdf")

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(mean_fpr, mean_tpr, color='b', label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc), lw=2, alpha=.8)
ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label=r'$\pm$ 1 std. dev.')
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc="lower right")
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/crossval_roc_mean_only.pdf")
plt.close('all')


res=pd.DataFrame({"c":best_c, "l1":best_l1, "max auc":max_auc, "valid auc":valid_auc})
print(res)
print(statistics.mean(max_auc))
print(statistics.mean(valid_auc))

#%%
#               PART2: retrain model on whole dataset, predict G2
import os
os.chdir("/data10/working_groups/balint_group/gargya.peter/R/uterus")

import dill                          
dill.load_session('globalsave_part1.pkl')


features = pd.read_table('uterus_rnaseq_VST.txt', sep="\t", index_col=0)
features = features[features['label']!="G2"]
features=features.replace("G1",0)
features=features.replace("G3",1)
print(features["label"].value_counts())

labels = np.array(features['label'])
features= features.drop('label', axis = 1)

X_train, X_test, y_train, y_test = train_test_split(features, labels,
                                                    test_size = test_size, random_state = random_state)
X_train=pd.DataFrame(X_train)
X_test=pd.DataFrame(X_test)
print([X_train.shape,y_train.shape,X_test.shape,y_test.shape])

best_model = LogisticRegressionCV(random_state=random_state, cv=cv, scoring=szempont, penalty="elasticnet", 
                          fit_intercept=True, solver="saga", Cs=10, n_jobs=15, max_iter=20000, 
                          l1_ratios=l1_ratios)
best_model.fit(X_train,y_train)
print("Best C: ", best_model.C_)
print("Best l1_ratio: ", best_model.l1_ratio_)
print ('Max auc_roc:', best_model.scores_[1].mean(axis=0).max())

dill.dump_session('globalsave_part2.pkl')

y_pred = best_model.predict(X_train)
y_pred_proba=best_model.predict_proba(X_train)[:,1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba, drop_intermediate=True)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(confusion_matrix(y_train,y_pred))
print(classification_report(y_train,y_pred))
print("Accuracy train: ", accuracy_score(y_train, y_pred))
print("AUC train: ", roc_auc_score(y_train, y_pred_proba))
print("Threshold value is:", optimal_threshold)
for i in range(len(y_pred)):
    if y_pred_proba[i]<optimal_threshold:
        y_pred[i]=0
    else: y_pred[i]=1

print(confusion_matrix(y_train,y_pred))
print(classification_report(y_train,y_pred))
print("Accuracy train after thresholding: ", accuracy_score(y_train, y_pred))


y_pred = best_model.predict(X_test)
y_pred_proba=best_model.predict_proba(X_test)[:,1]
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy test: ", accuracy_score(y_test, y_pred))
print("AUC test: ", roc_auc_score(y_test, y_pred_proba))
for i in range(len(y_pred)):
    if y_pred_proba[i]<optimal_threshold:
        y_pred[i]=0
    else: y_pred[i]=1

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy test after thresholding: ", accuracy_score(y_test, y_pred))


y_pred_proba=best_model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)

fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc = 'lower right')
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/test_roc.pdf")
plt.close('all')



features = pd.read_table('uterus_rnaseq_VST_G2.txt', sep="\t", index_col=0)
print(features["label"].value_counts())
y_test = np.array(features['label'])
X_test= features.drop('label', axis = 1)

y_pred_proba=pd.DataFrame(best_model.predict_proba(X_test)[:,1])
y_pred_proba.columns=["pred_proba"]
y_pred_proba["samples"]=X_test.index.values
y_pred_proba.to_csv("/data10/working_groups/balint_group/gargya.peter/R/uterus/G2_preds.txt", sep='\t',
                    index=False)

#%%
#                 PART3: important overlapping genes during CV-s
import os
os.chdir("/data10/working_groups/balint_group/gargya.peter/R/uterus")

import dill                          
dill.load_session('globalsave_part1.pkl')

big_model_AUCs=valid_auc
coefs_array=coefs_array/100
total_coefs=np.absolute(coefs_array).sum()
coefs_pd=pd.DataFrame(data=np.absolute(coefs_array), columns=X_train.columns.values).transpose().sort_values(0, axis=0, ascending=False)

toplot=pd.DataFrame(data=coefs_array, columns=X_train.columns.values).transpose()
toplot.columns=["coefs"]
toplot["abs"]=np.absolute(coefs_array).transpose()
toplot=toplot.sort_values("abs", axis=0, ascending=False).head(12)
toplot.to_csv("to_plot_barchart.txt", index=True, sep="\t")

ori_X_train=pd.DataFrame(X_train)
ori_X_test=pd.DataFrame(X_test)
loop_auc=[]
loop_auc_test=[]
loop_pvals=[]
loop_cum_coef=[]

for i in range(1,200):
    valid_auc=[]
    valid_auc_test=[]
    gene_mask=coefs_pd.index.values[:i]
    X_train=ori_X_train[gene_mask]
    X_test=ori_X_test[gene_mask]
                                
    for train_idx, val_idx in tqdm(cv_outer.split(X_train, y_train)):
        train_data, val_data = X_train.iloc[train_idx], X_train.iloc[val_idx]
        train_target, val_target = y_train[train_idx], y_train[val_idx]
        
        best_model = LogisticRegressionCV(random_state=random_state, cv=cv, scoring=szempont, penalty="elasticnet",
                          fit_intercept=True, solver="saga", Cs=10, n_jobs=15, max_iter=20000, 
                          l1_ratios=l1_ratios)
        best_model.fit(train_data, train_target)
        
        y_pred_prob = best_model.predict_proba(val_data)[:,1]
        auc = roc_auc_score(val_target, y_pred_prob)
        valid_auc.append(auc)
        mean_auc=statistics.mean(valid_auc)
        
        best_model.fit(X_train, y_train)
        y_pred_prob = best_model.predict_proba(X_test)[:,1]
        auc_test = roc_auc_score(y_test, y_pred_prob)
        valid_auc_test.append(auc_test)
        mean_auc_test=statistics.mean(valid_auc_test)
        
    loop_auc.append(mean_auc)
    loop_auc_test.append(mean_auc_test)
    loop_pvals.append(scipy.stats.wilcoxon(big_model_AUCs, valid_auc, alternative="two-sided").pvalue)
    loop_cum_coef.append(coefs_pd[0][0:i].sum())


dill.dump_session('globalsave_part3.pkl')

data=pd.DataFrame(loop_auc, columns=["auc"])
data["auc_test"]=loop_auc_test
data["pval"]=loop_pvals
data["cum_coef"]=loop_cum_coef
data.to_csv("results.txt", index=False, sep="\t")

fig, ax = plt.subplots()
ax.plot(range(1,200), loop_auc, "brown", label="Mean AUCs")
ax.plot(range(1,200), loop_auc_test, "green", label="Test AUCs")
plt.ylabel('AUC scores', fontsize=14)
plt.xlabel('Number of genes', fontsize=14)
plt.legend()
plt.title("Results of iterative analysis")
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/loop_AUCs_min_num_genes.pdf")
plt.close('all')



#%%
#                   PART4: results with min num genes
import os
os.chdir("/data10/working_groups/balint_group/gargya.peter/R/uterus")

import dill                          
dill.load_session('globalsave_part1.pkl')

big_model_AUCs=valid_auc
coefs_pd=pd.DataFrame(data=np.absolute(coefs_array), columns=X_train.columns.values).transpose().sort_values(0, axis=0, ascending=False)

gene_mask=coefs_pd.index.values[:12]
print(gene_mask)
X_train=X_train[gene_mask]
X_test=X_test[gene_mask]

best_model = LogisticRegressionCV(random_state=random_state, cv=cv, scoring=szempont, penalty="elasticnet",
                                  fit_intercept=True, solver="saga", Cs=10, n_jobs=15, max_iter=20000, 
                                  l1_ratios=l1_ratios)
best_model.fit(X_train, y_train)
print("Best C: ", best_model.C_)
print("Best l1_ratio: ", best_model.l1_ratio_)
print ('Max auc_roc:', best_model.scores_[1].mean(axis=0).max())


y_pred = best_model.predict(X_train)
y_pred_proba=best_model.predict_proba(X_train)[:,1]
fpr, tpr, thresholds = roc_curve(y_train, y_pred_proba, drop_intermediate=True)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
print(confusion_matrix(y_train,y_pred))
print(classification_report(y_train,y_pred))
print("Accuracy train: ", accuracy_score(y_train, y_pred))
print("AUC train: ", roc_auc_score(y_train, y_pred_proba))
print("Threshold value is:", optimal_threshold)
for i in range(len(y_pred)):
    if y_pred_proba[i]<optimal_threshold:
        y_pred[i]=0
    else: y_pred[i]=1

print(confusion_matrix(y_train,y_pred))
print(classification_report(y_train,y_pred))
print("Accuracy train after thresholding: ", accuracy_score(y_train, y_pred))


train_pred_proba=pd.DataFrame(y_pred_proba)
train_pred_proba["samples"]=X_train.index.values


y_pred = best_model.predict(X_test)
y_pred_proba=best_model.predict_proba(X_test)[:,1]
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy test: ", accuracy_score(y_test, y_pred))
print("AUC test: ", roc_auc_score(y_test, y_pred_proba))
for i in range(len(y_pred)):
    if y_pred_proba[i]<optimal_threshold:
        y_pred[i]=0
    else: y_pred[i]=1

print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test,y_pred))
print("Accuracy test after thresholding: ", accuracy_score(y_test, y_pred))


y_pred_proba=best_model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)
fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc = 'lower right')
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/test_roc_with_min_num_genes.pdf")
plt.close('all')


features = pd.read_table('uterus_rnaseq_VST_G2.txt', sep="\t", index_col=0)
print(features["label"].value_counts())
y_test = np.array(features['label'])
X_test= features.drop('label', axis = 1)

X_test=X_test[gene_mask]

y_pred_proba=pd.DataFrame(best_model.predict_proba(X_test)[:,1])
y_pred_proba.columns=["pred_proba"]
y_pred_proba["samples"]=X_test.index.values
y_pred_proba.to_csv("/data10/working_groups/balint_group/gargya.peter/R/uterus/G2_preds_with_mingenes.txt", sep='\t',
                    index=False)
