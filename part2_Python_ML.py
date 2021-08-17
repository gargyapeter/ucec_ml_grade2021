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
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import KernelPCA
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
#import seaborn as sn

szempont="roc_auc"
random_state=1
cv_outer = StratifiedShuffleSplit(n_splits=100, test_size=0.2, random_state=random_state)
cv=5
test_size=0.2
l1_ratios=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9]

#os.chdir("F:/Egyetem/TDK/TCGA_uterus")
os.chdir("/data10/working_groups/balint_group/gargya.peter/R/uterus")

features = pd.read_table('uterus_rnaseq_VST.txt', sep="\t", index_col=0)
features = features[features['label']!="G2"]
features=features.replace("G1",0)
features=features.replace("G3",1)
print(features["label"].value_counts())

#features2=features.drop(index=features.index.values[164])
labels = np.array(features['label'])

#PCA
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(features2)
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(X_scaled)
#principal_Df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#plt.figure()
#plt.figure(figsize=(10,10))
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.xlabel('PC1 '+str(round(pca.explained_variance_ratio_[0]*100,2))+"%",fontsize=20)
#plt.ylabel('PC2 '+str(round(pca.explained_variance_ratio_[1]*100,2))+"%",fontsize=20)
#plt.title("PCA of the whole dataset",fontsize=20)
#targets = [0, 1]
#colors = ['b', 'r']
#for target, color in zip(targets,colors):
#    indicesToKeep = features2['label'] == target
#    indicesToKeep.index=principal_Df.index.values
#    plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1'],
#                principal_Df.loc[indicesToKeep, 'principal component 2'], 
#                alpha=0.8, c = color, s = 50)
#
#plt.legend(["G1", "G3"],prop={'size': 15})
#plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/PCA_raw.png")
#plt.close('all')



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
#ax.legend(loc="lower right")
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
#os.chdir("F:/Egyetem/TDK/uterus")
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
#df_cm=pd.DataFrame(confusion_matrix(y_test,y_pred))
#plt.figure(figsize=(10,7))
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 25}, fmt='g', cmap='Reds')
#sn.set(font_scale=2)
#plt.xlabel("Predicted")
#plt.ylabel("True")
#plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/conf_matrix_test.png")
#plt.close('all')
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
#plt.rc("figure", facecolor="white")
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
#os.chdir("F:/Egyetem/TDK/uterus")
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

toplot=pd.DataFrame(data=coefs_array, columns=X_train.columns.values).transpose()
toplot.columns=["coefs"]
toplot["abs"]=np.absolute(coefs_array).transpose()
toplot=toplot.sort_values("abs", axis=0, ascending=False).head(200)
toplot.to_csv("to_plot_barchart200.txt", index=True, sep="\t")


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
#os.chdir("F:/Egyetem/TDK/uterus")
os.chdir("/data10/working_groups/balint_group/gargya.peter/R/uterus")

import dill                          
dill.load_session('globalsave_part1.pkl')

big_model_AUCs=valid_auc
coefs_pd=pd.DataFrame(data=np.absolute(coefs_array), columns=X_train.columns.values).transpose().sort_values(0, axis=0, ascending=False)


gene_mask=coefs_pd.index.values[:12]
print(gene_mask)
X_train=X_train[gene_mask]
X_test=X_test[gene_mask]

features["label"]=labels
gene_mask2=np.append(gene_mask,"label")
features2=features[gene_mask2]
features2.to_csv("tocluster_minnumgenes.txt", index=True, sep="\t")


#PCA
#scaler = StandardScaler()
#X_scaled = scaler.fit_transform(features2)
#pca = PCA(n_components=2)
#principalComponents = pca.fit_transform(X_scaled)
#principal_Df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#principal_Df = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_))
#plt.figure()
#plt.figure(figsize=(10,10))
#plt.xticks(fontsize=14)
#plt.yticks(fontsize=14)
#plt.xlabel('PC1 '+str(round(pca.explained_variance_ratio_[0]*100,2))+"%",fontsize=20)
#plt.ylabel('PC2 '+str(round(pca.explained_variance_ratio_[1]*100,2))+"%",fontsize=20)
#plt.title("PCA of the whole dataset",fontsize=20)
#targets = [0, 1]
#colors = ['b', 'r']
#for target, color in zip(targets,colors):
#    indicesToKeep = features2['label'] == target
#   indicesToKeep.index=principal_Df.index.values
#   plt.scatter(principal_Df.loc[indicesToKeep, 'principal component 1'],
#                principal_Df.loc[indicesToKeep, 'principal component 2'], 
#                alpha=0.8, c = color, s = 50)
#
#plt.legend(["G1", "G3"],prop={'size': 15})
#plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/PCA_raw_minnumgenes.png")
#plt.close('all')



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
#df_cm=pd.DataFrame(confusion_matrix(y_train,y_pred))
#import seaborn as sn
#plt.figure(figsize=(10,7))
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 25}, fmt='g', cmap='Reds')
#sn.set(font_scale=2)
#plt.xlabel("Predicted")
#plt.ylabel("True")
#plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/conf_matrix_train_minnumgenes.png")
#plt.close('all')
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
#df_cm=pd.DataFrame(confusion_matrix(y_test,y_pred))
#import seaborn as sn
#plt.figure(figsize=(10,7))
#sn.heatmap(df_cm, annot=True, annot_kws={"size": 25}, fmt='g', cmap='Reds')
#sn.set(font_scale=2)
#plt.xlabel("Predicted")
#plt.ylabel("True")
#plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/conf_matrix_test_minnumgenes.png")
#plt.close('all')
#plt.close('all')
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


test_pred_proba=pd.DataFrame(y_pred_proba)
test_pred_proba["samples"]=X_test.index.values
df_concat = pd.concat([train_pred_proba, test_pred_proba], axis=0)
df_concat.to_csv("find_badG1.txt", index=False, sep="\t")


y_pred_proba=best_model.predict_proba(X_test)[:,1]
fpr, tpr, threshold = metrics.roc_curve(y_test, y_pred_proba)
roc_auc = metrics.auc(fpr, tpr)
fig, ax = plt.subplots()
#plt.rc("figure", facecolor="white")
ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', label='Chance', alpha=.8)
ax.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
ax.set(xlim=[-0.05, 1.05], ylim=[-0.05, 1.05], title="Receiver operating characteristic")
ax.legend(loc = 'lower right')
plt.xlabel("False Positive Rate", fontsize=14)
plt.ylabel("True Positive Rate", fontsize=14)
ax.tick_params(axis='both', which='major', labelsize=14)
plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/test_roc_with_min_num_genes.pdf")
plt.close('all')



#PCA with decision surfce
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled= scaler.transform(X_test)

def BoundaryLine(kernel, algo_name):
    reduction = KernelPCA(n_components=2, kernel = kernel)
    x_train_reduced = reduction.fit_transform(X_train_scaled)
    x_test_reduced = reduction.transform(X_test_scaled)
    
    classifier = LogisticRegression()
    classifier.fit(x_train_reduced, y_train)
    #Train set boundary
    X_set, y_set = x_train_reduced, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.2, cmap = ListedColormap(('blue', 'red')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('blue', 'red'))(i), label = "G1")
    plt.title('{} Boundary Line with {} PCA (Train Set)' .format(algo_name, kernel))
    plt.xlabel('PC 1', fontsize=14)
    plt.ylabel('PC 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.legend().get_texts()[0].set_text('G1')
    plt.legend().get_texts()[1].set_text('G3')
    plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/PCA_trainset_minnumgenes.pdf")
    plt.close('all')
    
    #Test set boundary
    X_set, y_set = x_test_reduced, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.2, cmap = ListedColormap(('blue', 'red')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('blue', 'red'))(i), label = "G1")
    plt.title('{} Boundary Line with {} PCA (Test Set)' .format(algo_name, kernel))
    plt.xlabel('PC 1', fontsize=14)
    plt.ylabel('PC 2', fontsize=14)
    plt.tick_params(axis='both', which='major', labelsize=14)
    plt.tight_layout()
    plt.legend().get_texts()[0].set_text('G1')
    plt.legend().get_texts()[1].set_text('G3')
    plt.savefig("/data10/working_groups/balint_group/gargya.peter/R/uterus/PCA_testset_minnumgenes.pdf")
    plt.close('all')

BoundaryLine('linear', "Logistic Regression")



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





