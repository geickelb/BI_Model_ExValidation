
### windows for filtering from t_0
lower_window=(0-0.25)
upper_window=1
folder='24_hr_window'
nfolds=10
scoring='roc_auc' #neg_log_loss
n_iter=40 #for gridsearch
gridsearch=False #gridsearch=False means it does triaged hyperparameter combinations based on some algorithm. True= tests all 

save_boolean=False
###

import pandas as pd
import matplotlib.pyplot as plt
import os, sys
from pathlib import Path
import seaborn as sns
import numpy as np
import glob
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, accuracy_score, auc, precision_recall_fscore_support, pairwise, f1_score, log_loss, make_scorer
from sklearn.metrics import precision_score, recall_score
from sklearn import metrics
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
# from sklearn.externals.joblib import Memory
import joblib
#from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, Imputer
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.utils import validation
from scipy.sparse import issparse
from scipy.spatial import distance
from sklearn import svm

from sklearn.calibration import CalibratedClassifierCV

#importin xg boost and all needed otherstuff
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier #conda install -c conda-forge xgboost to install
##adding these, lets see if it helps with xgboost crash
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import pickle

#reducing warnings that are super common in my model
import warnings
from sklearn.exceptions import DataConversionWarning
warnings.simplefilter(action='ignore') #ignore all warnings

RANDOM_STATE = 15485867

# %matplotlib inline
plt.style.use('seaborn-white')




def evaluate(model, x, y):
    "simple classification evaluation metrics and output used in my hypertuning functions"
    from sklearn.metrics import log_loss
    
    y_hat = model.predict(x)
    y_hat_proba = model.predict_proba(x)[:, 1] 
    errors = abs(y_hat - y)
    mape = 100 * np.mean(errors / y)
    accuracy = 100 - mape
    auc=roc_auc_score(y, y_hat_proba)
    loss= log_loss(y, y_hat_proba)
        
#     print ('the AUC is: {:0.3f}'.format(auc))
#     print ('the logloss is: {:0.3f}'.format(loss))
#     print(confusion_matrix(y, y_hat))
#     print(classification_report(y,y_hat, digits=3))
    
    if scoring=='neg_log_loss':
        return_value=loss
    elif scoring=='roc_auc':
        return_value=auc
    else:
        raise ValueError
    
    return (return_value)


def hypertuning_fxn(X, y, nfolds, model , param_grid, z_subject_id, scoring=scoring, gridsearch=True, n_iter=20, verbose=False): 
    from sklearn.model_selection import GroupKFold
    
    np.random.seed(12345)
    if gridsearch==True:
        grid_search = GridSearchCV(estimator= model,
                                         param_grid=param_grid,
                                         cv=GroupKFold(nfolds),
                                         scoring=scoring,
                                         return_train_score=True,
                                         n_jobs = -1)
    else:
        grid_search = RandomizedSearchCV(estimator= model,
                                         param_distributions= param_grid,
                                         n_iter=n_iter,
                                         cv=GroupKFold(nfolds),
                                         scoring=scoring,
                                         return_train_score=True,
                                         random_state=12345,
                                         n_jobs = -1)
        
    grid_search.fit(X, y, groups=z_subject_id)    
    print(" scorer function: {}".format(scoring))
    print(" ##### CV performance: mean & sd scores #####")

    means = grid_search.cv_results_['mean_test_score']
    stds = grid_search.cv_results_['std_test_score']
    print('best cv score: {:0.3f}'.format(grid_search.best_score_))
    print('best cv params: ', grid_search.best_params_)

    worst_index=np.argmin(grid_search.cv_results_['mean_test_score'])
    print('worst cv score: {:0.3f}'.format(grid_search.cv_results_['mean_test_score'][worst_index]))
    print('worst cv params: ', grid_search.cv_results_['params'][worst_index])
    ##
    if verbose==True:
        for mean, std, params in zip(means, stds, grid_search.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r"% (mean, std * 2, params))

    print('##### training set performance #####\n')   
    print(' best hypertuned model training set performance:')
    best_random = grid_search.best_estimator_
    best_random_auc = evaluate(best_random, X, y)
    
    print(' worst hypertuned model training set performance:')
    worst_params= grid_search.cv_results_['params'][worst_index]
    worst_random=model.set_params(**worst_params)
    worst_random.fit(X,y)
    worst_random_auc = evaluate(worst_random, X, y)      
          
    print('relative scorer change of {:0.2f}%. between worst and best hyperparams on TRAINING set (may be overfit)'.format( 100 * (best_random_auc - worst_random_auc) / worst_random_auc))
    
    return(grid_search)


def hypertuned_cv_fxn(x, y, model_in, nfolds, z_subject_id, lr_override=False):
    """
    ### updating again on 10/14/22 to use a fit LR model instead of the model input 
    ###updating on 09/30/22 to use the fit model and find the threshold via cv. 
    
    the goal of this function is to take the best hypertuned model and 
    generate average and std for F-1, precision, recall, npv, and AUC across each fold.
    Ideally i could have generated this above in my hypertuning cv function,
    but it actually took less computational time to just rerun cv on the best performing evaluator and collect all of the averaged performance metrics
    
    lr_override: uses a fit l1 regression model ffit on x,y to assess high sensitivity threshold rather than the model_in
    
    """
    
    from sklearn.model_selection import GroupKFold
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score
    from sklearn.base import clone
    
    pos_label=1
#     model= clone(model_in, safe=True)
    model=model_in
    np.random.seed(12345)
    group_kfold = GroupKFold(n_splits=nfolds)
    group_kfold.get_n_splits(x, y, z_subject_id)

    f1_y_cv=[]
    auc_y_cv=[]
    prec_y_cv=[]
    recall_y_cv=[]
    npv_y_cv=[]
    tp_threshold_cv=[]   
    
    if lr_override==True:
        lr_model= LogisticRegression(random_state=12345,solver='liblinear', penalty='l2')
        lr_model.fit(x, y)
    

    for train_index, test_index in group_kfold.split(x, y, z_subject_id):
        x_train_cv, x_test_cv = x[train_index], x[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]
#         model.fit(x_train_cv, y_train_cv)  ###updating on 09/30/22 to use the fit model and find the threshold via cv. 
        
        y_proba = model.predict_proba(x_test_cv)[:,1]
        y_pred = model.predict(x_test_cv)

        
        fpr, tpr, thresholds = metrics.roc_curve(y_test_cv, y_proba, pos_label=pos_label)   
        #gathering the optimal youden_index and df of tpr/fpr for auc and index of that optimal youden. idx is needed in the roc
        ##only use this 
#         youden_threshold, roc_df, idx= optimal_youden_index(fpr, tpr, thresholds,tp90=True)
        
        if lr_override==True:
            y_proba_lr = lr_model.predict_proba(x_test_cv)[:,1]
            fpr_lr, tpr_lr, thresholds_lr = metrics.roc_curve(y_test_cv, y_proba_lr, pos_label=pos_label)    
            youden_threshold, roc_df, idx= optimal_youden_index(fpr_lr, tpr_lr, thresholds_lr,tp90=True)
#             print('you_thresh:{}'.format(youden_threshold))
        else: #only use this if model is well calibrated to the x,y data input. 
            youden_threshold, roc_df, idx= optimal_youden_index(fpr, tpr, thresholds,tp90=True)
            
        y_pred_youden = [1 if y >= youden_threshold else 0 for y in y_proba]
        tp_threshold_cv.append(youden_threshold)
        
        npv_y=confusion_matrix(y_test_cv, y_pred_youden)[0,0]/sum(np.array(y_pred_youden)==0)
        npv_y_cv.append(npv_y)

        prec_y= precision_score(y_true=y_test_cv, y_pred= y_pred_youden, pos_label=pos_label)
        prec_y_cv.append(prec_y)

        recall_y= recall_score(y_true=y_test_cv, y_pred= y_pred_youden, pos_label=pos_label)
        recall_y_cv.append(recall_y)

        f1_y= f1_score(y_true=y_test_cv, y_pred= y_pred_youden, pos_label=pos_label)
        f1_y_cv.append(f1_y)

        ###need to debug this.###
        auc_y=roc_auc_score(y_true=y_test_cv, y_score= y_proba)
        auc_y_cv.append(auc_y)
        
        youden_dic_cv= {'model':type(model).__name__, 
                'auc':np.mean(auc_y_cv),
                'auc_sd':np.std(auc_y_cv),
                'precision':np.mean(prec_y_cv),
                'precision_sd':np.std(prec_y_cv),
                'recall':np.mean(recall_y_cv),
                'recall_sd':np.std(recall_y_cv),
                'f1':np.mean(f1_y_cv),
                'f1_sd':np.std(f1_y_cv),
                'npv':np.mean(npv_y_cv),
                'npv_sd':np.std(npv_y_cv),
                'tp_threshold':np.mean(tp_threshold_cv),
                'tp_threshold_sd':np.std(tp_threshold_cv)}
        
    return(youden_dic_cv)




def saveplot(plt, figure_name, pubres=False):
    """
    simple function for saving plots
    """
    address = str(repository_path)+'/figures/{}_{}'.format(date,folder)
    print(address)

    if not os.path.exists(address):
        os.makedirs(address)
    if pubres==False:
        plt.savefig(address+"/{}.png".format(figure_name),bbox_inches='tight')
    else:
        plt.savefig(address+"/{}.png".format(figure_name),bbox_inches='tight', dpi=350)

def optimal_youden_index(fpr, tpr, thresholds, tp90=True):
    """
    inputs fpr, tpr, thresholds from metrics.roc(),
    outputs the clasification threshold, roc dataframe, and the index of roc dataframe for optimal youden index
    """
    #making dataframe out of the thresholds
    roc_df= pd.DataFrame({"thresholds": thresholds,"fpr":fpr, "tpr": tpr})
    roc_df.iloc[0,0] =1
    roc_df['yuden']= roc_df['tpr']-roc_df['fpr']
    
    if tp90==True:
        idx= roc_df[roc_df['tpr']>=0.9]['yuden'].idxmax() #changed this so now finds optimial yuden threshold but tp>=90%
    else:
        idx=roc_df['yuden'].idxmax() #MAX INDEX
    
    youden_threshold=roc_df.iloc[idx,0] #threshold for max youden
    return(youden_threshold, roc_df, idx)
    
    
    

    
def plot_roc(fpr, tpr, roc_auc,thresholds, tp_threshold, save=save_boolean,model_name=None, folder_name=None, file_name=None):
    """
    changed on 02/06/20 to accept thresholds and the specified tp_threshold to mark on the roc
    """
    plt.title('ROC with Training TPR>=0.9 Index')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    
    roc_df= pd.DataFrame({"thresholds": thresholds,"fpr":fpr, "tpr": tpr})
    roc_df.iloc[0,0] =1
    
    #finding the point on the line given threshold 0.5 (finding closest row in roc_df)
    og_idx=roc_df.iloc[(roc_df['thresholds']-0.5).abs().argsort()[:1]].index[0]
    plt.plot(roc_df.iloc[og_idx,1], roc_df.iloc[og_idx,2],marker='o', markersize=5, color="g")
#     plt.annotate(s="P(>=0.5)",xy=(roc_df.iloc[og_idx,1]+0.02, roc_df.iloc[og_idx,2]-0.04),color='g') #textcoords
    plt.annotate(text="P(>=0.5)",xy=(roc_df.iloc[og_idx,1]+0.02, roc_df.iloc[og_idx,2]-0.04),color='g') #textcoords


    #finding the point on the line given the best tuned threshold in the training set for tpr>=0.9
    idx=roc_df.iloc[(roc_df['thresholds']-tp_threshold).abs().argsort()[:1]].index[0]
    
    plt.plot(roc_df.iloc[idx,1], roc_df.iloc[idx,2],marker='o', markersize=5, color="r") ##
#     plt.annotate(s="TPR>=0.9",xy=(roc_df.iloc[idx,1]+0.02, roc_df.iloc[idx,2]-0.04),color='r' ) #textcoords
    plt.annotate(text="TPR>=0.9",xy=(roc_df.iloc[idx,1]+0.02, roc_df.iloc[idx,2]-0.04),color='r' ) #textcoords

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.grid(color='grey', linestyle='-', linewidth=1, alpha=0.2)
    
    if save==True:
        saveplot(plt, figure_name="{}_roc".format(model_name))
    else: pass
    
    plt.show()
    


def classifier_eval(model,
                    x,
                    y,
                    proba_input=False,
                    pos_label=1,
                    training=True,
                    train_threshold=None,
                    print_default=True,
                    model_name=None,
                    folder_name=None,
                    save=save_boolean):
    import sklearn.metrics as metrics
    from sklearn.metrics import precision_score, roc_auc_score, f1_score, recall_score
    """
    classification evaluation function. able to print/save the following:
    
    print/save the following:
        ROC curve marked with threshold for optimal youden (maximizing tpr+fpr with constraint that tpr>0.9)

        using 0.5 threshold:
            confusion matrix
            classification report
            npv
            accuracy

        using optimal youden (maximizing tpr+fpr with constraint that tpr>0.9):
            confusion matrix
            classification report
            npv
            accuracy
    
    output: 
        outputs modelname, auc, precision, recall, f1, and npv to a dictionary. 
    
    notes:
    youden's J statistic:
    J= sensitivity + specificity -1
    (truepos/ truepos+falseneg) + (true neg/ trueneg + falsepos) -1. 
    NOTE: with tpr>0.9 turned on, the youden statistic is basically just the furthest point on the line away from the midline with tpr>=0.9
    NOTE2: this function arguably does too much. in the future it may be better to seperate it out into more compartmental functions like with preprocessing().
    """
    
    if proba_input==True: 
        y_proba= model
        y_pred=[1 if y >= 0.5 else 0 for y in y_proba]
    
    else:
        model_name=type(model).__name__

        y_pred = model.predict(x)
        y_proba = model.predict_proba(x)[:,1]
        
    if training==True:
        
        fpr, tpr, thresholds = metrics.roc_curve(y, y_proba, pos_label=pos_label)
        roc_auc = metrics.auc(fpr, tpr)

        #gathering the optimal youden_index and df of tpr/fpr for auc and index of that optimal youden. idx is needed in the roc
        tp_threshold, roc_df, idx= optimal_youden_index(fpr, tpr, thresholds,tp90=True)
    
    else: #if training is not true, then we use the tuned threshold specified on the trainingset.
        fpr, tpr, thresholds = metrics.roc_curve(y, y_proba, pos_label=pos_label)
        roc_auc = metrics.auc(fpr, tpr)
        roc_df= pd.DataFrame({"thresholds": thresholds,"fpr":fpr, "tpr": tpr})
        roc_df.iloc[0,0] =1
        tp_threshold= train_threshold

    #plotting roc
    #plot_roc(fpr, tpr, roc_auc, threshold, save=save_boolean,model_name=None, folder_name=None, file_name=None
    plot_roc(fpr, tpr, roc_auc, thresholds, tp_threshold, save=save_boolean, model_name=model_name,folder_name=folder)
    plt.show(), plt.close()
    
    #printing npv, recall, precision, accuracy
    npv=confusion_matrix(y, y_pred)[0,0]/sum(np.array(y_pred)==0)
    prec= precision_score(y_true=y, y_pred= y_pred, pos_label=pos_label)
    recall= recall_score(y_true=y, y_pred= y_pred, pos_label=pos_label)
    f1= f1_score(y_true=y, y_pred= y_pred, pos_label=pos_label)
    confusion =pd.DataFrame(confusion_matrix(y, y_pred),
                                 index=['condition_neg','condition_pos'],
                                columns=['pred_neg','pred_pos'])
    
    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
    specificity = tn / (tn+fp)
    
    if save==True:
        save_df(confusion, df_name='{}_confusion_base'.format(model_name), rel_path='/tables/', verbose=False)
    
    if print_default==True: ###can opt to not print the 0.5 classification threshold classification report/conf matrix
        #plotting confusion matrixs
        print("\n******* Using 0.5 Classification Threshold *******\n")
        print(confusion)
        print('\n')
        print ('the Accuracy is: {:01.3f}'.format(accuracy_score(y, y_pred)))
        print ("npv: {:01.3f}".format(npv))
        print ('the classification_report:\n', classification_report(y,y_pred, digits=3))
    else:
        pass
    
    #### YOUDEN ADJUSTMENT #####

    print("\n******* Using Optimal TPR>=0.9 Classification Threshold *******\n")
    print("\nthe Youden optimal index is : {:01.3f}".format(train_threshold))

    y_pred_youden = [1 if y >= train_threshold else 0 for y in y_proba]

    npv_y=confusion_matrix(y, y_pred_youden)[0,0]/sum(np.array(y_pred_youden)==0)
    prec_y= precision_score(y_true=y, y_pred= y_pred_youden, pos_label=pos_label)
    recall_y= recall_score(y_true=y, y_pred= y_pred_youden, pos_label=pos_label)
    f1_y= f1_score(y_true=y, y_pred= y_pred_youden, pos_label=pos_label)
    auc_y=roc_auc_score(y_true=y, y_score= y_proba)
    
    
    ##plotting and saving confusion matrix
    confusion_youden=pd.DataFrame(confusion_matrix(y, y_pred_youden),
                                 index=['condition_neg','condition_pos'],
                                columns=['pred_neg','pred_pos'])
    
    if save==True:
        save_df(confusion_youden, df_name='{}_confusion_tuned'.format(model_name), rel_path='/tables/',verbose=False)
    
    #plotting confusion matrixs
    print('\n')
    print(confusion_youden)
    print('\n')
    print ('the Accuracy is: {:01.3f}'.format(accuracy_score(y, y_pred_youden)))
    print ("npv: {:01.3f}".format(npv_y))
    print ('the classification_report:\n', classification_report(y,y_pred_youden, digits=3))
    
    youden_dic= {'model':model_name, 'auc':auc_y, 'precision':prec_y, 'recall':recall_y, 'f1':f1_y, 'npv':npv_y,'threshold':tp_threshold}
    return(youden_dic)

