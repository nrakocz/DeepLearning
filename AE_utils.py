import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import MinMaxScaler



import seaborn as sns
# CONSTANTS
RANDOM_STATE = 101



def logLoss(y_true,y_pred,epsilon=1e-15):
    return -np.mean( (1-y_true)*np.log(1-y_pred+epsilon) + (y_true)*np.log(y_pred+epsilon),axis=1)



def generateReconstructHist2(X,Y,model,loss,val_range=None,wplt=True,v_version=False,normed=False):
    
    pred_X = model.predict(X)
    pred_Y = model.predict(Y)
    
    #if(v_version): # VAE has 2 outputs
        #pred_X = pred_X[0]
        #pred_Y = pred_Y[0]

    if(loss=='MSE'):
        non_death_errs = np.mean((X - pred_X)**2,axis=1)
        death_errs = np.mean((Y - pred_Y)**2,axis=1)
    else:
        px=pred_X #px = np.clip(pred_X,0,1)
        non_death_errs = logLoss(X,px)
        py = pred_Y #py = np.clip(pred_Y,0,1)
        death_errs = logLoss(Y,py)
        
    if(wplt):
        #plt.figure(figsize=(12,8))
        plt.hist(non_death_errs,label='Non Death',alpha=0.5,bins=50,range=val_range,normed=normed)
        plt.hist(death_errs, label='Death',alpha=0.5,bins=50,range=val_range,normed=normed)
        plt.legend()
        plt.title('Original vs. Deconstructed input - '+loss+' Histogram')
        plt.xlabel(loss)
        plt.ylabel('Frequency')
    
    return non_death_errs, death_errs

def plotRoc(nde,de,wplt=True):
    
    scaler = MinMaxScaler()
    res_score = scaler.fit_transform(np.concatenate((nde.reshape(-1,1),de.reshape(-1,1)),axis=0))

    nde_s = scaler.transform(nde.reshape(-1,1))
    de_s = scaler.transform(de.reshape(-1,1))
    y_score = np.concatenate((nde_s.reshape(-1,1),de.reshape(-1,1)),axis=0)
    y_test = np.concatenate((np.zeros((nde.shape[0],1)),np.ones((de.shape[0],1))),axis=0)

    from sklearn import svm, datasets
    from sklearn.metrics import roc_curve, auc

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    n_classes = 1
    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    if(wplt):
        plt.figure(figsize=(8,4))
        plt.plot(fpr[i], tpr[i], label='ROC curve (area = {:.6f})'.format(roc_auc[i]))
        plt.legend()
        
    return roc_auc[0]



def generateRecounstructionErrBarPlot(d1_true,d1_pred,d2_true,d2_pred,features,outcome=['outcome1','outcome2'],error_name='Reconstruction Error',feature_title='Features'):
    
    data1 = np.mean(np.abs(d1_true - d1_pred),axis=0)
    data2 = np.mean(np.abs(d2_true - d2_pred),axis=0)
        
    d1=pd.DataFrame(data1,columns=[error_name])
    d1[feature_title]=features
    d1['outcome']=outcome[0]
    d2=pd.DataFrame(data2,columns=[error_name])
    d2[feature_title]=features
    d2['outcome']=outcome[1]
    d3=pd.concat([d1,d2])


    plt.figure(figsize=(30,10))
    p1=sns.barplot(data=d3,x=feature_title,y=error_name,hue='outcome')
    p=plt.xticks(range(len(features)),features,rotation=90,)
    
    return d3

def viewFrstLayer(model,features):
    l = model.layers
    w = l[0].get_weights()

    plt.figure(figsize=(20,20))
    sns.heatmap(w[0],yticklabels=features,cmap='seismic')