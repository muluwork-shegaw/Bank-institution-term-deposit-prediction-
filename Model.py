# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from matplotlib  import rcParams



from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn import svm

from sklearn.preprocessing import OneHotEncoder,LabelEncoder
from sklearn.preprocessing import StandardScaler,MinMaxScaler

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


import time
time_start = time.time()

import scipy.stats as stat


from sklearn.metrics import *
import datetime



class ClassifierModeling:
    def __init__(self,model_name,X_train=None,y_train=None,X_test=None,y_test=None,kfold=None):
        
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.kfold =kfold
        self.y_pred = None
        self.model_name = model_name
        if self.model_name== "RandomForestClassifier":
            self.model = RandomForestClassifier()
        elif self.model_name == "LogisticRegression":
            self.model = LogisticRegression(solver='saga', random_state=0)
        elif self.model_name == "DecisionTreeClassifier":
            self.model = DecisionTreeClassifier()
        elif self.model_name == "XG_Boost":
            data_dmatrix = xgb.DMatrix(data=self.X_train,label=self.y_train)
            self.model = xgb.XGBClassifier()
            print()
        elif self.model_name =="Multilayer Perceptron":
            print("not implemented yet")
        elif self.model_name == "svm" :
            self.model = svm.SVC(kernel='linear', C=0.01)
        elif self.model_name == "adboost":
             self.model =AdaBoostClassifier()
        elif self.model_name == "gradienBoost":
             self.model = GradientBoostingClassifier()
        
    def fit(self):
        print("fitting the ",self.model_name)
        self.model.fit(self.X_train,self.y_train)

    def get_predicate(self): 
        print("predicting by ",self.model_name)
        self.y_pred = pd.Series(self.model.predict(self.X_test),name ="predict")
        return self.y_pred 
    def get_MSE(self):
        return mean_squared_error(self.y_test,self.y_pred)
    def get_score(self):
        return -(r2_score(self.y_test,self.y_pred))
        
    def get_loss(self):
        return np.sqrt(mean_squared_error(self.y_test,self.y_pred))
    def validate_model(self):
        print("validate the model")
        
        model_fit =pd.DataFrame()
        model_fit = pd.concat([self.y_pred, self.y_test], axis=1)
        matrix = confusion_matrix(self.y_test, self.y_pred)
        

        fig, axs = plt.subplots(1,3,squeeze=False,figsize=(15, 3))
        plt.rcParams.update({'font.size': 10})
        d= plot_confusion_matrix(self.model, self.X_test, self.y_test,
                                             display_labels=["yes","no"],
                                             cmap=plt.cm.Blues,ax=axs[0,2])
        d.ax_.set_title("{} confusion matrix".format(self.model_name))     
        total = float(len(model_fit))

        for ax in axs.flatten():
            plt.rcParams.update({'font.size': 16})
  
            for i, var in enumerate(model_fit.columns):
                ax  = sns.countplot(var, data=model_fit, ax=axs[0][i])
                ax .set_title(self.model_name)
                for p in ax.patches:
                    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
                    x = p.get_x() + p.get_width()
                    y = p.get_height()
                    ax .annotate(percentage, (x, y),ha='center')
        #fig.savefig("https://github.com/muluwork-shegaw/10Academy-week6/blob/master/data/{}.png".format(self.model_name))
     
        
        return matrix,model_fit
    def get_eff_model(self):
        if self.model_name != "svm":
            print("calculate  model performance ")
            metrics = pd.DataFrame()
            metrics["model"] = [self.model_name]
            metrics["MSE"] = mean_squared_error(self.y_test,self.y_pred)
            metrics["Loss"] = np.sqrt(mean_squared_error(self.y_test,self.y_pred))
            metrics["Score"] = -(r2_score(self.y_test,self.y_pred))
            metrics["Kappa"] = cohen_kappa_score(self.y_test, self.y_pred)
            metrics["ROC_Auc"] = roc_auc_score(self.y_test, self.y_pred)
            metrics["precision"] = precision_score(self.y_test, self.y_pred)
            metrics["recall"] = recall_score(self.y_test, self.y_pred)
            metrics["f1_score"] = f1_score(self.y_test, self.y_pred)
            metrics["accuracy"] = accuracy_score(self.y_test, self.y_pred)
        

        
            return metrics
    def get_accuracy_with_kfold(self):
       
        return cross_val_score(self.model,self.X_test, self.y_test,cv=self.kfold, scoring= 'accuracy').mean()
        
    def get_loss_with_kfold(self,valid_data,valid_targ,k_fold):
        return -(cross_val_score(self.model,self.X_test, self.y_test,cv=self.kfold, scoring= 'neg_log_loss').mean())
    def eff_model_with_kfold(self):
        
        if self.model_name != "svm":
            print("calculate  model performance with stratified k_fold")

            scoring = ["accuracy","roc_auc","neg_log_loss","r2",
                 "neg_mean_squared_error","neg_mean_absolute_error"] 

            metrics = pd.DataFrame()
            metrics["model"] = [self.model_name]
            for scor in scoring:
                score = []
                result = cross_val_score(estimator= self.model, X=self.X_test, y= self.y_test,cv=self.kfold,scoring=scor )
                score.append(result.mean())

                metrics[scor] =pd.Series(score)

            return metrics
        
    def get_feature_impo(self):
        if self.model_name != "LogisticRegression":
            feat_importance = pd.Series(self.model.feature_importances_, index=self.X_train.columns)
            feat_importance.plot(kind='bar')
            plt.show()
        return feat_importance
    def get_summary(self):# for feature importance of logistic regression
        if self.model_name == "LogisticRegression":

            denom = (2.0*(1.0+np.cosh(self.model.decision_function(X))))
            denom = np.tile(denom,(X.shape[1],1)).T
            F_ij = np.dot((X/denom).T,X) ## Fisher Information Matrix
            Cramer_Rao = np.linalg.inv(F_ij) ## Inverse Information Matrix
            sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
            z_scores = self.model.coef_[0]/sigma_estimates # z-score for eaach model coefficient
            p_values = [stat.norm.sf(abs(x))*2 for x in z_scores] ### two tailed test for p-values

            z_scores = z_scores
            p_values = p_values
            sigma_estimates = sigma_estimates
            F_ij = F_ij

            summary= pd.DataFrame()
            summary["features"] = self.X_train.columns
            summary["z_score"] = self.z_scores
            summary["p_value"] = self.p_values
            sns.barplot(summary["features"],summary["p_value"], data=summary)
        return summary

    def save_model(self):
        
        now = datetime.datetime.now().strftime('%Y-%m-%d')
        # Saving model to disk
        filename = now + '.pkl'
        pickle.dump(self.model, open(filename, 'wb'))
        return filename
    '''
        use stratified k-fold cross-validation 
        with imbalanced datasets to preserve the 
        class distribution in the train and test 
        sets for each evaluation of a given model.
        '''

    def make_it_stratified(self,data,target,reduction_model='pca',dim=7,show=False):
        X=data.drop(target,axis=1)
        y=data[target]
 
        eff =[]
        model_pred = []
        kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

        #enumerate the splits and summarize the distributions
        i=0
        for train_ix, test_ix in kfold.split(X, y):
            i =i+1
            print("k_fold -{}   with {} model".format(i,self.model_name))

            if reduction_model =='pca': # using PCA
                pca = PCA(n_components=7)
                reduced_df = pca.fit_transform( X) # reduce the dimention and convert to data frame
                      # columns=[f'pca {i}'  for i in range(1,8)]) 
              
            elif reduction_model =='tsne':# using TSNE
                tsne = TSNE(n_components = 7, n_iter = 300)
     
            # select rows
            self.X_train,self.X_test = reduced_df[train_ix], reduced_df[test_ix]
            self.y_train, self.y_test = y[train_ix], y[test_ix]

            self.fit()
            self.get_predicate()
            if show == True:
                matrix,model_fit=self.validate_model()
                model_pred.append(model_fit)
            eff.append(self.get_eff_model())    

            
        df_eff = pd.concat(eff)

        df_eff =pd.DataFrame(df_eff.mean()).transpose()
        df_eff.index=[self.model_name]

        return df_eff,model_pred
            
        
            

            
        

