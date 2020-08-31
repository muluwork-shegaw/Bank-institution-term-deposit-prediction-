from Model import ClassifierModeling
from Data import PreProcessor

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold

import datetime as dt
from matplotlib  import rcParams

data = pd.read_csv("data/bank-additional-full.csv", sep=';')
# data = pd.read_csv("https://github.com/muluwork-shegaw/10Academy-week6/blob/master/data/bank-additional-full.csv?raw=true",error_bad_lines=False,sep=';')

outlier_col = ['age', 'campaign','cons.conf.idx']
drop_col =  ["emp.var.rate","euribor3m",'day_of_week',"duration"]
processor = PreProcessor(data,'y',drop_col,outlier_col,"pca",7)
new_data,split_result = processor.pipe_and_filter()


#for later use
scaled_data = new_data[0]
reduced_data =new_data[1]

X_train =split_result[0]
X_test = split_result[1]
y_train =split_result[2]
y_test = split_result[3]


k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

classifiers =["RandomForestClassifier","LogisticRegression","DecisionTreeClassifier",
             "adboost","gradienBoost"]
print("---------------------------with startified k_fold distribution-----------------------------")

eff =[]
model_pred = []
for model_classifier in classifiers:
    classifier = ClassifierModeling(model_classifier)

    df_eff,model_pred = classifier.make_it_stratified(data=scaled_data,target="y")
    eff.append(df_eff)
    model_pred.append(model_pred)
print("\n-----------------model performance with stratified k_fold------------------")
df = pd.concat(eff)
print(df)

print("---------------------------with out k_fold distribution-----------------------------")
classifiers =["RandomForestClassifier","LogisticRegression","DecisionTreeClassifier",
              "svm","adboost","gradienBoost"]
model =[]
eff_kfold =[]
eff =[]
confu_matrix = []
for model_classifier in classifiers:
    print("{} is staring ....".format(model_classifier))
    classifier = ClassifierModeling(model_classifier,X_train,y_train,X_test,y_test,k_fold)
    classifier.fit()
    pridict = classifier.get_predicate()
    matrix,model_fit=classifier.validate_model()
    print("\n-------------confusion matrix---------")

    print("{0} {1}".format(model_classifier,matrix))
    confu_matrix.append(matrix)
    eff_kfold.append(classifier.eff_model_with_kfold())
    eff.append(classifier.get_eff_model())    
    model.append(classifier.model)
    print("Done !")

    
df_kfold =pd.concat(eff_kfold)


df = pd.concat(eff)
print("\n-----------------model performance with out stratified k_fold------------------")

print(df)

print("\n-------------confusion matrix---------")
print(confu_matrix)




