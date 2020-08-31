# import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt
from matplotlib  import rcParams










'''
________--------------------------________________________

'''



from sklearn.preprocessing import OneHotEncoder,LabelEncoder

from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
# from keras.models import Model
# from Keras.layers  import Input,Dense
# from keras import regularizers
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

import time



class PreProcessor:

    def __init__(self,data,target,drop_col,outlier_col,reduction_model,dim):
        self.data = data
        self.orig_data = data
        self.target = target
        self.drop_col =drop_col
        self.outlier_col =outlier_col
        
        self.reduction_model = reduction_model
        self.dim = dim
        
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None

        
        

    '''
       Missing Value Ratio
       calculate the missing value ration
       and drop that variable if it's missing value ratio 
       is greater than 20% and otherwise impute or drop that
       value

    ''' 

    def treat_missing(self,data):#feature selection based on missing value ratio
        cat_col = self.data.select_dtypes(exclude=['int64', 'float64']).columns #catagorical columns
        non_cat_col = self.data.select_dtypes(include=['int64', 'float64']).columns # non catagorical variable
    
        data[cat_col] = data[cat_col].replace(['unknown', 0], np.nan)
        missing_value = data.isnull().sum()/len(data)*100
        missing_var = []
        for i,col in enumerate(self.data.columns):
            if missing_value[i]>=20:   #setting the threshold as 20%
                missing_var.append(col)
                data.drop(col, axis=1,inplace=True)
        print("\n {} variables have dropped based on missing value ratio".format(missing_var))
          
        '''
            since the rest variable's missing value is 
            smaller we can impute them
        '''
        print("\n {} missing values have been dropped".format(data.isnull().sum().sum()))
        data=data.fillna(data.mode().iloc[0])
        
        self.data = data
        return data
    '''
        we may need to drop a columns which have a less variance and
        more correlated variables, since if they are more correlated 
        they have same impact in indecating something and if they have least
        variance which means they are almost constant values.
    '''
    def drope(self,data,col=None): # which is ongoing function,not implemented in optimized way yet.
        data.drop(col,axis =1,inplace=True)
        self.cat_col = data.select_dtypes(exclude=['int64', 'float64']).columns #catagorical columns
        self.non_cat = data.select_dtypes(include=['int64', 'float64']).columns #catagorical columns
        print("\n {} variable has been dropped based on high correlation and less variance analysis".format([col]))
        self.data = data
        return data

    def deal_outliers(self,data,columns):#handle outliers based on interquartile range(iqr)
        orig_data = data

        print("\n before dealing with the ouliers the shape of the data",
              data.shape)
        for col in columns:
            q1 = data[col].quantile(0.25) # quartile 1
            q3 = data[col].quantile(0.75) # quartile 4
            iqr = q3 - q1
            lower_bound = q1 -(1.5 * iqr) #lower whishker
            upper_bound = q3 +(1.5 * iqr) #upper whishker
            data = data[data[col]>lower_bound] #remove which are greater than upper whishker
            data = data[data[col]<upper_bound] #remove which are less than lower whishker

        print("after removing the outliers",data.shape)
        
        self.data = data
        self.orig_data = orig_data
        
        print("{}  data has been removed based on outlier analysis".format(len(self.orig_data)-len(self.data)))
        return data # return data with outlier and without outlier
    
    def encoder(self,data):# encode the catagorical value
        label_encoder = LabelEncoder()
        binary_col = []
        for col in (data.columns):
            if data[col].nunique() == 2:
                binary_col.append(col)
                data[col] = label_encoder.fit_transform(data[col])# encode binary variables
        print("\n{} variables have been encoded based on label encoding the rest encoded as dummy variable".format(binary_col))
        
        data = pd.get_dummies(data) #encode not binary variables
        self.data = data
        return data
    
    def scaler(self,data, scaler= MinMaxScaler()): # not implmented in optimized way yet
        data[:] = scaler.fit_transform(data[:]) #scaleing the given data
        self.data = data
        print("\n---- Scaling the data based on",scaler)
        return data
    
    def reduce_dimension(self,data,reduction_model = 'tsne',dim=3): # reduce the dimentions of a given data
        time_start = time.time()

        target = self.target
        data_1 = data
        d = data_1.drop(target,axis=1)
        
        print("\nThe data is reduced based on {0} algorithm to {1} dimentions".format(reduction_model,dim+1))
        print("Executing the reduction starts and wait ......")
        
        if reduction_model =='pca': # using PCA
            pca = PCA(n_components=dim)
            reduced_df = pd.DataFrame(pca.fit_transform(d), # reduce the dimention and convert to data frame
               columns=[f'pca {i}'  for i in range(1,dim+1)]) 
            print ('PCA done! Time elapsed: {} seconds'.format(time.time()-time_start))
            
        elif reduction_model =='tsne':
            tsne = TSNE(n_components = dim, n_iter = 300)
            reduced_df =pd.DataFrame(tsne.fit_transform(d),
                        columns=[f'tsne {i}'  for i in range(1,dim+1)])
            print ('t-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start))

    #     else:
    #         input_dim = d.shape[1]
    #         input_layer = Input(shape = (input_dim,))
    #         encoder_layer_1 = Dense(40,activation='tanh',
    #                         activity_regularizer=regularizers.l1(10e-5)(input))(input_layer)
    #         encoder_layer_2 = Dense(30,activation='tanh')(encoder_layer_1)
    #         encoder_layer_3 = Dense(encoding_dim,activation='tanh')(encoder_layer_2)
    #         #create encoder model
    #         encoder = Model(inputs = input_layer, outputs = encoder_layer_3)
    #         reduced_df= pd.DataFrame(encoder.predict(d))
                   
    # reset the index to get clean data(without nan)
                
        
        data_1[target].reset_index(drop=True, inplace=True)
        reduced_df.reset_index(drop=True, inplace=True)
        final_df = pd.concat([reduced_df,data_1[target]],axis=1)
                   
        self.data = final_df
        return final_df
    
    def split_data(self,data,target):  #Split the data
        
        print("\nspliting the data as train and test")
        X=data.drop(target,axis=1)
        y=data[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1,stratify =y,random_state=1)#use stratified distribution for imbalanced class distributin
        return X_train, X_test, y_train, y_test
                   
    def pipe_and_filter(self):# this function is to pipe earlier steps
        new = self.treat_missing(self.data)
        new = self.drope(self.data,col= self.drop_col)
        new = self.deal_outliers(self.data,columns = self.outlier_col)
        new = self.encoder(self.data)
        new_scaled = self.scaler(self.data)
        new = self.reduce_dimension(self.data,reduction_model = self.reduction_model , dim=self.dim)
        X_train, X_test, y_train, y_test =self.split_data(self.data,self.target)
        print("DONE!")
        
        result = [X_train, X_test, y_train, y_test]
        df =[new_scaled, new]
                   
        return df,result
        
        
            
        
    