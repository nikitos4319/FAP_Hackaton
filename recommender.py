import numpy as np
import pandas as pd
import xgboost as xgb
from math import sqrt

class Recommender:
    def __init__(self):
        self.embvecs = pd.read_csv('insight_face_model/vecs3.csv')
        self.usedGirls = set()
        self.imagemap = {}
        for index, row in (self.embvecs.iterrows()):
            vector = row.values[1:]
            name = row.values[0]
            self.imagemap[name] = vector
        
    def train(self,dataset):
        X_train = []
        y_train = []
        dataset=dataset[dataset['target']==1]
        for i in (range(dataset.shape[0])):
            if dataset['image_name'].iloc[i] in self.imagemap:
                X_train.append(self.imagemap[dataset['image_name'].iloc[i]])
                y_train.append(dataset['target'].iloc[i])
        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if dataset.shape[0]>300:
            self.model = xgb.XGBClassifier()
            self.model.fit(X_train, y_train)
            res = self.model.predict_proba(self.embvecs.drop('image_name',axis=1).values)[:,1]
            self.pred = pd.DataFrame({'image_name':self.embvecs.values[:,0],'res':res})
            self.pred = self.pred.sort_values(by='res', ascending=False)
			
        else:
            X_train = X_train-np.min(X_train,axis=0)
            X_train = np.divide(X_train, np.max(X_train,axis=0), out=np.zeros_like(X_train), where=np.max(X_train,axis=0)!=0)
            #X_train /= np.max(X_train,axis=0)
            #print(X_train.var(axis=0))
            var = (((X_train).var(axis=0)))
            mat = (X_train).mean(axis=0)
            tar = self.embvecs.values[:,1:]-np.min(self.embvecs.values[:,1:],axis=0)

            #tar = np.divide(tar, np.max(X_train,axis=0), out=np.zeros_like(), where=np.max(tar,axis=0)!=0)
            tar = tar/np.max(X_train)
            res = np.sum( np.abs(self.embvecs.values[:,1:] - mat)*(var),axis=1 )
            self.pred = pd.DataFrame({'image_name':self.embvecs.values[:,0],'res':res})
            self.pred = self.pred.sort_values(by='res')
        #print(self.pred.head(10))
        #print('Model updated')
		
		
    def updateUsed(self,name):
        self.usedGirls.add(name)
        
    def getSet(self):
        count = 0
        i = 0
        res = []
        while count < 10:
            #print(self.pred['image_name'].iloc[i])
            if not(self.pred['image_name'].iloc[i] in self.usedGirls):
                res.append({'image_name':self.pred['image_name'].iloc[i]})
                self.usedGirls.add(self.pred['image_name'].iloc[i])
                count+=1
            i+=1
        return res