import numpy as np
import pandas as pd
from math import sqrt
import face_recognition

class Recommender:
    def __init__(self):
        self.embvecs = pd.read_csv('face_recognizer_encodings.csv')
        self.usedGirls = set()
        self.imagemap = {}
        for index, row in (self.embvecs.iterrows()):
            vector = row.values[1:]
            name = row.values[0]
            self.imagemap[name] = vector
	
    def face_dist(self, allvecs, vec1):
        return np.linalg.norm((allvecs-vec1).astype(np.float64),axis=1)
	
    def train(self,dataset):
        X_train = []
        dataset=dataset[dataset['target']==1]
        for i in (range(dataset.shape[0])):
            if dataset['image_name'].iloc[i] in self.imagemap:
                X_train.append(self.imagemap[dataset['image_name'].iloc[i]])
        X_train = np.array(X_train)
        dist = np.zeros(self.embvecs.shape[0])
        for i in (range(X_train.shape[0])):
            vec1 = X_train[i]
            dist += self.face_dist(self.embvecs.values[:,1:], vec1)
        self.pred = pd.DataFrame({'image_name':self.embvecs.values[:,0],'dist':dist})
        self.pred = self.pred.sort_values(by = ['dist'])
		
		
    def updateUsed(self,name):
        self.usedGirls.add(name)
        
    def getSet(self):
        count = 0
        i = 0
        res = []
        while count < 10:
            if not(self.pred['image_name'].iloc[i] in self.usedGirls):
                res.append({'image_name':self.pred['image_name'].iloc[i]})
                self.usedGirls.add(self.pred['image_name'].iloc[i])
                count+=1
            i+=1
        return res