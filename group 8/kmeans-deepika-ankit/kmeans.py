import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing,cross_validation 
#from sklearn.cluster import KMeans
dataset = pd.read_csv('tit.csv')

class K_Means:
    def __init__(self, k=2,tol=0.001,max_iter=300):
        self.k=k
        self.tol=tol
        self.max_iter=max_iter
    def fit(self,dataset):
        self.centroids= {}
        for i in range(self.k):
            self.centroids[i]=dataset[i]
        for i in range(self.max_iter):
            self.classifications= {}
            for i in range(self.k):
                self.classifications[i]=[]
                    
            for featureset in dataset:
                distances=[np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            prev_centroids=dict(self.centroids)
            for classification in self.classifications:
                self.centroids[classification]=np.average(self.classifications[classification],axis=0)
            optimized=True    
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum(current_centroid-original_centroid*100.0)>self.tol:
                    #print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized=False
            if optimized:
                break  
    def predict(self,data):
        distances = [np.ling.norm(featureset-self.centriods[centroid]) for centroid in self.centroids]
        classification = (distances.index(min(distances)))
        return classification
dataset.convert_objects(convert_numeric=True)
dataset.head()
dataset.fillna(0,inplace=True) 

def handle_non_numerical_data(df):
    columns=dataset.columns.values
    for column in columns:
        text_digit_vals={}
        def convert_to_int(val):
            return text_digit_vals[val]
        if df[column].dtype!=np.int64 and df[column].dtype!=np.float64:
            column_contents=df[column].values.tolist()
            unique_elements=set(column_contents)
            x=0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    text_digit_vals[unique]=x
                    x+=1
            dataset[column]=list(map(convert_to_int,dataset[column]))
    return df        
dataset=handle_non_numerical_data(dataset)
dataset.head()
X=np.array(dataset.drop(['Survived'],1).astype(float))
X=preprocessing.scale(X)
y=np.array(dataset['Survived'])
clf=K_Means()
clf.fit(X)
correct=0
for i in range(len(X)):
    predict_me=np.array(X[i].astype(float))
    predict_me=predict_me.reshape(-1,len(predict_me))
    prediction=clf.predict(predict_me)
    if prediction == y[i]:
        correct+=1
        print(correct/len(X))       