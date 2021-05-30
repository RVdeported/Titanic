# -*- coding: utf-8 -*-
"""
Created on Sun May 30 12:15:36 2021

@author: Roman
"""


# libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.preprocessing import StandardScaler, OneHotEncoder

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression as LR

DATASET_TRAIN_PATH = "./input/train.csv"
DATASET_TEST_PATH = "./input/test.csv"
PREPARED_TRAIN_DATASET_PATH = "./output/train_prepared.csv"
PREPARED_TEST_DATASET_PATH = "./output/test_prepared.csv"
OUTPUT_PATH = "./output/test_predicted.csv"
MODEL_PATH = "./output/model.pickle"

RANDOM_VAL = 228



# import of dataframe
df_full = pd.DataFrame(pd.read_csv(DATASET_TRAIN_PATH))
df_full.index = df_full["PassengerId"]
df_full = df_full.drop(["PassengerId"], axis = 1)
df = df_full.drop(["Survived"], axis = 1).copy()
labels = pd.DataFrame(df_full["Survived"].copy())


class DataPipeLine:
    def __init__(self):
        self.titles = {
                'Mr' : 'Ordinary',
                'Mrs' : 'Ordinary',
                'Miss' : 'Ordinary',
                'Master' : 'High',
                'Rev' : 'Ordinary',
                'Dr' : 'High',
                'Mme' : 'Ordinary',
                'Ms' : 'Ordinary',
                'Major' : 'High',
                'Lady' : 'High',
                'Sir' : 'High',
                'Mlle' : 'Ordinary',
                'Col' : 'High',
                'Capt' : 'Ordinary',
                'the Countess' : 'High',
                'Jonkheer' : 'High',
                'Don' : 'High'
                }
        self.default_title = "Ordinary"
        
    
    def fit(self, df_, label_):
        #loc_df = df_.copy()
        pass

    def transform(self, df_, label_):
        loc_df = df_.copy()
        
        # title classification
        loc_df = self.addTitle(loc_df)
        loc_df["Title_class"] = loc_df["Title"].apply(self.title_class) == "High"
        
        
        return loc_df
    
    def addTitle(self, df_):
        res = df_.copy()
        res['Title'] = (df_["Name"].apply(lambda x: x[
                x.find(", ") + 2:
                x.find(".")]
            ))
        return res
    
    def title_class(self, x):
        try:
            return self.titles[x]
        except KeyError:
            return self.default_title



# passangers from 1st class were more likely to survive
plt.hist(df["Pclass"][label[label["Survived"] == 1].index], 
         density = 0.5)
plt.hist(df["Pclass"][label[label["Survived"] == 0].index],
         density = 0.5)
plt.show()

df["Title"].value_counts()



pipe = DataPipeLine()
df_prepared = pipe.transform(df, labels)
    



