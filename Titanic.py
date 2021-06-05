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
        self.numeric_features = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
        self.categoric_features = ["Name", "Sex", "Ticket", "Embarked"]
        
    
    def fit(self, df_, label_):
        self.mode = df_.mode()
        self.median = df_[self.numeric_features].median()
        self.avg = df_[self.numeric_features].mean()
        
        self.sc = StandardScaler(with_mean = False)
        self.encoder = OneHotEncoder()
        
        loc_df = self.transform(df_, False)
        self.sc.fit(loc_df)
        
        return [self.mode, self.median, self.avg]

    def transform(self, df_, scale = True, encode = True):
        loc_df = df_.copy()
        
        # NaN fill
        for n in self.numeric_features:
            loc_df[n] = loc_df[n].fillna(value = self.median[n])
        for n in self.categoric_features:
            loc_df[n] = loc_df[n].fillna(value = self.mode[n])
        
        # level
        loc_df["Level"] = loc_df["Cabin"].apply(self.get_level)
        loc_df.loc[(loc_df["Level"].isna()) & (loc_df["Fare"] < 8.0), "Level"] = 'F'
        loc_df.loc[(loc_df["Level"].isna()) & (loc_df["Fare"] > 25.0), "Level"] = 'B'
        loc_df.loc[loc_df["Level"].isna(), "Level"] = 'D'
        
        # title classification
        loc_df = self.addTitle(loc_df)
        loc_df["Title_class_high"] = loc_df["Title"].apply(self.title_class) == "High"
        loc_df = loc_df.drop(["Title"], axis = 1)
        
        # male / female
        loc_df["Is_female"] = loc_df["Sex"] == "female"
        loc_df = loc_df.drop(["Sex"], axis = 1)
        
        # age cat
        loc_df["Underage"] = loc_df["Age"] <= 14
        loc_df["Middleage"] = ((loc_df["Age"] > 14) & 
                               (loc_df["Age"] < 45))
        loc_df["Elderly"] = loc_df["Age"] >= 45
        
        # dropping other
        loc_df = loc_df.drop(["Name", "Cabin", "Ticket"], axis = 1)
  
        # other cat

        # dropping other
        loc_df["Embarked_S"] = loc_df["Embarked"] == "S"
        loc_df["Embarked_C"] = loc_df["Embarked"] == "C"
        loc_df["Embarked_Q"] = loc_df["Embarked"] == "Q"
        loc_df["Level_A"] = loc_df["Level"] == "A"
        loc_df["Level_B"] = loc_df["Level"] == "B"
        loc_df["Level_C"] = loc_df["Level"] == "C"
        loc_df["Level_D"] = loc_df["Level"] == "D"
        loc_df["Level_E"] = loc_df["Level"] == "E"
        
        
        loc_df = loc_df.drop(["Level"], axis = 1)
        loc_df = loc_df.drop(["Embarked"], axis = 1)
         
        if (scale):
            print("Scaling...")
            loc_df = pd.DataFrame(self.sc.transform(loc_df), columns = loc_df.columns)
        
        
        
        
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
        
    def get_level(self, x):
        try:
            return x[0]
        except TypeError:
            return None


# passangers from 1st class were more likely to survive
plt.hist(df["Pclass"][labels[labels["Survived"] == 1].index], 
         density = 0.5)
plt.hist(df["Pclass"][labels[labels["Survived"] == 0].index],
         density = 0.5)
plt.show()




pipe = DataPipeLine()
pipe.fit(df, labels)
df_prepared = pipe.transform(df)


