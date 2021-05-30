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




df = pd.DataFrame(pd.read_csv(DATASET_TRAIN_PATH))
df.info()
