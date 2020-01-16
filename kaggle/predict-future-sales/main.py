import numpy as np
import pandas as pd
import matplotlib
import seaborn as sns

import utils

data_path = "../data/competitive-data-science-predict-future-sales"

item_cat = pd.read_csv(data_path + '/item_categories.csv')
items = pd.read_csv(data_path + '/items.csv')
shops = pd.read_csv(data_path + '/shops.csv')
train = pd.read_csv(data_path + '/sales_train.csv')
test = pd.read_csv(data_path + '/test.csv')
sample_sub = pd.read_csv(data_path + '/sample_submission.csv')

utils.eda(train)


