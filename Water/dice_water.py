import sys
sys.path.append("..")
import dice_ml
from dice_ml.utils import helpers
from Dataloader import Water
from Adult.model import Net
import pandas as pd
import numpy as np
from Encoders.TabularEncoder import TabEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from category_encoders import *

from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import warnings

warnings.filterwarnings("ignore")

## 真实性验证分类器
NB = GaussianNB()
DT = DecisionTreeClassifier()
SVM = SVC()
MLP = MLPClassifier(max_iter=1000)
KNN = KNeighborsClassifier()

## 导入数据
adult_data = Water()
adult = adult_data.load_data()
X = adult_data.data
y = adult_data.target

encoder = TabEncoder(adult,adult_data.categoric)
X = encoder.encode(X)

## 按类别划分样本,并进行正则化
# 划分样本
low_income_samples = adult[adult["Potability"] == 0]
high_income_samples = adult[adult["Potability"] == 1]

# clf1 = Net("/datas/wangm/T-COL/Adult/v_model.joblib")

## 定义dice配置
#dataset = helpers.load_adult_income_dataset()
dataset = adult

target = dataset["Potability"]
train_dataset, test_dataset, y_train, y_test = train_test_split(dataset,
                                                                target,
                                                                test_size=0.2,
                                                                random_state=0,
                                                                stratify=target)
x_train = train_dataset.drop('Potability', axis=1)
x_test = test_dataset.drop('Potability', axis=1)
d = dice_ml.Data(dataframe=train_dataset,
                 continuous_features=["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"],
                 outcome_name='Potability')

numerical = ["ph","Hardness","Solids","Chloramines","Sulfate","Conductivity","Organic_carbon","Trihalomethanes","Turbidity"]
categorical = x_train.columns.difference(numerical)

categorical_transformer = Pipeline(steps=[
    ('target', TargetEncoder())])

transformations = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical)])

# Append classifier to preprocessing pipeline.
# Now we have a full prediction pipeline.
clf2 = Pipeline(steps=[('classifier', RandomForestClassifier())])
model = clf2.fit(x_train, y_train)

m = dice_ml.Model(model=model,backend="sklearn")

## 生成反事实
n_cfs = 5
n_query = 5

# high_income_query = high_income_samples.iloc[5:7]
low_income_query = pd.read_csv("samples.csv", header=0)

## 使用dice生成反事实解释

k_exp = dice_ml.Dice(d,m,method="kdtree")

dice_exp = k_exp.generate_counterfactuals(low_income_query.drop("Potability",axis=1),total_CFs=n_cfs*len(low_income_query),desired_class="opposite")
dice_CFs = dice_exp.cf_examples_list[0].final_cfs_df
dice_CFs.insert(9,"Potability",np.array([1 for i in range(len(dice_CFs))]))
dice_CFs.to_csv("dice_k.csv",index = False)
print("kdtree ending.")

r_exp = dice_ml.Dice(d,m,method="random")

dice_exp = r_exp.generate_counterfactuals(low_income_query.drop("Potability",axis=1),total_CFs=n_cfs*len(low_income_query),desired_class="opposite")
dice_CFs = dice_exp.cf_examples_list[0].final_cfs_df
# dice_CFs.insert(9,"Potability",np.array([1 for i in range(len(dice_CFs))]))
dice_CFs.to_csv("dice_r.csv",index = False)
print("random ending.")

g_exp = dice_ml.Dice(d,m,method="genetic")

dice_exp = g_exp.generate_counterfactuals(low_income_query.drop("Potability",axis=1),total_CFs=n_cfs*len(low_income_query),desired_class="opposite")
dice_CFs = dice_exp.cf_examples_list[0].final_cfs_df
# dice_CFs.insert(8,"income",np.array([1 for i in range(len(dice_CFs))]))
dice_CFs.to_csv("dice_g.csv",index = False)
print("genetic ending.")





