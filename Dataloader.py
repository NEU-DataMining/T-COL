import numpy as np
import pandas as pd
import os

PATH = os.path.join("/data/wangm/GLT-fastCFs/Datasets/")

class Climate():
    """
    UCI气象数据集
    Instances: 540
    Attributes: 18
    - 18个气象模型的参数[0,1]
    Classes: 2
    - 模拟结果: 成功1/失败0
    """
    climate = pd.read_csv(PATH+'pop_failures.csv',header=0)
    climate_data = pd.DataFrame(climate)
    climate_array = np.array(climate_data,dtype='float')
    def __init__(self):
        self.data = self.climate_array[:,:-1]
        self.target = self.climate_array[:,-1]
        pass
    def load_array(self):
        return self.climate_array
    def load_data(self):
        return self.climate_data
    pass

class Adult():
    """
    UCI成年人收入数据集，Dice文章筛选过的版本
    """
    # columns = ["age","workclass","fnlwgt","education","education-num","marital-status","occupation",
    #            "relationship","race","sex","capital-gain","capital-loss","hours-per-week","native-country","income"]
    adult = pd.read_csv(PATH+"adult_pre.csv",header=0,index_col=0)
    adult_data = pd.DataFrame(adult)
    # adult_data["income"].replace(" <=50K",0,inplace=True)           # 类别编码
    # adult_data["income"].replace(" >50K", 1, inplace=True)
    def __init__(self):
        self.data = self.adult_data.iloc[:,:-1]
        self.target = self.adult_data["income"]
        self.categoric = ["workclass","education","marital_status","occupation","race","gender"]
        self.continues = self.adult_data.columns.difference(self.categoric).drop("income").values.tolist()
        self.categorical_features = self.adult_data[self.categoric]
        self.continuous_features = self.adult_data[self.continues]
        pass
    def load_data(self):
        return self.adult_data
    pass

class Student():
    """
    UCI学生表现数据集
    """
    student = pd.read_csv(PATH+"students.csv",header=0)
    student_data = pd.DataFrame(student)
    def __init__(self):
        self.student_data["G3"] = self.student_data["G3"].apply(lambda x: self.class_transfer(x))
        self.data = self.student_data.iloc[:,:-1]
        self.target = self.student_data["G3"]
        self.categoric = ["school","sex","address","famsize","Pstatus","Mjob","Fjob","reason","guardian","schoolsup","famsup",
                          "paid","activities","nursery","higher","internet","romantic"]
        self.continues = self.student_data.columns.difference(self.categoric).drop("G3")
        self.categorical_features = self.student_data[self.categoric]
        self.continuous_features = self.student_data[self.continues]
    def class_transfer(self,x):
        if x >= 12:
            return 1
        else:
            return 0
        pass
    def load_data(self):
        return self.student_data
    pass

class German():
    """
    UCI德国信用卡数据集
    """
    columns = ["Estatus","Duration","Chistory","Purpose","Ncredit","Saccount","EmploySince","Insrate","Pstatus","Odebtors",
               "ResidenceSince","Property","Age","Insplans","House","NTBcredit","Job","NMpeople","Telephone","Fworker","credits"]
    german = pd.read_csv(PATH+"german.csv",names=columns)
    german_data = pd.DataFrame(german)
    german_data["credits"].replace(2,0,inplace=True)
    def __init__(self):
        self.data = self.german_data.iloc[:,:-1]
        self.target = self.german_data["credits"]
        self.categoric = ["Estatus","Chistory","Purpose","Saccount","EmploySince","Pstatus","Odebtors",
                            "Property","Insplans","House","Job","Telephone","Fworker"]
        self.continues = self.german_data.columns.difference(self.categoric).drop("credits").values.tolist()
        self.categorical_features = self.german_data[self.categoric]
        self.continuous_features = self.german_data[self.continues]
        pass
    def load_data(self):
        return self.german_data
    pass

class Shopping():
    """
    UCI消费者购物倾向数据集
    """
    shopping = pd.read_csv(PATH+"shopping.csv",header=0)
    shopping_data = pd.DataFrame(shopping)
    shopping_data["Revenue"].replace("F",0,inplace=True)
    shopping_data["Revenue"].replace("T", 1, inplace=True)
    def __init__(self):
        self.data = self.shopping_data.iloc[:,:-1]
        self.target = self.shopping_data["Revenue"]
        self.categoric = ["Month","OperatingSystems","Browser","Region","TrafficType","VisitorType","Weekend"]
        self.continues = self.shopping_data.columns.difference(self.categoric).drop("Revenue")
        self.categorical_features = self.shopping_data[self.categoric]
        self.continuous_features = self.shopping_data[self.continues]
        pass
    def load_data(self):
        return self.shopping_data
    pass

class Car():
    """
    UCI汽车满意度数据集
    """
    car = pd.read_csv(PATH+"car.csv",header=0)
    car_data = pd.DataFrame(car)
    car_data["class"].replace("unacc",0,inplace=True)
    car_data["class"].replace("acc", 1, inplace=True)
    def __init__(self):
        self.data = self.car_data.iloc[:,:-1]
        self.target = self.car_data["class"]
        self.categoric = ["buying","maint","doors","persons","lug_boot","safety"]
        self.continues = []
        self.categorical_features = self.car_data[self.categoric]
        self.continuous_features = self.car_data[self.continues]
        pass

    def load_data(self):
        return self.car_data

    pass

