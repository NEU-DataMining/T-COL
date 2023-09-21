from Dataloader import Adult, German, Airline, Titanic, Water, Phoneme
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
import numpy as np
from Encoders.TabularEncoder import TabEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import argparse

parse = argparse.ArgumentParser()
parse.add_argument("-d", "--data", choices=["german", "adult", "water", "titanic", "airline", "phoneme"], default="adult", help="Choosing a dataset.")
args = parse.parse_args()

if args.data == "adult":
    adult = Adult()
    pass
elif args.data == "german":
    adult = German()
    pass
elif args.data == "water":
    adult = Water()
    pass
elif args.data == "titanic":
    adult = Titanic()
    pass
elif args.data == "airline":
    adult = Airline()
    pass
elif args.data == "phoneme":
    adult = Phoneme()
    pass
else:
    adult = None
    pass

adult_data = adult.load_data()
adult_X,adult_y = adult.data,adult.target

cv = KFold(n_splits=10,random_state=0,shuffle=True)
adult_encoder = TabEncoder(adult_data,adult.categoric)

adult_X = adult_encoder.encode(adult_X)

adult_KNN = KNeighborsClassifier()
adult_MLP = MLPClassifier(max_iter=1500)
adult_SVM = SVC()
adult_DT = DecisionTreeClassifier()
adult_NB = GaussianNB()

adult_KNN_scores = cross_val_score(adult_KNN,adult_X,adult_y,cv = cv,scoring = "f1_macro")
adult_MLP_scores = cross_val_score(adult_MLP,adult_X,adult_y,cv = cv,scoring = "f1_macro")
adult_SVM_scores = cross_val_score(adult_SVM,adult_X,adult_y,cv = cv,scoring = "f1_macro")
adult_DT_scores = cross_val_score(adult_DT,adult_X,adult_y,cv = cv,scoring = "f1_macro")
adult_NB_scores = cross_val_score(adult_NB,adult_X,adult_y,cv = cv,scoring = "f1_macro")

print("KNN:",adult_KNN_scores,np.mean(adult_KNN_scores))
print("MLP:",adult_MLP_scores,np.mean(adult_MLP_scores))
print("SVM:",adult_SVM_scores,np.mean(adult_SVM_scores))
print("DT:",adult_DT_scores,np.mean(adult_DT_scores))
print("NB:",adult_NB_scores,np.mean(adult_NB_scores))