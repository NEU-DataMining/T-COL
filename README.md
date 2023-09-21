# T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems

T-COL is the code implementation of the paper [**T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems**](the/url/of/paper/need), which is also a python library to generate counterfactual explanations with the classifier considered as a black box.

We will continue to maintain and improve the program! If you are confused about any part of this project, please do not hesitate to contact [me](mailto:sci.m.wang@gmail.com) directly.

## Datasets
### Inner available datasets
- Adult Income: [Source](https://archive.ics.uci.edu/ml/datasets/adult)
- German Credit: [Source](https://www.kaggle.com/datasets/uciml/german-credit)
- Titanic: [Source](https://www.kaggle.com/competitions/titanic/data)
- Boston Housing: [Source](https://www.kaggle.com/competitions/boston-housing/data)
- Breast Cancer Wisconsin: [Source](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)
- Pima Indians Diabetes: [Source](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

### Customized datasets
You can generate counterfactual explanations for arbitrary datasets, you just need to define a python `class` in '.py' file like `Dataloader.py`:
```python
class MyData():
    my_data = pd.read_csv("path/to/csv/file",header=0,index_col=0)
    data = pd.DataFrame(my_data)
    def __init__(self):
        self.data = self.data.iloc[:,:-1]
        self.target = self.data["name of class"]
        self.categoric = ["colums which are categorical"]
        self.continues = self.data.columns.difference(self.categoric).drop("name of class").values.tolist()
        self.categorical_features = self.data[self.categoric]
        self.continuous_features = self.data[self.continues]
        pass
    def load_data(self):
        return self.data
    pass
```

Then, you can use your dataset in other files by `from Dataloader import MyData`

## Begin to use T-COL
### Requirements
- category_encoders==2.6.0
- dice_ml==0.9
- joblib==1.3.1
- numpy==1.24.3
- pandas==1.5.3
- prettytable==3.8.0
- scikit_learn==1.2.2
- torch==2.0.1
- tqdm==4.65.0

### Necessary files
```
data
├── Dataloader.py
├── Encoders
    ├── TabularEncoder.py
T-COL
├── ExtractProto.py
├── GLT.py
├── CreateCFsWithGLT.py
├── tree_path.json
├── vali_model.py
evaluation
├── preference.py
├── sparsity.py
├── datafidelity.py
```

### Files needed for each dataset
*Take adult income as an example.*
```
dataset
├── Datasets
    ├── adult_pre.csv
generate CEs
├── gen_adult.py
├── Adult
    ├── dice_adult.py
```


### Reappearance Experiments
If you just want to reappearance the results in the paper, please do not run the "generate.py". The samples will be random selected when run the "generate.py" and the result may change.
```bash
python preference.py -d [name_of_dataset]       # the name can be ["adult","german","titanic","water","phoneme"]
python sparsity.py -d [name_of_dataset]
python datafidelity.py -d [name_of_dataset]
```
If you want to generate new CEs and evaluation, the CEs can be generated following the bellow commands.
```bash
# First to generate counterfactual explanations for selected datasets.
python generate.py -p 'rep' -f 'rss' -d 'adult' -dp 3 -g 0 -vm 'RF' -dl 1 -sl 0 -n 5
# Then generate counterfactual explanations with Dice.
python ./Adult/dice_adult.py    # take adult as an example
# Then constract the results.
python preference.py -d [name_of_dataset]       # the name can be ["adult","german","titanic","water","phoneme"]
python sparsity.py -d [name_of_dataset]
python datafidelity.py -d [name_of_dataset]
```
You can also generate CEs by LLM (Spark) running the "Spark_CEs.py". Before you run this file, an [api](https://xinghuo.xfyun.cn/sparkapi) is needed.
```bash
python Spark_CEs.py -d [name_of_dataset]
```

### Usage Example
```python
from CreateCFsWithGLT import Creator
from joblib import dump
from vali_model import Net

dump(v_model,"path/to/v_model.joblib")
model = Net("path/to/v_model.joblib")

_data = German()
data = _data.load_data()
creator = Creator(model, data, s, _data.categorical_features, args.depth, args.d_label, n_ces)

proto = "good"
CEs = pd.DataFrame(creator.createCFs(args.proto, dataset=args.data, device=args.gpu, func = args.func), columns=data.columns.values)            # 默认german,cos,fcs a
CEs.to_csv("Adult/colt_good_fcs.csv",index=False)
```

## Citation
This paper hasn't been accepted, please cite the project on Github with the bib code bellow:
```bib
@misc{wang2023tcol,
      title = {T-COL: Generating Counterfactual Explanations for General User Preferences on Variable Machine Learning Systems},
      author = {Wang, Ming and Wang, Daling and Wu, Wenfang and Feng, Shi and Zhang, Yifei},
      year = {2023},
      month = {9},
      version = {0.2},
      url = {https://github.com/NEU-DataMining/T-COL}
}
```