# EXNO:4-Feature Scaling and Selection

## Name: SUBASH M
## Reg no: 212224220109
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
data=pd.read_csv("/content/income(1) (1).csv",na_values=[ " ?"])
data
```
![image](https://github.com/user-attachments/assets/b5e855b4-fd31-4d8d-af38-b2479aa93add)
```
data.isnull().sum()
```
![image](https://github.com/user-attachments/assets/b139ae86-b645-46e8-b647-2e65001f81a0)
```
missing=data[data.isnull().any(axis=1)]
missing
```

![image](https://github.com/user-attachments/assets/0ae84707-c717-4cb7-b289-20307dd79c62)
```
data2=data.dropna(axis=0)
data2
```

![image](https://github.com/user-attachments/assets/ef237434-70af-49f2-bfe3-2787fbfdcc01)
```
sal=data["SalStat"]
data2["SalStat"]=data["SalStat"].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(data2['SalStat'])
```

![image](https://github.com/user-attachments/assets/fcc313f5-6bb6-45f4-9a9c-7fa761724b6d)
```
sal2=data2['SalStat']
dfs=pd.concat([sal,sal2],axis=1)
dfs
```

![image](https://github.com/user-attachments/assets/5b624bf9-ca21-4a73-9331-a342a881f68e)
```
data2
```

![image](https://github.com/user-attachments/assets/bbb4596d-b813-4186-ab8f-2d46834f57ef)
```
new_data=pd.get_dummies(data2, drop_first=True)
new_data
```

![image](https://github.com/user-attachments/assets/93ee0fa8-2587-4e11-835d-5389a90a1c61)
```
columns_list=list(new_data.columns)
print(columns_list)
```

![image](https://github.com/user-attachments/assets/fa100185-2f95-4f33-92e4-7b425678ec78)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```

![image](https://github.com/user-attachments/assets/377c8b48-c6fb-48cc-8f71-aeea894be090)
```
y=new_data['SalStat'].values
print(y)
```

![image](https://github.com/user-attachments/assets/083bd89a-f540-4eca-8297-804096907a7b)
```
x=new_data[features].values
print(x)
```

![image](https://github.com/user-attachments/assets/d28d8e6d-7349-439b-9571-5a750bee5e1c)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3,random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors = 5)
KNN_classifier.fit(train_x,train_y)
```

![image](https://github.com/user-attachments/assets/9a9b0cf9-ac43-41fe-a24f-db51c19225a8)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y, prediction)
print(confusionMatrix)
```

![image](https://github.com/user-attachments/assets/095c3392-a650-47b2-a935-45c14f1c99ae)

```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```

![image](https://github.com/user-attachments/assets/76e64ca5-bacb-4037-9789-5cf232782b74)
```
print("Misclassified Samples : %d" % (test_y !=prediction).sum())
```

![image](https://github.com/user-attachments/assets/542d6d11-5fae-4f8b-a710-6518b206cdf8)
```
data.shape
```

![image](https://github.com/user-attachments/assets/5bf00213-75ba-441d-84fe-45a65396d6fb)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest, mutual_info_classif, f_classif
data={
    'Feature1': [1,2,3,4,5],
    'Feature2': ['A','B','C','A','B'],
    'Feature3': [0,1,1,0,1],
    'Target'  : [0,1,1,0,1]
}

df=pd.DataFrame(data)
x=df[['Feature1','Feature3']]
y=df[['Target']]
selector=SelectKBest(score_func=mutual_info_classif,k=1)
x_new=selector.fit_transform(x,y)
selected_feature_indices=selector.get_support(indices=True)
selected_features=x.columns[selected_feature_indices]
print("Selected Features:")
print(selected_features)
```

![image](https://github.com/user-attachments/assets/5382bbed-79ae-4832-a252-9b904a3f3e2c)
```
import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import seaborn as sns
tips=sns.load_dataset('tips')
tips.head()
```

![image](https://github.com/user-attachments/assets/e082228b-70a7-48d8-a8b1-4a6305661d60)
```
tips.time.unique()
```

![image](https://github.com/user-attachments/assets/8c96519f-2506-43bb-91ed-34905f761c48)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```

![image](https://github.com/user-attachments/assets/199088e5-a950-4c87-9fc3-c30cd735e194)
```
chi2,p,_,_=chi2_contingency(contingency_table)
print(f"Chi-Square Statistics: {chi2}")
print(f"P-Value: {p}")
```

![image](https://github.com/user-attachments/assets/8897836f-10b4-41e4-8e10-bf50ad9db066)

# RESULT:
       Thus, Feature selection and Feature scaling has been used on thegiven dataset.
