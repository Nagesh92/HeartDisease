import os
import pandas as pd
import numpy as np
import pickle

from sklearn.preprocessing import StandardScaler,MinMaxScaler,LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,ConfusionMatrixDisplay,confusion_matrix
from pandas.plotting import scatter_matrix


import matplotlib.pyplot as plt
import seaborn as sns

os.getcwd()

data = pd.read_csv('heart.csv',sep=',')
data

data.head(5)

data.tail(5)

## Listing columns in a data
data.columns

## statistics for the data
data.describe()

# Creating a dataframe object
df =pd.DataFrame(data)
df

# Checking for data types of data set
df.dtypes

## Check for missing values in the dataset
miss_val_perc = (df.isna().sum()/len(df))*100
miss_val_perc

print(df.groupby('HeartDisease').size())

## Visualizing the data
scatter_matrix(df)
plt.show()

## Box Plot
df.plot(kind='box', subplots=True, sharex=False, sharey=False)
plt.show()

## creating histogram
df.hist()
plt.show()


## Correlation matrix
#df.corr()

## check for numerical cols
numcols = df.select_dtypes(include=['int64','float64']).dtypes
numcols

## check for categorical columns
catcols = df.select_dtypes(include=['object']).dtypes
catcols



## Converting categorical to numerical variables
l = LabelEncoder()
l.fit(df.ChestPainType.drop_duplicates())
df.ChestPainType = l.transform(df.ChestPainType)
df.ChestPainType


l.fit(df.RestingECG.drop_duplicates())
df.RestingECG = l.transform(df.RestingECG)
df.RestingECG

l.fit(df.Sex.drop_duplicates())
df.Sex = l.transform(df.Sex)
df.Sex


l.fit(df.ExerciseAngina.drop_duplicates())
df.ExerciseAngina = l.transform(df.ExerciseAngina)
df.ExerciseAngina


l.fit(df.ST_Slope.drop_duplicates())
df.ST_Slope = l.transform(df.ST_Slope)
df.ST_Slope

df.Sex
df.ST_Slope.unique
df.ExerciseAngina
df.RestingECG
df.ChestPainType

df.ST_Slope




## Looking at Dataframe
df

## Defining dependent and Independent features
X = df.drop(['HeartDisease'],axis =1)
y = df.HeartDisease

X
y
## Splitt Data into train and test
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)
print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

y_train

## Build Logistic regression model
lr = LogisticRegression()
lr.fit(X_train,y_train)

lr_pred = lr.predict(X_test)
lr_Score = accuracy_score(y_test,lr_pred)
lr_Score

## Build a RandomForest Model
rf = RandomForestClassifier()
rf.fit(X_train,y_train)

rf_pred = rf.predict(X_test)
rf_score = accuracy_score(y_test,rf_pred)
rf_score
rf_pred
X



pickle.dump(rf,open('model.pkl','wb'))
model = pickle.load(open('model.pkl','rb'))

model.predict([[40,1,3,200,300,0,1,180,1,2.0,0]])





# ## Buid SVC Model
# from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(X_train,y_train)
# sv_pred =sv.predict(X_test)
# sv_score = accuracy_score(y_test,sv_pred)
# sv_score

# ## Build a DecisionTree Model
# from sklearn.tree import DecisionTreeClassifier
# dt = DecisionTreeClassifier()
# dt.fit(X_train,y_train)
# dt_pred = dt.predict(X_test)
# dt_score = accuracy_score(y_test,dt_pred)
# dt_score

