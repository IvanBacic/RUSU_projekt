import numpy as np
import pandas as pd

df = pd.read_csv("C:\\Users\Ivan\Desktop\RUSU\Strojno projekt\water_potability.csv")

#first 5 rows of data
print(df.head())

#shape of the data
print(df.shape)

#Check for missing values
print(df.isnull().sum())

#Dropping missing values
#because water quality is a sensitive data, we cannot tamper with the data by imputing mean, median, mode
df= df.dropna()

print(df.Potability.value_counts())

#Plots
import matplotlib.pyplot as plt
import seaborn as sns

df.Potability.value_counts().plot(kind ='pie')

zero  = df[df['Potability']==0]   #zero values in Potability column
one = df[df['Potability']==1]  # one values in Potability column


from sklearn.utils import resample
plt.figure(num=1)
plt.show()

#minority class that  is 1, we need to upsample/increase that class so that there is no bias
#n_samples = 1998 means we want 1998 sample of class 1, since there are 1998 samples of class 0
df_minority_upsampled = resample(one, replace = True, n_samples = 1200) 
#concatenate
df = pd.concat([zero, df_minority_upsampled])

from sklearn.utils import shuffle

df = shuffle(df) # shuffling so that there is particular sequence

plt.figure(num=2)
df.Potability.value_counts().plot(kind ='pie')
plt.show()


#understanding correlation
plt.figure(num=3,figsize = (12,7))
sns.heatmap(df.corr(), annot = True)
plt.show()


sns.scatterplot(x=df["ph"], y=df["Hardness"], hue=df.Potability,
data=df)
plt.show()

sns.scatterplot(x=df["ph"], y=df["Chloramines"], hue=df.Potability,
data=df)
plt.show()

sns.scatterplot(x=df["ph"], y=df["Chloramines"], hue=df.Potability,
data=df)
plt.show()

print(df.corr().abs()['Potability'].sort_values(ascending = False))

# Standardizing data
X = df.drop(['Potability'], axis = 1)
y = df['Potability']

from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
features= X.columns
X[features] = sc.fit_transform(X[features])


from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (GridSearchCV, train_test_split)
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.1)


lr = LogisticRegression(random_state=42) #solver lbfgs

knn = KNeighborsClassifier()

rf = RandomForestClassifier()



para_knn = {'n_neighbors':np.arange(1, 50)}  #parameters of knn
grid_knn = GridSearchCV(knn, param_grid=para_knn, cv=5) #search knn for 5 fold cross validation
 

#parameters for random forest
#n_estimators: The number of trees in the forest.
params_rf = {'n_estimators':[100,200, 350, 500], 'min_samples_leaf':[2, 10, 30]}
grid_rf = GridSearchCV(rf, param_grid=params_rf, cv=5)

grid_knn.fit(X_train, y_train)
grid_rf.fit(X_train, y_train)

print("Best parameters for KNN:", grid_knn.best_params_)
print("Best parameters for Random Forest:", grid_rf.best_params_)

lr = LogisticRegression(random_state=42,max_iter=500,solver='lbfgs') #solver lbfgs

knn = KNeighborsClassifier(n_neighbors=1)
rf = RandomForestClassifier(n_estimators=100, min_samples_leaf=2, random_state=42)

classifiers = [('Logistic Regression', lr), ('K Nearest Neighbours', knn),
               ('Random Forest', rf)]

from sklearn.metrics import accuracy_score

for classifier_name, classifier in classifiers:
 
    # Fit clf to the training set
    classifier.fit(X_train, y_train)    
   
    # Predict y_pred
    y_pred = classifier.predict(X_test)
    accuracy = accuracy_score(y_test,y_pred)
    

   
    # Evaluate clf's accuracy on the test set
    print('{:s} : {:.2f}'.format(classifier_name, accuracy))


from sklearn.metrics import classification_report

y_pred_rf= rf.predict(X_test)
print(classification_report(y_test, y_pred_rf))
