import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

## read the data 
df = pd.read_csv("./datasets/diabetes_data/diabetes.csv")

##Split independent and dependent variable
X = df.iloc[:,0:8]
y = df.iloc[:,-1]

X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.1)

##Drop columsn not neede
df.drop(['BMI', 'DiabetesPedigreeFunction'],axis=1,inplace=True)
#print(df.info())

## Do scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

## Train the model
clf = LogisticRegression()
clf.fit(X_train, y_train)

## Make predictions
y_pred = clf.predict(X_test)

##Check accuracy
print(accuracy_score(y_test, y_pred))


# plt.scatter(df['Glucose'],df['BMI'],c=df['Outcome'])
# plt.show()