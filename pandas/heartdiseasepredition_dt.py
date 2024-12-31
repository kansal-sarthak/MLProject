import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import cross_val_score

df = pd.read_csv("./datasets/heart_disease_data/heart.csv")

df['trestbps'].fillna(df['trestbps'].mean(), inplace=True)

X = df.iloc[:, 0:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
scaler.fit_transform(X_train)
scaler.transform(X_test)

type_classification = DecisionTreeClassifier(max_depth=8)
type_classification.fit(X_train, y_train)
y_pred = type_classification.predict(X_test)


print(accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred))

cvs = cross_val_score(type_classification, X, y, cv=10)
print("Multiple iteration average for modal performance: ",cvs.mean()*100)

