import numpy as np
import pandas as pd
import sklearn
import pandas as pd
from sklearn import preprocessing
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neural_network import MLPClassifier

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data"

# Assign colum names to the dataset
names = ['Alcohol','Malic Acid','Ash','Alcalinity of Ash','Magnesium','Total phenols','Flavanoids','Nonflavanoid Phenols',
         'Proanthocyanins','Color intensity','Hue','OD280/OD315 of diluted wines','Proline'] 

winedata = pd.read_csv(url, names=names)

# Assign data from first four columns to X variable
X = winedata.iloc[:, 0:13]

# Assign data from first fifth columns to y variable
y = []
for i in range(59):
  y.append(1)
for i in range(71):
  y.append(2)
for i in range(48):
  y.append(3)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)

scaler = StandardScaler()
scaler.fit(X_train)

X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(100, 500, 10), max_iter=2000)
mlp.fit(X_train, y_train)

predictions = mlp.predict(X_test)

# compare the predicted labels to the actual labels and conver to percentage to output
acc = metrics.accuracy_score(y_test, predictions)*100
print(f"{acc}% accuracy")
