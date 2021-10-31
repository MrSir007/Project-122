import pandas as pd
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score as acc

X = np.load("image.npz")["arr_0"]
y = pd.read_csv("labels.csv")
classes = ["A","B","C","D","E","F","G","H","J","K","L","M","N","O","P","Q","R","S","T","U","V","W","X","Y","Z"]
nClasses = len(classes)

samples = 5
figure = plt.figure(figsize=(nClasses*2,(1+samples*2)))
index = 0
for c in classes :
  indexes = np.flatnonzero(y == c)
  indexes = np.random.choice(indexes, samples, replace=False)
  a = 0
  for i in indexes :
    pltIndex = a * nClasses + index + 1
  heat = sb.heatmap(np.reshape(X[i], (22,30)), cmap=plt.cm.gray, xticklabels=False, yticklabels=False, cbar=False)
  '''plt.show()'''

xTrain, xTest, yTrain, yTest = tts(X, y, test_size=0.25, random_state=36)
ss = StandardScaler()
xTrain = ss.fit_transform(xTrain)
xTest = ss.fit_transform(xTest)
model = LogisticRegression(random_state=0)
model.fit(xTrain, yTrain)
yPredict = model.predict(xTest)

cm = pd.crosstab(yTest, yPredict, rownames=["Actual"], colnames=["Predicted"])
p = plt.figure(figsize=(10,10))
p = sb.heatmap(cm, annot=True, fmt="d", cbar=False)
plt.show(p)