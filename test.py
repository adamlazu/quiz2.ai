import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

dataset = pd.read_csv("./iris.csv")
X=[]
for i in range (len(dataset)):
    X.append(dataset.iloc[i, :-1].tolist())
y = dataset.iloc[:, -1].tolist()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=60)

def classify(sample):
    sepal_area= sample[0]*sample[1]
    petal_area = sample[2]*sample[3]
    sum_area = sepal_area+petal_area

    avg_sum_setosa = np.mean((dataset["sepallength"][dataset["class"]=="Iris-setosa"]*dataset["sepalwidth"][dataset["class"]=="Iris-setosa"])+(dataset["petallength"][dataset["class"]=="Iris-setosa"]*dataset["petalwidth"][dataset["class"]=="Iris-setosa"]))
    avg_sum_versicolor = np.mean((dataset["sepallength"][dataset["class"]=="Iris-versicolor"]*dataset["sepalwidth"][dataset["class"]=="Iris-versicolor"])+(dataset["petallength"][dataset["class"]=="Iris-versicolor"]*dataset["petalwidth"][dataset["class"]=="Iris-versicolor"]))
    avg_sum_virginica = np.mean((dataset["sepallength"][dataset["class"]=="Iris-virginica"]*dataset["sepalwidth"][dataset["class"]=="Iris-virginica"])+(dataset["petallength"][dataset["class"]=="Iris-virginica"]*dataset["petalwidth"][dataset["class"]=="Iris-virginica"]))

    diff_setosa = abs(sum_area-avg_sum_setosa)
    diff_versicolor = abs(sum_area-avg_sum_versicolor)
    diff_virginica = abs(sum_area-avg_sum_virginica)

    if min(diff_setosa,diff_versicolor,diff_virginica) == diff_setosa:
        return "Iris-setosa"
    elif min(diff_setosa,diff_versicolor,diff_virginica) == diff_versicolor:
        return "Iris-versicolor"
    else:
        return "Iris-virginica"
    






y_pred = [classify(sample) for sample in X_test]

accuracy = accuracy_score(y_test, y_pred)

print(f"accuracy = {accuracy}")