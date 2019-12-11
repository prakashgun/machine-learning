import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv('diabetes.csv')

zero_not_accepted = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']

for column in zero_not_accepted:
    dataset[column] = dataset[column].replace(0, np.NaN)
    mean = int(dataset[column].mean(skipna=True))
    dataset[column] = dataset[column].replace(np.NaN, mean)

x = dataset.iloc[:, 0:8]
y = dataset.iloc[:, 8]

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.2)
x_test_orig = x_test
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.fit_transform(x_test)

# import math
# print(math.sqrt(len(y_test)))

classifier = KNeighborsClassifier(n_neighbors=11, p=2, metric='euclidean')
classifier.fit(x_train, y_train)

y_pred = classifier.predict(x_test)

cm = confusion_matrix(y_test, y_pred)
print(cm)
print(f1_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
