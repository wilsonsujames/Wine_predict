import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

wine = pd.read_csv('./winequality-red.csv')

print(wine.head())

# 看fixed acidity跟品質的相關性
# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'fixed acidity', data = wine)
# plt.show()

# 看volatile acidity跟品質的相關性
# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'volatile acidity', data = wine)
# plt.show()

# 看citric acid跟品質的相關性
# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'citric acid', data = wine)
# plt.show()

# 看residual sugar跟品質的相關性
# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'residual sugar', data = wine)
# plt.show()


#Composition of chloride also go down as we go higher in the quality of the wine
# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'chlorides', data = wine)
# plt.show()



# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'free sulfur dioxide', data = wine)
# plt.show()


# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'total sulfur dioxide', data = wine)
# plt.show()

# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'sulphates', data = wine)
# plt.show()


# fig = plt.figure(figsize = (10,6))
# sns.barplot(x = 'quality', y = 'alcohol', data = wine)
# plt.show()


bins = (2, 6.5, 8)
group_names = ['bad', 'good']
wine['quality'] = pd.cut(wine['quality'], bins = bins, labels = group_names)

#Now lets assign a labels to our quality variable
label_quality = LabelEncoder()

#Bad becomes 0 and good becomes 1 
wine['quality'] = label_quality.fit_transform(wine['quality'])

print(wine['quality'].value_counts())

sns.countplot(wine['quality'])
plt.show()

X = wine.drop('quality', axis = 1)
y = wine['quality']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)

print(classification_report(y_test, pred_rfc))

print(confusion_matrix(y_test, pred_rfc))

sgd = SGDClassifier(penalty=None)
sgd.fit(X_train, y_train)
pred_sgd = sgd.predict(X_test)

print(classification_report(y_test, pred_sgd))