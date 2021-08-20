import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

ad_data = pd.read_csv('advertising.csv')
ad_data.columns
ad_data.head()
ad_data.info()
ad_data.describe()
#ad_data['Age'].plot.hist(bins = 30)

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male']]

y = ad_data['Clicked on Ad']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 101)

lm = LinearRegression()
lm.fit(X_train,y_train)
prediction = lm.predict(X_test)

def best_fit(X, Y):

    xbar = sum(X)/len(X)
    ybar = sum(Y)/len(Y)
    n = len(X) # or len(Y)

    numer = sum([xi*yi for xi,yi in zip(X, Y)]) - n * xbar * ybar
    denum = sum([xi**2 for xi in X]) - n * xbar**2

    b = numer / denum
    a = ybar - b * xbar

    print('best fit line:\ny = {:.2f} + {:.2f}x'.format(a, b))

    return a, b

# solution
a, b = best_fit(y_test, prediction)

plt.scatter(y_test, prediction)
yfit = [a + b * xi for xi in y_test]
plt.plot(y_test, yfit)

accuracy = lm.score(X_test,y_test)
print("Linear Regression score :",accuracy*100)

dtree = DecisionTreeClassifier()
dtree.fit(X_train,y_train)
predict = dtree.predict(X_test)
acc = dtree.score(X_test,y_test)
print("dtree score : ",acc*100)
print(classification_report(y_test,predict))

print("\n\n",confusion_matrix(y_test,predict))

error_rate = []
for i in range(1,40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i!=y_test))
    
    

plt.figure(figsize = (10,6))
plt.plot(range(1,40),error_rate,color = 'blue',marker ='o',ls = '--',markersize = 10,markerfacecolor = 'red')
plt.title('ERROR RATE VS K VALUE')
plt.xlabel('K')
plt.ylabel('Eroor rate')


kn = KNeighborsClassifier(n_neighbors=1)
kn.fit(X_train,y_train)
predknn = knn.predict(X_test)
print("KNN")
print(confusion_matrix(y_test,predknn))
print(classification_report(y_test,predknn))
