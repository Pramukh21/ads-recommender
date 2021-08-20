import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split

ad_data = pd.read_csv('advertising.csv')
ad_data.columns
ad_data.head()
ad_data.info()
ad_data.describe()
#ad_data['Age'].plot.hist(bins = 30)
ad_data[ad_data['Clicked on Ad']==1]['Age'].hist(bins = 100 , color = 'blue',label = 'Clicked on Ad 1',alpha = 0.6)
ad_data[ad_data['Clicked on Ad']==0]['Age'].hist(bins =100 , color = 'red',label = 'Clicked on Ad 0',alpha =0.6)
plt.legend()
plt.xlabel('Age')

X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
       'Daily Internet Usage','Male']]

y = ad_data['Clicked on Ad']


X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state = 101)

lm = LogisticRegression()
lm.fit(X_train,y_train)
prediction = lm.predict(X_test)

accuracy = lm.score(X_test,y_test)
print(accuracy*100)

from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))
print(confusion_matrix(y_test,prediction))