# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Data Collection
2. Display the first and last few rows of the dataset using head() and tail() to ensure it's loaded correctly.
3. Split the Data into Training and Testing Sets
4. Create a Linear Regression model: Use LinearRegression() from sklearn.linear_model to instantiate the model.
5. Use the trained model to make predictions on the test data (x_test) by calling the predict() method.
6. Calculate the evaluation metrics.
7. Visualize the training set results: Plot a scatter plot of the training data (x_train and y_train) and overlay the regression line (predicted values).

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:M.Mahalakshmi 
RegisterNumber: 24900868 
*/
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
print(df)
df.head(0)
df.tail(0)
print(df.head())
print(df.tail())
x=df.iloc[:,:-1].values
print(x)
y=df.iloc[:,-1].values
print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
print(y_pred)
print(y_test)
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
plt.scatter(x_test,y_test,color='black')
plt.plot(x_test,regressor.predict(x_test),color='red')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE=',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE=',mae)
rmse=np.sqrt(mse)
print("RMSE=",rmse)
```

## Output:
![simple linear regression model for predicting the marks scored](sam.png)
![download](https://github.com/user-attachments/assets/bacc74c2-540b-40e7-bc42-8db25e3f9ead)
![download](https://github.com/user-attachments/assets/80ef942b-1e59-403c-b9d0-e014ab0a960e)


## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
