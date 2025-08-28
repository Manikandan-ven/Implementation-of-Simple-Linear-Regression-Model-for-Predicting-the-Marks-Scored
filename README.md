# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Start the Program.
2. Import the necessary packages.
3. Read the given csv file and display the few contents of the data.
4. Assign the features for x and y respectively.
5. Split the x and y sets into train and test sets.
6. Convert the Alphabetical data to numeric using CountVectorizer.
7. Predict the number of spam in the data using SVC (C-Support Vector Classification) method of SVM (Support vector machine) in sklearn library.
8. Find the accuracy of the model. 9.End the Program.


## Program:
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('/content/student_scores.csv')
print("df.head")

df.head()

print("df.tail")

df.tail()

Y=df.iloc[:,1].values
print("Array of Y")
Y

X=df.iloc[:,:-1].values
print("Array of X")
X

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

print("Values of Y prediction")
Y_pred

print("Array values of Y test")
Y_test

plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Training Set Graph")
plt.show()

plt.scatter(X_test,Y_test,color="blue")
plt.plot(X_test,regressor.predict(X_test),color="black")
plt.title("Hours Vs Scores (Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
print("Test Set Graph")
plt.show()

print("Values of MSE, MAE and RMSE")
mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print("RMSE = ",rmse)

```
Program to implement the simple linear regression model for predicting the marks scored.

Developed by: MANIKANDAN V
RegisterNumber:  212224230151

## Output:


<img width="384" height="264" alt="image" src="https://github.com/user-attachments/assets/f36c1ab1-03fe-4305-afc5-25ef9fa2fc9d" />
<img width="326" height="255" alt="image" src="https://github.com/user-attachments/assets/7c8ef6d9-91a1-4194-8d8a-64cb91a95e8c" />
<img width="763" height="172" alt="image" src="https://github.com/user-attachments/assets/d5be8e03-9501-4d9d-a64a-a68d4edb5706" />
<img width="664" height="670" alt="image" src="https://github.com/user-attachments/assets/49ab0854-01e1-4dc2-9d44-7879890302d6" />
<img width="716" height="190" alt="image" src="https://github.com/user-attachments/assets/196cb3e2-dec1-4c2e-998a-c177efa0564a" />
<img width="603" height="154" alt="image" src="https://github.com/user-attachments/assets/b6dd8f39-1f0f-4000-b3f1-6192ea31fac5" />

## Training Set Graph
<img width="963" height="788" alt="image" src="https://github.com/user-attachments/assets/a28b36fa-90d3-48ec-b695-ce69fb750828" />
## Test Set Graph
<img width="958" height="769" alt="image" src="https://github.com/user-attachments/assets/5d4a21c8-6086-4684-86a2-ef61531397a2" />

<img width="721" height="326" alt="image" src="https://github.com/user-attachments/assets/aba18976-3aaf-4790-9152-5376f55732b2" />













## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
