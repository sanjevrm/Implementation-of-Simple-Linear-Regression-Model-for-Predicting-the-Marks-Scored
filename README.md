# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1. Import the standard Libraries.
2.Set variables for assigning dataset values.
3.Import linear regression from sklearn.
4.Assign the points for representing in the graph.
5.Predict the regression for marks by using the representation of the graph. Compare the graphs and hence we obtained the linear regression for the given datas.
```
## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by:JAGADEESH J
RegisterNumber:212223110015


from google.colab import drive
drive.mount('/content/gdrive')
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error,mean_squared_error
import matplotlib.pyplot as plt
*/
```
## Dataset:
```
a=pd.read_csv('/content/gdrive/MyDrive/student_scores.csv')
a
```
## Output:
![Screenshot 2024-09-08 184741](https://github.com/user-attachments/assets/dd013944-83a9-44b1-9791-9145bfb0ea2f)
## Head and Tail:
```
print(a.head())
print(a.tail())
```
## Output:
![Screenshot 2024-09-08 185113](https://github.com/user-attachments/assets/97bb4af2-91c5-42b2-8b23-c5f5bce853d8)
## Information of Dataset:
```
a.info()
```
## Output:
![image](https://github.com/user-attachments/assets/ac8e0bae-09ac-49b1-ae55-5b917fd9c1fa)
## x and y value:
```
x=a.iloc[:,:-1].values
print(x)
y=a.iloc[:,-1].values
print(y)
```
## Output:
![image](https://github.com/user-attachments/assets/2feb1714-6391-4258-8f0f-1265787b8bf9)
## Spliting for Training and Testing :
```
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
print(x_train.shape)
print(x_test.shape)
```
## Output:
![Screenshot 2024-09-08 192835](https://github.com/user-attachments/assets/5149b1e1-73fe-4612-86a2-9f9afc251767)
## Program:
```
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)
```
## Output:
![image](https://github.com/user-attachments/assets/0ce9219a-ac29-4851-85ba-aaeb10d4bcef)
## Training and Testing the Models:
```
y_pred = reg.predict(x_test)
print(y_pred)
print(y_test)
```
## Output:
![image](https://github.com/user-attachments/assets/82a27525-1fbb-4d4b-81f7-6b88fa2eebd7)
## MSE,MAE and RMSE:
```
mse=mean_squared_error(y_test,y_pred)
print('MSE =',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE =',mae)
rmse = np.sqrt(mse)
print('RMSE =',rmse)
```
## Output:
![image](https://github.com/user-attachments/assets/957602be-35ef-4aa6-ad73-0432c7eaa7fc)
## Graph Plotting:
```
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.title('Training set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
plt.scatter(x_test,y_test,color='green')
plt.plot(x_train,reg.predict(x_train),color='black')
plt.title('Test set(H vs S)')
plt.xlabel('Hours')
plt.ylabel('Scores')
plt.show()
```
## Output:
![image](https://github.com/user-attachments/assets/2806ae90-c08f-46f1-9a0e-6142cc05910b)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
