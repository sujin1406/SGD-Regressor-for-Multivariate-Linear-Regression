# SGD-Regressor-for-Multivariate-Linear-Regression

## AIM:
To write a program to predict the price of the house and number of occupants in the house with SGD regressor.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import required libraries and prepare the multivariate input data and target values.

2.Initialize the SGD Regressor with appropriate learning rate and iterations.

3.Train the model using the given dataset and predict the output values.

4.Compare actual and predicted values using graphical visualization.

## Program:
```
Program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor.
Developed by: SUJIN M L
RegisterNumber:  212225040435
```

```
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import SGDRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()

X = data.data[:, :3]

Y = np.column_stack((data.target, data.data[:, 6]))

print("X shape:", X.shape)
print("Y shape:", Y.shape)
print("Example X (first row):", X[0])
print("Example Y (first row):", Y[0])
```

<img width="712" height="135" alt="image" src="https://github.com/user-attachments/assets/c38ce0af-2aab-4f57-a8c9-617ad5e23336" />

```
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

print("Train shapes:", X_train.shape, Y_train.shape)
print("Test shapes: ", X_test.shape, Y_test.shape)
```

<img width="591" height="81" alt="image" src="https://github.com/user-attachments/assets/db041907-60e5-4c17-ab37-d5d75c59999c" />

```
scaler_X = StandardScaler()
scaler_Y = StandardScaler()


X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)

Y_train_scaled = scaler_Y.fit_transform(Y_train)
Y_test_scaled = scaler_Y.transform(Y_test)

print("Scaled X_train mean (approx):", X_train_scaled.mean(axis=0))
print("Scaled Y_train mean (approx):", Y_train_scaled.mean(axis=0))

```

<img width="836" height="92" alt="image" src="https://github.com/user-attachments/assets/f8f11f98-bdd4-45be-94a7-569a8dcefccd" />

```
sgd = SGDRegressor(max_iter=1000, tol=1e-3, random_state=42)
multi_output_sgd = MultiOutputRegressor(sgd)

multi_output_sgd.fit(X_train_scaled, Y_train_scaled)
```

<img width="580" height="205" alt="image" src="https://github.com/user-attachments/assets/4df797ca-7040-479e-b1f5-7da6f5c92b21" />

```
Y_pred_scaled = multi_output_sgd.predict(X_test_scaled)
Y_pred = scaler_Y.inverse_transform(Y_pred_scaled)
Y_test_orig = scaler_Y.inverse_transform(Y_test_scaled)

print("First 5 predictions (original units):")
print(Y_pred[:5])

```

<img width="425" height="170" alt="image" src="https://github.com/user-attachments/assets/f0d0a7c6-2090-4746-a49d-c0d7a86d0d4e" />

```
mse = mean_squared_error(Y_test_orig, Y_pred)
print("Mean Squared Error (multi-output):", mse)

mse_per_output = np.mean((Y_test_orig - Y_pred) ** 2, axis=0)
print("MSE per output:", mse_per_output)

```

<img width="642" height="62" alt="image" src="https://github.com/user-attachments/assets/a308c703-ee30-40f6-907e-6703807f10fe" />


```
for i in range(5):
    print(f"Example {i+1}")
    print("Inputs (raw):", X_test[i])
    print("True outputs:", Y_test_orig[i])
    print("Predicted   :", Y_pred[i])
    print("-" * 40)

```

<img width="582" height="610" alt="image" src="https://github.com/user-attachments/assets/11070676-51f5-46f3-8c69-b9001216d412" />

```
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = fetch_california_housing()
X, y = data.data[:, :3], data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)


sgd = SGDRegressor(max_iter=1000, tol=1e-3, eta0=0.01, learning_rate='constant', random_state=42)
sgd.fit(X_train, y_train)
sgd_pred = sgd.predict(X_test)

print("LinearRegression MSE:", mean_squared_error(y_test, lr_pred))
print("SGDRegressor MSE:", mean_squared_error(y_test, sgd_pred))

```

<img width="642" height="72" alt="image" src="https://github.com/user-attachments/assets/4a79a3ea-9c17-4021-af06-9f2cb4f96374" />


## Result:
Thus the program to implement the multivariate linear regression model for predicting the price of the house and number of occupants in the house with SGD regressor is written and verified using python programming.
