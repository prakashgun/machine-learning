import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model, datasets

house_price = [245, 312, 279, 308, 199, 219, 405, 324, 319, 255]
size = [1400, 1600, 1700, 1875, 1100, 1550, 2350, 2450, 1425, 1700]

size2 = np.array(size).reshape(-1, 1)

regr = linear_model.LinearRegression()
regr.fit(size2, house_price)

print("Coefficients: \n", regr.coef_)
print("Intercept: \n", regr.intercept_)


# Formula obtained by trained model
def graph(formula, x_range):
    x = np.array(x_range)
    y = eval(formula)
    plt.plot(x, y)


# size_new = 1400
# price = (size_new * regr.coef_) + regr.intercept_
# print(price)
# print(regr.predict([[size_new]]))

graph("(regr.coef_ * x) + regr.intercept_", range(1000, 2700))
plt.scatter(size, house_price, color='green')
plt.ylabel('House Price')
plt.xlabel('Size of the house')
plt.savefig("graph.png")
