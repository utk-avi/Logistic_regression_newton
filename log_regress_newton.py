import matplotlib.pyplot as plt
import numpy as np
import utility as util
from linear_mode1 import LinearModel

#loading the dataset
ds1_train_path = r'C:\Users\Aviral\Documents\Logistic_regression_newton\Data\ds1_train.csv'
ds1_valid_path = r'C:\Users\Aviral\Documents\Logistic_regression_newton\Data\ds1_valid.csv'

x_train, y_train = util.load_dataset('data/ds1_train.csv', add_intercept=True)
x_valid, y_valid = util.load_dataset('data/ds1_valid.csv', add_intercept=True)

plt.xlabel('x1')
plt.ylabel('x2')
plt.plot(x_train[y_train == 1, -2], x_train[y_train == 1, -1], 'bx', linewidth=2)
plt.plot(x_train[y_train == 0, -2], x_train[y_train == 0, -1], 'go', linewidth=2)

class LogisticRegression(LinearModel):

    def fit(self, x, y):

        def h(theta, x):
            return 1 / (1 + np.exp(-np.dot(x, theta)))
        def gradient(theta, x, y):
            m, _ = x.shape
            return -1 / m * np.dot(x.T, (y - h(theta, x)))
        def hessian(theta, x):
            m, _ = x.shape
            h_theta_x = np.reshape(h(theta, x), (-1, 1))
            return 1 / m * np.dot(x.T, h_theta_x * (1 - h_theta_x) * x)
        def next_theta(theta, x, y):
            return theta - np.dot(np.linalg.inv(hessian(theta, x)), gradient(theta, x, y))
        m,n = x.shape

        if self.theta is None:
            self.theta = np.zeros(n)

        
        old_theta = self.theta
        new_theta = next_theta(self.theta, x, y)
        while np.linalg.norm(new_theta - old_theta, 1) >= self.eps:
            old_theta = new_theta
            new_theta = next_theta(old_theta, x, y)

        self.theta = new_theta

    def predict(self, x):
         return x @ self.theta >= 0
# Training the model
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)

util.plot(x_train, y_train, theta=log_reg.theta)
plt.show()
print("Theta is: ", log_reg.theta)
print("The accuracy on training set is: ", np.mean(log_reg.predict(x_train) == y_train))
