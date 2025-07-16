import numpy as np
class LinearRegression():
     def __init__ (self, learning_rate=0.1, iterations=500):
          self.learning_rate = learning_rate
          self.iterations = iterations
          self.W = None
          self.b = None

     def fit(self, X, y):
          n, m = X.shape
          self.W = np.random.rand(m)
          self.b = 0.0
          for i in range(self.iterations):
               y_hat = self.predict(X)
               loss = self.compute_loss(y_hat, y)
               dW, dB = self.compute_grad(X, y_hat, y)
               self.W -= self.learning_rate*dW
               self.b -= self.learning_rate*dB

     def predict(self, X):
          return np.dot(X, self.W) + self.b

     def compute_loss(self, y_hat, y):
          return np.mean((y_hat-y)**2)

     def compute_grad(self, X, y_hat, y):
          n = X.shape[0]
          dW = 2.0/n * np.dot(X.T, y_hat-y)
          dB = 2.0/n * np.sum(y_hat-y, axis=0)
          return dW, dB
