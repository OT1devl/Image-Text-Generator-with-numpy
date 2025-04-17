import numpy as np

def ReLU(x, derv=False):
    if derv: return np.where(x>0, 1, 0)
    return np.maximum(x, 0)

def LeakyReLU(x, alpha=0.2, derv=False):
    if derv: return np.where(x>0, 1, alpha)
    return np.where(x>0, x, x*alpha)

def tanh(x, derv=False):
    if derv: return 1 - x**2 # x is already calculated as tanh
    return np.tanh(x)

def sigmoid(x, derv=False):
    if derv: return x * (1 - x) # x is already calculated as sigmoid
    return 1 / (1 + np.exp(-x))

def BinaryCrossEntropyLoss(y_true, y_pred, epsilon=1e-8, derv=False):
    if derv: return -y_true/(y_pred+epsilon)+(1-y_true)/(1-y_pred+epsilon)
    return np.mean(-y_true*np.log(y_pred+epsilon)-(1-y_true)*np.log(1-y_pred+epsilon))
