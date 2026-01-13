import numpy as np


def accuracy(y_true, y_pred):
    return (np.sum(y_true == y_pred) / y_true.shape[0]) if y_true.shape[0]!=0 else 0 
    
