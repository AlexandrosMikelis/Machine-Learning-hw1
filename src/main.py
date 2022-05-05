from modules.Dense import Dense
from modules.ReLu import ReLU
from modules.Sigmoid import Sigmoid
from modules.Sequential import Sequential
import numpy as np
from matplotlib import pyplot as plt

np.random.seed(1)


if __name__ == "__main__":

    Train_Samples = 1000000
    Test_Samples = 2000
    epochs = 1000

    val_len = 2000

    x0 = np.random.normal(loc=0.0, scale=1.0, size=(Train_Samples, 2))
    x0_test = np.random.normal(loc=0.0, scale=1.0, size=(Test_Samples,2))

    x1 = np.random.normal(loc=0.0, scale=1.0, size=(Train_Samples,2)) + np.random.choice([-1,1],size=(Train_Samples,2))
    x1_test = np.random.normal(loc=0.0, scale=1.0, size=(Test_Samples,2)) + np.random.choice([-1,1],size=(Test_Samples,2))

    X_train = np.vstack((x0_test,x1_test))
    y_train = np.vstack((np.zeros((Test_Samples,1)) , np.ones((Test_Samples,1))))

    X_val = np.vstack((x0[:val_len], x1[:val_len]))
    y_val = np.vstack((np.zeros((val_len, 1)), np.ones((val_len, 1))))

    X_test = np.vstack((x0,x1))
    y_test = np.vstack((np.zeros((Train_Samples, 1)), np.ones((Train_Samples, 1))))

    model = Sequential()
    model.add(Dense(X_train.shape[1],20))
    model.add(ReLU())
    model.add(Dense(20,1))
    model.add(Sigmoid())

    val,train = model.fit(X_train,y_train,X_val,y_val,epochs=epochs)

    plt.plot(train,label='train accuracy')
    plt.plot(val,label='val accuracy')
    plt.legend(loc='best')
    plt.grid()
    plt.show()

