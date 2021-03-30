import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras import Model
import matplotlib.pyplot as plt
from template import BaseModel

class LinearReg(BaseModel):
    def __init__(self, ndim):
        self.ndim = ndim
    def build(self):
        input = Input(shape=(self.ndim,))

        # hidden1 = Dense(5, use_bias=True)(input)
        # hidden2 = Dense(10, use_bias=True)(hidden1)
        # hidden3 = Dense(10, use_bias=True)(hidden2)
        # hidden4 = Dense(10, use_bias=True)(hidden3)
        # output = Dense(1, use_bias=True)(hidden4)

        hidden1 = Dense(10, use_bias=True)(input)
        output = Dense(1, use_bias=True)(hidden1)

        self.model = Model(inputs=input, outputs=output)

    def train(self, x_train, y_train):
        # train bang thuat toan gi
        self.model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
        return self.model.fit(x_train, y_train, epochs=1000, validation_split = 0.2)

# Chuan bi du lieu
nsample = 10
x_train = np.arange(nsample)
y_train = x_train * 3 + 5 + np.random.rand(1, nsample)
x_train = x_train.reshape(nsample, 1)
y_train = y_train.reshape(nsample, 1)
print(x_train.shape)
# print(y_train.shape)
model = LinearReg(1)
model.build()
model.summary()
hist = model.train(x_train, y_train)

plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['loss', 'validation'])
plt.show()

x_test = [20]
# y_test = 20 * 3 + 5
y_predict = model.predict(x_test)
print(y_predict)
# acc = accuracy(y_predict, y_test)
# model.save()
