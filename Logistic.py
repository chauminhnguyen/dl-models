import numpy as np
import tensorflow as tf
from keras.layers import Input, Dense
from keras import Model
import matplotlib.pyplot as plt
from keras.utils import to_categorical
from template import BasModel

class LogisticReg(BaseModel):
    def __init__(self, ndim):
        self.ndim = ndim
    def build(self):
        input = Input(shape=(self.ndim,))
        hidden1 = Dense(10, use_bias=True)(input)
        hidden2 = Dense(10, use_bias=True)(hidden1)
        hidden3 = Dense(10, use_bias=True)(hidden2)
        output = Dense(1, activation='sigmoid', use_bias=True)(hidden3)
        self.model = Model(inputs=input, outputs=output)

    def train(self, x_train, y_train):
        # train bang thuat toan gi
        self.model.compile(optimizer='sgd', loss='binary_crossentropy', metrics=['accuracy'])
        return self.model.fit(x_train, y_train, epochs=1000, validation_split = 0.2)

# Chuan bi du lieu

nsample = 50
x_train = np.random.rand(nsample, 5) * 10
y_train = np.asarray(np.sum(x_train, axis=1) / x_train.shape[1] >= 5)


# Build Model
model = LogisticReg(x_train.shape[1])
model.build()
hist = model.train(x_train, y_train)
# print(np.sum(x_train, axis=1) / x_train.shape[1] >= 5)

# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss', 'validation'])
# plt.show()

x_test = np.random.rand(5, 5) * 10
# print(x_test)
print(np.sum(x_test, axis=1) / x_test.shape[1])
y_predict = model.predict(x_test)
print(y_predict >= 0.5)
