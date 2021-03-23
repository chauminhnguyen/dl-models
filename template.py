# Khai bao thu vien
import numpy as np
import tensorflow as tf

# Chuan bi du lieu

# Tao mo hinh bang ki thuat OOP
class MyModel:
    def __init__(self):
        pass
    def build(self):
        pass
    def train(self):
        pass
    def predict(self):
        pass
    def summary(self):
        pass
    def save(self):
        pass
    def load(self):
        pass

# Su dung mo hinh de huan luyen va test
model = MyModel()
model.build(10, 0.0001)
model.train(x_train, y_train)
y_predict = model.predict(x_test)
acc = accuracy(y_predict, y_test)
model.save()
