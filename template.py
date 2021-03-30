# Khai bao thu vien
import numpy as np
import tensorflow as tf

# Chuan bi du lieu

# Tao mo hinh bang ki thuat OOP
class BaseModel:
    def __init__(self):
        pass
    def build(self):
        pass
    def train(self):
        pass
    def predict(self, x_test):
        return self.model.predict(x_test)
    def summary(self):
        self.model.summary()
    def save(self, path):
        self.model.save(path)
    def load(self):
        self.model.load(path)

# Su dung mo hinh de huan luyen va test
# model = MyModel()
# model.build(10, 0.0001)
# model.train(x_train, y_train)
# y_predict = model.predict(x_test)
# acc = accuracy(y_predict, y_test)
# model.save()
