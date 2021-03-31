from template import BaseModel
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from keras.utils import to_categorical
import numpy as np
from keras.optimizers import Adam


class VGG16(BaseModel):
    def __init__(self, shape, activation, num_classes):
        self.shape = shape
        self.activation = activation
        self.num_classes = num_classes
    def build(self):
        self.model = Sequential([
            Conv2D(64, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv1-1', input_shape=self.shape),
            Conv2D(64, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv1-2'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(128, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv2-1'),
            Conv2D(128, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv2-2'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(256, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv3-1'),
            Conv2D(256, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv3-2'),
            Conv2D(256, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv3-3'),
            MaxPooling2D(pool_size=(2,2)),
            Conv2D(512, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv4-1'),
            Conv2D(512, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv4-2'),
            Conv2D(512, activation=self.activation, kernel_size=(3, 3), padding='same', name='Conv4-3'),
            MaxPooling2D(pool_size=(2, 2)),
            Flatten(),
            Dense(4096, activation=self.activation),
            Dropout(0.2),
            Dense(4096, activation=self.activation),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax')
        ])
        return self.model

    def train(self, x_train, y_train, epoch=10,):
        self.model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['acc'])
        # create callbacks
        callbacks = [
            tf.keras.callbacks.ReduceLROnPlateau( factor = 0.1, patience = 3, min_lr = 0.00001, verbose = 1 )
        ]
        return self.model.fit(x_train, y_train, epochs=epoch, callbacks = callbacks, validation_split = 0.3)



from matplotlib import pyplot
from keras.datasets import fashion_mnist
# load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

num_classes = 10
img_width = 28
img_heigh = 28
img_ch = 1
input_shape = (img_width, img_heigh, img_ch)

# normalize data
trainX, testX = trainX / 255., testX / 255.

# reshape input
trainX = trainX.reshape(trainX.shape[0], *input_shape)
testX = testX.reshape(testX.shape[0], *input_shape)

# one-hot
trainy = tf.keras.utils.to_categorical(trainy)
testy = tf.keras.utils.to_categorical(testy)

# summarize loaded dataset
print('Train: X=%s, y=%s' % (trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (testX.shape, testy.shape))
# show the figure
pyplot.show()
model = VGG16((28, 28, 1), 'relu', 10)
model.build()
model.summary()
hist = model.train(trainX, trainy)
model.save('./')

test = np.expand_dims(testX[6], axis=0)
y_predict = model.predict(test)
pyplot.imshow(test[0,:,:,0], cmap=pyplot.get_cmap('gray'))
y = to_categorical(np.argmax(y_predict, axis = 1))
print(y)
