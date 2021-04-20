from template import BaseModel
from keras import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D
from keras.utils import to_categorical
import numpy as np
from keras.optimizers import Adam

class AlexNet(BaseModel):
    def __init__(self, shape, activation, num_classes):
        self.shape = shape
        self.activation = activation
        self.num_classes = num_classes

    def build(self):
        self.model = Sequential([
            Conv2D(96, activation=self.activation, kernel_size=11, strides=4, input_shape=self.shape),
            MaxPooling2D(pool_size=3, strides=2),
            Conv2D(256, activation=self.activation, kernel_size=5, padding='same'),
            MaxPooling2D(pool_size=3, strides=2),
            Conv2D(384, activation=self.activation, kernel_size=3, padding='same'),
            Conv2D(384, activation=self.activation, kernel_size=3, padding='same'),
            Conv2D(256, activation=self.activation, kernel_size=3, padding='same'),
            MaxPooling2D(pool_size=3, strides=2),
            Flatten(),
            Dense(4096, activation=self.activation, use_bias=True),
            Dropout(0.2),
            Dense(4096, activation=self.activation, use_bias=True),
            Dropout(0.2),
            Dense(self.num_classes, activation='softmax', use_bias=True)
        ])

        return self.model

    def train(self, x_train, y_train, epoch=50):
        self.model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['acc'])
        callbacks=[tf.keras.callbacks.ReduceLROnPlateau( factor = 0.1, patience = 3, min_lr = 0.00001, verbose = 1 )]
        return self.model.fit(x_train, y_train, epochs=epoch, callbacks = callbacks, validation_split = 0.3)


def process_images(image, label):
    # Normalize images to have a mean of 0 and standard deviation of 1
    image = tf.image.per_image_standardization(image)
    # Resize images from 32x32 to 277x277
    image = tf.image.resize(image, (227,227))
    return image, label



from matplotlib import pyplot
from keras.datasets import fashion_mnist
# load dataset
(trainX, trainy), (testX, testy) = fashion_mnist.load_data()

num_classes = 10

# normalize data
trainX, testX = trainX / 255., testX / 255.

trainX = np.array(trainX).reshape(trainX.shape[0], trainX.shape[1], trainX.shape[2], 1)
testX = np.array(testX).reshape(testX.shape[0], testX.shape[1], testX.shape[2], 1)

resized_trainX = tf.image.resize_with_pad(
    trainX[:10000], 227, 227, method=ResizeMethod.BILINEAR,
    antialias=False
)

resized_testX = tf.image.resize_with_pad(
    testX[:10000], 227, 227, method=ResizeMethod.BILINEAR,
    antialias=False
)

# one-hot
trainy = tf.keras.utils.to_categorical(trainy[:10000])
testy = tf.keras.utils.to_categorical(testy[:10000])

# summarize loaded dataset
print('Train: X=%s, y=%s' % (resized_trainX.shape, trainy.shape))
print('Test: X=%s, y=%s' % (resized_testX.shape, testy.shape))
# # show the figure
# pyplot.show()
model = AlexNet((227, 227, 1), 'relu', 10)
model.build()
model.summary()
hist = model.train(resized_trainX, trainy)
model.save('./')

test = np.expand_dims(resized_testX[6], axis=0)
y_predict = model.predict(test)
pyplot.imshow(test[0,:,:,0], cmap=pyplot.get_cmap('gray'))
y = to_categorical(np.argmax(y_predict, axis = 1))
print(y)
