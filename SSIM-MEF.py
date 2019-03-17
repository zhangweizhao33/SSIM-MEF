import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation,Input,Conv2D,BatchNormalization
from keras.optimizers import SGD
input= Input(shape=(200,200))
X=Conv2D(filters=100,kernel_size=(7,7),strides=(1,1),padding="valid")(input)
X=Conv2D(filters=100,kernel_size=(5,5),strides=(1,1),padding="valid")(X)
X=Conv2D(filters=100,kernel_size=(3,3),strides=(1,1),padding="valid")(X)
output=Conv2D(filters=100,kernel_size=(1,1),strides=(1,1),padding="valid")(X)