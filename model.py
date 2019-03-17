import keras
from keras.layers import Input, Dense,Conv2D,MaxPooling2D
from keras.models import Model
from keras.optimizers import SGD
from skimage import io,img_as_float32
import os
import glob
import numpy as np
from keras.utils.vis_utils import plot_model
from keras.optimizers import SGD
from keras.models import model_from_json
from keras.models import load_model
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from itertools import product
import json



def define_model():
        inputs = Input(shape=(200,200,9))
        X = Conv2D(filters=100, kernel_size=(7, 7), strides=(1, 1), padding="same")(inputs)
        X = Conv2D(filters=100, kernel_size=(5, 5), strides=(1, 1), padding="same")(X)
        X = Conv2D(filters=100, kernel_size=(3, 3), strides=(1, 1), padding="same")(X)
        X = Conv2D(filters=100, kernel_size=(1, 1), strides=(1, 1), padding="same")(X)
        outputs = Conv2D(filters=3, kernel_size=(1, 1), strides=(1, 1), padding="same")(X)
        model = Model(inputs=inputs, outputs=outputs)
        sgd = SGD(lr=0.001, decay=1e-10, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd,
                      loss='mean_squared_error',
                      metrics=['accuracy'])
        return model

def GetPatches(images,hdr, patchSize=200, stride=180):
    """
    把图片切分为patches
    :param images: 待切分图像
    :param hdr: 对应hdr
    :param patchSize: 切片大小
    :param stride: 切割步长，覆盖率为patchSize-stride
    :return: 切片
    """
    h, w, c = np.shape(images)[1],np.shape(images)[2],np.shape(images)[3]
    #assert(np.shape(images)[1::] == np.shape(hdr))
    input = []
    GT = []
    for i, j in product(range(0, h - stride, stride), range(0, w - stride, stride)):
        flag_i = i + patchSize > h
        flag_j = j + patchSize > w
        if flag_i and flag_j:
            input.append(images[:, - patchSize:, - patchSize:, :])
            GT.append(hdr[- patchSize:, - patchSize:, :])
        elif flag_i:
            input.append(images[:, - patchSize:, j: j + patchSize:, :])
            GT.append(hdr[- patchSize:, j: j + patchSize, :])
        elif flag_j:
            input.append(images[:, i: i + patchSize, - patchSize:, :])
            GT.append(hdr[i: i + patchSize, - patchSize:, :])
        else:
            GT.append(hdr[i: i + patchSize, j: j + patchSize, :])
            input.append(images[:, i: i + patchSize, j: j + patchSize, :])
    return np.squeeze(np.concatenate(np.split(np.stack(input),3,1),-1)),np.stack(GT)

def GetPatches2(images, patchSize=200, stride=180):
    """
    把图片切分为patches
    :param images: 待切分图像
    :param hdr: 对应hdr
    :param patchSize: 切片大小
    :param stride: 切割步长，覆盖率为patchSize-stride
    :return: 切片
    """
    h, w, c = np.shape(images)[1],np.shape(images)[2],np.shape(images)[3]
    #assert(np.shape(images)[1::] == np.shape(hdr))
    input = []
    GT = []
    for i, j in product(range(0, h - stride, stride), range(0, w - stride, stride)):
        flag_i = i + patchSize > h
        flag_j = j + patchSize > w
        if flag_i and flag_j:
            input.append(images[:, - patchSize:, - patchSize:, :])
            #GT.append(hdr[- patchSize:, - patchSize:, :])
        elif flag_i:
            input.append(images[:, - patchSize:, j: j + patchSize:, :])
            #GT.append(hdr[- patchSize:, j: j + patchSize, :])
        elif flag_j:
            input.append(images[:, i: i + patchSize, - patchSize:, :])
            #GT.append(hdr[i: i + patchSize, - patchSize:, :])
        else:
            #GT.append(hdr[i: i + patchSize, j: j + patchSize, :])
            input.append(images[:, i: i + patchSize, j: j + patchSize, :])
    return np.squeeze(np.concatenate(np.split(np.stack(input),3,1),-1))

def load_data(data_path):
        data_list = []
        gd_list=[]
        dirs = os.listdir(data_path)
        for dir in dirs:
            abs_dir = os.path.join(data_path, dir)
            if (os.path.isdir(abs_dir)):
                reference_path = os.path.join(abs_dir, "reference")
                if (os.path.exists(reference_path)):
                    img_path = glob.glob(os.path.join(reference_path, "*.tif"))
                    ue_image = img_as_float32(io.imread(img_path[0]))
                    ne_image = img_as_float32(io.imread(img_path[1]))
                    oe_image = img_as_float32(io.imread(img_path[2]))
                    fused_image = img_as_float32(io.imread(img_path[3]))
                    images=np.stack((ue_image,ne_image,oe_image))
                    input_patches, GT_patches = GetPatches(images, fused_image)
                    data_list.append(input_patches)
                    gd_list.append(GT_patches)
        return np.concatenate(data_list),np.concatenate(gd_list)

def train_model(result_path):
    model=define_model()
    model.summary()
    with open("./record.json", "w") as dump_f:
        json.dump(model.to_json(), dump_f)
    # plot_model(model, show_shapes=True,
    #            to_file=os.path.join(result_path, 'model.png'))
    X, Y = load_data(result_path)
    # split train and test data
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=2)

    # input data to model and train
    history = model.fit(X_train, Y_train, batch_size=2, epochs=10,
                        validation_data=(X_test, Y_test), verbose=1, shuffle=True)

    # evaluate the model
    loss, acc = model.evaluate(X_test, Y_test, verbose=0)
    print('Test loss:', loss)
    print('Test accuracy:', acc)
    model.save_weights("./my_model.h5")




if __name__=="__main__":
    # X, Y = load_data("D:\Train_OnlyGT")
    # train_model("D:\Train_OnlyGT")
    model = load_model("D:\Train_OnlyGT\my_model.h5")
    path = "D:\Train_OnlyGT\\16-10-10-a-01\\Reference"
    img_path = glob.glob(os.path.join(path, "*.tif"))
    ue_image = img_as_float32(io.imread(img_path[0]))
    ne_image = img_as_float32(io.imread(img_path[1]))
    oe_image = img_as_float32(io.imread(img_path[2]))
    ldrs = np.stack((ue_image, ne_image, oe_image))
    patches=GetPatches2(ldrs)
    x=model.predict(ldrs)



