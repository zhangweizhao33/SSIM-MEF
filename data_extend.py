import numpy as np
import random
import glob
import os
from skimage import io,img_as_float32
from itertools import product
from numpy import linalg as LA
import math
import tensorflow as tf
from keras.models import load_model
def AugmentData(input):
    """
    影像扩充包括几何扩充和颜色扩充
    :return:
    """
    def GeometricAugmentation(input):
        gInd = random.randint(0, 7)
        if gInd == 0:
            return input
        elif gInd == 1:  # 上下翻转
            return input[:, ::-1, :, :]
        elif gInd == 2:  # 左右翻转
            return input[:, :, ::-1, :]
        elif gInd == 3:  # 旋转90度（顺时针）
            return np.rollaxis(input[:, :, ::-1, :], 1, 3)
        elif gInd == 4:  # 旋转180度（顺时针）
            return input[:, ::-1, ::-1, :]
        elif gInd == 5:  # 旋转270度（顺时针）
            return np.rollaxis(input[:, ::-1, ::-1, :], 1, 3)
        elif gInd == 6:  # 45度轴对称（左上角为原点）
            return np.rollaxis(input, 1, 3)
        elif gInd == 7:  # -45度轴对称（左上角为原点）
            return np.rollaxis(input[:, ::-1, :, :], 1, 3)[:, ::-1, :, :]
    def ColorAugmentation(input):
        c = np.shape(input)[3]
        cInd = (np.arange(c)).tolist()
        random.shuffle(cInd)
        In = np.split(input, c, axis=3)
        Out = [In[i] for i in cInd]
        return np.squeeze(np.rollaxis(np.stack(Out), 0, 5))
    def ShiftAugmentation(input):
        return 1
    def RotationAugmentation(input):
        return 1
    return ColorAugmentation(GeometricAugmentation(input))
def Clamp(input, minVal=0, maxVal=1):
    output = np.copy(input)
    output[output < minVal] = minVal
    output[output > maxVal] = maxVal
    return output
gamma = 2.2
def LDRtoHDR(input, expo):
    return Clamp(input)**gamma / expo

def HDRtoLDR(input, expo):
    return Clamp((input * expo) ** (1 / gamma))

def LDRtoLDR(A, expA, expB):
    return HDRtoLDR(LDRtoHDR(A, expA), expB)

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

def GetPatches1(images,patchSize=200, stride=180):
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




# def caculate_Qmap(origin_ldrs,output_ldr):
#     global_mean_idensity=[np.mean(origin_ldrs[i]) for i in range(origin_ldrs.shape[0])]
#     img_patches=GetPatches(origin_ldrs)
#     output_patches=GetPatches(output_ldr)
#     Qlist=[]
#     for i in range(img_patches.shape[0]):
#         Qlist.append(caculate_SSIM(img_patches[i],output_patches[i],global_mean_idensity))
#     Qmap=np.reshape(np.stack(Qlist),(origin_ldrs.shape[1],origin_ldrs.shape[2]))
#     return Qmap







# def caculate_SSIM(origin_ldrs,output_ldr,global_mean_idensity):
#     xk = [origin_ldrs[i, :, :, :].ravel() for i in range(origin_ldrs.shape[0])]
#     k1=0.01
#     k2=0.02
#     '''l depends on the amount of bit of the sequence (l=255 for an 8-bit sequence)'''
#     l=255
#     c1=math.pow(k1*l,2)
#     c2=math.pow(k2*l,2)
#     sigma_g=0.2
#     sigma_l=0.2
#
#     '''caculate ideal contrast of fusion'''
#     l2_list=[LA.norm(x) for x in xk]
#     icontrast=max(l2_list)
#
#     '''caculate ideal structure of fusion'''
#     sk=[xk[i]/l2_list[i] for i in range(len(xk))]
#     p=math.tan((LA.norm(np.add(np.add(xk[0],xk[1]),xk[2]))/sum(l2_list))*math.pi/2)
#     wk=[math.pow(i,p) for i in l2_list]
#     mean_s=np.add(np.add(wk[0]/sum(wk)*sk[0],wk[1]/sum(wk)*sk[1]),wk[2]/sum(wk)*sk[2])
#     istructure=mean_s/LA.norm(mean_s)
#
#     '''caculate ideal luminance of fusion'''
#     lk=[np.mean(x) for x in xk]
#     u_list=[math.exp(-((math.pow(global_mean_idensity[i]-0.5,2)/(2*math.pow(sigma_g,2))))-((math.pow(lk[i]-0.5,2)/(2*math.pow(sigma_l,2))))) for i in range(len(lk))]
#     iluminance=(lk[0]*u_list[0]+lk[1]*u_list[1]+lk[2]*u_list[2])/sum(u_list)
#
#     '''caculate ideal fusion'''
#     x=icontrast*istructure+iluminance
#     MEF_SSIM=((2*np.mean(x)*np.mean(output_ldr)+c1)*(2*np.cov(x,output_ldr)[0][1]+c2))/((math.pow(np.mean(x),2)+math.pow(np.mean(x),2)+c1)*(np.cov(x)+np.cov(output_ldr)+c2))
#     return MEF_SSIM

def my_tf_cov(x,y):
    mean_x = tf.reduce_mean(x, axis=0, keep_dims=True)
    mean_y = tf.reduce_mean(y, axis=0, keep_dims=True)
    mx = tf.matmul(tf.transpose(mean_x), mean_y)
    vx = tf.matmul(tf.transpose(x), y)/tf.cast(tf.shape(x)[0], tf.float32)
    cov_xx = vx - mx
    return (tf.shape(x)[0]/(tf.shape(x)[0]-1))*cov_xx





def caculate_lk(input,ksize):
    lk=tf.reduce_mean(tf.nn.avg_pool3d(input,ksize=[1,1,ksize,ksize,1],strides=[1,1,1,1,1],padding="SAME"),axis=4)

    lk1=tf.expand_dims(lk,axis=4)
    lk1=tf.concat([lk1,lk1,lk1],axis=4)
    lk1=tf.reshape(lk1,shape=[lk1.shape[0],lk1.shape[1],lk1.shape[2]*lk1.shape[3]*lk1.shape[4]])
    return lk1

def caculate_Qmap2(input_ldrs,ksize=7,im_height=1000,im_width=1500,**kwargs):
    k1 = kwargs["k1"]
    k2 = kwargs["k2"]
    '''l depends on the amount of bit of the sequence ( l=255 for an 8-bit sequence)'''
    l = kwargs["l"]
    c1 = math.pow(k1 * l, 2)
    c2 = math.pow(k2 * l, 2)
    sigma_g = kwargs["sigma_g"]
    sigma_l = kwargs["sigma_g"]
    '''muc and lc are mid-intensity values(muc=lc=0.5 if source image normalized to [0,1])'''
    muc=kwargs["muc"]
    lc=kwargs["lc"]
    whole_image_tensor=tf.constant(input_ldrs)
    muk=tf.reduce_mean(whole_image_tensor,axis=[1,2,3])


    #image_patches=tf.constant(GetPatches(input_ldrs))
    image_patches=tf.expand_dims(input_ldrs,axis=0)
    #muk = tf.reduce_mean(whole_image_tensor, axis=[1, 2, 3])
    lk ,lk1= caculate_lk(image_patches,ksize=ksize)
    image_patches = tf.reshape(image_patches,shape=[image_patches.shape[0],image_patches.shape[1],image_patches.shape[2]*image_patches.shape[3]*image_patches.shape[4]])


    '''input_tensor[n,3,m] lk[n,3,m] ck_tensor[n,3] ic[n]'''
    mean_removed=tf.subtract(image_patches,lk)
    ck_tensor=tf.norm(mean_removed,ord="euclidean",axis=2)
    ic=tf.reduce_max(ck_tensor,axis=1)

    '''lk[n,3,m] uk[n,3,m] il[n,m],uk_sum[n,m]'''
    uk=tf.exp(tf.add(tf.negative(tf.divide(tf.pow(tf.subtract(lk,lc),2),2*tf.pow(sigma_l,2))),tf.reshape(tf.negative(tf.divide(tf.pow(tf.subtract(muk,muc),2),2*tf.pow(sigma_g,2))),shape=[1,3,1])))
    uk_sum=tf.reduce_sum(uk,axis=1)
    il=tf.divide(tf.reduce_sum(tf.multiply(lk,uk),axis=1),uk_sum)

    '''rk [n] sk[n,3,cN2]'''
    sk=tf.divide(mean_removed,tf.expand_dims(ck_tensor,axis=2))
    rk=tf.divide(tf.norm(tf.reduce_sum(mean_removed,axis=1),axis=1),tf.reduce_sum(ck_tensor,axis=1))

    '''p [n] WK[n,3] x[n,CN2] istructure[n,m]'''
    p=tf.tan(tf.multiply(rk,math.pi/2))
    p=tf.constant(4,shape=[1],dtype=tf.float32)
    wk=tf.pow(ck_tensor,tf.expand_dims(p,axis=1))
    wk_sum=tf.reduce_sum(wk,axis=1)
    s=tf.divide(tf.reduce_sum(tf.multiply(sk,tf.expand_dims(wk,axis=2)),axis=1),tf.expand_dims(wk_sum,axis=1))
    s_norm=tf.norm(s,ord="euclidean",axis=1)
    istructure=tf.divide(s,tf.expand_dims(s_norm,axis=1))
    x=tf.add(tf.multiply(istructure,tf.expand_dims(ic,axis=1)),il)
    # cov_xy = tf.diag_part(my_tf_cov(x, output_patches))
    # mux = tf.reduce_mean(x, 1)
    # muy = tf.reduce_mean(output_patches, 1)
    # cov_xx = tf.diag_part(my_tf_cov(x,x))
    # cov_yy = tf.diag_part(my_tf_cov(output_patches,output_patches))
    # S_map = (2*mux*muy+c1)*(2*cov_xy+c2)/((tf.pow(mux,2)+tf.pow(muy,2)+c1)*(tf.pow(cov_xx,2)+tf.pow(cov_yy,2)+c2))
    with tf.Session() as sess:
        output=sess.run(x)
    return output

def AggeratePatches(patches, batchsize=1,patchSize=200, stride=180, height=1000, width=1500):
    patches = np.reshape(patches,[patches.shape[0], patchSize, patchSize, 3])
    whole_image = np.empty([height, width, 3])
    count_image = np.empty([height, width])
    count_unit = np.ones([patchSize, patchSize])
    patch_index = 0
    for i, j in product(range(0, height - stride, stride), range(0, width - stride, stride)):
        flag_i = i + patchSize > height
        flag_j = j + patchSize > width
        if flag_i and flag_j:
            whole_image[- patchSize:, - patchSize:, :] += patches[patch_index]
            count_image[- patchSize:, - patchSize:] += count_unit
        elif flag_i:
            whole_image[ - patchSize:, j: j + patchSize:, :] += patches[patch_index]
            count_image[ - patchSize:, j: j + patchSize:] += count_unit
        elif flag_j:
            whole_image[ i: i + patchSize, - patchSize:, :] += patches[patch_index]
            count_image[ i: i + patchSize, - patchSize:] += count_unit
        else:
            whole_image[i: i + patchSize, j: j + patchSize, :] += patches[patch_index]
            count_image[i: i + patchSize, j: j + patchSize] += count_unit
        patch_index += 1
    whole_image[whole_image < 0] = 0
    final_result = np.divide(whole_image, np.expand_dims(count_image,axis=2))
    final_result=final_result*255
    final_result=final_result.astype("uint8")
    io.imsave("D:\whole_image_test.jpg",final_result)
    return 0

def save_result(input_patches,patch_size=200):
    image=np.reshape(input_patches,[input_patches.shape[0],patch_size,patch_size,3])
    image[image<0]=0
    image=image*255
    image=image.astype("uint8")
    for i in range(image.shape[0]):
      io.imsave("D:\im_result\\"+str(i+1)+".jpg",image[i])
    return 0




if __name__=="__main__":
    parameters={"k1":0.01,"k2":0.02,"l":255,"sigma_g":0.2,"sigma_l":0.2,"muc":0.5,"lc":0.5}
    patch_size=200
    model=load_model("./my_model.h5")
    le=io.imread("D:\Train_OnlyGT\\16-10-10-a-01\Reference\\262A1052.tif")
    ne = io.imread("D:\Train_OnlyGT\\16-10-10-a-01\Reference\\262A1053.tif")
    oe=io.imread("D:\Train_OnlyGT\\16-10-10-a-01\Reference\\262A1054.tif")
    le,ne,oe=img_as_float32(le),img_as_float32(ne),img_as_float32(oe)
    ldrs=np.stack((le,ne,oe))
    patches=GetPatches1(ldrs)
    output_patches=model.predict(patches)
    output_image=AggeratePatches(output_patches)
    # output=caculate_Qmap2(ldrs,**parameters)
    # print(np.min(output_image))
    # print(np.sum(output_image<0))
    # output_image[output_image<0]=0
    # output_image=output_image*255
    # output_image=output_image.astype("uint8")
    # io.imshow(output_image)
    # print(np.min(output_image))
    # result=np.reshape(output_image,size=[1000,1500,3])
    # io.imsave("./whole_image_7.jpg", result)
    #save_result(output,patch_size)
    # print(result[38])
    # print(result[39])


























