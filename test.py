import os
import numpy as np
import glob
import math
from skimage import io,color
from keras.models import load_model
from skimage import io,img_as_float32
import tensorflow as tf
def generate_gd(data_path):
    parameters = {"k1": 0.01, "k2": 0.02, "l": 255, "sigma_g": 0.2, "sigma_l": 0.2, "muc": 0.5, "lc": 0.5,"ksize":30}
    dirs = os.listdir(data_path)
    for dir in dirs:
        abs_dir = os.path.join(data_path, dir)
        if (os.path.isdir(abs_dir)):
            reference_path = os.path.join(abs_dir, "reference")
            if (os.path.exists(reference_path)):
                 if((os.path.exists(os.path.join(reference_path,"hdr.tif")))):
                    img_path = glob.glob(os.path.join(reference_path, "*.tif"))
                    ue_image = img_as_float32(io.imread(img_path[0]))
                    ne_image = img_as_float32(io.imread(img_path[1]))
                    oe_image = img_as_float32(io.imread(img_path[2]))
                    ldrs = np.stack((ue_image, ne_image, oe_image))
                    output = caculate_Qmap2(ldrs, **parameters)
                    output[output < 0] = 0
                    output = output * 255
                    output = output.astype("uint8")
                    result = np.reshape(output, [1000, 1500, 3])
                    io.imsave(reference_path + "\\hdr_"+str(parameters["ksize"])+".tif", result)

    return True
def caculate_lk(input,ksize):
    lk=tf.reduce_mean(tf.nn.avg_pool3d(input,ksize=[1,1,ksize,ksize,1],strides=[1,1,1,1,1],padding="SAME"),axis=4)
    # lk1=tf.expand_dims(lk,axis=4)
    # lk1=tf.concat([lk1,lk1,lk1],axis=4)
    # lk1=tf.reshape(lk1,shape=[lk1.shape[0],lk1.shape[1],lk1.shape[2]*lk1.shape[3]*lk1.shape[4]])
    return lk
def caculate_Qmap2(input_ldrs,ksize=7,im_height=1000,im_width=1500,**kwargs):
    k1 = kwargs["k1"]
    k2 = kwargs["k2"]
    '''l depends on the amount of bit of the sequence (l=255 for an 8-bit sequence)'''
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
    lk = caculate_lk(image_patches,ksize=ksize)
    #image_patches = tf.reshape(image_patches,shape=[image_patches.shape[0],image_patches.shape[1],image_patches.shape[2]*image_patches.shape[3]*image_patches.shape[4]])


    '''input_tensor[n,3,m] lk[n,3,m] ck_tensor[n,3] ic[n]'''
    mean_removed=tf.subtract(image_patches,tf.expand_dims(lk,-1))
    ck_tensor=tf.norm(tf.reshape(mean_removed,[mean_removed.shape[0],mean_removed.shape[1],mean_removed.shape[2]*mean_removed.shape[3]*mean_removed.shape[4]]),ord="euclidean",axis=2)
    ic=tf.reduce_max(ck_tensor,axis=1)

    '''lk[n,3,m] uk[n,3,m] il[n,m],uk_sum[n,m]'''
    uk=tf.exp(tf.add(tf.negative(tf.divide(tf.pow(tf.subtract(lk,lc),2),2*tf.pow(sigma_l,2))),tf.reshape(tf.negative(tf.divide(tf.pow(tf.subtract(muk,muc),2),2*tf.pow(sigma_g,2))),shape=[1,3,1,1])))
    uk_sum=tf.reduce_sum(uk,axis=1)
    il=tf.divide(tf.reduce_sum(tf.multiply(lk,uk),axis=1),uk_sum)

    '''rk [n] sk[n,3,cN2]'''
    sk=tf.divide(mean_removed,tf.reshape(ck_tensor,[1,3,1,1,1]))
    #rk=tf.divide(tf.norm(tf.reduce_sum(mean_removed,axis=1),axis=1),tf.reduce_sum(ck_tensor,axis=1))

    '''p [n] WK[n,3] x[n,CN2] istructure[n,m]'''
    #p=tf.tan(tf.multiply(rk,math.pi/2))
    p=tf.constant(4,shape=[1],dtype=tf.float32)
    wk=tf.pow(ck_tensor,tf.expand_dims(p,axis=1))
    wk_sum=tf.reduce_sum(wk,axis=1)
    s=tf.divide(tf.reduce_sum(tf.multiply(sk,tf.reshape(wk,[1,3,1,1,1])),axis=1),wk_sum)
    s_norm=tf.norm(tf.reshape(s,[s.shape[0],s.shape[1]*s.shape[2]*s.shape[3]]),ord="euclidean",axis=1)
    istructure=tf.divide(s,s_norm)
    x=tf.add(tf.multiply(istructure,ic),tf.expand_dims(il,axis=3))
    # cov_xy = tf.diag_part(my_tf_cov(x, output_patches))
    # mux = tf.reduce_mean(x, 1)
    # muy = tf.reduce_mean(output_patches, 1)
    # cov_xx = tf.diag_part(my_tf_cov(x,x))
    # cov_yy = tf.diag_part(my_tf_cov(output_patches,output_patches))
    # S_map = (2*mux*muy+c1)*(2*cov_xy+c2)/((tf.pow(mux,2)+tf.pow(muy,2)+c1)*(tf.pow(cov_xx,2)+tf.pow(cov_yy,2)+c2))
    with tf.Session() as sess:

        output=sess.run(x)

    return output
if __name__=="__main__":
    paths=["D:\Train_OnlyGT\\16-10-10-a-01\\Reference"]
    pathaaa="D:\Train_OnlyGT\\16-10-10-a-01\Reference","D:\Train_OnlyGT\\16-10-10-a-06\Reference","D:\Train_OnlyGT\\16-10-10-a-30\Reference"
    ldr_list=[]
    i=0
    for path in paths:
        i += 1
        img_path = glob.glob(os.path.join(path, "*.tif"))
        ue_image = img_as_float32(io.imread(img_path[0]))
        ne_image = img_as_float32(io.imread(img_path[1]))
        oe_image = img_as_float32(io.imread(img_path[2]))
        ldrs = np.stack((ue_image, ne_image, oe_image))
        #for sigma_g in range(10, 13, 1):
        parameters = {"k1": 0.01, "k2": 0.02, "l": 255, "sigma_g": 0.2, "sigma_l": 0.5, "muc": 0.5,
                      "lc": 0.5,
                      "ksize": 21}
        x= caculate_Qmap2(ldrs, **parameters)
        output = x * 255
        output = output.astype("uint8")
        # result = np.reshape(output, [1000, 1500, 3])
        io.imsave("./hdr_" + "_" + str(parameters["sigma_g"]) + "_" + str(parameters["sigma_g"]) + "_" + str(
            parameters["ksize"]) + ".tif", output)




