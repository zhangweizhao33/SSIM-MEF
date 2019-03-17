class MEF_SSIM(KE.Layer):
    def __init__(self, inputs=None, patch_size=8, **kwargs):
        super(MEF_SSIM, self).__init__(**kwargs)
        self.patch_size = patch_size
        self.trainable = False
        self.sigma_l = 0.2
        self.sigma_g = 0.2
        self.K1 = 0.01
        self.K2 = 0.03
        self.L = 255
        self.C1 = tf.constant((self.K1 * self.L) ** 2)
        self.C2 = tf.constant((self.K2 * self.L) ** 2)

        def call(self, inputs, output_shape=None, name=None):
            LDRs = inputs[0]
            Fused = inputs[1]
            patch_size = self.patch_size
            sigma_l = self.sigma_l
            sigma_g = self.sigma_g
            K1 = self.K1
            K2 = self.K2
            L = self.L
            C1 = self.C1
            C2 = self.C2

            if K.backend() == 'tensorflow':
                gMu_seq = K.tf.reduce_mean(LDRs, [1, 2, 3])
                LDRs = K.tf.concat(tf.split(K.tf.expand_dims(LDRs, 1), 3, -1), 1)
                lMu_seq = K.tf.reduce_mean(
                    K.tf.nn.avg_pool3d(LDRs, [1, 1, patch_size, patch_size, 1], strides=[1, 1, 1, 1, 1],
                                       padding='SAME'),
                    -1)
                lMu_sq_seq = K.tf.square(lMu_seq)
                sigma_sq_seq = K.tf.reduce_mean(
                    K.tf.nn.avg_pool3d(K.tf.square(LDRs), [1, 1, patch_size, patch_size, 1], strides=[1, 1, 1, 1, 1],
                                       padding='SAME'),
                    -1) - lMu_sq_seq
                patch_index = K.tf.argmax(sigma_sq_seq, 1)
                sigmaY_sq = K.tf.reduce_max(sigma_sq_seq, 1)
                LY = K.tf.exp(
                    -tf.reshape(K.tf.divide(tf.square(gMu_seq - 0.5), 2 * sigma_g ** 2), [-1, 1, 1, 1]) - K.tf.divide(
                        K.tf.square(lMu_seq - 0.5),
                        2 * sigma_l ** 2))
                LY_normed = K.tf.divide(LY, K.tf.expand_dims(K.tf.reduce_sum(LY, 1), 1))
                muY = K.tf.reduce_sum(LY_normed * lMu_seq, 1)
                muY_sq = K.tf.square(muY)

                muX = K.tf.reduce_mean(
                    K.tf.nn.avg_pool(Fused, [1, patch_size, patch_size, 1], [1, 1, 1, 1], padding='SAME'), -1)
                muX_sq = K.tf.square(muX)
                sigmaX_sq = K.tf.reduce_mean(
                    K.tf.nn.avg_pool(K.tf.square(Fused), [1, patch_size, patch_size, 1], [1, 1, 1, 1], padding='SAME'),
                    -1) - muX_sq

                A1_patches = 2 * muX * muY + C1
                B1_patches = muX_sq + muY_sq + C1
                B2_patches = sigmaX_sq + sigmaY_sq + C2
                sigmaXY = K.tf.reduce_mean(
                    K.tf.nn.avg_pool3d(K.tf.expand_dims(Fused, 1) * LDRs, [1, 1, patch_size, patch_size, 1],
                                       [1, 1, 1, 1, 1],
                                       padding='SAME'), [1, -1]) - muX * muY
                A2_patches = 2 * sigmaXY
                qmap = K.tf.divide(A1_patches * A2_patches, B1_patches * B2_patches)
                Q = K.tf.reduce_mean(qmap, [1, 2])
                return [qmap, Q]
            else:
                errmsg = '{} backend is not supported for layer{}'.format(K.backend(), type(self).__name__)
                raise NotImplementedError(errmsg)
                return [1, 1]

        def compute_output_shape(self, input_shape):
            output_shape = tuple(input_shape[0][0:-1])
            return [output_shape, output_shape]
