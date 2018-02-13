from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
from keras.layers.merge import concatenate
from keras.models import Model, load_model
import numpy as np
import tensorflow as tf

from keras import backend as K

IMG_HEIGHT = 256
IMG_WIDTH = 256
IMG_CHANNELS = 3

# Define IoU metric
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def unet_seg():

	inputs = Input((IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS))
	normalized_input = Lambda(lambda x: x / 255) (inputs)
	conv_1 = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal') (normalized_input)
	conv_1 = Dropout(0.2)(conv_1)
	conv_2 = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal') (conv_1)
	m_pool_1 = MaxPooling2D(pool_size=(2, 2))(conv_2)

	conv_3 = Conv2D(128, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal') (m_pool_1)
	conv_3 = Dropout(0.2)(conv_3)
	conv_4 = Conv2D(128, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal') (conv_3)
	m_pool_2 = MaxPooling2D(pool_size=(2, 2))(conv_4)

	conv_5 = Conv2D(256, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal') (m_pool_2)
	conv_5 = Dropout(0.2)(conv_5)
	conv_6 = Conv2D(256, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal') (conv_5)
	m_pool_3 = MaxPooling2D(pool_size=(2, 2))(conv_6)

	conv_7 = Conv2D(512, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(m_pool_3)
	conv_7 = Dropout(0.2)(conv_7)
	conv_8 = Conv2D(512, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_7)
	m_pool_4 = MaxPooling2D(pool_size=(2, 2))(conv_8)

	conv_9 = Conv2D(1024, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(m_pool_4)
	conv_9 = Dropout(0.2)(conv_9)
	conv_10 = Conv2D(1024, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(conv_9)
	

	up_sam_1 = Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same') (conv_10)
	con_cat_1 = concatenate([up_sam_1, conv_8])

	concat_conv_1 = Conv2D(512, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(con_cat_1)
	concat_conv_1 = Dropout(0.2)(concat_conv_1)
	concat_conv_2 = Conv2D(512, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat_conv_1)

	up_sam_2 = Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same') (concat_conv_2)
	con_cat_2 = concatenate([up_sam_2, conv_6])

	concat_conv_3 = Conv2D(256, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(con_cat_2)
	concat_conv_3 = Dropout(0.2)(concat_conv_3)
	concat_conv_4 = Conv2D(256, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat_conv_3)


	up_sam_3 = Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same') (concat_conv_4)
	con_cat_3 = concatenate([up_sam_3, conv_4])

	concat_conv_5 = Conv2D(128, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(con_cat_3)
	concat_conv_5 = Dropout(0.2)(concat_conv_5)
	concat_conv_6 = Conv2D(128, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat_conv_5)


	up_sam_4 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (concat_conv_6)
	con_cat_4 = concatenate([up_sam_4, conv_2])

	concat_conv_7 = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(con_cat_4)
	concat_conv_7 = Dropout(0.2)(concat_conv_7)
	concat_conv_8 = Conv2D(64, (3,3), padding = 'same', activation = 'relu', kernel_initializer = 'he_normal')(concat_conv_7)
	outputs = Conv2D(1, (1, 1), activation='sigmoid') (concat_conv_8)

	model = Model(inputs=[inputs], outputs=[outputs])
	model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
	model.summary()

	return model