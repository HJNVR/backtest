import numpy as np
import tensorflow as tf
import random as rn
import os

#--------------------------------------------------------------------------------------------------------------------------------
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Layer
from tensorflow.keras import optimizers,initializers,constraints,regularizers
from tensorflow.keras import backend as K


#--------------------------------------------------------------------------------------------------------------------------------

def fae_selector(X_train, kfnum=25):
    key_feture_number = kfnum
    seed=0
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)
    rn.seed(seed)
    #session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    session_conf =tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)

    from keras import backend as K

    #tf.set_random_seed(seed)
    tf.compat.v1.set_random_seed(seed)
    #sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)

    K.set_session(sess)
    #----------------------------Reproducible----------------------------------------------------------------------------------------

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


    #Import defined methods
    np.random.seed(seed)

    #--------------------------------------------------------------------------------------------------------------------------------
    class Feature_Select_Layer(Layer):
        
        def __init__(self, output_dim, l1_lambda, **kwargs):
            super(Feature_Select_Layer, self).__init__(**kwargs)
            self.output_dim = output_dim
            self.l1_lambda=l1_lambda

        def build(self, input_shape):
            self.kernel = self.add_weight(name='kernel',  
                                        shape=(input_shape[1],),
                                        initializer=initializers.RandomUniform(minval=0.999999, maxval=0.9999999, seed=seed),
                                        trainable=True,
                                        regularizer=regularizers.l1(self.l1_lambda),
                                        constraint=constraints.NonNeg())
            super(Feature_Select_Layer, self).build(input_shape)
        
        def call(self, x, selection=False,k=key_feture_number):
            kernel=self.kernel        
            if selection:
                kernel_=K.transpose(kernel)
                kth_largest = tf.math.top_k(kernel_, k=k)[0][-1]
                kernel = tf.where(condition=K.less(kernel,kth_largest),x=K.zeros_like(kernel),y=kernel)        
            return K.dot(x, tf.linalg.tensor_diag(kernel))

        def compute_output_shape(self, input_shape):
            return (input_shape[0], self.output_dim)

    #--------------------------------------------------------------------------------------------------------------------------------
    def Fractal_Autoencoder(p_data_feature,\
                            p_feture_number=key_feture_number,\
                            p_encoding_dim=key_feture_number,\
                            p_learning_rate=1E-3,\
                            p_l1_lambda=0.1,\
                            p_loss_weight_1=1,\
                            p_loss_weight_2=2,\
                            p_is_use_bias=True):
        
        input_img = Input(shape=(p_data_feature,), name='autoencoder_input')

        feature_selection = Feature_Select_Layer(output_dim=p_data_feature,\
                                                l1_lambda=p_l1_lambda,\
                                                input_shape=(p_data_feature,),\
                                                name='feature_selection')

        feature_selection_score=feature_selection(input_img)
        feature_selection_choose=feature_selection(input_img,selection=True,k=p_feture_number)

        encoded = Dense(p_encoding_dim,\
                        activation='linear',\
                        kernel_initializer=initializers.glorot_uniform(seed),\
                        use_bias=p_is_use_bias,\
                        name='autoencoder_hidden_layer')
        
        encoded_score=encoded(feature_selection_score)
        encoded_choose=encoded(feature_selection_choose)
        
        bottleneck_score=encoded_score
        bottleneck_choose=encoded_choose
        
        decoded = Dense(p_data_feature,\
                        activation='linear',\
                        kernel_initializer=initializers.glorot_uniform(seed),\
                        use_bias=p_is_use_bias,\
                        name='autoencoder_output')
        
        decoded_score =decoded(bottleneck_score)
        decoded_choose =decoded(bottleneck_choose)

        latent_encoder_score = Model(input_img, bottleneck_score)
        latent_encoder_choose = Model(input_img, bottleneck_choose)
        feature_selection_output=Model(input_img,feature_selection_choose)
        autoencoder = Model(input_img, [decoded_score, decoded_choose])
        
        autoencoder.compile(loss=['mean_squared_error','mean_squared_error'],\
                            loss_weights=[p_loss_weight_1, p_loss_weight_2],\
                            optimizer=optimizers.Adam(lr=p_learning_rate))
        
        print('Autoencoder Structure-------------------------------------')
        autoencoder.summary()
        return autoencoder,feature_selection_output,latent_encoder_score,latent_encoder_choose

    epochs_number=800
    batch_size_value=128
    is_use_bias=True

    F_AE,\
    feature_selection_output,\
    latent_encoder_score_F_AE,\
    latent_encoder_choose_F_AE=Fractal_Autoencoder(p_data_feature=X_train.shape[1],\
                                                p_feture_number=key_feture_number,\
                                                p_encoding_dim=key_feture_number,\
                                                p_learning_rate= 1E-3,\
                                                p_l1_lambda=0.1,\
                                                p_loss_weight_1=1,\
                                                p_loss_weight_2=2,\
                                                p_is_use_bias=is_use_bias)

    F_AE_history = F_AE.fit(X_train, [X_train,X_train],\
                        epochs=epochs_number,\
                        batch_size=batch_size_value,\
                        shuffle=True, verbose=0)

    selector = feature_selection_output

    return selector
    