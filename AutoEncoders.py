import tensorflow as tf
import keras
from keras.layers import Input, Dense, Activation
from keras.models import Model,Sequential, load_model
from keras import layers
from keras import backend as K
import numpy as np
from sklearn.model_selection import StratifiedKFold
import os
import sys

sys.path.insert(0, '/home/nrakocz/tools/')
from AE_utils import *

class AutoEncoder(object):
    
    def __init__(self,input_size, encoder_layers, decoder_layers, 
                 optimizer='adam', loss='binary_crossentropy',activation='relu',latent_activation='tanh',patience=20):
        
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.encoder_layers[-1] = encoder_layers[-1]
        self.optimizer = optimizer
        self.loss = loss
        self.patience = patience
        self.input_size = input_size
        self.input_data = Input(shape=(self.input_size,))
        self.activation = activation
        self.latent_activation = latent_activation
        
    def build(self):
        # ENCODER
        #encoder-input
        unit_size = self.input_size
        self.X = layers.Lambda(lambda x:x)(self.input_data)
       
        #encoder-hidden
        for i,unit_size in enumerate(self.encoder_layers[:-1]):
            self.X = Dense(unit_size, activation = self.activation,name='encoder_{}_layer'.format(i))(self.X)
        
        #encoder-output
        self.Z = Dense(self.encoder_layers[-1], activation = self.latent_activation, name='encoder_output_layer')(self.X)
        self.Encoder = Model(self.input_data,self.Z)
        
        # DECODER
        #decoder-input
        self.X = layers.Lambda(lambda x:x)(self.Z)
        
        #decoder-hidden
        for i,unit_size in enumerate(self.decoder_layers[:-1]):
            self.X = Dense(unit_size, activation = self.activation,name='decoder_{}_layer'.format(i))(self.X)
        
        #decoder-output
        self.decoded = Dense(self.input_size, activation = 'sigmoid', name='decoder_output_layer')(self.X)
        
        #COMPILE
        self.model = Model(self.input_data,self.decoded)
        self.model.compile(optimizer=self.optimizer,
            loss=self.loss,
            metrics=['mse'])
        
    def fit(self, X_train, X_test=None, epochs=10, batch_size=100,output_path=None):
        
        

        EarlyStop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=self.patience, verbose=1, mode='auto')
        validation_data = (X_test,X_test) if np.any(X_test) else None
        
        callbacks_list = [EarlyStop]
        
        if(output_path):        
            filepath = 'AE_models/'+output_path+'.hdf5'
            if(not os.path.exists('AE_models')): os.makedirs('AE_models')
            Check_point = keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)
            callbacks_list.append(Check_point)
            
        h=self.model.fit(X_train ,X_train,
                             epochs=epochs,
                             batch_size=batch_size,
                             validation_data=validation_data,
                             callbacks = callbacks_list)
        return h

 
    def predict(self,X):
        return self.model.predict(X)
    
    def printParams(self):
        print('\n'+ ('*'*50 + '\n')*4)
        print('Activation function: {}'.format(self.activation))
        print('Latent dimension: {}'.format(self.encoder_layers[-1]))
        print('Encoder hidden layers: {}'.format(self.encoder_layers))
        print('Decoder hidden layers: {}'.format(self.decoder_layers))
        print('Loss function: {}'.format(self.loss))
        print('\n'+ ('*'*50 + '\n')*4)

    def printSummary(self):
        print('-'*10 +'\nSUMMARY\n'+'-'*10+'\n')
        self.model.summary()
        
    def getModel(self):
        return self.model
    

# ======================================
# VAE
# ======================================
class CustomVariationalLayer(keras.layers.Layer):

    def vae_loss(self, x, z_decoded,z_log_var,z_mean): #
        x = K.flatten(x)
        z_decoded = K.flatten(z_decoded)
        #xent_loss = keras.metrics.binary_crossentropy(x, z_decoded)
        #xent_loss = keras.metrics.mean_squared_error(x, z_decoded)
        #xent_loss = keras.metrics.logcosh(x, z_decoded)
        kl_loss = -5e-4 * K.mean(
            1 + z_log_var - K.square(z_mean) - K.exp(z_log_var), axis=-1) #-5e-4
        return K.mean(kl_loss)

    def call(self, inputs):
        x = inputs[0]
        z_decoded = inputs[1]
        z_log_var = inputs[2]
        z_mean = inputs[3]
        loss = self.vae_loss(x, z_decoded,z_log_var,z_mean)#
        self.add_loss(loss, inputs=inputs)
        # We don't use this output.
        return x






class VariationalAutoEncoder(AutoEncoder):
    
    
    def build(self):
        
        #-----ENCODER-----------------------
        #encoder-input
        self.encoder_input_data = Input(shape=(self.input_size,))
        unit_size = self.input_size
        self.X = layers.Lambda(lambda x:x)(self.encoder_input_data)
       
        #encoder-hidden
        for i,unit_size in enumerate(self.encoder_layers[:-1]):
            self.X = Dense(unit_size, activation = self.activation,name='encoder_{}_layer'.format(i))(self.X)
        
        #encoder-output
        self.z_mean = layers.Dense(self.encoder_layers[-1], activation = self.activation, name='encoder_projection_mean')(self.X)
        self.z_log_var = layers.Dense(self.encoder_layers[-1], activation = self.activation, name='encoder_projection_log_ver')(self.X)
        self.encoder = Model(inputs=self.encoder_input_data,outputs=[self.z_mean,self.z_log_var],name='Encoder')
        
        #-----SAMPLING---------------------
        
        #input_data = self.input_data
        [self.Z_mean, self.Z_log_var] = self.encoder(self.input_data)
        self.Z = layers.Lambda(self.__sampling__,name='sampling_layer')([self.Z_mean, self.Z_log_var])
        
        #-----DECODER----------------------
        # This is the input where we will feed `z`.
        self.decoder_input = layers.Input(shape=(self.encoder_layers[-1],))
        
        # DECODER
        #decoder-input
        self.X = layers.Lambda(lambda x:x)(self.decoder_input)
        
        #decoder-hidden
        for i,unit_size in enumerate(self.decoder_layers[:-1]):
            self.X = Dense(unit_size, activation = self.activation,name='decoder_{}_layer'.format(i))(self.X)
        
        #decoder-output
        self.decoded = Dense(self.input_size, activation = 'sigmoid', name='decoder_output_layer')(self.X)
        # This is our decoder model.
        self.decoder = Model(self.decoder_input, self.decoded,name='Decoder')
        
        #-----OUTPUT-------------------------
        # We then apply it to `z` to recover the decoded `z`.
        self.z_decoded = self.decoder(self.Z)
        #------------------------------------
        
        #----Custom Loss---------------------
        # We call our custom layer on the input and the decoded output,
        # to obtain the final model output.
        y = CustomVariationalLayer()([self.input_data, self.z_decoded, self.Z_mean, self.Z_log_var])
        
        self.model = Model(self.input_data, outputs=[self.z_decoded,self.Z_mean])
        self.model.compile(optimizer='adam', loss=[self.loss,None],metrics=['mse'])

        
        
   
    def __sampling__(self,args):
        self.z_mean, self.z_log_var = args
        epsilon = K.random_normal(shape=(K.shape(self.z_mean)[0], K.shape(self.z_mean)[1]),
                                  mean=0., stddev=1.)
        return self.z_mean + K.exp(self.z_log_var) * epsilon     
        
        
        
    def printSummary(self):
        print('-'*10 +'\nSUMMARY\n'+'-'*10+'\n')
        print('VAE SUMMARY')
        self.model.summary()

        print('\n*****\nENCODER SUMMARY')
        self.encoder.summary()

        print('\n*****\nDECODER SUMMARY')
        self.decoder.summary()
        
    def predict(self,X):
        return self.model.predict(X)[0]
    
    def predict_projection(self,X):
        return self.model.predict(X)[1]
    
        
        

        
# ======================================
# VAE
# ======================================  

def AutoEncoderWrapper(X_train,y_train, encoder_hidden_layers, decoder_hidden_layers, activation='relu',loss='mse', score_loss='nll',
                      AE_type='VAE',X_test=None,y_test=None,cv = 10,batch_size=100,epochs=100):
    
    """ calculate reconstruction score by learning encoding-decoding using the distribution of the majoirty population only
        
        output:
            + res_tr - train data score
            + res_tst - test data score
            + ae - autoencoder model train on training data set
    """
    input_size = X_train.shape[1]
    kf = StratifiedKFold(n_splits=cv,shuffle=True,random_state=101)
    
    X_tr_maj = X_train[y_train==0]
    X_tr_minr = X_train[y_train==1]
    res_tr = np.zeros((X_train.shape[0],))
    res_tst = None
    num_of_epochs = np.zeros((cv,))
        
    
    if(AE_type=='AE'):
        ae = AutoEncoder(input_size,
                     encoder_hidden_layers,
                     decoder_hidden_layers,
                     activation=activation,
                     loss=loss)
    else:
        ae = VariationalAutoEncoder(input_size,
                     encoder_hidden_layers,
                     decoder_hidden_layers,
                     activation=activation,
                     loss=loss)
    
    ae.printParams()
    ae.build()
    ae.printSummary()
    init_weights = ae.model.get_weights()

       
    # train on train data
    for cv_i, (train_index, test_index) in enumerate(kf.split(X_train,y_train)):
        print('\n\nItteration {}/{}\n\n'.format(cv_i+1,cv))
        X_train_tr, X_train_ts, y_train_tr,y_train_ts = X_train[train_index],X_train[test_index],y_train[train_index],y_train[test_index]
        
        X_tr_maj_tr = X_train_tr[y_train_tr==0]
        X_tr_maj_ts = X_train_ts[y_train_ts==0]
        
        
        
        h=ae.fit(X_train=X_tr_maj_tr,
               X_test=X_tr_maj_ts,
               epochs=epochs,
               batch_size = batch_size)

        res_i = calc_ae_raw_score(ae,X_train_ts,score_loss)
        res_tr[test_index] = res_i
        
        print('\nAUC: ',plotRoc(res_i[y_train_ts==0],res_i[y_train_ts==1],wplt=False))
        num_of_epochs[cv_i]=len(h.epoch)

        
        ae.model.set_weights(init_weights) # reset model for CV
        
        
    # predict on minority samples from train data and on test data (if given).
    print('\n\nCreating Final Model\n\n')
    print(np.median(num_of_epochs))
    h=ae.model.fit(X_tr_maj,X_tr_maj,
           epochs=int(np.median(num_of_epochs)),
           batch_size = batch_size)

    res_tr[y_train==1] = calc_ae_raw_score(ae,X_tr_minr,score_loss)
    if(res_tst): res_tst = calc_ae_raw_score(ae,X_test,score_loss)
    
#     #scale raw scores between 0 to 1:
#     scaler = MinMaxScaler()
#     res_tr = scaler.fit_transform(res_tr.reshape(-1,1))
#     if(res_tst): res_tst = scaler.transform(res_tst.reshape(-1,1))
    
    return res_tr,res_tst,ae
    


        
        
def calc_ae_raw_score(ae,X,score_loss='nll'):
    if(score_loss=='nll'):
        return logLoss(X,ae.predict(X))
    else:
        return np.mean(X-ae.predict(X)**2,axis=1)
        


        
        
        

