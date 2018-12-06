import keras
from keras import layers, models, Model, Input, regularizers
from keras.callbacks import Callback
import keras.backend as K

import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype, is_numeric_dtype

import matplotlib.pyplot as plt
import pdb


class mixedInputModel():

    def __init__(self,df,y,cat_vars,emb_szs,hidden_layers,hidden_drop,l2=None,
                    emb_drop=0,cont_input_drop=0,output_size=1,use_bn=True,
                 is_reg=True,hidden_activation='relu',opt = None, metrics=['acc'],is_multi=False,
                debug=False,prefix='',bn_start=False):
    
        self.df,self.y, self.cat_vars, self.emb_szs                = df,y, cat_vars, emb_szs 
        self.hidden_layers, self.hidden_drop, self.emb_drop = hidden_layers, hidden_drop, emb_drop
        self.cont_input_drop, self.output_size, self.use_bn = cont_input_drop, output_size, use_bn
        self.is_reg, self.hidden_activation                 = is_reg, hidden_activation
        
        self.opt, self.metrics = opt,metrics
        self.is_multi = is_multi

        self.output_activation = 'sigmoid' if  is_reg else 'softmax' 
        self.loss =  'mse' if is_reg else 'binary_crossentropy' if self.is_multi \
                        else 'categorical_crossentropy' if (output_size>1) else 'binary_crossentropy'
        self.log = None
        self.debug = debug
        self.prefix=prefix
        self.bn_start = bn_start
        self.dnn_tensors = []
        self.best_model_path = None
        self.l2 = l2
     
    def genModel(self,):
        
        cat_vars_ind = False
        cont_vars_ind= False
        # INPUTS (TENSORS & DATA)
        self.input_tensors = []
        self.input_data_reform = []
        
        # CONTINOUS FEATURES AS INPUT
        if(self.df.shape[1]>len(self.cat_vars)):
            cont_vars_ind=True
            self.cont_df = self.df.drop(self.cat_vars,axis=1).values.astype('float32')    
            self.cont_input = layers.Input(shape=(self.cont_df.shape[1],))
            
            if(self.bn_start):
                cont = layers.BatchNormalization()(self.cont_input)
                cont = layers.Dropout(self.cont_input_drop)(cont)
            else:
                cont = layers.Dropout(self.cont_input_drop)(self.cont_input)
                
            self.input_tensors.append(self.cont_input)
            self.input_data_reform.append(self.cont_df)
        
        # CATEGORICAL FEATURES AS SEPERATE ARRAYS
        self.cat_values = []
        self.cat_df = self.df[self.cat_vars]
        for n,c in self.cat_df.items():
            self.cat_values.append(c.values.astype('float32'))

        # CREATE EMBEDDING LAYERS
        self.embeddings = []
        self.emb_inputs = []
        self.emb_models = []
        
        if(len(self.cat_vars)>0): #check we have cat vars
            cat_vars_ind = True
            for dict_sz,out_sz in self.emb_szs:
                input_layer_i = layers.Input(shape=(1,))
                emb_layer_i = layers.Embedding(dict_sz, out_sz, input_length=1)(input_layer_i)
                self.emb_inputs.append(input_layer_i)
                self.embeddings.append(emb_layer_i)
                self.emb_models.append(keras.Model(inputs = input_layer_i, outputs = emb_layer_i))

            ent_emb = keras.layers.Concatenate(name=self.prefix+'Embedding_layer')(self.embeddings)
            ent_emb = layers.Flatten()(ent_emb)
            ent_emb = layers.Dropout(self.emb_drop)(ent_emb)
            self.input_tensors += self.emb_inputs
            self.input_data_reform += self.cat_values

        
#         self.input_tensors = [self.cont_input]+self.emb_inputs
#         self.input_data_reform = [self.cont_df]+self.cat_values
        
        # CONCATENATE CONTINOUS AND EMBEDDED CATEGORICAL
        if(cat_vars_ind and cont_vars_ind):
            self.joint_tensor = keras.layers.Concatenate(name=self.prefix+'All_features')([ent_emb,cont])
        elif(cat_vars_ind):
            self.joint_tensor = ent_emb
        else:
            self.joint_tensor = cont
        

        # DENSE HIDDEN LAYERS
        self.output_tensor,self.mid_output_tensors = self.simpleDNN(self.joint_tensor)
                   
        # DEFINE AND COMPILE MODEL
        self.defineAndCompile()
    
    
    
    def simpleDNN(self,nn_input_tensor,hidden_layers=None,hidden_drop=None,balance_dr=False,l2=None):
        
        mid_outputs = []
        x = layers.Lambda(lambda x:x)(nn_input_tensor)
        
        if(hidden_layers is None): hidden_layers,hidden_drop,l2 = self.hidden_layers,self.hidden_drop,self.l2
        l2_reg = None if l2 is None else regularizers.l2(l=l2)
        
        for l,dr in zip(self.hidden_layers,self.hidden_drop):
            l_out = int(l/(1-dr)) if balance_dr else l
            x = layers.Dense(l_out,activation=self.hidden_activation,kernel_regularizer=l2_reg)(x)
            mid_outputs.append(x)
            if self.use_bn: x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr)(x)
            
     
        
        # OUTPUT  ACTIVATION
        if( (self.output_size>1) and (not self.is_reg)) and (not self.is_multi):
            self.output_activation = 'softmax'
        else:
            self.output_activation = 'sigmoid'
        
        return layers.Dense(self.output_size,name=self.prefix+'Output_Layer',activation=self.output_activation)(x),mid_outputs
    
    def defineAndCompile(self):
        
        self.struct_model = keras.Model(inputs=self.input_tensors,outputs=self.output_tensor)
        self.opt = self.opt if self.opt is not None else keras.optimizers.Adam() # keras.optimizers.SGD(momentum=0.9)
        self.compileModel(opt=self.opt)
        
        # COMPILE MODEL
    def compileModel(self,opt=None):
        if(opt is not None): self.opt = opt 
        self.struct_model.compile(optimizer=self.opt, loss=self.loss,metrics=self.metrics)
        self.init_weights = self.struct_model.get_weights()
        
    
    
    def addModelToJointFeat(self,new_df,input_tensor,concat_tensor):
        
        # APPEND NEW INPUT
        self.input_tensors.append(input_tensor)
        self.input_data_reform.append(new_df)
        
        # CONCATENATE NEW FEATURES
        self.joint_tensor = keras.layers.Concatenate(name=self.prefix+'new_joint_features')([self.joint_tensor,concat_tensor])
       
        # PASS THROUGH DNN
        self.output_tensor = self.simpleDNN(self.joint_tensor)
        
        # DEFINE AND COMPILE
        self.defineAndCompile()
        
    
    def fit(self, lr, batch_size = 128,epochs = 1, val_split = 0.0,validation_data=None,callbacks=None,
           cycle_length=1,mult_factor=1,sgdr=True,flr=False,save_best=False,model_path=None,early_stopping=True,
            patience=20,monitor='val_loss',baseline=None,sample_weight=None):
        
        K.set_value(self.struct_model.optimizer.lr, lr)
        callbacks = callbacks if callbacks is not None else []
        
        # LR SCHEDULE - SGDR
        schedule = SGDRScheduler(min_lr=lr/10,
                                 max_lr=lr,
                                 batch_size=batch_size,
                                 tr_sample_size = np.ceil((1-val_split)*self.df.shape[0]),
                                 lr_decay=1,
                                 cycle_length=cycle_length,
                                 mult_factor=mult_factor)
        if sgdr: callbacks.append(schedule)
        
        # LR SCHEDULE - FACTOR DECREASE
        if(flr):
            f_schedule = keras.callbacks.ReduceLROnPlateau(monitor=monitor, factor=0.1, patience=10, 
                                          min_delta=0.0001, cooldown=5, min_lr=0)
            callbacks.append(f_schedule)

        # SAVE BEST
        if(save_best):
            model_path = model_path if model_path is not None else './BestDNN_Model.h5'
            save_best = keras.callbacks.ModelCheckpoint(model_path, 
                                                    monitor= monitor,
                                                    verbose=0, 
                                                    save_best_only=True, save_weights_only=False,
                                                    mode='auto',
                                                    period=1)
            callbacks.append(save_best)
            self.best_model_path=model_path
        
        # EARLY STOPPING
        if(early_stopping):
            early_stop = keras.callbacks.EarlyStopping(monitor=monitor, 
                                          min_delta=0, 
                                          patience=patience, 
                                          verbose=0, 
                                          mode='auto', 
                                          baseline=baseline, 
                                          restore_best_weights=True)
            callbacks.append(early_stop)
            
            
            
            
        # FIT    
        if(self.debug==True): pdb.set_trace()
        log=self.struct_model.fit(x=self.input_data_reform,y=[self.y],
                 batch_size=batch_size,epochs=epochs,callbacks=callbacks,validation_split=val_split,
                             validation_data=validation_data,sample_weight=sample_weight)
        
        
        if(self.log is None):
            self.log = log 
        else:
            for key in log.history:
                self.log.history[key] = self.log.history[key] + log.history[key]
                
        self.schedule = schedule
            
    def loadBestModel(self):
        if(self.best_model_path is not None):
            self.struct_model = models.load_model(self.best_model_path)
        else:
            print('BEST MODEL NOT LOADED! model_path DOES NOT EXIST!')
        
    def loadModel(self,model_path):
        
        self.struct_model = models.load_model(model_path)
        
    def saveModel(self,model_path):
        self.struct_model.save(model_path)
        
    def print_acc(self):

        # summarize history for accuracy
        plt.figure(figsize=(10,10))
        plt.plot(self.log.history['acc'])
        plt.plot(self.log.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    def print_loss(self):
        plt.figure(figsize=(10,10))
        plt.plot(self.log.history['loss'])
        plt.plot(self.log.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()
        
    def print_full_recent_acc(self):

        plt.figure(figsize=(10,10))
        plt.plot(self.schedule.history['acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('batch')
        plt.legend(['train acc'], loc='upper left')
        plt.show()
        
    def print_full_recent_loss(self):
        plt.figure(figsize=(10,10))
        plt.plot(self.schedule.history['loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('batch')
        plt.legend(['train loss'], loc='upper left')
        plt.show()
        
    def print_recent_lr(self):
        plt.figure(figsize=(10,10))
        plt.plot(self.schedule.history['lr'])
        plt.title('model lr')
        plt.ylabel('lr')
        plt.xlabel('batch')
        plt.show()   
        
# class LRHistory(keras.callbacks.Callback):
#     def on_train_begin(self, logs={}):
#         self.lr = []

#     def on_batch_end(self, batch, logs={}):
#         self.losses.append(logs.get('loss'))
    
    def find_lr(self,min_lr=1e-5,max_lr=1.0,epochs=1,batch_size=256,linear=False):
        
       
        lr_finder = LRFinder(min_lr=min_lr, 
                             max_lr=max_lr,                               
                             epochs=epochs,
                             batch_size=batch_size,
                             tr_sample_size = np.ceil(self.df.shape[0]),
                             linear=linear
                            )
        
        
        self.struct_model.fit(x=[self.cont_df]+self.cat_values,y=[self.y],
                              batch_size=batch_size,
                              callbacks=[lr_finder])
        self.struct_model.set_weights(self.init_weights)

        lr_finder.plot_loss()
        self.lr_finder = lr_finder
        
    def getEmbeddings(self,data_df):
        """get dictionary of Data frames containing the Embedding wiegths learned by the network
        input data_df is needed for categoreis names"""
        
        self.embeddings = {}
        for i,l in enumerate(self.struct_model.get_weights()):
            if(i<=len(self.cat_vars)-1):
                col_name = self.cat_vars[i]
                embedding_names = ['UNKNOWN'] + list(data_df[col_name].cat.categories)
                self.embeddings[col_name] = pd.DataFrame(l,index=embedding_names)
                
        return self.embeddings
                
    def getMidModel(self,index):
        return keras.Model(inputs=self.input_tensors,outputs=self.mid_output_tensors[index])
    
    
    def getMidTensor(self,index):
        return self.mid_output_tensors[index]
        
    def addModel2DNN(self,new_df,input_tensor,concat_tensor,
                 tensor_index=-1, hidden_layers=None,hidden_drop=None):
        """
        input_tensor: input tensor of the added model
        concat_tensor: output tensor of the added model
        tensor_index: if dnn, index if the desired tensor in the dnn list of tensors"""
        # APPEND NEW INPUT
        self.input_tensors.append(input_tensor)
        self.input_data_reform.append(new_df)
        
        # CONCATENATE NEW TENSOR
        tensor_to_join_with = self.mid_output_tensors[tensor_index]
        self.new_joint_tensor = keras.layers.Concatenate(name=self.prefix+'new_joint')([tensor_to_join_with,concat_tensor])
       
        # PASS THROUGH DNN
        self.mid_output_tensors = self.mid_output_tensors[:tensor_index] # remove previous
        self.output_tensor,mid_output_tensors = self.simpleDNN(self.joint_tensor,hidden_layers=hidden_layers,hidden_drop=hidden_drop)
        
        
        self.mid_output_tensors = self.mid_output_tensors + [tensor_to_join_with] + mid_output_tensors
        
        # DEFINE AND COMPILE
        self.defineAndCompile()
    
    def addConst2DNN(self,new_input, tensor_index=-1,ni_drop=0.0,oi_drop=0.0, hidden_layers=None,hidden_drop=None):
        """
        new_input: constants to be added to model        
        tensor_index: index if the desired tensor in the dnn list of tensors"""
        
     
        new_input_tensor = layers.Input(shape=(new_input.shape[1],))
        
        # APPEND NEW INPUT
        self.input_tensors.append(new_input_tensor)
        self.input_data_reform.append(new_input)
        
        # CONCATENATE NEW TENSOR
        tensor_to_join_with = self.mid_output_tensors[tensor_index]
        
        x1 = layers.Dropout(ni_drop)(new_input_tensor)
        if self.use_bn: x2 = layers.BatchNormalization()(tensor_to_join_with)
        x2 = layers.Dropout(oi_drop)(x2)
        
        self.new_joint_tensor = keras.layers.Concatenate(name=self.prefix+'new_joint')([x1,x2])
       
        # PASS THROUGH DNN
        self.mid_output_tensors = self.mid_output_tensors[:tensor_index] # remove previous
        self.output_tensor,mid_output_tensors = self.simpleDNN(self.new_joint_tensor,hidden_layers=hidden_layers,hidden_drop=hidden_drop)
        
        
        self.mid_output_tensors = self.mid_output_tensors + [tensor_to_join_with] + mid_output_tensors
        
        # DEFINE AND COMPILE
        self.defineAndCompile()
        
#================================================================
#================================================================
#================================================================


def genStructModelInput(df,cat_vars):
    cat_values = []
    cat_df = df[cat_vars]
    for n,c in cat_df.items():
        cat_values.append(c.values.astype('float32'))
    
    if(df.shape[1]>len(cat_vars)):
        cont_df = df.drop(cat_vars,axis=1).values.astype('float32') 
        return [cont_df]+cat_values
    else:
        return cat_values


#================================================================
#================================================================
#================================================================



class SGDRScheduler(Callback):
    '''Cosine annealing learning rate scheduler with periodic restarts.
    # Usage
        ```python
            schedule = SGDRScheduler(min_lr=1e-5,
                                     max_lr=1e-2,
                                     steps_per_epoch=np.ceil(epoch_size/batch_size),
                                     lr_decay=0.9,
                                     cycle_length=5,
                                     mult_factor=1.5)
            model.fit(X_train, Y_train, epochs=100, callbacks=[schedule])
        ```
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        lr_decay: Reduce the max_lr after the completion of each cycle.
                  Ex. To reduce the max_lr by 20% after each cycle, set this value to 0.8.
        cycle_length: Initial number of epochs in a cycle.
        mult_factor: Scale epochs_to_restart after each full cycle completion.
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: http://arxiv.org/abs/1608.03983
    '''
    def __init__(self,
                 min_lr,
                 max_lr,
                 batch_size,
                 tr_sample_size,
                 lr_decay=1,
                 cycle_length=10,
                 mult_factor=2):

        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_decay = lr_decay

        self.batch_since_restart = 0
        self.next_restart = cycle_length

        self.steps_per_epoch = np.ceil(tr_sample_size/batch_size)

        self.cycle_length = cycle_length
        self.mult_factor = mult_factor

        self.history = {}

    def clr(self):
        '''Calculate the learning rate.'''
        fraction_to_restart = self.batch_since_restart / (self.steps_per_epoch * self.cycle_length)
        lr = self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(fraction_to_restart * np.pi))
        return lr

    def on_train_begin(self, logs={}):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.max_lr)

    def on_batch_end(self, batch, logs={}):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)

        self.batch_since_restart += 1
        K.set_value(self.model.optimizer.lr, self.clr())

    def on_epoch_end(self, epoch, logs={}):
        '''Check for end of current cycle, apply restarts when necessary.'''
        if epoch + 1 == self.next_restart:
            self.batch_since_restart = 0
            self.cycle_length = np.ceil(self.cycle_length * self.mult_factor)
            self.next_restart += self.cycle_length
            self.max_lr *= self.lr_decay
            self.best_weights = self.model.get_weights()

#     def on_train_end(self, logs={}):
#         '''Set weights to the values from the end of the most recent cycle for best performance.'''
#         self.model.set_weights(self.best_weights)
 
class LRFinder(Callback):
    
    '''
    A simple callback for finding the optimal learning rate range for your model + dataset. 
    
    # Usage
        ```python
            lr_finder = LRFinder(min_lr=1e-5, 
                                 max_lr=1e-2, 
                                 steps_per_epoch=np.ceil(epoch_size/batch_size), 
                                 epochs=3)
            model.fit(X_train, Y_train, callbacks=[lr_finder])
            
            lr_finder.plot_loss()
        ```
    
    # Arguments
        min_lr: The lower bound of the learning rate range for the experiment.
        max_lr: The upper bound of the learning rate range for the experiment.
        steps_per_epoch: Number of mini-batches in the dataset. Calculated as `np.ceil(epoch_size/batch_size)`. 
        epochs: Number of epochs to run experiment. Usually between 2 and 4 epochs is sufficient. 
        
    # References
        Blog post: jeremyjordan.me/nn-learning-rate
        Original paper: https://arxiv.org/abs/1506.01186
    '''
    
    def __init__(self,
                 batch_size,
                 tr_sample_size,
                 min_lr=1e-5,
                 max_lr=1.0,
                 epochs=1,
                 linear = False):
        
        super().__init__()
        
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.batch_size = batch_size
        self.tr_sample_size = tr_sample_size
        steps_per_epoch=np.ceil(tr_sample_size/batch_size)
        self.total_iterations = steps_per_epoch * epochs
        self.iteration = 0
        self.history = {}
        self.linear = linear
        
        ratio = max_lr/min_lr
        self.lr_mult = ratio**(1/steps_per_epoch)
        
    def clr(self):
        '''Calculate the learning rate.'''
        if(self.linear):
            x = self.iteration / self.total_iterations 
            return self.min_lr + (self.max_lr-self.min_lr) * x
        else:
            mult = self.lr_mult**self.iteration
            return self.min_lr * mult
        
    def on_train_begin(self, logs=None):
        '''Initialize the learning rate to the minimum value at the start of training.'''
        logs = logs or {}
        K.set_value(self.model.optimizer.lr, self.min_lr)
        
    def on_batch_end(self, epoch, logs=None):
        '''Record previous batch statistics and update the learning rate.'''
        logs = logs or {}
        self.iteration += 1

        self.history.setdefault('lr', []).append(K.get_value(self.model.optimizer.lr))
        self.history.setdefault('iterations', []).append(self.iteration)

        for k, v in logs.items():
            self.history.setdefault(k, []).append(v)
            
        K.set_value(self.model.optimizer.lr, self.clr())
 
    def plot_lr(self):
        '''Helper function to quickly inspect the learning rate schedule.'''
        plt.plot(self.history['iterations'], self.history['lr'])
        #plt.yscale('log')
        plt.xlabel('Iteration')
        plt.ylabel('Learning rate')
        
    def plot_loss(self):
        '''Helper function to quickly observe the learning rate experiment results.'''
        plt.plot(self.history['lr'], self.history['loss'])
        plt.xscale('log')
        plt.xlabel('Learning rate')
        plt.ylabel('Loss')

def createSimpleDNN(df,hidden_layers,hidden_drop,output_size,output_activation,input_drop=0,hidden_activation='relu',use_bn=True):
        
        input_tensor = layers.Input(shape=(df.shape[1],))
        x = layers.Dropout(input_drop)(input_tensor)
        
        for l,dr in zip(hidden_layers,hidden_drop):
            x = layers.Dense(l,activation=hidden_activation,)(x)
            if use_bn: x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr)(x)
                
        output_tensor = layers.Dense(output_size,activation=output_activation)(x)
        
        return input_tensor,output_tensor
