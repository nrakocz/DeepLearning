class mixedInputModel():

    def __init__(self,df,y,cat_vars,emb_szs,hidden_layers,hidden_drop,
                    emb_drop=0,cont_input_drop=0,output_size=1,use_bn=True,
                 is_reg=True,hidden_activation='relu',opt = None, metrics=['acc'],is_multi=False):
    
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
     
    def genModel(self,):
        
        # CATEGORICAL FEATURES AS SEPERATE ARRAYS
        self.cat_values = []
        cat_df = df[cat_vars]
        for n,c in cat_df.items():
            self.cat_values.append(c.values.astype('float32'))

        # CREATE EMBEDDING LAYERS
        self.embeddings = []
        self.emb_inputs = []
        self.emb_models = []

        for dict_sz,out_sz in self.emb_szs:
            input_layer_i = layers.Input(shape=(1,))
            emb_layer_i = layers.Embedding(dict_sz, out_sz, input_length=1)(input_layer_i)
            self.emb_inputs.append(input_layer_i)
            self.embeddings.append(emb_layer_i)
            self.emb_models.append(keras.Model(inputs = input_layer_i, outputs = emb_layer_i))

        ent_emb = keras.layers.Concatenate(name='Embedding_layer')(self.embeddings)
        ent_emb = layers.Flatten()(ent_emb)
        ent_emb = layers.Dropout(self.emb_drop)(ent_emb)


        # CONTINOUS FEATURES AS INPUT
        self.cont_df = df.drop(cat_vars,axis=1).values.astype('float32')    
        self.cont_input = layers.Input(shape=(self.cont_df.shape[1],))
        cont = layers.Dropout(self.cont_input_drop)(self.cont_input)

        # CONCATENATE CONTINOUS AND EMBEDDED CATEGORICAL
        x = keras.layers.Concatenate(name='All_features')([ent_emb,cont])

        # DENSE HIDDEN LAYERS
        for l,dr in zip(self.hidden_layers,self.hidden_drop):
            x = layers.Dense(l,activation=self.hidden_activation,)(x)
            if self.use_bn: x = layers.BatchNormalization()(x)
            x = layers.Dropout(dr)(x)
        
        # OUTPUT  ACTIVATION
        if( (self.output_size>1) and (not self.is_reg)) and (not self.is_multi):
            self.output_activation = 'softmax'
        else:
            self.output_activation = 'sigmoid'
        
        self.output_tensor = layers.Dense(self.output_size,name='Output_Layer',activation=self.output_activation)(x)        
        
            
        
        # DEFINE MODEL
        self.struct_model = keras.Model(inputs=[self.cont_input]+self.emb_inputs,outputs=self.output_tensor)
        self.input_tensors = [self.cont_input]+self.emb_inputs
        
        # COMPILE MODEL
        self.opt = self.opt if self.opt is not None else keras.optimizers.Adam() # keras.optimizers.SGD(momentum=0.9)
        self.struct_model.compile(optimizer=self.opt, loss=self.loss,metrics=self.metrics)

    def fit(self, lr, batch_size = 128,epochs = 1, val_split = 0.0,validation_data=None,callbacks=None,
           cycle_length=1,mult_factor=1,sgdr=True):
        
        
        callbacks = callbacks if callbacks is not None else []
        
        schedule = SGDRScheduler(min_lr=lr/10,
                                 max_lr=lr,
                                 batch_size=batch_size,
                                 tr_sample_size = np.ceil((1-val_split)*df.shape[0]),
                                 lr_decay=1,
                                 cycle_length=cycle_length,
                                 mult_factor=mult_factor)
        if sgdr: callbacks.append(schedule)
        
        #pdb.set_trace()  
        log=self.struct_model.fit(x=[self.cont_df]+self.cat_values,y=[self.y]
                 ,batch_size=batch_size,epochs=epochs,callbacks=callbacks,validation_split=val_split,
                             validation_data=validation_data)
        
        
        if(self.log is None):
            self.log = log 
        else:
            for key in log.history:
                self.log.history[key] = self.log.history[key] + log.history[key]
                
        self.schedule = schedule
            
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
        
