# DeepLearning
Deep Learning related repo

structuredDNN.py
==============
My implementation of Fastai's columnar-data architecture and fastai's course (+ some extras).
------------------

Usage example:
cat_sz = [(c, len(data_df_samp[c].cat.categories)+1) for c in cat_vars]

emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

mod = structuredDNN.mixedInputModel(df,y,cat_vars,emb_szs, [141,140],[0.4,0.4],
                      emb_drop=0.02,cont_input_drop=0.02,
                      output_size=1,is_reg=False)
                      

mod.genModel() # GENERATE MODEL

mod.struct_model.summary()

lr=1e-3

test_input = structuredDNN.genStructModelInput(df_test,cat_vars) # create test input for validation

mod.fit(lr,cycle_length=3,epochs=12,validation_data=(test_input,y_test))

mod.print_full_recent_loss()

mod.print_loss()

mod.loadBestModel() # LOAD BEST MODEL

y_pred = mod.struct_model.predict(x=test_input,batch_size=128) #PREDICT

mod.getEmbeddings(data_df) # GET EMBEDDING WEIGHTS

mod.addModelToJointFeat(df2,med_input_tensor,med_output_tensor) # ADD MODEL

test_input.append(df2_test) # ADD MODEL - create test input for joint model


