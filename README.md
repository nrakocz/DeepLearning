# DeepLearning
Deep Learning related repo

structuredDNN.py
==============
My implementation of Fastai's columnar-data architecture and fastai's course.
------------------

Usage example:
cat_sz = [(c, len(data_df_samp[c].cat.categories)+1) for c in cat_vars]

emb_szs = [(c, min(50, (c+1)//2)) for _,c in cat_sz]

mod = mixedInputModel(df,y,cat_vars,emb_szs, [141,140],[0.4,0.4],
                      emb_drop=0.02,cont_input_drop=0.02,
                      output_size=1,is_reg=False)
                      
mod.genModel()

mod.struct_model.summary()

lr=1e-3

mod.fit(lr,cycle_length=3,epochs=12,validation_data=(test_input,y_test))

mod.print_full_recent_loss()

mod.print_loss()


