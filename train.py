from keras.models import Model, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras import backend as K
from read_dataset import *
from model import *


X_train, Y_train, X_test =   get_nuclei_dataset()


model = unet_seg()




# Fit model
earlystopper = EarlyStopping(patience=5, verbose=1)
checkpointer = ModelCheckpoint('model-dsbowl2018-1.h5', verbose=1, save_best_only=True)
results = model.fit(X_train, Y_train, validation_split=0.25, batch_size=16, epochs=50, 
                    callbacks=[earlystopper, checkpointer])