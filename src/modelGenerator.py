from keras.layers import *
from keras.models import *
from keras.callbacks import *
import keras.backend as K

# TODO: more models!

"""
This function generates a Keras WaveNet model with the following input params:

PARAMS!!
"""
def create_wavenet_model(input_size: int, output_size: int, timestep_count: int) -> Sequential:
    
    # clear keras cache
    K.clear_session()
    
    # init model
    model = Sequential()
        
    #embedding layer
    model.add(Embedding(input_size, 100, input_length=timestep_count, trainable=True)) 

    model.add(Conv1D(64,3, padding='causal',activation='relu'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))
        
    model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    model.add(Conv1D(256,3,activation='relu',dilation_rate=4,padding='causal'))
    model.add(Dropout(0.2))
    model.add(MaxPool1D(2))

    #model.add(Conv1D(256,5,activation='relu'))    
    model.add(GlobalMaxPool1D())
        
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    # print model stats/summary
    model.summary()

    return model

# """
# """
# def create_lstm_model():
#   model = Sequential()
#   model.add(LSTM(128,return_sequences=True))
#   model.add(LSTM(128))
#   model.add(Dense(256))
#   model.add(Activation('relu'))
#   model.add(Dense(n_vocab))
#   model.add(Activation('softmax'))
#   model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')
#   model.summary()
#   return model