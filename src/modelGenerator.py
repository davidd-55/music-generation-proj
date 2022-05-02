from keras.layers import *
from keras.models import *
from keras.callbacks import *
from keras.losses import *
import keras.backend as K

# TODO: more models!

"""
This function generates a simple Keras NN model with the following input params:

"""
def create_basic_NN(input_size: int, output_size: int, timestep_count: int) -> Sequential:
    
    K.clear_session()
    
    model = Sequential()

    model.add(Input(shape=(timestep_count)))

    model.add(Dense(32, activation="tanh"))

    model.add(Dense(64, activation="tanh"))

    model.add(Dense(output_size, activation="tanh"))

    model.compile(loss=Hinge(), optimizer='adam')

    model.summary()

    return model


"""
This function generates a Keras WaveNet model with default params from our guide at


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

"""
This function generates a Keras WaveNet model with the following input params changed from the original wavenet model
Diffs:
- 1 fewer convolutional layer
- Dropout from 0.2 to 0.1
- MaxPool1D from 2 to 3

"""
def create_custom_wavenet_model(input_size: int, output_size: int, timestep_count: int) -> Sequential:
    
    # clear keras cache
    K.clear_session()
    
    # init model
    model = Sequential()
        
    #embedding layer
    model.add(Embedding(input_size, 100, input_length=timestep_count, trainable=True)) 

    model.add(Conv1D(64,3, padding='causal',activation='relu'))
    model.add(Dropout(0.1))
    model.add(MaxPool1D(3))
        
    model.add(Conv1D(128,3,activation='relu',dilation_rate=2,padding='causal'))
    model.add(Dropout(0.1))
    model.add(MaxPool1D(3))

    # Subtract a layer!

    model.add(GlobalMaxPool1D())
        
    model.add(Dense(256, activation='relu'))
    model.add(Dense(output_size, activation='softmax'))
        
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

    # print model stats/summary
    model.summary()

    return model