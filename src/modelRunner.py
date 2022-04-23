import argparse
import random
import numpy as np

from keras.models import *
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from midiLoader import load_midi_files, convert_to_midi
from dataCleaner import generate_io_sequence, map_notes_to_ints
from modelGenerator import create_wavenet_model

def train_model(
    model: Sequential,
    x_train: list,
    x_test: list,
    y_train: list,
    y_test: list) -> None:

    mc = ModelCheckpoint('best_model.h5', monitor='val_loss', mode='min', save_best_only=True,verbose=1)
    history = model.fit(np.array(x_train),np.array(y_train),batch_size=128,epochs=10, validation_data=(np.array(x_test),np.array(y_test)),verbose=1, callbacks=[mc])

def test_model(
    model: Sequential, 
    test_data: list, 
    timestep_count: int) -> np.ndarray:

    ind = np.random.randint(0,len(test_data)-1)

    random_music = test_data[ind]

    predictions=[]
    for i in range(10):

        random_music = random_music.reshape(1, timestep_count)

        prob  = model.predict(random_music)[0]
        y_pred= np.argmax(prob,axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
        random_music = random_music[1:]
    
    return np.array(predictions)

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--path", required=True)
    parser.add_argument("--path", default="/Users/daviddattile/Dev/music-generation-proj/music/demo/")
    parser.add_argument("--freq_threshold", type=int, default=50)
    parser.add_argument("--timestep_count", type=int, default=32)
    parser.add_argument("--instrument", default="Piano")
    args = parser.parse_args()

    # load notes from midi files
    loaded_notes = load_midi_files(args.path, args.instrument, args.freq_threshold)

    # generate model input (x), output (y) sequences of notes from loaded notes
    x, y = generate_io_sequence(loaded_notes, args.timestep_count)

    # map input (x), output (y) sequences of notes to unique ints;
    # store a map from int --> note
    x_seq, y_seq, unique_int_to_note = map_notes_to_ints(x, y)

    # TODO: args for t/t split!
    # split into train/test data
    x_tr, x_val, y_tr, y_val = train_test_split(x_seq, y_seq, test_size=0.2, random_state=0)

    # TODO #1 - wtf
    # TODO #2 - args???
    # define model
    model = create_wavenet_model(len(list(set(x_seq.ravel()))), len(list(set(y_seq))), args.timestep_count)

    # train model
    train_model(model, x_tr, x_val, y_tr, y_val)

    # load best fit model
    model = load_model('best_model.h5')

    # generate music! 
    predicted_music = test_model(model, x_val, args.timestep_count)

    # map from unique int --> note
    predicted_notes = [unique_int_to_note[i] for i in predicted_music]

    # save predicted notes to a file
    convert_to_midi(predicted_notes)

if __name__ == "__main__":
    main()