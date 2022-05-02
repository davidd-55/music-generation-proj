import argparse
import numpy as np

from keras.models import *
from keras.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split

from midiLoader import load_midi_files, convert_to_midi
from dataCleaner import generate_io_sequence, map_notes_to_ints
from modelGenerator import create_wavenet_model, create_custom_model_A

"""
From a trained model and some test data, generate a note array representation of a 
musical file. Specify the length (in timesteps) of the generated music.

Helper code from https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
"""
def generate_music(
    model: Sequential, 
    test_data: list, 
    timestep_count: int,
    generated_music_length: int) -> np.ndarray:

    ind = np.random.randint(0,len(test_data)-1)

    random_music = test_data[ind]

    predictions=[]
    for i in range(generated_music_length):

        random_music = random_music.reshape(1, timestep_count)

        prob  = model.predict(random_music)[0]
        y_pred= np.argmax(prob,axis=0)
        predictions.append(y_pred)

        random_music = np.insert(random_music[0],len(random_music[0]),y_pred)
        random_music = random_music[1:]
    
    return np.array(predictions)

"""
Run some stuff.
"""
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True)
    parser.add_argument("--train", action=argparse.BooleanOptionalAction)
    parser.add_argument("--saved_model_name", default="best_model.h5")
    parser.add_argument("--music_file_name", default="music.mid")
    parser.add_argument("--freq_threshold", type=int, default=50)
    parser.add_argument("--timestep_count", type=int, default=32)
    parser.add_argument("--tr_batch_size", type=int, default=128)
    parser.add_argument("--tr_epoch_count", type=int, default=50)
    parser.add_argument("--music_generate_length", type=int, default=100)
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

    # train if asked
    if (args.train):

        # TODO #1 - wtf
        # TODO #2 - args???
        # define model
        model = create_custom_model_A(len(list(set(x_seq.ravel()))), len(list(set(y_seq))), args.timestep_count)

        # train model
        mc = ModelCheckpoint(args.saved_model_name, monitor='val_loss', mode='min', save_best_only=True,verbose=1)
        model.fit(
            np.array(x_tr), np.array(y_tr),
            batch_size=args.tr_batch_size, epochs=args.tr_epoch_count, 
            validation_data=(np.array(x_val), np.array(y_val)), 
            verbose=1, callbacks=[mc]
            )

    # load specified model
    print("Loading model '" + args.saved_model_name + "'...")
    model = load_model(args.saved_model_name)

    # generate music! 
    print("Generating music...")
    predicted_music = generate_music(
        model, 
        x_val, 
        args.timestep_count,
        args.music_generate_length)

    # map from unique int --> note
    predicted_notes = [unique_int_to_note[i] for i in predicted_music]

    # save predicted notes to a file
    convert_to_midi(predicted_notes, args.music_file_name)
    print("Done!")

if __name__ == "__main__":
    main()