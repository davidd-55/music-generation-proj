# Music Generation Project for CS158
*By Emily T., Dave C., Sarah S., and David D.*

## Repo Summary:
This repo once downloaded/configured allows you to train a machine learning algorithm (of various types!) to generate music. For training, the model expects a directory of MIDI (musical instrument digital iunterface) files as input and will generate a MIDI file as output for you to convert to .mp3, .m4a, etc.

For instance, you can collect the MIDI format of music from an artist or genre ([check out piano-midi.de](http://www.piano-midi.de/midi_files.htm)), train the model, then generate music that hopefully sounds like that artist or genre!

The code for this project is based on the project described in [this article](https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/) by aravindpai.

## Required Packages:
This project was built using Python 3.9.12 and requires the following packages (Conda install commands provided):
* Music21 | `conda install -c conda-forge music21`
* NumPy | `conda install -c conda-forge numpy`
* Keras | `conda install -c conda-forge keras`
* TensorFlow | `conda install -c conda-forge tensorflow`
* SciKit Learn | `conda install -c anaconda scikit-learn`

## Running the Project:
In order to begin the training/output process, navigate to the directory you cloned the project into and run the command:
```
python modelRunner.py --path /path/to/training/data --saved_model_name your_model_name.h5
```

Include any of the parameters found below to change them from their default values!

## Available Module Parameters:
Below is a guide to the available module parameters. The parameter values demoed are the default values unless otherwise explicitly stated.

* Select paath to training data | `--path /path/to/data/` (no default)
* Train new model load prev. model | `--train` or `--no-train` (no default)
* Change name of saved or loaded model | `--saved_model_name model.h5` 
* Change name of generated MIDI file | `--music_file_name music.mid`
* Change note frequency requirement | `--freq_threshold 50`
* Change input data timestep chunk | `--timestep_count 32`
* Change model training examples per batch size | `--tr_batch_size 128`
* Change model training epoch count | `--tr_epoch_count 50`
* Change length of generated music (apprx. .5s) | `--music_generate_length 100`
* Change train/test data split (test data %) | `--train_test_split 0.2`
* Change input data instrument to filter by | `--instrument Piano`

If you would like to demo the basic neural network or WaveNet models, just switch the model in use in lines 77-79 in `modelRunner.py`!
