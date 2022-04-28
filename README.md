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

This readme will be updated to include the full list of available parameters in the near future as we finalize the project.
