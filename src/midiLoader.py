# lib for understanding music!
from music21 import *

import os
import numpy as np
from collections import Counter


"""
A helper function for reading/loading MIDI files

intstrument = Piano in starter!

Helper code from https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
"""
def read_midi(file: str, instrument_select: str="Piano") -> np.ndarray:
    
    print("Loading Music File:",file)
    
    notes = []
    notes_to_parse = None
    
    # parsing a midi file
    midi = converter.parse(file)
  
    # grouping based on different instruments
    s2 = instrument.partitionByInstrument(midi)

    # Looping over all the instruments
    for part in s2.parts:
    
        # select elements of only "instrument" type
        if instrument_select in str(part): 
        
            notes_to_parse = part.recurse() 
      
            # finding whether a particular element is note or a chord
            for element in notes_to_parse:
                
                # note
                if isinstance(element, note.Note):
                    notes.append(str(element.pitch))
                
                # chord
                elif isinstance(element, chord.Chord):
                    notes.append('.'.join(str(n) for n in element.normalOrder))

    return np.array(notes)


"""
A helper function for exporting note data to a MIDI file.

Helper code from https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
"""
def convert_to_midi(
    prediction_output: np.ndarray,
    output_filename: str,) -> None:
   
    offset = 0
    output_notes = []

    # create note and chord objects based on the values generated by the model
    for pattern in prediction_output:
        
        # pattern is a chord
        if ('.' in pattern) or pattern.isdigit():
            notes_in_chord = pattern.split('.')
            notes = []
            for current_note in notes_in_chord:
                
                cn=int(current_note)
                new_note = note.Note(cn)
                new_note.storedInstrument = instrument.Piano()
                notes.append(new_note)
                
            new_chord = chord.Chord(notes)
            new_chord.offset = offset
            output_notes.append(new_chord)
            
        # pattern is a note
        else:
            
            new_note = note.Note(pattern)
            new_note.offset = offset
            new_note.storedInstrument = instrument.Piano()
            output_notes.append(new_note)

        # increase offset each iteration so that notes do not stack
        offset += 1
    midi_stream = stream.Stream(output_notes)
    midi_stream.write('midi', fp=output_filename)


"""
A helper function for converting all MIDI (.mid) files in a specified 
path into a 2D ndarray of notes. Basically, it parses training data.

Allows for the specification of frequency for a particular note needed 
for it to be included in the resulting array; setting freq_threshold <= 0
yields all notes included.

Helper code from https://www.analyticsvidhya.com/blog/2020/01/how-to-perform-automatic-music-generation/
"""
def load_midi_files(path: str, instrument: str="Piano", freq_threshold: int=50)-> np.ndarray:
    # read all the filenames
    files = [i for i in os.listdir(path) if i.endswith(".mid")]

    # reading each midi file
    notes_array = np.array([read_midi(path+i, instrument) for i in files])

    # convert from 2D array --> 1D array
    notes = [element for note in notes_array for element in note]

    # computing frequency of each note
    freq_notes = dict(Counter(notes))

    # trimming notes by frequency occurring threshold
    frequent_notes = [note for note, count in freq_notes.items() if count >= freq_threshold]

    # create musical files with only most frequent notes
    trimmed_notes_array = []

    # loop through MIDI notes
    for notes in notes_array:
        temp=[]

        # loop through notes in specific MIDI
        for note in notes:

            # add note if it is included in frequency threshold
            if note in frequent_notes:
                temp.append(note) 

        # add specific MIDI notes to returned array           
        trimmed_notes_array.append(temp)
    
    # return 2D ndarray of trimmed notes
    return np.array(trimmed_notes_array)
    