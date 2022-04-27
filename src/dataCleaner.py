from typing import Tuple
import numpy as np

"""
Generates an x, y / input, output sequence for a model based on the 2D ndarray of
notes and timestep count passed as input. 

The output (label) for an array of notes over "x" amount of timesteps is the note immediately 
after the previous "x" notes.
"""
def generate_io_sequence(notes_array: np.ndarray, timestep_count: int) -> Tuple[np.ndarray, np.ndarray]:
    
    # init i/o arrays
    x = []
    y = []

    # loop through notes in training data
    for note in notes_array:

        # loop through notes in chunks of timestep_count
        for i in range(0, len(note) - timestep_count, 1):
            
            # preparing input and output sequences; input is
            # a 2D array where output is 1D
            input = note[i:i + timestep_count] # this is an array of notes from i to i+timestep_count
            output = note[i + timestep_count] # this is the single note after the timestep block
            
            # 2D array of notes input --> next note after input notes from example
            x.append(input)
            y.append(output)
            
    return np.array(x), np.array(y)

"""
Maps x, y / input, output sequence of notes to ints for network input. Returns
the 2D ndarray input sequence of ints, the 1D ndarray output sequence of ints,
and a dictionary mapping from int --> note.
"""
def map_notes_to_ints(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, dict]:
    
    # generate unique, unified input/output notes
    # ravel() changes a multi-dimensional array into a contiguous/flattened array
    unique_notes = list(set(np.append(x.ravel(), y, 0)))
    # unique_x = list(set(x.ravel()))
    # unique_y = list(set(y))
    

    # map unique notes to integers
    unique_note_to_int = dict((note, number) for number, note in enumerate(unique_notes))
    # x_note_to_int = dict((note, number) for number, note in enumerate(unique_x))
    # y_note_to_int = dict((note, number) for number, note in enumerate(unique_y))

    # prepare input (x) sequence of notes mapped to ints
    x_seq=[]
    for notes in x:
        temp=[]
        for note in notes:
            # assigning unique integer to every note
            temp.append(unique_note_to_int[note])
            # temp.append(x_note_to_int[note])
        x_seq.append(temp)
        
    x_seq = np.array(x_seq)

    # prepare output (y) sequence of notes mapped to ints
    y_seq = np.array([unique_note_to_int[i] for i in y])
    # y_seq = np.array([y_note_to_int[i] for i in y])

    # generate int --> note map for return
    unique_int_to_note = dict((number, note) for number, note in enumerate(unique_notes))

    return x_seq, y_seq, unique_int_to_note