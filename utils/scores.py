import numpy as np
import torch

def pitch_count(sequence, min_note=0, max_note=127):

    pitches = [p for p in sequence if p in range(0, max_note-min_note+1, 2)]
    unique_elements = torch.unique(pitches)
    pc = len(unique_elements)
    return pc


def note_count(sequence, min_note=0, max_note=127):

    pitches = [p for p in sequence if p in range(0, max_note-min_note+1, 2)]
    nc = len(pitches)
    return nc  


def pitch_range(sequence, min_note=0, max_note=127):

    pitches = [p for p in sequence if p in range(0, max_note-min_note+1, 2)]
    pr = np.max(pitches)-np.min(pitches)
    return pr


def average_pitch_interval(sequence, min_note=0, max_note=127):
    
    pitches = [p for p in sequence if p in range(0, max_note-min_note+1, 2)]
    pitches1 = pitches[:-1]
    pitches2 = pitches[1:]
    pitch_diff = pitches2-pitches1
    api = np.mean(pitch_diff)

    return api


def average_inter_onset_interval(sequence, min_note=0, max_note=127, num_time_shifts=10):
    shifts = [s for s in sequence if s in range((max_note-min_note+1), (max_note-min_note+1)+num_time_shifts)]
    aioi = np.mean(shifts)*(100//num_time_shifts)

    return aioi
