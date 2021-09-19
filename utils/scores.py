import math
import numpy as np
import torch
import muspy
from collections import defaultdict

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


def calculate_scores(midi_file, scores_to_calculate=['pitch_range', 'number_pitch_classes', 'polyphony']):
    scores = defaultdict(list)
    midi_obj = muspy.read_midi(midi_file)
    for score in scores_to_calculate:
        if score == 'pitch_range':
            scores['pitch_range'] = muspy.pitch_range(midi_obj)
        elif score == 'number_pitch_classes':
            scores['number_pitch_classes'] = muspy.n_pitch_classes_used(midi_obj)
        elif score == 'polyphony':
            scores['polyphony']  = muspy.polyphony(midi_obj)
        else:
            print('Score not found.')
    return scores

def calculate_scores_multi(midi_file_list, scores_to_calculate=['pitch_range', 'number_pitch_classes', 'polyphony']):
    scores = []
    for midi_file in midi_file_list:
        scores.append(calculate_scores(midi_file, scores_to_calculate))
    
    avg_scores = {s_name: np.mean([score[s_name] for score in scores], dtype=np.float32) for s_name in scores_to_calculate}

    return scores, avg_scores