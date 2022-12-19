#https://github.com/YatingMusic/remi/blob/master/chord_recognition.py
# REMI utils used to encode the data
import numpy as np
import miditoolkit
import copy

# parameters for input
DEFAULT_VELOCITY_BINS = np.linspace(0, 128, 32+1, dtype=np.int)
DEFAULT_FRACTION = 16
DEFAULT_DURATION_BINS = np.arange(60, 3841, 60, dtype=int)
DEFAULT_TEMPO_INTERVALS = [range(30, 90), range(90, 150), range(150, 210)]

# parameters for output
DEFAULT_RESOLUTION = 480


class MIDIChord(object):
    def __init__(self):
        # define pitch classes
        self.PITCH_CLASSES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        # define chord maps (required)
        self.CHORD_MAPS = {'maj': [0, 4],
                           'min': [0, 3],
                           'dim': [0, 3, 6],
                           'aug': [0, 4, 8],
                           'dom': [0, 4, 7, 10]}
        # define chord insiders (+1)
        self.CHORD_INSIDERS = {'maj': [7],
                               'min': [7],
                               'dim': [9],
                               'aug': [],
                               'dom': []}
        # define chord outsiders (-1)
        self.CHORD_OUTSIDERS_1 = {'maj': [2, 5, 9],
                                  'min': [2, 5, 8],
                                  'dim': [2, 5, 10],
                                  'aug': [2, 5, 9],
                                  'dom': [2, 5, 9]}
        # define chord outsiders (-2)
        self.CHORD_OUTSIDERS_2 = {'maj': [1, 3, 6, 8, 10],
                                  'min': [1, 4, 6, 9, 11],
                                  'dim': [1, 4, 7, 8, 11],
                                  'aug': [1, 3, 6, 7, 10],
                                  'dom': [1, 3, 6, 8, 11]}

    def note2pianoroll(self, notes, max_tick, ticks_per_beat):
        return miditoolkit.pianoroll.parser.notes2pianoroll(
                note_stream_ori=notes,
                max_tick=max_tick,
                ticks_per_beat=ticks_per_beat)

    def sequencing(self, chroma):
        candidates = {}
        for index in range(len(chroma)):
            if chroma[index]:
                root_note = index
                _chroma = np.roll(chroma, -root_note)
                sequence = np.where(_chroma == 1)[0]
                candidates[root_note] = list(sequence)
        return candidates

    def scoring(self, candidates):
        scores = {}
        qualities = {}
        for root_note, sequence in candidates.items():
            if 3 not in sequence and 4 not in sequence:
                scores[root_note] = -100
                qualities[root_note] = 'None'
            elif 3 in sequence and 4 in sequence:
                scores[root_note] = -100
                qualities[root_note] = 'None'
            else:
                # decide quality
                if 3 in sequence:
                    if 6 in sequence:
                        quality = 'dim'
                    else:
                        quality = 'min'
                elif 4 in sequence:
                    if 8 in sequence:
                        quality = 'aug'
                    else:
                        if 7 in sequence and 10 in sequence:
                            quality = 'dom'
                        else:
                            quality = 'maj'
                # decide score
                maps = self.CHORD_MAPS.get(quality)
                _notes = [n for n in sequence if n not in maps]
                score = 0
                for n in _notes:
                    if n in self.CHORD_OUTSIDERS_1.get(quality):
                        score -= 1
                    elif n in self.CHORD_OUTSIDERS_2.get(quality):
                        score -= 2
                    elif n in self.CHORD_INSIDERS.get(quality):
                        score += 1
                scores[root_note] = score
                qualities[root_note] = quality
        return scores, qualities

    def find_chord(self, pianoroll):
        chroma = miditoolkit.pianoroll.utils.tochroma(pianoroll=pianoroll)
        chroma = np.sum(chroma, axis=0)
        chroma = np.array([1 if c else 0 for c in chroma])
        if np.sum(chroma) == 0:
            return 'N', 'N', 'N', 0
        else:
            candidates = self.sequencing(chroma=chroma)
            scores, qualities = self.scoring(candidates=candidates)
            # bass note
            sorted_notes = []
            for i, v in enumerate(np.sum(pianoroll, axis=0)):
                if v > 0:
                    sorted_notes.append(int(i%12))
            bass_note = sorted_notes[0]
            # root note
            __root_note = []
            _max = max(scores.values())
            for _root_note, score in scores.items():
                if score == _max:
                    __root_note.append(_root_note)
            if len(__root_note) == 1:
                root_note = __root_note[0]
            else:
                #TODO: what should i do
                for n in sorted_notes:
                    if n in __root_note:
                        root_note = n
                        break
            # quality
            quality = qualities.get(root_note)
            sequence = candidates.get(root_note)
            # score
            score = scores.get(root_note)
            return self.PITCH_CLASSES[root_note], quality, self.PITCH_CLASSES[bass_note], score

    def greedy(self, candidates, max_tick, min_length):
        chords = []
        # start from 0
        start_tick = 0
        while start_tick < max_tick:
            _candidates = candidates.get(start_tick)
            _candidates = sorted(_candidates.items(), key=lambda x: (x[1][-1], x[0]))
            # choose
            end_tick, (root_note, quality, bass_note, _) = _candidates[-1]
            if root_note == bass_note:
                chord = '{}:{}'.format(root_note, quality)
            else:
                chord = '{}:{}/{}'.format(root_note, quality, bass_note)
            chords.append([start_tick, end_tick, chord])
            start_tick = end_tick
        # remove :None
        temp = chords
        while ':None' in temp[0][-1]:
            try:
                temp[1][0] = temp[0][0]
                del temp[0]
            except:
                print('NO CHORD')
                return []
        temp2 = []
        for chord in temp:
            if ':None' not in chord[-1]:
                temp2.append(chord)
            else:
                temp2[-1][1] = chord[1]
        return temp2

    def extract(self, notes):
        # read
        max_tick = max([n.end for n in notes])
        ticks_per_beat = 480
        pianoroll = self.note2pianoroll(
            notes=notes, 
            max_tick=max_tick, 
            ticks_per_beat=ticks_per_beat)
        # get lots of candidates
        candidates = {}
        # the shortest: 2 beat, longest: 4 beat
        for interval in [4, 2]:
            for start_tick in range(0, max_tick, ticks_per_beat):
                # set target pianoroll
                end_tick = int(ticks_per_beat * interval + start_tick)
                if end_tick > max_tick:
                    end_tick = max_tick
                _pianoroll = pianoroll[start_tick:end_tick, :]
                # find chord
                root_note, quality, bass_note, score = self.find_chord(pianoroll=_pianoroll)
                # save
                if start_tick not in candidates:
                    candidates[start_tick] = {}
                    candidates[start_tick][end_tick] = (root_note, quality, bass_note, score)
                else:
                    if end_tick not in candidates[start_tick]:
                        candidates[start_tick][end_tick] = (root_note, quality, bass_note, score)
        # greedy
        chords = self.greedy(candidates=candidates, 
                             max_tick=max_tick, 
                             min_length=ticks_per_beat)
        return chords


# define "Item" for general storage
class Item(object):
    def __init__(self, name, start, end, velocity, pitch):
        self.name = name
        self.start = start
        self.end = end
        self.velocity = velocity
        self.pitch = pitch

    def __repr__(self):
        return 'Item(name={}, start={}, end={}, velocity={}, pitch={})'.format(
            self.name, self.start, self.end, self.velocity, self.pitch)

# read notes and tempo changes from midi (assume there is only one track)
def read_items(file_path):
    midi_obj = miditoolkit.midi.parser.MidiFile(file_path)
    # note
    note_items = []
    notes = midi_obj.instruments[0].notes
    notes.sort(key=lambda x: (x.start, x.pitch))
    for note in notes:
        note_items.append(Item(
            name='Note', 
            start=note.start, 
            end=note.end, 
            velocity=note.velocity, 
            pitch=note.pitch))
    note_items.sort(key=lambda x: x.start)
    # tempo
    tempo_items = []
    for tempo in midi_obj.tempo_changes:
        tempo_items.append(Item(
            name='Tempo',
            start=tempo.time,
            end=None,
            velocity=None,
            pitch=int(tempo.tempo)))
    tempo_items.sort(key=lambda x: x.start)
    # expand to all beat
    max_tick = tempo_items[-1].start
    existing_ticks = {item.start: item.pitch for item in tempo_items}
    wanted_ticks = np.arange(0, max_tick+1, DEFAULT_RESOLUTION)
    output = []
    for tick in wanted_ticks:
        if tick in existing_ticks:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=existing_ticks[tick]))
        else:
            output.append(Item(
                name='Tempo',
                start=tick,
                end=None,
                velocity=None,
                pitch=output[-1].pitch))
    tempo_items = output
    return note_items, tempo_items

# quantize items
def quantize_items(items, ticks=120):
    # grid
    grids = np.arange(0, items[-1].start, ticks, dtype=int)
    # process
    for item in items:
        index = np.argmin(abs(grids - item.start))
        shift = grids[index] - item.start
        item.start += shift
        item.end += shift
    return items      

# extract chord
def extract_chords(items):
    method = MIDIChord()
    chords = method.extract(notes=items)
    output = []
    for chord in chords:
        output.append(Item(
            name='Chord',
            start=chord[0],
            end=chord[1],
            velocity=None,
            pitch=chord[2].split('/')[0]))
    return output

# group items
def group_items(items, max_time, ticks_per_bar=DEFAULT_RESOLUTION*4):
    items.sort(key=lambda x: x.start)
    downbeats = np.arange(0, max_time+ticks_per_bar, ticks_per_bar)
    groups = []
    for db1, db2 in zip(downbeats[:-1], downbeats[1:]):
        insiders = []
        for item in items:
            if (item.start >= db1) and (item.start < db2):
                insiders.append(item)
        overall = [db1] + insiders + [db2]
        groups.append(overall)
    return groups

# define "Event" for event storage
class Event(object):
    def __init__(self, name, time, value, text):
        self.name = name
        self.time = time
        self.value = value
        self.text = text

    def __repr__(self):
        return 'Event(name={}, time={}, value={}, text={})'.format(
            self.name, self.time, self.value, self.text)

# item to event
def item2event(groups):
    events = []
    n_downbeat = 0
    for i in range(len(groups)):
        if 'Note' not in [item.name for item in groups[i][1:-1]]:
            continue
        bar_st, bar_et = groups[i][0], groups[i][-1]
        n_downbeat += 1
        events.append(Event(
            name='Bar',
            time=None, 
            value=None,
            text='{}'.format(n_downbeat)))
        for item in groups[i][1:-1]:
            # position
            flags = np.linspace(bar_st, bar_et, DEFAULT_FRACTION, endpoint=False)
            index = np.argmin(abs(flags-item.start))
            events.append(Event(
                name='Position', 
                time=item.start,
                value='{}/{}'.format(index+1, DEFAULT_FRACTION),
                text='{}'.format(item.start)))
            if item.name == 'Note':
                # velocity
                velocity_index = np.searchsorted(
                    DEFAULT_VELOCITY_BINS, 
                    item.velocity, 
                    side='right') - 1
                events.append(Event(
                    name='Note Velocity',
                    time=item.start, 
                    value=velocity_index,
                    text='{}/{}'.format(item.velocity, DEFAULT_VELOCITY_BINS[velocity_index])))
                # pitch
                events.append(Event(
                    name='Note On',
                    time=item.start, 
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
                # duration
                duration = item.end - item.start
                index = np.argmin(abs(DEFAULT_DURATION_BINS-duration))
                events.append(Event(
                    name='Note Duration',
                    time=item.start,
                    value=index,
                    text='{}/{}'.format(duration, DEFAULT_DURATION_BINS[index])))
            elif item.name == 'Chord':
                events.append(Event(
                    name='Chord', 
                    time=item.start,
                    value=item.pitch,
                    text='{}'.format(item.pitch)))
            elif item.name == 'Tempo':
                tempo = item.pitch
                if tempo in DEFAULT_TEMPO_INTERVALS[0]:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[0].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[1]:
                    tempo_style = Event('Tempo Class', item.start, 'mid', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[1].start, None)
                elif tempo in DEFAULT_TEMPO_INTERVALS[2]:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 
                        tempo-DEFAULT_TEMPO_INTERVALS[2].start, None)
                elif tempo < DEFAULT_TEMPO_INTERVALS[0].start:
                    tempo_style = Event('Tempo Class', item.start, 'slow', None)
                    tempo_value = Event('Tempo Value', item.start, 0, None)
                elif tempo > DEFAULT_TEMPO_INTERVALS[2].stop:
                    tempo_style = Event('Tempo Class', item.start, 'fast', None)
                    tempo_value = Event('Tempo Value', item.start, 59, None)
                events.append(tempo_style)
                events.append(tempo_value)     
    return events

#############################################################################################
# WRITE MIDI
#############################################################################################
def word_to_event(words, word2event):
    events = []
    for word in words:
        event_name, event_value = word2event.get(word).split('_')
        events.append(Event(event_name, None, event_value, None))
    return events

def write_midi(words, word2event, output_path, prompt_path=None):
    events = word_to_event(words, word2event)
    # get downbeat and note (no time)
    temp_notes = []
    temp_chords = []
    temp_tempos = []
    for i in range(len(events)-3):
        if events[i].name == 'Bar' and i > 0:
            temp_notes.append('Bar')
            temp_chords.append('Bar')
            temp_tempos.append('Bar')
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Note Velocity' and \
            events[i+2].name == 'Note On' and \
            events[i+3].name == 'Note Duration':
            # start time and end time from position
            position = int(events[i].value.split('/')[0]) - 1
            # velocity
            index = int(events[i+1].value)
            velocity = int(DEFAULT_VELOCITY_BINS[index])
            # pitch
            pitch = int(events[i+2].value)
            # duration
            index = int(events[i+3].value)
            duration = DEFAULT_DURATION_BINS[index]
            # adding
            temp_notes.append([position, velocity, pitch, duration])
        elif events[i].name == 'Position' and events[i+1].name == 'Chord':
            position = int(events[i].value.split('/')[0]) - 1
            temp_chords.append([position, events[i+1].value])
        elif events[i].name == 'Position' and \
            events[i+1].name == 'Tempo Class' and \
            events[i+2].name == 'Tempo Value':
            position = int(events[i].value.split('/')[0]) - 1
            if events[i+1].value == 'slow':
                tempo = DEFAULT_TEMPO_INTERVALS[0].start + int(events[i+2].value)
            elif events[i+1].value == 'mid':
                tempo = DEFAULT_TEMPO_INTERVALS[1].start + int(events[i+2].value)
            elif events[i+1].value == 'fast':
                tempo = DEFAULT_TEMPO_INTERVALS[2].start + int(events[i+2].value)
            temp_tempos.append([position, tempo])
    # get specific time for notes
    ticks_per_beat = DEFAULT_RESOLUTION
    ticks_per_bar = DEFAULT_RESOLUTION * 4 # assume 4/4
    notes = []
    current_bar = 0
    for note in temp_notes:
        if note == 'Bar':
            current_bar += 1
        else:
            position, velocity, pitch, duration = note
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            # duration (end time)
            et = st + duration
            notes.append(miditoolkit.Note(velocity, pitch, st, et))
    # get specific time for chords
    if len(temp_chords) > 0:
        chords = []
        current_bar = 0
        for chord in temp_chords:
            if chord == 'Bar':
                current_bar += 1
            else:
                position, value = chord
                # position (start time)
                current_bar_st = current_bar * ticks_per_bar
                current_bar_et = (current_bar + 1) * ticks_per_bar
                flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
                st = flags[position]
                chords.append([st, value])
    # get specific time for tempos
    tempos = []
    current_bar = 0
    for tempo in temp_tempos:
        if tempo == 'Bar':
            current_bar += 1
        else:
            position, value = tempo
            # position (start time)
            current_bar_st = current_bar * ticks_per_bar
            current_bar_et = (current_bar + 1) * ticks_per_bar
            flags = np.linspace(current_bar_st, current_bar_et, DEFAULT_FRACTION, endpoint=False, dtype=int)
            st = flags[position]
            tempos.append([int(st), value])
    # write
    if prompt_path:
        midi = miditoolkit.midi.parser.MidiFile(prompt_path)
        #
        last_time = DEFAULT_RESOLUTION * 4 * 4
        # note shift
        for note in notes:
            note.start += last_time
            note.end += last_time
        midi.instruments[0].notes.extend(notes)
        # tempo changes
        temp_tempos = []
        for tempo in midi.tempo_changes:
            if tempo.time < DEFAULT_RESOLUTION*4*4:
                temp_tempos.append(tempo)
            else:
                break
        for st, bpm in tempos:
            st += last_time
            temp_tempos.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = temp_tempos
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]+last_time))
    else:
        midi = miditoolkit.midi.parser.MidiFile()
        midi.ticks_per_beat = DEFAULT_RESOLUTION
        # write instrument
        inst = miditoolkit.midi.containers.Instrument(0, is_drum=False)
        inst.notes = notes
        midi.instruments.append(inst)
        # write tempo
        tempo_changes = []
        for st, bpm in tempos:
            tempo_changes.append(miditoolkit.midi.containers.TempoChange(bpm, st))
        midi.tempo_changes = tempo_changes
        # write chord into marker
        if len(temp_chords) > 0:
            for c in chords:
                midi.markers.append(
                    miditoolkit.midi.containers.Marker(text=c[1], time=c[0]))
    # write
    midi.dump(output_path)