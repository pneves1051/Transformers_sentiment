#Carregar e criar lista de objetos de stream
for song in songList:
    score = converter.parse(song)
    k = score.analyze('key')
    i = interval.Interval(k.tonic, pitch.Pitch('C'))
    score = score.transpose(i)
    originalScores.append(score)      

from music21 import instrument

def monophonic(stream):
    try:
        length = len(instrument.partitionByInstrument(stream).parts)
    except:
        length = 0
    return length == 1


#Juntar notas para formar acordes
originalScores = [song.chordify() for song in originalScores]

# obj0, obj1, obj2 are created here...

# Saving the objects:
#with open('/content/drive/My Drive/Colab Notebooks/originalAll2.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
#    pickle.dump(originalScores, f)

# Getting back the objects:
with open('data/originalAll.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    originalScores = pickle.load(f)

from music21 import note, chord

#Define listas de listas vazias
originalChords = [[] for _ in originalScores]
originalDurations = [[] for _ in originalScores]
originalKeys = []

#Extrai notas, acordes, durações e tonalidades
for i, song in enumerate(originalScores):
    originalKeys.append(str(song.analyze('key')))
    for element in song:
        if isinstance(element, note.Note):
            originalChords[i].append(element.pitch)
            originalDurations[i].append(element.duration.quarterLength)
        elif isinstance(element, chord.Chord):
            originalChords[i].append('.'.join(str(n) for n in element.pitches))
            originalDurations[i].append(element.duration.quarterLength) 
        elif isinstance(element, note.Rest):
            originalChords[i].append("<rest>")
            originalDurations[i].append(element.duration.quarterLength)   

# obj0, obj1, obj2 are created here...

# Saving the objects:
with open('data/originalChords.pkl', 'wb') as f:
    pickle.dump(originalChords, f)
with open('data/originalDurations.pkl', 'wb') as f:
    pickle.dump(originalDurations, f)
with open('data/originalKeys.pkl', 'wb') as f:
    pickle.dump(originalKeys, f)

## AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA

# Getting back the objects:
with open('data/originalChords.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    originalChords = pickle.load(f)
with open('data/originalDurations.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    originalDurations = pickle.load(f)
with open('data/originalKeys.pkl', 'rb') as f:  # Python 3: open(..., 'rb')
    originalKeys = pickle.load(f)

#Concatenação de todos as listas de acordes e durações para um único vetor
#Tamanho das sequências de treinamento
seq_len = 100

music_tokenized = [[c + '_' + str(d) for c,d in zip(s,t)] for s,t in zip(originalChords, originalDurations)]
joined_music_tokenized = []
for i in range(len(music_tokenized[:200])):
  joined_music_tokenized += music_tokenized[i]

clean_music_tokenized = [song.copy() for song in music_tokenized]

#Música tokenizada, mas com a substituição dos tokens com menor frequência por '<unk>'
counter = Counter(joined_music_tokenized)
for i, song in enumerate(clean_music_tokenized):
  for j, element in enumerate(song):
    if(counter[element] < 3): clean_music_tokenized[i][j] = '<unk>'

music_tokenized[201]

#Acordes e durações únicas
unique = np.unique([element for song in clean_music_tokenized for element in song])

token_to_int = {c : i for i, c in enumerate(unique)}
int_to_token = {i : c for i, c in enumerate(unique)}

#Imprime o número de notas e acordes únicos
print(len(unique))

# Create training examples / targets
X = []
Y = []

for song in clean_music_tokenized:
  for i in range(len(song)-seq_len):
    X.append([token_to_int[c] for c in song[i: i+seq_len]])
    Y.append([token_to_int[c] for c in song[i+1 : i+1+seq_len]]) 

#Definição dos tamanhos dos datasets de treino e validação
train_size = round(0.9*(len(X)))
val_size = round(0.1*(len(X)))

assert train_size+val_size == len(X)
