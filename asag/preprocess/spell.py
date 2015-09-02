# Peter Norvigs spell correct
import re, collections

alphabet = 'abcdefghijklmnopqrstuvwxyz'

def words(text): return re.findall('[a-z]+', text.lower()) 

def train(features):
  model = collections.defaultdict(lambda: 1)
  for f in features:
    model[f] += 1
  return model

def compute_nwords(all_words):
  word_corpus = all_words + words(file('big.txt').read())
  return train(word_corpus)

def edits1(word):
   splits     = [(word[:i], word[i:]) for i in range(len(word) + 1)]
   deletes    = [a + b[1:] for a, b in splits if b]
   transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b)>1]
   replaces   = [a + c + b[1:] for a, b in splits for c in alphabet if b]
   inserts    = [a + c + b     for a, b in splits for c in alphabet]
   return set(deletes + transposes + replaces + inserts)

def known_edits2(word, NWORDS):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words, NWORDS): return set(w for w in words if w in NWORDS)

def correct(word, NWORDS):
    candidates = known([word], NWORDS) or known(edits1(word),NWORDS) or known_edits2(word, NWORDS) or [word]
    return max(candidates, key=NWORDS.get)
