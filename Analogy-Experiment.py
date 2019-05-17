from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors
import numpy as np

wordVectors = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True, limit=1000 )
wordVectors.init_sims(replace=True)
wordVectors.save('wordVectors')
wordVectors = KeyedVectors.load('wordVectors', mmap='r')


relations = ['capital-world', 'currency', 'city-in-state', 'family', 'gram1-adjective-to-adverb', 'gram2-opposite', 'gram3-comparative', 'gram6-nationality-adjective']
relationDict = dict((name, []) for name in relations)

lines = open("word-test.v1.txt", 'r').read().split(":")
# lines.pop(0)
lines.remove(lines[0])

for line in lines:
    line = line.split("\n")
    at = line[0].split(" ")[1]
    for key, val in relationDict.items():
        if at == key:
            # val +=[line[i+1] for i in range(len(line)-1)]
            for i in range(len(line) - 1):
                val.append(line[i+1])



def cosineSimilarity(v1, v2): #this function calculate cosine similarity between two vectors
    dot = np.dot(v1,v2)
    normV1 = np.linalg.norm(v1)
    normv2 = np.linalg.norm(v2)
    return dot / (normV1 + normv2)

def vectored(words): #this function finds vector
    Vec = wordVectors[words[1]] -\
          wordVectors[words[0]] + \
          wordVectors[words[2]]
    return Vec

AnsweredCorectly = 0
howManyQuestions = 0
for key, value in relationDict.items():
    if len(value) != 0:
        value.remove(value[len(value)-1])

    word = ""
    for word_pairs in value:
        result = 0
        try:
            word_pair = word_pairs.split(" ")
            VecRes = vectored(word_pair)
        except Exception as e:
            howManyQuestions += 1
            continue

        for WordControl in wordVectors.vocab:
            sim = cosineSimilarity(VecRes, wordVectors[WordControl])
            if sim > result:
                result = sim
                word = WordControl
        if word == word_pair[len(word_pair) - 1]:
            AnsweredCorectly += 1
            howManyQuestions += 1



        # print(word_pair, "////", word, "////", (AnsweredCorectly / howManyQuestions) * 100)

Evaluation = 100 * (AnsweredCorectly / howManyQuestions)
print('Testing accuracy for analogy experiment is ', Evaluation)
