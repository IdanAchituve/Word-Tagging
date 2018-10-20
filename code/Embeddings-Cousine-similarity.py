import numpy as np
import math
from collections import Counter

# Read data function
def read_data(fname):
    data = [] # list of lists. each list is a sentence
    for line in file(fname):
        word = line.strip("\n")
        data.append(word)
    return data


def cousine_distance(vec1, vec2):
    return vec1.dot(vec2)/math.sqrt(vec1.dot(vec1)*vec2.dot(vec2))

def most_similar(word, k, vocab, vectors):

    word_vec_ind = vocab.index(word) # get word index
    word_vec = vectors[word_vec_ind] # get word vector
    most_similar_words = {} # save the similarity of all words
    for idx, vec in enumerate(vectors):
        if idx == word_vec_ind: # no need to compare the word to itself
            continue
        sim = cousine_distance(vec,word_vec)
        most_similar_words[vocab[idx]] = sim

    d_counter = Counter(most_similar_words) # sort the dictionary according to value
    return d_counter.most_common(k)

def find_similar_words(vocab_path, vectors_path, out_path, words, k):

    vocab = read_data(vocab_path)
    vectors = np.loadtxt(vectors_path)

    similar_words = []
    for w in words:
        similar_words.append(most_similar(w, k, vocab, vectors))

    # write predictions to file
    file = open(out_path, "w")
    for idx, k_similar_words in enumerate(similar_words):
        file.write(words[idx] + ": ")
        for list_item in k_similar_words:
            word = list_item[0]
            similairty = str(list_item[1])
            file.write("(" + word + ", " + similairty + ")  ")
        file.write("\n")

if __name__ == '__main__':

    vocab_path = "../pre_trained/vocab.txt"
    vectors_path = "../pre_trained/wordVectors.txt"
    out_path = "./similar_words"
    words = ["dog","england","john","explode","office"]
    k = 5

    find_similar_words(vocab_path, vectors_path, out_path, words, k)