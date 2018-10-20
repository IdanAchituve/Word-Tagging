import tagger2 as t2
import dynet as dy
import numpy as np


# Read data function
def read_test_data(fname):
    data = [] # list of lists. each list is a sentence
    sentence = [] # list of tuples: (word in sentence, label)
    vocab = set() # a set of all unique words in the training set
    for line in file(fname):
        if line == "\n": # new sentence after line break
            data.append(sentence)
            sentence = []
            continue
        word = line.strip()
        sentence.append(word)
        vocab.add(word)
    return data, vocab


# the __call__ method applies the network to an input
def forward_calc(input, F2I, params, e, activate_sub_word):

    num_params = len(params)  # get the length of params vector
    lookup = e

    if activate_sub_word == False:  # for runs in which sub words are not taken into account
        emb_vectors = [lookup[F2I.get(i)] for i in input]  # get embedding of the words
    else:  # sum embedding of word, suffix and prefix for words that allow it
        emb_vectors = []
        for word in input:
            # get the word alone if len<=3 or word=start/end/unk
            if len(word) <= 3 or (word in ["<s>", "</s>", "UUUNKKK"]):
                emb_vectors.append(lookup[F2I.get(word)])
            else:
                pref = False
                suff = False
                # check if prefix exist in F2I. relevant for test/dev sets
                if F2I.has_key(word[:3]):
                    prefix_embd = lookup[F2I.get(word[:3])]
                    pref = True
                # check if suffix exist in F2I. relevant for test/dev sets
                if F2I.has_key(word[-3:]):
                    suffix_embd = lookup[F2I.get(word[-3:])]
                    suff = True

                word_embd = lookup[F2I.get(word)]

                # sum vectors of word with existing prefix/suffix
                if pref and suff:
                    sum_embd = dy.esum([prefix_embd, suffix_embd, word_embd])
                elif pref and suff == False:
                    sum_embd = dy.esum([prefix_embd, word_embd])
                elif suff and pref == False:
                    sum_embd = dy.esum([suffix_embd, word_embd])
                else:
                    sum_embd = dy.esum([word_embd])

                emb_vectors.append(sum_embd)

    net_input = dy.concatenate(emb_vectors)

    for i in xrange(0, num_params - 2,2):  # calculate the activation of each subsequent layers and apply the bernoulli mask
        W = dy.parameter(params[i])  # from parameters to expressions
        b = dy.parameter(params[i + 1])
        if i == 0:  # first layer
            activation = dy.tanh((W * net_input) + b)
        else:  # other layers
            activation = dy.tanh((W * activation) + b)

    W = dy.parameter(params[num_params - 2])  # from parameters to expressions
    b = dy.parameter(params[num_params - 1])
    net_output = dy.softmax((W * activation) + b)  # apply sfotmax on last layer
    return net_output


# return the forward pass output
def create_network_return_best(input, F2I, params, e, activate_sub_word):
    dy.renew_cg()
    out = forward_calc(input, F2I, params, e, activate_sub_word)
    return np.argmax(out.npvalue())


# get prediction on test set
def prediction(dataset, L2I, F2I, params, e, activate_sub_word):
    predictions = [] # save all predictions according to sentences
    for sentence in dataset:
        sentence = ['<s>', '<s>'] + sentence + ['</s>', '</s>']  # pad each sentence with BOS prefix and EOS suffix
        word_count = 0
        sentence_pred = []  # save actual class of prediction
        for word in sentence[2:(len(sentence) - 2)]:
            first_word = sentence[word_count]
            sec_word = sentence[word_count + 1]
            forth_word = sentence[word_count + 3]
            fifth_word = sentence[word_count + 4]

            context = [first_word, sec_word, word, forth_word, fifth_word]
            pred = create_network_return_best(context, F2I, params, e, activate_sub_word)

            # find actual label
            for label, num in L2I.items():
                if num == pred:
                    sentence_pred.append((word, label))
                    break
            word_count += 1
        predictions.append(sentence_pred)

    return predictions


def get_sub_words(word):
    words = []
    if len(word)<=3:
        words.append(word)
    else:
        prefix = word[:3]
        suffix = word[-3:]
        words = [word] + [prefix] + [suffix]
    return words


def run_test(train_path, test_path, vocab_path, test_output_path,
             hidden_layer, embedding_size, context_size, saved_model_path, activate_sub_word=False):

    from time import gmtime, strftime

    print strftime("%Y-%m-%d %H:%M:%S", gmtime())

    TRAIN, TRAIN_WORDS, TRAIN_LABELS = t2.read_data(train_path)  # get training set
    TEST, VOCAB_TEST = read_test_data(test_path)  # get dev set
    VOCAB = t2.read_data(vocab_path, "vocab")[0] # get vocabulary from pre-trained words

    # if the feature for taking into account sub words is activated
    if activate_sub_word == True:
        for word in TRAIN_WORDS.keys():
            if word not in ["<s>", "</s>", "UUUNKKK"]:
                words_from_word = get_sub_words(word)
                for part_word in words_from_word:
                    if TRAIN_WORDS.has_key(part_word) == False:
                        TRAIN_WORDS[part_word] = 1

    # if a word appeared in train but not in the vocab add it to vocab as well
    words_not_in_vocab = []
    for word in TRAIN_WORDS.keys():
        if word not in VOCAB:
            words_not_in_vocab.append(word)

    # add the train words to VOCAB
    for word in words_not_in_vocab:
        VOCAB.append(word)

    L2I = {l: i for i, l in enumerate(list(sorted(TRAIN_LABELS)))}  # enumerate the labels as 0,1,2,...
    F2I = {f: i for i, f in enumerate(list(sorted(VOCAB)))}  # enumerate the vocabulary as 0,1,2,...

    # change TEST dataset by replacing words that did not appear in the vocab with <UNK>
    TEST_ORIG = list(TEST)
    TEST = []
    for sentence in TEST_ORIG:
        new_sentence = []
        for word in sentence:
            word = word.lower()
            if word in VOCAB:
                new_sentence.append(word)
            else:
                new_sentence.append("UUUNKKK")
        TEST.append(new_sentence)

    m = dy.ParameterCollection()  # create parameter collection

    # define parameters
    pW1 = m.add_parameters((hidden_layer, embedding_size*context_size))
    pb1 = m.add_parameters(hidden_layer)
    pW2 = m.add_parameters((len(TRAIN_LABELS), hidden_layer))
    pb2 = m.add_parameters(len(TRAIN_LABELS))
    params=[pW1,pb1,pW2,pb2]
    e = m.add_lookup_parameters((len(VOCAB), embedding_size)) #
    # load the parameters
    m.populate(saved_model_path)

    predictions = prediction(TEST, L2I, F2I, params, e, activate_sub_word)

    # write predictions to file
    file = open(test_output_path, "w")
    for idx, sentence in enumerate(predictions):
        word_idx = 0
        for word,label in sentence:
            orig_word = TEST_ORIG[idx][word_idx]
            file.write(orig_word + " " + label + "\n")
            word_idx += 1
        file.write("\n")


if __name__ == '__main__':

    mission = "NER"
    train_path = "../ner/train"
    test_path = "../ner/test"
    vocab_path = "../pre_trained/vocab.txt"
    test_output_path = "/home/idan/Desktop/studies/deep_learning/ass2/ass2/To_Submit/resubmit/test4.ner"
    hidden_layer = 170
    embedding_size = 50  # embedding vector size
    context_size = 5
    saved_model_path = "/home/idan/Desktop/studies/deep_learning/ass2/ass2/tagger3/saved_models/model_NER_pre_trained_sub_words_0.7935"  # saving model path
    activate_sub_word = True
    run_test(train_path, test_path, vocab_path, test_output_path, hidden_layer, embedding_size, context_size, saved_model_path, activate_sub_word)
