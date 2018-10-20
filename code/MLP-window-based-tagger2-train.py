import dynet_config
import random
import numpy as np

dynet_config.set(random_seed=2, mem=2048, weight_decay=0.00000001) # configuration for dynet #weight_decay=0.0000001
import dynet as dy


# Read data function
def read_data(fname, file_type = "data"):
    data = []  # list of lists. each list is a sentence
    sentence = []  # list of tuples: (word in sentence, label)
    words = {}  # a set of all unique words in the training set
    labels = set()  # a set of all unique labels in the training set
    for line in file(fname):
        if file_type == "data":
            if line == "\n":  # new sentence after line break
                data.append(sentence)
                sentence = []
                continue
            word, label = line.strip().split()
            word = word.lower()
            sentence.append((word, label))
            if words.has_key(word):
                words[word] += 1
            else:
                words[word] = 1 # add word to set if not exist
            labels.add(label) # add label to set if not exist
        else:
            word = line.strip("\n")
            data.append(word)

    return data, words, labels


# create a class encapsulating the network
class TagNetwork(object):
    # add parameters to the parameter collection
    def __init__(self, pc, vocab_size, dim_embedding, dim_params, pre_trained_vec):

        self.params = []
        for first_layer, sec_layer in zip(dim_params, dim_params[1:]):
            pW = pc.add_parameters((sec_layer, first_layer), init=dy.GlorotInitializer())
            pb = pc.add_parameters(sec_layer)
            self.params.append(pW)  # pW with Xavier initialization
            self.params.append(pb)  # pb

        self.lookup = pc.add_lookup_parameters((vocab_size, dim_embedding))  # embedding matrix. dimensions: vocab_size X vector_size
        self.lookup.init_from_array(pre_trained_vec)

    # the __call__ method applies the network to an input
    def __call__(self, context, F2I, only_train_words, dropout_rate = 1.0, activate_sub_word = False, stop_updating_lookup = False):


        num_params = len(self.params)  # get the length of params vector
        lookup = self.lookup

        # if the user choose to continue learning the pre-trained words
        if not stop_updating_lookup:
            # if sub word feature is not activated
            if not activate_sub_word:
                emb_vectors = [lookup[F2I.get(i)] for i in context]  # get embedding of the words
            else:# sum embedding of word, suffix and prefix for words that allow it
                emb_vectors = []
                for word in context:
                    # get the word alone if len<=3 or word=start/end/unk
                    if len(word)<=3 or (word in ["<s>", "</s>", "UUUNKKK"]):
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

        # if the user choose to stop learning the pre-trained words
        if stop_updating_lookup:
            # if sub word feature is not activated
            if not activate_sub_word:
                emb_vectors = []
                for word in context:
                    # if it's a word from the corpus continue training it
                    if word in only_train_words:
                        emb_vectors.append(lookup[F2I.get(word)])
                    # if it's a word from the pre-train stop training it
                    else:
                        emb_vectors.append(dy.nobackprop(lookup[F2I.get(word)]))
            else:  # sum embedding of word, suffix and prefix for words that allow it
                emb_vectors = []
                for word in context:
                    # get the word alone if len<=3 or word=start/end/unk
                    if len(word) <= 3 or (word in ["<s>", "</s>", "UUUNKKK"]):
                        # if it's a word from the corpus continue training it
                        if word in only_train_words:
                            emb_vectors.append(lookup[F2I.get(word)])
                        # if it's a word from the pre-train stop training it
                        else:
                            emb_vectors.append(dy.nobackprop(lookup[F2I.get(word)]))
                    else:
                        pref = False
                        suff = False
                        # check if prefix exist in F2I. relevant for test/dev sets
                        if F2I.has_key(word[:3]):
                            # if it's a word from the corpus continue training it
                            if word[:3] in only_train_words:
                                prefix_embd = lookup[F2I.get(word[:3])]
                            # if it's a word from the pre-train stop training it
                            else:
                                prefix_embd = dy.nobackprop(lookup[F2I.get(word[:3])])
                            pref = True
                        # check if suffix exist in F2I. relevant for test/dev sets
                        if F2I.has_key(word[-3:]):
                            # if it's a word from the corpus continue training it
                            if word[-3:] in only_train_words:
                                suffix_embd = lookup[F2I.get(word[-3:])]
                            # if it's a word from the pre-train stop training it
                            else:
                                suffix_embd = dy.nobackprop(lookup[F2I.get(word[-3:])])
                            suff = True

                        # if it's a word from the corpus continue training it
                        if word in only_train_words:
                            word_embd = lookup[F2I.get(word)]
                        # if it's a word from the pre-train stop training it
                        else:
                            word_embd = dy.nobackprop(lookup[F2I.get(word)])


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

        for i in xrange(0, num_params - 2, 2):  # calculate the activation of each subsequent layers and apply the bernoulli mask
            W = dy.parameter(self.params[i])  # from parameters to expressions
            b = dy.parameter(self.params[i + 1])
            if i == 0:  # first layer
                activation = dy.tanh((W * net_input) + b)
            else:  # other layers
                activation = dy.tanh((W * activation) + b)

            if dropout_rate != 1.0:
                activation = dy.dropout(activation, dropout_rate)

        W = dy.parameter(self.params[num_params - 2])  # from parameters to expressions
        b = dy.parameter(self.params[num_params - 1])
        net_output = dy.softmax((W * activation) + b)  # apply sfotmax on last layer
        return net_output

    # compute loss during training
    def create_network_return_loss(self, input, expected_output, L2I, F2I, only_train_words, dropout_rate=1.0, activate_sub_word = False, stop_updating_lookup = False):
        dy.renew_cg()
        out = self(input, F2I, only_train_words, dropout_rate, activate_sub_word, stop_updating_lookup)
        loss = -dy.log(dy.pick(out, L2I.get(expected_output)))
        return loss

    # return the loss and prediction on unseen word in context
    def create_network_return_best(self, input, F2I, L2I, expected_output, activate_sub_word = False):
        dy.renew_cg()
        out = self(input, F2I, [], 1.0, activate_sub_word, False)
        loss = -dy.log(dy.pick(out, L2I.get(expected_output)))
        return np.argmax(out.npvalue()), loss

    def create_network_return_accuracy(self, dataset, L2I, F2I, mission, activate_sub_word=False):

        good = bad = 0.0
        cum_loss = 0.0
        dev_words = 0 # count number of dev words

        for sentence in dataset:
            sentence = [('<s>', 'dummy'), ('<s>', 'dummy')] + sentence + [('</s>', 'dummy'), ('</s>', 'dummy')]  # pad each sentence with BOS prefix and EOS suffix
            word_count = 0
            for word, label in sentence[2:(len(sentence) - 2)]:
                first_word = sentence[word_count][0]
                sec_word = sentence[word_count + 1][0]
                forth_word = sentence[word_count + 3][0]
                fifth_word = sentence[word_count + 4][0]
                context = [first_word, sec_word, word, forth_word, fifth_word]

                label_exist = False
                if L2I.has_key(label):
                    pred, loss = self.create_network_return_best(context, F2I, L2I, label, activate_sub_word)
                    cum_loss += loss.value()
                    label_ind = L2I.get(label)
                    label_exist = True
                if label_exist and pred == label_ind:
                    if mission == "POS" or (mission == "NER" and label != "O"):
                        good += 1
                else:
                    bad += 1

                word_count += 1
                dev_words += 1

        returned_values = [good / (good + bad), cum_loss / dev_words]
        return returned_values

def get_sub_words(word):
    words = []
    if len(word)<=3:
        words.append(word)
    else:
        prefix = word[:3]
        suffix = word[-3:]
        words = [word] + [prefix] + [suffix]
    return words


def GlorotInit(vocab_length, emb_vec_len):

    import math
    # initialization according to Xavier suggestion
    epsilon = math.sqrt(6.0) / math.sqrt(vocab_length + emb_vec_len)  # uniform range
    vec = np.random.uniform(-epsilon, epsilon, emb_vec_len)
    return vec

def run_experiment(mission, train_path, dev_path, vocab_path, vectors_path, context_size, hidden_layer_size,
                   initial_learning_rate, dropout_rate, epochs, saved_model_path, stop_embed_train = 30 ,activate_sub_word=False):

    from time import gmtime, strftime

    TRAIN, TRAIN_WORDS, TRAIN_LABELS = read_data(train_path)  # get training set
    DEV, DEV_WORDS, DEV_LABELS = read_data(dev_path)  # get dev set
    VOCAB = read_data(vocab_path, "vocab")[0] # get vocabulary from pre-trained words
    vectors = np.loadtxt(vectors_path) # get vectors of pre-trained words

    # if the feature for taking into account sub words is activated
    if activate_sub_word == True:
        for word in TRAIN_WORDS.keys():
            if word not in ["<s>", "</s>", "UUUNKKK"]:
                words_from_word = get_sub_words(word)
                for part_word in words_from_word:
                    if TRAIN_WORDS.has_key(part_word)==False:
                        TRAIN_WORDS[part_word] = 1

    # if a word appeared in train but not in the vocab add it to vocab as well
    words_not_in_vocab = []

    for word in TRAIN_WORDS.keys():
        if word not in VOCAB:
            words_not_in_vocab.append(word)
    
    # add to VOCAB and the vectors data structure words that appeared in the training but did not appear in VOCAB
    for word in words_not_in_vocab:
        VOCAB.append(word)
        word_vec = GlorotInit(len(VOCAB)+len(words_not_in_vocab), len(vectors[1]))
        vectors = np.vstack((vectors,word_vec))

    # parm dimensions:
    dim_params = [len(vectors[1]) * context_size, hidden_layer_size, len(TRAIN_LABELS)]

    # configure computation graph in dynet
    m = dy.ParameterCollection()  # create parameter collection
    network = TagNetwork(m, len(VOCAB), len(vectors[1]), dim_params, vectors)  # create network
    trainer = dy.SimpleSGDTrainer(m, initial_learning_rate)  # create trainer

    L2I = {l: i for i, l in enumerate(list(sorted(TRAIN_LABELS)))}  # enumerate the labels as 0,1,2,...
    F2I = {f: i for i, f in enumerate(list(sorted(VOCAB)))}  # enumerate the vocabulary as 0,1,2,...

    # change DEV dataset by replacing words that do not appear in the vocab with UUUNKKK
    TRAIN_CHANGED = list(TRAIN)
    for sentence in TRAIN_CHANGED:
        duplicate_sentence = False
        new_sentence = []
        if mission == 'NER':
            for word, label in sentence:
                if label != 'O':
                    duplicate_sentence = True
        for word, label in sentence:
            if word in words_not_in_vocab:
                new_sentence.append(("UUUNKKK", label))
                duplicate_sentence = True
            else:
                new_sentence.append((word, label))
        if duplicate_sentence == True:
            TRAIN.append(new_sentence)

    # change DEV dataset by replacing words that do not appear in the vocab with UUUNKKK
    DEV_CHANGED = list(DEV)
    DEV = []
    for sentence in DEV_CHANGED:
        new_sentence = []
        for word, label in sentence:
            if word in VOCAB:
                new_sentence.append((word, label))
            else:
                new_sentence.append(("UUUNKKK", label))
        DEV.append(new_sentence)

    print strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # training the model:
    random.seed(2)
    best_epoch = 0
    max_accuracy = 0
    stop_updating_lookup = False # incidate whether to stop training the lookup parameters
    for epoch in xrange(epochs):
        random.shuffle(TRAIN)
        cum_loss = 0
        examples = 0
        for sentence in TRAIN:
            sentence = [('<s>', 'dummy'), ('<s>', 'dummy')] + sentence + [('</s>', 'dummy'), ('</s>', 'dummy')]  # par each sentence add BOS prefix and EOS suffix
            word_count = 0 # count word in sentence
            for word, label in sentence[2:(len(sentence) - 2)]:
                context = [sentence[word_count][0], sentence[word_count + 1][0], word, sentence[word_count + 3][0],sentence[word_count + 4][0]]
                if epoch >= stop_embed_train:
                    stop_updating_lookup = True
                loss = network.create_network_return_loss(context, label, L2I, F2I, words_not_in_vocab, dropout_rate, activate_sub_word, stop_updating_lookup)
                cum_loss += loss.value()
                loss.backward()  # backpropogate the loss
                trainer.update()  # update model parameters according to gradients
                examples += 1
                word_count += 1

        train_loss = cum_loss / examples # train loss
        accuracy, loss = network.create_network_return_accuracy(DEV, L2I, F2I, mission, activate_sub_word) # get dev loss and accuracy
        print accuracy, loss, train_loss, strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # save best model
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            best_epoch = epoch
            m.save(saved_model_path)

    print "best epoch:", best_epoch


if __name__ == '__main__':

    print "----------------------POS pre trained embeddings-------------------"
    mission = "POS" # POS or NER
    train_path = "../pos/train" # train path
    dev_path = "../pos/dev" # dev path
    vocab_path = "../pre_trained/vocab.txt"
    vectors_path = "../pre_trained/wordVectors.txt"
    saved_model_path = "./saved_models/model_" + mission + "_pre_trained"  # saving model path
    context_size = 5 # window size
    hidden_layer_size = 220 # hidden layer size
    initial_learning_rate = 0.002 # learning rate
    dropout_rate = 0.5 # dropout
    epochs = 60 # number of epochs
    activate_sub_word = False
    stop_embed_train = 62
    run_experiment(mission, train_path, dev_path, vocab_path, vectors_path, context_size, hidden_layer_size,
                  initial_learning_rate, dropout_rate, epochs, saved_model_path, stop_embed_train, activate_sub_word)


