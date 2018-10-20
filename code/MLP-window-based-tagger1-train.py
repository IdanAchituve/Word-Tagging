import dynet_config
import random
import numpy as np

dynet_config.set(random_seed=2, mem=512,weight_decay=0.00000001) # configuration for dynet
import dynet as dy


# Read data function
def read_data(fname):
    data = []  # list of lists. each list is a sentence
    sentence = []  # list of tuples: (word in sentence, label)
    vocab = {}  # a set of all unique words in the training set
    labels = {}  # a set of all unique labels in the training set
    for line in file(fname):
        if line == "\n":  # new sentence after line break
            data.append(sentence)
            sentence = []
            continue
        word, label = line.strip().split()
        word = word.lower()
        sentence.append((word, label))
        if vocab.has_key(word):
            vocab[word] = vocab[word] + 1
        else:
            vocab[word] = 1

        if labels.has_key(label):
            labels[label] = labels[label] + 1
        else:
            labels[label] = 1

            # labels.add(label)
    return data, vocab, labels


# create a class encapsulating the network
class TagNetwork(object):
    # add parameters to the parameter collection
    def __init__(self, pc, vocab_size, dim_embedding, dim_params):

        self.params = []
        for first_layer, sec_layer in zip(dim_params, dim_params[1:]):
            pW = pc.add_parameters((sec_layer, first_layer), init=dy.GlorotInitializer())
            pb = pc.add_parameters(sec_layer)
            self.params.append(pW)  # pW with Xavier initialization
            self.params.append(pb)  # pb

        self.lookup = pc.add_lookup_parameters((vocab_size, dim_embedding))  # embedding matrix. dimensions: vocab_size X vector_size

    # the __call__ method applies the network to an input
    def __call__(self, context, F2I, dropout_rate, activate_sub_word):

        num_params = len(self.params)  # get the length of params vector
        lookup = self.lookup

        if activate_sub_word == False: # for runs in which sub words are not taken into account
            emb_vectors = [lookup[F2I.get(i)] for i in context]  # get embedding of the words
        else:# sum embedding of word, suffix and prefix for words that allow it
            emb_vectors = []
            for word in context:
                # get the word alone if len<=3 or word=start/end/unk
                if len(word)<=3 or (word in ["<BOS>", "<EOS>", "<UNK>"]):
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
                        sum_embd = dy.esum([prefix_embd,suffix_embd,word_embd])
                    elif pref and suff == False:
                        sum_embd = dy.esum([prefix_embd,word_embd])
                    elif suff and pref == False:
                        sum_embd = dy.esum([suffix_embd,word_embd])
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
    def create_network_return_loss(self, input, expected_output, L2I, F2I, dropout_rate=1.0, activate_sub_word = False):
        dy.renew_cg()
        out = self(input, F2I, dropout_rate, activate_sub_word)
        loss = -dy.log(dy.pick(out, L2I.get(expected_output)))
        return loss

    # return the loss and prediction on unseen word in context
    def create_network_return_best(self, input, F2I, L2I, expected_output, activate_sub_word = False):
        dy.renew_cg()
        out = self(input, F2I, 1.0, activate_sub_word)
        loss = -dy.log(dy.pick(out, L2I.get(expected_output)))
        return np.argmax(out.npvalue()), loss

    def create_network_return_accuracy(self, dataset, L2I, F2I, mission, activate_sub_word = False):

        good = bad = 0.0
        cum_loss = 0.0
        dev_words = 0 # count number of dev words

        pred_log_dev = open("./log/dev_pred_" + mission, 'w')

        for sentence in dataset:
            sentence = [('<BOS>', 'dummy'), ('<BOS>', 'dummy')] + sentence + [('<EOS>', 'dummy'), ('<EOS>', 'dummy')]  # pad each sentence with BOS prefix and EOS suffix
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

                str_predictions = word + "," + label + "-" + str(label_ind) + "," + str(pred)
                pred_log_dev.write(str_predictions + "\n")

            pred_log_dev.write("\n")
        pred_log_dev.write("end epoch!\n\n\n")
        returned_values = [good / (good + bad), cum_loss / dev_words]
        return returned_values

# get prfix a
def get_sub_words(word):
    words = []
    if len(word)<=3:
        words.append(word)
    else:
        prefix = word[:3]
        suffix = word[-3:]
        words = [word] + [prefix] + [suffix]
    return words

def run_experiment(mission, train_path, dev_path, embedding_size, context_size, hidden_layer_size,
                   initial_learning_rate, dropout_rate, epochs, saved_model_path, activate_sub_word=False):
    from time import gmtime, strftime

    TRAIN, VOCAB, LABELS = read_data(train_path)  # get training set
    DEV, VOCAB_DEV, LABELS_DEV = read_data(dev_path)  # get dev set

    VOCAB["<BOS>"] = 2  # a word for representing the beginning of a sentence
    VOCAB["<EOS>"] = 2  # a word for representing the end of a sentence
    VOCAB["<UNK>"] = 1  # a word for representing an unknown word in the dev set

    # add unknown replica for words that appeared only once in the train set
    TRAIN_CHANGED = list(TRAIN)
    for sentence in TRAIN_CHANGED:
        duplicate_sentence = False
        new_sentence = []
        for word, label in sentence:
            if VOCAB[word] == 1:
                new_sentence.append(("<UNK>", label))
                duplicate_sentence = True
            else:
                new_sentence.append((word, label))
        if duplicate_sentence==True:
            TRAIN.append(new_sentence)

    # change DEV dataset by replacing words that do not appear in the vocab with UUUNKKK
    DEV_CHANGED = list(DEV)
    DEV = []
    for sentence in DEV_CHANGED:
        new_sentence = []
        for word, label in sentence:
            if word in VOCAB.keys():
                new_sentence.append((word, label))
            else:
                new_sentence.append(("<UNK>", label))
        DEV.append(new_sentence)

    # if the feature for taking into account sub words is activated
    if activate_sub_word == True:
        for word in VOCAB.keys():
            if word not in ["<BOS>", "<EOS>", "<UNK>"]:
                words_from_word = get_sub_words(word)
                for part_word in words_from_word:
                    if VOCAB.has_key(part_word)==False:
                        VOCAB[part_word] = 1

    # parm dimensions:
    dim_params = [embedding_size * context_size, hidden_layer_size, len(LABELS)]

    # configure computation graph in dynet
    m = dy.ParameterCollection()  # create parameter collection
    network = TagNetwork(m, len(VOCAB), embedding_size, dim_params)  # create network
    trainer = dy.SimpleSGDTrainer(m, initial_learning_rate)  # create trainer

    L2I = {l: i for i, l in enumerate(list(sorted(LABELS.keys())))}  # enumerate the labels as 0,1,2,...
    F2I = {f: i for i, f in enumerate(list(sorted(VOCAB.keys())))}  # enumerate the vocabulary as 0,1,2,...

    print strftime("%Y-%m-%d %H:%M:%S", gmtime())

    # training the model:
    random.seed(2)
    best_epoch = 0
    max_accuracy = 0
    for epoch in xrange(epochs):
        random.shuffle(TRAIN)
        cum_loss = 0
        examples = 0
        for sentence in TRAIN:
            sentence = [('<BOS>', 'dummy'), ('<BOS>', 'dummy')] + sentence + [('<EOS>', 'dummy'), ('<EOS>', 'dummy')]  # par each sentence add BOS prefix and EOS suffix
            word_count = 0 # count word in sentence
            for word, label in sentence[2:(len(sentence) - 2)]:
                context = [sentence[word_count][0], sentence[word_count + 1][0], word, sentence[word_count + 3][0], sentence[word_count + 4][0]]
                loss = network.create_network_return_loss(context, label, L2I, F2I, dropout_rate, activate_sub_word)
                cum_loss += loss.value()
                loss.backward()  # backpropogate the loss
                trainer.update()  # update model parameters according to gradients
                examples+=1
                word_count += 1
        train_loss = cum_loss / examples # train loss
        accuracy, loss = network.create_network_return_accuracy(DEV, L2I, F2I, mission, activate_sub_word) # get dev loss and accuracy
        print accuracy, loss, train_loss, strftime("%Y-%m-%d %H:%M:%S", gmtime())

        # save best model
        if max_accuracy < accuracy:
            max_accuracy = accuracy
            best_epoch = epoch
            m.save(saved_model_path)

            """
            print dy.parameter(network.params[0]).value()
            print dy.parameter(network.params[1]).value()
            print dy.parameter(network.params[2]).value()
            print dy.parameter(network.params[3]).value()


            m2 = dy.ParameterCollection()  # create parameter collection
            e1 = m2.add_lookup_parameters((len(VOCAB), embedding_size))
            w1 = m2.add_parameters((hidden_layer_size,embedding_size * context_size))
            b1 = m2.add_parameters((hidden_layer_size))
            w2 = m2.add_parameters((len(LABELS), hidden_layer_size))
            b2 = m2.add_parameters((len(LABELS)))
            m2.populate(saved_model_path)
            dy.renew_cg()
            print dy.parameter(w1).value()
            print dy.parameter(b1).value()
            print dy.parameter(w2).value()
            print dy.parameter(b2).value()
            """


    print "best epoch:", best_epoch


if __name__ == '__main__':

    print "----------------------NER embeddings and part of words-------------------"
    mission = "POS"  # POS or NER
    train_path = "../pos/train"  # train path
    dev_path = "../pos/dev"  # dev path
    saved_model_path = "../tagger3/saved_models/model_" + mission + "_train_vocab_sub_words_trial!!"  # saving model path
    embedding_size = 50  # embedding vector size
    context_size = 5  # window size
    hidden_layer_size = 240  # hidden layer size
    initial_learning_rate = 0.001  # learning rate
    dropout_rate = 0.5  # dropout
    epochs = 5  # number of epochs
    activate_sub_word = True
    run_experiment(mission, train_path, dev_path, embedding_size, context_size, hidden_layer_size,
                   initial_learning_rate, dropout_rate, epochs, saved_model_path, activate_sub_word)
