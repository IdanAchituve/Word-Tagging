import numpy as np
import random
import dynet as dy


# biLSTM transducer class for section a
class biLSTMTaggerA(object):

    # constructor for creating parameter collection
    def __init__(self, vocab_size, bi_lstm1, bi_lstm2, embed_size, num_labels, model):
        # bi_lstm1 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]
        # bi_lstm2 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]

        self.params = {}

        # lookup:
        self.params["lookup"] = model.add_lookup_parameters((vocab_size, embed_size))

        # 1st bi-LSTM parameter collection:
        self.fw_builder1 = dy.LSTMBuilder(bi_lstm1[0], embed_size, bi_lstm1[1], model)
        self.bw_builder1 = dy.LSTMBuilder(bi_lstm1[2], embed_size, bi_lstm1[3], model)

        # 2nd bi-LSTM parameter collection:
        self.fw_builder2 = dy.LSTMBuilder(bi_lstm2[0], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[1], model)
        self.bw_builder2 = dy.LSTMBuilder(bi_lstm2[2], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[3], model)

        # out layer parameter collection:
        self.params["W"] = model.add_parameters((num_labels, bi_lstm2[1] + bi_lstm2[3]))
        self.params["b"] = model.add_parameters((num_labels))

    # create expressions and return output
    def __call__(self, words_sequence, word2int, vocab, dataset = "train"):

        lookup = self.params["lookup"]

        sequence = []
        if dataset == "train":
            for word, label in words_sequence:
                char_embed = []
                if word not in vocab:
                    word_chars = list(word)
                    for ch in word_chars:
                        char_embed.append(lookup[word2int.get(ch)])
                    s = dy.esum(char_embed)
                    sequence.append(s)
                else:
                    sequence.append(lookup[word2int.get(word)])
        else:
            for word in words_sequence:
                char_embed = []
                if word not in vocab:
                    word_chars = list(word)
                    for ch in word_chars:
                        char_embed.append(lookup[word2int.get(ch)])
                    s = dy.esum(char_embed)
                    sequence.append(s)
                else:
                    sequence.append(lookup[word2int.get(word)])

        # convert the parameter into an Expession (add it to graph)
        W = dy.parameter(self.params["W"])
        b = dy.parameter(self.params["b"])
        fw_lstm1 = self.fw_builder1.initial_state()
        bw_lstm1 = self.bw_builder1.initial_state()
        fw_lstm2 = self.fw_builder2.initial_state()
        bw_lstm2 = self.bw_builder2.initial_state()

        # get output vectors of all time steps for the first bi-lstm
        fw_lstm1_output = fw_lstm1.transduce(sequence)
        bw_lstm1_output = bw_lstm1.transduce(reversed(sequence))

        # concatenate backward vector to forward vector per each word
        bi1_output = [dy.concatenate([fw1, bw1]) for fw1, bw1 in zip(fw_lstm1_output, reversed(bw_lstm1_output))]

        # get output vectors of all time steps for the second bi-lstm
        fw_lstm2_output = fw_lstm2.transduce(bi1_output)
        bw_lstm2_output = bw_lstm2.transduce(reversed(bi1_output))

        # concatenate backward vector to forward vector per each 1st biLSTM vector
        bi2_output = [dy.concatenate([fw2, bw2]) for fw2, bw2 in zip(fw_lstm2_output, reversed(bw_lstm2_output))]

        # calc net output
        net_output = [dy.softmax(W*out + b) for out in bi2_output]

        return net_output

    # return the loss and prediction on test set
    def create_network_return_best(self, sentence, word2int, label2int, vocab):
        out = self(sentence, word2int, vocab)
        loss = [] # save loss for each word
        probs = [] #save net output for each word
        idx = 0
        for word, label in sentence:
            prob = -dy.log(dy.pick(out[idx], label2int.get(label)))
            loss.append(prob)
            probs.append(out[idx])
            idx +=1
        loss = dy.esum(loss)
        return loss, probs


# biLSTM transducer class for section b
class biLSTMTaggerB(object):

    # constructor for creating parameter collection
    def __init__(self, vocab_size, char_lstm, bi_lstm1, bi_lstm2, embed_size, num_labels, model):
        # bi_lstm1 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]
        # bi_lstm2 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]
        # char_lstm - [char_lstm_num_layers, char_lstm_num_units]

        self.params = {}

        # lookup:
        self.params["lookup"] = model.add_lookup_parameters((vocab_size, embed_size))

        #charachter level LSTM
        self.char_builder = dy.LSTMBuilder(char_lstm[0], embed_size, char_lstm[1], model)

        # 1st bi-LSTM parameter collection:
        self.fw_builder1 = dy.LSTMBuilder(bi_lstm1[0], char_lstm[1], bi_lstm1[1], model)
        self.bw_builder1 = dy.LSTMBuilder(bi_lstm1[2], char_lstm[1], bi_lstm1[3], model)

        # 2nd bi-LSTM parameter collection:
        self.fw_builder2 = dy.LSTMBuilder(bi_lstm2[0], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[1], model)
        self.bw_builder2 = dy.LSTMBuilder(bi_lstm2[2], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[3], model)

        # out layer parameter collection:
        self.params["W"] = model.add_parameters((num_labels, bi_lstm2[1] + bi_lstm2[3]))
        self.params["b"] = model.add_parameters((num_labels))

    # create expressions and return output
    def __call__(self, words_sequence, word2int, vocab, dataset = "train"):

        lookup = self.params["lookup"]
        char_lstm = self.char_builder.initial_state()

        sequence = []
        if dataset == "train":
            for word, label in words_sequence:
                char_embed = []
                word_chars = list(word)
                for ch in word_chars:
                    char_embed.append(lookup[word2int.get(ch)])
                sequence.append(char_lstm.transduce(char_embed)[-1])
        else:
            for word in words_sequence:
                char_embed = []
                word_chars = list(word)
                for ch in word_chars:
                    char_embed.append(lookup[word2int.get(ch)])
                sequence.append(char_lstm.transduce(char_embed)[-1])

        # convert the parameter into an Expession (add it to graph)
        W = dy.parameter(self.params["W"])
        b = dy.parameter(self.params["b"])
        fw_lstm1 = self.fw_builder1.initial_state()
        bw_lstm1 = self.bw_builder1.initial_state()
        fw_lstm2 = self.fw_builder2.initial_state()
        bw_lstm2 = self.bw_builder2.initial_state()


        # get output vectors of all time steps for the first bi-lstm
        fw_lstm1_output = fw_lstm1.transduce(sequence)
        bw_lstm1_output = bw_lstm1.transduce(reversed(sequence))

        # concatenate backward vector to forward vector per each word
        bi1_output = [dy.concatenate([fw1, bw1]) for fw1, bw1 in zip(fw_lstm1_output, reversed(bw_lstm1_output))]

        # get output vectors of all time steps for the second bi-lstm
        fw_lstm2_output = fw_lstm2.transduce(bi1_output)
        bw_lstm2_output = bw_lstm2.transduce(reversed(bi1_output))

        # concatenate backward vector to forward vector per each 1st biLSTM vector
        bi2_output = [dy.concatenate([fw2, bw2]) for fw2, bw2 in zip(fw_lstm2_output, reversed(bw_lstm2_output))]

        # calc net output
        net_output = [dy.softmax(W * out + b) for out in bi2_output]

        return net_output

    # return the loss and prediction on test set
    def create_network_return_best(self, sentence, word2int, label2int, vocab):
        out = self(sentence, word2int, vocab)
        loss = []  # save loss for each word
        probs = []  # save net output for each word
        idx = 0
        for word, label in sentence:
            prob = -dy.log(dy.pick(out[idx], label2int.get(label)))
            loss.append(prob)
            probs.append(out[idx])
            idx += 1
        loss = dy.esum(loss)
        return loss, probs


# biLSTM transducer class for section c
class biLSTMTaggerC(object):

    # constructor for creating parameter collection
    def __init__(self, vocab_size, bi_lstm1, bi_lstm2, embed_size, num_labels, model):
        # bi_lstm1 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]
        # bi_lstm2 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]

        self.params = {}

        # lookup:
        self.params["lookup"] = model.add_lookup_parameters((vocab_size, embed_size))

        # 1st bi-LSTM parameter collection:
        self.fw_builder1 = dy.LSTMBuilder(bi_lstm1[0], embed_size, bi_lstm1[1], model)
        self.bw_builder1 = dy.LSTMBuilder(bi_lstm1[2], embed_size, bi_lstm1[3], model)

        # 2nd bi-LSTM parameter collection:
        self.fw_builder2 = dy.LSTMBuilder(bi_lstm2[0], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[1], model)
        self.bw_builder2 = dy.LSTMBuilder(bi_lstm2[2], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[3], model)

        # out layer parameter collection:
        self.params["W"] = model.add_parameters((num_labels, bi_lstm2[1] + bi_lstm2[3]))
        self.params["b"] = model.add_parameters((num_labels))

    # create expressions and return output
    def __call__(self, words_sequence, word2int, vocab, dataset = "train"):

        # get prefix and suffix and sum them up with the word
        def add_sub_words_embd (word, word_embed):
            if len(word) <= 3:
                return dy.esum([word_embed])
            else:
                pref = False
                suff = False
                # check if prefix exist in F2I. relevant for test/dev sets
                if word2int.has_key(word[:3]):
                    prefix_embd = lookup[word2int.get(word[:3])]
                    pref = True
                # check if suffix exist in F2I. relevant for test/dev sets
                if word2int.has_key(word[-3:]):
                    suffix_embd = lookup[word2int.get(word[-3:])]
                    suff = True

                # sum vectors of word with existing prefix/suffix
                if pref and suff:
                    sum_embd = dy.esum([prefix_embd, suffix_embd, word_embed])
                elif pref and suff == False:
                    sum_embd = dy.esum([prefix_embd, word_embed])
                elif suff and pref == False:
                    sum_embd = dy.esum([suffix_embd, word_embed])
                else:
                    sum_embd = dy.esum([word_embed])

                return sum_embd

        lookup = self.params["lookup"]

        sequence = []
        if dataset == "train":
            for word, label in words_sequence:
                char_embed = []
                if word not in vocab: # for words not in vocab get char embeddings
                    word_chars = list(word)
                    for ch in word_chars:
                        char_embed.append(lookup[word2int.get(ch)])
                    s = dy.esum(char_embed)
                    sequence.append(add_sub_words_embd(word, s))
                else:
                    word_embed = lookup[word2int.get(word)]
                    sequence.append(add_sub_words_embd(word, word_embed))
        else:
            for word in words_sequence:
                char_embed = []
                if word not in vocab: # for words not in vocab get char embeddings
                    word_chars = list(word)
                    for ch in word_chars:
                        char_embed.append(lookup[word2int.get(ch)])
                    s = dy.esum(char_embed)
                    sequence.append(add_sub_words_embd(word, s))
                else:
                    word_embed = lookup[word2int.get(word)]
                    sequence.append(add_sub_words_embd(word, word_embed))

        # convert the parameter into an Expession (add it to graph)
        W = dy.parameter(self.params["W"])
        b = dy.parameter(self.params["b"])
        fw_lstm1 = self.fw_builder1.initial_state()
        bw_lstm1 = self.bw_builder1.initial_state()
        fw_lstm2 = self.fw_builder2.initial_state()
        bw_lstm2 = self.bw_builder2.initial_state()

        # get output vectors of all time steps for the first bi-lstm
        fw_lstm1_output = fw_lstm1.transduce(sequence)
        bw_lstm1_output = bw_lstm1.transduce(reversed(sequence))

        # concatenate backward vector to forward vector per each word
        bi1_output = [dy.concatenate([fw1, bw1]) for fw1, bw1 in zip(fw_lstm1_output, reversed(bw_lstm1_output))]

        # get output vectors of all time steps for the second bi-lstm
        fw_lstm2_output = fw_lstm2.transduce(bi1_output)
        bw_lstm2_output = bw_lstm2.transduce(reversed(bi1_output))

        # concatenate backward vector to forward vector per each 1st biLSTM vector
        bi2_output = [dy.concatenate([fw2, bw2]) for fw2, bw2 in zip(fw_lstm2_output, reversed(bw_lstm2_output))]

        # calc net output
        net_output = [dy.softmax(W*out + b) for out in bi2_output]

        return net_output

    # return the loss and prediction on test set
    def create_network_return_best(self, sentence, word2int, label2int, vocab):
        out = self(sentence, word2int, vocab)
        loss = [] # save loss for each word
        probs = [] #save net output for each word
        idx = 0
        for word, label in sentence:
            prob = -dy.log(dy.pick(out[idx], label2int.get(label)))
            loss.append(prob)
            probs.append(out[idx])
            idx +=1
        loss = dy.esum(loss)
        return loss, probs


# biLSTM transducer class for section d
class biLSTMTaggerD(object):

    # constructor for creating parameter collection
    def __init__(self, vocab_size, char_lstm, lin_dim, bi_lstm1, bi_lstm2, embed_size, num_labels, model):
        # bi_lstm1 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]
        # bi_lstm2 - [fw_lstm_num_layers, fw_lstm_num_units, bw_lstm_num_layers, bw_lstm_num_units]
        # char_lstm - [char_lstm_num_layers, char_lstm_num_units]
        # lin_dim - the number of units in the linear layer used for concatenation

        self.params = {}

        # lookup:
        self.params["lookup"] = model.add_lookup_parameters((vocab_size, embed_size))

        # charachter level LSTM
        self.char_builder = dy.LSTMBuilder(char_lstm[0], embed_size, char_lstm[1], model)

        # linear layer collection:
        self.params["W_con"] = model.add_parameters((lin_dim, embed_size + char_lstm[1]))
        self.params["b_con"] = model.add_parameters((lin_dim))

        # 1st bi-LSTM parameter collection:
        self.fw_builder1 = dy.LSTMBuilder(bi_lstm1[0], lin_dim, bi_lstm1[1], model)
        self.bw_builder1 = dy.LSTMBuilder(bi_lstm1[2], lin_dim, bi_lstm1[3], model)

        # 2nd bi-LSTM parameter collection:
        self.fw_builder2 = dy.LSTMBuilder(bi_lstm2[0], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[1], model)
        self.bw_builder2 = dy.LSTMBuilder(bi_lstm2[2], bi_lstm1[1] + bi_lstm1[3], bi_lstm2[3], model)

        # out layer parameter collection:
        self.params["W"] = model.add_parameters((num_labels, bi_lstm2[1] + bi_lstm2[3]))
        self.params["b"] = model.add_parameters((num_labels))

    # create expressions and return output
    def __call__(self, words_sequence, word2int, vocab, dataset = "train"):

        lookup = self.params["lookup"]
        char_lstm = self.char_builder.initial_state()
        W_con = dy.parameter(self.params["W_con"])
        b_con = dy.parameter(self.params["b_con"])

        sequence = []
        if dataset == "train":
            for word, label in words_sequence:
                char_embed = []
                word_chars = list(word)
                # get char embeddings of words
                for ch in word_chars:
                    char_embed.append(lookup[word2int.get(ch)])

                # get char LSTM encoding
                char_encoder = char_lstm.transduce(char_embed)[-1]

                if word not in vocab:
                    #curr_word_embed = dy.esum(char_embed)
                    curr_word_embed = lookup[word2int.get("<UNK>")]
                else:
                    curr_word_embed = lookup[word2int.get(word)]

                char_word_concat = dy.concatenate([curr_word_embed, char_encoder])
                sequence.append(W_con*char_word_concat + b_con)
        else:
            for word in words_sequence:
                char_embed = []
                word_chars = list(word)
                # get char embeddings of words
                for ch in word_chars:
                    char_embed.append(lookup[word2int.get(ch)])

                # get char LSTM encoding
                char_encoder = char_lstm.transduce(char_embed)[-1]

                if word not in vocab:
                    #curr_word_embed = dy.esum(char_embed)
                    curr_word_embed = lookup[word2int.get("<UNK>")]
                else:
                    curr_word_embed = lookup[word2int.get(word)]

                char_word_concat = dy.concatenate([curr_word_embed, char_encoder])
                sequence.append(W_con * char_word_concat + b_con)

        # convert the parameter into an Expession (add it to graph)
        W = dy.parameter(self.params["W"])
        b = dy.parameter(self.params["b"])
        fw_lstm1 = self.fw_builder1.initial_state()
        bw_lstm1 = self.bw_builder1.initial_state()
        fw_lstm2 = self.fw_builder2.initial_state()
        bw_lstm2 = self.bw_builder2.initial_state()

        # get output vectors of all time steps for the first bi-lstm
        fw_lstm1_output = fw_lstm1.transduce(sequence)
        bw_lstm1_output = bw_lstm1.transduce(reversed(sequence))

        # concatenate backward vector to forward vector per each word
        bi1_output = [dy.concatenate([fw1, bw1]) for fw1, bw1 in zip(fw_lstm1_output, reversed(bw_lstm1_output))]

        # get output vectors of all time steps for the second bi-lstm
        fw_lstm2_output = fw_lstm2.transduce(bi1_output)
        bw_lstm2_output = bw_lstm2.transduce(reversed(bi1_output))

        # concatenate backward vector to forward vector per each 1st biLSTM vector
        bi2_output = [dy.concatenate([fw2, bw2]) for fw2, bw2 in zip(fw_lstm2_output, reversed(bw_lstm2_output))]

        # calc net output
        net_output = [dy.softmax(W * out + b) for out in bi2_output]

        return net_output

    # return the loss and prediction on test set
    def create_network_return_best(self, sentence, word2int, label2int, vocab):
        out = self(sentence, word2int, vocab)
        loss = []  # save loss for each word
        probs = []  # save net output for each word
        idx = 0
        for word, label in sentence:
            prob = -dy.log(dy.pick(out[idx], label2int.get(label)))
            loss.append(prob)
            probs.append(out[idx])
            idx += 1
        loss = dy.esum(loss)
        return loss, probs


# run model on dev set
def test_model(mission, dev, transducer, word2int, label2int, vocab, dev_per_log, per_log, epoch, sentence_count, cum_good_bad):

    good = bad = 0.0
    cum_loss = 0.0
    num_examples = len(dev)
    idx = 0
    batch_size = 500

    for sentence in dev:
        if idx % batch_size == 0:
            dy.renew_cg()  # create new computation graph for each batch
            batch_preds = []  # batch predictions list
            batch_losses = []

        loss, sentence_prdictions = transducer.create_network_return_best(sentence, word2int, label2int, vocab)
        batch_losses.append(loss)
        word_idx = 0
        batch_sentence = []
        for word, label in sentence:
            batch_sentence.append((word, label, label2int.get(label), sentence_prdictions[word_idx]))
            word_idx += 1
        batch_preds.append(batch_sentence)

        # calc batch loss and print examples to log
        if idx % batch_size == (batch_size - 1) or idx == (num_examples - 1):
            # after accumulating the loss from the batch run forward-backward
            batch_loss = dy.esum(batch_losses) / batch_size  # sum the loss of the batch
            cum_loss += batch_loss.value()  # this calls forward on each sequence in the batch through the whole net

            # calculate the accuracy on the batch
            for sen_to_print in batch_preds:
                for word, real_label, label, pred in sen_to_print:
                    out_vec_vals = pred.npvalue()  # transform to numpy array
                    pred_class = np.argmax(out_vec_vals)  # get max value
                    if label == pred_class:
                        if mission == "POS" or (mission == "NER" and real_label != "O"):
                            good += 1
                            cum_good_bad[0] += 1
                    else:
                        bad += 1
                        cum_good_bad[1] += 1

            # write to log
            sentence_accuracy_batch = str(good / (good + bad))
            per_log.write(sentence_accuracy_batch + "," + str(idx + 1) + "\n")

        if sentence_count % 500 == (batch_size - 1) or (epoch == 4 and idx == (num_examples - 1)):
            sentence_accuracy = str(cum_good_bad[0] / (cum_good_bad[0] + cum_good_bad[1]))
            dev_per_log.write(sentence_accuracy + "," + str(sentence_count+1) + "\n")

        idx += 1
        sentence_count += 1

    returned_values = [cum_loss / num_examples, good / (good + bad), sentence_count, cum_good_bad]
    return returned_values


# train model
def train_model(mission, train, dev, vocab, epochs, batch_size, trainer, transducer, word2int, label2int, per_log, dev_per_log):

    from time import gmtime, strftime
    curr_time = str(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
    num_examples = len(train)
    print curr_time

    # print to screen and to log files
    str_header = "time\t\t\t\t"+ "#epochs\t\t"+ "train_loss\t"+ "train_accuracy\t"+ "test_loss\t\t"+ "test_accuracy\n"
    print str_header
    per_log.write(str_header)

    sentence_count = 0
    cum_good_bad = [0.0, 0.0]

    # training code: batched.
    for epoch in range(epochs):
        random.shuffle(train)
        cum_loss = 0
        good = bad = 0.0
        idx = 0

        for sentence in train:
            if idx % batch_size == 0:
                dy.renew_cg()  # create new computation graph for each batch
                batch_preds = []  # batch predictions list
                batch_losses = []

            loss, sentence_prdictions = transducer.create_network_return_best(sentence, word2int, label2int, vocab)
            batch_losses.append(loss)
            word_idx = 0
            batch_sentence = []
            for word, label in sentence:
                batch_sentence.append((word, label, label2int.get(label), sentence_prdictions[word_idx]))
                word_idx += 1
            batch_preds.append(batch_sentence)

            if (idx % batch_size == (batch_size - 1)) or idx == (num_examples - 1):
                # after accumulating the loss from the batch run forward-backward
                batch_loss = dy.esum(batch_losses) / batch_size  # sum the loss of the batch
                cum_loss += batch_loss.value()  # this calls forward on each sequence in the batch through the whole net
                batch_loss.backward()
                trainer.update()
                # calculate the accuracy on the batch
                for sen_to_print in batch_preds:
                    for word, real_label, label, pred in sen_to_print:
                        out_vec_vals = pred.npvalue()  # transform to numpy array
                        pred_class = np.argmax(out_vec_vals)  # get max value
                        if label == pred_class:
                            if mission == "POS" or (mission == "NER" and real_label != "O"):
                                good += 1
                        else:
                            bad += 1

            idx += 1

        # get train/dev loss and accuracy
        train_loss = str(cum_loss / num_examples)  # train loss
        train_accuracy = str(good / (good + bad))
        test_loss, dev_accuracy, sentence_count, cum_good_bad = test_model(mission, dev, transducer, word2int, label2int, vocab, dev_per_log, per_log, epoch, sentence_count, cum_good_bad)
        test_loss = str(test_loss)
        dev_accuracy= str(dev_accuracy)

        # print to screen and to log
        curr_time = str(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))
        print_str = curr_time + "\t" + str(epoch + 1) + "\t\t" + train_loss + "\t\t" + train_accuracy + "\t" + test_loss + "\t\t" + dev_accuracy + "\n"
        print print_str
        per_log.write("\n"+print_str)

    return dev_accuracy


# Read data function
def read_data(fname, mission):
    data = []  # list of lists. each list is a sentence
    sentence = []  # list of tuples: (word in sentence, label)
    for line in file(fname):
        if line == "\n":  # new sentence after line break
            data.append(sentence)
            sentence = []
            continue
        word, label = line.strip().split()
        sentence.append((word, label))
    return data


# read training set and change it accordingly
def get_train_dataset(train_file, mission, repre):

    dataset = read_data(train_file, mission) # read train data to list. each list item is a sentence. each sentence is a tuple

    # get all train words and labels
    words_list = []
    labels = set()
    for sentence in dataset:
        for word, tag in sentence:
            words_list.append(word)
            labels.add(tag)

    # count occurrences of each words
    from collections import Counter
    word_count = Counter(words_list)

    # get vocabulary of the most common words (the 80% most popular words)
    num_unique_words = len(word_count)
    vocab = set([x for x, c in word_count.most_common(int(round(num_unique_words*0.8)))])

    # if incorporating sub-words add them to vocabulary
    if repre == "c":
        temp_set = set()
        for word in vocab:
            if len(word)> 3:
                prefix = word[:3]
                suffix = word[-3:]
                temp_set.add(prefix)
                temp_set.add(suffix)
        vocab.union(temp_set)

    # add tokens to vocab
    import string
    vocab.add("<UNK>")
    ascii_char = list(string.printable)
    for ascii in ascii_char:
        vocab.add(ascii)

    return dataset, vocab, labels


# tagger main function
def tagger(repre, mission, char_lstm, lin_dim, bi_lstm1, bi_lstm2, embed_size, epochs, lr, batch_size, train_file, dev_file, per_log, dev_per_log, saved_model_path, test_model_after_train = False):

    # read train and dev data sets
    train, vocab, labels = get_train_dataset(train_file, mission, repre)
    dev = read_data(dev_file, mission)

    # define vocabulary and help structures
    word2int = {w: i for i, w in enumerate(vocab)}
    label2int = {l: i for i, l in enumerate(labels)}
    vocab_size = len(vocab)
    num_labels = len(labels)

    # create a transducer classifier
    m = dy.ParameterCollection()
    trainer = dy.AdamTrainer(m, lr) # define trainer # lr
    if repre == "a":
        transducer = biLSTMTaggerA(vocab_size, bi_lstm1, bi_lstm2, embed_size, num_labels, m) # create classifier
        dev_accuracy = train_model(mission, train, dev, vocab, epochs, batch_size, trainer, transducer, word2int,
                                   label2int, per_log, dev_per_log)
        dy.save(saved_model_path + "_" + dev_accuracy,[transducer.params["lookup"], transducer.fw_builder1, transducer.bw_builder1, transducer.fw_builder2, transducer.bw_builder2, transducer.params["W"], transducer.params["b"]])

    elif repre == "b":
        transducer = biLSTMTaggerB(vocab_size, char_lstm, bi_lstm1, bi_lstm2, embed_size, num_labels, m) # create classifier
        dev_accuracy = train_model(mission, train, dev, vocab, epochs, batch_size, trainer, transducer, word2int,
                                   label2int, per_log, dev_per_log)
        dy.save(saved_model_path + "_" + dev_accuracy,[transducer.params["lookup"], transducer.char_builder, transducer.fw_builder1, transducer.bw_builder1, transducer.fw_builder2, transducer.bw_builder2, transducer.params["W"], transducer.params["b"]])

    elif repre == "c":
        transducer = biLSTMTaggerC(vocab_size, bi_lstm1, bi_lstm2, embed_size, num_labels, m) # create classifier
        dev_accuracy = train_model(mission, train, dev, vocab, epochs, batch_size, trainer, transducer, word2int,
                                   label2int, per_log, dev_per_log)
        dy.save(saved_model_path + "_" + dev_accuracy,[transducer.params["lookup"],transducer.fw_builder1,transducer.bw_builder1,transducer.fw_builder2,transducer.bw_builder2,transducer.params["W"],transducer.params["b"]])

    else:
        transducer = biLSTMTaggerD(vocab_size, char_lstm, lin_dim, bi_lstm1, bi_lstm2, embed_size, num_labels, m) # create classifier
        dev_accuracy = train_model(mission, train, dev, vocab, epochs, batch_size, trainer, transducer, word2int,
                                   label2int, per_log, dev_per_log)
        dy.save(saved_model_path + "_" + dev_accuracy,[transducer.params["lookup"], transducer.char_builder, transducer.params["W_con"], transducer.params["b_con"], transducer.fw_builder1, transducer.bw_builder1, transducer.fw_builder2, transducer.bw_builder2, transducer.params["W"], transducer.params["b"]])

    if test_model_after_train == True:
        if mission == "NER":
            test_model_on_blind_set("./ner/test", saved_model_path + "_" + dev_accuracy +".ner", transducer, word2int, label2int, vocab)
        else:
            test_model_on_blind_set("./pos/test", saved_model_path + "_" + dev_accuracy +".pos", transducer, word2int, label2int, vocab)


# Read data function
def read_test_data(fname, dataset = "train"):
    data = []  # list of lists. each list is a sentence
    sentence = []  # list of tuples: (word in sentence, label)
    for line in file(fname):
        if line == "\n":  # new sentence after line break
            data.append(sentence)
            sentence = []
            continue
        if dataset == "train":
            word, label = line.strip().split()
            sentence.append((word, label))
        else:
            word = line.strip()
            sentence.append(word)
    return data



def test_model_on_blind_set(test_data, test_save_path, transducer, word2int, label2int, vocab):

    preds = []
    test = read_test_data(test_data, "test")
    orig_data_set = list(test)

    for sentence in test:
        dy.renew_cg()  # create new computation graph for each sentence
        word_prdictions = transducer.__call__(sentence, word2int, vocab, "test")
        sen_preds = []
        for idx, word in enumerate(sentence):
            label_num = np.argmax(word_prdictions[idx].npvalue())
            # find actual label
            for label, num in label2int.items():
                if num == label_num:
                    sen_preds.append((word, label))
                    break
        preds.append(sen_preds)

    # write predictions to file
    file = open(test_save_path, "w")
    for idx, sentence in enumerate(preds):
        word_idx = 0
        for word, label in sentence:
            orig_word = orig_data_set[idx][word_idx]
            file.write(orig_word + " " + label + "\n")
            word_idx += 1
        file.write("\n")


if __name__ == '__main__':

    import sys
    repre = sys.argv[1]
    train_file = sys.argv[2]
    saved_model_path = sys.argv[3]
    dev_file = sys.argv[4]
    mission = sys.argv[5].upper()
    lr = float(sys.argv[6])
    batch_size = int(sys.argv[7])
    bi_lstm1 = [int(x) for x in sys.argv[8].split(',')]
    bi_lstm2 = [int(x) for x in sys.argv[9].split(',')]
    embed_size = int(sys.argv[10])
    str_params = "representation: " + repre +"\n" + "mission: " + mission +"\n" + "bi_lstm1: " + str(bi_lstm1) +"\n" + "bi_lstm2: " + str(bi_lstm2) + "\n" + "embed_size: " + str(embed_size) +"\n" + "lr: " + str(lr) +"\n" + "batch_size: " + str(batch_size) +"\n"
    char_lstm = None
    lin_dim = None
    if repre == 'b' or repre == "d":
        char_lstm = [int(x) for x in sys.argv[11].split(',')]
        str_params += "char_lstm: " + str(char_lstm) + "\n"
    if repre == "d":
        lin_dim = int(sys.argv[12])
        str_params += "lin_dim: " + str(lin_dim) +"\n"
    str_params += "\n"

    # configurations:
    epochs = 5
    saved_model_path = saved_model_path+ "_" + mission + "_" + repre
    test_model_after_train = False

    # initialize logs:
    from time import gmtime, strftime
    curr_time = str(strftime("%Y-%m-%d_%H-%M-%S", gmtime()))

    # create log files
    per_log = open("./log/performance_" + mission + "_" + repre + "_" + curr_time, 'w')
    dev_per_log = open("./log/acc_dev_" + mission + "_" + repre + "_" + curr_time, 'w')

    per_log.write(str_params)
    dev_per_log.write(str_params)

    tagger(repre, mission, char_lstm, lin_dim, bi_lstm1, bi_lstm2, embed_size, epochs, lr, batch_size, train_file, dev_file, per_log, dev_per_log, saved_model_path, test_model_after_train)