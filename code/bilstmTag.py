import numpy as np
import dynet as dy


# biLSTM transducer class for section a
def transducerA(words_sequence, word2int, vocab, model_params):
    
    lookup, fw_builder1, bw_builder1, fw_builder2, bw_builder2, W, b = model_params

    sequence = []
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
    W = dy.parameter(W)
    b = dy.parameter(b)
    fw_lstm1 = fw_builder1.initial_state()
    bw_lstm1 = bw_builder1.initial_state()
    fw_lstm2 = fw_builder2.initial_state()
    bw_lstm2 = bw_builder2.initial_state()

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


# biLSTM transducer class for section b
def transducerB(words_sequence, word2int, vocab, model_params):
    
    lookup, char_builder, fw_builder1, bw_builder1, fw_builder2, bw_builder2, W, b = model_params

    char_lstm = char_builder.initial_state()

    sequence = []
    for word in words_sequence:
        char_embed = []
        word_chars = list(word)
        for ch in word_chars:
            char_embed.append(lookup[word2int.get(ch)])
        sequence.append(char_lstm.transduce(char_embed)[-1])

    # convert the parameter into an Expession (add it to graph)
    W = dy.parameter(W)
    b = dy.parameter(b)
    fw_lstm1 = fw_builder1.initial_state()
    bw_lstm1 = bw_builder1.initial_state()
    fw_lstm2 = fw_builder2.initial_state()
    bw_lstm2 = bw_builder2.initial_state()


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


# biLSTM transducer class for section c
def transducerC(words_sequence, word2int, vocab, model_params):

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

    lookup, fw_builder1, bw_builder1, fw_builder2, bw_builder2, W, b = model_params

    sequence = []
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
    W = dy.parameter(W)
    b = dy.parameter(b)
    fw_lstm1 = fw_builder1.initial_state()
    bw_lstm1 = bw_builder1.initial_state()
    fw_lstm2 = fw_builder2.initial_state()
    bw_lstm2 = bw_builder2.initial_state()

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


# biLSTM transducer class for section d
def transducerD(words_sequence, word2int, vocab, model_params):

    lookup, char_builder, W_con, b_con, fw_builder1, bw_builder1, fw_builder2, bw_builder2, W, b = model_params

    char_lstm = char_builder.initial_state()
    W_con = dy.parameter(W_con)
    b_con = dy.parameter(b_con)

    sequence = []
    for word in words_sequence:
        char_embed = []
        word_chars = list(word)
        # get char embeddings of words
        for ch in word_chars:
            char_embed.append(lookup[word2int.get(ch)])

        # get char LSTM encoding
        char_encoder = char_lstm.transduce(char_embed)[-1]

        if word not in vocab:
            # curr_word_embed = dy.esum(char_embed)
            curr_word_embed = lookup[word2int.get("<UNK>")]
        else:
            curr_word_embed = lookup[word2int.get(word)]

        char_word_concat = dy.concatenate([curr_word_embed, char_encoder])
        sequence.append(W_con*char_word_concat + b_con)

    # convert the parameter into an Expession (add it to graph)
    W = dy.parameter(W)
    b = dy.parameter(b)
    fw_lstm1 = fw_builder1.initial_state()
    bw_lstm1 = bw_builder1.initial_state()
    fw_lstm2 = fw_builder2.initial_state()
    bw_lstm2 = bw_builder2.initial_state()

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


# run model on test set
def test_model(repre, test, model_params, word2int, label2int, vocab):

    preds = []

    for sentence in test:
        dy.renew_cg()  # create new computation graph for each sentence
        if repre == "a":
            word_prdictions = transducerA(sentence, word2int, vocab, model_params)
        elif repre == "b":
            word_prdictions = transducerB(sentence, word2int, vocab, model_params)
        elif repre == "c":
            word_prdictions = transducerC(sentence, word2int, vocab, model_params)
        else:
            word_prdictions = transducerD(sentence, word2int, vocab, model_params)

        sen_preds = []
        for idx, word in enumerate(sentence):
            label_num = np.argmax(word_prdictions[idx].npvalue())
            # find actual label
            for label, num in label2int.items():
                if num == label_num:
                    sen_preds.append((word, label))
                    break
            # np.argmax(net_output.npvalue())
        preds.append(sen_preds)

    return preds


# Read data function
def read_data(fname, dataset = "train"):
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


# read training set and change it accordingly
def get_train_dataset(train_file, mission, repre):

    dataset = read_data(train_file) # read train data to list. each list item is a sentence. each sentence is a tuple

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
def tagger(repre, mission, train_file, input_file, saved_model_path, output_file):

    # read train and dev data sets
    train, vocab, labels = get_train_dataset(train_file, mission, repre)
    test = read_data(input_file, "test")

    # define vocabulary and help structures
    word2int = {w: i for i, w in enumerate(vocab)}
    label2int = {l: i for i, l in enumerate(labels)}

    # create a transducer classifier
    m = dy.ParameterCollection()
    if repre == "a":
        model_params = dy.load(saved_model_path, m)
    elif repre == "b":
        model_params = dy.load(saved_model_path, m)
    elif repre == "c":
        model_params = dy.load(saved_model_path, m)
    else:
        model_params = dy.load(saved_model_path, m)

    predictions = test_model(repre, test, model_params, word2int, label2int, vocab)

    # write predictions to file
    file = open(output_file, "w")
    for idx, sentence in enumerate(predictions):
        word_idx = 0
        for word, label in sentence:
            orig_word = test[idx][word_idx]
            file.write(orig_word + " " + label + "\n")
            word_idx += 1
        file.write("\n")


if __name__ == '__main__':

    import sys
    repre = sys.argv[1]
    saved_model_path = sys.argv[2]
    input_file = sys.argv[3]
    train_file = sys.argv[4]
    mission = sys.argv[5].upper()
    output_file = saved_model_path+ "." + mission.lower() + "_pred"

    tagger(repre, mission, train_file, input_file, saved_model_path, output_file)