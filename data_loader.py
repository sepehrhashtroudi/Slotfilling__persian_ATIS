# -*- coding: utf-8 -*-


import os
import random
import io
import numpy as np
from nltk.tag import StanfordNERTagger
from nltk import word_tokenize, pos_tag, ne_chunk

# st = StanfordNERTagger('NER/stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz',
# 					   'NER/stanford-ner/stanford-ner.jar',
# 					   encoding='utf-8')
# java_path = "C:/Program Files/Java/jdk1.8.0_162/bin/java.exe"
# os.environ['JAVAHOME'] = java_path

flatten = lambda l: [item for sublist in l for item in sublist]  # Two-dimensional exhibition into one dimension
index_seq2slot = lambda s, index2slot: [index2slot[i] for i in s]
index_seq2word = lambda s, index2word: [index2word[i] for i in s]


def data_pipeline(data, length=50):
    data = [t[:-1] for t in data]  # remove'\n'

    # One line of data like this：'BOS i want to fly from baltimore to dallas round trip EOS
    # \t O O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # Segmented into such a [original sentence word, annotated sequence，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]
    data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  # Remove BOS and EOS, and remove the corresponding label in the corresponding label sequence
    seq_in, seq_out, intent = list(zip(*data))
    return seq_in, seq_out, intent

def data_pipeline2(data, length=50):
    data = [t[:-1] for t in data]  # remove'\n'

    # One line of data like this：'BOS i want to fly from baltimore to dallas round trip EOS
    # \t O O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # Segmented into such a [original sentence word, annotated sequence，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" ")[:-1], t.split("\t")[1].split(" ")[-1]] for t in
            data]
    data = [[t[0][1:-1], t[1][1:-1], t[2]] for t in data]  # Remove BOS and EOS, and remove the corresponding label in the corresponding label sequence
    seq_in, seq_out, intent = list(zip(*data))
    return seq_in, seq_out, intent

def new_data_pipeline(data, length=50):
    data = [t[:-1] for t in data]  # remove'\n'

    # One line of data like this：'BOS i want to fly from baltimore to dallas round trip EOS
    # \t O O O O O O B-fromloc.city_name O B-toloc.city_name B-round_trip I-round_trip atis_flight'
    # Segmented into such a [original sentence word, annotated sequence，intent]
    data = [[t.split("\t")[0].split(" "), t.split("\t")[1].split(" "), t.split("\t")[2].split(" "),t.split("\t")[3]] for t in
            data]
    # data = [[t[0][1:-1], t[1][1:], t[2]] for t in data]  new dataset doesnt have BOS EOS
    seq_in, seq_out, pos_lable, intent = list(zip(*data))
    return seq_in, seq_out,pos_lable, intent

def data_add_feature(data):
    data = [t[:-1] for t in data]  # remove'\n'
    data = [[t.split("\t")[0].split(" ")[1:-1], t.split("\t")[1].split(" ")[1:-1], t.split("\t")[1].split(" ")[-1]] for t in data]
    with io.open("dataset/atis.test.w-pos-intent.iob", mode="w", encoding="utf-8") as file:
        for i,line in enumerate(data):
            # print(line[0])
            # tokenized_text = word_tokenize(line[0])
            word_pos = pos_tag(line[0])
            pos = [t[1] for t in word_pos] # select pos tags
            pos = " ".join(pos)
            label = " ".join(line[1])
            sentence = " ".join(line[0])
            new_data = sentence + '\t' + label + '\t' + pos + '\t' + line[2] + '\n'
            # print(new_data)
            file.write(new_data)
    file.close()

def get_info_from_training_data( seq_in, seq_out, intent):
    # seq_in, seq_out, intent = list(zip(*data))
    vocab = set(flatten(seq_in))
    slot_tag = set(flatten(seq_out))
    intent_tag = set(intent)
    # Generate word2index
    word2index = {'<PAD>': 0, '<UNK>': 1, '<SOS>': 2, '<EOS>': 3}
    for token in vocab:
        if token not in word2index.keys():
            word2index[token] = len(word2index)

    # Generate index2word
    index2word = {v: k for k, v in word2index.items()}
    # Generate tag2index
    tag2index = {'<PAD>': 0, '<UNK>': 1, "O": 2}
    for tag in slot_tag:
        if tag not in tag2index.keys():
            tag2index[tag] = len(tag2index)

    # Generate index2tag
    index2tag = {v: k for k, v in tag2index.items()}

    # Generate intent2index
    intent2index = {'<UNK>': 0}
    for ii in intent_tag:
        if ii not in intent2index.keys():
            intent2index[ii] = len(intent2index)

    # Generate index2intent
    index2intent = {v: k for k, v in intent2index.items()}
    return word2index, index2word, tag2index, index2tag, intent2index, index2intent


def getBatch(batch_size, train_data):
    random.shuffle(train_data)
    sindex = 0
    eindex = batch_size
    while eindex < len(train_data):
        batch = train_data[sindex:eindex]
        temp = eindex
        eindex = eindex + batch_size
        sindex = temp
        yield batch


def to_index(seq_in, seq_out, intent, word2index, slot2index, intent2index):
    new_train = []
    sin_ix = []
    sout_ix = []
    intent_ix = []
    for sent in seq_in:
        sin_ix.append(list(map(lambda i: word2index[i] if i in word2index else word2index["<UNK>"], sent)))

    for sent in seq_out:
        sout_ix.append(list(map(lambda i: slot2index[i] if i in slot2index else slot2index["<UNK>"], sent)))
    for sent in intent:
        intent_ix.append(intent2index[sent] if sent in intent2index else intent2index["<UNK>"])
    return sin_ix,sout_ix, intent_ix

def word_lable_intent_splitter(data, type):
    word = []
    true_length = []
    lable = []
    intent = []

    if(type == "index"):
        for sentence in data:
            word.append(sentence[0])
            true_length.append(sentence[1])
            lable.append(sentence[2])
            intent.append(sentence[3])
        return np.array(word),np.array(true_length), np.array(lable), np.array(intent)
        # return word, lable, intent
    if (type == "text"):
        for sentence in data:
            word.append(sentence[0])
            lable.append(sentence[1])
            intent.append(sentence[2])
        return word, lable, intent