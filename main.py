# -*- coding: utf-8 -*-
# coding=utf-8
# @author: cer
import numpy as np
import pickle

import data.load
import io
from metrics.accuracy import conlleval
from data_loader import *
import my_metrics as metric
from keras.models import Sequential,Model
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import SimpleRNN, GRU, LSTM
from keras.layers.core import Dense, Dropout
from keras.layers.wrappers import TimeDistributed
from keras.layers import Convolution1D, MaxPooling1D, Bidirectional,Input, Embedding, LSTM, Dense
import progressbar
from keras.utils import plot_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

# train_data = io.open("dataset/atis-2.train.w-intent.iob", mode="r", encoding="utf-8").readlines()
# test_data = io.open("dataset/atis-2.dev.w-intent.iob", mode="r", encoding="utf-8").readlines()

# train_data = io.open("dataset/atis.train.w-pos-intent.iob", mode="r", encoding="utf-8").readlines()
# test_data = io.open("dataset/atis.test.w-pos-intent.iob", mode="r", encoding="utf-8").readlines()

all_data = io.open("persian_dataset/all.iob", mode="r", encoding="utf-8").readlines()
train_data, test_data = train_test_split(all_data,test_size=0.2,random_state=42)
train_file = io.open("persian_dataset/train.iob", mode="w", encoding="utf-8")
test_file = io.open("persian_dataset/test.iob", mode="w", encoding="utf-8")
for line in train_data:
    train_file.write('%s' % line)
for line in test_data:
    test_file.write('%s' % line)
train_file.close()
test_file.close()

train_data_word_text, train_data_lable_text, train_data_intent_text = data_pipeline2(train_data)
test_data_word_text, test_data_lable_text, test_data_intent_text = data_pipeline2(test_data)
# train_data_word_text, train_data_lable_text, train_data_intent_text = data_pipeline(train_data)
# test_data_word_text, test_data_lable_text, test_data_intent_text = data_pipeline(test_data)
# print(train_data_word_text[1])
# print(train_data_lable_text[1])
# print(train_data_lable_text[1])
# data_add_feature(test_data)


word2index, index2word, slot2index, index2slot, intent2index, index2intent = \
    get_info_from_training_data(train_data_word_text, train_data_lable_text, train_data_intent_text)
# print("index2slot: ", index2slot)
train_data_word_index, train_data_lable_index, train_data_intent_index = to_index(train_data_word_text, train_data_lable_text, train_data_intent_text, word2index, slot2index, intent2index)
test_data_word_index, test_data_lable_index, test_data_intent_index = to_index(test_data_word_text, test_data_lable_text, test_data_intent_text, word2index, slot2index, intent2index)

# print(word2index)
# print(slot2index)
print(train_data_intent_index)
### Load Data
# train_set, valid_set, dicts = data.load.atisfull()
# w2idx, ne2idx, labels2idx = dicts['words2idx'], dicts['tables2idx'], dicts['labels2idx']
# # Create index to word/label dicts
# idx2w  = {w2idx[k]:k for k in w2idx}
# idx2ne = {ne2idx[k]:k for k in ne2idx}
# idx2la = {labels2idx[k]:k for k in labels2idx}
# ### Ground truths etc for conlleval
# train_x, train_ne, train_label = train_set
# val_x, val_ne, val_label = valid_set
#
# words_val = [ list(map(lambda x: idx2w[x], w)) for w in val_x]
# groundtruth_val = [ list(map(lambda x: idx2la[x], y)) for y in val_label]
# words_train = [ list(map(lambda x: idx2w[x], w)) for w in train_x]
# groundtruth_train = [ list(map(lambda x: idx2la[x], y)) for y in train_label]

### Model
n_classes = len(index2slot)
n_intent = len(index2intent)
n_vocab = len(index2word)
print(n_classes)
print(n_vocab)
print(n_intent)

# # Define model
# model = Sequential()
# model.add(Embedding(input_dim=n_vocab,output_dim=100)) # max number of vocab in data , size of output embedding vector
# # model.add(Convolution1D(64,5,border_mode='same', activation='relu'))
# model.add(Dropout(0.25))
# model.add(Bidirectional(LSTM(100,return_sequences=True)))
# model.add(TimeDistributed(Dense(n_classes, activation='softmax')))
# model.compile('rmsprop', 'categorical_crossentropy')


main_input = Input(name='main_input',shape=(None,))
# This embedding layer will encode the input sequence
# into a sequence of dense 512-dimensional vectors.
x = Embedding(output_dim=50, input_dim=n_vocab)(main_input)
# A LSTM will transform the vector sequence into a single vector,
# containing information about the entire sequence
lstm_out = Bidirectional(LSTM(units=50,return_sequences=True))(x)
lstm_out_slot = Bidirectional(LSTM(units=50,return_sequences=True))(lstm_out)
intent_out = Bidirectional(LSTM(units=50,return_sequences=False))(lstm_out)
main_out = TimeDistributed(Dense(n_classes, activation='softmax'))(lstm_out_slot)
intent_out = Dense(n_intent, activation='softmax')(intent_out)
model = Model(inputs= main_input , outputs= [main_out,intent_out])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
print(model.summary())


### Training
n_epochs = 100


best_val_f1 = 0
train_acc_epoch = []
test_acc_epoch = []
test_f1_epoch = []
for i in range(n_epochs):
    print("Epoch {}".format(i))
    
    print("Training =>")
    train_pred_label_index = []
    avgLoss = 0
    	
    bar = progressbar.ProgressBar(len(train_data_word_index))
    for n_batch, sent in bar(enumerate(train_data_word_index)):
        label = np.array(train_data_lable_index[n_batch])
        intent_label = np.array(train_data_intent_index[n_batch])
        # print(label.shape)
        label = np.eye(n_classes)[label][np.newaxis,:]
        intent_label = np.eye(n_intent)[intent_label][np.newaxis,:]
        # print(label.shape)
        sent = np.array(sent)
        # print(sent.shape)
        sent = sent[np.newaxis,:]
        # print(sent.shape)
        if sent.shape[1] > 1: #some bug in keras
            loss = model.train_on_batch(sent, [label,intent_label])
            # avgLoss += loss
        # print("batch:{}          ".format(n_batch))
    # avgLoss = avgLoss/n_batch
    
# accuracy
    slot_acc_list =[]
    intent_acc_list = []
    bar = progressbar.ProgressBar(len(train_data_word_index))
    for n_batch, sent in bar(enumerate(train_data_word_index)):
        label = np.array(train_data_lable_index[n_batch])
        intent_label = np.array(train_data_intent_index[n_batch])
        # print(label.shape)
        # print(label.shape)
        sent = np.array(sent)
        # print(sent.shape)
        sent = sent[np.newaxis,:]
        # print(sent.shape)
        pred = model.predict_on_batch(sent)
        slot_pred = np.argmax(np.array(pred[0]),-1)[0]
        intent_pred = np.argmax(np.array(pred[1]),-1)[0]
        train_pred_label_index.append(slot_pred)
        slot_acc = metric.accuracy_score(label,slot_pred)
        intent_acc_list.append(intent_pred == intent_label)
        slot_acc_list.append(slot_acc)
    print("slot accuracy{}".format(np.average(slot_acc_list)))
    print("intent accuracy{}".format(np.average(intent_acc_list)))
    train_acc_epoch.append(np.average(slot_acc_list))
    train_pred_lable_text = [ list(map(lambda x: index2slot[x], y)) for y in train_pred_label_index]
    # print(train_pred_lable_text[0])
    # print(train_data_lable_text[0])
    # print(train_data_word_text[0])

    # prec, rec, f1 = conlleval(train_pred_lable_text, train_data_lable_text, train_data_lable_text, 'r.txt')
    # train_f_scores.append(f1)
    # print("train:")
    # print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, prec, rec, f1))
    
    print("Validating =>")
    
    val_pred_label = []
    val_pred_intent = []
    avgLoss = 0
    train_f_scores = []
    test_f_scores = []
    test_slot_acc_list = []
    test_intent_acc_list = []
    bar = progressbar.ProgressBar(len(test_data_word_index))
    for n_batch, sent in bar(enumerate(test_data_word_index)):
        label = np.array(test_data_lable_index[n_batch])
        intent_label = np.array(test_data_intent_index[n_batch])
        # label = np.eye(n_classes)[label][np.newaxis,:]
        sent = np.array(sent)
        sent = sent[np.newaxis,:]
        
        # if sent.shape[1] > 1: #some bug in keras
        #     loss = model.test_on_batch(sent, label)
        #     avgLoss += loss

        pred = model.predict_on_batch(sent)
        slot_pred = np.argmax(np.array(pred[0]),-1)[0]
        intent_pred = np.argmax(np.array(pred[1]), -1)[0]
        val_pred_label.append(slot_pred)
        val_pred_intent.append(intent_pred)
        acc = metric.accuracy_score(label,slot_pred)
        f1  = f1_score(label,slot_pred,average='micro')
        test_f_scores.append(f1)
        test_slot_acc_list.append(acc)
        test_intent_acc_list.append(intent_pred == intent_label)
        # print("batch:{}          ".format(n_batch))
    print("slot test accuracy{}".format(np.average(test_slot_acc_list)))
    print("intent test accuracy{}".format(np.average(test_intent_acc_list)))
    test_acc_epoch.append(np.average(test_slot_acc_list))
    f1_score_this_epoch = np.average(test_f_scores)
    test_f1_epoch.append(np.average(test_f_scores))
    print("test f1:{}".format(np.average(test_f_scores)))
    test_pred_lable_text = [ list(map(lambda x: index2slot[x], y)) for y in val_pred_label]
    test_pred_intent_text = [ index2intent[i] for i in val_pred_intent]
    # for i in range(len(test_data_word_index)):
    #     if (test_intent_acc_list[i] != 1):
    #         print(test_pred_intent_text[i])
    #         print(test_data_intent_text[i])
    #     #     print(test_data_word_text[i])
    #     if(test_slot_acc_list[i] != 1):
    #         print(test_pred_lable_text[i])
    #         print(test_data_lable_text[i])
    #         print(test_data_word_text[i])
    #         # print(test_data_lable_index[i])
    #         # print(val_pred_label[i])
    #         print(test_slot_acc_list[i])
    avgLoss = avgLoss/n_batch

    plt.close()
    plt.plot(train_acc_epoch,label="Train")
    plt.plot(test_acc_epoch,label="Test")
    plt.xlabel("Epoch")
    plt.ylabel("Slot Label Acc")
    # plt.show(block=False)
    plt.legend()
    plt.savefig('foo.png')
    plt.pause(0.1)
    # predword_val = [ list(map(lambda x: index2slot[x], y)) for y in val_pred_label]
    # prec, rec, f1 = conlleval(predword_val, test_data_lable_text, test_data_lable_text, 'r.txt')
    # val_f_scores.append(f1)
    # print('Loss = {}, Precision = {}, Recall = {}, F1 = {}'.format(avgLoss, prec, rec, f1))

    if f1_score_this_epoch > best_val_f1:
    	best_val_f1 = f1_score_this_epoch
    	open('model_architecture.json','w').write(model.to_json())
    	model.save_weights('best_model_weights.h5',overwrite=True)
    print("Best validation F1 score = {}".format(best_val_f1))
    print()
    
