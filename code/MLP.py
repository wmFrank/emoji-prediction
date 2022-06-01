import jieba
import gensim
import csv
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
import joblib
import numpy as np
import time
import re

def getTrainData():
    file = open('Data/train.data','r',encoding = 'UTF-8-sig')
    file_obj = file.readlines()
    file.close()
    list = []
    for line in file_obj:
        line = line.replace('\n','')
        list.append(line)
    return list

def getTrainSolution():
    file = open('Data/train.solution','r',encoding = 'UTF-8-sig')
    file_obj = file.readlines()
    file.close()
    list = []
    for line in file_obj:
        line = line.replace('{','')
        line = line.replace('}','')
        line = line.replace('\n','')
        list.append(line)
    return list

def getEmoji():
    file = open('Data/emoji.data', 'r', encoding='UTF-8-sig')
    file_obj = file.readlines()
    file.close()
    list = []
    for line in file_obj:
        line = line.split('\t')[1]
        line = line.replace('\n','')
        list.append(line)
    return list

def getTestData():
    file = open('Data/test.data', 'r', encoding='UTF-8-sig')
    file_obj = file.readlines()
    file.close()
    list = []
    for line in file_obj:
        line = line.split('\t')[1]
        line = line.replace('\n','')
        list.append(line)
    return file_obj

def getCut(list_data):
    ls = []
    for line in list_data:
        line = re.sub("[\s+\.\/_,$%^*(+\"\']+|[+——、。~@#￥%&*（）]+", "", line)
        ls.append(list(jieba.cut(line)))
    return ls

def getVec1(list_traindata,list_testdata,list_trainsolution,list_emoji):
    x_train = []
    y_train = []
    x_test = []

    list_all = getCut(list_traindata)
    list_first = list_all.copy()
    list_second = getCut(list_testdata)
    list_all.extend(list_second)
    model = gensim.models.Word2Vec(list_all, min_count=1)
    #model.save('Vec_Model/GetModel')

    record = []
    for item in list_first:
        sum = np.zeros((100,))
        count = 0
        if len(item) != 0:
            record.append(1)
            for one in item:
                count = count + 1
                sum = sum + model.wv[one]
            result = sum / count
            x_train.append(list(result))
        else:
            record.append(0)

    for ind, item in enumerate(list_trainsolution):
        if record[ind] == 1:
            for index, value in enumerate(list_emoji):
                if item == value:
                    y_train.append(index)
                    break

    for item in list_second:
        sum = np.zeros((100,))
        count = 0
        if len(item) != 0:
            for one in item:
                count = count + 1
                sum = sum + model.wv[one]
            result = sum / count
            x_test.append(list(result))
        else:
            x_test.append(list(sum))

    return np.array(x_train), np.array(y_train), np.array(x_test)

def getVec2(list_traindata,list_testdata,list_trainsolution,list_emoji):
    x_train = []
    y_train = []
    x_test = []

    list_first = getCut(list_traindata)
    list_second = getCut(list_testdata)
    model = gensim.models.Word2Vec.load('Vec_Model/GetModel7')

    record = []
    for item in list_first:
        sum = np.zeros((100,))
        count = 0
        if len(item) != 0:
            record.append(1)
            for one in item:
                count = count + 1
                sum = sum + model.wv[one]
            result = sum / count
            x_train.append(list(result))
        else:
            record.append(0)

    for ind, item in enumerate(list_trainsolution):
        if record[ind] == 1:
            for index, value in enumerate(list_emoji):
                if item == value:
                    y_train.append(index)
                    break

    for item in list_second:
        sum = np.zeros((100,))
        count = 0
        if len(item) != 0:
            for one in item:
                count = count + 1
                sum = sum + model.wv[one]
            result = sum / count
            x_test.append(list(result))
        else:
            x_test.append(list(sum))

    return np.array(x_train), np.array(y_train), np.array(x_test)

def getPrediction2(x_test):
    mlp = joblib.load('Classifier_Model/MLP.pkl')
    y_test = mlp.predict(x_test)
    return y_test

def getPrediction1(x_train, y_train, x_test):
    mlp = MLPClassifier(hidden_layer_sizes = (115,115))
    mlp.fit(x_train,y_train)
    # joblib.dump(mlp,'Classifier_Model/MLP1.pkl')
    y_test = mlp.predict(x_test)
    return y_test

def main():
    list_traindata = getTrainData()
    list_trainsolution = getTrainSolution()
    list_emoji = getEmoji()
    list_testdata = getTestData()

    # choose getVec1 to train new word2vec model or choose getVec2 to use the existed well-trained word2vec model
    # x_train,y_train,x_test = getVec1(list_traindata,list_testdata,list_trainsolution,list_emoji)
    x_train, y_train, x_test = getVec2(list_traindata,list_testdata,list_trainsolution,list_emoji)

    # choose getPrediction1 to train new MLP model or choose getPrediction2 to use the existed well-trained MLP model
    # y_test = getPrediction1(x_train, y_train, x_test)
    y_test = getPrediction2(x_test)

    csvfile = open('Result/result0.csv', 'w', newline='')
    writer = csv.writer(csvfile)
    filehead = ['ID', 'Expected']
    writer.writerow(filehead)
    for pos, everyone in enumerate(y_test):
        lss = []
        lss.append(str(pos))
        lss.append(str(everyone))
        writer.writerow(lss)
    csvfile.close()
main()