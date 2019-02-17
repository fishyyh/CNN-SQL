#encoding:utf-8
from keras.models import load_model
from utils import URLDECODE
from gensim.models.word2vec import Word2Vec
from word import getVecs
import keras
import sys
import numpy as np
def init():	      
    model=load_model('bestcnn')
    w_model=Word2Vec.load("file/word2model")
    return model,w_model
def check(model,w_model,data):
    if data !=None:
        xx=[]
        filepath='file/INPUT_SHAPE'
        input_shape=[]
        with open(filepath,'r') as f :
            for line in f.readlines() :
                input_shape=int(line)
        if len(data.strip()): #判断是否是空行
            for text in URLDECODE(data) :
                #print(text)
                try:

                    xx.append(w_model[text])
                except KeyError:
                    continue
            xx=np.array(xx, dtype='float')

        #x=np.expand_dims(xx,1)
        if not len(xx):
            return [0]
        x=[]
        x.append(xx)
        x=np.array(x)
        #print(x)
        x=keras.preprocessing.sequence.pad_sequences(x,dtype='float32',maxlen=input_shape)

        result=model.predict_classes(x, batch_size=len(x))
        #print(result)
        return result
    else:
        return [0]

                


def convert2label(vector):
    for v in vector:
        if v==1:
            string_array='''<title>SQL injection</title>
            <body>
            <div align=center>Haha SQL injection was detected!</div>
            </body>
            '''
        elif v==2:
            string_array='''<title>XSS attack</title>
            <body>
            <div align=center>Haha XSS attack was detected!</div>
            </body>
            '''
        else:
            string_array='''<title>Normal input</title>
            <body>
            <div align=center>Nothing happen</div>
            </body>
            '''
    return string_array
    

