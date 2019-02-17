#encoding:utf-8
from keras.models import load_model
from utils import URLDECODE
from gensim.models.word2vec import Word2Vec
from word import getVecs
import keras
import numpy as np
filepath='file/INPUT_SHAPE'
input_shape=[]
with open(filepath,'r') as f:
	for line in f.readlines():
		input_shape=int(line)
		print(input_shape)
model=load_model('sqllearn')
w_model=Word2Vec.load("file/word2model")
test=getVecs('test.csv',w_model)
print(test)
x=[]
for i in test:
    x.append(i)

x=np.array(x)
print(x)
x=keras.preprocessing.sequence.pad_sequences(x,dtype='float32',maxlen=input_shape)

result=model.predict_classes(x, batch_size=len(x))
print(result)
def convert2label(vector):
    string_array=[]
    for v in vector:
        if v==1:
            string_array.append('SQL注入__')
        elif v==2:
            string_array.append('xss攻击__')
        else:
            string_array.append('正常输入__')

    return string_array
    
print (convert2label(result))
