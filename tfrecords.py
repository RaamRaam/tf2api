from .libraries import *
import tensorflow as tf
import numpy as np
import os
import math
import inspect
import functools
import time
import concurrent.futures
import json

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))
def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

type_map={'int' : (_int64_feature,tf.int64),'float' : (_float_feature,tf.float32),'str' : (_bytes_feature,tf.string),'list' : (_bytes_feature,tf.string),'numpy.uint8' : (_int64_feature,tf.int64),'numpy.float32' : (_float_feature,tf.float32),'numpy.float64' : (_float_feature,tf.float64),'numpy.ndarray' : (_bytes_feature,tf.string)}

@timer
def SaveTFRecordSet(path,data_dict):
    batch_size=len(data_dict[list(data_dict.keys())[0]])
    header = {}
    header['cols']=list(data_dict.keys())
    header['coltypes']=[str(j).replace('\'>','').replace('<class \'','')  for j in [type(data_dict[i][0]) for i in data_dict.keys()]]
    with open('/'.join(path.split('/')[:-1])+'/'+path.split('/')[-1].split('.')[0]+'.json', 'w') as outfile:
        json.dump(header, outfile)

    with tf.io.TFRecordWriter(path) as writer:
        for index in range(batch_size):
            content_dict={}
            for (col,coltype) in zip(header['cols'],header['coltypes']):
                if 'int' in coltype:
                    transformed_value=int(data_dict[col][index])
                elif 'float' in coltype:
                    transformed_value=data_dict[col][index]
                elif coltype=='str':
                    transformed_value=bytes(data_dict[col][index].strip(),'utf-8')
                elif coltype=='numpy.ndarray':
                    content_dict[col+'_h']=_int64_feature(data_dict[col][index].shape[0])
                    content_dict[col+'_w']=_int64_feature(data_dict[col][index].shape[1])
                    content_dict[col+'_d']=_int64_feature(data_dict[col][index].shape[2])
                    transformed_value = data_dict[col][index].tostring()
                elif coltype=='list':
                    list_to_array=np.asarray(data_dict[col][index])
                    content_dict[col+'_l']=_int64_feature(list_to_array.shape[0])
                    transformed_value = list_to_array.tostring()
                # print(col,coltype)
                content_dict[col]=type_map[coltype][0](transformed_value)
                example = tf.train.Example(features=tf.train.Features(feature=content_dict))
            writer.write(example.SerializeToString())
    return

def create_classfile(path,class_names):
  with open(path, 'w') as f:
    f.writelines([i+'\n' for i in class_names])



class ds(object):


    def __init__(self):
        self.ds=None
        self.length=0
        self.columns=[]
        

    

    def __parser(self,record,content_dict,head):
        parsed = tf.io.parse_single_example(record, content_dict)
        for (col,coltype) in zip(head[0],head[1]):
            if 'int' in coltype:
              test=tf.cast(parsed[col], tf.int32)
              del parsed[col]
              parsed[col]=test
            elif 'float' in coltype:
              test=tf.cast(parsed[col], tf.float32)
              del parsed[col]
              parsed[col]=test
            elif coltype=='numpy.ndarray':
                test=tf.io.decode_raw(parsed[col],tf.float64)
                h=parsed[col+'_h']
                w=parsed[col+'_w']
                d=parsed[col+'_d']
                test1=tf.reshape(test, [h,w,d])
                del parsed[col]
                parsed[col]=test1
            elif coltype=='list':
                test=tf.io.decode_raw(parsed[col], tf.float64)
                l=parsed[col+'_l']
                test1=tf.reshape(test, [l,])
                del parsed[col]
                parsed[col]=test1
        return parsed

    @timer
    def ReadTFRecordSet(self,tffilelist,parallelize):
        with open('/'.join(tffilelist[0].split('/')[:-1])+'/'+tffilelist[0].split('/')[-1].split('.')[0]+'.json') as json_file:
            header = json.load(json_file)
        content_dict={}
        for (col,coltype) in zip(header['cols'],header['coltypes']):
            if coltype=='numpy.ndarray':
                content_dict[col+'_h']=tf.io.FixedLenFeature([], tf.int64)
                content_dict[col+'_w']=tf.io.FixedLenFeature([], tf.int64)
                content_dict[col+'_d']=tf.io.FixedLenFeature([], tf.int64)
            elif coltype=='list':
                content_dict[col+'_l']=tf.io.FixedLenFeature([], tf.int64)
            content_dict[col]=tf.io.FixedLenFeature([], type_map[coltype][1])
        tfds=tf.data.TFRecordDataset(tf.data.Dataset.list_files(tffilelist),num_parallel_reads=parallelize)              
        # for i in tfds:
        #   parser(i,content_dict,[header['cols'],header['coltypes']])
        #   break
        self.ds=tfds.map(lambda record:self.__parser(record,content_dict,[header['cols'],header['coltypes']]),num_parallel_calls=parallelize)
        list_ds=list(self.ds)
        self.length=len(list_ds)
        self.columns=list_ds[0].keys()
        return self
        # return {'ds':ds,'len':len(list_ds), 'keys':list_ds[0].keys()}
        
    def reproduce(self):
        return ds()
    @timer
    def FilterTFRecordSet(self,col,filterlist):
            new_ds=self.reproduce()
            new_ds.ds=self.ds.filter(lambda y: tf.reduce_any(tf.math.equal(int(y[col]),filterlist)))
            list_ds=list(new_ds.ds)
            new_ds.length=len(list_ds)
            new_ds.columns=list_ds[0].keys()
            return new_ds

        
