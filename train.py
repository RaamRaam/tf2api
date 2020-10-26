from .libraries import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalMaxPool2D, BatchNormalization
from tqdm import tqdm_notebook as tqdm
import math
import datetime,time
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import pickle

tf.keras.backend.set_floatx('float16')


class Conv11(tf.keras.Model):
  def __init__(self, c_out):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=(1,1), padding="VALID", kernel_initializer=None, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

  def call(self, inputs):
    return self.conv(inputs)


class ConvBN(tf.keras.Model):
  def __init__(self, c_out, k=(3,3),s=(1,1),d=(1,1)):
    super().__init__()
    self.conv = tf.keras.layers.Conv2D(filters=c_out, kernel_size=k,strides=s,dilation_rate=d, padding="VALID", kernel_initializer=None, use_bias=False)
    self.bn = tf.keras.layers.BatchNormalization(momentum=0.9, epsilon=1e-5)

  def call(self, inputs):
    return tf.nn.relu(self.bn(self.conv(inputs)))
 


class train(object):
  def __init__(self):
    # hard coded
    self.optimizer=tf.keras.optimizers.SGD
    self.lossfunction=tf.keras.losses.SparseCategoricalCrossentropy()

    self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
    # self.trace=True
    self._global_step = 0
    self._start_epoch=0
    self._global_step_reminder = 0
    self.current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")    
    self.history={}


  def _initialize1_(self,datalen):
    self.lr=self._linear_lr_(datalen,self.batch_size,self.epochs,self.lr_mode,self.lr_peak,self.lr_repeat)
#     print(self.lr)
#     self.optimizer=self.optimizer(self.lr)
#     self.optimizer.learning_rate=self.lr
    self._train_summary_writer = tf.summary.create_file_writer(self._train_log)
    self._test_summary_writer = tf.summary.create_file_writer(self._test_log)

  def _initialize2_(self):
      self._log=self.log_path + '/' + self.name + '/' + self.current_time
      self._train_log=self._log+'/train_log'
      self._test_log=self._log+'/test_log'
      self._chosen_model_=self.model()


  def _savehistory_(self,epoch):
    vardict=dict([(i,self.__dict__[i]) for i in self.__dict__.keys() if any(ele in str(type(self.__dict__[i])) for ele in ['int','float'])])
    if epoch==0:
      self.history['epoch']=[]
      
      for i in vardict.keys():
        self.history[i]=[]
    self.history['epoch'].append(epoch+1)
    for i in vardict.keys():
      self.history[i].append(vardict[i])
    
  def _savelrhistory_(self,iteration,epoch):
    if epoch==0 and iteration==1:
      self.history['lr_epoch']=[]
      self.history['lr_step']=[]
      self.history['lr']=[]

    self.history['lr_step'].append(iteration)
    self.history['lr_epoch'].append(epoch+1) 
    self.history['lr'].append(self.__dict__['optimizer'].lr)

  @timer  
  def call(self,train_ds,test_ds):
    tf.keras.backend.set_floatx('float16')
    test_ds_batches = test_ds.ds.shuffle(test_ds.length).batch(self.batch_size).prefetch(self.batch_size)
    for epoch in range(self._start_epoch,self._start_epoch+self.epochs):   
      print('epoch',epoch)   
      if epoch==0:
        self._initialize2_()
      if epoch==self._start_epoch:
        self._initialize1_(train_ds.length)
      self.train_loss_metric.reset_states()
      self.train_accuracy_metric.reset_states()
      self.test_loss_metric.reset_states()
      self.test_accuracy_metric.reset_states()
      # if self.trace:
      #   tf.summary.trace_on(graph=True, profiler=False)

      train_ds_batches = train_ds.ds.shuffle(train_ds.length).batch(self.batch_size).prefetch(self.batch_size)
      for x in tqdm(train_ds_batches):
        self._global_step=self._global_step+1
        self.optimizer.learning_rate=self.lr

        self._savelrhistory_(self._global_step,epoch)
        with self._train_summary_writer.as_default():    
          tf.summary.scalar('LR', self.optimizer.lr, step=self._global_step_reminder+self._global_step)
        inputs=tf.cast(x['features'],tf.float16)
        labels=tf.cast(x['lables'],tf.int32)
        predictions=self._deep_learn_(inputs, labels, 'train')
        # if self.trace:
        #   with self.train_summary_writer.as_default():        
        #     tf.summary.trace_export(name='Architecture',step=0)#,profiler_outdir=self.train_log)
        #   tf.summary.trace_off()
        #   self.trace=False

      self._train_mean_loss = self.train_loss_metric.result().numpy()
      self._train_mean_accuracy = self.train_accuracy_metric.result().numpy()

      for x in test_ds_batches:
        inputs=tf.cast(x['features'],tf.float16)
        labels=tf.cast(x['lables'],tf.int32)
        predictions=self._deep_learn_(inputs, labels, 'test')
      self._test_mean_loss = self.test_loss_metric.result().numpy()
      self._test_mean_accuracy = self.test_accuracy_metric.result().numpy()

      with self._train_summary_writer.as_default():
        # tf.summary.scalar('LR', self.optimizer.lr, step=self.global_step_reminder+self.global_step)
        tf.summary.scalar('loss', self._train_mean_loss, step=epoch+1)
        tf.summary.scalar('accuracy', self._train_mean_accuracy, step=epoch+1)
        tf.summary.scalar('epochs', self.epochs, step=epoch+1)
        tf.summary.scalar('batch_size', self.batch_size, step=epoch+1)

      with self._test_summary_writer.as_default():
        tf.summary.scalar('loss', self._test_mean_loss, step=epoch+1)
        tf.summary.scalar('accuracy', self._test_mean_accuracy, step=epoch+1)
      self._savehistory_(epoch)
      print('Epoch: ', epoch+1, 'train loss:    ', self._train_mean_loss, '  train accuracy: ',self._train_mean_accuracy, '  test loss:    ', self._test_mean_loss, '  test accuracy: ',self._test_mean_accuracy)
    self._start_epoch=epoch+1  
    self._global_step_reminder=self._global_step
    self._global_step = 0
  
  @tf.function
  def _deep_learn_(self,inputs, labels, mode):
    with tf.GradientTape() as tape:
      predictions = self._chosen_model_(inputs)
      # regularization_loss = tf.math.add_n(model.losses)
      pred_loss = self.lossfunction(labels, predictions)
      total_loss = pred_loss #+ regularization_loss
      if mode=='train':
        gradients = tape.gradient(total_loss, self._chosen_model_.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self._chosen_model_.trainable_variables))
      
    if mode=='train':
      self.train_loss_metric.update_state(total_loss)
      self.train_accuracy_metric.update_state(labels, predictions)
    else:
      self.test_loss_metric.update_state(total_loss)
      self.test_accuracy_metric.update_state(labels, predictions)
    return predictions
      
  def evaluate(self,ds):
      ds = ds.batch(self.batch_size).prefetch(self.batch_size)
      lst_predictions1=[]
      lst_actuals=[]
      for x in ds:
        inputs=tf.cast(x['features'],tf.float16)
        labels=tf.cast(x['lables'],tf.int32)
        lst_predictions1.extend(list(self._deep_learn_(inputs, labels, 'test').numpy()))
        lst_actuals.extend(list(labels.numpy()))
        lst_predictions=[list(i) for i in lst_predictions1]
      return lst_actuals,lst_predictions


  def _linear_lr_(self,datalen,batch_size,epochs,lr_mode,lr_peak,lr_repeat):
    batches_per_epoch = datalen//batch_size + 1
    x = [i*batches_per_epoch for i in list(range(0,epochs,lr_repeat))+[list(range(0,epochs,lr_repeat))[-1]+lr_repeat]]

    if lr_mode=='stepup':
      z=[i+1 for i in x]
      x.extend(z[1:])
      x=sorted(x)[:-1]
      y=[lr_peak if i%2==0 else 0 for i in range(len(x))]

    if lr_mode=='stepdown':
      z=[i+1 for i in x]
      x.extend(z[1:])
      x=sorted(x)[:-1]
      y=[lr_peak if i%2==1 else 0 for i in range(len(x))]

    if lr_mode=='angledup':
      z=[round((x[i]+x[i+1])/2) for i in range(len(x)-1)]
      x.extend(z)
      x=sorted(x)
      y=[lr_peak if i%2==0 else 0 for i in range(len(x))]

    if lr_mode=='angleddown':
      z=[round((x[i]+x[i+1])/2) for i in range(len(x)-1)]
      x.extend(z)
      x=sorted(x)
      y=[lr_peak if i%2==1 else 0 for i in range(len(x))]

    if lr_mode=='constant':
      y=[lr_peak] * len(x)

    lr_schedule = lambda t: np.interp([t], x, y)[0]
    lr_func = lambda: lr_schedule(self._global_step)/batch_size
    return lr_func


  def save(self,path):
    self._chosen_model_.save(path +'/'+self.name +'/'+self.name+'.h5')
    file = open(path +'/'+self.name +'/'+self.name+'.pkl', 'wb')
    save_dict= dict([(i,self.__dict__[i]) for i in self.__dict__.keys() if any(ele in str(type(self.__dict__[i])) for ele in ['int','float','str', 'dict'])])
    pickle.dump(save_dict, file)
    file.close



  def load(self,path):
    self._chosen_model_ = tf.keras.models.load_model(path +'/'+self.name +'/'+self.name+'.h5')
    file = open(path +'/'+self.name +'/'+self.name+'.pkl', 'rb')
    hparams=pickle.load(file)
    file.close
    for i in hparams.keys():
      setattr(self, i, hparams[i])
    
