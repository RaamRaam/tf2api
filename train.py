from .libraries import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalMaxPool2D, BatchNormalization
from tqdm import tqdm_notebook as tqdm
import math
import datetime,time
from tensorflow import keras
from tensorflow.keras import layers

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
  def __init__(self, hparams):

    self.name=hparams['NAME']
    self.model=hparams['MODEL']
    self.train_ds=hparams['TRAIN_DS']
    self.test_ds=hparams['TEST_DS']
    self.epochs=hparams['EPOCHS']
    self.batch_size=hparams['BATCH_SIZE']

    self.trace=True
    self.global_step = 0
    self.start_epoch=0
    self.global_step_reminder = 0


    self.lr_peak=hparams['LR_PEAK']
    self.lr_repeat=hparams['LR_REPEAT']
    self.lr_interpolate=hparams['LR_INTERPOLATE']
    self.lr_modes=['constant','stepup','stepdown','angledup','angleddown']
    self.lr_mode=hparams['LR_MODE']
    self.lr=self.linear_lr(self.train_ds.length,self.batch_size,self.epochs,self.lr_mode,self.lr_peak,self.lr_repeat,self.lr_interpolate)

    self.optimizer=hparams['OPTIMIZER'](self.lr)
    self.lossfunction=hparams['LOSSFUNCTION']

    self.log_path=hparams['LOG_PATH']
    self.train_log=self.log_path+'/train_log'
    self.test_log=self.log_path+'/test_log'
    self.train_summary_writer = tf.summary.create_file_writer(self.train_log)
    self.test_summary_writer = tf.summary.create_file_writer(self.test_log)


    self.train_loss_metric = tf.keras.metrics.Mean(name='train_loss')
    self.train_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
    self.test_loss_metric = tf.keras.metrics.Mean(name='test_loss')
    self.test_accuracy_metric = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



  # def call(self):
  #   tf.keras.backend.set_floatx('float16')
  #   test_ds_batches = self.test_ds.ds.shuffle(self.batch_size).batch(self.batch_size).prefetch(self.batch_size)
  #   print('training....')
  #   t = time.time()
    
  #   inputs = keras.Input(shape=(28,28,1), name='Inputs')
  #   x = layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu', name='Conv_1')(inputs)
  #   x = layers.MaxPool2D(2,2)(x)
  #   x = layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu', name='Conv_2')(x)
  #   x = layers.MaxPool2D(2,2)(x)
  #   x = layers.Conv2D(filters=10,kernel_size=(3,3), activation='relu', name='Conv_3')(x)
  #   x = layers.GlobalAveragePooling2D()(x)
  #   # x = layers.Dense(10, activation='relu', name='dense_2')(x)
  #   outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
  #   model = keras.Model(inputs=inputs, outputs=outputs)
  #   train_acc_metric = keras.metrics.SparseCategoricalAccuracy()
  #   val_acc_metric = keras.metrics.SparseCategoricalAccuracy()

  #   if self.trace:
  #     tf.summary.trace_on(graph=True, profiler=False)
  #   for epoch in range(self.start_epoch,self.start_epoch+self.epochs):
  #     train_ds_batches = self.train_ds.ds.shuffle(self.train_ds.length).batch(self.batch_size).prefetch(self.batch_size)
  #     learnings=self.deep_learn(model, self.optimizer, None, train_ds_batches, test_ds_batches)
  #     lr=self.optimizer.learning_rate*self.batch_size
  #     train_loss=learnings[0][0]/self.train_ds.length
  #     train_acc=learnings[0][1]/self.train_ds.length
        
  #     val_loss=learnings[1][0]/self.test_ds.length
  #     val_acc=learnings[1][1]/self.test_ds.length
  #     time_taken=time.time() - t
  #     print("epoch: %0.3d \t lr:%0.2f \t train loss:%0.2f \t train acc:%2.2f \t  val loss:%0.2f \t val acc:%2.2f \t time:%0.2f" % (epoch+1,lr,train_loss,train_acc*100,val_loss,val_acc*100,time_taken))

  #     with self.train_summary_writer.as_default():
  #       tf.summary.scalar('loss', train_loss, step=epoch+1)
  #       tf.summary.scalar('accuracy', train_acc, step=epoch+1)
  #       tf.summary.scalar('epochs', self.epochs, step=epoch+1)
  #       tf.summary.scalar('batch_size', self.batch_size, step=epoch+1)

  #     with self.test_summary_writer.as_default():
  #       tf.summary.scalar('loss', val_loss, step=epoch+1)
  #       tf.summary.scalar('accuracy', val_acc, step=epoch+1)
  #   self.start_epoch=epoch+1  
  #   self.global_step_reminder=self.global_step
  #   self.global_step = 0


  @timer  
  def call(self):
    tf.keras.backend.set_floatx('float16')
    test_ds_batches = self.test_ds.ds.shuffle(self.test_ds.length).batch(self.batch_size).prefetch(self.batch_size)
    for epoch in range(self.epochs):
      self.train_loss_metric.reset_states()
      self.train_accuracy_metric.reset_states()
      self.test_loss_metric.reset_states()
      self.test_accuracy_metric.reset_states()
      if self.trace:
        tf.summary.trace_on(graph=True, profiler=False)

      train_ds_batches = self.train_ds.ds.shuffle(self.train_ds.length).batch(self.batch_size).prefetch(self.batch_size)
      tf.keras.backend.set_learning_phase(1)
      for x in train_ds_batches:
        self.global_step=self.global_step+1
        self.optimizer.learning_rate=self.lr
        inputs=tf.cast(x['features'],tf.float16)
        labels=tf.cast(x['lables'],tf.int32)
        self.deep_learn(inputs, labels, 'train')
        if self.trace:
          with train_summary_writer.as_default():        
            tf.summary.trace_export(name='Architecture',step=0)#,profiler_outdir=self.train_log)
          tf.summary.trace_off()
          self.trace=False

      self.train_mean_loss = self.train_loss_metric.result().numpy()
      self.train_mean_accuracy = self.train_accuracy_metric.result().numpy()

      tf.keras.backend.set_learning_phase(1)
      for x in train_ds_batches:
        inputs=tf.cast(x['features'],tf.float16)
        labels=tf.cast(x['lables'],tf.int32)
        self.deep_learn(inputs, labels, 'test')
      self.test_mean_loss = self.test_loss_metric.result().numpy()
      self.test_mean_accuracy = self.test_accuracy_metric.result().numpy()

      with train_summary_writer.as_default():
        tf.summary.scalar('loss', self.train_mean_loss, step=epoch+1)
        tf.summary.scalar('accuracy', self.train_mean_accuracy, step=epoch+1)
        tf.summary.scalar('epochs', self.epochs, step=epoch+1)
        tf.summary.scalar('batch_size', self.batch_size, step=epoch+1)

      with test_summary_writer.as_default():
        tf.summary.scalar('loss', self.test_mean_loss, step=epoch+1)
        tf.summary.scalar('accuracy', self.test_mean_accuracy, step=epoch+1)

      print('Epoch: ', epoch, '  loss:    ', train_mean_loss, '  accuracy: ',train_mean_accuracy, '  loss:    ', test_mean_loss, '  accuracy: ',test_mean_accuracy)
  
  @tf.function
  def deep_learn(self,inputs, labels, mode):
    # global global_step
    with tf.GradientTape() as tape:
      # predictions = model(inputs, training=True if mode=='train' else False)
      predictions = self.model(inputs)
      # regularization_loss = tf.math.add_n(model.losses)
      pred_loss = self.lossfunction(labels, predictions)
      total_loss = pred_loss #+ regularization_loss
    gradients = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
      
    if mode=='train':
      train_loss_metric.update_state(total_loss)
      train_accuracy_metric.update_state(labels, predictions)
    else:
      test_loss_metric.update_state(total_loss)
      test_accuracy_metric.update_state(labels, predictions)




  def linear_lr(self,data_len,batch_size,epochs,mode,peak_lr,repeat,interpolate):
    x=list(range(0,epochs+1,round(epochs*(1/repeat))))
    x= x + [epochs] if x[-1]!=epochs else x
    
    if mode=='stepup':
      z=[i+1 for i in x]
      x.extend(z[1:])
      x=sorted(x)[:-1]
      y=[peak_lr if i%2==0 else 0 for i in range(len(x))]
    if mode=='stepdown':
      z=[i+1 for i in x]
      x.extend(z[1:])
      x=sorted(x)[:-1]
      y=[peak_lr if i%2==1 else 0 for i in range(len(x))]

    if mode=='angledup':
      z=[round((x[i]+x[i+1])/2) for i in range(len(x)-1)]
      x.extend(z)
      x=sorted(x)
      y=[peak_lr if i%2==0 else 0 for i in range(len(x))]
    if mode=='angleddown':
      z=[round((x[i]+x[i+1])/2) for i in range(len(x)-1)]
      x.extend(z)
      x=sorted(x)
      y=[peak_lr if i%2==1 else 0 for i in range(len(x))]

    if mode=='constant':
      y=[peak_lr] * len(x)

    lr_schedule = lambda t: np.interp([t], x, y)[0]
    batches_per_epoch = data_len//batch_size + 1

    if interpolate:
      lr_func = lambda: lr_schedule(self.global_step/batches_per_epoch)/batch_size
    else:
      lr_func = lambda: lr_schedule(math.ceiling(self.global_step/batches_per_epoch))/batch_size
    return lr_func


  # @timer
  # def deep_learn(self,model, opt, loss, train, test):
  #   train_loss = test_loss = train_correct = test_correct  = 0.0
  #   tf.keras.backend.set_learning_phase(1)

  #   for x in tqdm(train):
  #     with tf.GradientTape() as tape:
  #       data=tf.cast(x['features'],tf.float16)
  #       labels=tf.cast(x['lables'],tf.int32)
  #       # print('predictioning')
  #       predictions = model(data)
  #       # print('got predictions')
  #       if self.trace:
  #         with self.train_summary_writer.as_default():        
  #           tf.summary.trace_export(name='Architecture',step=0)#,profiler_outdir=self.train_log)
  #         tf.summary.trace_off()
  #         self.trace=False
  #       loss = self.lossfunction(labels,predictions)
  #     grads = tape.gradient(loss, model.trainable_weights)
  #     self.global_step.assign_add(1)
  #     opt.apply_gradients(zip(grads, model.trainable_weights))
      
  #     # correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(predictions, axis = 1),tf.int32), labels), tf.float32))
  #     train_acc_metric(labels,predictions)

  #     train_loss = loss
  #     train_correct = train_acc_metric.result()
  #     train_acc_metric.reset_states()
  #     with self.train_summary_writer.as_default():
  #       tf.summary.scalar('LR', opt.learning_rate, step=self.global_step_reminder+self.global_step.numpy())
  #   train_metrics=(train_loss,train_correct)
    
    
  #   tf.keras.backend.set_learning_phase(0)
  #   for x in test:
  #     data=tf.cast(x['features'],tf.float16)
  #     labels=tf.cast(x['lables'],tf.int32)
  #     predictions = model(data)
  #     loss = self.lossfunction(labels,predictions)
  #     # correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(predictions, axis = 1),tf.int32), labels), tf.float32))
  #     val_acc_metric(labels,predictions)
  #     test_loss += loss
  #     test_correct += val_acc_metric.result()
  #     val_acc_metric.reset_states()
  #   test_metrics=(test_loss,test_correct)
  #   return (train_metrics,test_metrics)

  # @timer
  # def deep_learn(self,model, opt, loss, train, test):
  #   train_loss = test_loss = train_correct = test_correct  = 0.0
  #   tf.keras.backend.set_learning_phase(1)
    
  #   for x in tqdm(train):
  #     with tf.GradientTape() as tape:
  #       data=tf.cast(x['features'],tf.float16)
  #       labels=tf.cast(x['lables'],tf.int32)
  #       predictions = self.model(data)
  #       if self.trace:
  #         with self.train_summary_writer.as_default():        
  #           tf.summary.trace_export(name='Architecture',step=0)#,profiler_outdir=self.train_log)
  #         tf.summary.trace_off()
  #         self.trace=False
  #       loss = tf.reduce_sum(self.lossfunction(logits=predictions,labels=labels))
  #     grads = tape.gradient(loss, model.trainable_variables)
  #     self.global_step.assign_add(1)
  #     opt.apply_gradients(zip(grads, model.trainable_variables))
      
  #     correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(predictions, axis = 1),tf.int32), labels), tf.float32))
  #     train_loss += loss.numpy()
  #     train_correct += correct.numpy()
  #     with self.train_summary_writer.as_default():
  #       tf.summary.scalar('LR', opt.learning_rate, step=self.global_step_reminder+self.global_step.numpy())

  #   train_metrics=(train_loss,train_correct)
    
    
  #   tf.keras.backend.set_learning_phase(0)
  #   for x in test:
  #     data=tf.cast(x['features'],tf.float16)
  #     labels=tf.cast(x['lables'],tf.int32)
  #     predictions = model(data)
  #     loss = tf.reduce_sum(self.lossfunction(logits=predictions,labels=labels))
  #     correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(predictions, axis = 1),tf.int32), labels), tf.float32))
  #     test_loss += loss.numpy()
  #     test_correct += correct.numpy()
  #   test_metrics=(test_loss,test_correct)
  #   return (train_metrics,test_metrics)
