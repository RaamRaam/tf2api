from .libraries import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalMaxPool2D, BatchNormalization
from tqdm import tqdm_notebook as tqdm
import math
import datetime,time

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


class RamNet(tf.keras.Model):
  def __init__(self):
    super().__init__()
    pool = tf.keras.layers.MaxPooling2D()
    self.conv1 = tf2x.ConvBN(10)
    self.conv2 = tf2x.ConvBN(10)
    self.maxpool1 = tf.keras.layers.MaxPooling2D()
    self.conv3 = tf2x.ConvBN(10)
    self.conv4 = tf2x.ConvBN(10)
    self.maxpool2 = tf.keras.layers.MaxPooling2D()
    self.conv5 = tf2x.ConvBN(10)
    self.conv6 = tf2x.ConvBN(10)
    self.avgpool = tf.keras.layers.GlobalAvgPool2D()
    
  # @tf.function(input_signature=[tf.TensorSpec(shape=(None,28,28,1), dtype=tf.float16)])
  @tf.function
  def call(self, x):
    with tf.name_scope('block'):
      h=self.conv1(x) #28x28
      h=self.conv2(h) #28x28
      h=self.maxpool1(h) #14x14
      h=self.conv3(h) #14x14
      h=self.conv4(h) #14x14
      h=self.maxpool2(h) #7x7
      h=self.conv5(h) #7x7
      # h=self.conv6(h) #7x7
      h=self.avgpool(h)
    return h


class train(object):
  def __init__(self, hparams):
    self.name=hparams['NAME']
    self.model=hparams['MODEL']
    self.train_ds=hparams['TRAIN_DS']
    self.train_ds=hparams['TEST_DS']


    self.epochs=hparams['EPOCHS']
    self.batch_size=hparams['BATCH_SIZE']

    self.lr_peak=hparams['LR_PEAK']
    self.lr_repeat=hparams['LR_REPEAT']
    self.lr_interpolate=hparams['LR_INTERPOLATE']
    self.lr_modes=['constant','stepup','stepdown','angledup','angleddown']
    self.lr_mode=hparams['LR_MODE']
    self.lr=self.linear_lr(self.train_ds.length,self.batch_size,self.epochs,self.lr_mode,self.lr_peak,self.lr_repeat,self.lr_interpolate)

    self.log_path=hparams['LOG_PATH']
    self.train_log=self.log_path+'/train_log'
    self.test_log=self.log_path+'/test_log'
    self.train_summary_writer = tf.summary.create_file_writer(train_log)
    self.test_summary_writer = tf.summary.create_file_writer(test_log)

    self.global_step = tf.Variable(-1)

    self.optimizer=tf.keras.optimizers.SGD(self.lr)

  def call(self):
    tf.keras.backend.set_floatx('float16')
    test_ds_batches = test_ds.ds.shuffle(self.batch_size).batch(self.batch_size).prefetch(self.batch_size)
    print('training....')
    t = time.time()
    for epoch in range(self.epochs):
      train_ds_batches = train_ds.ds.shuffle(self.train_ds.length).batch(self.batch_size).prefetch(self.batch_size)
      learnings=self.deep_learn(self.model, opt, None, train_ds_batches, test_ds_batches)
      lr=opt.learning_rate*self.batch_size
      train_loss=learnings[0][0]/self.train_ds.length
      train_acc=learnings[0][1]/self.train_ds.length
        
      val_loss=learnings[1][0]/self.test_ds.length
      val_acc=learnings[1][1]/self.test_ds.length
      time_taken=time.time() - t
      print("epoch: %0.3d \t lr:%0.2f \t train loss:%0.2f \t train acc:%2.2f \t  val loss:%0.2f \t val acc:%2.2f \t time:%0.2f" % (epoch ,lr,train_loss,train_acc*100,val_loss,val_acc*100,time_taken))

      with train_summary_writer.as_default():
        tf.summary.scalar('loss', train_loss, step=epoch)
        tf.summary.scalar('accuracy', train_acc, step=epoch)
        # tf.summary.scalar('LR', lr, step=epoch)
      with test_summary_writer.as_default():
        tf.summary.scalar('loss', val_loss, step=epoch)
        tf.summary.scalar('accuracy', val_acc, step=epoch)


  def linear_lr(self,data_len,batch_size,epochs,mode,peak_lr,repeat,interpolate):
    # global global_step
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
      lr_func = lambda: lr_schedule(global_step/batches_per_epoch)/batch_size
    else:
      lr_func = lambda: lr_schedule(math.ceiling(global_step/batches_per_epoch))/batch_size
    return lr_func

  @timer
  def deep_learn(self,model, opt, loss, train, test):
    # global global_step
    train_loss = test_loss = train_correct = test_correct  = 0.0
    tf.keras.backend.set_learning_phase(1)
    
    for x in tqdm(train):
      with tf.GradientTape() as tape:
        data=tf.cast(x['features'],tf.float16)
        labels=tf.cast(x['lables'],tf.int32)
        predictions = model(data)
        loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions,labels=labels))
      grads = tape.gradient(loss, model.trainable_variables)
      self.global_step.assign_add(1)
      opt.apply_gradients(zip(grads, model.trainable_variables))
      
      correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(predictions, axis = 1),tf.int32), labels), tf.float32))
      train_loss += loss.numpy()
      train_correct += correct.numpy()
    train_metrics=(train_loss,train_correct)
    
    
    tf.keras.backend.set_learning_phase(0)
    for x in test:
      data=tf.cast(x['features'],tf.float16)
      labels=tf.cast(x['lables'],tf.int32)
      predictions = model(data)
      loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions,labels=labels))
      correct = tf.reduce_sum(tf.cast(tf.math.equal(tf.cast(tf.argmax(predictions, axis = 1),tf.int32), labels), tf.float32))
      test_loss += loss.numpy()
      test_correct += correct.numpy()
    test_metrics=(test_loss,test_correct)
    return (train_metrics,test_metrics)

