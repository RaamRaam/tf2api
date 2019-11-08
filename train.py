from .libraries import *
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, GlobalMaxPool2D, BatchNormalization
from tqdm import tqdm_notebook as tqdm

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




def linear_lr(data_len,batch_size,epochs,mode,peak_lr,repeat,interpolate):
  global global_step
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
def deep_learn(model, opt, loss, train, test):
  global global_step
  train_loss = test_loss = train_correct = test_correct  = 0.0
  tf.keras.backend.set_learning_phase(1)
  
  for x in tqdm(train):
    with tf.GradientTape() as tape:
      data=tf.cast(x['features'],tf.float16)
      labels=tf.cast(x['lables'],tf.int32)
      predictions = model(data)
      loss = tf.reduce_sum(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=predictions,labels=labels))
    grads = tape.gradient(loss, model.trainable_variables)
    global_step.assign_add(1)
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
