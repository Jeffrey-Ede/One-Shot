# -*- coding: utf-8 -*-
"""
Deep learning supersampling network for scanning transmission electron microscopy.

This is a standard convolutional network i.e. with batch norm and L2 regularization.

Acknowledgement: Initial testing of this network was performed with CIFAR-10 
in Google Colab
"""

import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

from tensorflow.contrib.layers.python.layers import initializers

import itertools

import time

from PIL import Image

cropsize = 256

## Load data
def flip_rotate(img):
    """Applies a random flip || rotation to the image, possibly leaving it unchanged"""

    choice = np.random.randint(0, 8)
    
    if choice == 0:
        return img
    if choice == 1:
        return np.rot90(img, 1)
    if choice == 2:
        return np.rot90(img, 2)
    if choice == 3:
        return np.rot90(img, 3)
    if choice == 4:
        return np.flip(img, 0)
    if choice == 5:
        return np.flip(img, 1)
    if choice == 6:
        return np.flip(np.rot90(img, 1), 0)
    if choice == 7:
        return np.flip(np.rot90(img, 1), 1)


def load_image(addr):
    """Read an image and make sure it is of the correct type. Optionally resize it"""

    if type(addr) == bytes:
        addr = addr.decode()

    img = np.load(addr)

    off_x = np.random.randint(0, 320-cropsize)
    off_y = np.random.randint(0, 320-cropsize)
    img = img[off_x:off_x+cropsize, off_y:off_y+cropsize]

    return img

def scale0to1(img):
    """Rescale image between 0 and 1"""

    img = img.astype(np.float32)

    min = np.min(img)
    max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.5)
    else:
        img = (img-min) / (max-min)

    return img.astype(np.float32)

def norm_img(img, min=None, max=None, get_min_and_max=False):
    
    if min == None:
        min = np.min(img)
    if max == None:
        max = np.max(img)

    if np.absolute(min-max) < 1.e-6:
        img.fill(0.)
    else:
        a = 0.5*(min+max)
        b = 0.5*(max-min)

        img = (img-a) / b

    if get_min_and_max:
        return img.astype(np.float32), (min, max)
    else:
        return img.astype(np.float32)

def preprocess(img):

    img[np.isnan(img)] = 0.
    img[np.isinf(img)] = 0.

    return img

def record_parser(record):
    """Parse files and generate lower quality images from them."""

    img = flip_rotate(preprocess(load_image(record)))
    lq = np.abs(img).astype(np.float32)

    img = np.angle(img).astype(np.float32)
    img = np.where(
        img < 0,
        2*img/np.pi + 1,
        1 - 2*img/np.pi
        )
    #img = (img.real/lq).astype(np.float32)
    #img = np.stack((img.real, img.imag), axis=-1).astype(np.float32)

    if np.sum(np.isfinite(img)) != np.product(img.shape) or np.sum(np.isfinite(lq)) != np.product(lq.shape):
        img = np.zeros((cropsize,cropsize))
        lq = np.zeros((cropsize,cropsize))

    return lq, img

def shaper(lq, img):

    lq = tf.reshape(lq, [cropsize, cropsize, 1])
    img = tf.reshape(img, [cropsize, cropsize, 1])

    return lq, img


def load_data(dir, subset, batch_size):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.list_files(dir+"*.npy")#subset+"/"+
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32, tf.float32]))
        dataset = dataset.map(shaper)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset.prefetch(buffer_size=10)

        iters = dataset.make_one_shot_iterator().get_next()

        #Add batch dimension size to graph
        for iter in iters:
            iter.set_shape([batch_size]+iter.get_shape().as_list()[1:])

        return iters

# Utility

def flip_and_rotate(x):
    """Random combination of flips and rotations."""

    for augmentator in [flip, rotate]:
        x = augmentator(x)

    return x


def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    # Rotate 0, 90, 180, 270 degrees
    return tf.image.rot90(x, tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))

def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x

def auto_name(name):
    """Append number to variable name to make it unique.
    
    Inputs:
        name: Initial variable name.
    Returns:
        Full variable name with number afterwards to make it unique.
    """

    scope = tf.contrib.framework.get_name_scope()
    vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope)

    names = [v.name for v in vars]
    
    #Increment variable number until unused name is found
    for i in itertools.count():
        short_name = name + "_" + str(i)
        sep = "/" if scope != "" else ""
        full_name = scope + sep + short_name
        if not full_name in [n[:len(full_name)] for n in names]:
            return short_name

def alrc(
    loss, 
    num_stddev_above=3, 
    num_stddev_below=None,
    decay=0.999, 
    mu1_start=25, 
    mu2_start=30**2, 
    in_place_updates=True
    ):
    """Adaptive learning rate clipping (ALRC) of outlier losses.
    
    Inputs:
        loss: Loss function to limit outlier losses of.
        num_stddev_above: Number of standard deviation above loss mean to limit it
        to. None for no clipping above.
        num_stddev_below: Number of standard deviation above loss mean to limit it
        to. None for no clipping below.
        decay: Decay rate for exponential moving averages used to track the first
        two raw moments of the loss.
        mu1_start: Initial estimate for the first raw moment of the loss.
        mu2_start: Initial estimate for the second raw moment of the loss.
        in_place_updates: If False, add control dependencies for moment tracking
        to tf.GraphKeys.UPDATE_OPS. This allows the control dependencies to be
        executed in parallel with other dependencies later.
    Return:
        Loss function with control dependencies for ALRC.
    """

    #Varables to track first two raw moments of the loss
    mu = tf.get_variable(
        auto_name("mu1"), 
        initializer=tf.constant(mu1_start, dtype=tf.float32))
    mu2 = tf.get_variable(
        auto_name("mu2"), 
        initializer=tf.constant(mu2_start, dtype=tf.float32))

    #Use capped loss for moment updates to limit the effect of outlier losses on the threshold
    sigma = tf.sqrt(mu2 - mu**2+1.e-8)

    #Clip loss if it's above or below the limits
    if num_stddev_below != None:
        alrc_below = tf.where(
            loss > mu-num_stddev_below*sigma, 
            loss, 
            loss/tf.stop_gradient(loss/(mu-num_stddev_below*sigma))
            )
    else:
        alrc_below = loss

    if num_stddev_above != None:
        loss = tf.where(
            loss < mu+num_stddev_above*sigma, 
            alrc_below, 
            loss/tf.stop_gradient(loss/(mu+num_stddev_above*sigma))
            )

    #Update moment moving averages
    mean_loss = tf.reduce_mean(loss)
    mean_loss2 = tf.reduce_mean(loss**2)

    update_ops = [
        mu.assign(decay*mu+(1-decay)*mean_loss), 
        mu2.assign(decay*mu2+(1-decay)*mean_loss2)
    ]

    if in_place_updates:
        with tf.control_dependencies(update_ops):
            loss = tf.identity(loss)
    else:
        #Control dependencies that can be executed in parallel with other update
        #ops. Often, these dependencies are added to train ops e.g. alongside
        #batch normalization update ops.
        for update_op in update_ops:
            tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, update_op)
        
    return loss
  

def spectral_norm(w, iteration=1, in_place_updates=False):
    """Spectral normalization. It imposes Lipschitz continuity by constraining the
    spectral norm (maximum singular value) of weight matrices.

    Inputs:
        w: Weight matrix to spectrally normalize.
        iteration: Number of times to apply the power iteration method to 
        enforce spectral norm.

    Returns:
        Weight matrix with spectral normalization control dependencies.
    """

    w0 = w
    w_shape = w.shape.as_list()
    w = tf.reshape(w, [-1, w_shape[-1]])


    u = tf.get_variable(auto_name("u"), 
                       [1, w_shape[-1]], 
                       initializer=tf.random_normal_initializer(mean=0.,stddev=0.03), 
                       trainable=False)

    u_hat = u
    v_hat = None
    for i in range(iteration):
        """
        power iteration
        Usually iteration = 1 will be enough
        """
        v_ = tf.matmul(u_hat, tf.transpose(w))
        v_hat = tf.nn.l2_normalize(v_)

        u_ = tf.matmul(v_hat, w)
        u_hat = tf.nn.l2_normalize(u_)

    u_hat = tf.stop_gradient(u_hat)
    v_hat = tf.stop_gradient(v_hat)

    sigma = tf.matmul(tf.matmul(v_hat, w), tf.transpose(u_hat))

    if in_place_updates:
        #In-place control dependencies bottlenect training
        with tf.control_dependencies([u.assign(u_hat)]):
            w_norm = w / sigma
            w_norm = tf.reshape(w_norm, w_shape)
    else:
        #Execute control dependency in parallel with other update ops
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u.assign(u_hat))

        w_norm = w / sigma
        w_norm = tf.reshape(w_norm, w_shape)

    return w_norm


def spectral_norm_conv(inputs,
                       num_outputs, 
                       stride=1, 
                       kernel_size=3, 
                       padding='VALID',
                       biases_initializer=tf.zeros_initializer()):
    """Convolutional layer with spectrally normalized weights."""
    
    w = tf.get_variable(auto_name("kernel"), shape=[kernel_size, kernel_size, inputs.get_shape()[-1], num_outputs])

    x = tf.nn.conv2d(input=inputs, filter=spectral_norm(w), 
                        strides=[1, stride, stride, 1], padding=padding)

    if biases_initializer != None:
        b = tf.get_variable(auto_name("bias"), [num_outputs], initializer=biases_initializer)
        x = tf.nn.bias_add(x, b)

    return x


def conv(
    inputs, 
    num_outputs, 
    kernel_size=3, 
    stride=1, 
    padding='SAME',
    data_format="NHWC",
    actv_fn=tf.nn.relu, 
    is_batch_norm=True,
    is_spectral_norm=False,
    is_depthwise_sep=False,
    extra_batch_norm=False,
    biases_initializer=tf.zeros_initializer,
    weights_initializer=initializers.xavier_initializer,
    transpose=False,
    is_training=True
    ):
    """Convenience function for a strided convolutional or transpositional 
    convolutional layer.
    
    Intro: https://towardsdatascience.com/intuitively-understanding-convolutions-for-deep-learning-1f6f42faee1.

    The order is: Activation (Optional) -> Batch Normalization (optional) -> Convolutions.

    Inputs: 
        inputs: Tensor of shape `[batch_size, height, width, channels]` to apply
        convolutions to.
        num_outputs: Number of feature channels to output.
        kernel_size: Side lenth of square convolutional kernels.
        stride: Distance between convolutional kernel applications.
        padding: 'SAME' for zero padding where kernels go over the edge.
        'VALID' to discard features where kernels go over the edge.
        activ_fn: non-linearity to apply after summing convolutions. 
        is_batch_norm: If True, add batch normalization after activation.
        is_spectral_norm: If True, spectrally normalize weights.
        is_depthwise_sep: If True, depthwise separate convolutions into depthwise
        spatial convolutions, then 1x1 pointwise convolutions.
        extra_batch_norm: If True and convolutions are depthwise separable, implement
        batch normalization between depthwise and pointwise convolutions.
        biases_initializer: Function to initialize biases with. None for no biases.
        weights_initializer: Function to initialize weights with. None for no weights.
        transpose: If True, apply convolutional layer transpositionally to the
        described convolutional layer.
        is_training: If True, use training specific operations e.g. batch normalization
        update ops.

    Returns:
        Output of convolutional layer.
    """

    x = inputs

    num_spatial_dims = len(x.get_shape().as_list()) - 2

    if biases_initializer == None:
        biases_initializer = lambda: None
    if weights_initializer == None:
        weights_initializer = lambda: None

    if not is_spectral_norm:
        #Convolutional layer without spectral normalization

        if transpose:
            stride0 = 1
            if type(stride) == list or is_depthwise_sep or stride % 1:
                #Apparently there is no implementation of transpositional  
                #depthwise separable convolutions, so bilinearly upsample then 
                #depthwise separably convolute
                if kernel_size != 1:
                    x = tf.image.resize_bilinear(
                        images=x,
                        size=stride if type(stride) == list else \
                        [int(stride*d) for d in x.get_shape().as_list()[1:3]],
                        align_corners=True
                        )
                stride0 = stride      
                stride = 1

            if type(stride0) == list and not is_depthwise_sep:
                layer = tf.contrib.layers.conv2d
            elif is_depthwise_sep:
                layer = tf.contrib.layers.separable_conv2d
            else:
                layer = tf.contrib.layers.conv2d_transpose

            x = layer(
                inputs=x,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                data_format=data_format,
                activation_fn=None,
                weights_initializer=weights_initializer(),
                biases_initializer=biases_initializer())
               
            if type(stride0) != list:
              if (is_depthwise_sep or stride0 % 1) and kernel_size == 1:
                  x = tf.image.resize_bilinear(
                      images=x,
                      size=[int(stride0*d) for d in x.get_shape().as_list()[1:3]],
                      align_corners=True
                      )   
        else:
            if num_spatial_dims == 1:
                layer = tf.contrib.layers.conv1d
            elif num_spatial_dims == 2:
                if is_depthwise_sep:
                    layer = tf.contrib.layers.separable_conv2d
                else:
                    layer = tf.contrib.layers.conv2d
            x = layer(
                inputs=x,
                num_outputs=num_outputs,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                data_format=data_format,
                activation_fn=None,
                weights_initializer=weights_initializer(),
                biases_initializer=biases_initializer())
    else:
        #Weights are spectrally normalized
        x = spectral_norm_conv(
            inputs=x,
            num_outputs=num_outputs, 
            stride=stride, 
            kernel_size=kernel_size, 
            padding=padding,
            biases_initializer=biases_initializer())

    if actv_fn:
        x = actv_fn(x)

    if is_batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training)

    return x


def residual_block(inputs, skip=2, is_training=True):
    """Residual block whre the input is added to the signal after skipping some
    layers. This architecture is good for learning purturbative transformations. 
    If no layer is provided, it defaults to a convolutional layer.

    Deep residual learning: https://arxiv.org/abs/1512.03385.

    Inputs:
        inputs: Tensor to apply residual block to. Outputs of every layer will 
        have the same shape.
        skip: Number of layers to skip before adding input to layer output.
        layer: Layer to apply in residual block. Defaults to convolutional 
        layer. Custom layers must support `inputs`, `num_outputs` and `is_training`
        arguments.

    Returns:
        Final output of residual block.
    """

    x = x0 = inputs

    def layer(inputs, num_outputs, is_training, is_batch_norm):
        
        x = conv(
            inputs=inputs, 
            num_outputs=num_outputs,
            is_training=is_training
            )

        return x

    for i in range(skip):
        x = layer(
            inputs=x, 
            num_outputs=x.get_shape()[-1], 
            is_training=is_training,
            is_batch_norm=i < skip - 1
        )

    x += x0

    x = tf.contrib.layers.batch_norm(x, is_training=is_training)

    return x
  
  
def transpose_Xception(
    inputs, 
    num_outputs, 
    stride=2, 
    actv_fn=tf.nn.relu,
    is_batch_norm=True,
    is_training=True
    ):
  """Transpositional Xception block for upsampling; rather than downsampling."""
  
  x = inputs
  
  if actv_fn:
      x = actv_fn(x)

  if is_batch_norm:
      x = tf.contrib.layers.batch_norm(x, is_training=is_training)
  
  x0 = conv(
      inputs=x, 
      num_outputs=num_outputs, 
      kernel_size=1,
      stride=stride,
      is_batch_norm=False,
      is_depthwise_sep=True,
      transpose=True
  )
  
  x = conv(
      inputs=x, 
      num_outputs=num_outputs, 
      kernel_size=3,
      stride=stride,
      is_batch_norm=False,
      is_depthwise_sep=True,
      transpose=True
  )
  
  x = conv(
      inputs=x, 
      num_outputs=num_outputs,
      is_depthwise_sep=True,
  )
  x = conv(
      inputs=x, 
      num_outputs=num_outputs,
      is_depthwise_sep=True,
  )
  print(x0, x)
  x += x0
  
  return x


def generator(inputs, num_outputs, is_training, is_depthwise_sep=False):
    """Convolutional neural network (CNN) for image supersampling.
  
    Args:
    Inputs: Images tensor with shape [batch_size, heigh, width, channels].
    num_outputs: Number of channels in network output.
    is_training: Bool indicating whether to use training operations
    
    Returns:
    Super-sampled images
    """

    x = inputs

    x = tf.contrib.layers.batch_norm(x, is_training=is_training)

    x = conv(
        x, 
        num_outputs=32,
        is_training=is_training
        )

    
    #Encoder
    layers = []
    for i in range(1, 5):

        layers.append(x)

        x = conv(
            x, 
            num_outputs=32*2**i, 
            stride=2,
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training
        )


    #Residual blocks
    for _ in range(6): #Number of blocks
        x = residual_block(
            x, 
            skip=3,
            is_training=is_training
        )


    #Decoder
    for i, layer in zip(range(3, -1, -1), reversed(layers)):

        x = conv(
            x, 
            num_outputs=32*2**i, 
            stride=2,
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training,
            transpose=True
        )


    x = conv(
        x, 
        num_outputs=32, 
        is_depthwise_sep=is_depthwise_sep,
        is_training=is_training,
    )

  
    #Project features onto output image
    x = conv(
        x,
        num_outputs=num_outputs,
        biases_initializer=None,
        actv_fn=None,
        is_batch_norm=False,
        is_training=is_training
    )
    
    return x



def generator1(inputs, num_outputs, is_training, is_depthwise_sep=False):
    """Convolutional neural network (CNN) for image supersampling.
  
    Args:
    Inputs: Images tensor with shape [batch_size, heigh, width, channels].
    num_outputs: Number of channels in network output.
    is_training: Bool indicating whether to use training operations
    
    Returns:
    Super-sampled images
    """

    x = inputs

    x = tf.contrib.layers.batch_norm(x, is_training=is_training)

    x = conv(
        x, 
        num_outputs=32,
        is_training=is_training
        )

    
    #Encoder
    layers = []
    for i in range(1, 5):

        layers.append(x)

        x = conv(
            x, 
            num_outputs=32*2**i, 
            stride=2,
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training
        )


    #Residual blocks
    for _ in range(6): #Number of blocks
        x = residual_block(
            x, 
            skip=3,
            is_training=is_training
        )


    #Decoder
    for i, layer in zip(range(3, -1, -1), reversed(layers)):

        x = conv(
            x, 
            num_outputs=32*2**i, 
            stride=2,
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training,
            transpose=True
        )
        x = tf.concat([x, layer], axis=-1)


    x = conv(
        x, 
        num_outputs=32, 
        is_depthwise_sep=is_depthwise_sep,
        is_training=is_training,
    )

  
    #Project features onto output image
    x = conv(
        x,
        num_outputs=num_outputs,
        biases_initializer=None,
        actv_fn=None,
        is_batch_norm=False,
        is_training=is_training
    )
    
    return x


def discriminator(inputs):
  """Predict whether inputs are real or generated."""
  
  x = inputs
  
  for layer_num, size in enumerate([64*2**i for i in range(4)]):
    x = conv(
        inputs=x, 
        num_outputs=size, 
        kernel_size=4, 
        stride=2,
        actv_fn=tf.nn.leaky_relu,
        is_batch_norm=False,
        is_spectral_norm=True
    )
    
    if layer_num == 1:
      layer = x
    
  x = tf.reduce_sum(
      x, 
      axis=[i for i in range(1, len(x.get_shape().as_list()))]
  )

  #So predictions have a features dimension. Not strictly necessary; however,
  #it may improve maintainability
  x = tf.expand_dims(x, axis=-1)
  
  return x, layer

def configure(
    inputs, 
    target_outputs, 
    is_training, 
    learning_rate, 
    beta1,
    is_depthwise_sep
  ):
  """Operations to calculate network losses and run training operations."""

  output = generator(
      inputs=inputs, 
      num_outputs=target_outputs.get_shape().as_list()[-1], 
      is_training=is_training,
      is_depthwise_sep=is_depthwise_sep
  )
  
  #Mean squared errors
  mse = 20*tf.reduce_mean( tf.square(output - target_outputs), axis=[1,2,3] )
  
  alrc_mse = mse#alrc(mse)
  alrc_mse = tf.reduce_mean(alrc_mse)

  mse = tf.reduce_mean(mse)

  #L2 regularization
  l2_loss = 0.#tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
  loss = alrc_mse + 1.e-5*l2_loss
  
  #Create optimizer for stochastic gradient descent (SGD)
  optimizer = tf.train.AdamOptimizer(
      learning_rate=learning_rate,
      beta1=beta1)
  
  #Update ops for batch normalisation
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  
  with tf.control_dependencies(update_ops):
    train_op = optimizer.minimize(loss)
    
  return train_op, mse, output

def experiment(report_every_n=100):
  """Run training operations, then validate.
  
  Args:
    report_every_n: Print loss every n training operations. 0 for no printing.
    
  Returns:
    Validation top-1 accuracy and a numpy array of training losses
  """
  
  #Placeholders to feed hyperparameters into graph
  learning_rate_ph = tf.placeholder(tf.float32, name="learning_rate")
  beta1_ph = tf.placeholder(
      tf.float32, 
      shape=(),
      name="beta1")
  is_training_ph = tf.placeholder(
      tf.bool, 
      name="is_training")
  mode_ph = tf.placeholder(
      tf.int32, 
      name="mode")

  def load_data_subset(subset):
      return load_data(
          dir="//Desktop-sa1evjv/f/wavefunctions/wavefunctions/",
          subset=subset, 
          batch_size=16
          )

  inputs, target_outputs = tf.case(
      {tf.equal(mode_ph, 0): lambda: load_data_subset("train"),
       tf.equal(mode_ph, 1): lambda: load_data_subset("val"),
       tf.equal(mode_ph, 2): lambda: load_data_subset("test")}
      )
  
  #Describe learning policy
  start_iter = 0
  train_iters = 1_000_000
  val_iters = 1_000
  
  learning_rate = 0.0003
  beta1 = 0.9
  
  #Configure operations
  train_op, loss, output = configure(
      inputs=inputs,
      target_outputs=target_outputs,
      is_training=is_training_ph, 
      learning_rate=learning_rate_ph, 
      beta1=beta1_ph,
      is_depthwise_sep=False
  )
  
  #Tensors to dump as visual output
  first_image = inputs[0]
  first_target_output = target_outputs[0]
  first_output = output[0]

  #Session configuration
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True #Only use required GPU memory
  config.gpu_options.force_gpu_compatible = True

  model_dir = "//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/wavefunctions/1/"
  
  #"//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/magnifier/"

  saver = tf.train.Saver(max_to_keep=1)

  log_filepath = model_dir + "log.txt"
  save_period = 1; save_period *= 3600
  with tf.Session(config=config) as sess, open(log_filepath, "a") as log_file:

    #Initialize network parameters
    feed_dict = feed_dict = {
      is_training_ph: np.bool(True),
      learning_rate_ph: np.float32(learning_rate),
      beta1_ph: np.float32(beta1),
      mode_ph: np.int32(0)
    }
    
    if start_iter:
        saver.restore(
            sess, 
            tf.train.latest_checkpoint(model_dir+"model/")
            )
    else:
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    #Training
    training_losses = np.zeros((train_iters-start_iter))
    time0 = time.time()
    for iter in range(start_iter, train_iters):
      
      is_halfway = iter >= train_iters // 2

      lr = learning_rate * 0.5**(iter//(train_iters//6))
      is_training = not is_halfway
      beta1 = 0.5 if is_halfway else 0.9
      
      #Feed values into training operations
      feed_dict = {
          is_training_ph: np.bool(is_training),
          learning_rate_ph: np.float32(lr),
          beta1_ph: np.float32(beta1),
          mode_ph: np.int32(0)
      }
      
      if iter in [0, 100, 500] or not iter % 25_000 or (0 <= iter < 10_000 and not iter % 1000) or iter == start_iter:
        _, step_loss, [step_image, step_target_output, step_output] = sess.run([
            train_op, 
            loss,
            [first_image, first_target_output, first_output]
            ],
            feed_dict=feed_dict)
          
        save_input_loc = model_dir+"input-"+str(iter)+".tif"
        save_truth_loc = model_dir+"truth-"+str(iter)+".tif"
        save_output_loc = model_dir+"output-"+str(iter)+".tif"
        Image.fromarray(step_image.reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
        Image.fromarray(step_target_output.reshape(cropsize, cropsize).astype(np.float32)).save( save_truth_loc )
        Image.fromarray(step_output.reshape(cropsize, cropsize).astype(np.float32)).save( save_output_loc )
      else:
        _, step_loss = sess.run([train_op, loss], feed_dict=feed_dict)
      
      training_losses[iter-start_iter] = step_loss
      
      output = f"Iter: {iter}, Loss: {step_loss}"
      if report_every_n:
        if not iter % report_every_n:
          print(output)

      log_file.write(output)

      if time.time() >= time0 + save_period:
          saver.save(sess, save_path=model_dir+"model/model", global_step=iter)
          time0 = time.time()
      
    #Validation - super important!
    dataset.initialize_iterator(sess=sess, mode="val")
      
    val_loss = 0.
    for iter in range(val_iters):
      
      feed_dict = {
          is_training_ph: np.bool(False),
          mode_ph: np.int32(1)
      }
      
      step_loss = sess.run(loss, feed_dict=feed_dict)
      val_loss += step_loss
      
    val_loss /= val_iters
    
  return val_loss, training_losses

#Reset so graph nodes to not accumulate in ipynb session memory.
tf.reset_default_graph()

#Run your experiment!
val_loss, training_losses = experiment(report_every_n=1)

#Report performance on validation set
print(f"Validation loss: {val_loss}")