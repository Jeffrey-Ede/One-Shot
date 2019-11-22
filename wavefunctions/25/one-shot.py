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

import queue


EXPER_NUM = 25
cropsize = 128#192#224#256
use_batch_norm = True
batch_norm_decay = 0.999
use_vbn = False
use_instance_norm = False#True
adversarial = True
use_spectral_norm = False#True
use_gradient_penalty = True#True]
standard_wass = False
use_l2_loss = False


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

    img = flip_rotate(img)

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

history = queue.Queue()
def record_parser(record):
    """Parse files and generate lower quality images from them."""

    if np.random.randint(0,2) and history.qsize() > 100:

        try: 
            (lq, img) = history.get()
            return lq, img
        except:
            pass

    img = flip_rotate(preprocess(load_image(record)))
    lq = np.abs(img).astype(np.float32)

    #img = np.angle(img).astype(np.float32)
    #img = np.where(
    #    img < 0,
    #    2*img/np.pi + 1,
    #    1 - 2*img/np.pi
    #    )
    #img = (img.real/lq).astype(np.float32)

    angle = np.angle(img)
    img = np.stack((np.cos(angle), np.sin(angle)), axis=-1).astype(np.float32)

    if np.sum(np.isfinite(img)) != np.product(img.shape) or np.sum(np.isfinite(lq)) != np.product(lq.shape):
        img = np.zeros((cropsize,cropsize,2))
        lq = np.zeros((cropsize,cropsize))

    try:
        history.put( (lq, img) )
    except:
        pass
        
    return lq, img

def shaper(lq, img):

    lq = tf.reshape(lq, [cropsize, cropsize, 1])
    img = tf.reshape(img, [cropsize, cropsize, 2])

    return lq, img


def load_data(dir, subset, batch_size):
    """Create a dataset from a list of filenames and shard batches from it"""

    with tf.device('/cpu:0'):

        dataset = tf.data.Dataset.list_files(dir+subset+"/"+"*.npy")
        dataset = dataset.shuffle(buffer_size=5000)
        dataset = dataset.repeat()
        dataset = dataset.map(
            lambda file: tf.py_func(record_parser, [file], [tf.float32, tf.float32])
            )
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
        name: Start of variable name.

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
    num_stddev=3, 
    decay=0.997, 
    mu1_start=5, 
    mu2_start=7**2, 
    in_place_updates=True
    ):
    """Adaptive learning rate clipping (ALRC) of outlier losses.
    
    Inputs:
        loss: Loss function to limit outlier losses of.
        num_stddev: Number of standard deviation above loss mean to limit it
        to.
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
    loss = tf.where(loss < mu+num_stddev*sigma, 
                   loss, 
                   loss/tf.stop_gradient(loss/(mu+num_stddev*sigma)))

    #Update moment moving averages
    mean_loss = tf.reduce_mean(loss)
    mean_loss2 = tf.reduce_mean(loss**2)
    update_ops = [mu.assign(decay*mu+(1-decay)*mean_loss), 
                  mu2.assign(decay*mu2+(1-decay)*mean_loss2)]
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
  

def spectral_norm(w, iteration=1, in_place_updates=True):
    """Spectral normalization. It imposes Lipschitz continuity by constraining the
    spectral norm (maximum singular value) of weight matrices.

    Inputs:
        w: Weight matrix to spectrally normalize.
        iteration: Number of times to apply the power iteration method to 
        enforce spectral norm.

    Returns:
        Weight matrix with spectral normalization control dependencies.
    """

    if not use_spectral_norm:
        return w

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


def spectral_norm_dense(
    inputs, 
    num_outputs,
    biases_initializer=tf.zeros_initializer()
    ):

    w = tf.get_variable(auto_name("weights"), shape=[inputs.get_shape()[-1], num_outputs])

    x = tf.matmul(inputs, spectral_norm(w))

    if biases_initializer != None:
        b = tf.get_variable(auto_name("bias"), [num_outputs], initializer=biases_initializer)
        x = tf.nn.bias_add(x, b)

    return x

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


std_actv = lambda x: tf.nn.leaky_relu(x, alpha=0.1)

def conv(
    inputs, 
    num_outputs, 
    kernel_size=3, 
    stride=1, 
    padding='SAME',
    data_format="NHWC",
    actv_fn=std_actv, 
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

    if is_batch_norm and use_batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=is_training)

    return x


def residual_block(inputs, skip=3, is_training=True):
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

    def layer(inputs, num_outputs, is_training, is_batch_norm, actv_fn):
        
        x = conv(
            inputs=inputs, 
            num_outputs=num_outputs,
            is_training=is_training,
            actv_fn=actv_fn
            )

        return x

    for i in range(skip):
        x = layer(
            inputs=x, 
            num_outputs=x.get_shape()[-1], 
            is_training=is_training,
            is_batch_norm=i < skip - 1,
            actv_fn=tf.nn.relu
        )

    x += x0

    if use_batch_norm:
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

    base_size = 32

    x = inputs

    x = tf.contrib.layers.batch_norm(x, is_training=is_training)

    x = conv(
        x, 
        num_outputs=32,
        is_training=is_training
        )
    
    #Encoder
    for i in range(1, 4):

        x = conv(
            x, 
            num_outputs=base_size*2**i, 
            stride=2,
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training,
            actv_fn=std_actv
        )

        if i == 2:
            low_level = x

    #Residual blocks
    for _ in range(6): #Number of blocks
        x = residual_block(
            x, 
            skip=3,
            is_training=is_training
        )


    #Decoder
    for i in range(2, -1, -1):

        x = conv(
            x, 
            num_outputs=base_size*2**i, 
            stride=2,
            is_depthwise_sep=is_depthwise_sep,
            is_training=is_training,
            transpose=True,
            actv_fn=std_actv
        )

        #if x.get_shape().as_list() == low_level.get_shape().as_list(): #Easy way to find concat level!
        #    x = tf.concat([x, low_level], axis=-1)

        #    for _ in range(3):
        #        x = conv(
        #            x, 
        #            num_outputs=base_size*2**i, 
        #            is_depthwise_sep=is_depthwise_sep,
        #            is_training=is_training,
        #        )

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
        is_batch_norm=True,
        is_training=is_training
    )

    x /= tf.sqrt(1.e-8 + tf.reduce_sum(x**2, axis=-1, keepdims=True))
    x0 = x
    x *= inputs

    return x, x0



def res_block(x, num_outputs, s=1):

    x0 = x
    start_channels = x.get_shape().as_list()[-1]

    if num_outputs != start_channels:
        x0 = conv(
            inputs=x0, 
            num_outputs=num_outputs, 
            kernel_size=1, 
            stride=1,
            actv_fn=None,
            is_batch_norm=False,
            is_spectral_norm=True
        )

    x = conv(
        inputs=x, 
        num_outputs=start_channels, 
        kernel_size=3, 
        stride=1,
        actv_fn=tf.nn.relu,
        is_batch_norm=False,
        is_spectral_norm=True
    )

    x = conv(
        inputs=x, 
        num_outputs=num_outputs, 
        kernel_size=3, 
        stride=1,
        actv_fn=tf.nn.relu,
        is_batch_norm=False,
        is_spectral_norm=True
    )

    x += x0

    if s > 1:
        #x0 = tf.layers.average_pooling2d(x0, s, s)
        x = tf.layers.average_pooling2d(x, s, s)

    return x

def large_discriminator(inputs):
    #Based on https://arxiv.org/pdf/1802.05637.pdf

    x = inputs

    for i in range(4):

        #x = res_block(x, 64*2**i, s=2)
        x = conv(
            inputs=x, 
            num_outputs=50*2**i, 
            kernel_size=4, 
            stride=1,
            actv_fn=tf.nn.leaky_relu,
            is_batch_norm=False,
            is_spectral_norm=True
        )

        x = conv(
            inputs=x, 
            num_outputs=50*2**i, 
            kernel_size=4, 
            stride=2,
            actv_fn=tf.nn.leaky_relu,
            is_batch_norm=False,
            is_spectral_norm=True
            )

    #for _ in range(4):
    #    x = res_block(x, 512)

    #for _ in range(3):
    #    x = conv(
    #        inputs=x, 
    #        num_outputs=400, 
    #        kernel_size=4, 
    #        stride=1,
    #        actv_fn=tf.nn.leaky_relu,
    #        is_batch_norm=False,
    #        is_spectral_norm=True
    #    )

    ##x = res_block(x, 1024, s=2)

    #x = conv(
    #    inputs=x, 
    #    num_outputs=800, 
    #    kernel_size=4, 
    #    stride=2,
    #    actv_fn=tf.nn.leaky_relu,
    #    is_batch_norm=False,
    #    is_spectral_norm=True
    #)

    x = conv(
        inputs=x, 
        num_outputs=800, 
        kernel_size=4, 
        stride=2,
        actv_fn=tf.nn.leaky_relu,
        is_batch_norm=False,
        is_spectral_norm=True
    )

    #x = tf.layers.flatten(x)
    #x = tf.expand_dims(tf.reduce_sum(x, axis=[1,2,3]), axis=-1)

    x = tf.reduce_sum(x, axis=[1,2])

    #x = tf.contrib.layers.fully_connected(x, 1)
    #x = tf.contrib.layers.fully_connected(x, 1, activation_fn=None, biases_initializer=None)

    #x = spectral_norm_dense(x, 2048, biases_initializer=None)
    #x = spectral_norm_dense(x, 1024, biases_initializer=None)


    #x = tf.expand_dims(tf.reduce_mean(x, axis=[1,2,3]), axis=-1)

    x = spectral_norm_dense(x, 1, biases_initializer=None)

    #x += 0.5

    #x = 0.5 - 0.1 + 1.1*tf.sigmoid(x)

    #x = 1 + tf.nn.elu(x)

    return x


def configure(
    inputs, 
    batch_size,
    target_outputs, 
    is_training, 
    learning_rate, 
    beta1,
    is_depthwise_sep,
    decay,
    gen_scale
    ):
  """Operations to calculate network losses and run training operations."""

  target_outputs0 = target_outputs

  with tf.variable_scope("gen"):
      output0, phase_components = generator(
          inputs=inputs, 
          num_outputs=target_outputs.get_shape().as_list()[-1], 
          is_training=is_training,
          is_depthwise_sep=is_depthwise_sep
      )
      output = output0
  
  if adversarial:
      #Theoretical argument for EMA tracking is in https://openreview.net/pdf?id=SJgw_sRqFQ

      #with tf.variable_scope("tracking/gen"):
      #    tracking_output = generator(
      #        inputs=inputs, 
      #        num_outputs=target_outputs.get_shape().as_list()[-1], 
      #        is_training=is_training,
      #        is_depthwise_sep=is_depthwise_sep
      #    )

      
      def amp(x):
          return 1 + tf.sqrt(1.e-8 + tf.reduce_sum(x**2, axis=-1, keepdims=True))

      output = tf.concat([inputs, phase_components], axis=-1)
      target_outputs = tf.concat([inputs, target_outputs], axis=-1)

      if use_gradient_penalty:
          x_hat = output + tf.random_uniform(output.get_shape().as_list())*(target_outputs-output)
          discr_batch = tf.concat([output, target_outputs, x_hat], axis=0)
      else:
          discr_batch = tf.concat([output, target_outputs], axis=0)

      with tf.variable_scope("main/discr"):
          preds = large_discriminator(discr_batch)

      #with tf.variable_scope("tracking/discr"):
      #    track_pred = large_discriminator(output)

      fake_pred = preds[:batch_size]
      real_pred = preds[batch_size:2*batch_size]

      if use_gradient_penalty:
        x_hat_pred = preds[2*batch_size:3*batch_size]

      if use_gradient_penalty:
          grad = tf.gradients(x_hat_pred, [x_hat])[0]
          grad_norm2 = tf.sqrt(1.e-6 + tf.reduce_sum(tf.square(grad), axis=[1,2,3]))
          gradient_penalty = tf.reduce_mean( (grad_norm2 - 1.)**2 )

      if use_gradient_penalty or standard_wass:
          discr_loss = tf.reduce_mean(fake_pred - real_pred) 
          gen_loss = -tf.reduce_mean(fake_pred) 
      else:
          #noise = tf.random_uniform(real_pred.get_shape().as_list(), maxval=0.05)
          discr_loss = tf.reduce_mean( (real_pred - 1)**2 + (fake_pred)**2 )
          gen_loss = tf.reduce_mean( (fake_pred - 1)**2 )
      
      if standard_wass:
          for v in tf.trainable_variables("main/discr"):
            tf.add_to_collection("clip_weights", v.assign(tf.clip_by_value(v, -0.01, 0.01)))

      #mu  = tf.get_variable(
      #  auto_name("avg_loss"), 
      #  initializer=tf.constant(0.707, dtype=tf.float32), 
      #  trainable=False
      #  )

      #mu_op = mu.assign(0.999*mu + 0.001*tf.sqrt(discr_loss))
      #tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, mu_op)

      #mu_scaled = mu/0.707
      #discr_lr_scale = tf.cond(mu_scaled > 0.6,  lambda: 1., lambda: (mu_scaled/0.6)**2 )

      if use_gradient_penalty:
          discr_loss += 10*gradient_penalty
          #discr_loss /= 100
          #gen_loss /= 100
      
      if use_l2_loss:
          gen_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables("gen")])
          discr_l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables("main/discr")])

          discr_loss += 5.e-5*discr_l2_loss
          gen_loss += 5.e-5*gen_l2_loss


      #discr_loss = tf.reduce_mean( tf.nn.relu(1-real_pred) +  tf.nn.relu(1+fake_pred), axis=-1 ) + 10*gradient_penalty #+ 1.e-5*discr_l2_loss
      #gen_loss = -tf.reduce_mean( fake_pred, axis=-1 )# + 5.e-5*gen_l2_loss

      #discr_loss = tf.reduce_mean(fake_pred - real_pred) / 1 + 10*gradient_penalty + 1.e-5*discr_l2_loss
      #gen_loss = -tf.reduce_mean(fake_pred) / 1 + 1.e-5*gen_l2_loss
      
      #Create optimizer for stochastic gradient descent (SGD)
      discr_optimizer = tf.train.AdamOptimizer(
          learning_rate=0.00005,
          beta1=0.5
          )
      #discr_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.00005, decay=0.5)

      #l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])

      #total_loss = gen_loss + discr_loss + 10*gradient_penalty + 5.e-5*l2_loss

      ##Tracking
      #for v, t in zip(tf.trainable_variables("main"), tf.trainable_variables("tracking")):
      #    tf.add_to_collection( tf.GraphKeys.UPDATE_OPS, t.assign(decay*t+(1-decay)*v) )

  else:
      #Mean squared errors
      mse = 10*tf.reduce_mean( tf.square(output - target_outputs), axis=[1,2,3] )
  
      alrc_mse = mse#alrc(mse)
      alrc_mse = tf.reduce_mean(alrc_mse)

      mse = tf.reduce_mean(mse)

      ##L2 regularization
      l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
  
      gen_loss = alrc_mse + 5.e-5*l2_loss

  #Create optimizer for stochastic gradient descent (SGD)
  gen_optimizer = tf.train.AdamOptimizer(
          learning_rate=0.0001,
          beta1=0.5
          )
  #gen_optimizer = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.5)


  #(
  #    learning_rate=learning_rate,
  #    beta1=beta1,
  #    beta2=0.9
  #    )

  #Update ops for batch normalisation
  update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
  
  with tf.control_dependencies(update_ops):
    if adversarial:
        #train_op = gen_optimizer.minimize(total_loss)
        gen_train_op = gen_optimizer.minimize(gen_loss, var_list=tf.trainable_variables("gen"))
        discr_train_op = discr_optimizer.minimize(discr_loss, var_list=tf.trainable_variables("main/discr"))
        train_op = [gen_train_op, discr_train_op]
    else:
        train_op = gen_optimizer.minimize(gen_loss)
    
  output_loss = {
      "Loss": tf.reduce_mean( tf.abs(phase_components - target_outputs0) ),
      "pred_real": tf.reduce_mean(real_pred),
      "pred_fake": tf.reduce_mean(fake_pred)
      }

  return train_op, output_loss, output0


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
  decay_ph = tf.placeholder(
      tf.float32, 
      shape=(),
      name="decay")
  gen_scale_ph = tf.placeholder(
      tf.float32, 
      shape=(),
      name="gen_scale")
  is_training_ph = tf.placeholder(
      tf.bool, 
      name="is_training")
  mode_ph = tf.placeholder(
      tf.int32, 
      name="mode")

  data_dir = "//Desktop-sa1evjv/h/wavefunctions/"
  batch_size = 24

  def load_data_subset(subset):
      return load_data(
          dir=data_dir,
          subset=subset, 
          batch_size=batch_size
          )

  inputs, target_outputs = tf.case(
      {tf.equal(mode_ph, 0): lambda: load_data_subset("train"),
       tf.equal(mode_ph, 1): lambda: load_data_subset("val"),
       tf.equal(mode_ph, 2): lambda: load_data_subset("test")}
      )
  
  #Describe learning policy
  start_iter = 4_234#0
  train_iters = 500_000
  val_iters = 1_000
  
  learning_rate = 0.0002
  beta1 = 0.9
  
  #Configure operations
  train_op, loss, output = configure(
      inputs=inputs,
      batch_size=batch_size,
      target_outputs=target_outputs,
      is_training=is_training_ph,
      learning_rate=learning_rate_ph, 
      beta1=beta1_ph,
      is_depthwise_sep=False,
      decay=decay_ph,
      gen_scale=gen_scale_ph
  )
  
  clip_op = tf.get_collection("clip_weights")

  #Tensors to dump as visual output
  first_image = inputs[0]
  first_target_output = target_outputs[0]
  first_output = output[0]

  #Session configuration
  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True #Only use required GPU memory
  config.gpu_options.force_gpu_compatible = True

  model_dir = f"//flexo.ads.warwick.ac.uk/Shared41/Microscopy/Jeffrey-Ede/models/wavefunctions/{EXPER_NUM}/"

  saver = tf.train.Saver(max_to_keep=1)
  noteable_saver = tf.train.Saver(max_to_keep=1)

  log_filepath = model_dir + "log.txt"
  save_period = 1; save_period *= 3600
  with tf.Session(config=config) as sess, open(log_filepath, "a") as log_file:

    #Initialize network parameters
    feed_dict = {
      is_training_ph: np.bool(True),
      learning_rate_ph: np.float32(learning_rate),
      beta1_ph: np.float32(beta1),
      mode_ph: np.int32(0),
      decay_ph: np.float32(0.),
      gen_scale_ph: np.float32(0.)
    }
    
    if start_iter:
        saver.restore(
            sess, 
            tf.train.latest_checkpoint(model_dir+"model/")
            )
    else:
        sess.run(tf.global_variables_initializer(), feed_dict=feed_dict)

    #Finalize graph to prevent additional nodes from being added
    #sess.graph.finalize()

    #Training
    avg_pred_fake = 0.3
    beta_pred_fake = 0.97
    time0 = time.time()
    for iter in range(start_iter, train_iters):
      
      is_halfway = iter >= train_iters // 2

      decay = 0.997 if iter else 0.
      lr = learning_rate #* 0.5**( max( iter//(train_iters//4), 3) )
      is_training = True#iter < 1_000 #not is_halfway
      beta1 = 0.9 if iter < 200_000 else 0.5
      
      gen_scale = 1.#0 if iter < 50 else 1.

      #Feed values into training operations
      feed_dict = {
          is_training_ph: np.bool(is_training),
          learning_rate_ph: np.float32(lr),
          beta1_ph: np.float32(beta1),
          mode_ph: np.int32(0),
          decay_ph: np.float32(decay),
          gen_scale_ph: np.float32(gen_scale)
      }

      if iter in [0, 100, 500] or not iter % 25_000 or (0 <= iter < 10_000 and not iter % 1000) or iter == start_iter:
        _, step_loss, [step_image, step_target_output, step_output] = sess.run([
            train_op, 
            loss,
            [first_image, first_target_output, first_output]
            ],
            feed_dict=feed_dict
            )
          
        save_input_loc = model_dir+"input-"+str(iter)+".tif"
        save_truth_loc = model_dir+"truth-"+str(iter)+".tif"
        save_output_loc = model_dir+"output-"+str(iter)+".tif"
        target_angle = np.angle(step_target_output[...,0] + 1j*step_target_output[...,1])
        output_angle = np.angle(step_output[...,0] + 1j*step_output[...,1])
        Image.fromarray(step_image.reshape(cropsize, cropsize).astype(np.float32)).save( save_input_loc )
        Image.fromarray(np.cos(target_angle).astype(np.float32)).save( save_truth_loc )
        Image.fromarray(np.cos(output_angle).astype(np.float32)).save( save_output_loc )
      else:
        if avg_pred_fake > 0.3 or use_gradient_penalty or standard_wass:
            step_train_op = train_op
        else:
            step_train_op = [train_op[0]]

        _, step_loss = sess.run([step_train_op, loss], feed_dict=feed_dict)

      if standard_wass:
        sess.run(clip_op)
      
      avg_pred_fake = beta_pred_fake*avg_pred_fake + (1-beta_pred_fake)*step_loss["pred_fake"]

      output = f"Iter: {iter}"
      for k in step_loss:
          output += f", {k}: {step_loss[k]}"

      if report_every_n:
        if not iter % report_every_n:
          print(output)

      if "nan" in output:
        saver.restore(
            sess, 
            tf.train.latest_checkpoint(model_dir+"model/")
            )
        #quit()

      log_file.write(output)

      if iter in [train_iters//2-1, train_iters-1]:
          noteable_saver.save(sess, save_path=model_dir+"noteable_ckpt/model", global_step=iter)
          time0 = time.time()
          start_iter = iter
      elif time.time() >= time0 + save_period:
          saver.save(sess, save_path=model_dir+"model/model", global_step=iter)
          time0 = time.time()
      
    #Validation - super important!
    val_loss = 0.
    for iter in range(val_iters):
      
      feed_dict = {
          is_training_ph: np.bool(False),
          mode_ph: np.int32(1),
          decay_ph: np.float32(decay)
      }
      
      step_loss = sess.run(loss, feed_dict=feed_dict)
      val_loss += step_loss
      
    val_loss /= val_iters
    
  return val_loss

if __name__ == "__main__":
    #Reset so graph nodes to not accumulate in ipynb session memory.
    tf.reset_default_graph()

    #Run your experiment!
    val_loss = experiment(report_every_n=1)

    #Report performance on validation set
    print(f"Validation loss: {val_loss}")
    with open(model_dir+"val_loss.txt", "w") as f:
        f.write(f"Val Loss: {val_loss}")