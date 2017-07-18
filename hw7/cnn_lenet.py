import numpy as np
import math
import scipy.io
import copy

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def load_mnist(fullset=True):
  """Load mnist dataset

  Args:
    fullset: whether use full MNIST or not

  Returns:
    x: shape = (784, samples)
    y: shape = (samples,)
  """
  mnist_dataset = scipy.io.loadmat('../mnist_all')

  # create train and test arrays and labels
  xtrain = np.vstack([mnist_dataset['train'+str(i)] for i in range(10)])
  xtest = np.vstack([mnist_dataset['test'+str(i)] for i in range(10)])
  ytrain = np.hstack([[i for _ in range(mnist_dataset['train'+str(i)].shape[0])]
                      for i in range(10)])
  ytest = np.hstack([[i for _ in range(mnist_dataset['test'+str(i)].shape[0])]
                     for i in range(10)])

  # normalize
  xtrain = xtrain.astype(np.double) / 255.0
  xtest = xtest.astype(np.double) / 255.0

  # random shuffle
  train_indices = range(xtrain.shape[0])
  np.random.shuffle(train_indices)
  xtrain.take(train_indices, axis=0, out=xtrain)
  ytrain.take(train_indices, axis=0, out=ytrain)

  test_indices = range(xtest.shape[0])
  np.random.shuffle(test_indices)
  xtest.take(test_indices, axis=0, out=xtest)
  ytest.take(test_indices, axis=0, out=ytest)

  # get validation set
  m_validate = 10000
  xvalidate = xtrain[0:m_validate, :]
  yvalidate = ytrain[0:m_validate]
  xtrain = xtrain[m_validate:, :]
  ytrain = ytrain[m_validate:]
  m_train = xtrain.shape[0]
  m_test = xtest.shape[0]

  # transpose feature matrices so columns are instances and rows are features
  xtrain = xtrain.T
  xtest = xtest.T
  xvalidate = xvalidate.T

  # create smaller set for testing purposes
  if not fullset:
    m_train_small = m_train/20
    m_test_small = m_test/20
    m_validate_small = m_validate/20
    xtrain = xtrain[:, 0:m_train_small]
    ytrain = ytrain[0:m_train_small]
    xtest = xtest[:, 0:m_test_small]
    ytest = ytest[0:m_test_small]
    xvalidate = xvalidate[:, 0:m_validate_small]
    yvalidate = yvalidate[0:m_validate_small]

  return [xtrain, ytrain, xvalidate, yvalidate, xtest, ytest]

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def init_convnet(layers):
  """Initialize parameters of each layer in LeNet

  Args:
    layers: a dictionary that defines LeNet

  Returns:
    params: a dictionary stores initialized parameters
  """

  params = {}

  h = layers[1]['height']
  w = layers[1]['width']
  c = layers[1]['channel']

  # Starts with second layer, first layer is data layer
  for i in range(2, len(layers)+1):
    params[i-1] = {}
    if layers[i]['type'] == 'CONV':
      scale = np.sqrt(3./(h*w*c))
      weight = np.random.rand(
        layers[i]['k'] * layers[i]['k'] * c / layers[i]['group'],
        layers[i]['num']
      )

      params[i-1]['w'] = 2*scale*weight - scale
      params[i-1]['b'] = np.zeros(layers[i]['num'])
      # update h, w and c, used in next layer
      h = (h + 2*layers[i]['pad'] - layers[i]['k']) / layers[i]['stride'] + 1
      w = (w + 2*layers[i]['pad'] - layers[i]['k']) / layers[i]['stride'] + 1
      c = layers[i]['num']
    elif layers[i]['type'] == 'POOLING':
      h = (h - layers[i]['k']) / layers[i]['stride'] + 1
      w = (w - layers[i]['k']) / layers[i]['stride'] + 1
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'IP' and layers[i]['init_type'] == 'gaussian':
      scale = np.sqrt(3./(h*w*c))
      params[i-1]['w'] = scale*np.random.randn(h*w*c, layers[i]['num'])
      params[i-1]['b'] = np.zeros(layers[i]['num'])
      h = 1
      w = 1
      c = layers[i]['num']
    elif layers[i]['type'] == 'IP' and layers[i]['init_type'] == 'uniform':
      scale = np.sqrt(3./(h*w*c))
      params[i-1]['w'] = 2*scale*np.random.rand(h*w*c, layers[i]['num']) - scale
      params[i-1]['b'] = np.zeros(layers[i]['num'])
      h = 1
      w = 1
      c = layers[i]['num']
    elif layers[i]['type'] == 'RELU':
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'ELU':
      params[i-1]['w'] = np.array([])
      params[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'LOSS':
      scale = np.sqrt(3./(h*w*c))
      num = layers[i]['num']
      # last layer is K-1
      params[i-1]['w'] = 2*scale*np.random.rand(h*w*c, num-1) - scale
      params[i-1]['b'] = np.zeros(num - 1)
      h = 1
      w = 1
      c = layers[i]['num']
  return params

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def conv_net(params, layers, data, labels):
  """

  Args:
    params: a dictionary that stores hyper parameters
    layers: a dictionary that defines LeNet
    data: input data with shape (784, batch size)
    labels: label with shape (batch size,)

  Returns:
    cp: train accuracy for the train data
    param_grad: gradients of all the layers whose parameters are stored in params

  """
  l = len(layers)
  batch_size = layers[1]['batch_size']
  assert layers[1]['type'] == 'DATA', 'first layer must be data layer'

  param_grad = {}
  cp = {}
  output = {}
  output[1] = {}
  output[1]['data'] = data
  output[1]['height'] = layers[1]['height']
  output[1]['width'] = layers[1]['width']
  output[1]['channel'] = layers[1]['channel']
  output[1]['batch_size'] = layers[1]['batch_size']
  output[1]['diff'] = 0

  for i in range(2, l):
    if layers[i]['type'] == 'CONV':
      output[i] = conv_layer_forward(output[i-1], layers[i], params[i-1])
    elif layers[i]['type'] == 'POOLING':
      output[i] = pooling_layer_forward(output[i-1], layers[i])
    elif layers[i]['type'] == 'IP':
      output[i] = inner_product_forward(output[i-1], layers[i], params[i-1])
    elif layers[i]['type'] == 'RELU':
      output[i] = relu_forward(output[i-1], layers[i])

  i = l
  assert layers[i]['type'] == 'LOSS', 'last layer must be loss layer'

  wb = np.vstack([params[i-1]['w'], params[i-1]['b']])
  [cost, grad, input_od, percent] = mlrloss(wb,
                                            output[i-1]['data'],
                                            labels,
                                            layers[i]['num'], 1)

  param_grad[i-1] = {}
  param_grad[i-1]['w'] = grad[0:-1, :]
  param_grad[i-1]['b'] = grad[-1, :]
  param_grad[i-1]['w'] = param_grad[i-1]['w'] / batch_size
  param_grad[i-1]['b'] = param_grad[i-1]['b'] / batch_size

  cp['cost'] = cost/batch_size
  cp['percent'] = percent

  # range: [l-1, 2]
  for i in range(l-1,1,-1):
    param_grad[i-1] = {}

    if layers[i]['type'] == 'CONV':
      output[i]['diff'] = input_od
      param_grad[i-1], input_od = conv_layer_backward(output[i],
                                                      output[i-1],
                                                      layers[i],
                                                      params[i-1])
    elif layers[i]['type'] == 'POOLING':
      output[i]['diff'] = input_od
      input_od = pooling_layer_backward(output[i],
                                        output[i-1],
                                        layers[i])
      param_grad[i-1]['w'] = np.array([])
      param_grad[i-1]['b'] = np.array([])
    elif layers[i]['type'] == 'IP':
      output[i]['diff'] = input_od
      param_grad[i-1], input_od = inner_product_backward(output[i],
                                                         output[i-1],
                                                         layers[i],
                                                         params[i-1])
    elif layers[i]['type'] == 'RELU':
      output[i]['diff'] = input_od
      input_od = relu_backward(output[i], output[i-1], layers[i])
      param_grad[i-1]['w'] = np.array([])
      param_grad[i-1]['b'] = np.array([])

    param_grad[i-1]['w'] = param_grad[i-1]['w'] / batch_size
    param_grad[i-1]['b'] = param_grad[i-1]['b'] / batch_size

  return cp, param_grad

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def im2col_conv(input_n, layer, h_out, w_out):
  """Convert columns to image

  Args:
    input_n: input data, shape=(h_in*w_in*c, )
    layer: one cnn layer, defined in testLeNet.py
    h_out: output height
    w_out: output width

  Returns:
    col: shape=(k*k*c, h_out*w_out)
  """
  h_in = input_n['height']
  w_in = input_n['width']
  c = input_n['channel']
  k = layer['k']
  stride = layer['stride']
  pad = layer['pad']

  im = np.reshape(input_n['data'], (h_in, w_in, c))
  # im = np.pad(im, ((pad, pad), (pad, pad), (0, 0)), mode='constant', constant_values=(0, 0))
  col = np.zeros((k*k*c, h_out*w_out))

  for h in range(h_out):
    for w in range(w_out):
      matrix_hw = im[h*stride: h*stride+k, w*stride: w*stride+k, :]
      col[:, h*w_out + w] = matrix_hw.flatten()

  return col

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def conv_layer_forward(input, layer, param):
  """Convolution forward

  Args:
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py
    param: parameters, a dictionary

  Returns:
    output: a dictionary contains output data and shape information
  """
  # get parameters
  h_in = input['height']
  w_in = input['width']
  c = input['channel']
  batch_size = input['batch_size']
  k = layer['k']
  pad = layer['pad']
  stride = layer['stride']
  group = layer['group']
  num = layer['num']
  
  # resolve output shape
  h_out = (h_in + 2*pad - k) / stride + 1
  w_out = (w_in + 2*pad - k) / stride + 1

  assert h_out == np.floor(h_out), 'h_out is not integer'
  assert w_out == np.floor(w_out), 'w_out is not integer'
  assert np.mod(c, group) == 0, 'c is not multiple of group'
  assert np.mod(num, group) == 0, 'c is not multiple of group'

  # set output shape
  output = {}
  output['height'] = h_out
  output['width'] = w_out
  output['channel'] = num
  output['batch_size'] = batch_size
  output['data'] = np.zeros((h_out * w_out * num, batch_size))

  input_n = {
    'height': h_in,
    'width': w_in,
    'channel': c,
  }
  for n in range(batch_size):
    input_n['data'] = input['data'][:, n]
    col = im2col_conv(input_n, layer, h_out, w_out)
    tmp_output = col.T.dot(param['w']) + param['b']
    output['data'][:, n] = tmp_output.flatten()

  return output

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def col2im_conv(col, input, layer, h_out, w_out):
  """Convert image to columns

  Args:
    col: shape = (k*k, c, h_out*w_out)
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py
    h_out: output height
    w_out: output width

  Returns:
    im: shape = (h_in, w_in, c)
  """
  h_in = input['height']
  w_in = input['width']
  c = input['channel']
  k = layer['k']
  stride = layer['stride']

  im = np.zeros((h_in, w_in, c))
  col = np.reshape(col, (k*k*c, h_out*w_out))
  for h in range(h_out):
    for w in range(w_out):
      im[h*stride: h*stride+k, w*stride: w*stride+k, :] = \
        im[h*stride: h*stride+k, w*stride: w*stride+k, :] + \
        np.reshape(col[:, h*w_out + w], (k, k, c))

  return im

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def conv_layer_backward(output, input, layer, param):
  """Convolution backward

  Args:
    output: a dictionary contains output data and shape information
    input: a dictionary contains output data and shape information
    layer: one cnn layer, defined in testLeNet.py
    param: parameters, a dictionary

  Returns:
    param_grad: a dictionary stores gradients of parameters
    input_od: gradients w.r.t input data
  """

  # get parameters
  h_in = input['height']
  w_in = input['width']
  c = input['channel']
  batch_size = input['batch_size']
  k = layer['k']
  num = layer['num']

  h_out = output['height']
  w_out = output['width']

  input_od = np.zeros(input['data'].shape)
  param_grad = {}
  param_grad['b'] = np.zeros(param['b'].shape)
  param_grad['w'] = np.zeros(param['w'].shape)

  input_n = {
    'height': h_in,
    'width': w_in,
    'channel': c,
  }
  for n in range(batch_size):
    input_n['data'] = input['data'][:, n]
    col = im2col_conv(input_n, layer, h_out, w_out)
    tmp_data_diff = np.reshape(output['diff'][:, n], (h_out*w_out, num))

    param_grad['b'] += np.sum(tmp_data_diff, axis=0)
    param_grad['w'] += col.dot(tmp_data_diff)
    col_diff = param['w'].dot(tmp_data_diff.T)

    im = col2im_conv(col_diff, input, layer, h_out, w_out)
    input_od[:, n] = im.flatten()

  return param_grad, input_od

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def pooling_layer_forward(input, layer):
  """Pooling forward

  Args:
    input: a dictionary contains output data and shape information
    layer: one cnn layer, defined in testLeNet.py

  Returns:
    output: a dictionary contains output data and shape information
  """
  h_in = input['height']
  w_in = input['width']
  c = input['channel']
  batch_size = input['batch_size']
  k = layer['k']
  pad = layer['pad']
  stride = layer['stride']

  h_out = (h_in + 2*pad - k) / stride + 1
  w_out = (w_in + 2*pad - k) / stride + 1

  assert h_out == np.floor(h_out), 'h_out is not integer'
  assert w_out == np.floor(w_out), 'w_out is not integer'

  output = {}
  output['height'] = h_out
  output['width'] = w_out
  output['channel'] = c
  output['batch_size'] = batch_size
  output['data'] = np.zeros((h_out * w_out * c, batch_size))

  # TODO: implement your pooling forward here

  # implementation begins
  reinput = np.reshape(input['data'], (h_in, w_in, c, batch_size))
  reoutput = np.reshape(output['data'], (h_out, w_out, c, batch_size))
  #print np.max(input['data'])
  #print "start!!"
  for e in range(batch_size):
    tmp_input = reinput[:,:,:,e:e + 1] 
    count_j = 0
    for j in range(0, h_in - k + 1, stride):
      count_l = 0
      for l in range(0, w_in - k + 1, stride):
        slice = tmp_input[j : j + k, l : l + k, :]
        slice = np.reshape(slice, (k * k, c, 1))
        slice = np.amax(slice, axis = 0)
        reoutput[count_j:count_j+1, count_l:count_l+1, :, e : e + 1] = slice
        count_l = count_l + 1
      count_j = count_j + 1
  output['data'] = np.reshape(reoutput, (h_out * w_out * c, batch_size)) 
  #print np.amax(output['data'])
  #print "forward_end"
  # implementation ends

  assert np.all(output['data'].shape == (h_out * w_out * c, batch_size)), 'output[\'data\'] has incorrect shape!'

  return output

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def pooling_layer_backward(output, input, layer):
  """Pooling backward

  Args:
    output: a dictionary contains output data and shape information
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py

  Returns:
    input_od: gradients w.r.t input data
  """
  c = input['channel']
  batch_size = input['batch_size']
  h_out = output['height']
  w_out = output['width']
  k = layer['k']

  input_od = np.zeros(input['data'].shape)

  # TODO: implement backward pass here
  #cpy = {}
  #cpy['height'] = input['height']
  #cpy['width'] = input['width']
  #cpy['channel'] = input['channel']
  #for i in range(input['data'].shape[1]):
    #cpy['data'] = input['data'][:,i]
    #tmp = im2col_conv(cpy, layer, h_out, w_out)
    #tmp = np.reshape(tmp, (k * k, c, h_out * w_out))
    #slice = np.argmax(tmp, axis = 0)
    #print slice

  # implementation begins
  #output['data'] shape (800, 64), output['diff'] shape (800, 64)
  h_in = input['height']
  w_in = input['width']
  stride = layer['stride']
  input_cal = np.reshape(input['data'], (h_in, w_in, c, batch_size))
  output_cal = np.reshape(output['diff'], (h_out, w_out, c, batch_size))
  input_chg_od = np.reshape(input_od, (h_in, w_in, c, batch_size))
  for e in range(batch_size):
    tmp_input = input_cal[:,:,:,e:e + 1] 
    for i in range(c):
      count_j = 0
      for j in range(0, h_in - k + 1, stride):
        count_l = 0
        for l in range(0, w_in - k + 1, stride):
          slice = tmp_input[j : j + k, l : l + k, i : i + 1] 
          #indices = np.where(slice == slice.max())
          indices0, indices1, indices2, indices3 = np.unravel_index(slice.argmax(),slice.shape)
          h_tmp =  j + indices0
          w_tmp =  l + indices1
          input_chg_od[h_tmp][w_tmp][i][e] += output_cal[count_j][count_l][i][e]
          count_l = count_l + 1
        count_j = count_j + 1
  # implementation ends
  input_od = np.reshape(input_chg_od, (h_in * w_in * c, batch_size))
  assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'

  return input_od

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def relu_forward(input, layer):
  """RELU foward

  Args:
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py

  Returns:
    output: a dictionary contains output data and shape information
  """
  output = {}
  output['height'] = input['height']
  output['width'] = input['width']
  output['channel'] = input['channel']
  output['batch_size'] = input['batch_size']
  output['data'] = np.zeros(input['data'].shape)

  # TODO: implement your relu forward pass here
  # implementation begins
  output['data'] = np.maximum(output['data'], input['data'])
  # implementation ends

  assert np.all(output['data'].shape == input['data'].shape), 'output[\'data\'] has incorrect shape!'

  return output

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def relu_backward(output, input, layer):
  """RELU backward

  Args:
    output: a dictionary contains output data and shape information
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py

  Returns:
    input_od: gradients w.r.t input data
  """
  input_od = np.zeros(input['data'].shape)

  # TODO: implement your relu backward pass here
  # implementation begins
  # output['diff'] shape is (500, 64)
  diff_relu = copy.deepcopy(input['data'])
  diff_relu[np.where(diff_relu > 0)] = 1
  diff_relu = np.maximum(diff_relu, input_od)
  input_od = np.multiply(output['diff'], diff_relu)
 
  # implementation ends

  assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'
  #tmp = finite_difference(output, input, 1) 
  return input_od

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def inner_product_forward(input, layer, param):
  """Fully connected layer forward

  Args:
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py
    param: parameters, a dictionary

  Returns:
    output: a dictionary contains output data and shape information
  """
  num = layer['num']
  batch_size = input['batch_size']

  output = {}
  output['height'] = 1
  output['width'] = 1
  output['channel'] = num
  output['batch_size'] = batch_size
  output['data'] = np.zeros((num, batch_size))

  # TODO: implement your inner product forward pass here
  # implementation begins
  w = np.transpose(param['w'])
  # b is (1, 500), now we need to add b for the output for each column, we need to duplicate from (1, 500) to (64, 500)
  b = np.array([param['b'],]*(output['data'].shape[1]))
  # transpose to (500, 64)
  b = np.transpose(b)
  output['data'] = np.dot(w, input['data']) + b
  # implementation ends

  assert np.all(output['data'].shape == (num, batch_size)), 'output[\'data\'] has incorrect shape!'

  return output

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def inner_product_backward(output, input, layer, param):
  """Fully connected layer backward

  Args:
    output: a dictionary contains output data and shape information
    input: a dictionary contains input data and shape information
    layer: one cnn layer, defined in testLeNet.py
    param: parameters, a dictionary

  Returns:
    para_grad: a dictionary stores gradients of parameters
    input_od: gradients w.r.t input data
  """
  param_grad = {}
  param_grad['b'] = np.zeros(param['b'].shape)
  param_grad['w'] = np.zeros(param['w'].shape)
  input_od = np.zeros(input['data'].shape)

  # TODO: implement your inner product backward pass here
  # implementation begins
  # w shape is (800, 500), input['data'] shape is (800, 64), and output['data'] shape, output['diff'] is (500, 64) where batch size is 64
  # the differential of b is 1, so the gradient of b equals to the output['diff'], divide batch_size or not is related to learning rate
  batch_size = input['batch_size']
  #param_grad['b'] = np.sum(output['diff'], axis = 1)
  input_od = np.dot(param['w'], output['diff'])


  # the differential of w is x, and it also need to multiply the output['diff'] for chain rule,(param_grad['w'] is (800, 500))
  for i in xrange(batch_size):
    param_grad['b'] += output['diff'][:,i]
    sub_diff = np.transpose(output['diff'][:,i])
    sub_x = input['data'][:,i]
    sub_diff = np.reshape(sub_diff, (output['diff'].shape[0], 1))
    sub_diff = np.transpose(sub_diff)
    sub_x = np.reshape(sub_x, (input['data'].shape[0], 1))
    every_batch = np.dot(sub_x, sub_diff) 
    param_grad['w'] += every_batch

  # the differential of x is w, and it also need to multiply the output['diff'] for chain rule
  # implementation ends

  assert np.all(input['data'].shape == input_od.shape), 'input_od has incorrect shape!'

  return param_grad, input_od

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def get_lr(step, epsilon, gamma, power):
  """Get the learning rate at step iter"""
  lr_t = epsilon / math.pow(1 + gamma*step, power)
  return lr_t

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def finite_difference(output, input, h):
  """
  Args:
    input: input layer of our CNN, dictionary which contains 'data'
    output: output layer of our CNN, dictionary which contains 'data' and 'diff'
    h (scalar): finite difference

  Output:
    input_od_approx (numpy array, size of input['data']): approximate gradient using finite difference w.r.t input data
  """
  x_plus_h = input
  x_plus_h['data'] = x_plus_h['data'] + h
  layer = 0
  fx_plus_h = relu_forward(x_plus_h, layer)
  input_od_approx = np.multiply(((fx_plus_h['data'] - output['data'])/h), output['diff'])
  return input_od_approx

########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def mlrloss(wb, X, y, K, prediction):
  """Loss layer

  Args:
    wb: concatenation of w and b, shape = (num+1, K-1)
    X: input data, shape = (num, batch size)
    y: ground truth label, shape = (batch size, )
    K: K distinct classes (0 to K-1)
    prediction: whether calculate accuracy or not

  Returns:
    nll: negative log likelihood, a scalar, no need to divide batch size
    g: gradients of parameters
    od: gradients w.r.t input data
    percent: accuracy for this batch
  """
  (_, batch_size) = X.shape
  theta = wb[:-1, :]
  bias = wb[-1, :]

  # Convert ground truth label to one-hot vectors
  I = np.zeros((K, batch_size))
  I[y, np.arange(batch_size)] = 1

  # Compute the values after the linear transform
  activation = np.transpose(X.T.dot(theta) + bias)
  activation = np.vstack([activation, np.zeros(batch_size)])

  # This rescales so that all values are negative, hence, no overflow
  # problems with the exp operation (a single +inf can blow things up)
  activation -= np.max(activation, axis=0)
  activation = np.exp(activation)

  # Convert to probabilities by normalizing
  prob = activation / np.sum(activation, axis=0)

  nll = 0
  od = np.zeros(prob.shape)
  nll = -np.sum(np.log(prob[y, np.arange(batch_size)]))

  if prediction == 1:
    indices = np.argmax(prob, axis=0)
    percent = len(np.where(y == indices)[0]) / float(len(y))
  else:
    percent = 0

  # compute gradients
  od = prob - I
  gw = od.dot(X.T)
  gw = gw[0:-1, :].T
  gb = np.sum(od, axis=1)
  gb = gb[0:-1]
  g = np.vstack([gw, gb])

  od = theta.dot(od[0:-1, :])

  return nll, g, od, percent

#############################################################################################
#########  YOU SHOULD MAINTAIN THE RETURN TYPE AND SHAPE AS PROVIDED IN STARTER CODE   ######
#############################################################################################

def sgd_momentum(w_rate, b_rate, mu, decay, params, param_winc, param_grad):
  """Update the parameters with sgd with momentum

  Args:
    #from get_lr
    w_rate (scalar): sgd rate for updating w
    #from get_lr
    b_rate (scalar): sgd rate for updating b
    mu (scalar): momentum
    decay (scalar): weight decay of w
    params (dictionary): original weight parameters
    param_winc (dictionary): buffer to store history gradient accumulation
    param_grad (dictionary): gradient of parameter

  Returns:
    params_ (dictionary): updated parameters
    param_winc_ (dictionary): gradient buffer of previous step
  """

  params_ = copy.deepcopy(params)
  param_winc_ = copy.deepcopy(param_winc)

  # TODO: your implementation goes below this comment
  # implementation begins
  # update w, and b in each param stracture
  for i in range(1, len(params) + 1):
    #update theta for w and w, Wreg is the gradiend descent of w after regularization
    Wreg = param_grad[i]['w'] + decay * params_[i]['w'] 
    param_winc_[i]['w'] = (mu * param_winc_[i]['w']) + (w_rate * Wreg)
    params_[i]['w'] = params_[i]['w'] - param_winc_[i]['w']

    #update theta for b and b, Breg is the gradiend descent of b after regularization
    param_winc_[i]['b'] = (mu * param_winc_[i]['b']) + (b_rate * param_grad[i]['b'])
    params_[i]['b'] = params_[i]['b'] - param_winc_[i]['b']
  
  # implementation ends

  assert len(params_) == len(param_grad), 'params_ does not have the right length'
  assert len(param_winc_) == len(param_grad), 'param_winc_ does not have the right length'
  return params_, param_winc_
