from math import exp
from random import random
from sklearn.model_selection import train_test_split
from sklearn import metrics
import numpy as np
import librosa
import tensorflow as tf
import torch
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Activation
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.layers import Conv1D
from keras.layers import MaxPooling2D
from keras.losses import MeanSquaredError
from google.colab import drive
drive.mount('/content/gdrive')

files = librosa.util.find_files('/content/gdrive/My Drive/Final Project 437/Number Recordings')
file_array = np.asarray(files)
data = []
nums = []
total_files = 0
for file in files: 
  print(file, total_files)
  x, sr = librosa.load(file, res_type='kaiser_fast')
  # Need to get number off file name
  number = file[-6:-4]
  if number[0] == '_':
    number = number[1]
  data.append(x)
  total_files += 1
  nums.append(number)

X = np.array(data, dtype=object)
y = np.array(nums, dtype=object)
i = 0
X_min = 1000000
while i < len(X):
  if X_min > len(X[i]):
    X_min = len(X[i])
  i += 1

#RNN

def initialize_network(n_inputs, n_hidden, n_outputs):
  network = list()  # initialize weights to random number in [0..1]
  hidden_layer = [{'weights':[random() for i in range(n_inputs+1)]} for i in range(n_hidden)]
  network.append(hidden_layer)
  output_layer = [{'weights':[random() for i in range(n_hidden+1)]} for i in range(n_outputs)]
  network.append(output_layer)
  return network


def activate(weights, inputs):
  activation = weights[-1]   # bias
  for i in range(len(weights)-1):
    activation += weights[i] * inputs[i]
  return activation


def transfer(activation): 
  return 1.0 / (1.0 + exp(-activation))


def forward_propagate(network, X, y):
  inputs = X
  for layer in network:
    new_inputs = []
    for node in layer:
      activation = activate(node['weights'], X)
      node['output'] = transfer(activation)
      new_inputs.append(node['output']) 
    inputs = new_inputs
  return inputs   


def transfer_derivative(output): 
  return output * (1.0 - output)


def backward_propagate_error(network, expected):
  for i in reversed(range(len(network))): 
    layer = network[i]
    errors = list()
    if i != len(network)-1:  
      for j in range(len(layer)):
        error = 0.0
        for node in network[i+1]:
          error += (node['weights'][j] * node['delta'])
        errors.append(error)
    else:   
      for j in range(len(layer)):
        node = layer[j]
        errors.append(expected[j] - node['output'])
    for j in range(len(layer)):
      network[i][j]['delta'] = errors[j] * transfer_derivative(node['output'])
  return network


def update_weights(network, x, eta):
  for i in range(len(network)):
    inputs = x
    if i != 0:
      inputs = [node['output'] for node in network[i-1]]
    for n in range(len(network[i])):
      node = network[i][n]
      for j in range(min(len(network[i][n]['weights']), len(inputs))):
        network[i][n]['weights'][j] += eta * node['delta'] * inputs[j]
      network[i][n]['weights'][-1] += eta * node['delta']
  return network


def train_network(network, X, y, eta, num_epochs, num_outputs):
  expected = np.full((50), 0)
  for epoch in range(num_epochs):
    sum_error = 0
    for i in range(len(y)):
      outputs = forward_propagate(network, X[i], y[i])
      expected[int(y[i])] = 1
      sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
      network = backward_propagate_error(network, expected)
      network = update_weights(network, X[i], eta)
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, eta, sum_error))
  return network


def test_network(network, X, y, num_outputs):
  expected = np.full((50), 0)
  sum_error = 0
  for i in range(len(y)):
    outputs = forward_propagate(network, X[i], y[i])
    expected[int(y[i])] = 1
    sum_error += sum([(expected[i] - outputs[i])**2 for i in range(len(expected))])
  print('mse of test data is', sum_error / float(len(y)))

def main_bc():
  n_inputs = X_min - 1
  n_outputs = 50  # possible class values are 0 through 49
  # Create the network
  network = initialize_network(n_inputs, 2, n_outputs)
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, test_size=0.33)
  # train network for 10 epochs using learning rate of 0.1 
  network = train_network(network, X_train, y_train, 0.1, 10, n_outputs)
  test_network(network, X_test, y_test, n_outputs)

#DNN
def main_dnn():
  #Data is a list of np.ndarrays, need to convert it to a list of tensors
  X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.67, test_size=0.33)
  max_len = 50335
  
  #X_train

  #Padding for all data to be used
  DNN_X_train = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in X_train])
  
  #Splitting for shortest example so all use that amount of data
  #DNN_X_train = []
  #for element in X_train:
  #  temp = (np.split(element, [X_min, X_min + 1]))
  #  DNN_X_train.append(temp[0])
  #DNN_X_train = np.array(DNN_X_train)

  DNN_X_train = tf.convert_to_tensor(DNN_X_train, dtype=tf.float32) 
  print("Shape of X: ", DNN_X_train.shape)

  #X_test

  #Padding for all data to be used
  DNN_X_test = np.asarray([np.pad(a, (0, max_len - len(a)), 'constant', constant_values=0) for a in X_test])

  #Splitting for shortest example so all use that amount of data
  #DNN_X_test = []
  #for element in X_test:
  #  temp = (np.split(element, [X_min, X_min + 1]))
  #  DNN_X_test.append(temp[0])
  #DNN_X_test = np.array(DNN_X_test)

  DNN_X_test = tf.convert_to_tensor(DNN_X_test, dtype=tf.float32) 
  print("Shape of X_test: ", DNN_X_test.shape)

  #y_train 
  DNN_y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)
  print("Shape of y: ", DNN_y_train.shape)

  #y_test
  DNN_y_test = tf.convert_to_tensor(y_test, dtype=tf.float32)
  print("Shape of y: ", DNN_y_test.shape)


  loss_fn = tf.keras.losses.MeanSquaredError()
  model = Sequential()
  model.add(tf.keras.Input(shape=(50335))) # padding up to most examples
  #model.add(tf.keras.Input(shape=(3165))) #cutting down to minimum data 

  for i in range(300):
    model.add(Dense(256, activation="relu"))
    model.add(Dropout(0.5))
  
  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(256))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))

  model.add(Dense(50))
  model.add(Activation('softmax'))

  model.compile(loss=loss_fn, metrics=['accuracy'], optimizer='adam')
  model.fit(DNN_X_train, DNN_y_train, epochs=10)
  #model.fit(DNN_dataset, epochs=10)
  model.evaluate(DNN_X_test, DNN_y_test)
  #model.evaluate(DNN_dataset)


if __name__ == "__main__":
  main_bc()
  main_dnn()





