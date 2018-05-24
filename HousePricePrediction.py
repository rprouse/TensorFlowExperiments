import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generate random houses between 1000 and 3500 square feet
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low=1000, high=3500, size=num_house)

# generate house prices from the house size with random noise
house_price = house_size * 100.0 + np.random.randint(low=20000, high=70000, size=num_house)

# plot the houses
# plt.plot(house_size, house_price, ".")
# plt.gcf().canvas.set_window_title("House Prices")
# plt.ylabel("Price")
# plt.xlabel("Size")
# plt.show()

# normalize the data to prevent underflows/overflows
def normalize(array):
  return (array - array.mean()) / array.std()

# use 70% of the samples for training
num_train_samples = math.floor(num_house * 0.7)

# define the training data
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# define test data
test_house_size = np.array(house_size[num_train_samples:])
test_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_price_norm = normalize(test_price)

# setup the tensor placeholders
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# define the variables holding the size_factor and price_offset we set during training
# we initialize both to random values based on the normal distribution
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# define operations for predicting the house prices
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# define the loss function (how much error) - mean squared error
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2)) / (2*num_train_samples)

# define the Gradient Decent optimizer that will minimize the loss in the cost operation
learning_rate = 0.1
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# initialize the variables
init = tf.global_variables_initializer()

# launch the graph in a session
with tf.Session() as sess:
  sess.run(init)

  # set how often we display training progress and the number of training iterations
  display_every = 2
  num_training_iter = 50

  for iteration in range(num_training_iter):
    for (x,y) in zip(train_house_size_norm, train_price_norm):
      sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

    # display the current status
    if(iteration + 1) % display_every == 0:
      c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
      print("Iteration #", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(c), \
            "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

  print("Training finished...")
  training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
  print("Trained cost=", training_cost, \
        "size_factor=", sess.run(tf_size_factor), "price_offset=", sess.run(tf_price_offset))

  train_house_size_mean = train_house_size.mean()
  train_house_size_std = train_house_size.std()
  train_price_mean = train_price.mean()
  train_price_std = train_price.std()

  plt.rcParams["figure.figsize"] = (10,8)
  plt.figure()
  plt.ylabel("Price")
  plt.xlabel("Size")
  plt.plot(train_house_size, train_price, 'go', label="Training Data")
  plt.plot(test_house_size, test_price, 'mo', label="Test Data")
  plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
           (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset)) * train_price_std + train_price_mean,
           label="Learning Regression")
  plt.legend(loc="upper left")
  plt.show()
