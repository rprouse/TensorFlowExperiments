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
plt.plot(house_size, house_price, ".")
plt.gcf().canvas.set_window_title("House Prices")
plt.ylabel("Price")
plt.xlabel("Size")
plt.show()