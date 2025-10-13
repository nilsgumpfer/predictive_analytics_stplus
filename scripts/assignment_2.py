import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.datasets import mnist


(train_images, train_labels), (val_images, val_labels) = mnist.load_data()
idx = 2
print(train_images[idx].shape, train_labels[idx])

plt.matshow(train_images[idx], cmap='Greys')
plt.show()
