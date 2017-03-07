from import_data import DataProvider
import numpy as np
import models

from keras.callbacks import Callback
from keras.models import load_model
import matplotlib.pyplot as plt

class LossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []
        self.val_loss = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
    def plot(self):
        fig, ax = plt.subplots()
        ax.plot(self.losses, 'C1', label='Train Loss.')
        ax.plot(self.val_loss, 'C2', label='Val Loss.')
        ax.set_title('Loss curves')
        ax.legend()
        plt.show()

# Load Data:
provider = DataProvider()
provider.load_file('./data/Center_Forward/')
provider.load_file('./data/Center_Backward/')

# Plot Angle Distribution
plt.hist(provider.get_angles(), bins='fd')  # plt.hist passes it's arguments to np.histogram
plt.title("Angles Histogram ")
plt.show()

# Exclude over represented data
bin_edges = provider.redistribute()
adjusted_angles = provider.get_angles()
plt.hist(adjusted_angles, bins=bin_edges)
plt.show()

# Project data has a relative path:
ref_data_folder = './data/RefData/'
#provider.load_file(ref_data_folder)

# Data augmentation: flip version of each of the 3 camera frames !
FRAMES_PER_SAMPLE = 6
[train_samples,validation_samples] = provider.split_samples()
train_generator = DataProvider.generator(ref_data_folder,train_samples, batch_size=128)
validation_generator = DataProvider.generator(ref_data_folder,validation_samples, batch_size=128)
print('Training samples: ' + str(FRAMES_PER_SAMPLE*len(train_samples)))

# Build
model = None
#model = load_model('model.h5')
if not model:
    model = models.build_Nvidia()
    model.compile(loss='mse',optimizer='adam')

# Train
history = LossHistory()
model.fit_generator(train_generator, samples_per_epoch= FRAMES_PER_SAMPLE*len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=FRAMES_PER_SAMPLE*len(validation_samples), nb_epoch=10,callbacks=[history])

model.save('model.h5')

history.plot()