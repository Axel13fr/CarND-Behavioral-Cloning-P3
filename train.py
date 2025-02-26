from import_data import DataProvider
import numpy as np
import models

from keras.callbacks import Callback
from keras.models import load_model
import matplotlib.pyplot as plt

''' Stores losses via Keras callbacks and provide plot functions
'''
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


###########################################################################################
############## TRAINING ###############

# Load Data:
provider = DataProvider()
provider.load_file('./data/Center_Forward/') # First track
provider.load_file('./data/Center_Backward/') # First track
provider.load_file('./data/2ndTrack/')
provider.load_file('./data/2ndTrack_LastTurns/')

# Project data has a relative path for images
ref_data_folder = './data/RefData/'
provider.load_file(ref_data_folder)

# Plot Angle Distribution
plt.hist(provider.get_angles(), bins=100)  # plt.hist passes it's arguments to np.histogram
plt.title("Angles Histogram ")
#plt.show()
plt.draw()
plt.pause(0.05)

# Exclude over represented data
bin_edges = provider.redistribute(cap_threshold=1000)
adjusted_angles = provider.get_angles()
plt.hist(adjusted_angles, bins=bin_edges)
plt.draw()
plt.pause(0.05)


# Data augmentation during generator: flip version of each frames
FRAMES_PER_SAMPLE = 2
[train_samples,validation_samples] = provider.split_samples()
train_generator = DataProvider.generator(ref_data_folder,train_samples, batch_size=256)
validation_generator = DataProvider.generator(ref_data_folder,validation_samples, batch_size=256)
print('Training samples: ' + str(FRAMES_PER_SAMPLE*len(train_samples)))

# Build
model = None
model = load_model('model.h5')
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