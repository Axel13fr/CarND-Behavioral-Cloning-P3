from import_data import DataProvider
import models

from keras.callbacks import Callback
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

# Main:
provider = DataProvider()

provider.load_file('./data/Center_Forward/')
provider.load_file('./data/Center_Backward/')
#provider.load_file('./data/First_backward/')
#provider.load_file('./data/First_forward/')
[train_samples,validation_samples] = provider.split_samples()
train_generator = DataProvider.generator(train_samples, batch_size=128)
validation_generator = DataProvider.generator(validation_samples, batch_size=128)

# Build
model = models.build_Nvidia()

# Train
history = LossHistory()
model.compile(loss='mse',optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch= 4*len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=4*len(validation_samples), nb_epoch=10,callbacks=[history])

model.save('model.h5')

history.plot()