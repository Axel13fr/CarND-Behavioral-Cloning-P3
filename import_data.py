import csv
import cv2
import numpy as np
import sklearn

class DataProvider():

    def __init__(self):
        self._images = None
        self._measurements = None
        self._samples = []

    @staticmethod
    def open_image(src_path):
        #file_name = src_path.split('/')[-1]
        #current_path = folder + 'IMG/' + file_name
        return cv2.imread(src_path)

    @staticmethod
    def open_images(csv_line):
        im_center = DataProvider.open_image(csv_line[0])
        im_left = DataProvider.open_image(csv_line[1])
        im_right = DataProvider.open_image(csv_line[2])
        return im_center,im_left,im_right

    @staticmethod
    def augment_sample(img,mes):
        image_flipped = np.fliplr(img)
        measurement_flipped = -mes
        return image_flipped,measurement_flipped

    def load_file(self,folder):
        with open(folder + "driving_log.csv") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                self._samples.append(line)

        return self._samples

    def split_samples(self):
        from sklearn.model_selection import train_test_split
        train_samples, validation_samples = train_test_split(self._samples, test_size=0.2)
        return train_samples, validation_samples

    def load_training(self,folder):
        lines = []
        with open(folder + "driving_log.csv") as csvfile:

            reader = csv.reader(csvfile)
            for line in reader:
                lines.append(line)
        images = []
        measurements = []
        for line in lines:
            [im_center, im_left, im_right] = self.open_images(line)
            images.append(im_center)
            images.append(im_left)
            images.append(im_right)

            measurement = float(line[3])
            measurements.append(measurement)
            # create adjusted steering measurements for the side camera images
            correction = 0.5 # this is a parameter to tune
            steering_left = measurement + correction
            steering_right = measurement - correction

            # Data Augmentation
            im,mes = self.augment_sample(im_center,measurement)
            images.append(im)
            measurements.append(mes)

        if self._images is None:
            self._images = np.array(images)
            self._measurements = np.array(measurements)
        else:
            self._images = np.concatenate([self._images, np.array(images)])
            self._measurements = np.concatenate([self._measurements, np.array(measurements)])

        return self._images,self._measurements
    @staticmethod
    def generator(samples,batch_size=32):

        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:

                    center_angle = float(batch_sample[3])

                    [im_center, im_left, im_right] = DataProvider.open_images(batch_sample)
                    images.append(im_center)
                    images.append(im_left)
                    images.append(im_right)

                    angles.append(center_angle)
                    # create adjusted steering measurements for the side camera images
                    correction = 0.4  # this is a parameter to tune
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction
                    angles.append(steering_left)
                    angles.append(steering_right)

                    # Data Augmentation
                    im, mes = DataProvider.augment_sample(im_center, center_angle)
                    images.append(im)
                    angles.append(mes)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

