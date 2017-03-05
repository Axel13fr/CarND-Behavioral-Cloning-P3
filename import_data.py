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
    def open_image(folder,src_path):
        # relative path case
        if src_path.startswith('IMG') or src_path.startswith(' IMG'):
            file_name = src_path.split('/')[-1]
            path = folder + 'IMG/' +  file_name
            img = cv2.imread(path)
        # absolute path case
        else:
            path = src_path
            img = cv2.imread(path)

        #if img is None:
        #    print('Error opening file:' + path)
        #    raise TypeError

        return img

    @staticmethod
    def open_images(folder,csv_line):
        im_center = DataProvider.open_image(folder,csv_line[0],)
        im_left = DataProvider.open_image(folder,csv_line[1])
        im_right = DataProvider.open_image(folder,csv_line[2])
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
                try:
                    float(line[3])
                except ValueError:
                    print("Could not convert data to a float:" + str(line))
                    raise

        return self._samples

    def split_samples(self):
        from sklearn.model_selection import train_test_split
        train_samples, validation_samples = train_test_split(self._samples, test_size=0.2)
        return train_samples, validation_samples

    @staticmethod
    def generator(folder,samples,batch_size=32):

        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for batch_sample in batch_samples:

                    center_angle = float(batch_sample[3])

                    [im_center, im_left, im_right] = DataProvider.open_images(folder,batch_sample)
                    images.append(im_center)
                    images.append(im_left)
                    images.append(im_right)

                    angles.append(center_angle)
                    # create adjusted steering measurements for the side camera images
                    correction = 0.25  # this is a parameter to tune
                    steering_left = center_angle + correction
                    steering_right = center_angle - correction
                    angles.append(steering_left)
                    angles.append(steering_right)

                    # Data Augmentation
                    im, mes = DataProvider.augment_sample(im_center, center_angle)
                    images.append(im)
                    angles.append(mes)
                    im, mes = DataProvider.augment_sample(im_left, center_angle)
                    images.append(im)
                    angles.append(mes)
                    im, mes = DataProvider.augment_sample(im_right, center_angle)
                    images.append(im)
                    angles.append(mes)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)

