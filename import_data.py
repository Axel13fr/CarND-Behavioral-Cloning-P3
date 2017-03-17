import csv
import cv2
import numpy as np
import sklearn

# A Sample
class Sample():
    def __init__(self,angle,img_path):
        self._angle = angle
        self._img_path = img_path

# Sample Container, File parser, Data generator
class DataProvider():

    def __init__(self):
        self._samples = []

    @staticmethod
    def open_image(folder,src_path):
        # relative path case
        if src_path.startswith('IMG') or src_path.startswith(' IMG'):
            file_name = src_path.split('/')[-1]
            path = folder + 'IMG/' +  file_name
            img =  cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
        # absolute path case
        else:
            path = src_path
            img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

        return img

    @staticmethod
    def flip_sample(img,mes):
        image_flipped = np.fliplr(img)
        measurement_flipped = -mes
        return image_flipped,measurement_flipped

    def load_file(self,folder):
        with open(folder + "driving_log.csv") as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                try:
                    float(line[3])
                except ValueError:
                    print("Could not convert data to a float:" + str(line))
                    raise

                # create adjusted steering measurements for the side camera images
                correction = 0.25  # this is a parameter to tune
                self._samples.append(Sample(float(line[3]),line[0]))
                self._samples.append(Sample(float(line[3])+ correction,line[1])) # left
                self._samples.append(Sample(float(line[3]) - correction, line[2]))  # right

        return self._samples

    def split_samples(self):
        from sklearn.model_selection import train_test_split
        train_samples, validation_samples = train_test_split(self._samples, test_size=0.2)
        return train_samples, validation_samples

    ''' Returns bin ranges which shall be filtered out because their bin count is over cap_threshold
        @return a list containing [bin_Start,bin_End, selection_ratio]
        @return bin_edges for later histogram reuse
    '''
    def get_angle_ranges_to_cap(self,cap_threshold,bins):
        MAX_SAMPLES = cap_threshold
        hist, bin_edges = np.histogram(self.get_angles(),bins)
        overpop_bin_indices = np.greater(hist, MAX_SAMPLES).nonzero()
        overpopulated_bin_starts = bin_edges[overpop_bin_indices]
        # Number of samples per bin
        bin_counts = hist[overpop_bin_indices]
        print("Total Bin Counts : " + str(sum(bin_counts)))

        bin_width = bin_edges[1] - bin_edges[0]

        capped_ranges = []
        for i, start in enumerate(overpopulated_bin_starts):
            # select ratio will be used to randomly excluse some samples
            select_ratio = MAX_SAMPLES / bin_counts[i]
            capped_ranges.append([start, start + bin_width, select_ratio])

        return capped_ranges, bin_edges

    ''' Cuts off overrepresented angle samples randomly within each bin having a count above the cap_threshold parameter
        @return: bin edges used to calculate the histogram to cut off 
    '''
    def redistribute(self,cap_threshold=200,bins=100):
        capped_ranges, bin_edges = self.get_angle_ranges_to_cap(cap_threshold,bins)
        print("Samples before redistribution: " + str(len(self._samples)))
        adjusted_samples = []
        for sample in self._samples:
            if not self.is_angle_excluded(sample._angle,capped_ranges):
                adjusted_samples.append(sample)
        self._samples = adjusted_samples
        print("Samples after redistribution: " + str(len(adjusted_samples)))

        return bin_edges

    ''' Checks if an angle is within a range which should be capped and randomly decides based on a the select ratio
        if it should be excluded or not
    '''
    @staticmethod
    def is_angle_excluded(angle,capped_ranges):
        # Search if angle shall be filtered out
        excluded = False
        for range in capped_ranges:
            if angle >= range[0] and angle <= range[1]:
                # random selection based on ratio
                ratio = range[2]
                if ratio < np.random.uniform(size=1):
                    excluded = True
        return excluded

    def get_angles(self):
        angles = []
        for sample in self._samples:
            angles.append(sample._angle)
        return angles

    @staticmethod
    def preprocessing(img):
        # RGB TO HSV then to only H
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)[:,:,0].reshape(160,320,1)

    @staticmethod
    def generator(folder,samples,batch_size=32):

        num_samples = len(samples)
        while 1:  # Loop forever so the generator never terminates
            sklearn.utils.shuffle(samples)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset + batch_size]

                images = []
                angles = []
                for sample in batch_samples:
                    img = DataProvider.open_image(folder,sample._img_path)
                    # Apply preprocessing for training too !
                    img = DataProvider.preprocessing(img)
                    #print(img.shape)
                    angle = sample._angle

                    # Read image, associate angle
                    angles.append(angle)
                    images.append(img)

                    # Data Augmentation: flip image
                    flipped_img, flipped_angle  = DataProvider.flip_sample(img, angle)
                    images.append(flipped_img)
                    angles.append(flipped_angle)

                X_train = np.array(images)
                y_train = np.array(angles)
                yield sklearn.utils.shuffle(X_train, y_train)