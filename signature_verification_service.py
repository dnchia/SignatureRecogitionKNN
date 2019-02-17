import os
from tqdm import tqdm as progress
from PIL import Image
import tensorflow as tf
import numpy as np


class SignatureVerificationService:

    __slots__ = ('_signatures_directory',
                 '_y_labels',
                 '_signatures_per_class',
                 '_signatures',
                 '_signatures_features',)

    def __init__(self, signatures_directory, num_of_classes, signatures_per_class):
        self._signatures_directory = signatures_directory
        self._y_labels = np.array(range(1, num_of_classes + 1))
        self._signatures_per_class = signatures_per_class

        self._load_normalized_signatures(signatures_directory, self._y_labels.size, self._signatures_per_class)

    def _load_normalized_signatures(self, signatures_directory, num_of_classes, signatures_per_class):
        images_have_been_normalized = True
        model_has_been_created = True
        self._signatures = []
        self._signatures_features = []

        progress_value = progress(total=num_of_classes * signatures_per_class, desc='Loading signatures', unit=' images')

        if not os.path.exists(signatures_directory + '_normalized'):
            os.mkdir(signatures_directory + '_normalized')
            images_have_been_normalized = False

        if not os.path.exists(signatures_directory + '_normalized_model'):
            os.mkdir(signatures_directory + '_normalized_model')
            model_has_been_created = False

        for class_folder_index in range(1, num_of_classes + 1):
            class_folder_number = '0'

            if class_folder_index < 10:
                class_folder_number += str(class_folder_index)
            else:
                class_folder_number = str(class_folder_index)

            self._signatures_features.append([])
            self._signatures.append([])

            if not images_have_been_normalized:
                os.mkdir(signatures_directory + '_normalized' + '/p' + class_folder_number)

            if not model_has_been_created:
                os.mkdir(signatures_directory + '_normalized_model' + '/p' + class_folder_number)

            for signature_file_index in range(1, signatures_per_class + 1):
                signature_file = '0'

                if signature_file_index < 10:
                    signature_file += str(signature_file_index)
                else:
                    signature_file = str(signature_file_index)

                if not model_has_been_created:
                    if not images_have_been_normalized:
                        loaded_image = Image.open(signatures_directory + '/p' + class_folder_number + '/p' + class_folder_number + 's' + signature_file + '.png')
                        loaded_image = loaded_image.resize((250, 250), Image.ANTIALIAS)
                        loaded_image = loaded_image.convert('L')
                        loaded_image.save(signatures_directory + '_normalized' + '/p' + class_folder_number + '/p' + class_folder_number + 's' + signature_file + '.png')
                        signature_image = loaded_image
                    else:
                        signature_image = Image.open(
                            signatures_directory + '_normalized' + '/p' + class_folder_number + '/p' + class_folder_number + 's' + signature_file + '.png')

                    signature_model = self._extract_signature_features_vector(np.asarray(signature_image))
                    self._signatures_features[class_folder_index - 1].append(signature_model)
                    self._signatures[class_folder_index - 1].append(signature_image)

                    signature_model_file = open(signatures_directory + '_normalized_model' + '/p' + class_folder_number + '/p' + class_folder_number + 's' + signature_file + '.csv', "+a")
                    for element in signature_model:
                        signature_model_file.write("%f\n" % element)

                else:
                    signature_model_lines = open(signatures_directory + '_normalized_model' + '/p' + class_folder_number + '/p' + class_folder_number + 's' + signature_file + '.csv', "r")
                    signature_model_lines = signature_model_lines.readlines()
                    signature_model = []
                    for line in signature_model_lines:
                        signature_model.append(float(line))
                    self._signatures_features[class_folder_index - 1].append(signature_model)

                progress_value.update()

    @staticmethod
    def _extract_signature_features_vector(signature):
        signature_features = []

        # Occupation percentage (back) of the signature
        # Vertical and horizontal projections (% of black per file and column)
        vertical_projection = []
        horizontal_projection = np.zeros(len(signature))

        dark_pixels = 0
        num_file = 0

        for file in signature:
            vertical_projection_var = 0

            for column in file:
                if column < 220:
                    dark_pixels += 1
                    vertical_projection_var += 1
                    horizontal_projection[num_file] += 1

            horizontal_projection[num_file] = horizontal_projection[num_file] / len(signature)
            vertical_projection.append(vertical_projection_var / len(file))
            num_file += 1

        percentage_of_dark = dark_pixels / (len(signature) * len(signature[0]))
        signature_features.append(percentage_of_dark)

        # Grid values (black maximums on 10x10 grids)
        grid_values = [min(i) for i in signature]
        grid_values = [min(grid_values[i-10:i]) / 255 for i in range(10, len(signature) + 1, 10)]

        for element in vertical_projection:
            signature_features.append(element)

        for element in horizontal_projection:
            signature_features.append(element)

        for element in grid_values:
            signature_features.append(element)

        return signature_features

    def build_model_with_signatures_features(self, division_between_training_and_test, number_of_executions):
        x_train_signatures, x_test_signatures, y_train_labels, y_test_labels = self.distribute_signatures(
            self._signatures_features,
            division_between_training_and_test)

        k = 3

        # Placeholders
        x_data_train = tf.placeholder(shape=[None, len(x_train_signatures[0])], dtype=tf.float32)
        x_data_test = tf.placeholder(shape=[len(x_test_signatures[0])], dtype=tf.float32)
        y_target_train = tf.placeholder(shape=[None], dtype=tf.int64)

        # Euclidean Distance
        distance = tf.negative(tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(x_data_train, x_data_test)), reduction_indices=1)))

        # Prediction: Get min distance neighbors
        values, indices = tf.nn.top_k(distance, k=k, sorted=True)

        nearest_neighbors = []
        for i in range(k):
            nearest_neighbors.append(y_target_train[indices[i]])

        neighbors_tensor = tf.stack(nearest_neighbors)
        y, idx, count = tf.unique_with_counts(neighbors_tensor)

        prediction = tf.slice(y, begin=[tf.argmax(count, 0)], size=tf.constant([1], dtype=tf.int64))[0]

        # Calculate how many loops over training data
        num_loops = int(np.ceil(len(x_test_signatures)))

        with tf.Session() as sess:
            average_accuracy = []

            for k in range(number_of_executions):
                accuracy = 0.
                predicted_labels = []

                for i in range(num_loops):
                    nn_index = sess.run(prediction, feed_dict={x_data_train: x_train_signatures,
                                                               x_data_test: x_test_signatures[i],
                                                               y_target_train: y_train_labels})

                    predicted_labels.append(nn_index)

                    if nn_index == y_test_labels[i]:
                        accuracy += 1. / len(x_test_signatures)

                print('Accuracy on test set: ' + str(accuracy))
                average_accuracy.append(accuracy)

                print('Predicted labels in iteration %d: ' % (k + 1))
                print(predicted_labels)
                print('Real labels in iteration %d: ' % (k + 1))
                print(y_test_labels)
                print()

                x_train_signatures, x_test_signatures, y_train_labels, y_test_labels = self.distribute_signatures(
                    self._signatures_features,
                    division_between_training_and_test)

            average_accuracy = sum(average_accuracy) / number_of_executions
            print('Average accuracy on test set: ' + str(average_accuracy))
        return

    @staticmethod
    def distribute_signatures(signatures_variable, percentage_between_training_and_test):
        x_train_signatures = []
        x_test_signatures = []
        y_train_labels = []
        y_test_labels = []

        for index in range(len(signatures_variable)):
            train_indices = np.random.choice(len(signatures_variable[index]),
                                             round(len(signatures_variable[index]) * percentage_between_training_and_test),
                                             replace=False)
            test_indices = np.array(list(set(range(len(signatures_variable[index]))) - set(train_indices)))

            for signature in train_indices:
                x_train_signatures.append(signatures_variable[index][signature])
                y_train_labels.append(index + 1)

            for signature in test_indices:
                x_test_signatures.append(signatures_variable[index][signature])
                y_test_labels.append(index + 1)

        return x_train_signatures, x_test_signatures, y_train_labels, y_test_labels
