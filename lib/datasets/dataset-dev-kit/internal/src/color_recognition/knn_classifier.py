#!/usr/bin/python
import csv
import random
import math
import operator
import cv2


# calculation of euclidean distance
def calculate_euclidean_distance(variable1, variable2, length):
    distance = 0
    for x in range(length):
        distance += pow(variable1[x] - variable2[x], 2)
    return math.sqrt(distance)


# get k nearest neighbors
def get_k_nearest_neighbors(training_feature_vector, test_instance, k):
    distances = []
    length = len(test_instance)
    for x in range(len(training_feature_vector)):
        dist = calculate_euclidean_distance(test_instance, training_feature_vector[x], length)
        distances.append((training_feature_vector[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


# votes of neighbors
def get_votes_of_neighbors(neighbors):
    all_possible_neighbors = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in all_possible_neighbors:
            all_possible_neighbors[response] += 1
        else:
            all_possible_neighbors[response] = 1
    sorted_votes = sorted(all_possible_neighbors.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_votes[0][0]


# Load image feature data to training feature vectors and test feature vector
def load_dataset(filename, filename2, training_feature_vector=[], test_feature_vector=[]):
    with open(filename) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            training_feature_vector.append(dataset[x])

    with open(filename2) as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)):
            for y in range(3):
                dataset[x][y] = float(dataset[x][y])
            test_feature_vector.append(dataset[x])


def classify_color(training_data, test_data):
    training_feature_vector = []  # training feature vector
    test_feature_vector = []  # test feature vector
    load_dataset(training_data, test_data, training_feature_vector, test_feature_vector)
    classifier_prediction = []  # predictions
    k = 5  # K value of k nearest neighbor
    for x in range(len(test_feature_vector)):
        neighbors = get_k_nearest_neighbors(training_feature_vector, test_feature_vector[x], k)
        result = get_votes_of_neighbors(neighbors)
        classifier_prediction.append(result)
    return classifier_prediction[0]
