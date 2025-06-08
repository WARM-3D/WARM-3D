import json
import numpy as np
from internal.src.color_recognition import knn_classifier
from internal.src.data_generation import color_histogram_feature_extraction
import os
import cv2

IMAGE_WIDTH = 1920
IMAGE_HEIGHT = 1200


def crop_center(img, cropx, cropy):  # to crop and get the center of the given image
    y, x, channels = img.shape
    startx = x // 2 - (cropx // 2)
    # starty = y // 2 - (cropy // 2)
    starty = y // 2
    return img[starty: starty + cropy, startx: startx + cropx]


def apply_k_mean_clustering(image):
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 3
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]

    # TODO:
    # collect all pixels of border in an array
    # cluster pixels (k=3)
    # find largest cluster
    # figure out color of largest cluster
    # if final vehicle color is equal to the color of largest cluster of border, then take second largest cluster color

    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # show the image
    # cv2.imshow('segmented frame', segmented_image)
    # cv2.waitKey(0)
    return segmented_image


def color_recognition(crop_img):
    # crop the detected vehicle image and get a image piece from center of it both for debugging and sending that image piece to color recognition module
    crop_height, crop_width, channels = crop_img.shape
    crop_img = crop_center(crop_img, crop_width // 2, crop_height // 2)
    # apply k means clustering
    crop_img = apply_k_mean_clustering(crop_img)
    color_rgb_str = color_histogram_feature_extraction.color_histogram_of_test_image(
        crop_img
    )  # send image piece to recognize vehicle color
    color_predicted = knn_classifier.classify_color(os.getcwd() + "/training.data", os.getcwd() + "/test.data")
    return color_predicted, color_rgb_str


input_files_labels = (
    "04_labels/"
)
input_files_images = (
    "03_images/"
)
idx_frame = 0
for label_file_name, image_file_name in zip(
        sorted(os.listdir(input_files_labels)), sorted(os.listdir(input_files_images))
):
    print("processing frame idx: ", str(idx_frame))
    json_file = open(
        os.path.join(input_files_labels, label_file_name),
    )
    img = cv2.imread(os.path.join(input_files_images, image_file_name), cv2.IMREAD_UNCHANGED)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img_vehicle = cv2.imread("/home/user/Pictures/red.png", cv2.IMREAD_UNCHANGED)
    # remove alpha channel when loading png image
    # img_vehicle = img_vehicle[:, :, :3]

    json_data = json.load(json_file)
    idx_obj = 0
    for label in json_data["labels"]:
        # print("obj index: ", str(idx_obj))

        x_min, y_min, x_max, y_max = label["box2d"]
        # print(x_min, y_min, x_max, y_max)
        img_vehicle = img[max(0, y_min): min(IMAGE_HEIGHT - 1, y_max), max(0, x_min): min(IMAGE_WIDTH - 1, x_max)]
        # cv2.imwrite("/home/user/Downloads/test.png", img_vehicle)
        color_predicted, color_rgb_str = color_recognition(img_vehicle)
        color_rgb_arr = color_rgb_str.split(",")
        color_hex = "#%02x%02x%02x" % (int(color_rgb_arr[0]), int(color_rgb_arr[1]), int(color_rgb_arr[2]))
        label["color_body"] = color_predicted
        label["color_body_hex"] = color_hex
        # print("color: " + color_predicted + ", rgb: " + color_rgb_str + ", hex: " + str(color_hex))
        idx_obj = idx_obj + 1

    with open(os.path.join(input_files_labels, label_file_name), "w", encoding="utf-8") as writer:
        json.dump(json_data, writer, ensure_ascii=True, indent=4)

    idx_frame = idx_frame + 1
