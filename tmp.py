import os.path

import cv2
import numpy as np
import random
import pyqtgraph as pg

def mosaic_data_augmentation(image_paths, labels, mosaic_ratio=0.5):
    num_images = len(image_paths)
    mosaic_images = []
    mosaic_labels = []

    for i in range(num_images):
        image = cv2.imread(file_path+image_paths[i])
        height, width, _ = image.shape

        if random.random() < mosaic_ratio:
            # Choose three additional images randomly
            indices = random.sample(range(num_images), 3)
            indices.append(i)

            # Calculate mosaic size
            mosaic_width = int(width / 2)
            mosaic_height = int(height / 2)

            # Initialize mosaic image and labels
            mosaic = np.zeros((mosaic_height * 2, mosaic_width * 2, 3), dtype=np.uint8)
            mosaic_label = []

            for j, index in enumerate(indices):
                # Load image and label
                image = cv2.imread(file_path+image_paths[index])
                #label = labels[index]

                # Resize image to mosaic size
                image = cv2.resize(image, (mosaic_width, mosaic_height))

                # Adjust label coordinates
                #label[:, [1, 3]] = label[:, [1, 3]] * (mosaic_width / width) + j % 2 * mosaic_width
                #label[:, [2, 4]] = label[:, [2, 4]] * (mosaic_height / height) + j // 2 * mosaic_height

                # Append the image and label to the mosaic
                mosaic[j // 2 * mosaic_height:(j // 2 + 1) * mosaic_height,
                       j % 2 * mosaic_width:(j % 2 + 1) * mosaic_width] = image
                #mosaic_label.extend(label.tolist())

            mosaic_images.append(mosaic)
            mosaic_labels.append(np.array(mosaic_label))

    return mosaic_images, mosaic_labels

#mosaic_data_augmentation('X:\毕业设计\源码\yolov5_7.0\yolov5\VOC2007_fer2013\images\\test','img')
#
file_path = 'X:\毕业设计\源码\yolov5_7.0\yolov5\VOC2007_fer2013\images\\val\\'
# items = os.listdir(file_path)
# img , lab = mosaic_data_augmentation(items,'happy')
# # cv2.imwrite('X:\毕业设计\源码\yolov5_7.0\yolov5\VOC2007_fer2013\images\\1.jpg',img[0])
# # cv2.imwrite('X:\毕业设计\源码\yolov5_7.0\yolov5\VOC2007_fer2013\images\\5.jpg',img[5])
# # cv2.imwrite('X:\毕业设计\源码\yolov5_7.0\yolov5\VOC2007_fer2013\images\\10.jpg',img[10])
# # cv2.imwrite('X:\毕业设计\源码\yolov5_7.0\yolov5\VOC2007_fer2013\images\\30.jpg',img[30])
# # cv2.imshow('img',img[5])
# # cv2.imshow('img',img[10])
# cv2.imshow('img',img[1])
# cv2.waitKey(0)


data = [1,2,3,4]
pg.plot(data)
