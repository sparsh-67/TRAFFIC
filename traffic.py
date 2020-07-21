import cv2
import numpy as np
import os
import sys
import tensorflow as tf

import glob

from sklearn.model_selection import train_test_split

EPOCHS = 30
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4


def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
	images=[]
	labels=[]
	dim=(IMG_WIDTH ,IMG_HEIGHT)
	for it in range(44):
		temp_img=[np.array(cv2.imread(file)) for file in glob.glob("gtsrb/"+str(it)+"/*.ppm")]
		temp_label=[it]*(len(temp_img))
		images=images+temp_img
		labels=labels+temp_label
	images=[cv2.cvtColor(bla,cv2.COLOR_BGR2RGB) for bla in images]
	images=[cv2.resize(np.array(img),dim) for img in images]
	return (np.array(images),np.array(labels))







def get_model():
	model = tf.keras.models.Sequential([
		tf.keras.layers.Conv2D(6, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
		tf.keras.layers.AveragePooling2D(),
		tf.keras.layers.Dropout(0.2),
		tf.keras.layers.Conv2D(16, (5, 5), activation="relu", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)),
		tf.keras.layers.AveragePooling2D(),
		tf.keras.layers.Flatten(),
		tf.keras.layers.Dense(120, activation="relu"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(84, activation="relu"),
		tf.keras.layers.Dropout(0.5),
		tf.keras.layers.Dense(43, activation="softmax")
	])
	model.compile(
	    optimizer="adam",
	    loss="categorical_crossentropy",
	    metrics=["accuracy"]
	)
	return model

if __name__ == "__main__":
    main()
