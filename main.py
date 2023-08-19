import keras.utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPool2D, ReLU, concatenate, Dropout
from keras.models import Model
from sklearn.model_selection import train_test_split
from keras import Sequential
from keras.layers.experimental.preprocessing import RandomFlip, RandomRotation
from keras.applications.inception_v3 import InceptionV3
from keras.models import model_from_json, load_model
from keras.optimizers import Adam
from keras.regularizers import l2

def convert_to_one_hot(Y, C):
    """
    This function implements one-hot encoding
    :param Y: Array to encode
    :param C: Number of classes to encode
    :return: One hot encoded array
    """
    Y = np.eye(C)[Y.reshape(-1)].T
    return Y

def convert_images_list_to_matrix(images_list, image_size = (224, 224)):
    """
    This function converts array of images to pixel matrix
    :param images_list: A python list that includes images
    :param image_size: image size (nw,nh)
    :return: Pixelled image array
    """
    num_images = len(images_list)
    image_size = image_size  # Size of the target variable
    images_array = np.zeros((num_images, *image_size, 3), dtype=np.uint8)

    for i, img in enumerate(images_list):
        img = img.convert("RGB")
        img = img.resize(image_size, Image.LANCZOS)
        img_array = np.array(img)
        images_array[i] = img_array

    return images_array

def plotImage(img_path, showTitle=False):
    """
    This function plots the image in the given file path.
    :param img_path: a string file path
    """
    img = Image.open(img_path)
    plt.imshow(img)
    if showTitle:
        title = img_path.split("/")[1].upper() + " | " + img_path.split("/")[2].split(".")[0]
        plt.title(title)
    plt.axis("off")
    plt.show()

def data_augmenter():
    """
    A data basic data augmenter
    :return: tf.keras.Sequential model
    """
    data_augmention = Sequential()
    data_augmention.add(RandomFlip("horizontal"))
    #data_augmention.add(RandomRotation(0.2))

    return data_augmention

def resnet_block(x, filters, kernel_size, strides=1, padding='same'):
  """ResNet block."""
  x = Conv2D(filters, kernel_size, strides=strides, padding=padding)(x)
  x = BatchNormalization()(x)
  x = Activation('relu')(x)
  x = Conv2D(filters, kernel_size, strides=1, padding=padding)(x)
  x = BatchNormalization()(x)
  return x + x

def resNet50(input_shape=(224, 224, 3), data_augmentation=data_augmenter()):
    """ResNet-50."""
    input_tensor = Input(input_shape)
    x = data_augmentation(input_tensor)
    x = Conv2D(64, (7,7), strides=(2,2), padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = MaxPool2D(3, strides=2)(x)

    x = resnet_block(x, 64, 3)
    x = resnet_block(x, 128, 3, strides=2)
    x = resnet_block(x, 256, 3, strides=2)
    x = resnet_block(x, 512, 3, strides=2)

    x = AveragePooling2D(7)(x)
    x = Flatten()(x)
    x = Dense(4, activation='softmax')(x)

    return Model(inputs = input_tensor, outputs=x)

def five_layered_cnn(input_shape=(224, 224, 3), data_augmentation=data_augmenter()):

    input_tensor = Input(input_shape)
    x = data_augmentation(input_tensor)
    Z1 = Conv2D(filters=32, kernel_size=4, padding="same", strides=1)(x)
    A1 = ReLU()(Z1)
    P1 = MaxPool2D(pool_size=8, strides=8, padding="same")(A1)
    Z2 = Conv2D(filters=64, kernel_size=2, padding="same", strides=1)(P1)
    A2 = ReLU()(Z2)
    P2 = MaxPool2D(pool_size=4, strides=4, padding="same")(A2)
    F = Flatten()(P2)
    outputs = Dense(units=4, activation="softmax")(F)

    model = Model(inputs=input_tensor, outputs=outputs)
    return  model

def inception_model(input_shape=(224, 224, 3), data_augmentation=data_augmenter()):
    input_tensor = Input(input_shape)
    x = data_augmentation(input_tensor)
    x1 = Conv2D(16, (1,1), padding="same", strides=1, activation='relu')(x)
    x2 = Conv2D(32, (3,3), padding="same", strides=1, activation='relu')(x1)
    x3 = Conv2D(64, (1,1), padding="same", strides=1, activation='relu')(x2)
    x4 = Conv2D(128, (5,5), padding="same", strides=1, activation='relu')(x3)
    x5 = Conv2D(filters=256, kernel_size=2, padding="same", strides=1, activation='relu')(x4)

    C = concatenate([x1,x2,x3,x4,x5], axis=3)
    x6 = MaxPool2D(pool_size=4, strides=4, padding="same")(C)
    F = Flatten()(x6)
    outputs = Dense(units=4,  activation="softmax")(F)

    model = Model(inputs=input_tensor, outputs=outputs)
    return model

def ten_layered_cnn(input_shape=(224, 224, 3), data_augmentation=data_augmenter()):
    input_tensor = Input(input_shape)
    x = data_augmentation(input_tensor)
    Z1 = Conv2D(filters=32, kernel_size=4, padding="same", strides=1)(x)
    A1 = ReLU()(Z1)
    P1 = MaxPool2D(pool_size=4, strides=4, padding="same")(A1)

    Z2 = Conv2D(filters=64, kernel_size=2, padding="same", strides=1)(P1)
    A2 = ReLU()(Z2)
    P2 = MaxPool2D(pool_size=4, strides=4, padding="same")(A2)

    Z3 = Conv2D(filters=128, kernel_size=2, padding="same", strides=1)(P2)
    A3 = ReLU()(Z3)
    dropout = Dropout(0.33)(A3)
    P3 = MaxPool2D(pool_size=2, strides=2, padding="same")(dropout)

    Z4 = Conv2D(filters=256,kernel_size=2 , padding="same", strides=1)(P3)
    A4 = ReLU()(Z4)
    P4 = MaxPool2D(pool_size=2, strides=2, padding="same")(A4)

    F = Flatten()(P4)
    D = Dense(units=128, activation="relu")(F)
    outputs = Dense(units=4, activation="softmax")(D)

    model = Model(inputs=input_tensor, outputs=outputs)
    return model

def fifteen_layered_cnn(input_shape=(224, 224, 3), data_augmentation=data_augmenter()):
    input_tensor = Input(input_shape)
    x = data_augmentation(input_tensor)
    Z1 = Conv2D(filters=32, kernel_size=4, padding="same", strides=1)(x)
    A1 = ReLU()(Z1)
    P1 = MaxPool2D(pool_size=4, strides=4, padding="same")(A1)

    Z2 = Conv2D(filters=64, kernel_size=2, padding="same", strides=1)(P1)
    A2 = ReLU()(Z2)
    P2 = MaxPool2D(pool_size=2, strides=2, padding="same")(A2)

    Z3 = Conv2D(filters=128, kernel_size=2, padding="same", strides=1)(P2)
    A3 = ReLU()(Z3)
    P3 = MaxPool2D(pool_size=2, strides=2, padding="same")(A3)

    Z4 = Conv2D(filters=256, kernel_size=2, padding="same", strides=1)(P3)
    A4 = ReLU()(Z4)
    P4 = MaxPool2D(pool_size=2, strides=2, padding="same")(A4)

    Z5 = Conv2D(filters=512, kernel_size=2, padding="same", strides=1)(P4)
    A5 = ReLU()(Z5)
    P5 = MaxPool2D(pool_size=2, strides=2, padding="same")(A5)

    F = Flatten()(P5)
    D1 = Dense(units=128, activation="relu", kernel_regularizer=l2(0.01))(F)
    D2 = Dense(units=64, activation="relu")(D1)
    #dropout = Dropout(0.2)(D2)
    outputs = Dense(units=4, activation="softmax")(D2)

    model = Model(inputs=input_tensor, outputs=outputs)
    return model
def print_dict(dictionary):
    """
    This function prints given dictionary
    :param dictionary: A python dict
    :return: None
    """
    print("\n=======================================")
    print("THE FINAL RATING AVERAGES OF IMAGE CATEGORIES")
    for key in dictionary:
        print(f"{key} -> {dictionary[key]}")
    print("=======================================")

if __name__ == '__main__':

    # Data loading
    df = pd.read_csv("reviews.csv")

    # A sample from the data
    img_path = df.iloc[489,3]
    plotImage(img_path, True)

    # Head of the data frame
    print("Head of the initial data frame:")
    print(df.head())
    print("------------------------------------")

    # Value counts of rating_category column
    rating_categories = df["rating_category"]
    print(rating_categories.value_counts())
    print("------------------------------------")

    plt.hist(rating_categories)
    plt.title("Histogram of Rating Categories")
    plt.show()

    # Label Encoding
    l_encoder = LabelEncoder()
    rating_categories = l_encoder.fit_transform(rating_categories)
    print("First 10 encoded variables:")
    print(rating_categories[:10])
    # taste = 3, menu = 1, outdoor_atmosphere = 2, indoor_atmosphere = 0
    print("Length of rating_categories: ", len(rating_categories))
    print("------------------------------------")

    # Fetching photo paths from df
    photo_paths = df["photo"].values
    print("Some photo paths from the photo_paths array:")
    print(photo_paths)
    print("Length of photo_paths array: " ,len(photo_paths))
    print("------------------------------------")

    # Photo path to images from PIL
    images = []
    for img_path in photo_paths:
        img = Image.open(img_path)
        images.append(img)

    # Separating data as train and test set
    X_train_orig, X_test_orig, Y_train_orig, Y_test_orig = train_test_split(images, rating_categories, test_size= 0.25, random_state=42)

    print("Shapes of some important arrays:")
    X_train_orig = convert_images_list_to_matrix(X_train_orig)
    print("X_train_orig's shape =", X_train_orig.shape)
    X_test_orig = convert_images_list_to_matrix(X_test_orig)
    print("X_test_orig's shape =", X_test_orig.shape)
    print("Y_train_orig's shape =", Y_train_orig.shape)
    print("------------------------------------")

    # Data scaling
    X_train = X_train_orig / 255.
    X_test = X_test_orig / 255.

    # Converting label encoded Y's to one hot
    Y_train = convert_to_one_hot(Y_train_orig, 4).T
    Y_test = convert_to_one_hot(Y_test_orig, 4).T

    # Use saved model?
    useSavedModel = True
    if useSavedModel:
        model = load_model("model.h5")
    else:
        # Creating the model
        model = fifteen_layered_cnn()

        epochs = 30
        l_rate = 4e-4
        adam = Adam(learning_rate=l_rate, ema_momentum=0.9, weight_decay=(l_rate/epochs))
        model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

        # Fit the model
        model.fit(X_train, Y_train, epochs=epochs, batch_size=32)
    print(model.summary())

    # Evaluate the model on test data
    print("Performance of the model on X_test set:")
    model.evaluate(X_test, Y_test)
    print("------------------------------------")

    # Save the model if you want
    save = False
    if save:
        model.save("model.h5")
        print("The model is save to model.h5 file.")
        print("------------------------------------")

    # Load your own restaurant images
    new_reviews = pd.read_csv("sepetcioglu_restaurant.csv")
    img_path = new_reviews.iloc[3,0]
    plotImage(img_path)

    # Head of the new_reviews
    print("Head of the new_reviews data frame:")
    print(new_reviews.head())
    print("------------------------------------")

    # Value counts of new rating_category column
    new_rating_categories = new_reviews["rating_category"]
    print(new_rating_categories.value_counts())
    print("------------------------------------")

    # Encoding new reviews
    new_rating_categories = l_encoder.transform(new_rating_categories)
    Y_true = keras.utils.to_categorical(new_rating_categories)
    print("Y_true values:")
    print(Y_true)
    # taste = 3, menu = 1, outdoor_atmosphere = 2, indoor_atmosphere = 0
    print("------------------------------------")

    # Fetching photo paths from new reviews
    new_photo_paths = new_reviews["photo"].values

    # Photo path to images from PIL
    new_photos = []
    for img_path in new_photo_paths:
        img = Image.open(img_path)
        new_photos.append(img)

    # Converting image to pixel matrix
    X_new = convert_images_list_to_matrix(new_photos)

    # Making predictions
    Y_pred = model.predict(X_new)
    Y_pred = np.round(Y_pred, 2)
    print("Predictions of the model on new reviews:")
    print(Y_pred)
    print("------------------------------------")

    # Similarity of Y_true and Y_pred
    number_of_same_rows = np.sum(np.all(Y_true == Y_pred, axis=1))
    print("Number of comments that the model categorizes correctly in new comments is : ", number_of_same_rows)
    print(f"Model's accuracy on new comments is : %{round(number_of_same_rows/len(new_photo_paths) * 100, 2)}")
    print("------------------------------------")

    # Converting one hot matrices to encoded array
    Y_pred_labels = np.argmax(Y_pred, axis=1)
    Y_true_labels = new_rating_categories

    # Dictionary to store averages per category
    category_averages = {"taste":[0,0], "indoor_atmosphere":[0,0], "outdoor_atmosphere":[0,0], "menu":[0,0]}

    # Looping new reviews
    for i in range(len(Y_pred_labels)):
        if Y_pred_labels[i] == 3:   # taste
            category_averages["taste"][0] += new_reviews.iloc[i,1]
            category_averages["taste"][1] += 1
        elif Y_pred_labels[i] == 2: # outdoor_atmosphere
            category_averages["outdoor_atmosphere"][0] += new_reviews.iloc[i,1]
            category_averages["outdoor_atmosphere"][1] += 1
        elif Y_pred_labels[i] == 1: # menu
            category_averages["menu"][0] += new_reviews.iloc[i,1]
            category_averages["menu"][1] += 1
        else:                       # indoor_atmosphere
            category_averages["indoor_atmosphere"][0] += new_reviews.iloc[i, 1]
            category_averages["indoor_atmosphere"][1] += 1

    # Rating averages of pictures by category
    for key in category_averages:
        category_averages[key] = round(category_averages[key][0] / category_averages[key][1], 2)

    # Final averages of category ratings
    print_dict(category_averages)
