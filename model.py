
# coding: utf-8

# ### Importing Section
# 

# In[1]:


import csv
import cv2

import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential, Model
from keras.layers import Lambda, Cropping2D, MaxPooling2D, Conv2D, Dropout
from keras.layers.core import Dense, Activation, Flatten 

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


# ### Learning Parameters
# 

# In[ ]:


BATCH_SIZE = 32
RIGHT_LEFT_CORNER_THERESHOLD = 0.2
EPOCHS = 5
AUGMENTATION_FACTOR = 6


# ### Loading images

# In[ ]:


data_set = []
train_set, valid_set, test_set = [], [], []

def load_csv_data():
    spamreader = csv.DictReader(open('./data/driving_log.csv'), delimiter=',')
    for row in spamreader:
        data_set.append(row)
                            
load_csv_data()

train_set, test_set = train_test_split(data_set, test_size=0.2)
train_set, valid_set = train_test_split(train_set, test_size=0.1)

number_of_total_set = AUGMENTATION_FACTOR * len(data_set)
number_of_train_set = AUGMENTATION_FACTOR * len(train_set)
number_of_valid_set = AUGMENTATION_FACTOR * len(valid_set)
number_of_test_set = AUGMENTATION_FACTOR * len(test_set)

print("Total size => {} samples".format(number_of_total_set))
print("Training size => {} samples".format(number_of_train_set))
print("Test size => {} samples".format(number_of_test_set))
print("Validation size => {} samples".format(number_of_valid_set))


# ### Augmentation

# In[9]:


def load_image(path):
    path = path.replace(" ", "")
    image = cv2.imread("./data/{}".format(path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    return image

def flip_image(image):
    return np.fliplr(image)

def change_brightness(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image[:, :, 2] = np.minimum(image[:, :, 2] * (0.5 + np.random.uniform()), 255)
    
    return cv2.cvtColor(image, cv2.COLOR_HSV2RGB)

def random_shadow(image):
    
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    width, height, _ = image.shape
    
    x1, x2 = np.random.randint(0, width / 2), np.random.randint(width / 2, width)
    y1, y2 = np.random.randint(0, height / 2), np.random.randint(height / 2, height)
    
    mask = 0 * image[:, :, 1]
    
    X_m = np.mgrid[0 : image.shape[0], 0 : image.shape[1]][0]
    Y_m = np.mgrid[0 : image.shape[0], 0 : image.shape[1]][1]

    mask[(np.abs(X_m - x1) + np.abs(X_m - x2) + np.abs(Y_m - y1) + np.abs(Y_m - y2) - np.abs(x2 - x1) - np.abs(y2 - y1)) == 0] = 1
    
    image[:, :, 1][mask == 1] = np.minimum(image[:, :, 1][mask == 1] * (0.5 + np.random.uniform()), 255)

    return cv2.cvtColor(image, cv2.COLOR_HLS2RGB)


# ### Temp Section!

# In[1]:


def temp_function():
    
    i1 = load_image("IMG/center_2016_12_01_13_30_48_287.jpg")
    print(i1.shape)
    
    i1 = cv2.GaussianBlur(i1, (5, 5), 0)
    
    print(i1.shape)

    i2 = random_shadow(i1)
    i3 = change_brightness(i1)
    
    fig, axs = plt.subplots(1, 3, figsize=(50, 10))
    axs = axs.ravel()
    
    axs[0].imshow(i1.squeeze())
    axs[0].axis('off')
    
    axs[1].imshow(i2.squeeze())
    axs[1].axis('off')
    
    axs[2].imshow(i3.squeeze())
    axs[2].axis('off')
    
    plt.show()


def temp_function_2():
    from keras.models import load_model

    model = load_model("model.h5")
    
    for i in ["center_2016_12_01_13_30_48_287.jpg", "left_2016_12_01_13_43_56_225.jpg", "center_2016_12_01_13_41_13_967.jpg"]:
        image = load_image("IMG/{}".format(i))
        print(image.shape)
        image_array = np.asarray(image)
        angle = float(model.predict(image_array[None, :, :, :], batch_size=1))

        print(angle)

# temp_function()
# temp_function_2()


# ### Data Generator

# In[ ]:


def load_batch(data_set, batch_size):
    
    while(1):
        shuffle(data_set)
    
        for offset in range(0, len(data_set), batch_size):
            sample_set = data_set[offset : offset + batch_size]
            X_train, Y_train = [], []

            for sample in sample_set:
                steering_center = float(sample["steering"])
            
                steering_left = steering_center + RIGHT_LEFT_CORNER_THERESHOLD
                steering_right = steering_center - RIGHT_LEFT_CORNER_THERESHOLD
                
                left_image = load_image(sample["left"])
                center_image = load_image(sample["center"])
                right_image = load_image(sample["right"])
                
                images = [left_image, center_image, right_image]
                angles = [steering_left, steering_center, steering_right]                    
                
                for i in range(len(images)):

                    images[i] = cv2.GaussianBlur(images[i], (5, 5), 0)
                    
                    if np.random.rand() > 0.6:
                        images[i] = change_brightness(images[i])
                        
                    if np.random.rand() > 0.8:
                        images[i] = random_shadow(images[i])
                    
                    X_train.append(images[i])
                    Y_train.append(angles[i])                                
                    
                    X_train.append(flip_image(images[i]))
                    Y_train.append(-angles[i])
                    
            
            X_train = np.array(X_train)
            Y_train = np.array(Y_train)
 
            yield shuffle(X_train, Y_train)


# ### Visualizing

# In[ ]:


def visualize_augmentation():
    
    showing_samples = 2
    
    images, angles = next(load_batch(train_set, showing_samples))
    
    print("Images shapes => {}".format(images[0].shape))
    
    fig, axs = plt.subplots(showing_samples, AUGMENTATION_FACTOR, figsize=(20, 5))
    axs = axs.ravel()
    
    for i in range(len(images)):
        axs[i].imshow(images[i].squeeze())
        axs[i].set_title("Angle = {0:.2f}".format(angles[i]))
        axs[i].axis('off')
    
    plt.show()

def visualize_angles():
    global data_set
    
    angles = []
    
    for sample in data_set:
        angles.append(float(sample['steering']))
        angles.append(float(sample['steering']) + RIGHT_LEFT_CORNER_THERESHOLD)
        angles.append(float(sample['steering']) - RIGHT_LEFT_CORNER_THERESHOLD)
        
        angles.append(-(float(sample['steering'])))
        angles.append(-(float(sample['steering']) + RIGHT_LEFT_CORNER_THERESHOLD))
        angles.append(-(float(sample['steering']) - RIGHT_LEFT_CORNER_THERESHOLD))
    
    angles = sorted(angles)
    
    fig = plt.figure(figsize=(12, 6))

    plt.hist(angles, 20, range=[-1, 1], histtype='barstacked')
    plt.show()
    
def visualize_learning(history_object):
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('Model mean squared error loss')
    plt.ylabel('Mean squared error loss')
    plt.xlabel('Epoch')
    plt.legend(['Training set', 'Validation set'], loc='upper right')
    plt.show()
    
# visualize_augmentation()
# visualize_angles()


# ### Network Artitucture

# In[ ]:


def build_network():
    model = Sequential()

    model.add(Lambda(lambda x: (x / 255.) - 0.5, input_shape=(160, 320, 3)))
    model.add(Cropping2D(cropping=((70, 25), (0, 0)), input_shape=(160, 320, 3)))
    
    model.add(Conv2D(24, 5, strides=(2, 2), activation="elu"))
    model.add(Conv2D(36, 5, strides=(2, 2), activation="elu"))
    model.add(Conv2D(48, 5, strides=(2, 2), activation="elu"))
    
    model.add(Conv2D(64, 3, strides=(1, 1), activation="elu"))
    model.add(Conv2D(64, 3, strides=(1, 1), activation="elu"))
    
    model.add(Flatten())
    model.add(Dropout(0.5))
    
    model.add(Dense(100, activation='elu'))
    model.add(Dense(50, activation='elu'))
    model.add(Dense(10, activation='elu'))
    model.add(Dense(1))
    
    return model


# ### Training

# In[ ]:


train_generator = load_batch(train_set, BATCH_SIZE)
validation_generator = load_batch(valid_set, BATCH_SIZE)
test_generator = load_batch(test_set, BATCH_SIZE)

network = build_network()
network.compile(loss='mse', optimizer='adam')

try:
    history_object = network.fit_generator(train_generator, steps_per_epoch=number_of_train_set, validation_data=validation_generator, validation_steps=number_of_valid_set, epochs=EPOCHS, verbose=1)
except KeyboardInterrupt:
    pass

network.save("model.h5")

for key, value in history_object.history.items():
    print('{}: {}'.format(key, value))
    
visualize_learning(history_object)
    
metrics = network.evaluate_generator(test_generator, steps=number_of_test_set)
print("Test set error => {}".format(metrics))

