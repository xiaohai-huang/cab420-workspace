import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report


def load_data(data_directory="small_flower_dataset"):
    """Preparation of the training, validation and test sets.
    
    data_directory: the directory that contains the image data
    """
    class_names = os.listdir(data_directory)
    size = (224, 224)

    images = []
    labels = []
    for folder_name in class_names:
        folder_path = os.path.join(data_directory, folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            img = tf.image.resize_with_pad(cv2.imread(fpath), *size).numpy()
            images.append(img)
            labels.append(class_names.index(folder_name))

    images = np.array(images)
    labels = np.array(labels)
    # split into 3 sets
    X_train, X_other, Y_train, Y_other = train_test_split(images, labels, test_size=0.3, random_state=333)
    X_val, X_test, Y_val, Y_test = train_test_split(X_other, Y_other, test_size=0.5, random_state=333)
    return (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names

initial_epochs = 30
fine_tune_epochs = 10
def train_model(dataset, learning_rate=0.01, momentum=0.0):
    """Train a MobileNetV2 model.
    
    dataset: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names
    learning_rate: the learning rate for SGD optimizer
    momentum: the momentum value for SGD optimizer

    return:
    model: the trained model
    history: the history dictionary that contains information about 
    [accuracy, val_accuracy, loss, val_loss]
    """
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), _ = dataset

    num_classes = 5 # num folders
    IMG_SHAPE = (224, 224, 3)
    base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
    base_model.trainable = False

    preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input
    global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
    prediction_layer = tf.keras.layers.Dense(num_classes)


    inputs = keras.Input(shape=IMG_SHAPE)
    x = preprocess_input(inputs)
    x = base_model(x, training=False)
    x = global_average_layer(x)
    x = layers.Dropout(0.2)(x)
    outputs = prediction_layer(x)
    model = keras.Model(inputs, outputs)


    model.compile(optimizer=tf.keras.optimizers.SGD(
                            learning_rate=learning_rate, 
                            momentum=momentum,
                            nesterov=False),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    history = model.fit(X_train, Y_train, epochs=initial_epochs, validation_data=(X_val, Y_val))

    # Fine tuning
    # Un-freeze the top layers of the model

    base_model.trainable = True
    print("Number of layers in the base model: ", len(base_model.layers))

    # Fine-tune from this layer onwards
    fine_tune_at = 100

    # Freeze all the layers before the `fine_tune_at` layer
    for layer in base_model.layers[:fine_tune_at]:
        layer.trainable = False

    fine_tune_learning_rate = 0.00001
    model.compile(optimizer=tf.keras.optimizers.SGD(
                            learning_rate=fine_tune_learning_rate, 
                            momentum=momentum,
                            nesterov=False),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
    
    total_epochs =  initial_epochs + fine_tune_epochs

    history_fine = model.fit(X_train, Y_train,
                         epochs=total_epochs,
                         initial_epoch=history.epoch[-1],
                         validation_data=(X_val, Y_val))

    final_history = {}
    final_history["accuracy"] = history.history["accuracy"] + history_fine.history["accuracy"]
    final_history["val_accuracy"] = history.history["val_accuracy"] + history_fine.history["val_accuracy"]
    final_history["loss"] = history.history["loss"] + history_fine.history["loss"]
    final_history["val_loss"] = history.history["val_loss"] + history_fine.history["val_loss"]

    return model, final_history

def eval_model(model, history, dataset, info=""):
    """Plot the training and validation errors vs time as well as 
    the training and validation accuracies.

    model: a DCNN model
    history: the history dict
    dataset: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names
    
    return the latest validation accuracy
    """
    acc = history['accuracy']
    val_acc = history['val_accuracy']

    loss = history['loss']
    val_loss = history['val_loss']

    plt.figure(figsize=(10, 10))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend()
    plt.title('Training and Validation Accuracy'+info)

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.plot([initial_epochs-1,initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning')
    plt.legend()
    plt.title('Training and Validation Loss'+info)
    plt.xlabel('epoch')
    plt.show()

    # plot confusion matrix
    (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names = dataset
    predition = np.argmax(model.predict(X_test), axis=1)
    loss, test_accuracy = model.evaluate(X_test, Y_test)

    fig = plt.figure(figsize=[10, 10])    
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f"Test Accuracy: {test_accuracy}")  

    confusion_mtx = tf.math.confusion_matrix(Y_test, predition) 
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=class_names, yticklabels=class_names, 
            annot=True, fmt='g', ax=ax)

    # print F1-Score
    print(classification_report(predition, Y_test))

    # Get the last val accuracy
    return val_acc[-1]

def eval_first_model(dataset, learning_rate=0.01, momentum=0.0):
    """Compile and train your model with an SGD
    optimizer using the following parameters
    learning_rate=0.01, momentum=0.0, nesterov=False.
    
    parameters:
        dataset: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names
        learning_rate: the learning rate for SGD optimizer
        momentum: the momentum value for SGD optimizer
    """
    info = "(learning_rate=0.01, momentum=0.0, nesterov=False)"
    model, history = train_model(dataset, learning_rate, momentum)
    eval_model(model, history, dataset,info)
    loss, accuracy = model.evaluate(*dataset[2])
    print('Test accuracy :', accuracy)

def eval_three_learning_rate(dataset, learning_rates=[]):
    """Experiment with 3 different orders of magnitude for the learning rate. 
    Plot the results, draw conclusions.
    
    parameters:
        dataset: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names
        learning_rates: a list of learning rates for SGD optimizer
    
    return: the best learning rate which has the highest validation accuracy
    """
    best_learning_rate = -1
    best_val_accuracy = -1
    for learning_rate in learning_rates:
        info = f"(learning_rate={learning_rate}-momentum=0.0)"
        model, history = train_model(dataset, learning_rate)
        model.save(info)
        val_acc =  eval_model(model, history, dataset,info)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_learning_rate = learning_rate
        loss, accuracy = model.evaluate(*dataset[2])
        print(f'{info}. Test accuracy :', accuracy)

    print(f"The best learning rate is {best_learning_rate}, it achieved {best_val_accuracy} Validation Accuracy")

    return best_learning_rate

def eval_three_momentum(dataset, best_learning_rate, momentum_values=[]):
    """With the best learning rate that you found in the previous task, 
    add a non zero momentum to the training with the SGD optimizer.
    Plot the results, draw conclusions.
    
    parameters:
        dataset: (X_train, Y_train), (X_val, Y_val), (X_test, Y_test), class_names
        best_learning_rate: the learning rate for SGD optimizer
        momentum_values: a list of momentum values for SGD optimizer
    """
    best_momentum = -1
    best_val_accuracy = -1
    for momentum in momentum_values:
        info = f"(learning_rate={best_learning_rate}-momentum={momentum})"
        model, history = train_model(dataset, best_learning_rate, momentum)
        model.save(info)
        val_acc = eval_model(model, history, dataset,info)

        if val_acc > best_val_accuracy:
            best_val_accuracy = val_acc
            best_momentum = momentum
        loss, accuracy = model.evaluate(*dataset[2])
        print(f'{info}. Test accuracy :', accuracy)
    
    print(f"The best momentum is {best_momentum}, it achieved {best_val_accuracy} Validation Accuracy")

if __name__ == "__main__":
    # load the small flower dataset and split into training, validation and test sets.
    dataset = load_data()

    # train a MobileNetV2 network
    # model, history = train_model(dataset, learning_rate=0.01, momentum=0.0)

    # Plot the training and validation errors vs time as well as 
    # the training and validation accuracies
    eval_first_model(dataset)

    # Experiment with 3 different orders of magnitude for the learning rate.
    best_learning_rate = eval_three_learning_rate(dataset,learning_rates=[0.001, 0.01, 1])

    # add a non zero momentum to the training with the SGD optimizer 
    # (consider 3 values for the momentum)
    eval_three_momentum(dataset, best_learning_rate, momentum_values=[0.5, 0.9, 0.98])