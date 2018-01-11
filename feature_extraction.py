import pickle
import tensorflow as tf
import numpy as np
from keras.models import Model
from keras.layers import Input, Flatten, Dense

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('training_file', '', "Bottleneck features training file (.p)")
flags.DEFINE_string('validation_file', '', "Bottleneck features validation file (.p)")

flags.DEFINE_integer('epochs', 50, "The number of epochs.")
flags.DEFINE_integer('batch_size', 256, "The batch size.")


def load_bottleneck_data(training_file, validation_file):
    """
    Utility function to load bottleneck features.

    Arguments:
        training_file - String
        validation_file - String
    """
    print("Training file", training_file)
    print("Validation file", validation_file)

    with open(training_file, 'rb') as f:
        train_data = pickle.load(f)
    with open(validation_file, 'rb') as f:
        validation_data = pickle.load(f)

    X_train = train_data['features']
    y_train = train_data['labels']
    X_val = validation_data['features']
    y_val = validation_data['labels']

    return X_train, y_train, X_val, y_val


def main(_):
    # load bottleneck data
    X_train, y_train, X_val, y_val = load_bottleneck_data(FLAGS.training_file, FLAGS.validation_file)

    print(X_train.shape, y_train.shape)
    print(X_val.shape, y_val.shape)

    # Define model and hyperparams
    if FLAGS.training_file.split('_')[1] == 'cifar10':
        num_classes = 10
    elif FLAGS.training_file.split('_')[1] == 'traffic':
        num_classes = 43
    else:
        raise ValueError("couldn't infer number of classes from file name - check code")

    # This returns a tensor
    input_shape = X_train.shape[1:]
    x = Input(shape=input_shape)

    # a layer instance is callable on a tensor, and returns a tensor
    y = Flatten()(x)
    
    # output_shape = (None, num_classes)
    # Using softmax, so output will be probability distribution with num_classes many column matrices
    y = Dense(num_classes, activation='softmax')(y) # output_shape = (None, num_classes)
    
    model = Model(x, y)

    # y_train is a sparse representation (ie each example is a single number refering to the index of
    # the correct class). therefore, can use sparse loss algorithm to take care of transformation
    # of sparse representation to one-hot vector so that can perform cross entropy between softmax
    # output and one-hot labels
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


    # Train model
    model.fit(X_train, y_train, batch_size=FLAGS.batch_size, nb_epoch=FLAGS.epochs,
              validation_data=(X_val, y_val), shuffle=True)  # starts training


# parses flags and calls the `main` function above
if __name__ == '__main__':
    tf.app.run()
