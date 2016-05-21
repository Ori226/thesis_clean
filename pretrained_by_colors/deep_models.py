import keras
from keras.layers import Convolution2D, Activation, MaxPooling2D, Flatten, Dense, LSTM, Dropout
from keras.models import Sequential
from keras.regularizers import l2

__author__ = 'ORI'


def create_compile_cnn_model():
    model = Sequential()

    number_of_time_stamps = 20
    number_of_out_channels = 10
    number_of_in_channels = 55

    model.add(Convolution2D(nb_filter=10,
                            nb_col=number_of_out_channels,
                            nb_row=1,
                            input_shape=(1, number_of_time_stamps, number_of_in_channels),
                            border_mode='same',
                            init='glorot_normal'))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(1, number_of_in_channels)))
    model.add(
        Convolution2D(nb_filter=number_of_out_channels, nb_row=6, nb_col=1, border_mode='same', init='glorot_normal'))
    model.add(MaxPooling2D(pool_size=(20, 1)))
    model.add(Activation('relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Activation('relu'))
    model.add(Dense(2))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='sgd')
    return model


def create_compile_lstm_model():
    """
    define the neural network model:
    :return:
    """

    model_lstm = Sequential()

    model_lstm.add(LSTM(input_dim=55, output_dim=20, return_sequences=True))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(LSTM(input_dim=20, output_dim=20, return_sequences=False))
    model_lstm.add(Dense(2, W_regularizer=l2(0.06)))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


def create_compile_lstm_model_letter():
    """
    define the neural network model:
    """
    model_lstm = Sequential()

    model_lstm.add(LSTM(input_dim=55, output_dim=20, return_sequences=True))
    model_lstm.add(Dropout(0.01))
    model_lstm.add(LSTM(input_dim=20, output_dim=20, return_sequences=False))
    #     model_lstm.add(Dropout(0.01))
    #     model_lstm.add(LSTM(input_dim=20, output_dim=20,return_sequences=False))
    model_lstm.add(Dense(5, W_regularizer=l2(0.006)))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


def create_compile_dense_model():
    """
    define the neural network model:
    """
    model_lstm = Sequential()
    model_lstm.add(keras.layers.core.Flatten(input_shape=(55, 100)))
    model_lstm.add(Dense(input_dim=55 * 100, output_dim=30, W_regularizer=l2(0.06)))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(2))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


def create_small_compile_dense_model():
    """
    define the neural network model:
    """
    model_lstm = Sequential()
    model_lstm.add(keras.layers.core.Flatten(input_shape=(55, 25)))
    model_lstm.add(Dense(input_dim=55 * 25, output_dim=20))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(output_dim=20, W_regularizer=l2(0.06)))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(2))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


def create_small_compile_dense_model_color():
    """
    define the neural network model:
    """
    model_lstm = Sequential()
    model_lstm.add(keras.layers.core.Flatten(input_shape=(55, 25)))
    model_lstm.add(Dense(input_dim=55 * 25, output_dim=20))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(output_dim=20, W_regularizer=l2(0.06)))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(5))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm


def create_small_compile_dense_model_color_less_params():
    """
    define the neural network model:
    """
    model_lstm = Sequential()
    model_lstm.add(keras.layers.core.Flatten(input_shape=(55, 25)))
    model_lstm.add(Dense(input_dim=55 * 25, output_dim=5))
    model_lstm.add(Dropout(0.3))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(output_dim=5, W_regularizer=l2(0.06)))
    model_lstm.add(Activation('tanh'))
    model_lstm.add(Dense(5))
    model_lstm.add(Activation('softmax'))
    model_lstm.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model_lstm