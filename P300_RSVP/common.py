from abc import ABCMeta, abstractmethod

from keras.utils.np_utils import to_categorical
from scipy import stats
from sklearn.cross_validation import StratifiedShuffleSplit


class GeneralModel(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def predict(self, _X):
        pass

    @abstractmethod
    def fit(self, _X, _y):
        pass

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_params(self):
        pass


class LSTM_EEG(GeneralModel):
    def get_params(self):
        super(LSTM_EEG, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_EEG, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_EEG, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_EEG, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units

        '''
        define the neural network model:

        '''
        from keras.models import Sequential
        from keras.layers.recurrent import LSTM
        from keras.layers.core import Dense, Dropout, Activation
        from keras.regularizers import l2

        self.model = Sequential()
        self.model.add(LSTM(input_dim=55, output_dim=_num_of_hidden_units, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(LSTM(input_dim=_num_of_hidden_units, output_dim=_num_of_hidden_units, return_sequences=False))
        self.model.add(Dense(2, W_regularizer=l2(0.06)))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def fit(self, _X, y):
        from keras.callbacks import ModelCheckpoint

        _y = to_categorical(y)

        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))
        self.model.fit(stats.zscore(_X[sss[0][0]], axis=1), _y[sss[0][0]],
                       nb_epoch=20, show_accuracy=True, verbose=1, validation_data=(
                stats.zscore(_X[sss[0][1]], axis=1), _y[sss[0][1]]),
                       class_weight={0: 1, 1: self.positive_weight},
                       callbacks=[checkpointer])

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))


class LSTM_CNN_EEG(GeneralModel):
    def get_params(self):
        super(LSTM_CNN_EEG, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_CNN_EEG, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_CNN_EEG, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_CNN_EEG, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units

        '''
        define the neural network model:

        '''
        # from keras.layers.extra import *

        from keras.models import Sequential
        # from keras.initializations import norRemal, identity
        from keras.layers.recurrent import GRU
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, TimeDistributedDense, Reshape
        # from keras.layers.wrappers import TimeDistributed
        from keras.layers.convolutional import MaxPooling2D
        from keras.layers.core import Permute




        maxToAdd = 200
        # define our time-distributed setup
        model = Sequential()

        model.add(TimeDistributedDense(10, input_shape=(maxToAdd, 55)))
        # model.add(Convolution2D(1, 1, 10, border_mode='valid', input_shape=(1,maxToAdd, 55)))
        model.add(Activation('tanh'))
        model.add(
            Reshape((1, maxToAdd, 10)))  # this line updated to work with keras 1.0.2
        model.add(Convolution2D(3, 20, 1, border_mode='valid')) # org
        model.add(Activation('tanh'))
        model.add(Convolution2D(1, 1, 1, border_mode='same'))  # org
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(20, 1), border_mode='valid'))
        model.add(Permute((2, 1, 3)))
        model.add(Reshape((9, 10)))  # this line updated to work with keras 1.0.2
        model.add(GRU(output_dim=20, return_sequences=False))
        #
        model.add(Dense(2, activation='softmax'))


        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy')
        self.model = model

        # model.predict(np.random.rand(28, 200, 55).astype(np.float32)).shape

        print model.layers[-1].output_shape
        # print "2 {} {}".format(model.layers[1].output_shape[-3:], (1, maxToAdd, np.prod(model.layers[1].output_shape[-3:])))
        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def fit(self, _X, y):
        from keras.callbacks import ModelCheckpoint

        _y = to_categorical(y)
        # _X = np.expand_dims(np.expand_dims(_X,3),4).transpose([0,1,3,2,4])


        # (batch, times, color_channel, x, y)

        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))

        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=50, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))


class LSTM_CNN_EEG_Comb(GeneralModel):
    def get_params(self):
        super(LSTM_CNN_EEG_Comb, self).get_params()
        return self.model.get_weights()

    def get_name(self):
        super(LSTM_CNN_EEG_Comb, self).get_name()
        return self.__class__.__name__ + "_" + str(self._num_of_hidden_units) + "_" + str(self.positive_weight)

    def reset(self):
        super(LSTM_CNN_EEG_Comb, self).reset()
        self.model.set_weights(self.original_weights)

    def __init__(self, positive_weight, _num_of_hidden_units):
        super(LSTM_CNN_EEG_Comb, self).__init__()
        self.positive_weight = positive_weight
        self._num_of_hidden_units = _num_of_hidden_units

        '''
        define the neural network model:

        '''
        # from keras.layers.extra import *

        from keras.models import Sequential
        # from keras.initializations import norRemal, identity
        from keras.layers.recurrent import GRU, LSTM
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, TimeDistributedDense, Reshape
        # from keras.layers.wrappers import TimeDistributed
        from keras.layers.convolutional import MaxPooling2D
        from keras.layers.core import Permute

        from keras.regularizers import l2, activity_l2

        maxToAdd = 200
        # define our time-distributed setup
        model = Sequential()

        model.add(Reshape((1, maxToAdd, 55), input_shape=(maxToAdd, 55)))  # this line updated to work with keras 1.0.2
        # model.add(TimeDistributedDense(10, input_shape=(maxToAdd, 55)))
        model.add(Convolution2D(3, 12, 55, border_mode='valid', W_regularizer=l2(0.1)))  # org
        model.add(Activation('tanh'))
        model.add(MaxPooling2D(pool_size=(12, 1), border_mode='valid'))
        model.add(Permute((2, 1, 3)))
        model.add(Reshape((model.layers[-1].output_shape[1],
                           model.layers[-1].output_shape[2])))  # this line updated to work with keras 1.0.2
        model.add(LSTM(output_dim=10, return_sequences=False))
        #
        model.add(Dense(2, activation='softmax'))

        model.compile(optimizer='rmsprop',
                      loss='categorical_crossentropy')
        self.model = model

        # model.predict(np.random.rand(28, 200, 55).astype(np.float32)).shape

        print model.layers[-1].output_shape
        # print "2 {} {}".format(model.layers[1].output_shape[-3:], (1, maxToAdd, np.prod(model.layers[1].output_shape[-3:])))
        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def fit(self, _X, y):
        from keras.callbacks import ModelCheckpoint

        _y = to_categorical(y)
        # _X = np.expand_dims(np.expand_dims(_X,3),4).transpose([0,1,3,2,4])


        # (batch, times, color_channel, x, y)

        checkpointer = ModelCheckpoint(filepath=r"c:\temp\25_dec_lstm_with_ds_22.hdf5", verbose=1, save_best_only=True)
        sss = list(StratifiedShuffleSplit(_y[:, 0], n_iter=1, test_size=0.1))

        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=1, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

    def predict(self, _X):
        return self.model.predict(stats.zscore(_X, axis=1))

class CNN_2011_EEG(GeneralModel):
    def __init__(self, positive_weight):
        super(CNN_2011_EEG, self).__init__()
        from keras.models import Sequential
        from keras.layers.convolutional import Convolution2D
        from keras.layers.core import Dense, Activation, Flatten, Reshape
        from keras.layers.convolutional import MaxPooling2D
        from keras.regularizers import l2
        number_of_time_stamps = 200
        number_of_in_channels = 55
        number_of_out_channels =10
        self.model = Sequential()
        self.positive_weight = positive_weight
        self.model.add(Reshape((1,number_of_time_stamps, number_of_in_channels ), input_shape=(number_of_time_stamps, number_of_in_channels)))
        self.model.add(Convolution2D(nb_filter=10,
                                nb_col=number_of_out_channels,
                                nb_row=1,
                                input_shape=(1, number_of_time_stamps, number_of_in_channels),
                                border_mode='same',
                                init='glorot_normal',W_regularizer=l2(0.01)))

        self.model.add(Activation('tanh'))
        self.model.add(MaxPooling2D(pool_size=(1, number_of_in_channels)))
        self.model.add(
            Convolution2D(nb_filter=number_of_out_channels, nb_row=6, nb_col=1, border_mode='same', init='glorot_normal',W_regularizer=l2(0.01)))
        self.model.add(MaxPooling2D(pool_size=(20, 1)))
        self.model.add(Activation('tanh'))
        self.model.add(Flatten())
        self.model.add(Dense(100))
        self.model.add(Activation('tanh'))
        self.model.add(Dense(2))
        self.model.add(Activation('softmax'))
        self.model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        self.original_weights = self.model.get_weights()
        """ :type Sequential"""

    def predict(self, _X):
        super(CNN_2011_EEG, self).predict(_X)
        return self.model.predict(stats.zscore(_X, axis=1))

    def get_params(self):
        return self.model.get_weights()

    def fit(self, _X, _y):
        _y = to_categorical(_y)
        self.model.fit(stats.zscore(_X, axis=1), _y,
                       nb_epoch=5, show_accuracy=True, verbose=1,
                       class_weight={0: 1, 1: self.positive_weight})

    def get_name(self):
        return self.__class__.__name__ + "_" + str(self.positive_weight)


    def reset(self):
        self.model.set_weights(self.original_weights)

