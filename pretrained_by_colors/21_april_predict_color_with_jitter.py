import pickle

import numpy as np
from keras.callbacks import ModelCheckpoint

from keras.utils.np_utils import to_categorical
from scipy import stats
from sklearn.utils import shuffle
import keras

from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4
from experiments.pretrained_by_colors.color_utils import create_color_dictionary, get_color_from_stimuli
from experiments.pretrained_by_colors.deep_models import create_compile_lstm_model_letter, \
    create_small_compile_dense_model_color

data_source_dir = r'C:\Users\ORI\Documents\Thesis\dataset_all\\'
__author__ = 'ORI'
# I should learn how to load libraries in a more elegant way




# print color_dictionary[27]

'''
define the neural network model:
'''


def create_evaluation_data(gcd_res, down_samples_param):
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'], gcd_res['target'],
                                    -200, 800)

    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    test_data_gcd, test_target_gcd = temp_data_for_eval[gcd_res['train_mode'] != 1], data_for_eval[1][
        gcd_res['train_mode'] != 1]
    return test_data_gcd, test_target_gcd


def downsample_data(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval


def create_train_data_color(gcd_res, down_samples_param):
    last_time_stamp = 800
    fist_time_stamp = -200
    color_dictionary = create_color_dictionary()
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    print data_for_eval[0].shape
    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    all_data = temp_data_for_eval[np.all([gcd_res['train_mode'] != 3], axis=0)]
    categorical_tags = to_categorical(
        get_color_from_stimuli(gcd_res['stimulus'][gcd_res['train_mode'] != 3], color_dictionary))

    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags


def create_letter_test_data_color(gcd_res, down_samples_param):
    last_time_stamp = 800
    fist_time_stamp = -200
    color_dictionary = create_color_dictionary()
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    print data_for_eval[0].shape
    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    all_data = temp_data_for_eval[np.all([gcd_res['train_mode'] == 3], axis=0)]

    categorical_tags = to_categorical(
        get_color_from_stimuli(gcd_res['stimulus'][gcd_res['train_mode'] == 3], color_dictionary))
    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags


def create_data_for_compare_by_repetition(file_name):
    gcd_res = readCompleteMatFile(file_name)
    sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
                       train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
                       stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
    return sub_gcd_res


data_set_locations = ["RSVP_Color116msVPicr.mat",
                      "RSVP_Color116msVPpia.mat",
                      "RSVP_Color116msVPfat.mat",
                      "RSVP_Color116msVPgcb.mat",
                      "RSVP_Color116msVPgcc.mat",
                      "RSVP_Color116msVPgcd.mat",
                      "RSVP_Color116msVPgcf.mat",
                      "RSVP_Color116msVPgcg.mat",
                      "RSVP_Color116msVPgch.mat",
                      "RSVP_Color116msVPiay.mat",
                      "RSVP_Color116msVPicn.mat"];


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.val_losses = []
        self.val_accuracies = []
        self.train_losses = []
        self.train_accuracies = []

    def on_epoch_end(self, batch, logs={}):

        self.val_losses.append(logs.get('val_loss'))
        self.val_accuracies.append(logs.get('val_acc'))
        self.train_losses.append(logs.get('loss'))
        self.train_accuracies.append(logs.get('acc'))



if __name__ == "__main__":


    model = create_compile_lstm_model_letter()
    model_mlp = create_small_compile_dense_model_color()
    original_weights = model.get_weights()

    # create_train_data(gcd_res,8)

    original_weights_mlp = model_mlp.get_weights()
    model.set_weights(original_weights)
    model_mlp.set_weights(original_weights_mlp)


    results = []


    history = LossHistory()
    history_mlp = LossHistory()
    all_history = []

    for subject_name in data_set_locations:
        file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(subject_name)
        model.set_weights(original_weights)
        model_mlp.set_weights(original_weights_mlp)
        print "subject_name : {0}".format(subject_name)
        gcd_res = readCompleteMatFile(file_name)
        subject_results = dict()

        down_sample_param = 8
        train_data, train_tags = create_train_data_color(gcd_res, down_samples_param=down_sample_param)
        print train_tags.shape
        shuffeled_samples, suffule_tags = shuffle(train_data, train_tags, random_state=0)
        test_data, test_tags = create_letter_test_data_color(gcd_res, down_samples_param=down_sample_param)

        train_history = model.fit(stats.zscore(shuffeled_samples, axis=1), suffule_tags,
                                  nb_epoch=50, show_accuracy=True, verbose=1,
                                  validation_data=(stats.zscore(test_data, axis=1), test_tags),
                                  callbacks=[history])
        train_history_mlp = model_mlp.fit(stats.zscore(shuffeled_samples, axis=1), suffule_tags,
                                          nb_epoch=50, show_accuracy=True, verbose=1,
                                          validation_data=(stats.zscore(test_data, axis=1), test_tags),
                                          callbacks=[history_mlp])
        print history.val_accuracies
        #         print model.evaluate(stats.zscore(test_data, axis=1), test_tags, show_accuracy=True)




        all_history.append(dict(subject_name=subject_name,
                                lstm_history=dict(
                                val_accuracies=history.val_accuracies,
                                val_losses=history.val_losses,
                                train_accuracies=history.train_accuracies,
                                train_losses=history.train_losses,
                                weights=model_mlp.get_weights()),
                           mlp_hisotry=dict(
                                val_accuracies=history_mlp.val_accuracies,
                                val_losses=history_mlp.val_losses,
                                train_accuracies=history_mlp.train_accuracies,
                                train_losses=history_mlp.train_losses,
                           weights=model.get_weights())))
        break
        # model.save_weights('{0}_model_weights.h5'.format(subject_name), overwrite=True)

    pickle.dump( all_history, open( "save_21_april_predict_color_take2.p", "wb" ) )
