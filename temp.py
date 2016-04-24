__author__ = 'ORI'

import numpy as np
import sys
from OriKerasExtension.ThesisHelper import readCompleteMatFile, ExtractDataVer4

from keras.callbacks import ModelCheckpoint
from keras.utils.np_utils import to_categorical
from scipy import stats

from sklearn.utils import shuffle

import keras
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from OriKerasExtension.P300Prediction import accuracy_by_repetition, create_target_table
from keras.regularizers import l2
import pickle
import matplotlib.pyplot as plt



def downsample_data(data, number_of_original_samples, down_samples_param):
    new_number_of_time_stamps = number_of_original_samples / down_samples_param

    temp_data_for_eval = np.zeros((data.shape[0], new_number_of_time_stamps, data.shape[2]))

    for new_i, i in enumerate(range(0, number_of_original_samples, down_samples_param)):
        temp_data_for_eval[:, new_i, :] = np.mean(data[:, range(i, (i + down_samples_param)), :], axis=1)
    return temp_data_for_eval

def create_train_data(gcd_res, down_samples_param):
    all_positive_train = []
    all_negative_train = []


    last_time_stamp = 800
    fist_time_stamp = -200


    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)


    print data_for_eval[0].shape
    temp_data_for_eval = downsample_data(data_for_eval[0],data_for_eval[0].shape[1], down_samples_param)

    positive_train_data_gcd = temp_data_for_eval[
        np.all([gcd_res['train_mode'] != 3, gcd_res['target'] == 1], axis=0)]
    negative_train_data_gcd = temp_data_for_eval[
        np.all([gcd_res['train_mode'] != 3, gcd_res['target'] == 0], axis=0)]
    all_positive_train.append(positive_train_data_gcd)
    all_negative_train.append(negative_train_data_gcd)

    positive_train_data_gcd = np.vstack(all_positive_train)
    negative_train_data_gcd = np.vstack(all_negative_train)

    all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])

    all_tags = np.vstack(
        [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))])
    categorical_tags = to_categorical(all_tags)

    # shuffeled_samples, suffule_tags = shuffle(all_data, categorical_tags, random_state=0)
    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags

def create_test_data(gcd_res, down_samples_param):
    all_positive_train = []
    all_negative_train = []


    last_time_stamp = 800
    fist_time_stamp = -200


    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)


    print data_for_eval[0].shape
    temp_data_for_eval = downsample_data(data_for_eval[0],data_for_eval[0].shape[1], down_samples_param)

    positive_train_data_gcd = temp_data_for_eval[
        np.all([gcd_res['train_mode'] == 3, gcd_res['target'] == 1], axis=0)]
    negative_train_data_gcd = temp_data_for_eval[
        np.all([gcd_res['train_mode'] == 3, gcd_res['target'] == 0], axis=0)]
    all_positive_train.append(positive_train_data_gcd)
    all_negative_train.append(negative_train_data_gcd)

    positive_train_data_gcd = np.vstack(all_positive_train)
    negative_train_data_gcd = np.vstack(all_negative_train)

    all_data = np.vstack([positive_train_data_gcd, negative_train_data_gcd])

    all_tags = np.vstack(
        [np.ones((positive_train_data_gcd.shape[0], 1)), np.zeros((negative_train_data_gcd.shape[0], 1))])
    categorical_tags = to_categorical(all_tags)

    # shuffeled_samples, suffule_tags = shuffle(all_data, categorical_tags, random_state=0)
    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags


def create_data_for_compare_by_repetition(file_name):
    gcd_res = readCompleteMatFile(file_name)
    sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] == 3],
                       train_block=gcd_res['train_block'][gcd_res['train_mode'] == 3],
                       stimulus=gcd_res['stimulus'][gcd_res['train_mode'] == 3])
    return sub_gcd_res

if __name__ == "__main__":

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
    #second_layer_model.fit(color_features, speller_train_tags, nb_epoch=1, show_accuracy=True, class_weight={0: 1, 1: 50})
    file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(data_set_locations[0])

    gcd_res = readCompleteMatFile(file_name)
    down_sample_param = 8
    test_data_gcd, test_target_gcd = create_test_data(gcd_res, down_samples_param=down_sample_param)
    print test_data_gcd.shape
    # test_features = result_func(stats.zscore(test_data_gcd, axis=1).astype(np.float32))
    # test_prediction = second_layer_model.predict(test_features)


    sub_gcd_res = create_data_for_compare_by_repetition(file_name)
    # sub_gcd_res = dict(train_trial=gcd_res['train_trial'][gcd_res['train_mode'] != 1],
    # train_block=gcd_res['train_block'][gcd_res['train_mode'] != 1],
    # stimulus=gcd_res['stimulus'][gcd_res['train_mode'] != 1])
    print np.sort(np.argsort(sub_gcd_res['stimulus']).reshape(30, -1), axis=1).flatten().shape
    _, _, gt_data_for_sum = create_target_table(sub_gcd_res, test_target_gcd[:,1])
    # _, _, actual_data_for_sum = create_target_table(sub_gcd_res, test_prediction[:, 1])
    # subject_results[i] = dict(test_prediction=test_prediction,
    #                           acc_by_rep=accuracy_by_repetition(actual_data_for_sum, gt_data_for_sum, number_of_repetition=10))

    # print "accuracy_by_repetition {0}".format(
    #     accuracy_by_repetition(actual_data_for_sum, gt_data_for_sum, number_of_repetition=10))
    # pass
