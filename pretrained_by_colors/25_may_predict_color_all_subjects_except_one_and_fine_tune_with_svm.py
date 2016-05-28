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

def create_train_data_color_all_mode(gcd_res, down_samples_param):
    last_time_stamp = 800
    fist_time_stamp = -200
    color_dictionary = create_color_dictionary()
    data_for_eval = ExtractDataVer4(gcd_res['all_relevant_channels'], gcd_res['marker_positions'],
                                    gcd_res['target'], fist_time_stamp, last_time_stamp)

    print data_for_eval[0].shape
    temp_data_for_eval = downsample_data(data_for_eval[0], data_for_eval[0].shape[1], down_samples_param)

    all_data = temp_data_for_eval
    categorical_tags = to_categorical(
        get_color_from_stimuli(gcd_res, color_dictionary))

    shuffeled_samples, suffule_tags = (all_data, categorical_tags)
    return shuffeled_samples, suffule_tags


def create_letter_test_data_color(gcd_res, down_samples_param, jitter=20):
    last_time_stamp = 800 + jitter
    fist_time_stamp = -200 +jitter
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

def read_single_subject(target_subject,jitter = 20):
    file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(target_subject)

    gcd_res = readCompleteMatFile(file_name)

    down_sample_param = 8
    train_data, train_tags = create_train_data_color(gcd_res,
                                                                                     down_samples_param=down_sample_param)

    test_data, test_tags = create_letter_test_data_color(gcd_res,
                                                         down_samples_param=down_sample_param, jitter=jitter)

    return train_data, train_tags, test_data, test_tags

def read_all_except_one(target_subject):
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

    data_set_locations = ["RSVP_Color116msVPicr.mat",
                          "RSVP_Color116msVPpia.mat",
                          "RSVP_Color116msVPfat.mat",
                          "RSVP_Color116msVPiay.mat",
                          "RSVP_Color116msVPicn.mat"];
    train_data = None
    train_tags = None

    for subject_name in data_set_locations:
        file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(subject_name)
        if target_subject == subject_name:
            continue
        gcd_res = readCompleteMatFile(file_name)

        down_sample_param = 8
        single_subjects_train_data, single_subjects_train_tags = create_train_data_color(gcd_res,
                                                                                         down_samples_param=down_sample_param)
        if train_data is None:
            train_data = single_subjects_train_data
            train_tags = single_subjects_train_tags
        else:
            train_data = np.vstack([train_data, single_subjects_train_data])
            train_tags = np.vstack([train_tags, single_subjects_train_tags])

    gcd_res = readCompleteMatFile(r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(target_subject))
    test_data, test_tags = create_letter_test_data_color(gcd_res,
                                                                                     down_samples_param=down_sample_param)

    return train_data, train_tags, test_data, test_tags


def read_all_except_one(target_subject, skip_subject=False):
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

    data_set_locations = ["RSVP_Color116msVPicr.mat",
                          "RSVP_Color116msVPpia.mat",
                          "RSVP_Color116msVPfat.mat",
                          "RSVP_Color116msVPiay.mat",
                          "RSVP_Color116msVPicn.mat"];
    train_data = None
    train_tags = None

    for subject_name in data_set_locations:
        file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(subject_name)
        if skip_subject:
            if target_subject == subject_name:
                continue
        gcd_res = readCompleteMatFile(file_name)

        down_sample_param = 8
        single_subjects_train_data, single_subjects_train_tags = create_train_data_color(gcd_res,
                                                                                         down_samples_param=down_sample_param)
        if train_data is None:
            train_data = single_subjects_train_data
            train_tags = single_subjects_train_tags
        else:
            train_data = np.vstack([train_data, single_subjects_train_data])
            train_tags = np.vstack([train_tags, single_subjects_train_tags])

    gcd_res = readCompleteMatFile(r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(target_subject))
    test_data, test_tags = create_letter_test_data_color(gcd_res,
                                                                                     down_samples_param=down_sample_param)

    return train_data, train_tags, test_data, test_tags

class TrainedModelData(object):
    def __init__(self, trained_model_file):
        super(TrainedModelData, self).__init__()
        # self.trained_model_data = pickle.load(open(
        #     r"C:\git\thesis_clean_v2\experiments\pretrained_by_colors\save_26_april_predict_color_all_subjects_aggregated.p",
        #     "rb"))
        print "start reading"
        self.trained_model_data = pickle.load(open(
            trained_model_file,
            "rb"))

        print "done reading"



    def get_lstm_weight(self, subject_name):
        print "subject_name:{0}".format(subject_name)
        print len(self.trained_model_data)
        subject_data = next(x for x in self.trained_model_data if x[0]['subject_name'] == subject_name)[0]
        return subject_data['lstm_history']['weights']

    def get_mlp_weight(self, subject_name):
        subject_data = next(x for x in self.trained_model_data if x[0]['subject_name'] == subject_name)[0]
        return subject_data['mlp_history']['weights']


import os
from sklearn import svm
from sklearn.lda import LDA
if __name__ == "__main__":
    script_file_name = os.path.splitext(os.path.basename(__file__))[0]

    load_trained_model = True


    train_model_weight_file = r"C:\git\thesis_clean_v2\experiments\pretrained_by_colors\save_28_april_predict_color_all_subjects_aggregated_except_one.p"

    # trained_model = TrainedModelData(train_model_weight_file)
    # print "start compiling"
    # model = create_compile_lstm_model_letter()
    # model_mlp = create_small_compile_dense_model_color()
    # print "done compiling"
    # original_weights = model.get_weights()
    #
    #
    #
    # original_weights_mlp = model_mlp.get_weights()
    # model.set_weights(original_weights)
    # model_mlp.set_weights(original_weights_mlp)


    results = []


    history = LossHistory()
    history_mlp = LossHistory()
    all_history = []

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

    for subject_name in data_set_locations:
        train_data, train_tags, test_data, test_tags = read_all_except_one(target_subject=subject_name)
        # clf = svm.SVC(max_iter=2)
        clf = LDA()
        # clf.fit(X, y)


        clf.fit(X=train_data.reshape(train_data.shape[0],-1),y=np.argmax(train_tags,axis=1))
        prediction = clf.predict(test_data.reshape(test_data.shape[0],-1))
        num_of_acc = [x == y for x,y in zip(prediction, np.argmax(test_tags,axis=1))]
        print "name:{} acc:{}".format(subject_name, np.sum(num_of_acc).astype(np.float) / prediction.shape[0])
        # down_sample_param = 8
        # train_data, train_tags = create_train_data_color(gcd_res, down_samples_param=down_sample_param)
        # print train_tags.shape
        # shuffeled_samples, suffule_tags = shuffle(train_data, train_tags, random_state=0)
        #
        # # if load_trained_model:
        # #     model.set_weights(trained_model.get_lstm_weight(subject_name))
        # #     model_mlp.set_weights(trained_model.get_mlp_weight(subject_name))
        # #
        # # train_history_mlp = model_mlp.fit(stats.zscore(shuffeled_samples, axis=1), suffule_tags,
        # #                                   nb_epoch=20, show_accuracy=True, verbose=1,
        # #                                   validation_data=(stats.zscore(test_data, axis=1), test_tags),
        # #                                   callbacks=[history_mlp])
        # #
        # # train_history = model.fit(stats.zscore(shuffeled_samples, axis=1), suffule_tags,
        # #                           nb_epoch=20, show_accuracy=True, verbose=1,
        # #                           validation_data=(stats.zscore(test_data, axis=1), test_tags),
        # #                           callbacks=[history])
        #
        # all_training_history = [dict(subject_name=subject_name,
        #                     lstm_history=dict(
        #                         val_accuracies=history.val_accuracies,
        #                         val_losses=history.val_losses,
        #                         train_accuracies=history.train_accuracies,
        #                         train_losses=history.train_losses,
        #                         weights=model.get_weights()),
        #                     mlp_history=dict(
        #                         val_accuracies=history_mlp.val_accuracies,
        #                         val_losses=history_mlp.val_losses,
        #                         train_accuracies=history_mlp.train_accuracies,
        #                         train_losses=history_mlp.train_losses,
        #                         weights=model_mlp.get_weights()))]
        # all_history.append(all_training_history)



    """
    fine tuning phase
    """

    pickle.dump(all_history, open("save_" +script_file_name+".p", "wb"))
    #
    # subject_results = dict()
    # for subject_name in data_set_locations:
    #     file_name = r'C:\Users\ORI\Documents\Thesis\dataset_all\{0}'.format(subject_name)
    #     test_data, test_tags = create_letter_test_data_color(gcd_res, down_samples_param=down_sample_param)
    #
    #     #         print model.evaluate(stats.zscore(test_data, axis=1), test_tags, show_accuracy=True)
    #
    #
    #
    #
    #     all_history.append(dict(subject_name=subject_name,
    #                             lstm_history=dict(
    #                             val_accuracies=history.val_accuracies,
    #                             val_losses=history.val_losses,
    #                             train_accuracies=history.train_accuracies,
    #                             train_losses=history.train_losses,
    #                             weights=model.get_weights()),
    #                        mlp_history=dict(
    #                             val_accuracies=history_mlp.val_accuracies,
    #                             val_losses=history_mlp.val_losses,
    #                             train_accuracies=history_mlp.train_accuracies,
    #                             train_losses=history_mlp.train_losses,
    #                        weights=model_mlp.get_weights())))
    #
    #     # model.save_weights('{0}_model_weights.h5'.format(subject_name), overwrite=True)
    #
    # pickle.dump( all_history, open( "save_25_april_predict_color_all_subjects.p", "wb" ) )
