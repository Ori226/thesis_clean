__author__ = 'ORI'
ABC_list = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 't', 'u',
            'v', 'w', 'x', 'y', 'z', '_', '.', '!', '<']

def create_color_dictionary():
    color_dictionary_ = dict()
    for letter_i, letter in enumerate(ABC_list):
        print letter_i
        #     red (fRyGk<)
        if letter in list('fRyGk<'.lower()):
            color_dictionary_[letter_i + 1] = 0
        # blue (iSwc_N)
        if letter in list('iSwc_N'.lower()):
            color_dictionary_[letter_i + 1] = 1
        # green (TBMqAH),
        if letter in list('TBMqAH'.lower()):
            color_dictionary_[letter_i + 1] = 2
        # black (LdvOz.).
        if letter in list('LdvOz.'.lower()):
            color_dictionary_[letter_i + 1] = 3
        # white (pJUX!E)
        if letter in list('pJUX!E'.lower()):
            color_dictionary_[letter_i + 1] = 4
    return color_dictionary_


def get_color_from_stimuli(stimulus_vetor, color_dictionary):
    #     red (fRyGk<),
    #     white (pJUX!E),
    #     blue (iSwc_N),
    #     green (TBMqAH),
    #     black (LdvOz.).
    return [color_dictionary[x] for x in stimulus_vetor]