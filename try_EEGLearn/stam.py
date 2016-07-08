from  eeg_cnn_lib import gen_images
import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
if __name__ == "__main__":

    res = gen_images(sio.loadmat(r'c:\temp\2map.txt')['two_d_map'].T, np.ones((1,63)),30,)
    plt.imshow(res[0,0,:,:],interpolation='none')
    plt.show()
    print "sas"