import numpy as np

import matplotlib.pyplot as plt


if __name__ == '__main__':

    em_data = np.load('em_data.npy')
    occam_data = np.load('occam_data.npy')

    # time vs number of measurements
    occam_time = np.mean(occam_data[:, :, 1], axis=1)
    em_time = np.mean(em_data[:, :, 1], axis=1)
    xticks = 1 + np.arange(len(occam_time)).astype(np.int)
    plt.plot(xticks, occam_time)
    plt.plot(xticks, em_time)
    plt.xticks(xticks)
    plt.yscale('log')
    plt.title('OccamSAM vs EM Performance')
    plt.xlabel('Maximum Number of Measurements per Frame')
    plt.ylabel('Avg. Time to Convergence (s)')
    plt.legend(['OccamSAM', 'EM'])
    plt.show()

    # accuracy scatter
    occam_error = occam_data[2, :, 0].flatten()
    occam_time = occam_data[2, :, 1].flatten()
    em_error = em_data[2, :, 0].flatten()
    em_time = em_data[2, :, 1].flatten()
    plt.scatter(occam_time, occam_error)
    plt.scatter(em_time, em_error)
    plt.xscale('log')
    plt.title('Maximum Number of Measurements = 3')
    plt.xlabel('Time to Convergence (s)')
    plt.ylabel('Mean Error')
    plt.legend(['OccamSAM', 'EM'])
    plt.show()


