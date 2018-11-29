import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import pydart2 as pydart
import time, sys



if __name__ == "__main__":
    if len(sys.argv) > 1:
        fname = sys.argv[1]
    else:
        assert False

    data = np.loadtxt(fname)

    plt.ion()
    plt.show()

    current_step = 0
    max_step = len(data)
    data_dim = len(data[0])
    while True:
        plt.clf()
        plt.plot(np.arange(data_dim), data[current_step])
        plt.legend()
        plt.draw()
        plt.pause(0.001)

        goal_frame = input('input frame number to go to, or play a range by inputting \"start_frame-to_frame\", max frame ' + str(max_step))
        if '-' not in goal_frame:
            current_step = int(goal_frame)
        else:
            start_frame = int(goal_frame.split('-')[0])
            end_frame = int(goal_frame.split('-')[1])
            for i in range(start_frame, end_frame):
                if i > max_step - 1:
                    print('Exceed max frame!')
                    break
                plt.clf()
                plt.plot(np.arange(data_dim), data[i])
                plt.legend()
                plt.draw()
                plt.pause(0.002)

        if current_step > max_step - 1:
            print('Exceed max frame!')
            current_step = 0
        if current_step < 0:
            current_step = max_step - 1

