import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import numpy as np

import pydart2 as pydart
import time, sys



if __name__ == "__main__":
    if len(sys.argv) > 1:
        fnames = sys.argv[1:]
    else:
        assert False

    data_sets = [np.loadtxt(fname) for fname in fnames]



    current_step = 0
    max_step = np.min([len(data) for data in data_sets])
    data_dim = len(data_sets[0][0])

    while True:
        mode = input('Select visualization mode: 1. per frame, 2. per dimension\n')
        if int(mode) == 1:
            plt.ion()
            plt.show()

            while True:
                plt.clf()
                for i in range(len(data_sets)):
                    plt.plot(np.arange(data_dim), data_sets[i][current_step], label=str(i))
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
                        for j in range(len(data_sets)):
                            plt.plot(np.arange(data_dim), data_sets[j][i])
                        plt.legend()
                        plt.draw()
                        plt.pause(0.002)

                if current_step > max_step - 1:
                    print('Exceed max frame!')
                    current_step = 0
                if current_step < 0:
                    current_step = max_step - 1
        elif int(mode) == 2:
            current_range = np.arange(data_dim)
            same_plot = True
            while True:
                plt.clf()
                for id, d in enumerate(current_range):
                    if same_plot:
                        plt.subplot(len(current_range), 1, id+1)
                    for i in range(len(data_sets)):
                        plt.plot(data_sets[i][:, d], label=str(i)+'_'+str(d))
                    plt.legend()
                plt.draw()
                plt.pause(0.001)

                sel_dims = input('select dimensions to visualize, seperated by space. Use \'all\' to '+
                                 'visualize all dimensions. Use \'same\' to toggle visualize on the same plot.\n')
                if sel_dims == 'all':
                    current_range = np.arange(len(data_sets))
                elif sel_dims == 'same':
                    same_plot = not same_plot
                else:
                    current_range = [int(v) for v in sel_dims.split(' ')]
