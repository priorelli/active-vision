import sys
import argparse
import numpy as np
import config as c


# Normalize data
def normalize(x, limits):
    limits = np.array(limits)

    x_norm = (x - limits[0]) / (limits[1] - limits[0])
    x_norm = x_norm * 2 - 1

    return x_norm


# Denormalize data
def denormalize(x, limits):
    limits = np.array(limits)

    x_denorm = (x + 1) / 2
    x_denorm = x_denorm * (limits[1] - limits[0]) + limits[0]

    return x_denorm


# Add Gaussian noise to array
def add_gaussian_noise(array, noise):
    sigma = noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))


# Parse arguments for simulation
def get_sim_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--manual-control',
                        action='store_true', help='Start manual control')
    parser.add_argument('-i', '--infer',
                        action='store_true', help='Estimate depth')
    parser.add_argument('-r', '--reach',
                        action='store_true', help='Fixate target')
    parser.add_argument('-b', '--both',
                        action='store_true', help='Estimate depth and '
                                                  'fixate target')
    parser.add_argument('-a', '--ask-params',
                        action='store_true', help='Ask parameters')

    args = parser.parse_args()
    return args


# Parse arguments for plots
def get_plot_options():
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dynamics',
                        action='store_true', help='Plot dynamics')
    parser.add_argument('-v', '--video',
                        action='store_true', help='Record video')

    args = parser.parse_args()
    return args


# Print simulation info
def print_info(trial, success, step):
    sys.stdout.write('\rTrial: {:4d}({:4d})/{:d} \t '
                     'Step: {:5d}/{:d} \t Accuracy: {:6.2f}%'
                     .format(trial + 1, int(success), c.n_trials,
                             step + 1, c.n_steps,
                             success * 100 / (trial + 1)))
    sys.stdout.flush()


# Print debug info
def print_debug(step, mu, lkh, E_g, E_i):
    if c.debug:
        np.set_printoptions(precision=4, suppress=True)

        names = ['Step', 'mu_abs', 'mu_theta', 'mu_cam',
                 'lkh_abs', 'lkh_theta', 'lkh_prop',
                 'lkh_cam', 'e_cam', 'e_vis', 'e_mu_cam']
        vars_ = [step, *mu, lkh['abs'], lkh['theta'], lkh['prop'],
                 lkh['cam'], E_g[0], E_g[1], E_i[2]]

        for name, var in zip(names, vars_):
            print(name + ':', var, '\n')
        input()


# Compute score
def get_score(ground, est, mean=True):
    error = np.linalg.norm(ground - est, axis=2)
    f_error = error[:, -1]
    acc = (f_error < c.reach_dist) * 100

    time = []
    for ep in error:
        reached = np.where(ep < c.reach_dist)
        if reached[0].size > 0:
            time.append(reached[0][0])

    if mean:
        return np.mean(acc), np.mean(f_error), np.mean(time)
    else:
        return acc, f_error, time


# Print score
def print_score(log, time):
    score = get_score(log.pos, log.est_pos)

    print('\n' + '=' * 30)
    for m, measure in enumerate(('Acc', 'Error', 'Time')):
        print('{:s}:\t\t{:.2f}'.format(measure, score[m]))
    print('\nTime elapsed:\t{:.2f}'.format(time))

    print('\n' + '=' * 30)
    print('\nTarget pos:\t{:+5.2f} {:+5.2f}'.format(*log.pos[-1, -1]))
    print('Est target pos:\t{:+5.2f} {:+5.2f}\n'.format(*log.est_pos[-1, -1]))

    print('\t\t\tLEFT\t\t\tRIGHT')
    print('Angles:\t\t\t{:+4.2f}\t\t\t{:+4.2f}'.format(*log.angles[-1, -1]))
    print('Est angles:\t\t{:+4.2f}\t\t\t{:+4.2f}'.format(
        *log.est_angles[-1, -1]))

    print('Lengths:\t\t{:+5.2f}\t\t\t{:+5.2f}'.format(*log.lengths[-1, -1]))
    print('Est lengths:\t\t{:+5.2f}\t\t\t{:+5.2f}'.format(
        *log.est_lengths[-1, -1]))

    print('Target cam:\t\t{:+5.2f}\t\t\t{:+5.2f}'.format(*log.cam[-1, -1]))
    print('Est target cam:\t\t{:+5.2f}\t\t\t{:+5.2f}'.format(
        *log.est_cam[-1, -1]))
