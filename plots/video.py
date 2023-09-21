import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.transforms import Bbox, BboxTransformTo
from matplotlib.lines import Line2D
from pylab import tight_layout
import time
import sys
import config as c
from environment.eyes import Eyes


def record_video(log, width):
    # Initialize arm
    arm = Eyes()

    # Load variables
    n_t = log['angles'].shape[0] * log['angles'].shape[1]

    angles = log['angles'].reshape(n_t, 2)
    est_angles = log['est_angles'].reshape(n_t, 2)

    pos = log['pos'].reshape(n_t, c.n_dim)
    est_pos = log['est_pos'].reshape(n_t, c.n_dim)

    cam = log['cam'].reshape(n_t, c.n_eyes)
    est_cam = log['est_cam'].reshape(n_t, c.n_eyes)

    int_point = log['int_point'].reshape(n_t, c.n_dim)

    # Create plot
    fig, axs = plt.subplots(1, figsize=(16, 14))

    c.height /= 2
    c.width *= 1.5

    # Display parameters
    scale = 1000
    offset_main = np.array([50, 3 * c.height / 4])
    offset_eyes = [np.array([c.width / 4, c.height / 4]),
                   np.array([c.width * 3 / 4, c.height / 4])]

    def animate(n):
        if (n + 1) % 10 == 0:
            sys.stdout.write('\rTrial: {:d} \tStep: {:d}'
                             .format(int(n / c.n_steps) + 1,
                                     (n % c.n_steps) + 1))
            sys.stdout.flush()

        # Clear plot
        axs.clear()
        axs.get_xaxis().set_visible(False)
        axs.get_yaxis().set_visible(False)
        axs.set_xlim(0, c.width)
        axs.set_ylim(0, c.height)
        tight_layout()

        # Draw frame lines
        axs.plot((0, c.width), (c.height / 2, c.height / 2),
                 color='black', zorder=2, linewidth=15)
        axs.plot((c.width / 2, c.width / 2), (0, c.height / 2),
                 color='black', zorder=2, linewidth=15)

        axs.plot((0, c.width), (offset_main[1], offset_main[1]),
                 color='black', zorder=0, linewidth=2)
        axs.plot((offset_main[0], offset_main[0]), (c.height / 2, c.height),
                 color='black', zorder=0, linewidth=2)

        axs.plot((0, c.width), (c.height / 4, c.height / 4),
                 color='black', zorder=0, linewidth=2)
        axs.plot((c.width / 4, c.width / 4), (0, c.height / 2),
                 color='black', zorder=0, linewidth=2)
        axs.plot((c.width * 3 / 4, c.width * 3 / 4), (0, c.height / 2),
                 color='black', zorder=0, linewidth=2)

        # Draw eyes
        for i in range(c.n_eyes):
            eye_w = np.array([0, c.eye_lengths[i]]) + offset_main

            angle = np.radians(angles[n, 0])
            angle -= np.radians(angles[n, 1]) if i == 0 else \
                -np.radians(angles[n, 1])

            inf_pos = eye_w + np.array([1000 * np.cos(angle),
                                        1000 * np.sin(angle)])

            axs.scatter(*eye_w, color='b', s=2.5 * scale, zorder=0)

            plt.plot((eye_w[0], inf_pos[0]), (eye_w[1], inf_pos[1]),
                     color='b', zorder=1, linewidth=4)

        # Draw targets on main screen
        pos_w = pos[n] + offset_main
        est_pos_w = est_pos[n] + offset_main

        axs.scatter(*pos_w, color='r', s=1.5 * scale, zorder=0)
        if c.height / 2 < est_pos_w[1] < c.height:
            axs.scatter(*est_pos_w, color='orange', s=1.5 * scale, zorder=0)

        # Draw targets on eye screens
        for i in range(c.n_eyes):
            cam_w = [0, cam[n, i] * c.scale] + offset_eyes[i]
            if 0 < cam_w[1] < c.height / 2:
                axs.scatter(*cam_w, color='r', s=1.5 * scale, zorder=0)

            est_cam_w = [0, est_cam[n, i] * c.scale] + offset_eyes[i]
            if 0 < est_cam_w[1] < c.height / 2:
                axs.scatter(*est_cam_w, color='orange', s=1.5 * scale, zorder=0)

        # # Draw trajectories
        pos_w_all = pos + offset_main
        est_pos_w_all = est_pos + offset_main
        int_point_w_all = int_point + offset_main

        axs.scatter(*pos_w_all[n - (n % c.n_steps): n + 1].T,
                    color='darkred', linewidth=width + 2, zorder=2)
        axs.scatter(*est_pos_w_all[n - (n % c.n_steps): n + 1].T,
                    color='darkorange', linewidth=width + 2, zorder=2)
        axs.scatter(*int_point_w_all[n - (n % c.n_steps): n + 1].T,
                    color='lightblue', linewidth=width + 2, zorder=2)

    # start = time.time()
    # ani = animation.FuncAnimation(fig, animate, n_t)
    # writer = animation.writers['ffmpeg'](fps=500)
    # ani.save('plots/video.mp4', writer=writer)
    # print('\nTime elapsed:', time.time() - start)

    for i in range(0, c.n_steps, c.n_cycle):
        animate(i - 1 if i > 0 else i)
        plt.savefig('plots/frame%d' % i)

    # animate(100)#c.n_steps - 400)
    # plt.savefig('plots/frame%d' % 100)
