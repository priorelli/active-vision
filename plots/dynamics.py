import numpy as np
import matplotlib.pyplot as plt
import config as c


def plot_dynamics(log, width):
    # Load variables
    pos = log['pos'][-1]
    est_pos = log['est_pos'][-1]

    cam = log['cam'][-1]
    est_cam = log['est_cam'][-1]

    angles = log['angles'][-1]

    # Plots
    e_cam = np.array([0, 0]) - est_cam
    e_pos = np.linalg.norm(pos - est_pos, axis=1)

    fig, axs = plt.subplots(3, figsize=(60, 30))

    axs[0].set_title('Absolute error')
    axs[0].plot(e_pos, lw=width)

    axs[1].set_title('Projection error')
    axs[1].plot(e_cam[:, 0], lw=width)
    axs[1].plot(e_cam[:, 1], lw=width)

    axs[2].set_title('Angles')
    axs[2].plot(angles[:, 0], lw=width)
    axs[2].plot(angles[:, 1], lw=width)

    plt.tight_layout()
    fig.savefig('plots/dynamics_' + c.log_name, bbox_inches='tight')
    # plt.show()
