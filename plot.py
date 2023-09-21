import seaborn as sns
import numpy as np
import utils
import config as c
from plots.dynamics import plot_dynamics
from plots.video import record_video

sns.set_theme(style='darkgrid', font_scale=3.0)


def main():
    width = 5

    # Parse arguments
    options = utils.get_plot_options()

    # Choose plot to display
    if options.dynamics:
        log = np.load('simulation/log_' + c.log_name + '.npz')
        plot_dynamics(log, width)

    elif options.video:
        log = np.load('simulation/log_' + c.log_name + '.npz')
        record_video(log, width)


if __name__ == '__main__':
    main()
