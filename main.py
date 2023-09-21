import utils
import config as c
from simulation.manual_control import ManualControl
from simulation.inference import Inference


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.manual_control:
        sim = ManualControl()

    else:
        if options.infer:
            c.gain_ext = 4.0
            c.pi_cam = 1.0
            c.task = 'infer'

        elif options.reach:
            c.gain_ext = 4.0
            c.pi_cam = 1.0
            c.task = 'reach'

        elif options.both:
            c.gain_ext = 10.0
            c.pi_cam = 1.2
            c.task = 'both'

        sim = Inference()

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
