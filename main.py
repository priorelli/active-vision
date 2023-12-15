import utils
import config as c
import time
from simulation.manual_control import ManualControl
from simulation.inference import Inference


def main():
    # Parse arguments
    options = utils.get_sim_options()

    # Choose simulation
    if options.manual_control:
        c.fps = 200
        sim = ManualControl()

    elif options.infer:
        c.gain_abs = 4.0
        c.pi_cam = 1.0
        c.task = 'infer'
        sim = Inference()

    elif options.reach:
        c.gain_abs = 4.0
        c.pi_cam = 1.0
        c.task = 'reach'
        sim = Inference()

    elif options.both:
        c.gain_abs = 10.0
        c.pi_cam = 1.2
        c.task = 'both'
        sim = Inference()

    else:
        print('Choose task:')
        print('0 --> infer depth')
        print('1 --> fixate target')
        print('2 --> infer and fixate')
        task = input('Task: ')

        print('\nChoose eye angles:')
        print('0 --> parallel')
        print('1 --> convergent')
        angles = input('Angles: ')

        print('\nChoose context:')
        print('0 --> static environment')
        print('1 --> dynamic environment')
        context = input('Context: ')

        if task == '0':
            c.gain_abs = 4.0
            c.pi_cam = 1.0
            c.task = 'infer'
        elif task == '1':
            c.gain_abs = 4.0
            c.pi_cam = 1.0
            c.task = 'reach'
        else:
            c.gain_abs = 10.0
            c.pi_cam = 1.2
            c.task = 'both'

        c.eye_angles = [0, 0] if angles == '0' else [10, 10]

        c.context = 'static' if context == '0' else 'dynamic'

        time.sleep(0.5)
        sim = Inference()

    # Run simulation
    sim.run()


if __name__ == '__main__':
    main()
