from pyglet.window import key
import utils
import config as c
from environment.window import Window


# Define manual control class
class ManualControl(Window):
    def __init__(self):
        super().__init__()
        # Initialize target
        self.sample_target()

    def update(self, dt):
        # Get action from user
        action = self.get_pressed()

        # Update eyes
        self.eyes.update(action)

        # Update target
        if c.context == 'dynamic':
            self.move_target()

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, self.success, self.step)

        # Reset trial
        self.step += 1
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        # Simulation done
        if self.trial == c.n_trials - 1:
            self.stop()
        else:
            # Sample target
            self.sample_target()

            self.step = 0
            self.trial += 1

    # Get action from user input
    def get_pressed(self):
        return [(key.Z in self.keys) - (key.X in self.keys),
                (key.LEFT in self.keys) - (key.RIGHT in self.keys)]
