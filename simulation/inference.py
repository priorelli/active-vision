import time
import utils
import config as c
from environment.window import Window
from environment.log import Log
from simulation.agent import Agent


# Define inference class
class Inference(Window):
    def __init__(self):
        super().__init__()
        # Initialize target
        self.sample_target()

        # Initialize agent
        self.agent = Agent(self.eyes)
        self.agent.init_belief()

        if c.task == 'reach':
            self.agent.mu_ext[0] = utils.normalize(
                self.target_pos, c.norm_cart)

        # Initialize error tracking
        self.log = Log()

        self.time = time.time()

    def update(self, dt):
        # Get observations
        S = self.get_visual_obs(), self.get_prop_obs()

        # Perform free energy step
        action = self.agent.inference_step(S, self.step)

        # Update eyes
        self.eyes.update(action)

        # Move objects
        if c.context == 'dynamic':
            self.move_target()

        # Track log
        self.log.track(self.step, self.trial, self.agent,
                       self.eyes, self.target_pos, S[0])

        # Print info
        if (self.step + 1) % 100 == 0:
            utils.print_info(self.trial, self.success, self.step)

        # Reset trial
        self.step += 1
        if self.step == c.n_steps:
            self.reset_trial()

    def reset_trial(self):
        self.success += self.task_done(self.agent.mu_ext[0],
                                       self.agent.mu_cam[0])

        # Simulation done
        if self.trial == c.n_trials - 1:
            self.log.success = self.success
            utils.print_score(self.log, time.time() - self.time)
            self.log.save_log()
            self.stop()
        else:
            # Sample target
            self.sample_target()

            if c.task == 'reach':
                self.agent.mu_ext[0] = utils.normalize(
                    self.target_pos, c.norm_cart)

            self.step = 0
            self.trial += 1
