import numpy as np
import config as c
import utils


# Define log class
class Log:
    def __init__(self):
        # Initialize logs
        self.actions = np.zeros((c.n_trials, c.n_steps, c.n_eyes))

        self.angles = np.zeros((c.n_trials, c.n_steps, 2))
        self.est_angles = np.zeros_like(self.angles)

        self.lengths = np.zeros((c.n_trials, c.n_steps, c.n_eyes))
        self.est_lengths = np.zeros_like(self.lengths)

        self.pos = np.zeros((c.n_trials, c.n_steps, c.n_dim))
        self.est_pos = np.zeros_like(self.pos)

        self.cam = np.zeros((c.n_trials, c.n_steps, c.n_eyes))
        self.est_cam = np.zeros_like(self.cam)

        self.int_point = np.zeros((c.n_trials, c.n_steps, c.n_dim))

        self.success = np.zeros(c.n_trials)

    # Track logs for each time step
    def track(self, step, trial, agent, eyes, target_pos, s_vis):
        self.actions[trial, step] = agent.a

        self.angles[trial, step] = eyes.angles
        est_angles = utils.denormalize(agent.mu_theta[0], c.norm_polar)
        self.est_angles[trial, step] = est_angles

        self.lengths[trial, step] = eyes.lengths
        est_lengths = utils.denormalize(agent.mu_len, c.norm_cart)
        self.est_lengths[trial, step] = est_lengths

        self.pos[trial, step] = target_pos
        est_pos = utils.denormalize(agent.mu_abs[0], c.norm_cart)
        self.est_pos[trial, step] = est_pos

        self.cam[trial, step] = utils.denormalize(s_vis, c.norm_cart)
        est_cam = utils.denormalize(agent.mu_cam[0], c.norm_cart)
        self.est_cam[trial, step] = est_cam

        b1, b2 = eyes.lengths
        m1 = np.radians(eyes.angles[0] - eyes.angles[1])
        m2 = np.radians(eyes.angles[0] + eyes.angles[1])
        if m2 - m1 != 0:
            x_int = (b1 - b2) / (m2 - m1)
            y_int = (b1 * m2 - m1 * b2) / (m2 - m1)
            self.int_point[trial, step] = x_int, y_int
        else:
            self.int_point[trial, step] = -100, 0

    # Save log to file
    def save_log(self):
        np.savez_compressed('simulation/log_' + c.log_name,
                            actions=self.actions,
                            angles=self.angles, est_angles=self.est_angles,
                            lengths=self.lengths, est_lengths=self.est_lengths,
                            pos=self.pos, est_pos=self.est_pos,
                            cam=self.cam, est_cam=self.est_cam,
                            int_point=self.int_point, success=self.success)
