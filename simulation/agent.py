import numpy as np
import utils
import config as c


class Agent:
    def __init__(self, eyes):
        self.get_eye = eyes.get_eye
        self.get_cam = eyes.get_cam
        self.focal_norm = utils.normalize(c.focal, c.norm_cart)

        # Initialize beliefs and action
        self.mu_ext = np.zeros((c.n_orders, c.n_dim))
        self.mu_int = np.zeros((c.n_orders, 2))
        self.mu_cam = np.zeros((c.n_orders, c.n_eyes))
        self.mu_len = np.zeros(c.n_eyes)

        self.a = np.zeros(c.n_eyes)

        self.gain_a = c.gain_a
        self.gain_ext = c.gain_ext
        self.k_cam = c.k_cam
        self.go = 1.0

    # Initialize belief
    def init_belief(self):
        self.mu_ext[0, 0] = self.focal_norm * 5
        self.mu_int[0] = utils.normalize(c.eye_angles, c.norm_polar)
        self.mu_len = utils.normalize(c.eye_lengths, c.norm_cart)

        mu_eye = self.get_eye(self.mu_ext[0], lengths=self.mu_len)
        self.mu_cam[0] = self.get_cam(mu_eye, self.focal_norm)

    # Get predictions
    def get_p(self):
        p_cam = self.g_cam()
        p_vis = self.mu_cam[0].copy()
        p_prop = self.mu_int[0].copy()

        return p_cam, p_vis, p_prop

    # Get camera points prediction
    def g_cam(self):
        angles_denorm = utils.denormalize(self.mu_int[0], c.norm_polar)

        p_eye = self.get_eye(self.mu_ext[0], angles_denorm, self.mu_len)

        return self.get_cam(p_eye, self.focal_norm)

    # Get generative prediction errors
    def get_e_g(self, S, P):
        obs = [self.mu_cam[0], *S]
        Pi = [c.pi_cam, c.pi_vis, c.pi_prop]

        E_g = [(s - p) * pi for s, p, pi in zip(obs, P, Pi)]

        return E_g

    # Get intentions
    def get_i(self):
        i_ext = self.mu_ext[0].copy()
        # angles_denorm = utils.denormalize(self.mu_int[0], c.norm_polar)
        # tan = np.tan(np.radians(angles_denorm))
        # dist = self.mu_len[1] - self.mu_len[0]
        # if tan[1] - tan[0] != 0:
        #     i_ext[0] = dist / (tan[1] - tan[0])

        i_int = utils.normalize([30, 10], c.norm_polar)

        i_cam = np.array([0, 0])

        return i_ext, i_int, i_cam

    # Get dynamics prediction errors
    def get_e_mu(self, I):
        E_i = (I[0] - self.mu_ext[0]) * c.k_ext, \
            (I[1] - self.mu_int[0]) * c.k_int, \
            (I[2] - self.mu_cam[0]) * self.k_cam

        return (self.mu_ext[1] - E_i[0], self.mu_int[1] - E_i[1],
                self.mu_cam[1] - E_i[2])

    # Get likelihood components
    def get_likelihood(self, E_g):
        lkh = {}

        lkh['ext'], lkh['int'] = self.grad_cam(E_g[0])
        lkh['cam'] = E_g[1].copy()
        lkh['prop'] = E_g[2].copy()

        lkh['forward_cam'] = E_g[0].copy()

        return lkh

    # Get gradient of camera image
    def grad_cam(self, e_cam):
        lkh_eye = np.zeros((c.n_eyes, c.n_dim))

        angles_denorm = utils.denormalize(self.mu_int[0], c.norm_polar)
        p_eye = self.get_eye(self.mu_ext[0], angles_denorm, self.mu_len)

        for i in range(c.n_eyes):
            px, py = p_eye[i]

            grad_eye = np.array([-(self.focal_norm * py) / (px ** 2),
                                 self.focal_norm / px])

            lkh_eye[i] = grad_eye.dot(e_cam[i])

        lkh_ext = np.zeros(c.n_dim)
        lkh_int = np.zeros(2)

        for i in range(c.n_eyes):
            angle = angles_denorm[0]
            angle -= angles_denorm[1] if i == 0 else -angles_denorm[1]

            cos = np.cos(np.radians(angle))
            sin = np.sin(np.radians(angle))

            length = self.mu_len[i]
            wx, wy = self.mu_ext[0]

            grad_ext = np.array([[cos, -sin],
                                 [sin, cos]])

            grad_int = np.array([cos * (wy - length) - sin * wx,
                                 -sin * (wy - length) - cos * wx])

            # grad_len = length [-sin, -cos]

            # Gradient of normalization
            grad_int *= np.pi / 180
            grad_int *= (c.norm_polar[1] - c.norm_polar[0])

            # if i == 0:  # CLOSE ONE EYE
            lkh_ext += grad_ext.dot(lkh_eye[i])
            lkh_int[0] += grad_int.dot(lkh_eye[i])
            lkh_int[1] -= grad_int.dot(lkh_eye[i]) if i == 0 else \
                -grad_int.dot(lkh_eye[i])

        return lkh_ext, lkh_int

    # Get belief update
    def get_mu_dot(self, lkh, E_mu):
        mu_ext_dot = np.zeros_like(self.mu_ext)
        mu_int_dot = np.zeros_like(self.mu_int)
        mu_cam_dot = np.zeros_like(self.mu_cam)

        mu_ext_dot[0] = self.mu_ext[1] + lkh['ext']
        mu_int_dot[0] = self.mu_int[1] + lkh['int'] + lkh['prop']
        mu_cam_dot[0] = self.mu_cam[1] + lkh['cam'] - lkh['forward_cam']

        if c.task in ['reach', 'both']:
            mu_ext_dot[1] -= E_mu[0]
            mu_int_dot[1] -= E_mu[1]
            mu_cam_dot[1] -= E_mu[2]

        return mu_ext_dot, mu_int_dot, mu_cam_dot

    # Get action update
    def get_a_dot(self, e_prop):
        return -c.dt * e_prop

    # Integrate with gradient descent
    def integrate(self, mu_ext_dot, mu_int_dot, mu_cam_dot, a_dot):
        # Update belief
        self.mu_ext[0] += c.dt * mu_ext_dot[0] * self.gain_ext
        self.mu_ext[1] += c.dt * mu_ext_dot[1] * self.gain_ext
        self.mu_ext[0] = np.clip(self.mu_ext[0], (self.focal_norm * 2, -1), 1)

        self.mu_int[0] += c.dt * mu_int_dot[0] * c.gain_int
        self.mu_int[1] += c.dt * mu_int_dot[1] * c.gain_int
        self.mu_int[0] = np.clip(self.mu_int[0], (-1, 0), (1, 0.5))

        self.mu_cam[0] += c.dt * mu_cam_dot[0] * c.gain_cam
        self.mu_cam[1] += c.dt * mu_cam_dot[1] * c.gain_cam
        self.mu_cam[0] = np.clip(self.mu_cam[0], -1, 1)

        # Update action
        self.a += c.dt * a_dot * self.gain_a

    def set_mode(self, step):
        if c.task == 'reach':
            self.gain_ext = 0.0

        elif c.task == 'infer':
            self.gain_a = 0.0

        elif c.task == 'both':
            if int(step / c.n_cycle) % 2 == 0:
                self.k_cam = 0.0
                # c.pi_vis = 1.0

                self.gain_ext = c.gain_ext
                # c.gain_int = 0.0

                self.gain_a = 0.0
                # self.go = 0.0
            else:
                self.k_cam = c.k_cam
                # c.pi_vis = 0.0

                self.gain_ext = 0.0
                # c.gain_int = 1.0

                self.gain_a = c.gain_a
                # self.go = 1.0

    # Run an inference step
    def inference_step(self, S, step):
        self.set_mode(step)

        # Get predictions
        P = self.get_p()

        # Get generative prediction errors
        E_g = self.get_e_g(S, P)

        # Get intentions
        I = self.get_i()

        # Get intention prediction errors
        E_mu = self.get_e_mu(I)

        # Get likelihood components
        likelihood = self.get_likelihood(E_g)

        # Get belief update
        mu_dot = self.get_mu_dot(likelihood, E_mu)

        # Get action update
        a_dot = self.get_a_dot(E_g[2])

        # Print debug info
        utils.print_debug(step, (self.mu_ext[0], self.mu_int,
                          self.mu_cam), likelihood, E_g, E_mu)

        # Update
        self.integrate(*mu_dot, a_dot)

        return utils.denormalize(self.a, c.norm_polar) * self.go
