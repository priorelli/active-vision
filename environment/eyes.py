import numpy as np
import config as c


class Eyes:
    def __init__(self):
        self.angles = np.array(c.eye_angles)
        self.vel = np.zeros_like(self.angles)

        self.lengths = np.array(c.eye_lengths)

    # Get eye frame of reference
    def get_eye(self, point, angles=None, lengths=None):
        if angles is None:
            angles = self.angles
        if lengths is None:
            lengths = self.lengths

        point_hom = np.array([*point, 1])
        eye_points = np.zeros((c.n_eyes, 2))

        for i in range(c.n_eyes):
            # Vergence-accomodation
            angle = angles[0]
            angle -= angles[1] if i == 0 else -angles[1]

            cos = np.cos(np.radians(angle))
            sin = np.sin(np.radians(angle))

            rel = np.array([[cos, sin, -sin * lengths[i]],
                            [-sin, cos, -cos * lengths[i]],
                            [0, 0, 1]])

            eye_points[i] = rel.dot(point_hom)[:2]

        return eye_points

    # Get camera projection
    def get_cam(self, points, focal=c.focal):
        point_hom = np.c_[points, np.ones(c.n_eyes)]

        cam_points = np.zeros(c.n_eyes)
        for i in range(c.n_eyes):
            cam = np.array([[1, 0, 0],
                            [0, focal, 0]])

            cam_points[i] = cam.dot(point_hom[i])[1:]
            cam_points[i] /= point_hom[i, 0]

        return cam_points

    # Update eyes with velocity
    def update(self, vel):
        self.vel = np.array(vel)
        self.angles = self.angles + c.dt * self.vel
        self.angles[0] = np.clip(self.angles[0], *c.norm_polar)
        self.angles[1] = np.clip(self.angles[1], 0, c.norm_polar[1] / 2)

    # Set eyes rotation in absolute angles
    def set_rotation(self, angles):
        self.angles = np.array(angles)
