import numpy as np
import pyglet
from pyglet.shapes import Circle, Rectangle, Line
import utils
import config as c
from environment.eyes import Eyes


# Define window class
class Window(pyglet.window.Window):
    def __init__(self):
        super().__init__(c.width, c.height, 'Active vision',  vsync=False)
        # Initialize arm
        self.eyes = Eyes()

        # Initialize target and obstacle
        self.target_pos = np.zeros(c.n_dim)
        self.target_dir = np.zeros(c.n_dim)

        # Initialize agent
        self.agent = None

        # Initialize simulation variables
        self.step, self.trial, self.success = 0, 0, 0

        self.keys = set()
        self.batch = pyglet.graphics.Batch()
        self.fps_display = pyglet.window.FPSDisplay(self)
        self.offset_main = np.array([50, 3 * c.height / 4])
        self.offset_eyes = [np.array([c.width / 4, c.height / 4]),
                            np.array([c.width * 3 / 4, c.height / 4])]

        # Set background
        pyglet.gl.glClearColor(1, 1, 1, 1)

    def on_key_press(self, sym, mod):
        self.keys.add(sym)

    def on_key_release(self, sym, mod):
        self.keys.remove(sym)

    def on_draw(self):
        self.clear()
        objects = self.draw_screen()
        self.batch.draw()
        self.fps_display.draw()

    # Update function to override
    def update(self, dt):
        pass

    # Run simulation with custom update function
    def run(self):
        if c.fps == 0:
            pyglet.clock.schedule(self.update)
        else:
            pyglet.clock.schedule_interval(self.update, 1 / c.fps)
        pyglet.app.run()

    # Stop simulation
    def stop(self):
        pyglet.app.exit()
        self.close()

    # Draw screen
    def draw_screen(self):
        objects = set()

        objects = self.draw_lines(objects)
        objects = self.draw_main(objects)

        objects = self.draw_eye(objects, 0)
        objects = self.draw_eye(objects, 1)

        return objects

    # Draw frame lines
    def draw_lines(self, objects):
        objects.add(Line(0, c.height / 2, c.width, c.height / 2,
                         width=8, color=(0, 0, 0), batch=self.batch))
        objects.add(Line(c.width / 2, 0, c.width / 2, c.height / 2,
                         width=8, color=(0, 0, 0), batch=self.batch))

        objects.add(Line(0, self.offset_main[1], c.width, self.offset_main[1],
                         width=1, color=(0, 0, 0), batch=self.batch))
        objects.add(Line(self.offset_main[0], c.height / 2,
                         self.offset_main[0], c.height,
                         width=1, color=(0, 0, 0), batch=self.batch))

        objects.add(Line(0, c.height / 4, c.width, c.height / 4,
                         width=1, color=(0, 0, 0), batch=self.batch))
        objects.add(Line(c.width / 4, 0, c.width / 4, c.height / 2,
                         width=1, color=(0, 0, 0), batch=self.batch))
        objects.add(Line(c.width * 3 / 4, 0, c.width * 3 / 4, c.height / 2,
                         width=1, color=(0, 0, 0), batch=self.batch))

        return objects

    # Draw main screen
    def draw_main(self, objects):
        # Draw targets
        target_w = self.target_pos + self.offset_main
        objects.add(Circle(*target_w, 10, segments=20,
                           color=(200, 50, 0), batch=self.batch))

        try:
            est_target_w = utils.denormalize(
                self.agent.mu_ext[0], c.norm_cart) + self.offset_main
            if c.height / 2 < est_target_w[1] < c.height:
                objects.add(Circle(*est_target_w, 10, segments=20,
                                   color=(200, 100, 0), batch=self.batch))
        except AttributeError:
            pass

        # Draw eyes
        for e in range(2):
            eye_w = np.array([0, self.eyes.lengths[e]]) + self.offset_main
            objects.add(Circle(*eye_w, 15, segments=20,
                               color=(0, 100, 200), batch=self.batch))
            angle = Rectangle(*eye_w, 2, c.height,
                              color=(0, 100, 200), batch=self.batch)

            # Sum vergence-accomodation for absolute angles
            angle.rotation = -self.eyes.angles[0] + 90
            angle.rotation += self.eyes.angles[1] if e == 0 \
                else -self.eyes.angles[1]

            objects.add(angle)

        return objects

    # Draw eye screen
    def draw_eye(self, objects, i):
        cam = utils.denormalize(self.get_visual_obs(), c.norm_cart)
        cam_w = [0, cam[i] * c.scale] + self.offset_eyes[i]
        if 0 < cam_w[1] < c.height / 2:
            objects.add(Circle(*cam_w, 10, segments=20,
                               color=(200, 50, 0), batch=self.batch))

        try:
            est_cam = utils.denormalize(self.agent.mu_cam[0], c.norm_cart)
            est_cam_w = [0, est_cam[i] * c.scale] + self.offset_eyes[i]
            if 0 < est_cam_w[1] < c.height / 2:
                objects.add(Circle(*est_cam_w, 10, segments=20,
                                   color=(200, 100, 0), batch=self.batch))
        except AttributeError:
            pass

        return objects

    # Get proprioceptive observation
    def get_prop_obs(self):
        return utils.normalize(self.eyes.angles, c.norm_polar)

    # Get visual observation
    def get_visual_obs(self):
        rel_points = self.eyes.get_rel(self.target_pos)
        cam_points = self.eyes.get_cam(rel_points)

        #  Use nonuniform fovea resolution
        cam_noise = utils.add_gaussian_noise(
            cam_points, np.exp(np.abs(cam_points) / 1.5) * c.w_vis)

        return utils.normalize(cam_noise, c.norm_cart)

    # Generate target
    def sample_target(self):
        if c.target_pos == [0, 0]:
            angle_max = c.norm_polar[1] // 2
            radius_min = c.focal * 5

            angle = np.random.rand() * 2 * angle_max - angle_max
            radius = np.random.rand() * (c.norm_cart[1] - radius_min) \
                + radius_min

            x = radius * np.cos(np.radians(angle))
            y = radius * np.sin(np.radians(angle))

            self.target_pos = np.array([x, y])

            # Fixate target
            # self.eyes.set_rotation([angle, np.degrees(np.arctan2(
            #     c.eye_lengths[0], x))])

        else:
            self.target_pos = np.array(c.target_pos)

        # Sample velocity
        angle = np.random.rand() * 2 * np.pi
        self.target_dir = np.array((np.cos(angle), np.sin(angle)))

    # Move target
    def move_target(self):
        self.target_pos += c.target_vel * self.target_dir

        # Bounce
        if not c.focal * 5 < self.target_pos[0] < c.width:
            self.target_dir = -self.target_dir
        if not -c.height / 4 < self.target_pos[1] < c.height / 4:
            self.target_dir = -self.target_dir

    # Check if trial is successful
    def task_done(self, mu_abs, mu_cam):
        mu_ext_denorm = utils.denormalize(mu_abs, c.norm_cart)
        mu_cam_denorm = utils.denormalize(mu_cam, c.norm_cart)

        dist_infer = np.linalg.norm(self.target_pos - mu_ext_denorm)
        dist_reach = np.linalg.norm(np.array([0, 0]) - mu_cam_denorm)

        if c.task == 'infer':
            return dist_infer < c.reach_dist
        elif c.task == 'reach':
            return dist_reach < c.reach_dist * 2
        return dist_infer < c.reach_dist and dist_reach < c.reach_dist * 2
