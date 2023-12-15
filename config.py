# Window
width = 600
height = 600
scale = 10
fps = 0
debug = 0

# Agent
dt = 0.7
gain_a = 2.0
gain_abs = 10.0
gain_theta = 1.0
gain_cam = 1.0

pi_cam = 1.2
pi_vis = 1.0
pi_prop = 1.0

w_vis = 1.0  # 0.0

lambda_abs = 0.0
lambda_theta = 0.0
lambda_cam = 1.5

# Inference
task = 'both'  # reach, infer, both
context = 'dynamic'  # static, dynamic
log_name = ''

target_vel = 0.01
reach_dist = 5.0

n_steps = 10000
n_cycle = 100
n_trials = 1
n_orders = 2

# Eyes
target_pos = [0, 0]

eye_angles = [0, 0]  # [10, 10]
eye_lengths = [40.0, -40.0]
focal = 10.0

n_eyes = len(eye_angles)
n_dim = len(target_pos)

norm_polar = [-50.0, 50.0]
norm_cart = [-250, 250]
