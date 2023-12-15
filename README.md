# Active vision

<p align="center">
  <img src="/reference/images/env.png">
</p>

This is the project related to the paper [Active vision in binocular depth estimation: a top-down perspective](https://www.mdpi.com/2313-7673/8/5/445). It describes a hierarchical active inference model with two parallel pathways corresponding to the agent's eyes. Each pathway performs a roto-translation of a target encoded in absolute coordinates, followed by a perspective projection. The latter is compared with the visual observation of the point in the projective plane of the eye. The proposed model can estimate the target depth through inference. Concurrently, by imposing attractors in the projective planes, an additional vergence-accommodation belief can fixate the target.

Video simulations are found [here](https://priorelli.github.io/projects/3_active_vision/).

Check [this](https://priorelli.github.io/blog/) and [this](https://priorelli.github.io/projects/) for additional guides and projects.

## HowTo

### Start the simulation

The simulation can be launched through *main.py*, either with the option `-m` for manual control, `-i` for depth estimation, `-r` for target fixation, `-b` for simultaneous estimation and fixation, or `-a` for choosing the parameters from the console. If no option is specified, the last one will be launched. For the manual control simulation, the eyes can be rotated with the keys `Z`, `X`, `LEFT`, `RIGHT`.

Plots can be generated through *plot.py*, either with the option `-d` for the belief trajectories, or `-v` for generating a video of the simulation.

The folder *reference/video/* contains a few videos about depth estimation and target fixation in both static and dynamic environments.

The window is divided into three frames. The top frame shows the task from above, while the two bottom frames displays the visual observations from the perspective of the eyes. The red and orange circles correspond to the real and estimated target.

### Advanced configuration

More advanced parameters can be manually set from *config.py*. Custom log names are set with the variable `log_name`. The number of trials, steps, and cycles can be set with the variables `n_trials`, `n_steps`, and `n_cycle` respectively.

The parameters `gain_abs`, `gain_theta`, and `gain_cam` control the magnitude of the update step of the beliefs.

The parameter `w_vis` controls the magnitude of the Gaussian error of the visual observations.

The variable `task` affects the goal of the active inference agent, and can assume the following values:
1. `infer`: the agent has to infer the depth of the target without rotating the eyes;
2. `reach`: the agent has to fixate on the target. Note that in this case the belief of the point in absolute coordinates is set to the correct value and kept fixed throughout the trial;
3. `both`: the agent has to simultaneously infer and fixate on the target. In this case, the absolute belief is free to change.

The variable `context` specifies whether (`dynamic`) or not (`static`) the target moves. The velocity is set by `target_vel`.

The target position can be manually set through the variable `target_pos`. If it is set to (0, 0), it will be randomly sampled at each trial.

The parameters of the eyes, i.e., the angles, distance from the origin, and focal plane can be manually set through the variables `eye_angles`, `eye_lengths`, and `focal`.

### Agent

The script *simulation/inference.py* contains a subclass of `Window` in *environment/window.py*, which is in turn a subclass `pyglet.window.Window`. The only overriden function is `update`, which defines the instructions to run in a single cycle. Specifically, the subclass `Inference` initializes the agent and the target; during each update, it retrieves proprioceptive and visual observations through functions defined in *environment/window.py*, calls the function `inference_step` of the agent, and finally moves the eyes and the target.

The function `inference_step` of the class `Agent` in *simulation/agent.py* executes a single update step. In particular, the function `get_p` returns visual, projective, and proprioceptive predictions. Note that the predictions of the points in the reference frames relative to the eyes are computed implicitly through the function `get_rel`. The function `get_i` returns future beliefs depending on the agent's intentions, e.g., reach a point in absolute coordinates, a point projected in the eye planes, or rotate by a specific angle. Functions `get_e_g` and `get_e_mu` compute sensory and dynamics prediction errors, respectively. The function `get_likelihood` backpropagates the sensory errors toward the beliefs, calling the function `grad_cam` to compute the gradients of the absolute belief and the vergence-accommodation angles. Note that this function could infer the belief over the eye lengths `mu_len` through the variable `grad_len`. Note also that the absolute belief is affected by the likelihoods of both eyes, and letting it depend only on a single contribution is the equivalent of closing an eye. Finally, the function `mu_dot` computes the total belief updates, also considering the backward and forward errors `E_mu` of the dynamics functions.

The function `set_mode` is used to control the action-perception cycles for simultaneous inference and fixation.

Useful trajectories computed during the simulations are stored through the class `Log` in *environment/log.py*.

Note that all the variables are normalized between -1 and 1 to ensure that every contribution to the belief updates has the same magnitude.

## Required libraries

matplotlib==3.8.1

numpy==1.26.2

pyglet==2.0.5

seaborn==0.13.0
