# Binocular depth perception in Active Inference

<p align="center">
  <img src="/reference/images/env.png">
</p>

<p align="justify">
Depth estimation is an ill-posed problem: objects of different shapes or dimensions may project to the same 2D image on the retina. Our brain uses several cues for depth perception, including monocular cues such as motion parallax and binocular cues such as binocular disparity. However, it is still unclear how the computations required for depth perception might be implemented in a biologically plausible way. State-of-the-art approaches to depth estimation in machine learning use biologically unrealistic deep neural networks. Here, instead, we propose a biologically plausible approach to depth estimation under the active inference framework. We show that depth estimation can be performed by inverting a generative model that predicts the 2D projection of the eyes from a 3D belief over an object. The generative model performs a series of affine transformations and it computes beliefs about depth by averaging the prediction errors coming from each projection. In turn, depth perception becomes more precise after fixating the object with the eyes - and this kind of oculomotor control can be realized in active inference by defining simple attractors into the model dynamics. The proposed approach requires only local (top-down and bottom-up) message passing that can be implemented in biologically plausible neural circuits.
</p>

**Reference Paper**: 

## How To

<p align="justify">
The simulation can be launched through <b>main.py</b>. either with the option "-m" for manual control, "-i" for only estimating the depth of the target with fixed eye angles, "-r" for fixating the target with the belief depth set to the correct value, or "-b" for both depth estimation and target fixation. If no option is specified, the later simulation will be launched. For the manual control simulation, the keys Z and X can be used for accomodation, while LEFT and RIGHT for vergence.
</p>

More advanced parameters can be manually set from **config.py**.

<p align="justify">
Plots can be generated through <b>plot.py</b>, either with the option "-d" for the dynamics, or "-v" for generating a video of the simulation.
</p>

