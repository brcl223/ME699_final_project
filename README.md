# ME699 Robotic Control Final Project

Authors:
* Benton Clark
* Ethan Howell
* Brian Moberly


### Running the Code
To run the code, move into the src directory and run

```sh
julia --color=yes demonstration.jl
```

It may take a few minutes to get up in running, but should list a progress report as each phase of the demonstration is worked through. Once completed, a window should pop-up, and within a few seconds the demonstration will begin. The demonstration includes three phases:

* Moving from home to intial state of object
* Moving the object from the initial state to the goal state
* Moving from the goal state back to the home state

During each phase, the following show up in the demonstration:

* The green line represents the planned trajectory for the end effector
* The red line represents the actual trajectory followed for the end effector when the control was applied
* The blue point cloud dots represent the noisy end effector configurations

If successful, the arm should move towards the object, move it from the start platform to the end platform, and finally return home. Several collision obstacles are present in the simulation, including the platforms the object is located on. The arm should (approximately) avoid these objects while moving the mass from the start to the goal state.

It should be noted that the mass of the object is unknown, and an adaptive PD controller is used to move the mass from intial to goal state.

### Changing Variables

The mass, initial state and goal state for the ball are all controllable variables in this simulation. If so desired, the mass of the object can be changed by editing the `src/robot_heavy.urdf` file. Specifically, the mass and moment of inertia values can be tweaked for the last link.

To change the initial and goal state for the object, edit the x,y and/or z coordinates on line 110 and 116 of `src/demonstration.jl` for the starting platform positions. The ball is automatically placed on top of the start platform, and the end platform position will be calculated when planning the trajectory from initial state to goal state.

### Code

The three primary targets for our project were the mass estimated PD controller, the adaptive PD controller, and the Kalman filter. The following files contain the code related to these tasks:

* `src/utils/controllers.jl`
* `src/utils/kalman.jl`

The remaining files located in the `src/utils` folder pertain to various ancillary tasks needed to complete the final demonstration. Each file is written according to the function it serves in the project.

For training the mass neural network, the following files are present:

* `src/train_mass_nn.jl`
* `src/sample_mass.jl`

The model itself is output to `src/models/mass_nn.bson` where it can be loaded with the BSON Julia package. A demonstration of testing the neural network can be found in `src/test_mass_nn.jl`. Finally, the data samples collected are placed in `src/data/gravity_points.jld`, where the JLD Julia package can be used to load/save the data points.
