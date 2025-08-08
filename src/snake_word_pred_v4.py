
from typing import Dict, Union
import numpy as np

from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.spaces import Box
import os
import json
import numpy as np
from sklearn.feature_selection import mutual_info_regression
from jax import lax

import jax
import jax.numpy as jnp

@jax.jit
def mutual_info_histogram(x, y, n_bins=8):
    N_BINS = 1000
    x = x.flatten()
    y = y.flatten()
    x_min, x_max = jnp.min(x), jnp.max(x)
    y_min, y_max = jnp.min(y), jnp.max(y)
    x_min = jnp.where(x_max == x_min, x_min - 0.5, x_min)
    x_max = jnp.where(x_max == x_min, x_max + 0.5, x_max)
    y_min = jnp.where(y_max == y_min, y_min - 0.5, y_min)
    y_max = jnp.where(y_max == y_min, y_max + 0.5, y_max)
    x_bin = jnp.floor((x - x_min) / (x_max - x_min) * n_bins).astype(jnp.int32)
    y_bin = jnp.floor((y - y_min) / (y_max - y_min) * n_bins).astype(jnp.int32)
    x_bin = jnp.clip(x_bin, 0, n_bins - 1)
    y_bin = jnp.clip(y_bin, 0, n_bins - 1)
    joint = x_bin * n_bins + y_bin
    counts = jnp.bincount(joint,length=N_BINS*N_BINS)
    counts = counts.reshape((N_BINS,N_BINS))
    p_xy = counts / x.shape[0]
    p_x = jnp.sum(p_xy, axis=1, keepdims=True)
    p_y = jnp.sum(p_xy, axis=0, keepdims=True)
    mask = (p_xy > 0)
    mi = jnp.sum(jnp.where(mask, 
                           p_xy * (jnp.log(p_xy) - jnp.log(p_x) - jnp.log(p_y)), 
                           0.0))
    return mi

@jax.jit
def computePredInfo(raw_data, maxT=100, bins=100):
    data = jnp.stack(raw_data, axis=0)
    T,D = data.shape
    maxT = min(maxT, T - 1)
    
    data_past = lax.dynamic_slice(operand=data,start_indices= (0, 0), slice_sizes=(maxT, D))       # shape [maxT, D]
    data_future = lax.dynamic_slice(data, (1, 0), (maxT, D))

    def mi_for_dim(i):
        return mutual_info_histogram(data_past[:, i], data_future[:, i], n_bins=bins)

    mi_vals = jax.vmap(mi_for_dim)(jnp.arange(D))
    return jnp.mean(mi_vals,dtype=np.float32)

class snakeEnvPred(MujocoEnv, utils.EzPickle):
    r"""
    ## Description
    This environment corresponds to the Swimmer environment described in Rémi Coulom's PhD thesis ["Reinforcement Learning Using Neural Networks, with Applications to Motor Control"](https://tel.archives-ouvertes.fr/tel-00003985/document).
    The environment aims to increase the number of independent state and control variables compared to classical control environments.
    The swimmers consist of three or more segments ('***links***') and one less articulation joints ('***rotors***') - one rotor joint connects exactly two links to form a linear chain.
    The swimmer is suspended in a two-dimensional pool and always starts in the same position (subject to some deviation drawn from a uniform distribution),
    and the goal is to move as fast as possible towards the right by applying torque to the rotors and using fluid friction.

    ## Notes

    The problem parameters are:
    Problem parameters:
    * *n*: number of body parts
    * *m<sub>i</sub>*: mass of part *i* (*i* ∈ {1...n})
    * *l<sub>i</sub>*: length of part *i* (*i* ∈ {1...n})
    * *k*: viscous-friction coefficient

    While the default environment has *n* = 3, *l<sub>i</sub>* = 0.1, and *k* = 0.1.
    It is possible to pass a custom MuJoCo XML file during construction to increase the number of links, or to tweak any of the parameters.


    ## Action Space
    ```{figure} action_space_figures/swimmer.png
    :name: swimmer
    ```

    The action space is a `Box(-1, 1, (2,), float32)`. An action represents the torques applied between *links*

    | Num | Action                             | Control Min | Control Max | Name (in corresponding XML file) | Joint | Type (Unit)  |
    |-----|------------------------------------|-------------|-------------|----------------------------------|-------|--------------|
    | 0   | Torque applied on the first rotor  | -1          | 1           | motor1_rot                       | hinge | torque (N m) |
    | 1   | Torque applied on the second rotor | -1          | 1           | motor2_rot                       | hinge | torque (N m) |


    ## Observation Space
    The observation space consists of the following parts (in order):

    - *qpos (3 elements by default):* Position values of the robot's body parts.
    - *qvel (5 elements):* The velocities of these individual body parts (their derivatives).

    By default, the observation does not include the x- and y-coordinates of the front tip.
    These can be included by passing `exclude_current_positions_from_observation=False` during construction.
    In this case, the observation space will be a `Box(-Inf, Inf, (10,), float64)`, where the first two observations are the x- and y-coordinates of the front tip.
    Regardless of whether `exclude_current_positions_from_observation` is set to `True` or `False`, the x- and y-coordinates are returned in `info` with the keys `"x_position"` and `"y_position"`, respectively.

    By default, however, the observation space is a `Box(-Inf, Inf, (8,), float64)` where the elements are as follows:

    | Num | Observation                          | Min  | Max | Name (in corresponding XML file) | Joint | Type (Unit)              |
    | --- | ------------------------------------ | ---- | --- | -------------------------------- | ----- | ------------------------ |
    | 0   | angle of the front tip               | -Inf | Inf | free_body_rot                    | hinge | angle (rad)              |
    | 1   | angle of the first rotor             | -Inf | Inf | motor1_rot                       | hinge | angle (rad)              |
    | 2   | angle of the second rotor            | -Inf | Inf | motor2_rot                       | hinge | angle (rad)              |
    | 3   | velocity of the tip along the x-axis | -Inf | Inf | slider1                          | slide | velocity (m/s)           |
    | 4   | velocity of the tip along the y-axis | -Inf | Inf | slider2                          | slide | velocity (m/s)           |
    | 5   | angular velocity of front tip        | -Inf | Inf | free_body_rot                    | hinge | angular velocity (rad/s) |
    | 6   | angular velocity of first rotor      | -Inf | Inf | motor1_rot                       | hinge | angular velocity (rad/s) |
    | 7   | angular velocity of second rotor     | -Inf | Inf | motor2_rot                       | hinge | angular velocity (rad/s) |
    | excluded | position of the tip along the x-axis | -Inf | Inf | slider1                          | slide | position (m)           |
    | excluded | position of the tip along the y-axis | -Inf | Inf | slider2                          | slide | position (m)           |


    ## Rewards
    The total reward is: ***reward*** *=* *forward_reward - ctrl_cost*.

    - *forward_reward*:
    A reward for moving forward,
    this reward would be positive if the Swimmer moves forward (in the positive $x$ direction / in the right direction).
    $w_{forward} \times \frac{dx}{dt}$, where
    $dx$ is the displacement of the (front) "tip" ($x_{after-action} - x_{before-action}$),
    $dt$ is the time between actions, which depends on the `frame_skip` parameter (default is 4),
    and `frametime` which is $0.01$ - so the default is $dt = 4 \times 0.01 = 0.04$,
    $w_{forward}$ is the `forward_reward_weight` (default is $1$).
    - *ctrl_cost*:
    A negative reward to penalize the Swimmer for taking actions that are too large.
    $w_{control} \times \|action\|_2^2$,
    where $w_{control}$ is `ctrl_cost_weight` (default is $10^{-4}$).

    `info` contains the individual reward terms.


    ## Starting State
    The initial position state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{5}, reset\_noise\_scale \times I_{5}]}$.
    The initial velocity state is $\mathcal{U}_{[-reset\_noise\_scale \times I_{5}, reset\_noise\_scale \times I_{5}]}$.

    where $\mathcal{U}$ is the multivariate uniform continuous distribution.


    ## Episode End
    ### Termination
    The Swimmer never terminates.

    ### Truncation
    The default duration of an episode is 1000 timesteps.


    ## Arguments
    Swimmer provides a range of parameters to modify the observation space, reward function, initial state, and termination condition.
    These parameters can be applied during `gymnasium.make` in the following way:

    ```python
    import gymnasium as gym
    env = gym.make('Swimmer-v5', xml_file=...)
    ```

    | Parameter                                  | Type      | Default       |Description                                                                                                                                                                                                  |
    |--------------------------------------------| --------- |-------------- |-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
    |`xml_file`                                  | **str**   |`"swimmer.xml"`| Path to a MuJoCo model                                                                                                                                                                                      |
    |`forward_reward_weight`                     | **float** | `1`           | Weight for _forward_reward_ term (see `Rewards` section)                                                                                                                                                    |
    |`ctrl_cost_weight`                          | **float** | `1e-4`        | Weight for _ctrl_cost_ term (see `Rewards` section)                                                                                                                                                         |
    |`reset_noise_scale`                         | **float** | `0.1`         | Scale of random perturbations of initial position and velocity (see `Starting State` section)                                                                                                               |
    |`exclude_current_positions_from_observation`| **bool**  | `True`        | Whether or not to omit the x- and y-coordinates from observations. Excluding the position can serve as an inductive bias to induce position-agnostic behavior in policies (see `Observation Space` section) |


    ## Version History
    * v5:
        - Minimum `mujoco` version is now 2.3.3.
        - Added support for fully custom/third party `mujoco` models using the `xml_file` argument (previously only a few changes could be made to the existing models).
        - Added `default_camera_config` argument, a dictionary for setting the `mj_camera` properties, mainly useful for custom environments.
        - Added `env.observation_structure`, a dictionary for specifying the observation space compose (e.g. `qpos`, `qvel`), useful for building tooling and wrappers for the MuJoCo environments.
        - Return a non-empty `info` with `reset()`, previously an empty dictionary was returned, the new keys are the same state information as `step()`.
        - Added `frame_skip` argument, used to configure the `dt` (duration of `step()`), default varies by environment check environment documentation pages.
        - Restored the `xml_file` argument (was removed in `v4`).
        - Added `forward_reward_weight`, `ctrl_cost_weight`, to configure the reward function (defaults are effectively the same as in `v4`).
        - Added `reset_noise_scale` argument to set the range of initial states.
        - Added `exclude_current_positions_from_observation` argument.
        - Replaced `info["reward_fwd"]` and `info["forward_reward"]` with `info["reward_forward"]` to be consistent with the other environments.
    * v4: All MuJoCo environments now use the MuJoCo bindings in mujoco >= 2.1.3.
    * v3: Support for `gymnasium.make` kwargs such as `xml_file`, `ctrl_cost_weight`, `reset_noise_scale`, etc. rgb rendering comes from tracking camera (so agent does not run away from screen).
    * v2: All continuous control environments now use mujoco-py >= 1.50.
    * v1: max_time_steps raised to 1000 for robot based tasks. Added reward_threshold to environments.
    * v0: Initial versions release.
    """

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
    }
    @staticmethod
    def computePredInfo(raw_data, maxT=100, bins=100):
        data = jnp.stack(raw_data, axis=0)
        T,D = data.shape
        maxT = min(maxT, T - 1)
        
        data_past = lax.dynamic_slice(operand=data,start_indices= (0, 0), slice_sizes=(maxT, D))       # shape [maxT, D]
        data_future = lax.dynamic_slice(data, (1, 0), (maxT, D))

        def mi_for_dim(i):
            return mutual_info_histogram(data_past[:, i], data_future[:, i], n_bins=bins)

        mi_vals = jax.vmap(mi_for_dim)(jnp.arange(D))
        return jnp.mean(mi_vals,dtype=np.float32)
    @staticmethod
    def mutual_info_histogram(x, y, n_bins=8):
        N_BINS = 1000
        x = x.flatten()
        y = y.flatten()
        x_min, x_max = jnp.min(x), jnp.max(x)
        y_min, y_max = jnp.min(y), jnp.max(y)
        x_min = jnp.where(x_max == x_min, x_min - 0.5, x_min)
        x_max = jnp.where(x_max == x_min, x_max + 0.5, x_max)
        y_min = jnp.where(y_max == y_min, y_min - 0.5, y_min)
        y_max = jnp.where(y_max == y_min, y_max + 0.5, y_max)
        x_bin = jnp.floor((x - x_min) / (x_max - x_min) * n_bins).astype(jnp.int32)
        y_bin = jnp.floor((y - y_min) / (y_max - y_min) * n_bins).astype(jnp.int32)
        x_bin = jnp.clip(x_bin, 0, n_bins - 1)
        y_bin = jnp.clip(y_bin, 0, n_bins - 1)
        joint = x_bin * n_bins + y_bin
        counts = jnp.bincount(joint,length=N_BINS*N_BINS)
        counts = counts.reshape((N_BINS,N_BINS))
        p_xy = counts / x.shape[0]
        p_x = jnp.sum(p_xy, axis=1, keepdims=True)
        p_y = jnp.sum(p_xy, axis=0, keepdims=True)
        mask = (p_xy > 0)
        mi = jnp.sum(jnp.where(mask, 
                            p_xy * (jnp.log(p_xy) - jnp.log(p_x) - jnp.log(p_y)), 
                            0.0))
        return mi


    def __init__(
        self,
        xml_file: str = "./assets/snakeMotors2_14_rough.xml",
        frame_skip: int = 10,
        default_camera_config: Dict[str, Union[float, int]] = {},
        forward_reward_weight: float = 1.0,
        com_velocity_weight: float = 3.0,
        distance_reward_weight: float = .1,
        ctrl_cost_weight: float = 1e-4,
        reset_noise_scale: float = 0.1,
        epsilon_product: float = 0.25,
        predSamples: int = 40,
        exclude_current_positions_from_observation: bool = False,
        **kwargs,
    ):
        utils.EzPickle.__init__(
            self,
            xml_file,
            frame_skip,
            default_camera_config,
            forward_reward_weight,
            ctrl_cost_weight,
            reset_noise_scale,
            exclude_current_positions_from_observation,
            **kwargs,
        )

        self._forward_reward_weight = forward_reward_weight
        self._ctrl_cost_weight = ctrl_cost_weight
        self._com_velocity_weight = com_velocity_weight
        self._distance_reward_weight = distance_reward_weight
        self._reset_noise_scale = reset_noise_scale
        self.distanceFromOrigin = 0
        self._exclude_current_positions_from_observation = (
            exclude_current_positions_from_observation
        )

        MujocoEnv.__init__(
            self,
            xml_file,
            frame_skip,
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

        obs_size = (
            self.data.qpos.size
            + self.data.qvel.size
            - 2 * exclude_current_positions_from_observation
        )
        self.observation_space = Box(
            low=-np.inf, high=np.inf, shape=(obs_size,), dtype=np.float64
        )

        self.observation_structure = {
            "skipped_qpos": 2 * exclude_current_positions_from_observation,
            "qpos": self.data.qpos.size
            - 2 * exclude_current_positions_from_observation,
            "qvel": self.data.qvel.size,
        }
        # settings_path = os.path.join(os.path.dirname(__file__), "../../../settings.json")
        # with open(settings_path, "r") as f:
        #     settings = json.load(f)

        # self.cpgH = DiscreteMatsuokaCPG(NUMBER_OF_JOINTS,**cpgSettings)
        # self.cpgV = DiscreteMatsuokaCPG(NUMBER_OF_JOINTS,**cpgSettings)
        # self.cpgHstate = np.zeros(4*NUMBER_OF_JOINTS)
        # self.cpgVstate = np.zeros(4*NUMBER_OF_JOINTS)
        self.nJoints = int((self.data.qpos.size - 7/ 2))
        self.jointHist = []

        self._predSamples = predSamples
        self._epsilon_product = epsilon_product
        # self.action_space = Box(
        #     low=-10.0, high=10.0, shape=(2*self.nJoints,), dtype=np.float32
        # )
        self.stepCounter = 0
    def calculateCenterofMass(self):
        mass = self.model.body_mass
        xpos = self.model.body_pos
        centerOfMass = np.sum([mass[i]*xpos[i] for i in range(len(mass))],axis = 0)/np.sum(mass)
        return centerOfMass

    def control_cost(self, action):
        control_cost = self._ctrl_cost_weight * np.sum(np.square(action))
        return control_cost

    target_pos = np.array([10.0, 10.0])

    def step(self, action):
        comBefore = self.calculateCenterofMass()
        xy_position_before = self.data.qpos[0:2].copy()
        # ueV,ufV = self.cpgV.tonic_input_from_action(action[:self.nJoints])
        # ueH,ufH = self.cpgH.tonic_input_from_action(action[self.nJoints:])
        # self.cpgVstate, outputV= self.cpgV.step(self.cpgHstate,self.dt,ueV,ufV)
        # self.cpgHstate,outputH = self.cpgH.step(self.cpgVstate,self.dt,ueH,ufH)
        # action = np.concatenate([outputV,outputH])
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.data.qpos[0:2].copy()

        distance_from_target_before = np.linalg.norm(xy_position_before - self.target_pos, ord=2)
        distance_from_target_after = np.linalg.norm(xy_position_after - self.target_pos, ord=2)

        velocity_to_target = (distance_from_target_before - distance_from_target_after) / self.dt

        xy_velocity = (xy_position_after - xy_position_before) / self.dt

        x_velocity, y_velocity = xy_velocity
        comAfter = self.calculateCenterofMass()
        comVelocity = (comAfter - comBefore) / self.dt
        comXVelocity,comYVelocity,comZVelocity = comVelocity
        observation = self._get_obs()

        distance_from_origin = xy_position_after[1]
        qV = self.data.qpos[7:7+ self.nJoints].copy()
        qH = self.data.qpos[7+self.nJoints:7+2*self.nJoints].copy()
        jointT = np.concatenate([qV,qH])
        self.jointHist.append(jointT)
        self.stepCounter +=1
        distance_from_origin = xy_position_after[1]
        reward, reward_info = self._get_rew(velocity_to_target, action,comYVelocity,distance_from_origin)
        info = {
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": distance_from_origin,
            "distance_from_target": distance_from_target_after,
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "center_of_mass": comAfter,
            "env": "Base env",
            **reward_info,
        }

        if self.render_mode == "human":
            self.render()
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info

    def _get_rew(self, velocity_to_target: float, action,comYVelocity,distance_from_origin):
        # forward_reward = self._forward_reward_weight * y_velocity
        # comVelocity_reward = self._com_velocity_weight * comYVelocity
        # signVel  = np.sign(velocity_to_target)
        # forward_reward =  velocity_to_target
        # distance_reward = self._distance_reward_weight * distance_from_origin
        # ctrl_cost = self.control_cost(action)

        # reward = signVel*(forward_reward)**2

        jointHist = self.jointHist[-self._predSamples:].copy() if len(self.jointHist) > self._predSamples else self.jointHist.copy()
        if self.stepCounter > 1 and self._epsilon_product != 1:
            predReward = computePredInfo(jointHist).item()
            
        #else:
            predReward = 1
        #predReward = 1
        forward_reward = velocity_to_target
        reward = np.sign(forward_reward) * (np.power(np.abs(forward_reward), self._epsilon_product) * np.power(predReward, 1 - self._epsilon_product))**2


        reward_info = {
            "reward_forward": forward_reward,
            "prediction_reward": predReward,
            # "reward_ctrl": -ctrl_cost,
            "total_reward": reward,
        }

        return reward, reward_info

    def _get_obs(self):
        position = self.data.qpos.flatten()
        velocity = self.data.qvel.flatten()

        if self._exclude_current_positions_from_observation:
            position = position[2:]

        observation = np.concatenate([position, velocity]).ravel()
        return observation

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )

        self.set_state(qpos, qvel)
        # self.cpgHstate = np.zeros(4*self.nJoints)
        # self.cpgVstate = np.zeros(4*self.nJoints)
        self.jointHist = []
        self.stepCounter = 0
        observation = self._get_obs()
        return observation

    def _get_reset_info(self):
        return {
            "x_position": self.data.qpos[0],
            "y_position": self.data.qpos[1],
            "distance_from_origin": np.linalg.norm(self.data.qpos[0:2], ord=2),
        }
