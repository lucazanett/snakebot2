
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
            
        else:
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
