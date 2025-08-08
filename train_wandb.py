
import gymnasium as gym
import wandb
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback,EvalCallback,CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import os
from typing import Callable
import argparse
import uuid
import random
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to the GPU you want to use

from tqdm import tqdm

class WandbWithGradientsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # Log environment info
        for info in self.locals.get("infos", []):
            for k, v in info.items():
                if isinstance(v, (int, float)):
                    wandb.log({k: v, "step": self.num_timesteps})
            if "episode" in info:
                reward = info["episode"]["r"]
                length = info["episode"]["l"]
                wandb.log({
                    "episode_reward": reward,
                    "episode_length": length,
                    "episode_reward_mean": reward / length,
                    "step": self.num_timesteps,
                    "final_distance_from_target": info["episode"]["distance_from_target"],
                    "step_target_reached": info["episode"]["step_target_reached"],
                    
                })

        return True

    def _on_rollout_end(self):
        # Compute and log gradient norm
        if hasattr(self.model.policy, "parameters"):
            total_norm = 0.0
            for p in self.model.policy.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            total_norm = total_norm ** 0.5
            wandb.log({"grad_norm": total_norm, "step": self.num_timesteps})



cfg = {
    "algo": "PPO",
    "env": "mySnake",
    "max_n_steps": 8192,
    "xml_file": os.path.join(os.path.dirname(__file__),"assets", "snakeMotors2_14_highRough.xml"),
    "total_timesteps": 8_192_000,
    "policy": "MlpPolicy",
    "learning_rate": "lin_0.0003",
    "batch_size": 64,
    "n_steps": 2048,
    "gamma": 0.99,
    "n_envs":4
}
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--eps", type=float, default=0.25, help="Epsilon product ")
    parser.add_argument("--xml_file", type=str, default="snakeMotors2_14_highRough.xml", help="XML file for the environment")
    args = parser.parse_args()
    eps = args.eps
    cfg["xml_file"] = cfg["xml_file"].replace(cfg["xml_file"].split("/")[-1], args.xml_file)
    terrain = cfg["xml_file"].split("/")[-1].split("_")[-1].split(".")[0]
    def linear_schedule(initial_value: float) -> Callable[[float], float]:
        """
        Linear learning rate schedule.

        :param initial_value: Initial learning rate.
        :return: schedule that computes
        current learning rate depending on remaining progress
        """
        def func(progress_remaining: float) -> float:
            """
            Progress will decrease from 1 (beginning) to 0.

            :param progress_remaining:
            :return: current learning rate
            """
            return progress_remaining * initial_value

        return func
    identifier = uuid.uuid4().hex[:10]
    cfg["identifier"] = identifier
    cfg["color"] = random.randint(0, 1000)    
    run = wandb.init(project="snakebot-training-Unige", 
            config=cfg,
            name = f"snakebot_{terrain}_{eps}_{identifier}",
            sync_tensorboard=True,
            tags=[terrain, f"eps_{eps}"],)

    if cfg["n_envs"]==1:
        from gymnasium.envs.registration import register
        
        register(
            id="mySnake",
            entry_point = "src.snake_word_pred_v4:snakeEnvPred",
            max_episode_steps=cfg["max_n_steps"],
        )
        
        env = gym.make(cfg["env"], render_mode="rgb_array", xml_file=cfg["xml_file"])
        env = Monitor(env,info_keywords = ('distance_from_target','step_target_reached',))
        
    else:
        from stable_baselines3.common.vec_env import SubprocVecEnv,VecMonitor
        from gymnasium.envs.registration import register
                

        def make_env(seed):
            def _init():

                register(
                                id="mySnake",
                                entry_point="src.snake_word_pred_v4:snakeEnvPred",
                                max_episode_steps=cfg["max_n_steps"],
                            )
                env = gym.make(cfg["env"], render_mode="rgb_array", xml_file=cfg["xml_file"],epsilon_product=eps)
                env = Monitor(env,info_keywords = ('distance_from_target','step_target_reached',))
                env.reset(seed=seed)
                return env
            return _init
        
        
        num_envs = cfg["n_envs"]
        env_fns = [make_env(seed=i) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns,start_method="spawn")



    model = PPO(
        "MlpPolicy",    
        env,
        verbose=1,
        learning_rate=wandb.config.learning_rate if isinstance(wandb.config.learning_rate, float) else linear_schedule(float(cfg["learning_rate"].split("_")[1])),
        batch_size=wandb.config.batch_size,
        n_steps=wandb.config.n_steps,
        gamma=wandb.config.gamma,
        tensorboard_log=f"runs/{run.id}",  # Rely on wandb for logging
    )

    model.learn(total_timesteps=wandb.config.total_timesteps, 
                callback=[WandbWithGradientsCallback(), 
                        WandbCallback(verbose=2,),CheckpointCallback(save_freq=8192*4 // cfg["n_envs"], save_path=f'checkpoints/check_{terrain}_{eps}_{identifier}', name_prefix=f"ppo")],)
    os.makedirs("checkpoints", exist_ok=True)
    model.save("checkpoints/ppo_mujoco_model")
    wandb.finish()

    # env0 = env.env.env.env.env
    # for i in range(5):
    #     obs, _ = env0.reset()
    #     done = False

    #     tot_reward = 0

    #     progress_bar = tqdm(range(1000), desc=f"Episode {i+1}")

    #     for i in progress_bar:
    #         action, _ = model.predict(obs)
    #         obs,rewards,done,info, _ = env0.step(action)
    #         tot_reward += rewards
    #         progress_bar.set_postfix({"reward": tot_reward})

