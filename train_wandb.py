
import gymnasium as gym
import wandb
from sbx import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
import os
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

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
                    "step": self.num_timesteps
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
    "max_n_steps": 10000,
    "xml_file":"./assets/snakeMotors2_14_rough.xml",
    "total_timesteps": 1_000_000,
    "policy": "MlpPolicy",
    "learning_rate": 3e-4,
    "batch_size": 64,
    "n_steps": 2048,
    "gamma": 0.99,
    "n_envs":4
}
if __name__ == "__main__":
    run = wandb.init(project="snakebot-training", 
            config=cfg,
            sync_tensorboard=True, )

    if cfg["n_envs"]==1:
        from gymnasium.envs.registration import register
        
        register(
            id="mySnake",
            entry_point = "src.snake_word_pred_v4:snakeEnvPred",
            max_episode_steps=cfg["max_n_steps"],
        )
        
        env = gym.make(cfg["env"], render_mode="rgb_array", xml_file=cfg["xml_file"])
        env = Monitor(env)
        
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
                env = gym.make(cfg["env"], render_mode="rgb_array", xml_file=cfg["xml_file"])
                env = Monitor(env)
                env.reset(seed=seed)
                return env
            return _init
        
        
        num_envs = cfg["n_envs"]
        env_fns = [make_env(seed=i) for i in range(num_envs)]
        env = SubprocVecEnv(env_fns)



    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=wandb.config.learning_rate,
        batch_size=wandb.config.batch_size,
        n_steps=wandb.config.n_steps,
        gamma=wandb.config.gamma,
        tensorboard_log=f"runs/{run.id}",  # Rely on wandb for logging
    )

    model.learn(total_timesteps=wandb.config.total_timesteps, 
                callback=[WandbWithGradientsCallback(), 
                        WandbCallback(verbose=2,)])
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

