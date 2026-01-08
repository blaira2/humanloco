
import numpy as np
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder
import matplotlib.pyplot as plt
import glob
import os

from humanoid_variants import VARIANTS
from morph_humanoid_env import MorphHumanoidEnv, compute_start_height
from balance_humanoid_env import BalanceHumanoidEnv


default_params = dict(
    TORSO_RADIUS=0.12,
    TORSO_HALF=0.35,
    TORSO_LENGTH=0.7,

    HEAD_RADIUS=0.11,
    HEAD_HEIGHT=0.45,

    HIP_OFFSET=0.1,
    THIGH_LENGTH=0.34,
    SHIN_LENGTH=0.30,
    THIGH_RADIUS=0.06,
    SHIN_RADIUS=0.05,

    FOOT_RADIUS=0.075,

    ARM_OFFSET=0.18,
    ARM_HEIGHT=0.35,
    ARM_RADIUS=0.04,
    ARM_LENGTH=0.25
)

MORPH_KEYS = [
    "TORSO_RADIUS", "TORSO_HALF", "TORSO_LENGTH",
    "HEAD_RADIUS",
    "ARM_RADIUS", "ARM_LENGTH", "ARM_OFFSET", "ARM_HEIGHT",
    "HIP_OFFSET",
    "THIGH_LENGTH", "SHIN_LENGTH",
    "THIGH_RADIUS", "SHIN_RADIUS",
]

def generate_xml(template_file, out_file, **params):
    with open(template_file, "r") as f:
        xml = f.read()

    # Replace all template parameters
    for key, val in params.items():
        xml = xml.replace(f"{{{key}}}", str(val))

    with open(out_file, "w") as f:
        f.write(xml)

    print("Generated:", out_file)

def preview(xml):
    env = gym.make("Humanoid-v5",
                   xml_file=os.path.abspath(xml),
                   render_mode="rgb_array")
    obs, info = env.reset()
    frame = env.render()
    plt.imshow(frame)
    plt.axis("off")
    plt.title(xml)
    plt.show()
    env.close()


def make_env(xml_file, morph_params, seed=0):
    def _init():
        env = MorphHumanoidEnv(
            xml_file=xml_file,
            morph_params=morph_params,
            render_mode=None        # IMPORTANT: no rendering in training
        )
        env.reset(seed=seed)
        return env
    return _init


def make_balance_env(xml_file, downward_accel_weight=1.0, morph_params=None, seed=0):
    def _init():
        env = BalanceHumanoidEnv(
            xml_file=xml_file,
            downward_accel_weight=downward_accel_weight,
            morph_params=morph_params,
            render_mode=None,  # IMPORTANT: no rendering in training
        )
        env.reset(seed=seed)
        return env
    return _init

class VideoEveryNEpisodesCallback(BaseCallback):
    def __init__(self, video_every, xml_file, morph, out_dir, verbose=0):
        super().__init__(verbose)
        self.video_every = video_every
        self.xml_file = xml_file
        self.morph = morph
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)
        self.episode_count = 0

    def _on_step(self):
        # Check how many episodes finished this step
        done_array = self.locals.get("dones")
        if done_array is None:
            return True

        for done in done_array:
            if done:
                self.episode_count += 1

                if self.episode_count % self.video_every == 0:
                    self._record_video()

        return True

    def _record_video(self):
        video_folder = os.path.join(self.out_dir, f"ep_{self.episode_count}")
        os.makedirs(video_folder, exist_ok=True)

        # Create a recordable environment
        env = MorphHumanoidEnv(
            xml_file=self.xml_file,
            morph_params=self.morph,
            render_mode="rgb_array"
        )
        env = VecVideoRecorder(
            DummyVecEnv([lambda: env]),
            video_folder,
            record_video_trigger=lambda x: x == 0,
            video_length=1000,
            name_prefix=f"ep_{self.episode_count}"
        )

        obs = env.reset()
        done = False
        while not done:
            action, _ = self.model.predict(obs, deterministic=True)
            obs, _, done, _ = env.step(action)

        env.close()
        print(f"[🎥] Saved video at episode {self.episode_count} → {video_folder}")

policy_kwargs = dict( # MLP
        net_arch=[128, 96, 64]  # 3 hidden layers shared by pi & vf
    )

def train_balance_env(
        variant_name,
        cfg,
        xml_file,
        timesteps=300_000,
        parallel_envs=3,
        initial_learning_rate=2e-4,
        video_every=10,
        pretrained_model=None):

    xml_path = os.path.abspath(xml_file)

    # ---------- CREATE VEC ENV ----------
    env_fns = [
        make_balance_env(xml_path, cfg, seed=i)
        for i in range(parallel_envs)
    ]
    if parallel_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    vec_env = VecMonitor(vec_env)
    vec_env.save(f"./norms/{variant_name}_vecnorm.pkl")

    # ---------- LOGGER ----------
    logger = configure(folder=f"./logs_{variant_name}", format_strings=["stdout", "csv", "tensorboard"])

    # learning
    final_lr = initial_learning_rate *.1
    lr_schedule = LinearSchedule(
        start=5e-4,
        end=final_lr,
        end_fraction=.9
    )

    # ---------- MODEL ----------
    if pretrained_model is None:
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            n_steps=2048 // parallel_envs,   # good rule-of-thumb
            batch_size=128,
            learning_rate=lr_schedule,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1
        )
    elif isinstance(pretrained_model, (str, os.PathLike)):
        model = PPO.load(pretrained_model, env=vec_env)
        model.learning_rate = lr_schedule
        model.lr_schedule = lr_schedule
    else:
        model = pretrained_model
        model.set_env(vec_env)
        model.learning_rate = lr_schedule
        model.lr_schedule = lr_schedule
    model.set_logger(logger)

    # ---------- CALLBACK FOR VIDEO ----------
    video_cb = VideoEveryNEpisodesCallback(
        video_every=video_every,
        xml_file=xml_path,
        morph=cfg,
        out_dir=f"{variant_name}_videos"
    )

    debug_cb = RewardDebugCallback()

    # ---------- callback for incrementing forward reward scale ----------
    forward_cb = ForwardRampCallback(total_timesteps=timesteps)

    # ---------- TRAIN ----------
    print(f"\n🚀 Training PPO for {variant_name} ...")
    model.learn(total_timesteps=timesteps, callback=[video_cb, debug_cb, forward_cb])

    # ---------- SAVE ----------
    model.save(f"{variant_name}_ppo.zip")
    vec_env.close()
    print(f"✔️ Training complete for {variant_name}")

    return model

def train_variant(
        variant_name,
        cfg,
        xml_file,
        timesteps=300_000,
        parallel_envs=3,
        initial_learning_rate=2e-4,
        video_every=10,
        pretrained_model=None):

    xml_path = os.path.abspath(xml_file)

    # ---------- CREATE VEC ENV ----------
    env_fns = [
        make_env(xml_path, cfg, seed=i)
        for i in range(parallel_envs)
    ]
    if parallel_envs == 1:
        vec_env = DummyVecEnv(env_fns)
    else:
        vec_env = SubprocVecEnv(env_fns)
    # Normalize observations and rewards
    vec_env = VecNormalize(
        vec_env,
        norm_obs=False,
        norm_reward=False,
        clip_obs=10.0,
        clip_reward=10.0,
    )
    vec_env = VecMonitor(vec_env)
    vec_env.save(f"./norms/{variant_name}_vecnorm.pkl")

    # ---------- LOGGER ----------
    logger = configure(folder=f"./logs_{variant_name}", format_strings=["stdout", "csv", "tensorboard"])

    # learning
    final_lr = initial_learning_rate *.1
    lr_schedule = LinearSchedule(
        start=5e-4,
        end=final_lr,
        end_fraction=.9
    )

    # ---------- MODEL ----------
    if pretrained_model is None:
        model = PPO(
            "MlpPolicy",
            vec_env,
            policy_kwargs=policy_kwargs,
            n_steps=2048 // parallel_envs,   # good rule-of-thumb
            batch_size=128,
            learning_rate=lr_schedule,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1
        )
    elif isinstance(pretrained_model, (str, os.PathLike)):
        model = PPO.load(pretrained_model, env=vec_env)
        model.learning_rate = lr_schedule
        model.lr_schedule = lr_schedule
    else:
        model = pretrained_model
        model.set_env(vec_env)
        model.learning_rate = lr_schedule
        model.lr_schedule = lr_schedule
    model.set_logger(logger)

    # ---------- CALLBACK FOR VIDEO ----------
    video_cb = VideoEveryNEpisodesCallback(
        video_every=video_every,
        xml_file=xml_path,
        morph=cfg,
        out_dir=f"{variant_name}_videos"
    )

    debug_cb = RewardDebugCallback()

    # ---------- callback for incrementing forward reward scale ----------
    forward_cb = ForwardRampCallback(total_timesteps=timesteps)

    # ---------- TRAIN ----------
    print(f"\n🚀 Training PPO for {variant_name} ...")
    model.learn(total_timesteps=timesteps, callback=[video_cb, debug_cb, forward_cb])

    # ---------- SAVE ----------
    model.save(f"{variant_name}_ppo.zip")
    vec_env.close()
    print(f"✔️ Training complete for {variant_name}")

    return model


def collect_trajectories(model, variant_name, cfg, num_episodes=15):
    xml_path = os.path.abspath(cfg["xml"])
    base_env = DummyVecEnv([make_env(xml_path, cfg, seed=0)])
    # ----- Try to load VecNormalize stats -----
    vecnorm_path = f"./norms/{variant_name}_vecnorm.pkl"
    if os.path.exists(vecnorm_path):
        env = VecNormalize.load(vecnorm_path, base_env)
        # Inference mode: do NOT update stats, and keep raw rewards
        env.training = False
        env.norm_reward = False
        print(f"[collect] Loaded VecNormalize stats from {vecnorm_path}")
    else:
        env = base_env
        print("[collect] VecNormalize stats not found, using unnormalized env.")

    if isinstance(env, VecNormalize):
        raw_env = env.venv.envs[0]
    else:
        raw_env = env.envs[0]
    mj_env = raw_env.unwrapped

    morph_vec = np.array([cfg[k] for k in MORPH_KEYS], dtype=np.float32)

    traj = {k: [] for k in ["qpos","qvel","action","reward",
                            "next_qpos","next_qvel","morph","phase"]}

    for ep in range(num_episodes):
        obs = env.reset()
        done = False
        step = 0

        while not done:
            qpos = mj_env.data.qpos.copy()
            qvel = mj_env.data.qvel.copy()
            phase = (step % 200) / 200.0
            action, _ = model.predict(obs, deterministic=True)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            next_qpos = mj_env.data.qpos.copy()
            next_qvel = mj_env.data.qvel.copy()

            traj["qpos"].append(qpos)
            traj["qvel"].append(qvel)
            traj["action"].append(action)
            traj["reward"].append(reward)
            traj["next_qpos"].append(next_qpos)
            traj["next_qvel"].append(next_qvel)
            traj["morph"].append(morph_vec)
            traj["phase"].append(phase)

            obs = next_obs
            step += 1

    np.savez_compressed(f"{variant_name}_traj.npz", **traj)
    print("Saved:", f"{variant_name}_traj.npz")
    env.close()


class RewardDebugCallback(BaseCallback):
    """
    Accumulates reward component sums per env and prints episode averages.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.ep_sums = None
        self.ep_len = None

    def _on_training_start(self) -> None:
        n_envs = self.training_env.num_envs
        self.ep_sums = []
        self.ep_len = [0] * n_envs
        for _ in range(n_envs):
            self.ep_sums.append({
                "total": 0.0,
                "forward": 0.0,
                "alive": 0.0,
                "lateral_penalty": 0.0,
                "com_penalty": 0.0,
                "angular_penalty": 0.0,
                "accel_penalty": 0.0,
                "energy_penalty": 0.0,
            })

    def _on_step(self) -> bool:
        rewards = self.locals["rewards"]  # shape (n_envs,)
        infos = self.locals["infos"]  # list of dicts
        dones = self.locals["dones"]  # shape (n_envs,)

        n_envs = len(dones)
        for i in range(n_envs):
            info = infos[i]
            s = self.ep_sums[i]


            s["total"] += float(rewards[i])
            s["forward"] += info.get("forward_reward", 0.0)
            s["alive"] += info.get("alive_reward", 0.0)
            s["lateral_penalty"] += info.get("lateral_penalty", 0.0)
            s["com_penalty"] += info.get("com_penalty", 0.0)
            s["angular_penalty"] += info.get("angular_penalty", 0.0)
            s["accel_penalty"] += info.get("accel_penalty", 0.0)
            s["energy_penalty"] += info.get("energy_penalty", 0.0)

            self.ep_len[i] += 1

            if dones[i]:
                L = self.ep_len[i] or 1
                print(
                    f"[env {i}] ep_len={L:4d} | "
                    f"R_mean={s['total'] / L: .3f} | "
                    f"fwd={s['forward'] / L: .3f} | "
                    f"alive={s['alive'] / L: .3f} | "
                    f"lat_p={s['lateral_penalty'] / L: .3f} | "
                    f"accel_p={s['accel_penalty'] / L: .3f} | "
                    f"com_p={s['com_penalty'] / L: .3f} | "
                    f"ang_p={s['angular_penalty'] / L: .3f} | "
                    f"energy_p={s['energy_penalty'] / L: .3f}"
                )

                # reset for next episode
                self.ep_sums[i] = {k: 0.0 for k in s}
                self.ep_len[i] = 0

        return True

class ForwardRampCallback(BaseCallback):
    def __init__(self, total_timesteps, start=0.2, end=0.5, verbose=0):
        super().__init__(verbose)
        self.total_timesteps = total_timesteps
        self.start = start
        self.end = end

    def _on_step(self) -> bool:
        progress = min(self.num_timesteps / self.total_timesteps, 1.0)
        scale = self.start + progress * (self.end - self.start)

        self.training_env.env_method("set_forward_scale", scale)
        return True
