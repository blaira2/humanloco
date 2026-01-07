
import numpy as np
import gymnasium as gym
import mujoco as mj
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecVideoRecorder
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.utils import LinearSchedule
from gymnasium.envs.mujoco.humanoid_v5 import HumanoidEnv
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize, VecVideoRecorder
import matplotlib.pyplot as plt
import glob
import os


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

VARIANTS = {
    "variant1": dict(
        xml="humanoid_variant1.xml",
        TORSO_RADIUS=0.12, TORSO_HALF=0.35, TORSO_LENGTH=0.7,
        HEAD_RADIUS=0.11, HEAD_HEIGHT=0.45,
        ARM_RADIUS=0.04, ARM_LENGTH=0.25, ARM_OFFSET=0.18, ARM_HEIGHT=0.35,
        HIP_OFFSET=0.10,
        THIGH_LENGTH=0.34, SHIN_LENGTH=0.30,
        THIGH_RADIUS=0.06, SHIN_RADIUS=0.05,
        FOOT_RADIUS=0.075,
    ),
    "variant2": dict(
        xml="humanoid_variant2.xml",
        TORSO_RADIUS=0.10, TORSO_HALF=0.25, TORSO_LENGTH=0.55,
        HEAD_RADIUS=0.10, HEAD_HEIGHT=0.38,
        ARM_RADIUS=0.035, ARM_LENGTH=0.20, ARM_OFFSET=0.16, ARM_HEIGHT=0.32,
        HIP_OFFSET=0.10,
        THIGH_LENGTH=0.34, SHIN_LENGTH=0.30,
        THIGH_RADIUS=0.055, SHIN_RADIUS=0.045,
        FOOT_RADIUS=0.07,
    ),
    "variant3": dict(
        xml="humanoid_variant3.xml",
        TORSO_RADIUS=0.13, TORSO_HALF=0.45, TORSO_LENGTH=0.85,
        HEAD_RADIUS=0.115, HEAD_HEIGHT=0.52,
        ARM_RADIUS=0.045, ARM_LENGTH=0.33, ARM_OFFSET=0.20, ARM_HEIGHT=0.40,
        HIP_OFFSET=0.10,
        THIGH_LENGTH=0.34, SHIN_LENGTH=0.30,
        THIGH_RADIUS=0.06, SHIN_RADIUS=0.05,
        FOOT_RADIUS=0.075,
    ),
    "variant4": dict(
        xml="humanoid_variant4.xml",
        TORSO_RADIUS=0.12, TORSO_HALF=0.35, TORSO_LENGTH=0.7,
        HEAD_RADIUS=0.11, HEAD_HEIGHT=0.45,
        ARM_RADIUS=0.04, ARM_LENGTH=0.25, ARM_OFFSET=0.18, ARM_HEIGHT=0.35,
        HIP_OFFSET=0.10,
        THIGH_LENGTH=0.42, SHIN_LENGTH=0.36,
        THIGH_RADIUS=0.055, SHIN_RADIUS=0.045,
        FOOT_RADIUS=0.075,
    ),
    "variant5": dict(
        xml="humanoid_variant5.xml",
        TORSO_RADIUS=0.12, TORSO_HALF=0.33, TORSO_LENGTH=0.65,
        HEAD_RADIUS=0.11, HEAD_HEIGHT=0.43,
        ARM_RADIUS=0.04, ARM_LENGTH=0.25, ARM_OFFSET=0.18, ARM_HEIGHT=0.34,
        HIP_OFFSET=0.10,
        THIGH_LENGTH=0.26, SHIN_LENGTH=0.22,
        THIGH_RADIUS=0.06, SHIN_RADIUS=0.05,
        FOOT_RADIUS=0.072,
    ),
    "variant6": dict(
        xml="humanoid_variant6.xml",
        TORSO_RADIUS=0.16, TORSO_HALF=0.40, TORSO_LENGTH=0.75,
        HEAD_RADIUS=0.13, HEAD_HEIGHT=0.47,
        ARM_RADIUS=0.055, ARM_LENGTH=0.25, ARM_OFFSET=0.19, ARM_HEIGHT=0.35,
        HIP_OFFSET=0.12,
        THIGH_LENGTH=0.34, SHIN_LENGTH=0.30,
        THIGH_RADIUS=0.075, SHIN_RADIUS=0.065,
        FOOT_RADIUS=0.085,
    ),
    "variant7": dict(
        xml="humanoid_variant7.xml",
        TORSO_RADIUS=0.09, TORSO_HALF=0.30, TORSO_LENGTH=0.60,
        HEAD_RADIUS=0.09, HEAD_HEIGHT=0.40,
        ARM_RADIUS=0.03, ARM_LENGTH=0.28, ARM_OFFSET=0.16, ARM_HEIGHT=0.34,
        HIP_OFFSET=0.09,
        THIGH_LENGTH=0.34, SHIN_LENGTH=0.30,
        THIGH_RADIUS=0.045, SHIN_RADIUS=0.04,
        FOOT_RADIUS=0.065,
    ),
    "default_var": dict(
        xml="humanoid_default.xml",
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
        # FOOT_OFFSET=0.00,

        ARM_OFFSET=0.18,
        ARM_HEIGHT=0.35,
        ARM_RADIUS=0.04,
        ARM_LENGTH=0.25
    ),
"test_var": dict(
        xml="humanoid_custom.xml",
        TORSO_RADIUS=0.12,
        TORSO_HALF=0.35,
        TORSO_LENGTH=0.7,

        HEAD_RADIUS=0.11,
        HEAD_HEIGHT=0.45,

        HIP_OFFSET=0.1,
        THIGH_LENGTH=0.4,
        SHIN_LENGTH=0.4,
        THIGH_RADIUS=0.06,
        SHIN_RADIUS=0.05,

        FOOT_RADIUS=0.075,
        # FOOT_OFFSET=0.00,

        ARM_OFFSET=0.18,
        ARM_HEIGHT=0.35,
        ARM_RADIUS=0.04,
        ARM_LENGTH=0.25
    ),
}

MORPH_KEYS = [
    "TORSO_RADIUS", "TORSO_HALF", "TORSO_LENGTH",
    "HEAD_RADIUS",
    "ARM_RADIUS", "ARM_LENGTH", "ARM_OFFSET", "ARM_HEIGHT",
    "HIP_OFFSET",
    "THIGH_LENGTH", "SHIN_LENGTH",
    "THIGH_RADIUS", "SHIN_RADIUS",
]

def compute_start_height(params):
    """
    Compute correct starting torso height so feet rest on the floor.
    """
    thigh = params["THIGH_LENGTH"]
    shin = params["SHIN_LENGTH"]
    foot = params["FOOT_RADIUS"]

    SAFETY = 0.14   # extra buffer so initial state avoids penetration
    return thigh + shin + foot + SAFETY


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

def train_variant_ppo(
        variant_name,
        cfg,
        xml_file,
        timesteps=300_000,
        parallel_envs=3,
        initial_learning_rate=2e-4,
        video_every=10):

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

def continue_training(variant_name, cfg, more_timesteps=300_000, parallel_envs=3, video_every=100):
    xml_path = os.path.abspath(cfg["xml"])

    # --- Recreate base envs ---
    env_fns = [make_env(xml_path, cfg, seed=i) for i in range(parallel_envs)]

    if parallel_envs == 1:
        base_vec = DummyVecEnv(env_fns)
    else:
        base_vec = SubprocVecEnv(env_fns)

    # --- Load VecNormalize stats ---
    vecnorm_path = f"./norms/{variant_name}_vecnorm.pkl"
    vec_env = VecNormalize.load(vecnorm_path, base_vec)
    vec_env = VecMonitor(vec_env)

    vec_env.training = True
    vec_env.norm_reward = True

    # ---------- LOGGER ----------
    logger = configure(folder=f"./logs_{variant_name}", format_strings=["stdout", "csv", "tensorboard"])

    # --- Load PPO and attach env ---
    model_path = f"{variant_name}_ppo.zip"
    model = PPO.load(model_path, env=vec_env)
    model.set_logger(logger) #

    video_cb = VideoEveryNEpisodesCallback(
        video_every=video_every,
        xml_file=xml_path,
        morph=cfg,
        out_dir=f"{variant_name}_videos"
    )

    # --- Continue training ---
    model.learn(total_timesteps=more_timesteps, reset_num_timesteps=False, callback=video_cb)

    # --- Save updated model + stats ---
    model.save(f"{variant_name}_ppo_continued.zip")
    vec_env.save(f"./norms/{variant_name}_vecnorm.pkl")

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


class MorphHumanoidEnv(HumanoidEnv):
    """
    Custom Humanoid environment that:
      • Adjusts healthy_z_range based on morphology
      • Uses a more robust fall detection method
      • Avoids instant terminal states for crouching or short robots
      • Still terminates properly when falling / lying down
    """

    def __init__(self, xml_file, morph_params, **kwargs):
        self.morph = morph_params

        # -----------------------------
        # Compute morphology-based torso height
        # -----------------------------
        thigh = morph_params.get("THIGH_LENGTH", 0.34)
        shin  = morph_params.get("SHIN_LENGTH", 0.30)
        torso = morph_params.get("TORSO_LENGTH", 0.70)
        head  = morph_params.get("HEAD_HEIGHT", 0.45)
        foot  = morph_params.get("FOOT_RADIUS", 0.075)

        self.base_height = thigh + shin + foot       # hip -> ground distance
        upper_height = torso + head             # hip -> head distance
        standing_height = self.base_height + upper_height
        self.start_height = compute_start_height(morph_params)

        # Store for debugging
        self.morph_standing_height = standing_height
        self._prev_action = None
        # -----------------------------
        # Compute robust healthy_z_range
        # -----------------------------
        # Instead of:
        #   healthy if standing_height * 0.5 < z < standing_height * 1.5
        #
        # Use:
        #   • Torso must be ABOVE 20% of standing height
        #   • Torso must be BELOW 200% (sanity)
        #   • Allows crouching/kneeling but not lying flat
        #
        min_healthy = 0.15 * standing_height
        max_healthy = 2.0 * standing_height

        self.custom_healthy_min = min_healthy
        self.custom_healthy_max = max_healthy

        #stay healthy
        self._steps_alive = 0  # counts how many steps we stay alive

        self.forward_scale = .25
        self.prev_com_margin = 0

        super().__init__(xml_file=xml_file,forward_reward_weight=0.0, terminate_when_unhealthy=True, **kwargs)

        #cache some body parts
        self.torso_geom_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM, "torso1")
        self.head_geom_id  = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_GEOM,"head")
        self.torso_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "torso")
        self.left_foot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "left_foot")
        self.right_foot_body_id = mj.mj_name2id(self.model, mj.mjtObj.mjOBJ_BODY, "right_foot")

        self.init_qpos[2] = self.start_height

        # Override the internal range the parent env will check
        self._healthy_z_range = (self.custom_healthy_min,
                                 self.custom_healthy_max)

        # -----------------------------
        # 3. Additional robustness parameters
        # -----------------------------
        self.max_tilt = np.deg2rad(70)  # if torso tilts >70°, consider fallen
        self.min_base_height = 0.03      # if the pelvis hits the floor → terminal

    def step(self, action):
        # ---- call original HumanoidEnv step ----
        obs, base_reward, terminated, truncated, info = super().step(action)

        # ----------------------
        # Base kinematics
        # ----------------------
        x_vel = float(self.data.qvel[0])  # forward speed
        y_vel = float(self.data.qvel[1])  # lateral speed
        z_vel = float(self.data.qvel[2]) # downward speed
        x_vel = max(x_vel, 0.0)  # no reward for walking backwards
        ang_vel = self.data.qvel[3:6]  # angular vel


        # Alive reward
        # small constant per timestep + big penalty on fall
        alive_step = 1
        self._steps_alive += 1
        alive_reward = alive_step if not (terminated or truncated) else 0.0
        #terminal penalty shrinks over time
        survival_frac = np.clip(self._steps_alive / 1000, 0.0, 1.0)
        max_penalty = 150.0
        terminal_penalty = max_penalty * (1.0 - survival_frac)
        if not terminated:  # only if it actually fell, not time-limit
            terminal_penalty = 0.0

        #  Forward reward
        # reward grows with speed but saturates, and is posture-gated
        x_vel_clipped = min(x_vel, 3.0)  # cap speed for reward purposes
        forward_base = self.forward_scale * x_vel_clipped

        # energy penalty = discourage huge torques
        action = np.asarray(action)
        energy = np.sum(action**2)
        n = len(action)
        energy_penalty = 1 * (energy / n)

        forward_reward = 4.5 * forward_base

        # -----------------
        # Minor Penalties
        # -----------------
        # downward penalty
        downward_speed = max(0.0, -z_vel)  # positive when going down
        downward_penalty = 0.2 * (1 - self.forward_scale) * downward_speed

        # sideways = discourage strafing
        lateral_penalty = 0.25 * abs(y_vel)

        # --- Center of mass and feet geometry ---
        com = np.array(self.data.subtree_com[self.torso_body_id])  # [x, y, z]
        com_xy = com[:2]
        lf_xy = self.data.xipos[self.left_foot_body_id, :2]
        rf_xy = self.data.xipos[self.right_foot_body_id, :2]
        feet_mid = 0.5 * (lf_xy + rf_xy)
        # is com near any balance point
        d_left = np.linalg.norm(com_xy - lf_xy)
        d_right = np.linalg.norm(com_xy - rf_xy)
        d_mid = np.linalg.norm(com_xy - feet_mid)
        d_min = float(min(d_left, d_right, d_mid))
        d_excess = max(1e-4, d_min - 0.1)
        d_min_dt = (d_excess - self.prev_com_margin) / self.model.opt.timestep
        self.prev_com_margin = d_excess
        com_penalty = .3 * d_min_dt
        if com_penalty < 0:
            com_penalty = .75 * com_penalty # weaker reward than penalty



        # angular velocity penalty
        angular_penalty = .05 * np.linalg.norm(ang_vel[:2])  # pitch/roll wobble


        # Combine

        reward = (
                alive_reward
                + forward_reward
                - terminal_penalty
                - lateral_penalty
                - downward_penalty
                - angular_penalty
                - energy_penalty
                - com_penalty
        )

        info["energy_penalty"] = float(energy_penalty)
        info["forward_reward"] = float(forward_reward)
        info["lateral_penalty"] = float(lateral_penalty)
        info["down_penalty"] = float(downward_penalty)
        info["com_penalty"] = float(com_penalty)
        info["angular_penalty"] = float(angular_penalty)
        info["alive_reward"] = float(alive_reward)


        return obs, reward, terminated, truncated, info

    # -----------------------------------
    #               RESET
    # -----------------------------------
    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)

        self._steps_alive = 0
        self._prev_action = None
        # If initial reset is below threshold, lift robot a little
        torso_z = obs[0]
        if torso_z < self.custom_healthy_min:
            delta = self.custom_healthy_min - torso_z + 0.05
            self.data.qpos[2] += delta
            mj.mj_forward(self.model, self.data)
            obs = self._get_obs()

        return obs, info


    # -----------------------------------
    #           CUSTOM HEALTH CHECK
    # -----------------------------------
    def _is_healthy(self, state):
        """
        Improved fall detection:
          • Torso above minimum height
          • Torso below maximum height
          • Torso not excessively tilted
          • Pelvis not touching floor
        """

        torso_z = state[2]
        min_z, max_z = self._healthy_z_range

        # Height check
        if not (min_z <= torso_z <= max_z):
            return False

        # Pelvis height (qpos index 2 is torso, index 5 is pelvis if using Humanoid-v5 defaults)
        pelvis_z = float(self.data.qpos[2])
        if pelvis_z < self.min_base_height:
            return False

        # Torso tilt check
        # Orientation is quaternion (w, x, y, z), Humanoid-v5 places torso orientation early in qpos
        torso_quat = self.data.xquat[1]   # body index 1 = torso in default humanoid
        w, x, y, z = torso_quat

        # Convert quaternion → tilt angle
        tilt_angle = 2 * np.arccos(abs(w))
        if tilt_angle > self.max_tilt:
            return False

        return True

    def set_forward_scale(self, scale: float):
        self.forward_scale = float(scale)


    # -----------------------------------
    # Optional diagnostics helper
    # -----------------------------------
    def print_diagnostics(self, state):
        z = state[2]
        min_z, max_z = self._healthy_z_range

        print("\n--- HEALTH DIAGNOSTICS ---")
        print(f"Standing height:       {self.morph_standing_height:.3f}")
        print(f"Healthy z-range:       [{min_z:.3f}, {max_z:.3f}]")
        print(f"Current torso z:       {z:.3f}")
        print(f"Base pelvis height:    {float(self.data.qpos[2]):.3f}")

        # tilt
        w, x, y, zq = self.data.xquat[1]
        tilt = 2 * np.arccos(abs(w))
        print(f"Torso tilt angle deg:  {np.rad2deg(tilt):.2f}")
        print("--------------------------")

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
                "down_penalty": 0.0,
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
            s["down_penalty"] += info.get("down_penalty", 0.0)
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
                    f"down_p={s['down_penalty'] / L: .3f} | "
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