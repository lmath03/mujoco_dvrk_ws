import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback

def linear_schedule(initial_value):
    def func(progress_remaining):
       return progress_remaining * initial_value
    return func

def make_env(rank):
    def _init():
        env = PandaReachEnv("ik_sceneBYEBYE.xml")
        env.reset(seed=rank)
        return env
    return _init

class PandaReachEnv(gym.Env):
    def __init__(self, model_path="ik_sceneBYEBYE.xml"):
        super().__init__()
        
        # ==========================================
        # MOTORE FISICO 
        # ==========================================
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")    # Punto 1
        self.box_id2 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box2")  # Punto 2
        self.box_id3 = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box3")  # Punto 3
        self.link7_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "link7")
        
        # ==========================================
        # DEFINIZIONE SPAZI
        # ==========================================
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # Giunti(7) + Vel(7) + Distanza_Target_Attivo(3) + Orientamento(3) = 20
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float32)

        self.max_steps = 3000
        self.current_step = 0
        self.prev_dist = 0.0

        # Timeout di sicurezza per singolo target (se l'agente fatica all'inizio)
        self.max_steps_per_target = 350

        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # Sequenza 1 -> 2 -> 3 -> 2 per 3 cicli completi + finale in 1
        cycle = [self.box_id, self.box_id2, self.box_id3, self.box_id2]
        self.target_sequence = cycle * 3 + [self.box_id]
        
        self.current_target_index = 0
        self.active_target_id = self.target_sequence[self.current_target_index]
        
        self.steps_on_current_target = 0
        self.target_hold_counter = 0  # Contatore dei 5 step entro 0.02m

        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "custom_home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        noise = self.np_random.uniform(low=-0.05, high=0.05, size=7)
        self.data.qpos[:7] += noise
        
        mujoco.mj_forward(self.model, self.data)
        
        target_pos = self.data.xpos[self.active_target_id]
        ee_pos = self.data.site_xpos[self.site_id]
        self.prev_dist = np.linalg.norm(ee_pos - target_pos)
        
        return self._get_obs(), {}

    def _get_obs(self):
        giunti = self.data.qpos[:7].copy()
        vel_angolare = self.data.qvel[:7].copy()
        
        target_pos = self.data.xpos[self.active_target_id].copy()
        punta_pos = self.data.site_xpos[self.site_id].copy()
        relative_target = target_pos - punta_pos

        osservazione = np.concatenate([
            giunti,             # 7
            vel_angolare,       # 7
            relative_target,    # 3
        ])

        return osservazione.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        self.steps_on_current_target += 1

        max_delta = 0.05
        action_delta = action * max_delta

        current_qpos = self.data.qpos[:7].copy()
        target_qpos = current_qpos + action_delta

        low_limit = self.model.jnt_range[:7, 0]
        high_limit = self.model.jnt_range[:7, 1]
        self.data.ctrl[:7] = np.clip(target_qpos, low_limit, high_limit)
        
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.sync()
            
        # 1. Distanza attuale dal target attivo
        ee_pos = self.data.site_xpos[self.site_id]
        target_pos = self.data.xpos[self.active_target_id]
        distanza = np.linalg.norm(ee_pos - target_pos)
        
        # 2. Controllo vicinanza (< 0.02m per 5 step consecutivi)
        is_close = (distanza < 0.02)
        if is_close:
            self.target_hold_counter += 1
        else:
            self.target_hold_counter = 0

        target_completato = (self.target_hold_counter >= 5)
        timeout_target = (self.steps_on_current_target >= self.max_steps_per_target)

        # 3. Calcolo della Reward per lo step corrente
        reward, info_reward = self._compute_reward(action, distanza, is_close, target_completato)

        # 4. Transizione al Target Successivo
        if target_completato or timeout_target:
            self.current_target_index += 1
            self.target_hold_counter = 0
            self.steps_on_current_target = 0
            
            # Se ha concluso i 3 cicli di saluto, ricomincia la sequenza
            if self.current_target_index >= len(self.target_sequence):
                self.current_target_index = 0

            self.active_target_id = self.target_sequence[self.current_target_index]
            
            # CRITICO: Reset di prev_dist per il nuovo target per evitare sbalzi di reward!
            ee_pos = self.data.site_xpos[self.site_id]
            nuovo_target_pos = self.data.xpos[self.active_target_id]
            self.prev_dist = np.linalg.norm(ee_pos - nuovo_target_pos)
        else:
            self.prev_dist = distanza

        obs = self._get_obs()
        done = (self.current_step >= self.max_steps)
        
        # Aggiungiamo le info per eventuale log (opzionale, ma utile per debug)
        info = info_reward

        return obs, reward, done, False, info

    def _compute_reward(self, action, distanza, is_close, target_completato):
        # 1. reward_distance: Penality basata sulla distanza dal target
        reward_near_weight = 30.0 
        reward_distance = -distanza * reward_near_weight
        
        # 2. Premio/Penalità per il Progresso (Avvicinamento relativo)
        progresso = self.prev_dist - distanza
        reward_distance += progresso * 30.0
        
        # ELIMINATO: self.prev_dist = distanza (viene già fatto in step() in modo più sicuro!)
        
        # 3. reward_control: Penalità per azioni ampie
        reward_control_weight = 1.5 
        reward_control = -np.sum(np.square(action)) * reward_control_weight
        
        # 4. Bonus specifici
        bonus = 0.00
        if is_close:
            bonus += 20.0
        if target_completato:
            bonus += 100.0

        total_reward = reward_distance + reward_control + bonus
        
        info = {
            "reward_distance": reward_distance,
            "reward_control": reward_control,
            "reward_bonus": bonus
        }

        return total_reward, info

# ==========================================================
# 3. L'AGENTE (Il "Cervello" PPO)
# ==========================================
if __name__ == "__main__":


    # =========================
    # VISUALIZZAZIONE 
    # =========================

    env = PandaReachEnv("ik_sceneBYEBYE.xml")

    model = PPO.load("modelli_Panda_BYEBYE/reach_modeRandBYEBYE_29999952_steps", env=env)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        obs, _ = env.reset()

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            time.sleep(0.01)
            if done:
                obs, _ = env.reset()