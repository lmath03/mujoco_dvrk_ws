import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(rank):
    def _init():
        env = PandaReachEnv("ik_scene.xml")
        env.reset(seed=rank)
        return env
    return _init

class PandaReachEnv(gym.Env):
    def __init__(self, model_path="ik_scene.xml"):
        super().__init__()
        
        # ==========================================
        #  MOTORE FISICO 
        # ==========================================
        # Carichiamo il progetto (model) e lo stato (data) direttamente in RAM
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        
        # Salviamo gli ID degli oggetti per interrogarli velocemente in seguito
        self.site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper")
        self.box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_box")
        # Salviamo l'ID della geometria dell'ostacolo
        self.obs_geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "obs_geom")
        self.obstacle_mocap_id = self.model.body('obstacle_box').mocapid[0]     # Recuperiamo l'ID mocap associato al corpo
        
        # ==========================================
        # DEFINIZIONE DEL GIOCO 
        # ==========================================
        # Azioni: 7 valori continui per i 7 motori del Panda
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Osservazioni: 16 valori totali
        # Giunti(7) + Target(3) + End-Effector(3) = 13
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(24,), dtype=np.float32)

        # Variabili di gestione dell'episodio
        self.max_steps = 2000
        self.current_step = 0
        
        # Variabili per tracciare il progresso (le usavi in _compute_reward)
        self.prev_dist = 0.0

        # Variabile per il viewer
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0

        self.obstacle_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 1. RESET FISICO: Usiamo il keyframe "custom_home"
        # Questo rimette all'istante tutti i 7 motori nella posizione iniziale.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "custom_home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Aggiungiamo un piccolo disturbo casuale (rumore) ai giunti
        noise = self.np_random.uniform(low=-0.1, high=0.1, size=7)
        self.data.qpos[:7] += noise
        
        # 2. AGGIORNAMENTO: Diciamo a MuJoCo di ricalcolare la fisica dopo aver teletrasportato il robot
        mujoco.mj_forward(self.model, self.data)
        
        # 3. LETTURA INIZIALE: Preparazione delle variabili per il calcolo del progresso
        target_pos = self.data.xpos[self.box_id]
        ee_pos = self.data.site_xpos[self.site_id]
        
        self.prev_dist = np.linalg.norm(ee_pos - target_pos)
        
        # Restituiamo la prima osservazione e il dizionario vuoto richiesto da Gymnasium
        return self._get_obs(), {}

    def _get_obs(self):
        # ATTENZIONE
        # utilizziamo .copy() alla fine delle letture per accedere alla memoria scritta in C++
        # senza creare un collegamento alla memoria in tempo reale che potrebbe limitare LR nel salvataggio di nuovi dati
        
        # Giunti (i primi 7 valori dell'array delle posizioni qpos)
        giunti = self.data.qpos[:7].copy()
        # Target (Coordinate X,Y,Z del cubo)
        target = self.data.xpos[self.box_id].copy()
        # Punta (Coordinate X,Y,Z del sito 'gripper')
        punta = self.data.site_xpos[self.site_id].copy()
        # Velocità panda
        velocita = self.data.qvel[:7].copy()
        # Posizione dell'ostacolo
        ostacolo = self.data.geom_xpos[self.obs_geom_id].copy()
        # Velocità ostacolo
        vel_obst =  self.obstacle_velocity

        relative_target = target - punta
        relative_obs = ostacolo - punta

        relative_velocity = -vel_obst

        distance_obs = np.linalg.norm(relative_obs) + 1e-6

        closing_velocity = np.dot(
            relative_obs / distance_obs,
            relative_velocity
        )

        # Unione di tutto in un array di 16 elementi
        osservazione = np.concatenate([
            giunti,                    # 7
            relative_target,           # 3
            relative_obs,              # 3
            velocita,                  # 7
            vel_obst,                  # 3
            np.array([closing_velocity])  # 1
        ])

        return osservazione.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1

        # Tempo della simulazione
        sim_time = self.data.time

        # Definizione parametri dell'ampiezza
        dx = 0.35        #ampiezza
        vdx = 0.0       # velocità dell'ostacolo
        centro_y = 5.0  #punto centrale del percorso

        #a aggiornamento pos ostacolo
        new_pos_y = centro_y + dx * math.sin(vdx * sim_time)
        self.data.mocap_pos[self.obstacle_mocap_id][1] = new_pos_y

        #salviamo la velocità analiticamente per poi usarla nel get_obs
        vel_y = dx * vdx * math.cos(vdx * sim_time)
        self.obstacle_velocity = np.array([0.0, vel_y, 0.0])

        # Definiamo quanto il robot può muoversi al massimo in uno step (es. 0.05 radianti)
        max_delta = 0.04
        action_delta = action * max_delta

        # Calcoliamo la nuova posizione partendo da dove siamo ora
        current_qpos = self.data.qpos[:7].copy()
        target_qpos = current_qpos + action_delta

        # Applichiamo i limiti fisici per sicurezza
        low_limit = self.model.jnt_range[:7, 0]
        high_limit = self.model.jnt_range[:7, 1]
        
        # APPLICAZIONE AZIONE
        # Mappiamo l'azione (-1, 1) del RL sui limiti reali dei giunti del robot.
        # MuJoCo userà i suoi attuatori interni per muovere il robot verso queste posizioni.
        self.data.ctrl[:7] = np.clip(target_qpos, low_limit, high_limit)
        
        # SIMULAZIONE FISICA
        # Facciamo avanzare la fisica
        # Possiamo ripetere mj_step più volte (es. 5) per rendere la simulazione più stabile.
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)

        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.sync()
            
        # LETTURA STATO
        obs = self._get_obs()

        # contollo collisioni
        collision = False
        # cicliamo su tutti i punti di contatto fisici attuali
        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if contact.geom1 == self. obs_geom_id or contact.geom2 == self.obs_geom_id:
                collision = True
                break
        
        # CALCOLO REWARD
        reward, _  = self._compute_reward(action, collision)
        
        # CONTROLLO FINE EPISODIO
        terminated = False
        if self.current_step >= self.max_steps:
            terminated = True
        else:
            terminated = False

        return obs, reward, terminated, False, {}

    def _compute_reward(self, action, collision):
        ee_pos = self.data.site_xpos[self.site_id]
        target_pos = self.data.xpos[self.box_id]
        distanza = np.linalg.norm(ee_pos - target_pos)

        obstacle_pos = self.data.geom_xpos[self.obs_geom_id]
        dist_obs = np.linalg.norm(ee_pos - obstacle_pos)

        # 1. Distanza (meno ripida per evitare salti enormi di reward)
        reward = -distanza * 5.0

        # 2. Penalità per progresso 
        progresso = self.prev_dist - distanza
        reward += progresso * 30.0
        self.prev_dist = distanza

        safe_distance = 0.18
        if dist_obs < safe_distance:
            reward -= (safe_distance - dist_obs) * 40.0

        # 3. Penalità Collisione
        if collision:
            reward -= 20.0

        # 4. Controllo Successo
        successo = False
        if distanza < 0.05:  # Soglia realistica per dire "preso!"
            reward += 100.0
            successo = True

        # 5. Fluidità (penalizza azioni troppo grandi/brusche)
        reward -= 0.1 * np.sum(np.square(action))

        return reward, successo
       
# ==========================================================
# 3. L'AGENTE (Il "Cervello" PPO)
# ==========================================
if __name__ == "__main__":
    # =========================
    # TRAINING PARALLELO
    # =========================

    num_cpu = 5  # o quanti core vuoi
    envs = SubprocVecEnv([make_env(i) for i in range(num_cpu)])
    model = PPO.load("reach_mode_BASE2", env=envs)
    #model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log="./tensorboard_logs/")
    model.learn(total_timesteps=5000000)
    model.save("reach_mode_BASE3")
    envs.close()

    # =========================
    # VISUALIZZAZIONE 1
    # =========================

    env = PandaReachEnv("ik_scene.xml")

    model = PPO.load("reach_mode_BASE3", env=env)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer
        obs, _ = env.reset()

        while viewer.is_running():
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, _, _ = env.step(action)

            time.sleep(0.01)
            if done:
                obs, _ = env.reset()

