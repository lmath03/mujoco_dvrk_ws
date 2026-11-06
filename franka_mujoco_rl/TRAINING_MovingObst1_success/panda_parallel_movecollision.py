import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
import math
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv

# Creazione dell'ambiente per il training parallelo (SubprocVecEnv richiede una funzione che crei un'istanza dell'ambiente per ogni processo)
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
        
        # Osservazioni: 52 valori totali   ---------- Definizione dello spazio
        # Giunti(7) + AngVelJoints(7) + VelObstacle(3) + TargetEefDirection(3) + Alert(8link * 4dati ciascuno = 32) = 52
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(52,), dtype=np.float32)

        # Variabili di gestione dell'episodio
        self.max_steps = 2000   # n step per episodio
        self.current_step = 0
        
        # Variabili per tracciare il progresso
        self.prev_dist = 0.0

        # Variabile per il viewer
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)        #serve a gymnasium
        self.current_step = 0

        self.obstacle_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        # 1. RESET FISICO: Usiamo il keyframe "custom_home"
        # Questo rimette all'istante tutti i 7 motori nella posizione iniziale.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "custom_home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)

        # Aggiungiamo un piccolo disturbo casuale (rumore) ai giunti, per renderlo randomizzato/causale
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

    def _get_obs(self):     # Restituzione delle osservazioni ( con _)
        
        # 1. Dati base

        # ATTENZIONE
        # utilizziamo .copy() alla fine delle letture per accedere alla memoria scritta in C++
        # senza creare un collegamento alla memoria in tempo reale che potrebbe limitare LR nel salvataggio di nuovi dati
        
        # Giunti (i primi 7 valori dell'array delle posizioni qpos)
        giunti = self.data.qpos[:7].copy()
        # Velocità Angolare dei giunti (i primi 7 valori dell'array delle velocità qvel)
        vel_angolare = self.data.qvel[:7].copy()
        # Posizione dell'ostacolo
        ostacolo_pos = self.data.geom_xpos[self.obs_geom_id].copy()
        # Posizione dell'obiettivo
        target_pos = self.data.xpos[self.box_id].copy()
        # Posizione della punta/eef
        punta_pos = self.data.site_xpos[self.site_id].copy()
        # Velocità dell'ostacolo
        vel_obst = self.obstacle_velocity.copy()

        # 2. Direzione verso l'obiettivo (rispetto al gripper)
        relative_target = target_pos - punta_pos

        # 3. Scansione di TUTTI i link per il pericolo
        # Definisce la lista dei nomi di tutti i segmenti fisici del Panda e crea una lista vuota (danger_data) per accumulare i dati dul pericolo di ciascuno
        nomi_link = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]
        danger_data = []

        for nome in nomi_link:    # Ciclo for per tutti i link  
            link_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nome)     # Estrazione posizione cartesiana (XYZ) del link
            pos_link = self.data.xpos[link_id].copy()
            
            # Velocità Lineare del link (conversione da angolare a m/s)
            vel_full = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, link_id, vel_full, 0) # Velocità effettiva del link
            vel_lineare = vel_full[3:]      # Velocità lineare del link (scartando i primi 3 dati della rotazione)
            
            # Calcolo Closing Velocity
            rel_obs = ostacolo_pos - pos_link                   # Vettore distanza link-ostacolo
            dist = np.linalg.norm(rel_obs) + 1e-6               # Distanza geometrica link-ostacolo (1e-6 per evitare il crash matematico del pc)
            rel_vel = vel_lineare - self.obstacle_velocity      # Velocità relativa tra il link e l'ostacolo che si muove (differenza di velocità)
            
            # Prodotto scalare per la velocità di avvicinamento: calcola la proiezione della velocità sulla direzione dell'ostacolo
            # se positivo => ostacolo si sta avvicinando. se positivo => pericolo si allontana
            cv = np.dot(rel_obs / dist, rel_vel)
            
            # Aggiungiamo all'osservazione: Vettore Posizione (3) + Closing Vel (1)
            danger_data.append(rel_obs)
            danger_data.append(np.array([cv]))

        # 4. Unione finale (52 elementi)
        osservazione = np.concatenate([
            giunti,             # 7
            vel_angolare,       # 7
            vel_obst,            # 3
            relative_target,    # 3
            *danger_data        # 32 (8 link * 4 dati)
        ])

        return osservazione.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1      #Conteggio degli step

        # Tempo della simulazione
        sim_time = self.data.time

        # Definizione parametri dell'ampiezza
        dx = 0.7        #ampiezza
        vdx = 0.5       #velocità dell'ostacolo
        centro_y = 0.0  #punto centrale del percorso

        # Aggiornamento pos ostacolo
        new_pos_y = centro_y + dx * math.sin(vdx * sim_time)
        self.data.mocap_pos[self.obstacle_mocap_id][1] = new_pos_y

        # Salviamo la velocità analiticamente per poi usarla nel get_obs
        vel_y = dx * vdx * math.cos(vdx * sim_time) # Velocità sinusoidale in y dell'ostacolo
        self.obstacle_velocity = np.array([0.0, vel_y, 0.0])

        # Definiamo quanto il robot può muoversi al massimo in uno step (es. 0.05 radianti)
        max_delta = 0.05                            # rendiamo il movimento "teletrasporto" in modo incrementale e fluido
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
        for _ in range(5):                              # Avvia un ciclo che ripete l'istruzione successiva per 5 volte consecutive
            mujoco.mj_step(self.model, self.data)       # Funzione nativa di MuJoCo che fa avanzare il tempo e la fisica del mondo di un piccolissimo "passo"

        if hasattr(self, 'viewer') and self.viewer is not None:     # Controlla se nell'ambiente esiste la variabile viewer e se è attiva
            self.viewer.sync()                                      # Invia i dati fisici appena calcolati alla scheda grafica per aggiornare la finestra 3D
            
        # LETTURA STATO
        obs = self._get_obs()           # Il robot si è mosso di n (mj_step=5) passi. Mandiamo l'osservazione al robot

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

        # 1. Distanza 
        reward = -distanza * 10.0

        # 2. Penalità per progresso 
        progresso = self.prev_dist - distanza
        reward += progresso * 30.0
        self.prev_dist = distanza

        # 3. SICUREZZA MULTI-LINK (La parte importante)
        # Controlliamo il pericolo per i link più critici (es. Gomito e Mano)
        punti_critici = ["link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand"]
        max_penalty = 0
        
        for nome in punti_critici:          # Ciclo per valutare la sicurezza di ciascun pezzo del braccio 
            idx = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, nome)     # Lettura pos braccio
            pos_link = self.data.xpos[idx]                                          
            dist_link_obs = np.linalg.norm(pos_link - obstacle_pos)                 # Distanza tra link e ostacolo
            
            # Calcolo velocità di avvicinamento per questo link
            v_full = np.zeros(6)
            mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, idx, v_full, 0)
            v_rel = v_full[3:] - self.obstacle_velocity                                     # Velocità relativa - velocità di movimento dell ostacolo
            c_speed = np.dot(v_rel, (obstacle_pos - pos_link) / (dist_link_obs + 1e-6))     # Calcola la closingspeed trdimensionale (velocità di avvicinamento). Se c_speed>0 ostacolo verso link
            
            # Se il link è vicino e corre verso l'ostacolo, accumula penalità
            if dist_link_obs < 0.20 and c_speed > 0:
                penalty = (c_speed * 50.0) / (dist_link_obs + 0.05)
                max_penalty = max(max_penalty, penalty)
        
        reward -= max_penalty

        # 4. Penalità Collisione
        if collision:
            reward -= 20.0

        # 5. Controllo Successo
        successo = distanza < 0.05
        if successo:
            reward += 12.0 # Premio per ogni istante che RESTI vicino
            if distanza < 0.02:
                reward += 22.0 # Bonus precisione
                if distanza < 0.005:
                    reward += 50.0

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
    model = PPO.load("reach_mode_25M", env=envs)
    #model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log="./tensorboard_logs/")
    #model = PPO("MlpPolicy", envs, verbose=1)
    model.learn(total_timesteps=5000000)
    model.save("reach_mode_30M")

    model = PPO.load("reach_mode_30M", env=envs)
    model.learn(total_timesteps=5000000)
    model.save("reach_mode_35M")

    # model = PPO.load("reach_mode_10M", env=envs)
    # model.learn(total_timesteps=5000000)
    # model.save("reach_mode_15M")

    # model = PPO.load("reach_mode_15M", env=envs)
    # model.learn(total_timesteps=5000000)
    # model.save("reach_mode_20M")

    envs.close()

    # =========================
    # VISUALIZZAZIONE 
    # =========================

    # env = PandaReachEnv("ik_scene.xml")

    # model = PPO.load("reach_mode_BASE", env=env)

    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    #     env.viewer = viewer
    #     obs, _ = env.reset()

    #     while viewer.is_running():
    #         action, _ = model.predict(obs, deterministic=True)
    #         obs, reward, done, _, _ = env.step(action)

    #         time.sleep(0.01)
    #         if done:
    #             obs, _ = env.reset()
