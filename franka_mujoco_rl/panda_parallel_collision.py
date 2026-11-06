import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
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
        
        # ==========================================
        # DEFINIZIONE DEL GIOCO 
        # ==========================================
        # Azioni: 7 valori continui per i 7 motori del Panda
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Osservazioni: 16 valori totali
        # Giunti(7) + Target(3) + End-Effector(3) = 13
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(23,), dtype=np.float32)

        # Variabili di gestione dell'episodio
        self.max_steps = 750
        self.current_step = 0
        
        # Variabili per tracciare il progresso (le usavi in _compute_reward)
        self.prev_dist = 0.0

        # Variabile per il viewer
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
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
        # Velocità
        velocita = self.data.qvel[:7].copy()
        # Posizione dell'ostacolo
        ostacolo = self.data.geom_xpos[self.obs_geom_id].copy()
        
        # Unione di tutto in un array di 16 elementi
        osservazione = np.concatenate([giunti, target, punta, ostacolo, velocita])
        
        return osservazione.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1

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
        if collision:
            terminated = False 
        elif self.current_step >= self.max_steps:
            terminated = True
            
        return obs, reward, terminated, False, {}

    # def _compute_reward(self, action, collision):
    #     # Prendiamo le posizioni attuali
    #     target_pos = self.data.xpos[self.box_id]
    #     ee_pos = self.data.site_xpos[self.site_id]

    #     # Calcolo Distanze
    #     distanza = np.linalg.norm(ee_pos - target_pos)
    #     # Reward di Distanza
    #     reward = -distanza * 50.0
    #     # Progresso (Bonus se si è avvicinato rispetto al passo precedente)
    #     progresso = self.prev_dist - distanza
    #     reward += progresso * 60.0
    #     self.prev_dist = distanza

    #     #perdita di punti in caso di collisione
    #     if collision:
    #         reward -= 500


    #     # Bonus di Successo 
    #     if distanza < 0.1:
    #         reward += 50
    #     if distanza < 0.08:
    #         reward += 75 # Bonus per il successo
    #     if distanza < 0.004:
    #         print("Distanza Buona!")
    #         reward += 200

    #     # Super-Bonus di successo
    #     if distanza < 0.002:
    #         print("Sei molto vicino!!")
    #         reward +=300
    #     if distanza < 0.001 :
    #         print("TARGET COLPITO PERFETTAMENTE")
    #         reward += 1500
        
    #     # Fluidità (Penalità per azioni troppo brusche)
    #     reward -= 0.1 * np.sum(np.square(action))

    #     return reward, False

    def _compute_reward(self, action, collision):
        ee_pos = self.data.site_xpos[self.site_id]
        target_pos = self.data.xpos[self.box_id]
        distanza = np.linalg.norm(ee_pos - target_pos)

        # 1. Distanza (meno ripida per evitare salti enormi di reward)
        reward = -distanza * 5.0

        # 2. Penalità Collisione
        if collision:
            reward -= 50.0

        # 3. Controllo Successo
        successo = False
        if distanza < 0.05:  # Soglia realistica per dire "preso!"
            reward += 100.0
            successo = True

        # 4. Fluidità (penalizza azioni troppo grandi/brusche)
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
    model = PPO.load("panda_reach_8M_2", env=envs)
    #model = PPO("MlpPolicy", envs, verbose=1, tensorboard_log="./tensorboard_logs/")
    model.learn(total_timesteps=2000000)
    model.save("panda_reach_10M_2")
    envs.close()

    # =========================
    # VISUALIZZAZIONE 1
    # =========================

    env = PandaReachEnv("ik_scene.xml")

    model = PPO.load("panda_reach_10M_2", env=env)

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer

        model.learn(total_timesteps=5000)

    model.save("panda_reach_10M_visualizzato")

    # env = PandaReachEnv(model_path="ik_scene.xml")

    # print("Apertura finestra grafica in corso...")
    
    # # Apriamo il visualizzatore PRIMA di far partire PPO
    # with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
    #     env.viewer = viewer  # Diamo il viewer all'ambiente così fa il sync nello step()
        
    #     print("Inizializzazione del modello PPO...")
    #     model1 = PPO.load("panda_reach_mujoco_v1", env=env, tensorboard_log="./tensorboard_logs/")
        
    #     print("Inizio allenamento LIVE!")
    #     # Ho messo 300k passi perché 100k sono pochi per imparare bene a raggiungere il target
    #     model1.learn(total_timesteps=2000)
        
    #     model1.save("panda_reach_mujoco_v2")
    #     print("Modello salvato con successo!")
