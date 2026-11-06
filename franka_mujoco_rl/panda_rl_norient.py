import gymnasium as gym
from gymnasium import spaces
import numpy as np
import mujoco
import mujoco.viewer
import time
from stable_baselines3 import PPO

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
        
        # ==========================================
        # DEFINIZIONE DEL GIOCO 
        # ==========================================
        # Azioni: 7 valori continui per i 7 motori del Panda
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        
        # Osservazioni: 16 valori totali
        # Giunti(7) + Target(3) + End-Effector(3) + Vettore Z della punta(3) = 16
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(13,), dtype=np.float32)

        # Variabili di gestione dell'episodio
        self.max_steps = 500
        self.current_step = 0
        
        # Variabili per tracciare il progresso (le usavi in _compute_reward)
        self.prev_dist = 0.0
        #self.prev_orient_dist = 0.0

        # Variabile per il viewer
        self.viewer = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        
        # 1. RESET FISICO: Usiamo il keyframe "custom_home"
        # Questo rimette all'istante tutti i 7 motori nella posizione iniziale.
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "custom_home")
        mujoco.mj_resetDataKeyframe(self.model, self.data, key_id)
        
        # 2. AGGIORNAMENTO: Diciamo a MuJoCo di ricalcolare la fisica dopo aver teletrasportato il robot
        mujoco.mj_forward(self.model, self.data)
        
        # 3. LETTURA INIZIALE: Preparazione delle variabili per il calcolo del progresso
        target_pos = self.data.xpos[self.box_id]
        ee_pos = self.data.site_xpos[self.site_id]
        
        self.prev_dist = np.linalg.norm(ee_pos - target_pos)
        
        # Estraiamo il vettore Z dall'orientamento del gripper
        # site_xmat è una lista piatta di 9 numeri, la convertiamo in matrice 3x3 e prendiamo l'ultima colonna
        # tip_z_vector = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
        # target_z = np.array([0, 0, -1]) # Vogliamo che punti verso il basso
        
        #self.prev_orient_dist = np.linalg.norm(tip_z_vector - target_z)
        
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
        # Orientamento (Vettore Z della punta)
        #orientamento = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2].copy()
        
        # Unione di tutto in un array di 16 elementi
        osservazione = np.concatenate([giunti, target, punta])
        
        return osservazione.astype(np.float32)
    
    def step(self, action):
        self.current_step += 1
        
        # APPLICAZIONE AZIONE
        # Mappiamo l'azione (-1, 1) del RL sui limiti reali dei giunti del robot.
        # MuJoCo userà i suoi attuatori interni per muovere il robot verso queste posizioni.
        self.data.ctrl[:7] = action
        
        # SIMULAZIONE FISICA
        # Facciamo avanzare la fisica
        # Possiamo ripetere mj_step più volte (es. 5) per rendere la simulazione più stabile.
        for _ in range(5):
            mujoco.mj_step(self.model, self.data)
        
        if hasattr(self, 'viewer') and self.viewer is not None:
            self.viewer.sync()
            
        # LETTURA STATO
        obs = self._get_obs()
        
        # CALCOLO REWARD (Usiamo la tua logica di Coppelia)
        reward = self._compute_reward(action)
        
        # CONTROLLO FINE EPISODIO
        terminated = False
        if self.current_step >= self.max_steps:
            terminated = True
            
        return obs, reward, terminated, False, {}

    def _compute_reward(self, action):
        # Prendiamo le posizioni attuali
        target_pos = self.data.xpos[self.box_id]
        ee_pos = self.data.site_xpos[self.site_id]
        # tip_z_vector = self.data.site_xmat[self.site_id].reshape(3, 3)[:, 2]
        # target_z = np.array([0, 0, -1])

        # Calcolo Distanze
        distanza = np.linalg.norm(ee_pos - target_pos)
        #orient_dist = np.linalg.norm(tip_z_vector - target_z)
        # Reward di Distanza
        reward = -distanza * 50.0
        # Progresso (Bonus se si è avvicinato rispetto al passo precedente)
        progresso = self.prev_dist - distanza
        reward += progresso * 30.0
        self.prev_dist = distanza

        # Orientamento
        # reward -= orient_dist * 10.0
        # progresso_orient = self.prev_orient_dist - orient_dist
        # reward += progresso_orient * 50.0
        # self.prev_orient_dist = orient_dist

        # Bonus di Successo 
        if distanza < 0.1:
            reward += 50
        if distanza < 0.05:
            reward += 75 # Bonus per il successo
            print("Distanza Buona!")
        if distanza < 0.004:
            print("Distanza Buona!")
            reward += 200

        # if orient_dist < 0.26:  #se l'orientamento varia di 0.26 (circa 15 gradi)
        #     reward += 20
        # if orient_dist < 0.122: #7 gradi
        #     reward += 75

        # Super-Bonus di successo
        if distanza < 0.007:
            print("Sei molto vicino!!")
            reward +=300
        if distanza < 0.001:
            print("TARGET COLPITO PERFETTAMENTE")
            reward += 1500
        
        # Fluidità (Penalità per azioni troppo brusche)
        reward -= 0.1 * np.sum(np.square(action))

        return reward
    
# ==========================================================
# 3. L'AGENTE (Il "Cervello" PPO)
# ==========================================
if __name__ == "__main__":
    env = PandaReachEnv(model_path="ik_scene.xml")

    print("Apertura finestra grafica in corso...")
    
    # Apriamo il visualizzatore PRIMA di far partire PPO
    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        env.viewer = viewer  # Diamo il viewer all'ambiente così fa il sync nello step()
        
        print("Inizializzazione del modello PPO...")
        model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard_logs/")
        
        print("Inizio allenamento LIVE! Mettiti comodo e guarda il robot...")
        # Ho messo 300k passi perché 100k sono pochi per imparare bene a raggiungere il target
        model.learn(total_timesteps=500000)
        
        model.save("panda_reach_mujoco_v1")
        print("Modello salvato con successo!")

