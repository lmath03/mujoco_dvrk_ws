#!/usr/bin/env python3
import gymnasium as gym
import rospy
import numpy as np
from gymnasium import spaces
from stable_baselines3 import PPO

from std_srvs.srv import Empty #per il reset temporale

from scipy.spatial.transform import Rotation as R #per la conversione da quaternione a matrice di rotazione

# Import messaggi ROS
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped

# ==========================================================
# 1. INTERFACCIA ROBOT (I "Sensi" e i "Muscoli")
# ==========================================
class PandaRobotInterface:
    def __init__(self):
        self.ee_pos = np.zeros(3)           #memorizziamo l end effector del robot
        self.current_joints = np.zeros(7)   #memorizziamo i giunti del robot

        self.joint_pub = rospy.Publisher('/joint_states', JointState, queue_size=1)
        rospy.Subscriber("/franka_pub/pose_link_8", PoseStamped, self._ee_cb)

        rospy.wait_for_service('/reset_robot', timeout=5.0)
        self.reset_robot_srv = rospy.ServiceProxy('/reset_robot', Empty)
        
    def _ee_cb(self, msg):  #funzione che ascolta messaggio msg da coppelia e lo traduce in numeri
        x = msg.pose.position.x      #prendiamo la posizione dell'end effector
        y = msg.pose.position.y
        z = msg.pose.position.z
        self.ee_pos = np.array([x, y, z])   #utilizziamo un array numpy con la posizione per l'IA

        #prendiamo il quaternione per l'orientamento del tip
        qx = msg.pose.orientation.x
        qy = msg.pose.orientation.y
        qz = msg.pose.orientation.z
        qw = msg.pose.orientation.w

        #conversione in matrice di rotazione
        rot = R.from_quat([qx, qy, qz, qw])

        #calcolo dove punta l'asse Z del tip rispetto al mondo
        self.tip_z_vector = rot.apply([0, 0, 1])

    def move(self, joints):
        msg = JointState()  #creazione del contenitore con il messaggio

        #diamo un nome ai giunti (deve essere uguale a quelli di coppelia)
        msg.name = ['Franka_joint1','Franka_joint2','Franka_joint3','Franka_joint4',
                    'Franka_joint5','Franka_joint6','Franka_joint7']
        
        #inseriamo i valori numerici degli angoli
        msg.position = joints.tolist()

        #spediamo il messaggio sul canale ROS
        self.joint_pub.publish(msg)

    def reset_sim(self):
        self.reset_robot_srv()

# ==========================================================
# 2. AMBIENTE GYMNASIUM (Le "Regole del Gioco")
# ==========================================
class PandaReachEnv(gym.Env):
    def __init__(self, robot_interface):
        super(PandaReachEnv, self).__init__()   #attiva gymnasium che le regole standard (poi le altre le si forniscono)

        self.prev_dist = 0.0 #per il progresso distanza
        self.prev_orient_dist = 0.0 # Per il progresso orientamento
        #colleghiamo il robot all'ambiente
        self.robot = robot_interface
        
        # Azione: Spostamento dei 7 giunti. Definiamo cosa può FARE il robot 
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(7,), dtype=np.float32)
        # Osservazione: Giunti(7) + Target(3) + EE(3) + Tip_Z_Vector(3) = 16. Definiamo cosa può vedere il robot
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)

        # definiamo i parametri dell'obbiettivo
        self.target_pos = np.array([0.6, 0.0, 0.3])

        #definiione della posizione corrente dei giunti (iniziale)
        self.current_joints = np.zeros(7)
        self.max_steps = 200        #quanti passi fa prima del reset
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)    #comando gymnasium per resettare e tenere traccia dei parametri interni
        self.current_step = 0       #reset del cronometro dei passi dell'episodio. Ogni tentativo parte dal passo zero
        
        # RIAVVIA LA SCENA DI COPPELIASIM!
        self.robot.reset_sim()
        
        # Qui definiremo la posa iniziale "ad arco" (7 angoli in radianti)
        self.current_joints = np.array([0.0, -0.5, 0.0, -2.0, 0.0, 1.5, 0.7])
        
        # Forziamo il robot a stare fermo lì all'inizio del turno
        self.robot.move(self.current_joints)

        #Piccolo sleep per ricevere la ee_pose corretta via ROS
        rospy.sleep(0.1) 
        self.prev_dist = np.linalg.norm(self.robot.ee_pos - self.target_pos)
        
        #Inizializza l'orientamento precedente al reset
        target_z = np.array([0, 0, -1])
        self.prev_orient_dist = np.linalg.norm(self.robot.tip_z_vector - target_z)
        
        #restituiamo la prima cosa che il robot vede e un dizionario vuoti
        return self._get_obs(), {}  #restituisce la "vista" attuale del robot

    def step(self, action):
        # 1. Aumentiamo il contatore passi
        self.current_step += 1      #aumento dell contatore di passi
        
        # 2. Traduzione azione: effettua piccoli passi3. AGGIUNGI QUESTO: 
        step_size = 0.05
        self.current_joints = self.current_joints + (action * step_size)    #muoviamo il giunto di soli 0.025 radianti, rendendo il robot fluido

        self.current_joints = np.clip(self.current_joints, -2.8, 2.8)       #limitiamo 
        # 3. Movimento del robot
        self.robot.move(self.current_joints)    #robot si muove aggiornando nuovi giunti. Diamo il comando a coppeliaSIm
        rospy.sleep(0.05)       #aspettiamo che proceda

        # 4. Osservazione: si vede dove è finito il braccio
        obs =self._get_obs()

        # 5. Reward
        reward = self._compute_reward(action)

        # 6. Fine partita
        terminated = self._check_done()

        return obs, reward, terminated, False, {}


    def _get_obs(self):
        giunti = self.current_joints    #prendiamo i 7 angoli dei giunti
        target = self.target_pos        #prendiamo le coordinate dell'obbiettivo (dove deve andare)
        punta = self.robot.ee_pos       #prendiamo le coordinate reali della punta del simulatore
        orientamento = self.robot.tip_z_vector #3 elementi dell'orientamento del tip

        osservazione = np.concatenate([giunti, target, punta, orientamento])  #unione di tutto in un unico vettore di 16 elementi

        return osservazione.astype(np.float32)      #restituisco il vettore in formato float32 (standard per l'IA)

    def _compute_reward(self, action):

        # PARTE DISTANZA -------------------------------------
        distanza = np.linalg.norm(self.robot.ee_pos - self.target_pos)  #utilizziamo la distanza euclidea, per calcolare la distanza tra obbiettivo e coordinate tip
       
        reward = -distanza * 27 #maggiore è la distanza, maggiore sarà la penalità
        
        #progresso distanza
        progresso = self.prev_dist - distanza 
        reward += progresso * 100  # Se progresso è positivo (si avvicina), il reward sale   
        
        self.prev_dist = distanza # Aggiorna per il prossimo passo

        # PARTE ORIENTAMENTO -------------------------------------
        target_z = np.array([0, 0, -1])

        orient_dist = np.linalg.norm(self.robot.tip_z_vector - target_z)    #calcolo la "distanza" dall'orientamento verticale corretto

        reward -= orient_dist *15
        
        #progresso orientamento
        progresso_orient = self.prev_orient_dist - orient_dist
        reward += progresso_orient * 50

        self.prev_orient_dist = orient_dist

        if distanza < 0.05:
            reward += 50
        if distanza < 0.01:
            reward += 75 # Bonus per il successo
        if distanza < 0.004:
            print("Distanza Buona!")
            reward += 200

        if orient_dist < 0.26:  #se l'orientamento varia di 0.26 (circa 15 gradi)
            reward += 20
        if orient_dist < 0.122: #7 gradi
            reward += 75

        if distanza < 0.007 and orient_dist < 0.15:
            print("Sei molto vicino!!")
            reward +=300
        
        if distanza < 0.001 and orient_dist < 0.05:
            print("TARGET COLPITO PERFETTAMENTE")
            reward += 1500

        # FLUIDITA -------------------------------------
        reward -= 0.1 * np.sum(np.square(action))

        return reward

    def _check_done(self):
        distanza = np.linalg.norm(self.robot.ee_pos - self.target_pos)
        
        # Finisce se: tempo scaduto OPPURE target raggiunto
        if self.current_step >= self.max_steps:
            return True
        return False

# ==========================================================
# 3. L'AGENTE (Il "Cervello" PPO)
# ==========================================
def train_agent(env):
    # ----------------- TRAINING DA ALLENAMENTO PRECEDENTE -----------------
    
    print("Caricamento del modello precedente...")
    # SB3 gestisce internamente la memoria e le reti neurali
    model = PPO.load("panda_model_perfect_reach_v2.zip", env=env, tensorboard_log="./tensorboard_logs/")
    model.learn(total_timesteps=100000, reset_num_timesteps=False)
    model.save("panda_model_perfect_reach_v3")
    return model

    #----------------- TRAINING ALLENAMENTO NUOVO -----------------
    # print("Creazione di un nuovo modello")
    # model = PPO(
    #     "MlpPolicy", 
    #     env, 
    #     verbose=1, 
    #     tensorboard_log="./tensorboard_logs/",
    #     learning_rate=0.0003 # Velocità di apprendimento standard
    # )
    # print("Inizio addestramento completo...")
    # model.learn(total_timesteps=100000)

    # model.save("panda_model_perfect_reach_v1")
    # print("Modello salvato come: panda_model_perfect_reach_v1")
    
    # return model

# ==========================================================
# 4. ESECUZIONE (Il "Main")
# ==========================================
if __name__ == "__main__":
    rospy.init_node('panda_rl_modular')
    
    # Istanziamo i moduli
    robot = PandaRobotInterface()
    env = PandaReachEnv(robot)
    
    print("Inizio Fase 1: Solo Reach...")
    train_agent(env)