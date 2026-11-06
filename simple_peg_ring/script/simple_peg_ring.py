#!/usr/bin/env python3
import numpy

numpy.set_printoptions(suppress=True)

import rospy

import time
import crtk
import dvrk
import math
import numpy
import PyKDL
import sys

import tf

from geometry_msgs.msg import PoseStamped, Pose, Point, Quaternion, PoseArray
try:
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    client = RemoteAPIClient()
    sim = client.getObject('sim')
    print("Connessione a Coppelia riuscita!")
except ImportError:
    # Se la libreria zmq non c'è, definiamo sim come None per non far crashare tutto
    sim = None
    print("Libreria Coppelia non trovata, il parenting non funzionerà.")

class SimplePegRing():
    def __init__(self,ral):

        self.pose_name_ = rospy.get_param('~pose_name','pose_name_default')
        self.rings_topic_name_ = rospy.get_param('~ring_poses','rings/poses')
        self.pegs_topic_name_ = rospy.get_param('~pegs_poses','pegs/poses')
        
        self.pose_pub_ = rospy.Publisher(self.pose_name_, PoseStamped, queue_size=1)

        self.rings_sub_ = rospy.Subscriber(self.rings_topic_name_, PoseArray, self.rings_callback) #riceve le posizioni degli anelli nello spazio
        self.pegs_sub_ = rospy.Subscriber(self.pegs_topic_name_, PoseArray, self.pegs_callback) #riceve la posizione dei pioli nello spazio
        

        self.rings_poses_ = [] # PoseArray()
        self.pegs_poses_ = [] # PoseArray()

        self.ral = ral

        self.psm1_ = dvrk.psm(ral = self.ral,       #cra gli oggetti software per controllare i bracci fisici/simulati Braccio nome:PSM1 e 
                            arm_name = 'PSM1',      #ha un intervallo di aggiornamento
                            expected_interval = 0.01)

        #self.psm2_ = dvrk.psm(ral = self.ral,       #cra gli oggetti software per controllare i bracci fisici/simulati Braccio nome:PSM2 e 
        #                    arm_name = 'PSM2',      #ha un intervallo di aggiornamento
        #                    expected_interval = 0.01)
        
        self.expected_interval_ = 0.01

        self.psm1_base_to_world_ = self.getMatrix([-1.538, -0.138, 0.799,-0.000, 0.000, 0.574, 0.819]) #È una matrice fissa che descrive dove si trova la base del robot rispetto al centro del mondo di CoppeliaSim. È fondamentale perché il simulatore vede l'anello in un punto, ma il robot deve sapere dove si trova quel punto rispetto alla propria "spalla".
        #self.psm2_base_to_world_ = self.getMatrix([-0.657, -1.400, -0.799, -0.000, -0.000, -0.819, 0.574])
        
#---------------------------------------------------------------------------------------------------------
    # [blue,cyan,red,yellow,purple,green]
    def rings_callback(self, msg):          #
        self.rings_poses_ = msg.poses       #
        # print(len(self.rings_poses_))     # Queste funzioni vengono eseguite automaticamente ogni
                                            # Volta che CoppeliaSim invia una nuova posizione. Esse aggiornano le liste self.rings_poses_ e
    # [left, right]                         # self.pegs_poses_ con le coordinate aggiornate.
    def pegs_callback(self, msg):           #
        self.pegs_poses_ = msg.poses        #
        # print(len(self.pegs_poses_))      #

    # x,y,z,qx,qy,qz,qw
    def getMatrix(self,pose):               #Prende una posizione (X, Y, Z) e un orientamento (quaternione) e li trasforma in una matrice di rototraslazione 4×4. Questo è il formato matematico standard in robotica per calcolare i movimenti.
        trasl_matrix = tf.transformations.translation_matrix( [pose[0], pose[1], pose[2]])
        rot_matrix = tf.transformations.quaternion_matrix([pose[3],pose[4],pose[5],pose[6]])
        T_matrix = numpy.dot(trasl_matrix, rot_matrix)
        return T_matrix

    def home(self, arm):
        self.ral.check_connections()
        print('starting enable')
        arm.enable(10) 
        
        print('starting home')
        arm.home(10)
        
        # AGGIUNGI QUESTE RIGHE: forza l'attesa finché il robot non è davvero pronto
        rospy.sleep(2.0) 
        if not arm.is_homed():
            print("Il braccio non ha completato l'homing!")
    
    def prepare_cartesian(self, arm):       #Configura il braccio in modo che lo strumento sia perfettamente verticale e pronto a muoversi nello spazio X, Y, Z (movimento cartesiano) invece di muovere solo i singoli giunti.
        print('preparing for cartesian motion')
    
        # 1. Tentativo di lettura sicura della posizione attuale
        try:
            # Prova a leggere il setpoint (posizione teorica)
            current_joints = arm.setpoint_jp()
        except:
            try:
                # Se fallisce, prova a leggere la posizione misurata (sensori reali)
                print("Setpoint non disponibile, leggo measured_jp...")
                current_joints = arm.measured_jp()
            except:
                # Se falliscono entrambi, il robot è probabilmente disconnesso o spento
                print("ERRORE: Impossibile leggere la posizione del robot. Verifica Coppelia!")
                return

        # 2. Crea il nuovo obiettivo (goal) basandoti sulla posizione trovata
        goal = numpy.copy(current_joints)
        
        # Imposta i valori standard per la cinematica cartesiana
        goal[0] = 0.0
        goal[1] = 0.0
        goal[2] = 0.12 # Distanza di sicurezza per uscire dalla cannula
        goal[3] = 0.0
        
        # Invia il comando di movimento
        arm.move_jp(goal).wait()

    def convertToPSM_Base(self, object_matrix, psm_base_matrix):        #Questa è la funzione che devi completare. Serve a rispondere alla domanda: "Se l'anello è nel punto A del mondo, dove deve andare la punta del mio robot per toccarlo?".
        inv_psm_matrix = numpy.linalg.inv(psm_base_matrix)
        T_psm_object = numpy.dot(inv_psm_matrix, object_matrix)
        return T_psm_object
#---------------------------------------------------------------------------------------------------------
    def test_kinematics(self):              #Un esempio di movimento Cartesiano. Dice al robot di andare in un punto specifico dello spazio usando coordinate metriche (metri). Include anche il comando jaw.close() per chiudere le pinze.       
        self.home(self.psm1_)
        self.prepare_cartesian(self.psm1_)

        initial_cartesian_position = PyKDL.Frame()
        initial_cartesian_position.p = self.psm1_.setpoint_cp().p
        initial_cartesian_position.M = self.psm1_.setpoint_cp().M
        goal = PyKDL.Frame()
        goal.p = self.psm1_.setpoint_cp().p
        goal.M = self.psm1_.setpoint_cp().M
        
        amplitude = 0.1 # 5 cm

         # first motion
        goal.p[0] =  initial_cartesian_position.p[0]  + amplitude
        goal.p[1] =  initial_cartesian_position.p[1]
        goal.p[2] =  initial_cartesian_position.p[2]
        
        self.psm1_.move_cp(goal).wait(is_busy = True)
        self.psm1_.jaw.close().wait() 
        
        # else :
        #     print('[ERROR] maybe run coppelia')

    def poseMsgToList(self, pose_msg):
        return [
            pose_msg.position.x,
            pose_msg.position.y,
            pose_msg.position.z,
            pose_msg.orientation.x,
            pose_msg.orientation.y,
            pose_msg.orientation.z,
            pose_msg.orientation.w
        ]

    def wait_for_setpoint_jp(self, arm, timeout=5.0):
        start = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            try:
                arm.setpoint_jp()
                return
            except RuntimeWarning:
                if rospy.Time.now().to_sec() - start > timeout:
                    raise RuntimeError('setpoint_jp non disponibile')
                rospy.sleep(0.1)

    def wait_for_robot(self, arm):
        print("Attendo PSM1...")
        while not rospy.is_shutdown():
            try:
                arm.measured_jp()
                print("PSM1 pronto")
                return
            except:
                rospy.sleep(0.1)

    def run_spiral_trajectory(self):      #Un esempio di movimento Joint (dei giunti). Muove i motori del robot seguendo un'equazione matematica per disegnare una spirale nell'aria.     
        print('starting spiral trajectory')
        # get current position
        initial_joint_position = numpy.copy(self.psm1_.setpoint_jp())
        print('testing spiral trajectory with initial joint positions: %s' % str(initial_joint_position))
        
        amplitude = math.radians(10.0)  # +/- 25 degrees for joint movement
        duration = 30  # seconds, adjust as needed
        samples = duration / self.expected_interval_
        
        # create a new goal starting with the current position
        goal_p = numpy.copy(initial_joint_position)
        goal_v = numpy.zeros(goal_p.size)
        start = time.time()

        sleep_rate = self.ral.create_rate(1.0 / self.expected_interval_)
        
        for i in range(int(samples)):
            # Compute angle for circular motion (spiral)
            angle = i * math.radians(360.0) / samples
            
            # Change in x, y, z to create a spiral trajectory
            # Assuming joint[0] and joint[1] are the ones for spiral, and joint[5] is adjusted for z height
            
            # For a spiral, modify the first two joints (0 and 1) to move in a circle
            goal_p[0] = initial_joint_position[0] + amplitude * math.cos(angle)
            goal_p[1] = initial_joint_position[1] + amplitude * math.sin(angle)
            
            # Gradual increase in z-axis (or joint[5]) for the spiral effect
            goal_p[3] = initial_joint_position[5] + 0.05 * i  # Adjust the z-axis increment (0.01 can be changed)

            # Set the velocity for smooth movement
            goal_v[0] = amplitude * -math.sin(angle)
            goal_v[1] = amplitude * math.cos(angle)
            goal_v[3] = 0.01  # Smooth increment for the z-axis

            # goal_p[2] =  0.11
            # goal_v[2] = 0.01

            goal_p[4] = 0
            goal_v[4] = 0
            goal_p[5] = 0
            goal_v[5] = 0

            # Maintain the rest of the joints at their initial positions (if they are not moving)
            for j in range(2, 5):
                goal_p[j] = initial_joint_position[j]
                goal_v[j] = 0

            # Send the new joint positions and velocities to the robot
            self.psm1_.servo_jp(goal_p, goal_v)
            
            # Sleep to maintain the specified rate
            sleep_rate.sleep()

        actual_duration = time.time() - start
        print('Spiral trajectory complete in %2.2f seconds (expected %2.2f)' % (actual_duration, duration))

    def move_to_pose(self, arm, T_psm_object, x_offset, y_offset, z_offset=0.0):
        goal = PyKDL.Frame()
        goal.p[0] = T_psm_object[0,3] + x_offset
        goal.p[1] = T_psm_object[1,3] + y_offset
        goal.p[2] = T_psm_object[2,3] + z_offset
        
        # Attesa robusta per la matrice di rotazione
        timeout = 5.0
        start_t = rospy.Time.now().to_sec()
        while not rospy.is_shutdown():
            try:
                goal.M = arm.measured_cp().M
                break
            except:
                if rospy.Time.now().to_sec() - start_t > timeout:
                    print("TIMEOUT: Impossibile ottenere measured_cp")
                    return
                rospy.sleep(0.1)
        
        arm.move_cp(goal).wait()

    def pick_and_place1(self, arm, ring_pose, peg_pose, psm_base, ring_name):
        
        ring = sim.getObject(':/' + ring_name)
        punta = sim.getObject(':/J3_dx_TOOL1')
        sim.resetDynamicObject(ring)

        ring_list = self.poseMsgToList(ring_pose)
        peg_list  = self.poseMsgToList(peg_pose)

        T_ring = self.getMatrix(ring_list)
        T_peg  = self.getMatrix(peg_list)

        T_ring_psm = self.convertToPSM_Base(T_ring, psm_base)
        T_peg_psm  = self.convertToPSM_Base(T_peg,  psm_base)

        # sopra anello
        rospy.sleep(2.0)
        self.move_to_pose(arm, T_ring_psm, x_offset=0.0, y_offset=0.0, z_offset=0.03)
        arm.jaw.open().wait() #apro la pinza
        rospy.sleep(2.0)

        #mi avvicino all'anello
        self.move_to_pose(arm, T_ring_psm, x_offset=0.0, y_offset=0.00, z_offset=0.024)
        rospy.sleep(2.0)

        #presa dell'oggetto
        arm.jaw.close().wait()   
        rospy.sleep(3.0)
        sim.setObjectParent(ring, punta, True) #aggancia anello
        rospy.sleep(1.0)

        # sopra piolo
        self.move_to_pose(arm, T_peg_psm, x_offset=0.0, y_offset=0.0, z_offset=0.05)
        rospy.sleep(2.0)

        #rotazione del wrist
        #joints = arm.measured_jp()
        #rospy.sleep(2.0)
        #joints[4] += 1.5
        #rospy.sleep(2.0)
        #arm.move_jp(joints).wait()
        #rospy.sleep(2.0)

        # mi avvicino al piolo
        self.move_to_pose(arm, T_peg_psm, x_offset= 0.007, y_offset= 0.0017, z_offset=0.03)
        rospy.sleep(1.0)

        #drop sul piolo
        sim.setObjectParent(ring, -1, True) #sgancia anello
        arm.jaw.open().wait()
        #sim.setObjectInt32Param(h_anello, 3004, 0)
        rospy.sleep(1.0)
        
        self.move_to_pose(arm, T_peg_psm, x_offset=0.0, y_offset=0.0, z_offset=0.05)
        rospy.sleep(2.0)

        #rotazione del wrist
        #joints = arm.measured_jp()
        #rospy.sleep(2.0)
        #joints[4] -= 1.5
        #rospy.sleep(2.0)
        #arm.move_jp(joints).wait()
        #rospy.sleep(5.0)

    def pick_and_place(self, arm, ring_pose, peg_pose, psm_base):

        ring_list = self.poseMsgToList(ring_pose)
        peg_list  = self.poseMsgToList(peg_pose)

        T_ring = self.getMatrix(ring_list)
        T_peg  = self.getMatrix(peg_list)

        T_ring_psm = self.convertToPSM_Base(T_ring, psm_base)
        T_peg_psm  = self.convertToPSM_Base(T_peg,  psm_base)

        # sopra anello
        rospy.sleep(1.0)
        self.move_to_pose(arm, T_ring_psm, z_offset=0.03)
        rospy.sleep(1.0)
        arm.jaw.open().wait() #apro la pinza
        rospy.sleep(1.0)

        #mi avvicino all'anello
        self.move_to_pose(arm, T_ring_psm, z_offset=0.02)
        rospy.sleep(1.0)

        #presa dell'oggetto
        sim.setObjectParent(h_anello, h_punta, True) #aggancia anello
        rospy.sleep(1.0)
        arm.jaw.close().wait()   
        rospy.sleep(1.0)

        # sopra piolo
        self.move_to_pose(arm, T_peg_psm, z_offset=0.05)
        rospy.sleep(1.0)

        #rotazione del wrist
        #joints = arm.measured_jp()
        #rospy.sleep(1.0)
        #joints[3] += 0.7854
        #rospy.sleep(1.0)
        #arm.move_jp(joints).wait()
        #rospy.sleep(1.0)

        # mi avvicino al piolo
        self.move_to_pose(arm, T_peg_psm, z_offset=0.015)
        rospy.sleep(1.0)

        #drop sul piolo
        arm.jaw.open().wait()
        #sim.setObjectParent(h_anello, -1, True) #sgancia anello
        #sim.setObjectInt32Param(h_anello, 3004, 0)
        rospy.sleep(1.0)

        #rotazione del wrist
        #joints = arm.measured_jp()
        #ospy.sleep(1.0)
        #joints[3] -= 0.7854
        #rospy.sleep(1.0)
        #arm.move_jp(joints).wait()
        #rospy.sleep(1.0)
        
        self.move_to_pose(arm, T_peg_psm, z_offset=0.05)
        rospy.sleep(1.0)



    def update(self):
        print ('UPDATE')
        # print(len(self.rings_poses_))



#------------------------------------------------------------------

def main():
    
    # Inizializza il nodo ROS e l'oggetto
    ral = crtk.ral('dvrk_simple_peg_ring') #inizializza il nodo ROS
    spr = SimplePegRing(ral) #crea l'oggetto con tutti ibracci

    #print("\n--- ANALISI ALIAS SCENA ---")
   # try:
    #    all_obj = sim.getObjectsInTree(sim.handle_scene)
    #    for o in all_obj:
    #        alias = sim.getObjectAlias(o)
    #        if 'ring' in alias.lower():
    #            print(f"Anello trovato: '{alias}'")
    #    print("---------------------------\n")
    #except Exception as e:
    #    print(f"Errore scansione: {e}")

    def task():

        # attendo pose da Coppelia
        rospy.sleep(2.0)

        arm = spr.psm1_
        base = spr.psm1_base_to_world_

        # ATTESA EXTRA: Fondamentale per evitare il RuntimeWarning
        print("Sincronizzazione con il controller...")
        rospy.sleep(3.0)

        
        spr.home(arm)
        spr.wait_for_robot(arm)
        spr.prepare_cartesian(arm)

        #try:
        #    posizione_iniziale_jp = numpy.copy(arm.setpoint_jp())
        #except RuntimeWarning:
        #    print("Setpoint non disponibile, uso measured_jp")
        #    posizione_iniziale_jp = numpy.copy(arm.measured_jp())

        #rospy.sleep(2.0)

        # solo 2 pioli
        pegA = spr.pegs_poses_[0]
        pegB = spr.pegs_poses_[1]

        # pick & place espliciti
        ring3 = spr.rings_poses_[3]
        spr.pick_and_place1(arm, ring3, pegB, base, 'ring7')
        #spr.pick_and_place(arm, ring3, pegB, base)

        ring4 = spr.rings_poses_[4]
        spr.pick_and_place1(arm, ring4, pegB, base, 'ring10')
        #spr.pick_and_place(arm, ring4, pegB, base)

        ring5 = spr.rings_poses_[5]
        spr.pick_and_place1(arm, ring5, pegB, base, 'ring11')
        #spr.pick_and_place(arm, ring5, pegB, base)

        #ritorno alla posizione iniziale
        rospy.sleep(2.0)
        print("Task concluso. Ritorno alla posizione iniziale...")
        spr.home(arm)
        spr.wait_for_robot(arm)
        spr.prepare_cartesian(arm)
        rospy.sleep(2.0)
        #arm.move_jp(posizione_iniziale_jp).wait()
        #rospy.sleep(2.0)
        print("Robot in posizione. Fine programma.")
        

    ral.spin_and_execute(task)
    #ral.spin_and_execute(simple_peg_ring.test_kinematics)
    

if __name__ == '__main__':
    main()
