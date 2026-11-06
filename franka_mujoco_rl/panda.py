import mujoco
import mujoco.viewer
import numpy as np
import time

model_path = "ik_scene.xml" 
model = mujoco.MjModel.from_xml_path(model_path)
data = mujoco.MjData(model)

print("Available sites in this model:")
for i in range(model.nsite):
    print(f"- {mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)}")


integration_dt = 0.05 # Slightly smaller dt for smoother IK visualization
damping = 1e-4

ee_name = "gripper" 
try:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_name)
    
except:
    print(f"Warning: Site '{ee_name}' not found. Check your panda.xml for the correct site name.")
    site_id = 0 

box_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_box")

def run_ik():
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Reset to 'home' keyframe to start near the cube
        key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "custom_home")
        mujoco.mj_resetDataKeyframe(model, data, key_id)
        
        # Initialize kinematics before the loop starts
        mujoco.mj_forward(model, data)
        
        while viewer.is_running():
            step_start = time.time()

            # Get current position of the box and the gripper
            target_pos = data.xpos[box_id]
            current_pos = data.site_xpos[site_id]
            
            # Calculate the error (difference)
            error = target_pos - current_pos
            
            # Stop moving if we are extremely close to prevent jitter
            if np.linalg.norm(error) > 0.005:
                # Get the Position Jacobian
                jacp = np.zeros((3, model.nv)) 
                mujoco.mj_jacSite(model, data, jacp, None, site_id)
                
                # Solve for joint velocities using Damped Least Squares
                # dq = J^T * inv(J*J^T + damping*I) * error
                dq = jacp.T @ np.linalg.solve(jacp @ jacp.T + damping * np.eye(3), error)

                # Integrate the positions purely kinematically
                mujoco.mj_integratePos(model, data.qpos, dq, integration_dt)

            # CRITICAL FIX: Use mj_forward instead of mj_step.
            # mj_step runs the physics engine (gravity, collisions).
            # mj_forward just updates the spatial positions based on our new qpos.
            mujoco.mj_forward(model, data)
            viewer.sync()

            # Real-time sync
            time_until_next_step = integration_dt - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    run_ik()