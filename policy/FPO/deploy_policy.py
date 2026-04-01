# import packages and module here
import numpy as np
from .dp_model import DP
import yaml

def encode_obs(observation):  # Post-Process Observation
    head_cam = (np.moveaxis(observation["observation"]["head_camera"]["rgb"], -1, 0) / 255)
    left_cam = (np.moveaxis(observation["observation"]["left_camera"]["rgb"], -1, 0) / 255)
    right_cam = (np.moveaxis(observation["observation"]["right_camera"]["rgb"], -1, 0) / 255)
    obs = dict(
        head_cam=head_cam,
        left_cam=left_cam,
        right_cam=right_cam,
    )
    obs["agent_pos"] = observation["joint_action"]["vector"]
    return obs


def get_model(usr_args):  # from deploy_policy.yml and eval.sh (overrides)
    # Your_Model = None
    # ...
    ckpt_file = f"./policy/FPO/checkpoints/{usr_args['task_name']}-{usr_args['ckpt_setting']}-{usr_args['expert_data_num']}-{usr_args['seed']}/{usr_args['checkpoint_num']}.ckpt"
    action_dim = usr_args['left_arm_dim'] + usr_args['right_arm_dim'] + 2 # 2 gripper
    
    load_config_path = f'./policy/FPO/diffusion_policy/config/robot_fpo_{action_dim}.yaml'
    with open(load_config_path, "r", encoding="utf-8") as f:
        model_training_config = yaml.safe_load(f)
    
    n_obs_steps = model_training_config['n_obs_steps']
    n_action_steps = model_training_config['n_action_steps']

    return DP(ckpt_file, n_obs_steps=n_obs_steps, n_action_steps=n_action_steps)
    # return Your_Model  # return your policy model


def eval(TASK_ENV, model, observation):
    """
    All the function interfaces below are just examples
    You can modify them according to your implementation
    But we strongly recommend keeping the code logic unchanged
    """
    obs = encode_obs(observation)  # Post-Process Observation
    instruction = TASK_ENV.get_instruction()

    if len(
            model.obs_cache
    ) == 0:  # Force an update of the observation at the first frame to avoid an empty observation window, `obs_cache` here can be modified
        model.update_obs(obs)

    actions = model.get_action()  # Get Action according to observation chunk

    for action in actions:  # Execute each step of the action
        # see for https://robotwin-platform.github.io/doc/control-robot.md more details
        TASK_ENV.take_action(action, action_type='qpos') # joint control: [left_arm_joints + left_gripper + right_arm_joints + right_gripper]
        # TASK_ENV.take_action(action, action_type='ee') # endpose control: [left_end_effector_pose (xyz + quaternion) + left_gripper + right_end_effector_pose + right_gripper]
        # TASK_ENV.take_action(action, action_type='delta_ee') # delta endpose control: [left_end_effector_delta (xyz + quaternion) + left_gripper + right_end_effector_delta + right_gripper]
        observation = TASK_ENV.get_obs()
        obs = encode_obs(observation)
        model.update_obs(obs)  # Update Observation, `update_obs` here can be modified


def reset_model(model):  
    # Clean the model cache at the beginning of every evaluation episode, such as the observation window
    # pass
    model.reset_obs()
