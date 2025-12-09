import os
import copy
import torch
import numpy as np
import random
import argparse

from humanoid import LEGGED_GYM_ROOT_DIR, LEGGED_GYM_ENVS_DIR


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_"):
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            element = class_to_dict(val)
        result[key] = element
    return result


def update_class_from_dict(obj, dict):
    for key, val in dict.items():
        attr = getattr(obj, key, None)
        if isinstance(attr, type):
            update_class_from_dict(attr, val)
        else:
            setattr(obj, key, val)
    return


def set_seed(seed):
    if seed == -1:
        seed = np.random.randint(0, 10000)
    print("Setting seed: {}".format(seed))

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SimParams:
    """MuJoCo simulation parameters (replaces gymapi.SimParams)"""
    def __init__(self):
        self.dt = 0.005
        self.substeps = 1
        self.up_axis = 1  # 0 is y, 1 is z
        self.gravity = [0., 0., -9.81]
        self.use_gpu_pipeline = False
        
        # PhysX parameters (mapped to MuJoCo equivalents)
        self.physx = self.PhysxParams()
    
    class PhysxParams:
        def __init__(self):
            self.num_threads = 10
            self.solver_type = 1  # 0: pgs, 1: tgs
            self.num_position_iterations = 4
            self.num_velocity_iterations = 0
            self.contact_offset = 0.01
            self.rest_offset = 0.0
            self.bounce_threshold_velocity = 0.5
            self.max_depenetration_velocity = 1.0
            self.max_gpu_contact_pairs = 2**23
            self.default_buffer_size_multiplier = 5
            self.contact_collection = 2
            self.use_gpu = False
            self.num_subscenes = 0


def parse_sim_params(args, cfg):
    """Parse simulation parameters for MuJoCo
    
    Args:
        args: Command line arguments
        cfg: Configuration dictionary
    
    Returns:
        SimParams object configured for MuJoCo
    """
    # Initialize sim params
    sim_params = SimParams()
    
    # Set GPU pipeline flag
    sim_params.use_gpu_pipeline = args.use_gpu_pipeline if hasattr(args, 'use_gpu_pipeline') else False
    sim_params.physx.use_gpu = args.use_gpu if hasattr(args, 'use_gpu') else False
    
    # If sim options are provided in cfg, parse them and update
    if "sim" in cfg:
        parse_sim_config(cfg["sim"], sim_params)
    
    # Override num_threads if passed on the command line
    if hasattr(args, 'num_threads') and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads
    
    return sim_params


def parse_sim_config(cfg_sim, sim_params):
    """Parse simulation configuration dictionary into SimParams
    
    Args:
        cfg_sim: Configuration dictionary for simulation
        sim_params: SimParams object to update
    """
    if hasattr(cfg_sim, 'dt'):
        sim_params.dt = cfg_sim.dt
    if hasattr(cfg_sim, 'substeps'):
        sim_params.substeps = cfg_sim.substeps
    if hasattr(cfg_sim, 'up_axis'):
        sim_params.up_axis = cfg_sim.up_axis
    if hasattr(cfg_sim, 'gravity'):
        sim_params.gravity = cfg_sim.gravity
    
    # Parse PhysX parameters
    if hasattr(cfg_sim, 'physx'):
        physx_cfg = cfg_sim.physx
        if hasattr(physx_cfg, 'num_threads'):
            sim_params.physx.num_threads = physx_cfg.num_threads
        if hasattr(physx_cfg, 'solver_type'):
            sim_params.physx.solver_type = physx_cfg.solver_type
        if hasattr(physx_cfg, 'num_position_iterations'):
            sim_params.physx.num_position_iterations = physx_cfg.num_position_iterations
        if hasattr(physx_cfg, 'num_velocity_iterations'):
            sim_params.physx.num_velocity_iterations = physx_cfg.num_velocity_iterations
        if hasattr(physx_cfg, 'contact_offset'):
            sim_params.physx.contact_offset = physx_cfg.contact_offset
        if hasattr(physx_cfg, 'rest_offset'):
            sim_params.physx.rest_offset = physx_cfg.rest_offset
        if hasattr(physx_cfg, 'bounce_threshold_velocity'):
            sim_params.physx.bounce_threshold_velocity = physx_cfg.bounce_threshold_velocity
        if hasattr(physx_cfg, 'max_depenetration_velocity'):
            sim_params.physx.max_depenetration_velocity = physx_cfg.max_depenetration_velocity


def get_load_path(root, load_run=-1, checkpoint=-1):
    try:
        runs = os.listdir(root)
        # TODO sort by date to handle change of month
        runs.sort()
        if "exported" in runs:
            runs.remove("exported")
        last_run = os.path.join(root, runs[-1])
    except:
        raise ValueError("No runs in this directory: " + root)
    if load_run == -1:
        load_run = last_run
    else:
        load_run = os.path.join(root, load_run)

    if checkpoint == -1:
        models = [file for file in os.listdir(load_run) if "model" in file]
        models.sort(key=lambda m: "{0:0>15}".format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)

    load_path = os.path.join(load_run, model)
    return load_path


def update_cfg_from_args(env_cfg, cfg_train, args):
    # seed
    if env_cfg is not None:
        # num envs
        if args.num_envs is not None:
            env_cfg.env.num_envs = args.num_envs
    if cfg_train is not None:
        if args.seed is not None:
            cfg_train.seed = args.seed
        # alg runner parameters
        if args.max_iterations is not None:
            cfg_train.runner.max_iterations = args.max_iterations
        if args.resume:
            cfg_train.runner.resume = args.resume
        if args.experiment_name is not None:
            cfg_train.runner.experiment_name = args.experiment_name
        if args.run_name is not None:
            cfg_train.runner.run_name = args.run_name
        if args.load_run is not None:
            cfg_train.runner.load_run = args.load_run
        if args.checkpoint is not None:
            cfg_train.runner.checkpoint = args.checkpoint

    return env_cfg, cfg_train


def get_args():
    """Parse command line arguments for MuJoCo training
    
    Returns:
        Namespace with parsed arguments
    """
    parser = argparse.ArgumentParser(description="RL Policy Training with MuJoCo")
    
    # Task and training parameters
    parser.add_argument(
        "--task",
        type=str,
        default="pai_ppo",
        help="Resume training or start testing from a checkpoint. Overrides config file if provided.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=False,
        help="Resume training from a checkpoint",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        help="Name of the experiment to run or load. Overrides config file if provided.",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        default="v1",
        help="Name of the run. Overrides config file if provided.",
    )
    parser.add_argument(
        "--load_run",
        type=str,
        help="Name of the run to load when resume=True. If -1: will load the last run. Overrides config file if provided.",
    )
    parser.add_argument(
        "--checkpoint",
        type=int,
        help="Saved model checkpoint number. If -1: will load the last checkpoint. Overrides config file if provided.",
    )
    
    # Display and device parameters
    parser.add_argument(
        "--headless",
        action="store_true",
        default=False,
        help="Force display off at all times",
    )
    parser.add_argument(
        "--horovod",
        action="store_true",
        default=False,
        help="Use horovod for multi-gpu training",
    )
    parser.add_argument(
        "--rl_device",
        type=str,
        default="cuda:0",
        help="Device used by the RL algorithm, (cpu, gpu, cuda:0, cuda:1 etc..)",
    )
    
    # Environment parameters
    parser.add_argument(
        "--num_envs",
        type=int,
        help="Number of environments to create. Overrides config file if provided.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed. Overrides config file if provided.",
    )
    parser.add_argument(
        "--max_iterations",
        type=int,
        help="Maximum number of training iterations. Overrides config file if provided.",
    )
    
    # Simulation device parameters (MuJoCo compatible)
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cuda:0",
        help="Physics simulation device (cpu or cuda:0, cuda:1, etc.)",
    )
    parser.add_argument(
        "--compute_device_id",
        type=int,
        default=0,
        help="Compute device ID",
    )
    parser.add_argument(
        "--graphics_device_id",
        type=int,
        default=0,
        help="Graphics device ID",
    )
    parser.add_argument(
        "--num_threads",
        type=int,
        default=0,
        help="Number of physics threads",
    )
    parser.add_argument(
        "--use_gpu",
        action="store_true",
        default=False,
        help="Use GPU for physics (tensor operations)",
    )
    parser.add_argument(
        "--use_gpu_pipeline",
        action="store_true",
        default=False,
        help="Use GPU pipeline",
    )
    parser.add_argument(
        "--subscenes",
        type=int,
        default=0,
        help="Number of subscenes",
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Name alignment for compatibility
    args.sim_device_id = args.compute_device_id
    args.sim_device_type = "cuda" if "cuda" in args.sim_device else "cpu"
    
    # Set physics engine to MuJoCo
    args.physics_engine = "mujoco"
    
    return args


def export_policy_as_jit(actor_critic, path):
    """Export policy as TorchScript
    
    Args:
        actor_critic: Actor-critic model
        path: Export directory path
    """
    os.makedirs(path, exist_ok=True)
    path = os.path.join(path, "policy_torch.pt")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    traced_script_module = torch.jit.script(model)
    traced_script_module.save(path)
    print(f"Policy exported as TorchScript to: {path}")


def export_policy_to_onnx(actor_critic, path, num_observations=705):
    """Export policy as ONNX format
    
    Args:
        actor_critic: Actor-critic model
        path: Export directory path
        num_observations: Number of observations (frame_stack * num_single_obs)
    """
    import os
    import copy
    import torch

    os.makedirs(path, exist_ok=True)
    model_path = os.path.join(path, "policy_onnx.onnx")
    model = copy.deepcopy(actor_critic.actor).to("cpu")
    model.eval()  # Set model to evaluation mode

    # Create example input
    # You need to modify according to the actual input shape of the model
    batch_size = 1  # Can be adjusted as needed
    dummy_input = torch.randn(batch_size, num_observations)

    # Export model as ONNX format
    torch.onnx.export(
        model,                      # Model
        dummy_input,                # Example input
        model_path,                 # Export path
        export_params=True,         # Export model parameters
        opset_version=11,           # ONNX opset version, can be adjusted as needed
        do_constant_folding=True,   # Perform constant folding optimization
        input_names=['input'],      # Input node name, can be adjusted as needed
        output_names=['output'],    # Output node name, can be adjusted as needed
    )
    print(f"Model exported as ONNX format, saved to: {model_path}")
