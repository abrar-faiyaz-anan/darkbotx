import sys
import math
import os
import xml.etree.ElementTree as ET

import trimesh

import numpy as np

import torch
from torch import Tensor

import genesis as gs
from genesis import Scene
from genesis.utils.geom import trans_quat_to_T, transform_quat_by_quat, quat_to_R, quat_to_xyz

from rsl_rl.env import VecEnv

from tensordict import TensorDict

from typing import Any, cast, override

from config import SJoint


def get_urdf_meshes(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()
    base_dir = os.path.dirname(urdf_path)
    mesh_paths = []
    
    for mesh in root.findall(".//collision/geometry/mesh") + root.findall(".//visual/geometry/mesh"):
        filename = mesh.get('filename')
        if filename:
            if filename.startswith("package://"):
                filename = filename.replace("package://", "")
            full_path = os.path.join(base_dir, filename)
            scale_str = mesh.get('scale', '1.0 1.0 1.0')
            scale = np.array([float(s) for s in scale_str.split()])
            mesh_paths.append((full_path, scale))
            
    return list({(p, tuple(s)) for p, s in mesh_paths})

def resample_boundary(points: np.ndarray, num_points: int) -> np.ndarray:
    """Resamples a 2D polygon boundary to have exactly `num_points` points.
    
    The points are assumed to form a closed loop.
    """
    if len(points) == 0:
        return np.zeros((num_points, points.shape[1]))
        
    # Ensure it's closed to calculate full perimeter
    closed_pts = np.vstack([points, points[0]])
    # Compute segment lengths
    dists = np.sqrt(np.sum(np.diff(closed_pts, axis=0)**2, axis=1))
    cum_dists = np.concatenate(([0.0], np.cumsum(dists)))
    total_dist = cum_dists[-1]
    
    if total_dist < 1e-6:
        # Fallback if points are coincident
        indices = np.linspace(0, len(points) - 1, num_points)
        return points[np.round(indices).astype(int)]
        
    # Query points at equal intervals
    query_dists = np.linspace(0, total_dist, num_points, endpoint=False)
    
    # Interpolate x and y coordinates
    resampled = np.zeros((num_points, points.shape[1]))
    for d in range(points.shape[1]):
        resampled[:, d] = np.interp(query_dists, cum_dists, closed_pts[:, d])
        
    return resampled


class GraspEnv(VecEnv):
    def __init__(self, CONFIG: dict[str, Any]) -> None :
        # self.num_envs: int = CONFIG.get("num_envs", 1)
        self.cfg: dict[str, Any] | object = CONFIG
        self.num_envs: int = self.cfg.get("num_envs", 1)
        self.num_actions: int = self.cfg.get("num_actions", 2)
        self.max_episode_length: int | Tensor = cast(int, self.cfg.get("max_episode_length", 300))
        self.episode_length_buf: torch.Tensor = torch.zeros(self.num_envs, dtype=torch.long, device=gs.device)
        # pyrefly: ignore [bad-assignment]
        self.device = gs.device
        self.scene = gs.Scene(
            viewer_options=gs.options.ViewerOptions(
                res = self.cfg.get("image_resolution", (1280, 960)),
                camera_pos=(.3, -0.1, 0.3),
                camera_lookat=(0.1, -.1, 0.1),
                camera_fov=55,
            ),
            rigid_options=gs.options.RigidOptions(
                noslip_iterations=5,
            ),
            show_viewer=self.cfg.get("show_viewer", False),
            profiling_options=gs.options.ProfilingOptions(
                show_FPS=False,
            ),
        )
        self.scene.add_entity(gs.morphs.Plane())
        self.robot = Manipulator(self.scene, self.cfg)
        
        config_obj_type = self.cfg.get("object_type", "random")
        self.object_configs = self.cfg.get("object_configs", {})
        self.object_names = list(self.object_configs.keys())
        
        # Precompute local 2D periphery points resampled to exactly 16 points for each config
        num_points = self.cfg.get("num_periphery_points", 16)
        local_peripheries = {}
        
        for name, obj_kwargs in self.object_configs.items():
            is_urdf = "file" in obj_kwargs
            if is_urdf:
                urdf_path = cast(str, obj_kwargs.get("file"))
                if not os.path.isabs(urdf_path):
                    urdf_path = os.path.join(os.path.dirname(__file__), "..", urdf_path)
                meshes_info = get_urdf_meshes(urdf_path)
                if meshes_info:
                    mesh_file, scale = meshes_info[0]
                    t_mesh = cast(trimesh.Trimesh, trimesh.load(mesh_file))
                    t_mesh.apply_scale(scale)
                    path2d = t_mesh.projected(normal=[0, 0, 1])
                    boundary_points = path2d.discrete[0][:-1]
                    resampled = resample_boundary(boundary_points, num_points)
                    pts_3d = np.zeros((num_points, 3))
                    pts_3d[:, :2] = resampled
                    local_peripheries[name] = torch.tensor(pts_3d, dtype=torch.float32, device=gs.device)
                else:
                    local_peripheries[name] = torch.zeros((num_points, 3), device=gs.device)
            else:
                size: list[float] = cast(list[float], obj_kwargs.get("size", [0.03, 0.03, 0.03]))
                lx, ly = size[0], size[1]
                boundary_points = np.array([
                    [lx/2, ly/2],
                    [-lx/2, ly/2],
                    [-lx/2, -ly/2],
                    [lx/2, -ly/2]
                ])
                resampled = resample_boundary(boundary_points, num_points)
                pts_3d = np.zeros((num_points, 3))
                pts_3d[:, :2] = resampled
                local_peripheries[name] = torch.tensor(pts_3d, dtype=torch.float32, device=gs.device)
                
        # Now create list of morphs, one for each env
        spawn_pos = self.cfg.get("object_spawn_pos", (-0.0061, -0.0617, 0.015))
        self.spawn_pos = torch.tensor(spawn_pos, device=gs.device).expand(self.num_envs, -1)
        
        spawn_yaw = self.cfg.get("object_spawn_yaw", 0.0)
        self.spawn_yaw = torch.tensor(spawn_yaw, device=gs.device).expand(self.num_envs)
        
        morphs = []
        self.env_object_names = []
        local_peripheries_list = []
        
        for i in range(self.num_envs):
            if config_obj_type == "random":
                name = self.object_names[i % len(self.object_names)]
            else:
                name = config_obj_type
                
            self.env_object_names.append(name)
            obj_kwargs = self.object_configs[name]
            is_urdf = "file" in obj_kwargs
            if is_urdf:
                morph = gs.morphs.URDF(pos=spawn_pos, **obj_kwargs)
            else:
                morph = gs.morphs.Box(pos=spawn_pos, **obj_kwargs)
            morphs.append(morph)
            local_peripheries_list.append(local_peripheries[name])
            
        # Stack local periphery points so it's a [num_envs, num_points, 3] tensor
        self._local_periphery = torch.stack(local_peripheries_list).to(gs.device)
        
        self.objects = self.scene.add_entity(
            morphs,
            material=gs.materials.Rigid(friction=2.0),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ColorTexture(
                    color=(1.0, 0.0, 0.0),
                ),
            ),
            vis_mode=self.cfg.get("vis_mode", None),
        )

        self._debug_draw: bool = self.cfg.get("debug_draw", False)
        self._debug_dashboard: bool = self.cfg.get("debug_dashboard", False)
        self.debugline: bool = self.cfg.get("debugline", False)
        self._use_mtrick: bool = self.cfg.get("use_mtrick", False)

        if self._use_mtrick:
            from mtrick import Tracker
            self._tracker = Tracker("live_periphery")

        self.scene.build(n_envs=self.num_envs, env_spacing=(1, 1))
        
        # Make the gripper joints much stiffer so they can apply enough force to hold the object
        _kp = self.robot._robot_entity.get_dofs_kp()
        _kp[:] = 500.0  # stiffer arm
        _kp[SJoint.GRIPPER_LEFT] = 5000.0
        _kp[SJoint.GRIPPER_RIGHT] = 5000.0
        self.robot._robot_entity.set_dofs_kp(_kp)
        
        _kv = self.robot._robot_entity.get_dofs_kv()
        _kv[:] = 20.0
        _kv[SJoint.GRIPPER_LEFT] = 100.0
        _kv[SJoint.GRIPPER_RIGHT] = 100.0
        self.robot._robot_entity.set_dofs_kv(_kv)

    
    @override
    def step(self, actions: Tensor) -> tuple[TensorDict, Any, Tensor, dict[str, Any]]:
        """
        actions: [num_envs, 2] tensor containing [angle, squeeze_width]
        """
        # 1. Rotation [RL]
        self.robot.apply_rotation_action(actions[:, 0])
        for _ in range(40):
            self.scene.step()

        # 2. Move to grasp position [Manual]
        self.robot.move_to_grasp_position()
        for _ in range(40):
            self.scene.step()

        # actions[:, 1] = torch.full_like(actions[:, 1], -1.0)

        # 3. Squeeze [RL]
        self.robot.apply_squeeze_action(actions[:, 1])
        for _ in range(40):
            self.scene.step()

        # >>>>>>>>>>>>>>>>>>>>> REWARD PHASE I >>>>>>>>>>>>>>>>>>>>
        # Compute alignment and width at the moment of the grasp (pre-lift)
        alignment_reward, periphery_width = self.compute_alignment_and_width_reward(actions)

        # 4. Lift up — multi-stage to prevent flinging the object [Manual]
        grasp_shoulder = 1.7453
        final_shoulder = 0.34
        lift_steps = 4
        for i in range(1, lift_steps + 1):
            t = i / lift_steps
            shoulder = grasp_shoulder + t * (final_shoulder - grasp_shoulder)
            self.robot.move_to_lift_position(shoulder_target=shoulder, gripper_action=actions[:, 1])
            for _ in range(25):
                self.scene.step()

        # >>>>>>>>>>>>>>>>>>>>> REWARD PHASE II >>>>>>>>>>>>>>>>>>>>
        reward = self.compute_final_reward(alignment_reward, periphery_width)


        self._last_reward = reward
        
        # update time
        self.episode_length_buf += 1
        
        if self.debugline:
            # Print the last value of the left gripper finger
            qpos = self.robot._robot_entity.get_qpos()
            left_gripper_val = qpos[:, SJoint.GRIPPER_LEFT]
            print(f"Last left gripper finger value: {left_gripper_val.tolist()}")
        
        # 4. End Episode
        # For a single-step approach, every step is a terminal evaluation state.
        done = torch.ones(self.num_envs, dtype=torch.bool, device=gs.device)

        info: dict[str, Any] = {
            # 0.0 because it's a true terminal state (success/failure), not a truncation timeout
            "time_outs": torch.zeros(self.num_envs, dtype=torch.float32, device=gs.device)
        }
        
        # RL frameworks expect the environment to auto-reset when 'done' is True.
        if done.any():
            self._reset_idx(done)
            self.episode_length_buf[done] = 0
        
        return self.get_observations(), reward, done, info

    def compute_alignment_and_width_reward(self, actions: Tensor) -> tuple[Tensor, Tensor]:
        # >>>>>>>>>>>>>>>>>>>>> ALIGNMENT REWARD >>>>>>>>>>>>>>>>>>>>
        quat = self.objects.get_quat()
        qpos = self.robot._robot_entity.get_qpos()
        gripper_angle = qpos[:, SJoint.WRIST_ROLL]
        
        # Get 2D periphery points in world frame relative to object center
        R = quat_to_R(quat)
        rotated_periphery = torch.einsum('nij,nkj->nki', R, self._local_periphery)
        pts = rotated_periphery[:, :, :2]  # [num_envs, num_points, 2]
        
        # Calculate wall vectors and their angles
        pts_next = torch.roll(pts, shifts=-1, dims=1)
        wall_vectors = pts_next - pts
        wall_angles = torch.atan2(wall_vectors[..., 1], wall_vectors[..., 0])  # [num_envs, num_points]
        
        # Compute difference between gripper angle and all wall angles (modulo pi)
        # Since gripper joint rotation is negated relative to the world yaw frame,
        # we add the angles (which is equivalent to subtracting the negated wall angles in joint space)
        angle_diff = (gripper_angle.unsqueeze(-1) + wall_angles) % math.pi
        angle_diff = torch.minimum(angle_diff, math.pi - angle_diff)  # [num_envs, num_points]
        
        # Find the minimum difference (closest wall alignment)
        min_angle_diff, closest_idx = torch.min(angle_diff, dim=-1)  # [num_envs]
        alignment_reward = torch.exp(-5.0 * min_angle_diff)

        # <<<<<<<<<<<<<<<<<<<<< ALIGNMENT REWARD <<<<<<<<<<<<<<<<<<<<


        # >>>>>>>>>>>>>>>>>>>>> WIDTH REWARD >>>>>>>>>>>>>>>>>>>>

        # Calculate the projection width of the 2D periphery at the gripper angle
        # Since gripper joint rotation is negated relative to the world yaw frame,
        # the world gripper yaw is -gripper_angle. The squeeze direction is parallel
        # to the world gripper yaw, which corresponds to [cos(g), -sin(g)].
        u = torch.stack([torch.cos(gripper_angle), -torch.sin(gripper_angle)], dim=-1)  # [num_envs, 2]
        projections = (pts * u.unsqueeze(1)).sum(dim=-1)  # [num_envs, num_points]
        periphery_width = projections.max(dim=-1).values - projections.min(dim=-1).values  # [num_envs]

        # <<<<<<<<<<<<<<<<<<<<< WIDTH REWARD <<<<<<<<<<<<<<<<<<<<

        # >>>>>>>>>>>>>>>>>>>>> LIFT REWARD >>>>>>>>>>>>>>>>>>>>
        return alignment_reward, periphery_width

    def compute_final_reward(
        self, 
        alignment_reward: Tensor, 
        periphery_width: Tensor
    ) -> Tensor:
        pos = self.objects.get_pos()
        success = (pos[:, 2] > 0.1).float()

        periphery_reward = torch.exp(-20.0 * periphery_width)

        # Combined reward: multiplicative composition of alignment/width plus success scaled by alignment
        reward = alignment_reward * periphery_reward + success * alignment_reward

        # Diagnostic printing (runs during manual test/evaluation where num_envs == 1)
        if self.num_envs == 1:
            quat = self.objects.get_quat()
            rpy = quat_to_xyz(quat, rpy=True)
            yaw = rpy[:, 2:3]
            qpos = self.robot._robot_entity.get_qpos()
            gripper_angle = qpos[:, SJoint.WRIST_ROLL]
            R = quat_to_R(quat)
            rotated_periphery = torch.einsum('nij,nkj->nki', R, self._local_periphery)
            pts = rotated_periphery[:, :, :2]
            pts_next = torch.roll(pts, shifts=-1, dims=1)
            wall_vectors = pts_next - pts
            wall_angles = torch.atan2(wall_vectors[..., 1], wall_vectors[..., 0])
            angle_diff = (gripper_angle.unsqueeze(-1) + wall_angles) % math.pi
            angle_diff = torch.minimum(angle_diff, math.pi - angle_diff)
            min_angle_diff, closest_idx = torch.min(angle_diff, dim=-1)

            g_angle = gripper_angle[0].item()
            g_angle_deg = math.degrees(g_angle)
            min_diff_deg = math.degrees(min_angle_diff[0].item())
            
            raw_wall_joint_angle = -wall_angles[0, closest_idx[0]].item()
            wrapped_wall_joint_angle = g_angle + ((raw_wall_joint_angle - g_angle + math.pi/2) % math.pi - math.pi/2)
            closest_wall_angle_deg = math.degrees(wrapped_wall_joint_angle)
            
            actual_yaw_deg = math.degrees(yaw[0, 0].item())
            spawn_yaw_deg = math.degrees(self.spawn_yaw[0].item())
            width_mm = periphery_width[0].item() * 1000.0
            height_m = pos[0, 2].item()
            
            print(f"[Diag] Gripper Angle: {g_angle_deg:.2f}°, Closest Wall Angle: {closest_wall_angle_deg:.2f}°")
            print(f"[Diag] Min Angle Diff: {min_diff_deg:.2f}°, Actual Object Yaw: {actual_yaw_deg:.2f}°, Spawn Yaw: {spawn_yaw_deg:.2f}°")
            print(f"[Diag] Periphery Width at Gripper Angle: {width_mm:.2f} mm")
            print(f"[Diag] Height: {height_m:.4f} m (Success: {success[0].item()})")
            print(f"[Diag] PeripheryWidth Reward: {periphery_reward[0].item():.4f})")
            print(f"[Diag] Alignment      Reward: {alignment_reward[0].item():.4f}")
            print(f"[Diag] Final Reward: {reward[0].item():.4f}")
            sys.stdout.flush()

        return reward

    def reset(self)-> TensorDict:
        self._reset_idx()
        return self.get_observations()

    def _reset_idx(self, envs_idx: None | Tensor = None)-> None:
        """Reset specified environments.

        Parameters
        ----------
        envs_idx : torch.Tensor or None
            Boolean mask of shape (num_envs,) or integer indices for selective reset, or None for full reset.
        """
        # Safely convert boolean masks to integer indices for Genesis API compatibility
        if envs_idx is not None and envs_idx.dtype == torch.bool:
            envs_idx = envs_idx.nonzero(as_tuple=False).flatten()
            if len(envs_idx) == 0:
                return

        self.robot.reset(envs_idx)


        # >>>>>>>>>>>>>>>>> RESET OBJECT >>>>>>>>>>>>>>>>>
        # Get random value of -90 to +90 (in radians) | pi = 180
        self.spawn_yaw = (
            torch.rand(self.num_envs) * 2 * math.pi - math.pi
        ) * 0.5

        q_yaw = torch.stack(
            [
                torch.cos(self.spawn_yaw / 2),
                torch.zeros(self.num_envs),
                torch.zeros(self.num_envs),
                torch.sin(self.spawn_yaw / 2),
            ],
            dim=-1,
        )

        q_downward = torch.tensor([0.0, 1.0, 0.0, 0.0]).expand(self.num_envs, -1)
        goal_yaw = transform_quat_by_quat(q_yaw, q_downward)
        
        if envs_idx is None:
            self.objects.set_pos(self.spawn_pos, skip_forward=True)
            self.objects.set_quat(goal_yaw, skip_forward=False)
        else:
            self.objects.set_pos(self.spawn_pos, envs_idx=envs_idx, skip_forward=True)
            self.objects.set_quat(goal_yaw, envs_idx=envs_idx, skip_forward=False)
        
        # <<<<<<<<<<<<<<<<<<<<< RESET OBJECT <<<<<<<<<<<<<<<<<<<<
        

    @override
    def get_observations(self)-> TensorDict:
        """
        Being called after resetting and at the end of a step, means that it will only output the inital position, quaternions, etc, keep this in mind.
        """
        pos = self.objects.get_pos() # [num_envs, 3]
        quat = self.objects.get_quat() # [num_envs, 4]
        # 1. Height
        height = pos[:, 2:3]
        
        # 2. 2D Periphery relative to object center
        R = quat_to_R(quat) # [num_envs, 3, 3]
        
        # Rotate corners
        rotated_periphery = torch.einsum('nij,nkj->nki', R, self._local_periphery)
        
        # Extract x,y and flatten
        num_points = self._local_periphery.shape[1]
        periphery_2d = rotated_periphery[:, :, :2].reshape(self.num_envs, num_points * 2)
        
        # 3. Gripper State (Wrist rotation and gripper left)
        qpos = self.robot._robot_entity.get_qpos()
        angle = qpos[:, SJoint.WRIST_ROLL:SJoint.WRIST_ROLL+1] 
        width = qpos[:, SJoint.GRIPPER_LEFT:SJoint.GRIPPER_LEFT+1] 
        gripper_state = torch.cat([angle, width], dim=-1)

        # 4. Object Yaw
        rpy = quat_to_xyz(quat, rpy=True)
        yaw = rpy[:, 2:3]
        
        # Draw the periphery for env_idx 0 in the 3D viewer
        if self._debug_draw:
            # world_periphery_0 = rotated_periphery[0] + pos[0]
            # self.draw_debug_periphery(world_periphery_0)
            self.draw_debug_periphery(rotated_periphery[0])

        # --- DASHBOARD ---
        if self._debug_dashboard:
            sys.stdout.write("\033[H\033[J") # Move to home, clear screen
            print("====== DARKBOTX ENVIRONMENT DASHBOARD ======")
            print(f"Reward:          {self._last_reward[0].item():.4f}")
            print(f"pos[0]:          {pos[0].tolist()}")
            print(f"quat[0]:         {quat[0].tolist()}")
            print(f"yaw[0]:          {yaw[0].item():.4f}")
            print(f"height[0]:       {height[0].item():.4f}")
            print(f"gripper_state:   {gripper_state[0].tolist()}")
            print("============================================")
            sys.stdout.flush()
        
        # --- MTRICK LIVE DASHBOARD UPDATES ---
        if self._use_mtrick:
            
            # Extract x,y for env_idx = 0
            local_2d = rotated_periphery[0, :, :2].tolist()
            
            # Close the polygons in the 2D plot
            if len(local_2d) > 0:
                local_2d.append(local_2d[0])
                
            self._tracker.log_trajectory(local_2d, [])

        return TensorDict({
            "object_2d_profile": periphery_2d,
            "current_gripper_state": gripper_state
        }, batch_size=[self.num_envs])


    def draw_debug_periphery(self, world_periphery: torch.Tensor, radius: float = 0.0035, color: tuple[float, float, float, float]=(0.0, 1.0, 0.0, 1.0)):
        """Draws the 2D periphery points in the Genesis viewer for debugging."""
        self.scene.clear_debug_objects()
        self.scene.draw_debug_spheres(poss=world_periphery.cpu().numpy(), radius=radius, color=color)

    def draw_debug_frame(self, pos: torch.Tensor, quat: torch.Tensor, axis_len: float = 0.05, radius: float = 0.002, env_idx: int = 0):
        """Draws a coordinate frame using Genesis's built-in draw_debug_frame."""
        # Convert position and quaternion to a 4x4 transformation matrix T
        pos_slice = pos[env_idx:env_idx+1]
        quat_slice = quat[env_idx:env_idx+1]
        T = trans_quat_to_T(pos_slice, quat_slice)[0]
        
        # Genesis's internal method requires CPU numpy arrays for drawing
        self.scene.draw_debug_frame(T.cpu().numpy(), axis_length=axis_len, axis_radius=radius)


class Manipulator:
    def __init__(self, scene: Scene, cfg: dict[str, Any]) -> None:
        morph = gs.morphs.URDF(file=cfg["path"], fixed=True)
        material = gs.materials.Rigid(friction=2.0)
        self._robot_entity = scene.add_entity(material=material, morph=morph)
        
        _init_qpos_deg = torch.tensor(cfg["rl_start_angles"], dtype=torch.float32, device=gs.device)
        self._init_qpos = torch.deg2rad(_init_qpos_deg)
        self._limits_cache: dict[int, tuple[float, float]] = {}

    def _get_limit(self, dof_idx: int) -> tuple[float, float]:
        if not self._limits_cache:
            for joint in self._robot_entity.joints:
                if len(joint.dofs_idx_local) > 0:
                    idx = joint.dofs_idx_local[0]
                    limit = joint.dofs_limit
                    if limit[0] is not None:
                        self._limits_cache[idx] = (float(limit[0][0]), float(limit[0][1]))
        return self._limits_cache[dof_idx]

    
    def reset(self, envs_idx: None | Tensor = None, skip_forward: bool =True) -> None:
        self._robot_entity.set_qpos(
            self._init_qpos,
            envs_idx=envs_idx,
            zero_velocity=True,
            skip_forward=skip_forward,
        )
        # Establish PD controller targets so all joints are actively held in place
        # Exclude GRIPPER_RIGHT (mimic joint) — its motion is driven by the mimic
        # constraint on GRIPPER_LEFT. Setting a controller target on it would fight
        # the constraint and prevent GRIPPER_LEFT from reaching its target.
        controlled_dofs = [j.value for j in SJoint if j != SJoint.GRIPPER_RIGHT]
        self._robot_entity.control_dofs_position(
            position=self._init_qpos[controlled_dofs].expand(self._robot_entity.get_qpos().shape[0], -1),
            dofs_idx_local=controlled_dofs,
            envs_idx=envs_idx,
        )

    def rotation_scale(self, action: Tensor, motor_idx: int) -> Tensor:
        action_clipped = torch.clamp(action, min=-1.0, max=1.0)
        
        lower_rad, upper_rad = self._get_limit(motor_idx)
        # Uncomment these if you want to restrict the limit to half of 180 i.e. (90 deg)
        # lower_rad = lower_rad / 2.0
        # upper_rad = upper_rad / 2.0
        return lower_rad + ((action_clipped + 1.0) / 2.0) * (upper_rad - lower_rad)
    
    def prismatic_scale(self, action: Tensor, motor_idx: int) -> Tensor:
        action_clipped = torch.clamp(action, min=-1.0, max=1.0)
        
        lower_m, upper_m = self._get_limit(motor_idx)
        return lower_m + ((action_clipped + 1.0) / 2.0) * (upper_m - lower_m)

    def apply_rotation_action(self, rotation: Tensor)-> None:
        """
        Apply the rotation action from the RL agent.
        """
        targets: dict[SJoint, int | float | Tensor]  = {
            SJoint.WRIST_ROLL: self.rotation_scale(rotation, SJoint.WRIST_ROLL),
        }
        self.control_motors(targets)

    def apply_squeeze_action(self, squeeze: Tensor)-> None:
        """
        Apply the squeeze action from the RL agent.
        """
        targets: dict[SJoint, int | float | Tensor]  = {
            SJoint.GRIPPER_LEFT: self.prismatic_scale(squeeze, SJoint.GRIPPER_LEFT),
        }
        self.control_motors(targets)

    def move_to_grasp_position(self):
        targets: dict[SJoint, int | float | Tensor] = {
            # HiwonderJoint.BASE: 0.0,
            SJoint.SHOULDER: 1.9722, # 113 degrees
            SJoint.ELBOW: 0.4223, # 24.2 degrees
            # HiwonderJoint.WRIST: 0.1745,
            # HiwonderJoint.WRIST_ROLL: 0.1745,
            SJoint.GRIPPER_LEFT: 0.02, # fully open (meters)
        }
        self.control_motors(targets)

    def move_to_lift_position(self, shoulder_target: float = 0.34, gripper_action: Tensor | None = None):
        targets: dict[SJoint, int | float | Tensor] = {
            SJoint.SHOULDER: shoulder_target,
        }
        if gripper_action is not None:
            targets[SJoint.GRIPPER_LEFT] = self.prismatic_scale(gripper_action, SJoint.GRIPPER_LEFT)
            
        self.control_motors(targets)


    def control_motors(self, targets: dict[SJoint, int | float | Tensor]) -> None:
        """
        Controls multiple motors at once without changing the target state of other motors.
        
        Args:
            targets: A dictionary mapping motor indices to target values (radians for revolute, meters for prismatic).
        """
        if not targets:
            return

        dof_indices = list(targets.keys())
        
        # Build a [num_envs, len(targets)] position tensor
        vals: list[Tensor] = []
        for idx in dof_indices:
            v: Tensor | float | int = targets[idx]
            if isinstance(v, Tensor):
                vals.append(v.unsqueeze(-1) if v.dim() == 1 else v)
            else:
                vals.append(torch.full((self._robot_entity.get_qpos().shape[0], 1), float(v), device=gs.device))
        
        position = torch.cat(vals, dim=-1)
        
        self._robot_entity.control_dofs_position(
            position=position,
            dofs_idx_local=dof_indices,
        )


