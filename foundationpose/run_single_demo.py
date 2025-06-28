import os
import argparse
import logging
import numpy as np
import trimesh
import cv2
import imageio

from estimater import *
from datareader import *
import nvdiffrast.torch as dr

def set_logging_format():
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

def set_seed(seed):
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    code_dir = os.path.dirname(os.path.realpath(__file__))

    parser.add_argument('--test_scene_dir', type=str, default=f'{code_dir}/demo_data/mustard0')
    parser.add_argument('--est_refine_iter', type=int, default=5)
    parser.add_argument('--debug', type=int, default=2)
    parser.add_argument('--debug_dir', type=str, default=f'{code_dir}/debug')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    debug = args.debug
    debug_dir = args.debug_dir
    os.makedirs(debug_dir, exist_ok=True)

    reader = YcbineoatReader(video_dir=args.test_scene_dir, shorter_side=None, zfar=np.inf)

    scorer = ScorePredictor()
    refiner = PoseRefinePredictor()
    glctx = dr.RasterizeCudaContext()

    mask_root = os.path.join(args.test_scene_dir, 'masks')
    mesh_root = os.path.join(args.test_scene_dir, 'mesh')

    pose_output_dir = os.path.join(debug_dir, 'poses')
    vis_output_dir = os.path.join(debug_dir, 'vis_all')
    os.makedirs(pose_output_dir, exist_ok=True)
    os.makedirs(vis_output_dir, exist_ok=True)

    frame_ids = sorted([f.split('.')[0] for f in os.listdir(os.path.join(args.test_scene_dir, 'rgb')) if f.endswith('.png')])
    object_names = sorted([
        d for d in os.listdir(mask_root)
        if os.path.isdir(os.path.join(mask_root, d)) and
        os.path.exists(os.path.join(mesh_root, d + '.obj'))
    ])

    print(f'Found {len(object_names)} object folders with matching meshes.')

    for frame_str in frame_ids:
        frame_idx = int(frame_str)
        logging.info(f'Processing frame {frame_idx}')
        color = reader.get_color(frame_idx)
        depth = reader.get_depth(frame_idx)

        for obj_name in object_names:
            mask_file = os.path.join(mask_root, obj_name, f'{frame_str}.png')
            mesh_file = os.path.join(mesh_root, obj_name + '.obj')

            if not os.path.exists(mesh_file) or not os.path.exists(mask_file):
                logging.warning(f"Skipping {obj_name} (missing mesh or mask for frame {frame_str})")
                continue

            logging.info(f'  Estimating pose for object: {obj_name}')

            mesh = trimesh.load(mesh_file, process=False)
            mesh.apply_scale(0.001)  # Ensure mesh is in meters

            mask = cv2.imread(mask_file, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                logging.warning(f"Could not read mask file: {mask_file}")
                continue
            mask = cv2.resize(mask, (reader.W, reader.H), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 0)

            to_origin, extents = trimesh.bounds.oriented_bounds(mesh)
            bbox = np.stack([-extents / 2, extents / 2], axis=0).reshape(2, 3)

            est = FoundationPose(
                model_pts=mesh.vertices,
                model_normals=mesh.vertex_normals,
                mesh=mesh,
                scorer=scorer,
                refiner=refiner,
                debug_dir=debug_dir,
                debug=debug,
                glctx=glctx
            )

            try:
                pose = est.register(K=reader.K, rgb=color, depth=depth, ob_mask=mask, iteration=args.est_refine_iter)
                center_pose = pose @ np.linalg.inv(to_origin)

                pose_path = os.path.join(pose_output_dir, f'{frame_str}_{obj_name}.txt')
                np.savetxt(pose_path, pose.reshape(4, 4))

                vis = draw_posed_3d_box(reader.K, img=color.copy(), ob_in_cam=center_pose, bbox=bbox)
                vis = draw_xyz_axis(vis, ob_in_cam=center_pose, scale=0.1, K=reader.K, thickness=2)
                vis_path = os.path.join(vis_output_dir, f'{frame_str}_{obj_name}.png')
                imageio.imwrite(vis_path, vis)
            except Exception as e:
                logging.error(f"Pose estimation failed for {obj_name} in frame {frame_str}: {e}")

    logging.info('✅ All pose estimations completed.')
