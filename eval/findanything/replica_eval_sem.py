import os
import os.path
import argparse
from typing import List
from typing import Dict

import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import torch

from pointclouds import Pointclouds

from conceptgraphs.conceptgraph.utils.eval import compute_confmatrix, compute_pred_gt_associations, compute_metrics
from conceptgraphs.conceptgraph.dataset.replica_constants import (
    REPLICA_EXISTING_CLASSES, REPLICA_CLASSES,
    REPLICA_SCENE_IDS, REPLICA_SCENE_IDS_,
)

import findanything_eval.vis_util as fa_vis
import findanything_eval.FindAythingReaderSem as fa_reader

'''
Script to evaluate semantic consistency scores (Accuracy fMIOU) for Replica Dataset.

Mostly based on the implementation of Concept-Graphs:
https://github.com/concept-graphs/concept-graphs/blob/main/conceptgraph/scripts/eval_replica_semseg.py#L175

'''

# Set to true if Submaps have been created using (externally provided) ground truth poses, to ensure correct alignment 
GT_POSES = False

# User Input:
list_of_sequences = ["office0", "office1", "office2", "office3", "office4", "room0", "room1", "room2"]

# Function to return 6DoF poses from OKVIS
def read_okvis_poses(path):

    # Read Trajectory
    trajectory = np.genfromtxt(path, delimiter=',', skip_header=True)

    # Orientations
    quats = trajectory[:,4:8]
    rots = R.from_quat(quats).as_matrix() # (N,3,3)

    # translations
    trans = trajectory[:,1:4]

    # Create batched rigid transformation matrices
    n = quats.shape[0]
    transformation_matrices = np.zeros((n, 4, 4))
    
    transformation_matrices[:, :3, :3] = rots
    transformation_matrices[:, :3, 3] = trans       # Set translation part
    transformation_matrices[:, 3, 3] = 1 # (N,4,4)

    return torch.from_numpy(transformation_matrices)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--replica_seq_path", type=str, required=True,
        help="Path to generated Replica sequence"
    )
    parser.add_argument(
        "--n_exclude", type=int, default=6, choices=[1, 4, 6],
        help='''Number of classes to exclude:
        1: exclude "other"
        4: exclude "other", "floor", "wall", "ceiling"
        6: exclude "other", "floor", "wall", "ceiling", "door", "window"
        ''',
    )
    parser.add_argument(
            "--device", type=str, default="cuda:0"
    )

    parser.add_argument(
        "--results_path", type=str, default="", 
        help="path to results base directory"
    )

    parser.add_argument(
        "--gt_path", type=str, default="./replica-semantic-saved-maps-gt",
        help="Path to GT Labeled Point Cloud"
    )

    parser.add_argument(
        "--query_path", type=str, default="./text_queries/",
        help="Path to Input Text Query CLIP embeddings"
    ) 

    parser.add_argument(
        "--debug", action='store_true',
        help="visualize 3d labeled point clouds"
    )

    return parser


def eval_replica(
    scene_id: str,
    class_names: List[str],
    class_feats: torch.Tensor,
    args: argparse.Namespace,
    class_all2existing: torch.Tensor,
    ignore_index=[],
    gt_class_only: bool = True, # only compute the conf matrix for the GT classes
):

    R_align = np.array([[1,0,0],[0,-1,0], [0,0,-1]])
    class2color = fa_vis.get_random_colors(len(class_names))

    '''Load the GT point cloud'''
    gt_pc_path = os.path.join(
        args.gt_path, scene_id , "Sequence_1", "saved-maps-gt"
    )

    gt_pose_path = os.path.join(
        args.gt_path, scene_id , "Sequence_1", "traj_w_c.txt"
    )

    gt_map = Pointclouds.load_pointcloud_from_h5(gt_pc_path)
    gt_poses = np.loadtxt(gt_pose_path)
    gt_poses = torch.from_numpy(gt_poses.reshape(-1, 4, 4)).float()

    gt_xyz = gt_map.points_padded[0]
    gt_color = gt_map.colors_padded[0]
    gt_embedding = gt_map.embeddings_padded[0]  # (N, num_class)
    gt_class = gt_embedding.argmax(dim=1)  # (N,)
    gt_class = class_all2existing[gt_class]  # (N,)
    assert gt_class.min() >= 0
    assert gt_class.max() < len(REPLICA_EXISTING_CLASSES)

    # transform pred_xyz and gt_xyz according to the first pose in gt_poses
    gt_xyz = gt_xyz @ gt_poses[0, :3, :3].t() + gt_poses[0, :3, 3]

    gt_pose_path2 = os.path.join(
        args.replica_seq_path, scene_id, "mav0/traj.txt"
    )
    gt_poses2 = np.loadtxt(gt_pose_path2)
    gt_poses2 = torch.from_numpy(gt_poses2.reshape(-1, 4, 4)).float()
    gt_alignment = gt_poses2[2].inverse()
    
    # Get the set of classes that are used for evaluation
    all_class_index = np.arange(len(class_names))
    ignore_index = np.asarray(ignore_index)
    if gt_class_only:
        # Only consider the classes that exist in the current scene
        existing_index = gt_class.unique().cpu().numpy()
        non_existing_index = np.setdiff1d(all_class_index, existing_index)
        ignore_index = np.append(ignore_index, non_existing_index)
        print(
            "Using only the classes that exists in GT of this scene: ",
            len(existing_index),
        )

    keep_index = np.setdiff1d(all_class_index, ignore_index)

    print(
        f"{len(keep_index)} classes remains. They are: ",
        [(i, class_names[i]) for i in keep_index],
    )
    
    '''Load the predicted point cloud'''

    results_path = os.path.join(
        args.results_path, scene_id
    )
    
    print(f"Loading results in {results_path}")
    objectMaps = fa_reader.FindAnythingReaderSem(results_path)
    okvisPoses = read_okvis_poses(os.path.join(results_path, 'okvis2-slam-final_trajectory.csv'))

    # Compute the CLIP similarity for the mapped objects and assign class to them
    object_ids, object_feats = objectMaps.get_stacked_values_torch("features")
    object_feats = object_feats.to(args.device)
    class_feats = torch.from_numpy(np.stack(class_feats)).to(args.device)
    object_feats = object_feats / object_feats.norm(dim=-1, keepdim=True) # (num_objects, D)
    object_class_sim = object_feats @ class_feats.T # (num_objects, num_classes)
    
    # suppress the logits to -inf that are not in torch.from_numpy(keep_class_index)
    object_class_sim[:, ignore_index] = -1e10
    object_class = object_class_sim.argmax(dim=-1) # (num_objects,)
    n_objects = len(object_class)
    print(f"Found {n_objects} objects for evaluation")
    
    pred_xyz = []
    pred_class = []

    for i in range(n_objects):
        obj_pcl = objectMaps.objectMap['pcl'][object_ids[i]]
        pred_xyz.append(obj_pcl)
        pred_class.append(np.ones(obj_pcl.shape[0]) * object_class[i].item())
        
    pred_xyz = torch.from_numpy(np.concatenate(pred_xyz, axis=0))
    if not GT_POSES:
        pred_xyz = (pred_xyz.float() @ okvisPoses[0,:3,:3].t().float() + okvisPoses[0,:3,3]) @ R_align
    else: 
        pred_xyz = pred_xyz.float()
    pred_class = torch.from_numpy(np.concatenate(pred_class, axis=0)).long()

    ## Align Poitn Clouds
    if not GT_POSES:
        gt_aligned =  gt_xyz.view(-1,3).double() @ gt_alignment[:3,:3].t().double() + gt_alignment[:3,3].double()
        gt_xyz = gt_aligned
    else:
        gt_xyz = gt_xyz.view(-1,3).double()


    # Compute the associations between the predicted and ground truth point clouds
    idx_pred_to_gt, idx_gt_to_pred = compute_pred_gt_associations(
        pred_xyz.unsqueeze(0).cuda().contiguous().float(),
        gt_xyz.unsqueeze(0).cuda().contiguous().float(),
    )
    
    # Only keep the points on the 3D reconstructions that are mapped to
    # GT point that is in keep_index
    label_gt = gt_class[idx_pred_to_gt.cpu()]
    pred_keep_idx = torch.isin(label_gt, torch.from_numpy(keep_index))
    pred_xyz = pred_xyz[pred_keep_idx]
    pred_class = pred_class[pred_keep_idx]
    idx_pred_to_gt = idx_pred_to_gt[pred_keep_idx]
    idx_gt_to_pred = None  # not to be used
    
    # Compute the confusion matrix
    confmatrix = compute_confmatrix(
        pred_class.cuda(),
        gt_class.cuda(),
        idx_pred_to_gt,
        idx_gt_to_pred,
        class_names,
    )
    
    assert confmatrix.sum(0)[ignore_index].sum() == 0
    assert confmatrix.sum(1)[ignore_index].sum() == 0
    
    if args.debug:

        color_legend = fa_vis.create_image_with_legend(255.0 * class2color[keep_index], [class_names[int(i)] for i in keep_index])
        color_legend = cv2.cvtColor(color_legend, cv2.COLOR_BGR2RGB)

        # Display the image
        cv2.imshow("Image with Legend", color_legend)
        cv2.waitKey(0)
        
        # '''Visualization for debugging'''
        cf = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0., 0., 0.]))

        # GT point cloud in open3d
        existing_gt_labels_idx = torch.isin(gt_class, torch.from_numpy(keep_index))
        existing_gt_xyz = gt_xyz[existing_gt_labels_idx]
        existing_gt_labels = gt_class[existing_gt_labels_idx]

        gt_pcd = o3d.geometry.PointCloud()
        gt_pcd.points = o3d.utility.Vector3dVector(existing_gt_xyz.view(-1,3).cpu().double().numpy())
        gt_pcd.colors = o3d.utility.Vector3dVector(class2color[existing_gt_labels.cpu()])
        
        # predicted point cloud in open3d
        pred_pcd = o3d.geometry.PointCloud()
        pred_pcd.points = o3d.utility.Vector3dVector(pred_xyz.view(-1,3).cpu().double().numpy())
        pred_pcd.colors = o3d.utility.Vector3dVector(class2color[pred_class.cpu()])

        # Create a list of geometries and a visibility dictionary
        geometries = [gt_pcd, pred_pcd, cf] 
        geometries = [gt_pcd, pred_pcd]
        visibility = {id(gt_pcd): True, id(pred_pcd): True}

        # Function to toggle visibility
        def toggle_visibility(vis, geometry_id):
            view_control = vis.get_view_control()

            # Save the current camera parameters
            camera_params = view_control.convert_to_pinhole_camera_parameters()

            if visibility[geometry_id]:
                vis.remove_geometry(geometry_map[geometry_id])
            else:
                vis.add_geometry(geometry_map[geometry_id])
            visibility[geometry_id] = not visibility[geometry_id]

             # Reapply the saved camera parameters
            view_control.convert_from_pinhole_camera_parameters(camera_params)
        
            return False

        # Map geometries to their IDs for easy lookup
        geometry_map = {id(pcd): pcd for pcd in geometries}

        # Define key callbacks
        def toggle_pcd1(vis):
            return toggle_visibility(vis, id(gt_pcd))

        def toggle_pcd2(vis):
            return toggle_visibility(vis, id(pred_pcd))

        # Add geometries and start visualization with key callbacks
        key_to_callback = {
            ord("1"): toggle_pcd1,  # Press '1' to toggle the first point cloud
            ord("2"): toggle_pcd2,  # Press '2' to toggle the second point cloud
        }

        o3d.visualization.draw_geometries_with_key_callbacks(geometries, key_to_callback)
    
    return confmatrix, keep_index


def main(args: argparse.Namespace):
    
    # map REPLICA_CLASSES to REPLICA_EXISTING_CLASSES
    class_all2existing = torch.ones(len(REPLICA_CLASSES)).long() * -1
    for i, c in enumerate(REPLICA_EXISTING_CLASSES):
        class_all2existing[c] = i
    class_names = [REPLICA_CLASSES[i] for i in REPLICA_EXISTING_CLASSES]

    # Get also (normaized) feature vectors for all classes
    class_embeddings = []
    for c in class_names:
        embedding_path = os.path.join(args.query_path, c+".txt")
        
        if(os.path.exists(embedding_path)):
            feature_embedding = np.genfromtxt(embedding_path, delimiter=',')
            print(f"Loaded embedding for class {c} from {embedding_path}")
            feature_embedding/=np.linalg.norm(feature_embedding)
            class_embeddings.append(feature_embedding)
        else:
            print(f"ERROR! No Embedding available for class: {c} at {class_names.index(c)}")
            class_embeddings.append(np.zeros(768))
    
    
    if args.n_exclude == 1:
        exclude_class = [class_names.index(c) for c in [
            "other"
        ]]
    elif args.n_exclude == 4:
        exclude_class = [class_names.index(c) for c in [
            "other", "floor", "wall", "ceiling"
        ]]
    elif args.n_exclude == 6:
        exclude_class = [class_names.index(c) for c in [
            "other", "floor", "wall", "ceiling", "door", "window"
        ]]
    else:
        raise ValueError("Invalid n_exclude: %d" % args.n_exclude)
    
    print("Excluding classes: ", [(i, class_names[i]) for i in exclude_class])
    

    # Compute Confusion Matrices
    conf_matrices = {}
    conf_matrix_all = 0
    for scene_id in list_of_sequences:
        print("Evaluating on:", scene_id)
        conf_matrix, keep_index = eval_replica(
            scene_id = scene_id,
            class_names = class_names,
            class_feats = class_embeddings,
            args = args,
            class_all2existing = class_all2existing,
            ignore_index = exclude_class,
        )
        
        conf_matrix = conf_matrix.detach().cpu()
        conf_matrix_all += conf_matrix

        conf_matrices[scene_id] = {
            "conf_matrix": conf_matrix,
            "keep_index": keep_index,
        }
        
    # Remove the rows and columns that are not in keep_class_index
    conf_matrices["all"] = {
        "conf_matrix": conf_matrix_all,
        "keep_index": conf_matrix_all.sum(axis=1).nonzero().reshape(-1),
    }
    
    results = []
    for scene_id, res in conf_matrices.items():
        conf_matrix = res["conf_matrix"]
        keep_index = res["keep_index"]
        conf_matrix = conf_matrix[keep_index, :][:, keep_index]
        keep_class_names = [class_names[i] for i in keep_index]

        mdict = compute_metrics(conf_matrix, keep_class_names)
        results.append(
            {
                "scene_id": scene_id,
                "miou": mdict["miou"] * 100.0,
                "mrecall": np.mean(mdict["recall"]) * 100.0,
                "mprecision": np.mean(mdict["precision"]) * 100.0,
                "mf1score": np.mean(mdict["f1score"]) * 100.0,
                "fmiou": mdict["fmiou"] * 100.0,
            }
        )

    print(results)
    

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    
