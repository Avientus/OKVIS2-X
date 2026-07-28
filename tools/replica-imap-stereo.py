#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Imperial College London
# SPDX-FileCopyrightText: 2024 Smart Robotics Lab, Technical University of Munich
# SPDX-FileCopyrightText: 2024-2026 Sotiris Papatheodorou
# SPDX-License-Identifier: BSD-3-Clause
#
# replica-imap-stereo.py - generate stereo images for iMAP Replica sequences
#
# Setup
# =====
# 1. Create a conda environment as follows:
#     conda create -n replica-imap-stereo python=3.8 cmake=3.14
#     conda activate replica-imap-stereo
#     conda install habitat-sim=0.2.2 -c conda-forge -c aihabitat
# 2. Download the Replica dataset (31.6 GB download + 42.3 GB to extract) using
#    this script:
#    https://raw.githubusercontent.com/facebookresearch/Replica-Dataset/refs/heads/main/download.sh
# 3. Download the iMAP Replica sequences (11.6 GB download + 11.8 GB to extract)
#    from here: https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip
#
# Usage
# =====
# Run the script with the -h option to display the usage help.

import PIL
import argparse
import habitat_sim as hs
import json
import magnum
import math
import numpy as np
import os
import os.path
import quaternion
import shutil
import sys

from typing import List, Tuple


def f_to_hfov(f: float, width: int) -> float:
    """Convert focal length in pixels to horizontal field of view in degrees.
    https://github.com/facebookresearch/habitat-sim/issues/402"""
    return math.degrees(2.0 * math.atan(width / (2.0 * f)))


def split_pose(T: np.ndarray) -> Tuple[np.ndarray, np.quaternion]:
    """Split a pose stored in a 4x4 matrix into a position vector and an
    orientation quaternion."""
    return T[0:3, 3], quaternion.from_rotation_matrix(T[0:3, 0:3]).normalized()


class Scene:
    SCENES = ['office_0', 'office_1', 'office_2', 'office_3', 'office_4',
              'room_0', 'room_1', 'room_2']
    RATE_HZ = 10 # Arbitrary, used to produce the timestamps.

    def __init__(self, replica_dir: str, imap_dir: str, out_dir: str, name: str,
                 baseline: float) -> None:
        self._out_dir = os.path.join(out_dir, name, 'mav0')

        self._imap_dir = os.path.join(imap_dir, name.replace('_', ''))
        traj_filename = os.path.join(self._imap_dir, 'traj.txt')
        self._traj_T_WC = np.genfromtxt(traj_filename).reshape((-1, 4, 4))
        self._baseline = baseline

        self.name = name
        self.scene_filename = os.path.join(replica_dir, name, 'habitat',
                                           'replica_stage.stage_config.json')
        with open(os.path.join(imap_dir, 'cam_params.json')) as f:
            self.intrinsics = json.load(f)['camera']

        self.write_cam_data_csv('cam0')
        self.write_cam_data_csv('cam1')
        self.write_cam_sensor_yaml('cam0', '(left)')
        self.write_cam_sensor_yaml('cam1', '(right)')
        shutil.copy(traj_filename, os.path.join(self._out_dir, 'traj.txt'))
        try:
            os.symlink('cam0', os.path.join(self._out_dir, 'rgb2'))
        except FileExistsError:
            pass

    def num_poses(self) -> int:
        return self._traj_T_WC.shape[0]

    def T_WC(self, i: int) -> np.ndarray:
        """Return the ith pose of the z-forward, x-right camera frame C, as used
        in the iMAP Replica sequences, expressed in the world frame W as a 4x4
        array."""
        return self._traj_T_WC[i,...]

    def write_image(self, img: np.ndarray, img_idx: int, cam_name: int) -> None:
        dir = os.path.join(self._out_dir, cam_name, 'data')
        os.makedirs(dir, exist_ok=True)
        filename = os.path.join(dir, '{:012d}.png'.format(Scene._stamp_ns(img_idx)))
        PIL.Image.fromarray(img).save(filename, optimize=True)

    def write_cam_data_csv(self, cam_name: str) -> None:
        dir = os.path.join(self._out_dir, cam_name)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'data.csv'), 'w') as f:
            f.write('#timestamp [ns],filename\n')
            for i in range(self.num_poses()):
                stamp_ns = Scene._stamp_ns(i)
                f.write('{:012d},{:012d}.png\n'.format(stamp_ns, stamp_ns))

    def write_cam_sensor_yaml(self, cam_name: str, cam_desc: str) -> None:
        dir = os.path.join(self._out_dir, cam_name)
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'sensor.yaml'), 'w') as f:
            if cam_name == 'cam1':
                x = self._baseline
            else:
                x = 0.0
            if cam_name.startswith('depth'):
                cam_type = 'depth-'
            else:
                cam_type = ''
            f.write('sensor_type: {}camera\n'.format(cam_type))
            f.write('comment: {} {}\n'.format(cam_name, cam_desc))
            f.write('\n')
            f.write('# Camera extrinsics (S) w.r.t. the frame of cam0 (B).\n')
            f.write('T_BS:\n')
            f.write('  cols: 4\n')
            f.write('  rows: 4\n')
            f.write('  data: [1.0, 0.0, 0.0, {},\n'.format(x))
            f.write('         0.0, 1.0, 0.0, 0.0,\n')
            f.write('         0.0, 0.0, 1.0, 0.0,\n')
            f.write('         0.0, 0.0, 0.0, 1.0]\n')
            f.write('\n')
            f.write('# Camera intrinsics.\n')
            f.write('rate_hz: {}\n'.format(Scene.RATE_HZ))
            f.write('resolution: [{}, {}]\n'.format(self.intrinsics['w'],
                                                    self.intrinsics['h']))
            f.write('camera_model: pinhole\n')
            f.write('intrinsics: [{}, {}, {}, {}] # fu, fv, cu, cv\n'.format(
                self.intrinsics['fx'], self.intrinsics['fy'],
                self.intrinsics['cx'], self.intrinsics['cy']))
            f.write('distortion_model: radial-tangential\n')
            f.write('distortion_coefficients: [0, 0, 0, 0] # no distortion\n')

    def copy_depth(self, img_idx: int) -> None:
        dir = os.path.join(self._out_dir, 'depth0', 'data')
        os.makedirs(dir, exist_ok=True)
        src = os.path.join(self._imap_dir, 'results', 'depth{:06d}.png'.format(img_idx))
        dst = os.path.join(dir, '{:012d}.png'.format(Scene._stamp_ns(img_idx)))
        shutil.copy(src, dst)

    @staticmethod
    def _stamp_ns(i: int) -> int:
        """Return the timestamp in nanoseconds corresponding to frame index i."""
        return 1000000000 * i // Scene.RATE_HZ


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="""Generate stereo color images in the EuRoC format for an
        iMAP Replica sequence. The left RGB camera (cam0) coincides with the
        color iMAP Replica one, while the right color camera (cam1) exists only
        in the generated dataset. If no SCENE is provided, generate a dataset
        for all scenes.""")
    parser.add_argument('replica_dir', metavar='REPLICA_DIR',
        help='the directory containing all Replica scenes')
    parser.add_argument('imap_dir', metavar='IMAP_DIR',
        help='the directory containing all iMAP Replica sequences')
    parser.add_argument('out_dir', metavar='OUT_DIR',
        help='the directory to write the generated datasets into')
    parser.add_argument('scenes', metavar='SCENE', nargs='*',
        help='the name of a Replica scene to generate a dataset for')
    parser.add_argument('-b', '--baseline', metavar='M', type=float, default=0.06,
        help='the baseline in meters of the stereo RGB cameras (default: 0.06)')
    parser.add_argument('-c', '--copy-depth', action='store_true',
        help='copy the depth images from IMAP_DIR into OUT_DIR')
    args = parser.parse_args()
    if not args.scenes:
        args.scenes = Scene.SCENES
    for scene in args.scenes:
        if scene not in Scene.SCENES:
            print('{} error: invalid scene name: {}'.format(
                os.path.basename(sys.argv[0]), scene), file=sys.stderr)
            print('Valid scene names:', Scene.SCENES, file=sys.stderr)
            sys.exit(2)

    # The -z-forward, x-right habitat-sim camera frame Ch (thanks Meta)
    # expressed in the z-forward, x-right camera frame C of the iMAP Replica
    # sequences.
    T_CCh = np.array([[1.0,  0.0,  0.0, 0.0],
                      [0.0, -1.0,  0.0, 0.0],
                      [0.0,  0.0, -1.0, 0.0],
                      [0.0,  0.0,  0.0, 1.0]])

    for name in args.scenes:
        scene = Scene(args.replica_dir, args.imap_dir, args.out_dir, name,
                      args.baseline)
        print('Processing {}: {} poses'.format(scene.name, scene.num_poses()))

        # Create the RGB sensor used in the iMAP Replica sequences.
        cam0_cfg = hs.CameraSensorSpec()
        cam0_cfg.uuid = 'cam0'
        cam0_cfg.sensor_type = hs.SensorType.COLOR
        cam0_cfg.sensor_subtype = hs.SensorSubType.PINHOLE
        cam0_cfg.resolution = [scene.intrinsics['h'], scene.intrinsics['w']]
        cam0_cfg.near = 0.00001
        cam0_cfg.far = 1000
        cam0_cfg.hfov = f_to_hfov(scene.intrinsics['fx'], scene.intrinsics['w'])
        cam0_cfg.position = magnum.Vector3.zero_init()
        # Create the second RGB sensor, offset horizontally to the right from
        # cam0 by args.baseline.
        cam1_cfg = hs.CameraSensorSpec()
        cam1_cfg.uuid = 'cam1'
        cam1_cfg.sensor_type = cam0_cfg.sensor_type
        cam1_cfg.sensor_subtype = cam0_cfg.sensor_subtype
        cam1_cfg.resolution = cam0_cfg.resolution
        cam1_cfg.near = cam0_cfg.near
        cam1_cfg.far = cam0_cfg.far
        cam1_cfg.hfov = cam0_cfg.hfov
        cam1_cfg.position = args.baseline * hs.geo.RIGHT
        # Attach the cameras to an agent.
        agent_cfg = hs.AgentConfiguration(height=0, sensor_specifications=[cam0_cfg, cam1_cfg])
        # Create the simulator.
        sim_cfg = hs.SimulatorConfiguration()
        sim_cfg.scene_id = scene.scene_filename
        sim = hs.Simulator(hs.Configuration(sim_cfg, [agent_cfg]))

        if args.copy_depth:
            scene.write_cam_data_csv('depth0')
            scene.write_cam_sensor_yaml('depth0', '(left)')

        # Render an image for each trajectory pose.
        for i in range(scene.num_poses()):
            if sys.stdout.isatty():
                print('{}: frame {:4d}/{}\r'.format(name, i+1, scene.num_poses()))
            T_WCh = scene.T_WC(i) @ T_CCh
            t_WCh, q_WCh = split_pose(T_WCh)
            sim.get_agent(0).set_state(hs.agent.AgentState(t_WCh, q_WCh))
            observation = sim.get_sensor_observations()
            for c in [0, 1]:
                cam = 'cam{}'.format(c)
                scene.write_image(observation[cam], i, cam)
            if args.copy_depth:
                scene.copy_depth(i)
        if sys.stdout.isatty():
            print()
