import os
import numpy as np
import pandas as pd
import plyfile
import torch

class FindAnythingReaderSem:

    def __init__(self, sequence_path: str):
        print(f"Trying to load sequence under {sequence_path}")

        objectMapDict = {}
        objectMapDict['pcl'] = {}
        objectMapDict['features'] = {}
        

        # Open CSV files with results:
        objectmaps = [om for om in os.listdir(sequence_path)
                        if om.startswith('objectMap')]
        for objectMap in objectmaps:
            print(f"\tProcessing object map: {objectMap}")
            # Regular expression to match the pattern and extract the number
            kfId = int(objectMap[9:objectMap.index('.')])

            # Read Object Map CSV data
            object_data = pd.read_csv(os.path.join(sequence_path,objectMap))

            # Open Corresponding Mesh file
            # mesh_file_name = os.path.join(sequence_path, 'mesh_kf%04d.ply' % (kfId))
            mesh_file_name = os.path.join(sequence_path, 'mesh_kf%06d.ply' % (kfId))
            print(f"Opening mesh: {mesh_file_name}")
            with open(mesh_file_name, "rb") as f:
                mesh = plyfile.PlyData.read(f)

                # Iterating objects
                for _, row in object_data.iterrows():

                    feature = row.loc[' f0' : ' f767'].to_numpy().astype(np.float32)
                    oid = np.int32(row.loc['ObjectId'])

                    objectMapDict['features'][oid] = feature

                    # print(f"Checking object ID {oid} and look in mesh {mesh_file_name}")
                    # pcl_oid = mesh['vertex'][mesh['vertex']['segment_id'] == oid]
                    pcl_oid = mesh['vertex'][mesh['vertex']['id'] == oid]

                    # Shape for gradslam compatibility: (batch_size, N, 3)
                    pcl_xyz = np.ndarray(shape=(len(pcl_oid), 3))
                    for i, v in enumerate(pcl_oid):
                        pcl_xyz[i, 0] = v['x']
                        pcl_xyz[i, 1] = v['y']
                        pcl_xyz[i, 2] = v['z']
                    
                    # print(f" -- pcl size: {pcl_xyz.shape}")
                    objectMapDict['pcl'][oid] = pcl_xyz

        self.objectMap = objectMapDict

    def get_stacked_values_torch(self, key):
        values = []
        ids = []

        # Iterating 
        try:
                
            for v in self.objectMap[key]:
                value = torch.from_numpy(self.objectMap[key][v].astype(np.float64))
                values.append(value)
                ids.append(v)
        except Exception as e:
            import pdb
            pdb.set_trace()
            print(f"{e}: ")

        return ids, torch.stack(values,dim = 0)

