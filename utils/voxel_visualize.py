import numpy as np
import torch
import trimesh
import open3d as o3d
import time

from data.make_dataset import voxelize_scan
from utils.sdf import create_grid

def vis():
    resolution = 128 # Voxel resolution
    lattice = False
    b_min = np.array([-1.2, -1.2, -1.2])
    b_max = np.array([1.2, 1.2, 1.2])
    step = 3000

    with torch.no_grad():
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        points_npy = coords.reshape(3, -1).T
        iters = len(points)//step + 1

        for it in range(iters):
            it_v_npy = points_npy[it*step:(it+1)*step]

    if lattice:
        ver = it_v_npy        
    else:
        scan = trimesh.load('../IPNet/data_pool/mano/testdir/32_03r.obj')
        _, ver = voxelize_scan(scan)                
    pcd = o3d.geometry.PointCloud()
    factor = max(1, int(len(ver) / 20000))
    print(ver[::factor].astype(np.float64).dtype)    
    pcd.points = o3d.utility.Vector3dVector(ver[::factor].astype(np.float64))

    # Define the voxel size and bounds for the VoxelGrid
    voxel_size = 1/resolution  # Adjust as needed
    max_bound = np.max(ver, axis=0) + voxel_size
    min_bound = np.min(ver, axis=0) - voxel_size

    # Create a VoxelGrid from the point cloud
    voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud_within_bounds(
        pcd, voxel_size, min_bound, max_bound
    )

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name='vis vox', width=800, height=600)
    vis.add_geometry(voxel_grid)
    vis.run()
    time.sleep(0)
    vis.destroy_window()


if __name__ == "__main__":
    vis()