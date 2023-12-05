import os 
import numpy as np
import torch
import trimesh
import argparse

from utils.sdf import create_grid

parser = argparse.ArgumentParser(description="convert ply to obj directory-wise")
parser.add_argument('-m', '--mode', type=str)
args = parser.parse_args()
jobs_done = 0

clue_train_endswith = 'r.obj'
clue_test_endswith = '.obj'

def voxelize_scan(scan):
    resolution = 128 # Voxel resolution
    b_min = np.array([-1.2, -1.2, -1.2])
    b_max = np.array([1.2, 1.2, 1.2])
    step = 3000

    vertices = scan.vertices
    bounding_box = (vertices.max(0) - vertices.min(0))[1]

    vertices = vertices / bounding_box * 1.5
    trans = (vertices.max(0) + vertices.min(0))/2
    vertices = vertices - trans

    factor = max(1, int(len(vertices) / 20000)) # We will subsample vertices when there's too many in a scan !

    print("Voxelizing input scan:")
    # NOTE: It was easier and faster to just get distance to vertices, instead of voxels carrying inside/outside information,
    # which will only be possible for closed watertight meshes.
    with torch.no_grad():
        v = torch.FloatTensor(vertices).cuda()
        coords, mat = create_grid(resolution, resolution, resolution, b_min, b_max)
        points = torch.FloatTensor(coords.reshape(3, -1)).transpose(1, 0).cuda()
        points_npy = coords.reshape(3, -1).T
        iters = len(points)//step + 1

        all_distances = []
        for it in range(iters):
            it_v = points[it*step:(it+1)*step]
            it_v_npy = points_npy[it*step:(it+1)*step]
            distance = ((it_v.unsqueeze(0) - v[::factor].unsqueeze(1))**2).sum(-1)
            #contain = scan.contains(it_v_npy)
            distance = distance.min(0)[0].cpu().data.numpy()
            all_distances.append(distance)
            #contains = scan.contains(points_npy)
            signed_distance = np.concatenate(all_distances)

        voxels = signed_distance.reshape(resolution, resolution, resolution)
    return voxels, vertices

def convert_scan(path):
    global jobs_done 

    scan = trimesh.load(path)
    voxels_list = []
    vertices_list = []
    angle_per = 120
    periodic = 360//angle_per
    for j in range(periodic):
        if j != 0:
            rotation_matrix = trimesh.transformations.rotation_matrix(
                np.radians(angle_per), 
                [0, 1, 0]  # y axis
            )            
            scan.apply_transform(rotation_matrix)        
        voxels, normalized_vertices = voxelize_scan(scan)
        idx_778 = np.random.randint(0, len(normalized_vertices), 778)
        voxels_list.append(voxels)
        vertices_list.append(normalized_vertices[idx_778])        

    jobs_done += 1
    print(f'jobs done : {jobs_done}')

    return voxels_list, vertices_list

def make_dataset(mode):
    '''use only right hand dataset (test dataset are with only right hand ...)'''
    if mode == "train":
        scans_dir = '/workspace/IPNet/data_pool/mano/handsOnly_SCANS'              
        data_list = [f for f in os.listdir(scans_dir) if f.endswith(clue_train_endswith)]         
        dataset_name = 'train.npz'
    elif mode == "test":
        # scans_dir = '/workspace/IPNet/data_pool/mano/handsOnly_testDataset_SCANS'            
        scans_dir = '/workspace/IPNet/data_pool/mano/testdir2'            
        data_list = [f for f in os.listdir(scans_dir) if f.endswith(clue_test_endswith)]          
        dataset_name = 'test.npz'
    else:
        assert 0, "No such mode supported!"

    voxels_list = []
    vertices_list = []
    angle_per = 120
    periodic = 360//angle_per
    for i, f in enumerate(data_list):
        print(f'[MAKEDATASET/INFO] converting ... ({i+1}/{len(data_list)})')
        scan = trimesh.load(scans_dir + '/' + f)
        for j in range(periodic):
            if j != 0:
                rotation_matrix = trimesh.transformations.rotation_matrix(
                    np.radians(angle_per), 
                    [0, 1, 0]  # y axis
                )            
                scan.apply_transform(rotation_matrix)
            voxels, normalized_vertices = voxelize_scan(scan)
            voxels_list.append(voxels)
            vertices_list.append(normalized_vertices)
    np_voxels = np.asarray(voxels_list, dtype=object)
    np_vertices = np.asarray(vertices_list, dtype=object)
    save_path = scans_dir + '/' + dataset_name
    np.savez_compressed(save_path, voxels=np_voxels, vertices=np_vertices)


def make_dataset_multithreading(mode):
    '''use only right hand dataset (test dataset are with only right hand ...)'''
    if mode == "train":
        scans_dir = '/workspace/IPNet/data_pool/mano/handsOnly_SCANS'              
        data_list = [f for f in os.listdir(scans_dir) if f.endswith(clue_train_endswith)]         
        # scans_dir = '/workspace/LVD/data'
        # data_list = [f for f in os.listdir(scans_dir) if f.endswith('obj')]         
        dataset_name = 'train_120deg.npz'
    elif mode == "test":
        scans_dir = '/workspace/IPNet/data_pool/mano/handsOnly_testDataset_SCANS'            
        data_list = [f for f in os.listdir(scans_dir) if f.endswith(clue_test_endswith)]          
        dataset_name = 'test.npz'
    else:
        assert 0, "No such mode supported!"

    print(f'[MAKEDATASET/INFO] converting ... ')
    import concurrent.futures
    paths = [scans_dir + '/' + f for f in data_list]    
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(convert_scan, paths))
    np_results = np.asarray(results, dtype=object)
    np_voxels = np.stack(np.concatenate(np_results[:, 0], axis=-1)).astype(np.float32)
    np_vertices = np.stack(np.concatenate(np_results[:, 0], axis=-1)).astype(np.float32)
    save_path = scans_dir + '/' + dataset_name
    np.savez_compressed(save_path, voxels=np_voxels, vertices=np_vertices)


if __name__ == "__main__":
    make_dataset_multithreading(args.mode)
