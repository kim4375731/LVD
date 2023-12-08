import numpy as np
import torch
import trimesh
import argparse
parser = argparse.ArgumentParser(description="read obj file and save it after rotating")
parser.add_argument('-pf', '--path-from', type=str)
parser.add_argument('-pt', '--path-to', type=str)
args = parser.parse_args()

from utils.sdf import create_grid

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

def rotate_and_save_obj(input_file, output_file, angle, axis):
    """
    Rotates an OBJ file and saves it as another OBJ file.
    
    :param input_file: Path to the input OBJ file
    :param output_file: Path to the output OBJ file
    :param angle: Rotation angle in degrees
    :param axis: Axis of rotation ('x', 'y', or 'z')
    """
    # Load the mesh
    mesh = trimesh.load(input_file)
    voxels, normalized_vertices = voxelize_scan(mesh)
    print(f'before: {normalized_vertices[0]}')

    # Define the rotation based on the axis
    if axis == 'x':
        rotation_axis = [1, 0, 0]
    elif axis == 'y':
        rotation_axis = [0, 1, 0]
    else: # axis == 'z'
        rotation_axis = [0, 0, 1]
    
    # Convert the angle from degrees to radians and apply the rotation
    rotation_matrix = trimesh.transformations.rotation_matrix(
        np.radians(angle), 
        rotation_axis
    )
    mesh.apply_transform(rotation_matrix)

    voxels, normalized_vertices = voxelize_scan(mesh)
    print(f'after: {normalized_vertices[0]}')

    # Save the rotated mesh
    mesh.export(output_file)

def rotate_obj(path_from, path_to):
    # Example usage
    angle = 45  # rotation angle in degrees
    axis = 'y'  # rotation axis ('x', 'y', or 'z')

    # Read, rotate, and write the OBJ
    rotate_and_save_obj(path_from, path_to, angle, axis)



if __name__ == "__main__":
    rotate_obj(args.path_from, args.path_to)
