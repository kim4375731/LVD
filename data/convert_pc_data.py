import meshio 
import os
import argparse

parser = argparse.ArgumentParser(description="convert ply to obj directory-wise")
parser.add_argument('-m', '--mode', type=str)
parser.add_argument('-pf', '--path-from', type=str)
parser.add_argument('-pt', '--path-to', type=str)
args = parser.parse_args()

# path_from_list = os.path.abspath(args.path_from).split("/")[:-1]
# args.path_from = '/' + os.path.join(path_from_list)
# path_to_list = os.path.abspath(args.path_to).split("/")[:-1]
# args.path_to = '/' + os.path.join(path_to_list)
args.path_from = os.path.abspath(args.path_from)
args.path_to = os.path.abspath(args.path_to)

def normalize_obj(path_from, path_to):
    assert os.path.exists(path_from), "[IPNet/N/ERROR] there should be right path from which some obj are loaded!"
    assert os.path.exists(path_to), "[IPNet/N/ERROR] there should be right path to which some obj are delivered!"
    from utils.preprocess_scan import process
    lists = [f for f in os.listdir(path_from) if f.endswith("obj")]
    for i, f in enumerate(lists):
        fname = f.split('.')[0]        
        f = path_from + f'/{f}'
        print(f"[IPNet/N/INFO] normalizing {f} ... ({i+1}/{len(lists)})")
        _, _ = process(path_from, "none", fname, path_to)


def voxelize_obj(path_from, path_to):
    assert os.path.exists(path_from), "[IPNet/V/ERROR] there should be right path from which some obj are loaded!"
    assert os.path.exists(path_to), "[IPNet/V/ERROR] there should be right path to which some npy are delivered!"
    from utils.voxelized_pointcloud_sampling import voxelized_pointcloud_sampling
    bb_min = -1
    bb_max = 1.
    EXT = "01"
    REDO = False
    RES = 128
    NUM_PT = 5000
    lists = [f for f in os.listdir(path_from) if f.endswith("_scaled.obj")]
    for i, f in enumerate(lists):
        fname = f.split('.')[0]        
        f = path_from + f'/{f}'
        voxelized_pointcloud_sampling(f, fname, path_to, res=RES, num_points=NUM_PT, bounds=(bb_min, bb_max), ext=EXT)


def convert_ply2obj(path_from, path_to):
    assert os.path.exists(path_from), "[IPNet/C/ERROR] there should be right path from which some ply are loaded!"
    assert os.path.exists(path_to), "[IPNet/C/ERROR] there should be right path to which some obj are delivered!"
    lists = [f for f in os.listdir(path_from) if f.endswith("ply")]
    for i, f in enumerate(lists):        
        fname = f.split('.')[0]        
        f = path_from + f'/{f}'
        print(f"[MESHIO/INFO] converting {f} ... ({i+1}/{len(lists)})")
        mesh = meshio.read(f)
        meshio.write(os.path.join(path_to, fname+".obj"), mesh)


if __name__ == "__main__":
    mode = args.mode
    if mode == 'normalize':
        normalize_obj(args.path_from, args.path_to)
    elif mode == 'voxelize':
        voxelize_obj(args.path_from, args.path_to)
    elif mode == 'convert':
        convert_ply2obj(args.path_from, args.path_to)
    else:
        assert 0, '[IPNet/ERROR] no such supported convert mode!'
