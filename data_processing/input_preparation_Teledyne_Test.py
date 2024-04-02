from os.path import join, exists, dirname, abspath
from sklearn.neighbors import KDTree
import numpy as np
import os, pickle, argparse, sys

BASE_DIR = dirname(abspath(__file__))
ROOT_DIR = dirname(BASE_DIR)
sys.path.append(BASE_DIR)
sys.path.append(ROOT_DIR)

from helper_ply import write_ply
from tool import DataProcessing as DP

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/nas2/jacob/data/Teledyne_Test/test/original_las', help='')
    parser.add_argument('--output_path', type=str, default='/nas2/jacob/data/Teledyne_Test/test', help='original dataset path')
    parser.add_argument('--grid_size', type=float, default=0.06, help='sampling grid size default=0.2')
    FLAGS = parser.parse_args()
    dataset_path = FLAGS.dataset_path
    preparation_types = ['grid']  # Grid sampling & Random sampling
    grid_size = FLAGS.grid_size
    random_sample_ratio = 10
    files = np.sort([join(dataset_path, i) for i in os.listdir(dataset_path)])
    length = files.shape[0]
    total_lenght = files.shape[0]
    c = 0
    total_org=0
    total_sub=0
    print('grid_size: ' , grid_size)
    for sample_type in preparation_types:
        for pc_path in files:
            cloud_name = pc_path.split('/')[-1][:-4]
            #print('start to process:', cloud_name)

            # create output directory
            out_folder = join(FLAGS.output_path, sample_type + '_{:.3f}'.format(grid_size))
            os.makedirs(out_folder) if not exists(out_folder) else None

            # check if it has already calculated
            if exists(join(out_folder, cloud_name + '_KDTree.pkl')):
                print(cloud_name, 'already exists, skipped')
                continue

            xyz, i = DP.read_las_no_label_no_norm(pc_path)
            labels = None
            org_point_number = i.shape[0]
            #os.makedirs(join(out_folder, 'pt_clouds')) if not exists(join(out_folder, 'pt_clouds')) else None
            sub_ply_file = join(out_folder, cloud_name + '.ply')
            sub_xyz, sub_i = DP.grid_sub_sampling(xyz, i, labels, grid_size)
            
            
            sub_point_number = sub_i.shape[0]
            total_org += org_point_number
            total_sub += sub_point_number
            print(cloud_name + '| ' + str(org_point_number) + '| ' + str(sub_point_number))
            c += 1
            if c == total_lenght:
                print('total points' + '| ' + str(total_org) + '| ' + str(total_sub))
                
                
            #sub_rgb = sub_rgb / 255.0
            write_ply(sub_ply_file, [sub_xyz, sub_i], ['x', 'y', 'z', 'intensity', 'num_return'])

            search_tree = KDTree(sub_xyz, leaf_size=50)
            #os.makedirs(join(out_folder, 'KDTree')) if not exists(join(out_folder, 'KDTree')) else None
            kd_tree_file = join(out_folder, cloud_name + '_KDTree.pkl')
            with open(kd_tree_file, 'wb') as f:
                pickle.dump(search_tree, f)

            proj_idx = np.squeeze(search_tree.query(xyz, return_distance=False))
            proj_idx = proj_idx.astype(np.int32)
            #os.makedirs(join(out_folder, 'proj')) if not exists(join(out_folder, 'proj')) else None
            proj_save = join(out_folder, cloud_name + '_proj.pkl')
            with open(proj_save, 'wb') as f:
                pickle.dump([proj_idx, labels], f)
