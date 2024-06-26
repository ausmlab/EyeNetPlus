from os.path import join
from EyeNetPlus import Network
from tester_S3DIS import ModelTester
from helper_ply import read_ply
from tool import ConfigS3DIS as cfg
from tool import DataProcessing as DP
import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os


class S3DIS:
    def __init__(self, test_area_idx):
        self.path = cfg.data_set_dir
        self.label_to_names = {0: 'ceiling',
                               1: 'floor',
                               2: 'wall',
                               3: 'beam',
                               4: 'column',
                               5: 'window',
                               6: 'door',
                               7: 'table',
                               8: 'chair',
                               9: 'sofa',
                               10: 'bookcase',
                               11: 'board',
                               12: 'clutter'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.num_per_class = np.zeros(self.num_classes)
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)
        for ignore_label in self.ignored_labels:
            self.num_per_class = np.delete(self.num_per_class, ignore_label)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'grid_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                print('validation',cloud_name)
                cloud_split = 'validation'
            else:
                print('training',cloud_name)
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            
            # compute num_per_class in training set
            if cloud_split == 'training':
                self.num_per_class += DP.get_num_class_from_label(sub_labels, self.num_classes)

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():

            # Generator loop
            for i in range(num_per_epoch):  # num_per_epoch

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)
                
                #collect points for base receptive field
                if len(points) < cfg.num_points * 4 //7:
                    base_queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    base_queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points * 4 //7)[1][0]
                
                # Shuffle index for base
                base_queried_idx = DP.shuffle_idx(base_queried_idx)

                # Get corresponding points and colors based on the index
                base_queried_pc_xyz = points[base_queried_idx]
                base_queried_pc_xyz[:, 0:2] = base_queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                base_queried_pc_colors = self.input_colors[split][cloud_idx][base_queried_idx]  
                base_queried_pc_labels = self.input_labels[split][cloud_idx][base_queried_idx]
                base_queried_pc_labels = np.array([self.label_to_idx[l] for l in base_queried_pc_labels])
                
                # Collect points and colors for medium receptive field
                base_dists = np.sum(np.square((points[base_queried_idx] - pick_point).astype(np.float32)), axis=1)
                base_r = np.sqrt(np.max(base_dists))
                
                medium_r = base_r*2
                
                ind = self.input_trees[split][cloud_idx].query_radius(pick_point, r=medium_r)[0]
                medium_queried_idx = np.setdiff1d(ind, base_queried_idx,assume_unique=True)
                
                medium_queried_idx = DP.shuffle_idx(medium_queried_idx)[:cfg.num_points * 3 //7]
                
                medium_queried_pc_xyz = points[medium_queried_idx]
                medium_queried_pc_xyz[:, 0:2] = medium_queried_pc_xyz[:, 0:2] - pick_point[:, 0:2]
                
                medium_queried_pc_colors = self.input_colors[split][cloud_idx][medium_queried_idx]
                medium_queried_pc_labels = self.input_labels[split][cloud_idx][medium_queried_idx]
                medium_queried_pc_labels = np.array([self.label_to_idx[l] for l in medium_queried_pc_labels])
                
                
                if len(points) < cfg.num_points * 4 //7:
                    base_queried_pc_xyz, base_queried_pc_colors, base_queried_idx, base_queried_pc_labels = \
                        DP.data_aug(base_queried_pc_xyz, 
                                    base_queried_pc_colors, 
                                    base_queried_pc_labels,
                                    base_queried_idx, 
                                    cfg.num_points * 4 //7)
                    
                if len(medium_queried_pc_xyz) < cfg.num_points * 3 //7:
                    print(len(points))
                    medium_queried_pc_xyz, medium_queried_pc_colors, medium_queried_idx, medium_queried_pc_labels = \
                        DP.data_aug(medium_queried_pc_xyz,
                                    medium_queried_pc_colors,
                                    medium_queried_pc_labels,
                                    medium_queried_idx,
                                    cfg.num_points * 3 //7)
                    
                #concatenate base and medium indexes
                query_idx = np.concatenate((base_queried_idx, medium_queried_idx))
                
                # Update the possibility of the selected points
                dists = np.sum(np.square((points[query_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][query_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))
                
                #combine medium and base points
                queried_pc_xyz = np.concatenate((base_queried_pc_xyz, medium_queried_pc_xyz), axis = 0)
                queried_pc_colors = np.concatenate((base_queried_pc_colors, medium_queried_pc_colors), axis = 0)
                queried_pc_labels = np.concatenate((base_queried_pc_labels, medium_queried_pc_labels), axis = 0)


                #TODO: I have to add functions that does not take number of return as features 
                if True:
                    yield (queried_pc_xyz,
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           query_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes


    def get_tf_mapping2(self):

        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            #separate points to base and medium receptive field
            b_batch_xyz, m_batch_xyz = batch_xyz[:,:cfg.num_points * 4 //7,:], batch_xyz[:,cfg.num_points * 4 //7:,:]
            b_batch_features, m_batch_features = batch_features[:,:cfg.num_points * 4 //7,:], batch_features[:,cfg.num_points * 4 //7:,:]
            b_batch_xyz_opp = b_batch_xyz
            m_batch_xyz_opp = m_batch_xyz
            b_input_points = []
            b_input_neighbors = []
            b_input_pools = []
            b_input_up_samples = []


            m_input_points =[]
            m_input_neighbors = []
            m_input_pools = []
            m_input_up_samples = []
            
            #currently it always assume the last subsampling ratio to be 4. Is it even possible to use 2 like original RandLA-Net Implementation?
            for i in range(cfg.num_layers):
                #processing base points
                neigh_idx = tf.py_func(DP.knn_search, [b_batch_xyz, b_batch_xyz, cfg.k_n[i]], tf.int32)
                sub_points = b_batch_xyz[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                if i == 0:
                    sub_features = b_batch_features[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                    m_batch_features = tf.concat((sub_features,m_batch_features), 1)
                pool_i = neigh_idx[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, b_batch_xyz, 1], tf.int32)
                
                b_input_points.append(b_batch_xyz)
                b_input_neighbors.append(neigh_idx)
                b_input_pools.append(pool_i)
                b_input_up_samples.append(up_i)
                
                
                if cfg.sub_sampling_ratio[i] == cfg.connection_ratio:
                    b_batch_xyz = sub_points
                else:
                    addtional_sampling_ratio = cfg.connection_ratio//cfg.sub_sampling_ratio[i]
                    b_batch_xyz = sub_points[:, :tf.shape(sub_points)[1] // addtional_sampling_ratio, :]
                
                #processing medium points
                m_input_data = tf.concat((b_batch_xyz, m_batch_xyz), 1)
                m_neigh_idx = tf.py_func(DP.knn_search, [m_input_data, m_input_data, cfg.k_n[i]], tf.int32)
                
                m_b_neigh_idx = m_neigh_idx[:,:tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_m_neigh_idx = m_neigh_idx[:,tf.shape(b_batch_xyz)[1]: tf.shape(b_batch_xyz)[1] + tf.shape(m_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_pool_i = tf.concat((m_b_neigh_idx, m_m_neigh_idx), 1)
                
                b_sub_points = b_batch_xyz[:, :tf.shape(b_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_batch_xyz = m_batch_xyz[:, :tf.shape(m_batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                m_sub_points = tf.concat((b_sub_points, m_batch_xyz), 1)
                
                m_up_i = tf.py_func(DP.knn_search, [m_sub_points, m_input_data, 1], tf.int32)
                
                m_input_points.append(m_input_data)#[12288,3072, 768, 192, 48]
                m_input_neighbors.append(m_neigh_idx)
                m_input_pools.append(m_pool_i)
                m_input_up_samples.append(m_up_i)
                ##########################################################################################
            
            opp = b_batch_xyz_opp[:, tf.shape(b_batch_xyz_opp)[1] // cfg.sub_sampling_ratio[0]:, :]
            cat_xyz = tf.concat((b_input_points[1],opp, m_batch_xyz_opp), axis = 1)
            reorder_idx = tf.py_func(DP.knn_search, [cat_xyz, batch_xyz, 1], tf.int32)
            
            
            input_list = b_input_points + b_input_neighbors + b_input_pools + b_input_up_samples + m_input_points + m_input_neighbors + m_input_pools + m_input_up_samples
            input_list += [b_batch_features, m_batch_features, batch_labels, batch_pc_idx, batch_cloud_idx, reorder_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = S3DIS(test_area)
    dataset.init_input_pipeline()
    
    cfg.log_file_name = cfg.log_file_name + '_' + str(test_area)
    cfg.train_sum_dir = cfg.train_sum_dir + '_' + str(test_area)
    cfg.saving_path = cfg.saving_path + '_' + str(test_area)
    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        chosen_snap = FLAGS.model_path
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)
    else:
        raise ValueError('mode not supported')
