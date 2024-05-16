from os import makedirs
from os.path import exists, join
from helper_ply import read_ply, write_ply
import tensorflow as tf
import numpy as np
from tool import DataProcessing as DP
import time
from sklearn.metrics import confusion_matrix


def log_string(out_str, log_out):
    log_out.write(out_str + '\n')
    log_out.flush()
    print(out_str)


class ModelTester:
    def __init__(self, model, dataset, restore_snap=None):
        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        # Add a softmax operation for predictions
        self.prob_logits = tf.nn.softmax(model.logits)
        self.test_probs = [np.zeros((l.data.shape[0], model.config.num_classes), dtype=np.float16)
                           for l in dataset.input_trees['test']]

        self.log_out = open('log_test_' + dataset.name + '.txt', 'a')

    def test(self, model, dataset, num_votes=100):

        # Smoothing parameter for votes
        test_smooth = 0.98

        # Initialise iterator with train data
        self.sess.run(dataset.test_init_op)

        # Test saving path
        saving_path = time.strftime('results/Log_Semantic3D_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'predictions')) if not exists(join(test_path, 'predictions')) else None
        makedirs(join(test_path, 'probs')) if not exists(join(test_path, 'probs')) else None

        #####################
        # Network predictions
        #####################

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:

            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],)

                stacked_probs, stacked_labels, point_idx, cloud_idx = self.sess.run(ops, {model.is_training: False})
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    inds = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][inds] = test_smooth * self.test_probs[c_i][inds] + (1 - test_smooth) * probs
                step_id += 1
                log_string('Epoch {:3d}, step {:3d}. min possibility = {:.1f}'.format(epoch_id, step_id, np.min(
                    dataset.min_possibility['test'])), self.log_out)

            except tf.errors.OutOfRangeError:

                # Save predicted cloud
                new_min = np.min(dataset.min_possibility['test'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.log_out)

                if last_min + 4 < new_min:

                    print('Saving clouds')

                    # Update last_min
                    last_min = new_min

                    # Project predictions
                    print('\nReproject Vote #{:d}'.format(int(np.floor(new_min))))
                    t1 = time.time()
                    files = dataset.test_file_name
                    i_test = 0
                    for i, file_path in enumerate(files):
                        # Get file
                        points = self.load_evaluation_points(file_path)
                        points = points.astype(np.float16)

                        # Reproject probs
                        probs = np.zeros(shape=[np.shape(points)[0], 8], dtype=np.float16)
                        proj_index = dataset.test_proj[i_test]

                        probs = self.test_probs[i_test][proj_index, :]

                        # Insert false columns for ignored labels
                        probs2 = probs
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs2 = np.insert(probs2, l_ind, 0, axis=1)

                        # Get the predicted labels
                        preds = dataset.label_values[np.argmax(probs2, axis=1)].astype(np.uint8)

                        # Save plys
                        cloud_name = file_path.split('/')[-1]

                        # Save ascii preds
                        ascii_name = join(test_path, 'predictions', dataset.ascii_file_name[cloud_name])
                        np.savetxt(ascii_name, preds, fmt='%d')
                        log_string(ascii_name + 'has saved', self.log_out)
                        i_test += 1

                    t2 = time.time()
                    print('Done in {:.1f} s\n'.format(t2 - t1))
                    self.sess.close()
                    return

                self.sess.run(dataset.test_init_op)
                epoch_id += 1
                step_id = 0
                continue
        return

    @staticmethod
    def load_evaluation_points(file_path):
        data = read_ply(file_path)
        return np.vstack((data['x'], data['y'], data['z'])).T

class ModelTester_validation:
    def __init__(self, model, dataset, restore_snap=None, base_only=False):
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)
        makedirs('test_log') if not exists('test_log') else None
        self.Log_file = open('test_log/log_test_' + dataset.name + '.txt', 'a')

        # Create a session for running Ops on the Graph.
        on_cpu = False
        if on_cpu:
            c_proto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            c_proto = tf.ConfigProto()
            c_proto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=c_proto)
        self.sess.run(tf.global_variables_initializer())

        # Load trained model
        if restore_snap is not None:
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)

        self.prob_logits = tf.nn.softmax(model.logits)

        # Initiate global prediction over all test clouds
        self.test_probs = [np.zeros(shape=[l.shape[0], model.config.num_classes], dtype=np.float32)
                           for l in dataset.input_labels['validation']]

    def test(self, model, dataset, num_votes=1):

        # Smoothing parameter for votes
        test_smooth = 0.95

        # Initialise iterator with validation/test data
        self.sess.run(dataset.val_init_op)

        # Number of points per class in validation set
        val_proportions = np.zeros(model.config.num_classes, dtype=np.float32)
        i = 0
        for label_val in dataset.label_values:
            if label_val not in dataset.ignored_labels:
                val_proportions[i] = np.sum([np.sum(labels == label_val) for labels in dataset.val_labels])
                i += 1

        # Test saving path
        saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
        test_path = join('test', saving_path.split('/')[-1])
        makedirs(test_path) if not exists(test_path) else None
        makedirs(join(test_path, 'val_preds')) if not exists(join(test_path, 'val_preds')) else None

        step_id = 0
        epoch_id = 0
        last_min = -0.5

        while last_min < num_votes:
            try:
                ops = (self.prob_logits,
                       model.labels,
                       model.inputs['input_inds'],
                       model.inputs['cloud_inds'],
                       model.accuracy)
                
                stacked_probs, stacked_labels, point_idx, cloud_idx, acc = self.sess.run(ops, {model.is_training: False})
                print('step' + str(step_id) + ' acc:' + str(acc))
                stacked_probs = np.reshape(stacked_probs, [model.config.val_batch_size, model.config.num_points,
                                                           model.config.num_classes])

                for j in range(np.shape(stacked_probs)[0]):
                    probs = stacked_probs[j, :, :]
                    p_idx = point_idx[j, :]
                    c_i = cloud_idx[j][0]
                    self.test_probs[c_i][p_idx] = test_smooth * self.test_probs[c_i][p_idx] + (1 - test_smooth) * probs
                step_id += 1

            except tf.errors.OutOfRangeError:

                new_min = np.min(dataset.min_possibility['validation'])
                log_string('Epoch {:3d}, end. Min possibility = {:.1f}'.format(epoch_id, new_min), self.Log_file)

                if last_min + 1 < new_min:

                    # Update last_min
                    last_min += 1

                    # Show vote results (On subcloud so it is not the good values here)
                    log_string('\nConfusion on sub clouds', self.Log_file)
                    confusion_list = []

                    num_val = len(dataset.input_labels['validation'])

                    for i_test in range(num_val):
                        probs = self.test_probs[i_test]
                        for l_ind, label_value in enumerate(dataset.label_values):
                            if label_value in dataset.ignored_labels:
                                probs = np.insert(probs, l_ind, 0, axis=1)
                        preds = dataset.label_values[np.argmax(probs, axis=1)].astype(np.int32)
                        labels = dataset.input_labels['validation'][i_test]

                        # Confs
                        confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]

                    # Regroup confusions
                    C = np.sum(np.stack(confusion_list), axis=0).astype(np.float32)
                    
                    
                    for l_ind, label_value in reversed(list(enumerate(dataset.label_values))):
                        if label_value in dataset.ignored_labels:
                            C = np.delete(C, l_ind, axis=0)
                            C = np.delete(C, l_ind, axis=1)

                    # Rescale with the right number of point per class
                    C *= np.expand_dims(val_proportions / (np.sum(C, axis=1) + 1e-6), 1)

                    # Compute IoUs
                    IoUs = DP.IoU_from_confusions(C)
                    m_IoU = np.mean(IoUs)
                    s = '{:5.2f} | '.format(100 * m_IoU)
                    for IoU in IoUs:
                        s += '{:5.2f} '.format(100 * IoU)
                    log_string(s + '\n', self.Log_file)

                    #if int(np.ceil(new_min)) % 1 == 0:
                    if int(np.ceil(new_min)) % (num_votes + 1) == 0:

                        # Project predictions
                        log_string('\nReproject Vote #{:d}'.format(int(np.floor(new_min))), self.Log_file)
                        proj_probs_list = []

                        for i_val in range(num_val):
                            # Reproject probs back to the evaluations points
                            proj_idx = dataset.val_proj[i_val]
                            probs = self.test_probs[i_val][proj_idx, :]
                            probs2 = probs
                            for l_ind, label_value in enumerate(dataset.label_values):
                                if label_value in dataset.ignored_labels:
                                    probs2 = np.insert(probs2, l_ind, 0, axis=1)
                            proj_probs_list += [probs2]

                        # Show vote results
                        log_string('Confusion on full clouds', self.Log_file)
                        preds_list = []
                        labels_list = []
                        confusion_list = []
                        for i_test in range(num_val):
                            # Get the predicted labels
                            preds = dataset.label_values[np.argmax(proj_probs_list[i_test], axis=1)].astype(np.uint8)
                            # Confusion
                            labels = dataset.val_labels[i_test]
                            preds_list += [preds]
                            labels_list += [labels]
                            
                            acc = np.sum(preds == labels) / len(labels)
                            #log_string(dataset.input_names['validation'][i_test] + ' Acc:' + str(acc), self.Log_file)

                            confusion_list += [confusion_matrix(labels, preds, dataset.label_values)]
                            #name = dataset.input_names['validation'][i_test] + '.ply'
                            #original_data_path = '/nas2/jacob/data/semantic3d/original_ply/' + dataset.input_names['validation'][i_test] + '.ply'
                            #data = read_ply(original_data_path)
                            #print(data.shape)
                            #write_ply(join(test_path, 'val_preds', name), [data['x'], data['y'], data['z'], data['red'], data['green'], data['blue'], preds, labels], ['x','y','z','red','green','blue','pred', 'label'])

                        # Regroup confusions
                        preds = np.concatenate(preds_list)
                        labels = np.concatenate(labels_list)
                        acc = np.sum(preds == labels) / (labels.shape[0])
                        log_string('Overall Acc:' + str(acc), self.Log_file)
                        C = np.sum(np.stack(confusion_list), axis=0)

                        IoUs = DP.IoU_from_confusions(C)
                        m_IoU = np.mean(IoUs)
                        s = '{:5.2f} | '.format(100 * m_IoU)
                        for IoU in IoUs:
                            s += '{:5.2f} '.format(100 * IoU)
                        log_string('-' * len(s), self.Log_file)
                        log_string(s, self.Log_file)
                        log_string('-' * len(s) + '\n', self.Log_file)
                        print('finished \n')
                        #self.sess.close()
                        #return

                self.sess.run(dataset.val_init_op)
                epoch_id += 1
                step_id = 0
                continue

        return