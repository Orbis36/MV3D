import cv2
import numpy as np
from config import cfg
import os
import glob
from sklearn.utils import shuffle
from utils.check_data import check_preprocessed_data, get_file_names
import net.processing.boxes3d  as box
from multiprocessing import Process, Queue as Queue, Value, Array
# import queue
import time
import random

import config
import os
import numpy as np
import glob
import cv2
from kitti_data.pykitti.tracklet import parseXML, TRUNC_IN_IMAGE, TRUNC_TRUNCATED
from config import cfg
import data
import net.utility.draw as draw
from Object_data import *
from raw_data import *
from utils.training_validation_data_splitter import TrainingValDataSplitter
import pickle
import array
import data
from sklearn.utils import shuffle
import threading
import matplotlib.pyplot as plt


def draw_bbox_on_rgb(rgb, boxes3d, one_frame_tag):

    img = draw.draw_box3d_on_camera(rgb, boxes3d)
    new_size = (img.shape[1] // 3, img.shape[0] // 3)
    img = cv2.resize(img, new_size)
    path = os.path.join(config.cfg.LOG_DIR, '%s.png' % one_frame_tag.replace('/', '_'))
    cv2.imwrite(path, img)
    print('write %s finished' % path)


def draw_bbox_on_lidar_top(top, boxes3d, one_frame_tag):
    path = os.path.join(config.cfg.LOG_DIR, '%s.png' % one_frame_tag.replace('/', '_'))
    top_image = data.draw_top_image(top)
    top_image = data.draw_box3d_on_top(top_image, boxes3d, color=(0, 0, 80))
    cv2.imwrite(path, top_image)
    print('write %s finished' % path)


use_thread = True


class BatchLoading2:
    def __init__(self, tags=None, queue_size=20, require_shuffle=False, require_log=False, is_testset=False,
                 n_skip_frames=0, random_num=666, is_raw = True, is_valid =False,data_cate = 'training'):
        self.is_testset = is_testset
        self.is_raw = is_raw
        self.is_valid = is_valid
        self.shuffled = require_shuffle
        self.random_num = random_num
        self.preprocess = data.Preprocess()
        self.hres = 0.35
        self.vres = 0.4
        self.vfov = (-24.9,2.0)
        self.y_fudge = 5

        if is_raw:
            self.raw_img = Image()
            self.raw_lidar = Lidar()
            self.raw_tracklet = Tracklet()
        else:
            self.raw_img = Image_3ddata(data_cate)
            self.raw_lidar = Lidar_3ddata(data_cate)
            self.raw_tracklet = Tracklet_3ddata(data_cate)
            self.calib = Calib_3ddata(data_cate)


        if tags:
            self.tags = [tag for i, tag in enumerate(tags) if i % (n_skip_frames + 1) == 0]
        else:
            self.tags = [x for x in self.raw_img.files_path_mapping.keys()]


        # shuffle val/train tags
        if self.shuffled:
            self.tags = shuffle(self.tags, random_state=self.random_num)

        self.tag_index = 0
        self.size = len(self.tags)

        self.require_log = require_log
        self.flip_axis = 1 # if axis=1, flip from y=0. If axis=0, flip from x=0
        self.flip_rate = 2 # if flip_rate is 2, means every two frames

        self.cache_size = queue_size
        self.loader_need_exit = Value('i', 0)

        if use_thread:
            self.prepr_data = []
            self.lodaer_processing = threading.Thread(target=self.loader)
        else:
            self.preproc_data_queue = Queue()
            self.buffer_blocks = [Array('h', 41246691) for i in range(queue_size)]
            self.blocks_usage = Array('i', range(queue_size))
            self.lodaer_processing = Process(target=self.loader)
        self.lodaer_processing.start()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.loader_need_exit.value = True
        if self.require_log: print('set loader_need_exit True')
        self.lodaer_processing.join()
        if self.require_log: print('exit lodaer_processing')

    def keep_gt_inside_range(self, train_gt_labels, train_gt_boxes3d):

        train_gt_labels = np.array(train_gt_labels, dtype=np.int32)
        train_gt_boxes3d = np.array(train_gt_boxes3d, dtype=np.float32)
        if train_gt_labels.shape[0] == 0:
            return False, None, None
        assert train_gt_labels.shape[0] == train_gt_boxes3d.shape[0]

        # get limited train_gt_boxes3d and train_gt_labels.
        keep = np.zeros((len(train_gt_labels)), dtype=bool)

        for i in range(len(train_gt_labels)):
            if box.box3d_in_top_view(train_gt_boxes3d[i]):
                keep[i] = 1

        # if all targets are out of range in selected top view, return True.
        if np.sum(keep) == 0:
            return False, None, None

        train_gt_labels = train_gt_labels[keep]
        train_gt_boxes3d = train_gt_boxes3d[keep]
        return True, train_gt_labels, train_gt_boxes3d

    def load_from_one_tag(self, one_frame_tag):
        if self.is_testset:
            obstacles = None
        else:
            obstacles = self.raw_tracklet.load(one_frame_tag)
        rgb = self.raw_img.load(one_frame_tag)
        lidar = self.raw_lidar.load(one_frame_tag)
        if not self.is_raw:
            calib = self.calib.load(one_frame_tag)

            return obstacles, rgb, lidar, calib

        return obstacles, rgb, lidar

    def preprocess_one_frame(self, rgb, lidar, obstacles):
        rgb = self.preprocess.rgb(rgb)
        top = self.preprocess.lidar_to_top(lidar)

        if self.is_testset:
            return rgb, top, None, None
        # 每次只有一个
        boxes3d = [self.preprocess.bbox3d(obs) for obs in obstacles]
        labels = [self.preprocess.label(obs) for obs in obstacles]
        # flip in y axis.
        if self.is_aug and len(boxes3d) > 0:
            if self.tag_index % self.flip_rate == 1:
                top, rgb, boxes3d = self.preprocess.flip(rgb, top, boxes3d, axis=1)
            elif self.tag_index % self.flip_rate == 2:
                top, rgb, boxes3d = self.preprocess.flip(rgb, top, boxes3d, axis=0)
        return rgb, top, boxes3d, labels

    def preprocess_one_3d_frame(self, rgb, lidar, boxes_velo):

        # boxes_velo_use = boxes_velo.copy()
        # resize to 1242
        rgb = self.preprocess.rgb(rgb)
        # 标志位， box的后处理等到project时再做， 否则给出的lidar不是原来的
        # 继而转移矩阵不成立，无法还原到image平面
        param = {}

        # 这里每个预处理需要返回的box只是FV和BEV需要用的
        if not self.is_valid:
            method = [np.random.uniform(0, 1), np.random.uniform(0, 1), np.random.uniform(0, 1)]
            if method[0] > 1:
                rgb, lidar, s = self.preprocess.scale(lidar, rgb)
                boxes_velo[:, :, 0] *= s
                boxes_velo[:, :, 1] *= s
                boxes_velo[:, :, 2] *= s
                param['scale'] = s
            if method[1] > 1:
                rgb, lidar, t = self.preprocess.translate(lidar, rgb)
                boxes_velo[:, :, 1] += t[1]
                param['translate'] = t[1]
            if method[2] > 0.5:
                rgb, lidar = self.preprocess.flip(lidar, rgb, axis=1)
                boxes_velo[:, :, 1] = -boxes_velo[:, :, 1]
                param['flip'] = True
        top = self.preprocess.lidar_to_top(lidar)

        return rgb, top, lidar, param

    def compute_3D_box_velo(self, h, w, l, x, y, z, yaw):
        '''
        Return:3Xn in cam2 coordinate
        '''
        # 建立旋转矩阵R
        R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
        # 计算8个顶点坐标
        x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
        # 使用旋转矩阵变换坐标
        corners_3d_velo = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        # 最后在加上中心点
        corners_3d_velo += np.vstack([x, y, z])
        return corners_3d_velo

    def parseBox(self, bboxes, calib):

        boxes = []
        labels = []
        for bbox in bboxes:
            expand_box = np.concatenate((bbox['bbox'],np.ones([1,8])),axis = 0) #(4,8)
            temp = np.dot(np.linalg.pinv(calib['r_0']), expand_box)
            velo = np.dot(np.linalg.pinv(calib['v2c']), temp).T
            boxes.append(velo[:, :-1])
            labels.append(int(bbox['label']))
        return np.array(boxes), labels


    def get_shape(self):
        # todo for tracking, it means wasted a frame which will cause offset.
        train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, _, _, _, _ = self.load()
        top_shape = train_tops[0].shape
        front_shape = train_fronts[0].shape
        rgb_shape = train_rgbs[0].shape

        return top_shape, front_shape, rgb_shape

    def data_preprocessed(self):
        # only feed in frames with ground truth labels and bboxes during training, or the training nets will break.
        skip_frames = True
        calib = None

        while skip_frames:
            fronts = []
            frame_tag = self.tags[self.tag_index]
            if self.is_raw:
                # 这里boxes3d包含了这一帧中所有的box, label 默认只检查车
                obstacles, rgb, lidar = self.load_from_one_tag(frame_tag)
                rgb, top, boxes3d, labels = self.preprocess_one_frame(rgb, lidar, obstacles)
                # 这里box为(8,3)大小
            else:
                frame_tag = 'training/' + str(frame_tag).zfill(5)
                #frame_tag = 'training/06772'
                # 这里出来的是rect坐标系下的box， lidar已经被裁切过范围
                boxes_rect, rgb, lidar, calib = self.load_from_one_tag(frame_tag)
                # 投影到Velo下，labels是所有标签，这里只做car，即0-1
                boxes_velo, labels = self.parseBox(boxes_rect, calib)
                # 预处理，这里bbox在RGB的调整全部在后面做
                if len(boxes_velo) != 0:
                    rgb, top, lidar, param = self.preprocess_one_3d_frame(rgb, lidar, boxes_velo)
                fronts, move = self.GetFront(lidar, calib)

            if self.require_log and not self.is_testset:
                draw_bbox_on_rgb(rgb, boxes_velo, frame_tag)
                draw_bbox_on_lidar_top(top, boxes_velo, frame_tag)

            self.tag_index += 1

            # reset self tag_index to 0 and shuffle tag list
            if self.tag_index >= self.size:
                self.tag_index = 0
                if self.shuffled:
                    self.tags = shuffle(self.tags, random_state=self.random_num)
            skip_frames = False

            # 直接在这一部将dont care去掉，他们不在视野范围内
            if not self.is_testset:
                is_gt_inside_range, batch_gt_labels_in_range, batch_gt_boxes3d_in_range = \
                    self.keep_gt_inside_range(labels, boxes_velo)
                labels = batch_gt_labels_in_range
                boxes_velo = batch_gt_boxes3d_in_range

                # if no gt labels inside defined range, discard this training frame.
                if not is_gt_inside_range:
                    skip_frames = True

        return np.array([rgb]), np.array([top]), np.array([fronts]), np.array([labels]), \
               np.array([boxes_velo]), calib, frame_tag, move, param,

    @staticmethod
    def coord_transfer(points):

        v_fov = (-24.8, 2.0)
        # 这里注意 需要反转y才可以获得与image对应的FV
        x_lidar = points[:, 0]
        y_lidar = -points[:, 1]
        z_lidar = points[:, 2]
        r_lidar = points[:, 3]  # Reflectance
        # Distance relative to origin when looked from top

        # Absolute distance relative to origin
        d_lidar = np.sqrt(x_lidar ** 2 + y_lidar ** 2, z_lidar ** 2)

        yaw = np.arctan2(y_lidar, x_lidar)
        pitch = np.arcsin(z_lidar / d_lidar)

        fov_up = (v_fov[1] / 180) * np.pi
        fov_down = (v_fov[0] / 180) * np.pi
        fov_rad = fov_up - fov_down

        v = 0.5 * (yaw / np.pi + 1.0) * 1024
        u = (1.0 - (pitch + abs(fov_down)) / fov_rad) * 64

        return u, v, r_lidar, d_lidar, z_lidar

    def GetFront(self, lidar, calib):

        u, v, r_lidar, d_lidar, z_lidar = self.coord_transfer(lidar)

        min_v = min(v)
        min_u = min(u)

        v -= min(v)
        u -= min(u)

        u[np.isnan(u)] = 0
        v[np.isnan(v)] = 0

        u_bound = int(max(u)) + 1
        if u_bound < 64:
            u_bound = 64

        v_bound = int(max(v)) + 1
        if v_bound < 512:
            v_bound = 512

        front = np.zeros((u_bound, v_bound, 3))

        for idx in range(u.shape[0]):
            front[int(u[idx]), int(v[idx]), 0] = r_lidar[idx]
            front[int(u[idx]), int(v[idx]), 1] = d_lidar[idx]
            front[int(u[idx]), int(v[idx]), 2] = z_lidar[idx]

        return front[:64, :512, :], (min_v, min_u)


    def find_empty_block(self):
        idx = -1
        for i in range(self.cache_size):
            if self.blocks_usage[i] == 1:
                continue
            else:
                idx = i
                break
        return idx

    def loader(self):
        if use_thread:
            while self.loader_need_exit.value == 0:

                if len(self.prepr_data) >= self.cache_size:
                    time.sleep(1)
                    # print('sleep ')
                else:
                    self.prepr_data = [(self.data_preprocessed())] + self.prepr_data
                    # print('data_preprocessed')
        else:
            while self.loader_need_exit.value == 0:
                empty_idx = self.find_empty_block()
                if empty_idx == -1:
                    time.sleep(1)
                    # print('sleep ')
                else:
                    prepr_data = (self.data_preprocessed())
                    # print('data_preprocessed')
                    dumps = pickle.dumps(prepr_data)
                    length = len(dumps)
                    self.buffer_blocks[empty_idx][0:length] = dumps[0:length]

                    self.preproc_data_queue.put({
                        'index': empty_idx,
                        'length': length
                    })

        if self.require_log: print('loader exit')

    def load(self):
        if use_thread:
            while len(self.prepr_data) == 0:
                time.sleep(1)
            data_ori = self.prepr_data.pop()

        else:

            # print('self.preproc_data_queue.qsize() = ', self.preproc_data_queue.qsize())
            info = self.preproc_data_queue.get(block=True)
            length = info['length']
            block_index = info['index']
            dumps = self.buffer_blocks[block_index][0:length]

            # set flag
            self.blocks_usage[block_index] = 0

            # convert to bytes string
            dumps = array.array('B', dumps).tostring()
            data_ori = pickle.loads(dumps)

        return data_ori

    def get_frame_info(self):
        return self.tags[self.tag_index]


if __name__ == '__main__':
    # testing image testing, single frames
    # batch frame testing.
    dataset_dir = cfg.PREPROCESSED_DATA_SETS_DIR

    dates_to_drivers = {'1': ['11']}
    # dates_to_drivers = {'Round1Test': ['19_f2']}
    # load_indexs = None
    # batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs, is_testset=True)
    # # get_shape is used for getting shape.
    # top_shape, front_shape, rgb_shape = batches.get_shape()
    # for i in range(1000):
    #     train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d = batches.load(2, batch=True,
    #                                                                                            shuffled=False)

    # this code is for single testing.
    # load_indexs = ['00000', '00001', '00002', '00003']
    # batches = batch_loading(dataset_dir, dates_to_drivers, load_indexs, is_testset=True)
    #
    # for i in range(1000):
    #     train_rgbs, train_tops, train_fronts, train_gt_labels, train_gt_boxes3d, handle_id = batches.load(1, False)
    train_key_list = ['nissan_pulling_away',
                      'nissan_pulling_up_to_it',
                      'suburu_follows_capture',
                      'nissan_pulling_to_left',
                      'nissan_driving_past_it',
                      'nissan_pulling_to_right',
                      'suburu_driving_away',
                      'nissan_following_long',
                      'suburu_driving_parallel',
                      'suburu_driving_towards_it',
                      'suburu_pulling_to_left',
                      'suburu_not_visible',

                      'suburu_leading_front_left',
                      'ped_train',
                      'bmw_following_long',
                      'cmax_following_long',
                      'suburu_following_long',
                      'suburu_driving_past_it',
                      'nissan_brief',
                      'suburu_leading_at_distance']

    train_key_full_path_list = [os.path.join(cfg.RAW_DATA_SETS_DIR, key) for key in train_key_list]
    train_value_list = [os.listdir(value)[0] for value in train_key_full_path_list]

    train_n_val_dataset = [k + '/' + v for k, v in zip(train_key_list, train_value_list)]

    splitter = TrainingValDataSplitter(train_n_val_dataset)

    # bl = BatchLoading2(splitter.training_bags, splitter.training_tags)

    with BatchLoading2(tags=splitter.training_tags, require_shuffle=True, random_num=np.random.randint(100)) as bl:
        time.sleep(5)
        for i in range(5):
            t0 = time.time()
            data = bl.load()
            print('use time =', time.time() - t0)
            print(data)
            time.sleep(3)

        print('Done')
