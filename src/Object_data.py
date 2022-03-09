from config import cfg
from raw_data import read_objects
import os
import config
import numpy as np
import glob
import cv2

class Get_3ddata():

    def __init__(self, is_train_data):
        self.data_attr = is_train_data

    def Getpath(self,sub_dir):
        raw_dir = cfg.RAW_DATA_SETS_DIR
        mapping = {}
        # foreach dir1
        for dir1 in glob.glob(os.path.join(raw_dir, sub_dir, self.data_attr, '*')):
            files_path = glob.glob(os.path.join(dir1, '*'))
            files_path.sort()
            for i, file_path in enumerate(files_path):
                key = '%s/%05d' % (self.data_attr, i)
                mapping[key] = file_path
        return mapping


class Image_3ddata(Get_3ddata):

    def __init__(self, is_train_data):
        super().__init__(is_train_data)
        self.files_path_mapping= self.Getpath('image')

    def load(self, frame_tag:str)-> np.ndarray:

        img = cv2.imread(self.files_path_mapping[frame_tag])
        return img


class Lidar_3ddata(Get_3ddata):

    def __init__(self,is_train_data):
        super().__init__(is_train_data)
        self.files_path_mapping = self.Getpath('velodyne')

    def load(self, frame_tag: str) -> np.dtype:
        lidar = np.fromfile(self.files_path_mapping[frame_tag], np.float32)
        lidar = lidar.reshape((-1, 4))

        # LIMIT RANGE, NUMBER FROM PAPER, same effect as projection to 2d-rgb
        lidar = lidar[lidar[:, 0] > 0]
        lidar = lidar[lidar[:, 0] < 70.4]
        lidar = lidar[lidar[:, 1] > -40]
        lidar = lidar[lidar[:, 1] < 40]

        return lidar


class Tracklet_3ddata(Get_3ddata):

    def __init__(self, is_train_data):
        super().__init__(is_train_data)
        self.frames_object = self.Getpath('labels')
        self.labels_map = {
            # background
            'background': 0,
            # car
            'Van': 0, 'Truck': 0, 'Car': 1, 'Tram': 0,
            # Pedestrianx
            'Pedestrian': 0}


    def load(self, frame_tag: str):

        objs = self.frames_object[frame_tag]
        boxes = []
        with open(objs, 'r') as f:
            for line in f:
                box = {}
                fields = line.split(' ')
                # xmin，ymin，xmax，ymax
                Boudary_2d = np.array(fields[4:8], dtype=np.float32)
                # length, width, height
                h, w, l = np.array(fields[8:11], dtype=np.float32)
                # x,y,z
                x,y,z = np.array(fields[11:14], dtype=np.float32)
                # Rotation
                yaw = float(fields[14])
                R = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
                # 计算8个顶点坐标
                x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
                y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
                z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]
                # 使用旋转矩阵变换坐标
                corners_3d_cam_rect = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
                # 最后在加上中心点
                corners_3d_cam_rect += np.vstack([x, y, z])
                box['bbox'] = corners_3d_cam_rect
                if fields[0] in self.labels_map.keys():
                    label = self.labels_map[fields[0]] #if config.cfg.SINGLE_CLASS_DETECTION == False else 1
                else:
                    label = self.labels_map['background']
                box['label'] = label
                boxes.append(box)

        return boxes


class Calib_3ddata(Get_3ddata):

    def __init__(self, is_train_data):
        super().__init__(is_train_data)
        self.frames_object = self.Getpath('calib')

    def load(self, frame_tag: str):

        calib_parameter = {}
        objs = self.frames_object[frame_tag]
        projection = self.read_calib_file(objs)
        calib_parameter['p2'] = projection['P2'].reshape(3, 4)

        v2c = projection['Tr_velo_to_cam'].reshape(3, 4)
        calib_parameter['v2c'] = np.concatenate((v2c, np.array([0, 0, 0, 1]).reshape(1, -1)), axis=0)

        r_0 = projection['R0_rect'].reshape(3, 3)
        r_0_temp = np.concatenate((r_0, np.array([0, 0, 0]).reshape(1, 3)), axis=0)
        calib_parameter['r_0'] = np.concatenate((r_0_temp, np.array([0, 0, 0, 1]).reshape(4, 1)), axis=1)

        return calib_parameter

    def read_calib_file(self, filepath):
        ''' Read in a calibration file and parse into a dictionary.
        Ref: https://github.com/utiasSTARS/pykitti/blob/master/pykitti/utils.py
        '''
        data = {}
        with open(filepath, 'r') as f:
            for line in f.readlines():
                line = line.rstrip()
                if len(line) == 0: continue
                key, value = line.split(':', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

