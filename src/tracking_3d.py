import net.utility.draw as draw
import numpy as np
import argparse

from tqdm import tqdm
import mv3d
import os
import cv2

from utils.batch_loading import BatchLoading2 as BatchLoading
from net.processing.boxes3d import decomposebox_3ddata


def str2bool(v):
    if v.lower() in ('true'):
        return True
    elif v.lower() in ('false'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='tracking')
    parser.add_argument('-n', '--tag', type=str, nargs='?', default='unknown_tag',
                        help='set log tag')
    parser.add_argument('-w', '--weights', type=str, nargs='?', default='',
                        help='set weights tag name')
    parser.add_argument('-t', '--fast_test', type=str2bool, nargs='?', default=False,
                        help='set fast_test model')
    parser.add_argument('-s', '--n_skip_frames', type=int, nargs='?', default=0,
                        help='set number of skip frames')
    args = parser.parse_args()

    print('\n\n{}\n\n'.format(args))
    tag = args.tag
    if tag == 'unknown_tag':
        tag = input('Enter log tag : ')
        print('\nSet log tag :"%s" ok !!\n' % tag)
    weights_tag = args.weights if args.weights != '' else None

    val_tags = list(np.load('val_tags.npy'))
    # val_tags = ['training/'+str(x).zfill(5) for x in val_tags]

    output_dir = ['../eval/prediction/data/', '../eval/prediction/picture/']
    temp = []
    for dir in output_dir:
        if not os.path.exists(dir):
            os.makedirs(dir)


    with BatchLoading(tags=val_tags, queue_size=1, require_shuffle=False, random_num=666, is_raw=False, is_valid=True) as validation:
        top_shape, front_shape, rgb_shape = validation.get_shape()
        predict = mv3d.Predictor(top_shape, front_shape, rgb_shape, log_tag=tag, weights_tag=weights_tag)

        for i in tqdm(range(validation.size - 1)):
            # 这里的坐标均在velo坐标系下
            rgb, top, front, label, bboxes, calib, frame_tag, move, param = validation.load()
            # 验证这里没问题
            if param:
                print('Not empty')
            # 这里的box是velo
            boxes3d, probs = predict(top, front, rgb, calib, move, param, Kitti_3d=True)
            img = draw.draw_box3d_on_camera(rgb[0], boxes3d,probs=probs,calib=calib)
            cv2.imwrite(output_dir[1]+frame_tag[-4:]+'.png', img)
            # 这里需要velo下坐标 ->
            all_data = decomposebox_3ddata(boxes3d, calib, probs)

            name = str(frame_tag).split('/')[1].zfill(6) + '.txt'
            filename = output_dir[0] + name
            f = open(filename, "w+")
            if all_data is not []:
                for line in all_data:
                        f.write(" ".join(str(x) for x in line) + '\n')
            f.close()
            temp.append(int(frame_tag.split('/')[1]))
            # break
        np.save('../eval/real_use', np.array(temp))
