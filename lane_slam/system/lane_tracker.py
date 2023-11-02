#!/usr/bin/env python
# coding: utf-8
# @author: Zhijian Qiao
# @email: zqiaoac@connect.ust.hk
import os

import open3d as o3d
import numpy as np
from lane_slam.system.lane_ui import LaneUI
from lane_slam.assoc_utils import KnnASSOC
from lane_slam.frame import Frame
from misc.config import cfg
from typing import List, Optional, Dict
from lane_slam.lane_feature import LaneFeature
from misc.lie_utils import get_pose2d_noise, inv_se3, compute_rpe
from copy import deepcopy
from misc.plot_utils import visualize_points_list

class LaneTracker(LaneUI):
    def __init__(self, bag_file):
        super(LaneTracker, self).__init__(bag_file)
    def tracking_init(self):

        self.cur_frame:Frame = None
        self.prev_frame:Frame = None
        self.frames:Optional[Dict[int, Frame]] = {}
        self.gt_pose:Optional[List[np.ndarray]] = []
        self.raw_pose:Optional[List[np.ndarray]] = []
        self.time_stamp:Optional[List[float]] = []

        self.lanes_in_map:Optional[Dict[int, LaneFeature]] = {}
        self.assoc_model = KnnASSOC()

        self.max_lane_id = -1
        self.max_frame_id = -1

        self.odo_meas = None
        self.odo_noise = deepcopy(cfg.pose_update.odom_noise)

    def lane_association(self):

        #1.
        lm_list = self.get_candi_lanes(self.cur_frame)#获取地图中现存的车道线，  输入的cur_frame并没有使用
        self.assoc_model.set_landmark(lm_list)#构建地图中现有车道线的kdtree
        #2.
        det_list = self.cur_frame.get_lane_features()#获取当前帧的lane
        self.assoc_model.set_deteciton(det_list, self.cur_frame.T_wc)# 会计算当前帧每个点的阈值参数 ，并赋值给了det_knn_thd

        #very important function!!!!!
        #3.
        #返回的A是匹配对 是一个list 每个匹配对里面先是地图中的lane id  然后才是当前帧感知的laneid
        A, stats = self.assoc_model.association()

        # if cfg.debug_flag:
        #     self.visualize_association(lm_list, det_list, self.cur_frame.T_wc, A=A)
        #4.将匹配对数据结构进行处理
        A = np.array(A).reshape(-1, 2)
        #遍历当前帧的所有lane
        for i in range(len(det_list)):
            if i in A[:, 1]:
                j = A[A[:, 1] == i, 0][0]#拿到当前帧对应地图lane中的序号
                det_list[i].id = lm_list[j].id#改变当前帧感知中的序号，赋值为地图lane中的序号
            elif det_list[i].self_check():
                det_list[i].id = self.max_lane_id + 1
                self.max_lane_id += 1
            else:
                det_list[i].id = -1
                continue

    #主要是对当前帧的车道线根据range进行过滤，并生成当前帧的数据            
    def odometry(self, lane_pts_c, cam0_pose, timestamp):
        self.gt_pose.append(cam0_pose)
        if self.prev_frame is not None:
            self.odo_meas = inv_se3(self.gt_pose[-2]) @ self.gt_pose[-1]#矩阵乘法
            if self.add_odo_noise:
                pose_noisy = get_pose2d_noise(self.odo_noise[2], self.odo_noise[3])
                self.odo_meas = self.odo_meas @ pose_noisy
            T_wc = self.prev_frame.T_wc @ self.odo_meas
            self.raw_pose.append(self.raw_pose[-1] @ self.odo_meas)#just used for evaluation
        else:
            self.raw_pose.append(cam0_pose)#just used for evaluation
            T_wc = cam0_pose

        # transform lane points from current frame to world frame
        #对当前帧感知到的所有车道线进行range删除点
        lane_pts_c = self.get_lane_in_range(lane_pts_c)
        self.max_frame_id  = self.max_frame_id + 1
        self.cur_frame = Frame(self.max_frame_id, lane_pts_c, T_wc, timestamp)#very important function
        self.frames[self.max_frame_id] = self.cur_frame

    def get_candi_lanes(self, frame):
        return list(self.lanes_in_map.values())

    def evaluate_pose(self):
        gt_poses = self.gt_pose
        gt_points = [pose[:3, 3] for pose in gt_poses]
        est_poses = [frame.T_wc for frame in self.frames.values()]
        raw_poses = self.raw_pose
        num = len(gt_poses)
        assert len(gt_poses) == len(est_poses) == len(raw_poses)
        gt_delta = [0] + [np.linalg.norm(gt_points[i] - gt_points[i - 1]) for i in range(1, len(gt_points))]
        gt_distance = np.cumsum(gt_delta)
        intervals = cfg.evaluation.intervals

        stats = {}
        stats['path_length'] = gt_distance[-1]
        for interval in intervals:
            eval_pair = []
            for i in range(num):
                for j in range(i+1, num):
                    if gt_distance[j] - gt_distance[i] > interval:
                        eval_pair.append((i, j))
                        break

            error_rot = []
            error_trans = []
            error_rot_raw = []
            error_trans_raw = []
            for i, j in eval_pair:
                delta_deg, delta_xyz = compute_rpe(est_poses[i], est_poses[j], gt_poses[i], gt_poses[j])
                error_rot.append(delta_deg)
                error_trans.append(delta_xyz)
                delta_deg, delta_xyz = compute_rpe(raw_poses[i], raw_poses[j], gt_poses[i], gt_poses[j])
                error_rot_raw.append(delta_deg)
                error_trans_raw.append(delta_xyz)
            stats[interval] = {
                'error_rot': error_rot,
                'error_trans': error_trans,
                'error_rot_raw': error_rot_raw,
                'error_trans_raw': error_trans_raw,
            }
        # for interval, value in stats.items():
        #     if len(value['error_rot']) > 0:
        #         print('interval: ', interval,
        #               'error rot: {:.3f}/{:.3f}'.format(np.mean(value['error_rot']), np.mean(value['error_rot_raw'])),
        #               'error trans: {:.3f}/{:.3f}'.format(np.mean(value['error_trans']), np.mean(value['error_trans_raw'])))
        #
        # gt_points = np.concatenate([pose[:3, 3] for pose in gt_poses], axis=0).reshape(-1, 3)
        # est_points = np.concatenate([frame.T_wc[:3, 3] for frame in self.frames.values()], axis=0).reshape(-1, 3)
        # raw_points = np.concatenate([pose[:3, 3] for pose in raw_poses], axis=0).reshape(-1, 3)
        # visualize_points_list([gt_points, est_points, raw_points])

        return stats



