import os
import numpy as np
from glob import glob
import cv2
import pandas as pd


class DatasetVisualizer:
    def __init__(self, dataset, vis_cfg, point_view=None, bev_opt=None):
        '''
        dataset: XxxReader class instance
        vis_cfg: e.g. {'image': {'sensor_id': 'camera_front'},
                       'point_cloud': {'sensor_id': 'lidar', 'axis'='optical'},
                       ...}
        point_view: 'bev' or 'image'
        bev_opt: e.g. {'resolution': 0.05, 'color': 'jet',
                       'range_min': [x_min, y_min, z_min], 'range_max': [x_max, y_max, z_max]
                      }
        '''
        self.dataset = dataset
        self.vis_cfg = vis_cfg
        self.point_view = point_view
        self.bev_opt = bev_opt
        self.functions = {'image': self.vis_image, 'box2d': self.vis_box2d,
                          'lane': self.vis_lane,
                          'point_cloud': self.vis_point_cloud,
                          'box3d': self.vis_box3d}
        self.results = {}
        self.categories = None

    def visualize(self, index, wait=0):
        self.result = {}
        self.categories = self.dataset.get_class(index)
        for key in self.vis_cfg:
            self.functions[key](index)
        cv2.waitKey(wait)

    def vis_image(self, index):
        image = self.dataset.get_image(index, **self.vis_cfg['image'])
        if image is not None:
            self.results['image'] = image
            cv2.imshow('image', image)

    def vis_box2d(self, index):
        bboxes = self.dataset.get_box2d(index, **self.vis_cfg['box2d'])
        img = self.dataset.get_image(index, **self.vis_cfg['box2d']).copy()
        for bbox, category in zip(bboxes, self.categories):
            y, x, h, w = bbox
            cv2.rectangle(img, (x-w//2, y-h//2), (x+w//2, y+h//2), (0, 255, 0), thickness=2)
            cv2.putText(img, category, (x-w//2, y-h//2-5), cv2.FONT_ITALIC, 0.5, (0, 0, 255), thickness=2)
        cv2.imshow('box2d', img)

    def vis_lane(self, index, color=(0, 255, 0), thickness=2):
        img = self.dataset.get_image(index, **self.vis_cfg['box2d']).copy()
        lane_data = self.dataset.get_lane(index)
        for lane_polygon in lane_data:
            lane_polygon = np.array(lane_polygon).reshape(-1, 2).astype(np.int32)
            for point in lane_polygon:
                cv2.circle(img, tuple(point), 4, color, -1)  # -1 fills the circle

            cv2.polylines(img, [lane_polygon], isClosed=False, color=color, thickness=thickness)
        cv2.imshow('lane', img)


        lane_3p_img = self.dataset.get_image(index, **self.vis_cfg['box2d']).copy()
        lane_3p = lane_3_points(lane_data)
        for lane_polygon in lane_3p:
            lane_polygon = np.array(lane_polygon).reshape(-1, 2).astype(np.int32)
            for point in lane_polygon:
                cv2.circle(lane_3p_img, tuple(point), 4, color, -1)  # -1 fills the circle

            cv2.polylines(lane_3p_img, [lane_polygon], isClosed=False, color=color, thickness=thickness)

        cv2.imshow('lane3p', lane_3p_img)

    def vis_point_cloud(self, index):
        pcd = self.dataset.get_point_cloud(index, **self.vis_cfg['point_cloud'])
        if pcd is None:
            return
        if self.point_view == 'bev':
            box3d = self.dataset.get_box3d(index, **self.vis_cfg['box3d'])
            bev = self.convert_bev(pcd, self.bev_opt, box3d)
            self.results['bev'] = bev
            cv2.imshow('bev', bev)
        if self.point_view == 'image':
            depthmap = self.dataset.get_depth_map(index, **self.vis_cfg['point_cloud'])
            # intrinsic = self.dataset.get_intrinsic(**self.vis_cfg['intrinsic'])
            # pcd_img = self.draw_points_on_image(pcd, intrinsic, self.results['image'])
            self.results['pcd_img'] = depthmap
            cv2.imshow('pcd_img', depthmap)

    def convert_bev(self, pcd, bev_opt, box3d):

        R = np.array([[-1, 0, 0], [0, 0, -1], [0, -1, 0]]) # 90 0 180
        R_cam = np.array([[0, 0, -1], [-1, 0, 0], [0, -1, 0]])  # 90 0 -90

        box3d_corners = self.convert_box3d_bev(box3d, R_cam)
        pcd = (R @ pcd.T).T
        bev_shape = ((bev_opt['range_max'][1] - bev_opt['range_min'][1]) // bev_opt['resolution'],
                     (bev_opt['range_max'][0] - bev_opt['range_min'][0]) // bev_opt['resolution'])
        bev = np.zeros((int(bev_shape[0]), int(bev_shape[1]), 3), dtype=np.float32)
        offset = np.array([bev_opt['range_min'][1], bev_opt['range_min'][0], bev_opt['range_min'][2]])
        scale = 1 / bev_opt['resolution']
        pcd_bev = ((pcd - offset) * scale).astype(np.int32)
        pcd_bev = pcd_bev[(pcd_bev[:, 0] >= 0) & (pcd_bev[:, 0] < bev_shape[0]) &
                          (pcd_bev[:, 1] >= 0) & (pcd_bev[:, 1] < bev_shape[1])]
        bev[pcd_bev[:, 0], pcd_bev[:, 1], 0] = pcd_bev[:, 2]
        bev[pcd_bev[:, 0], pcd_bev[:, 1], 1] = pcd_bev[:, 2]
        bev[pcd_bev[:, 0], pcd_bev[:, 1], 2] = pcd_bev[:, 2]
        bev_draw = bev.astype(np.uint8)
        black = np.zeros(bev_draw.shape)
        bev_colored = self.pcl_img_to_colormap(bev_draw, black)

        box3d_corners = np.array([(bev_corner - offset[[1, 0], np.newaxis]) * scale \
                                  for bev_corner in box3d_corners]).astype(np.int32)
        bev_colored = self.draw_bev_box(box3d_corners, bev_colored)

        return bev_colored

    def pcl_img_to_colormap(self, src, dst):
        _, bi_img = cv2.threshold(src, 1, 255, cv2.THRESH_BINARY)
        min2 = np.unique(src)[1]
        src_clip = np.clip(src, min2, np.max(src))
        src_norm = (((src_clip-min2) / (np.max(src_clip)-min2)) * 255).astype(np.uint8)
        src_colored = cv2.applyColorMap(src_norm, cv2.COLORMAP_JET)
        dst = cv2.copyTo(src_colored, bi_img, dst)
        return dst

    def vis_box3d(self, index):
        box3d = self.dataset.get_box3d(index, **self.vis_cfg['box3d'])
        img = self.dataset.get_image(index, **self.vis_cfg['box3d']).copy()
        box3d_img = self.draw_box3d_image(box3d, img)
        self.results['box3d_img'] = box3d_img
        cv2.imshow('box3d', box3d_img)
        # bev = self.results['bev'] #if 'bev' in self.results else self.get_empty_bev()
        # self.results['bev'] = self.drwa_box3d_bev(box3d, bev)

    def convert_box3d_bev(self, box3d, R):
        bev_corners = []
        for b in box3d:
            corners_3d = self.convert_point_to_corners(b)
            bev_corners.append((R @ corners_3d)[:2, :4])
        return np.array(bev_corners).astype(np.float32)

    def draw_bev_box(self, points, bev_img):
        for point in points:
            for i in range(4):
                pt1 = (point[0, i], point[1, i])
                pt2 = (point[0, (i + 1) % 4], point[1, (i + 1) % 4])
                bev_img = cv2.line(bev_img, pt1, pt2, (0, 0, 255), 1)
        return bev_img

    def draw_box3d_image(self, box3d, img):
        for b in box3d:
            x, y, z, w, l, h, _, yaw, _ = b
            corners_3d_cam2 = self.convert_point_to_corners(b)
            intrinsic = self.dataset.get_intrinsic()
            pj_mat = np.hstack((intrinsic, np.zeros((3, 1))))
            corners_2d = self.project_to_image(corners_3d_cam2.T, pj_mat)
            corners_2d = corners_2d.astype(np.int32)

            for i in range(4):
                pt1 = (corners_2d[i][0], corners_2d[i][1])
                pt2 = (corners_2d[i + 4][0], corners_2d[i + 4][1])
                img = cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                pt1 = (corners_2d[i][0], corners_2d[i][1])
                pt2 = (corners_2d[(i + 1) % 4][0], corners_2d[(i + 1) % 4][1])
                img = cv2.line(img, pt1, pt2, (0, 255, 0), 2)

                pt1 = (corners_2d[i + 4][0], corners_2d[i + 4][1])
                pt2 = (corners_2d[(i + 1) % 4 + 4][0], corners_2d[(i + 1) % 4 + 4][1])
                img = cv2.line(img, pt1, pt2, (0, 255, 0), 2)
        return img

    def convert_point_to_corners(self, box3d):
        x, y, z, w, l, h, _, yaw, _ = box3d
        r_yaw = np.array([[np.cos(yaw), 0, np.sin(yaw)],
                            [0, 1, 0],
                            [-np.sin(yaw), 0, np.cos(yaw)]])
        x_corners = [l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2]
        y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
        z_corners = [w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2]

        R = r_yaw
        corners_3d_cam2 = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))
        corners_3d_cam2 += np.vstack([x, y, z])
        return corners_3d_cam2

    def project_to_image(self, pts_3d, P):
        pts_3d_hom = np.hstack((pts_3d, np.ones((pts_3d.shape[0], 1))))
        pts_2d_hom = np.dot(pts_3d_hom, P.T)
        pts_2d = pts_2d_hom[:, :2] / np.abs(pts_2d_hom[:, 2:3])
        return pts_2d


def calculate_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def find_center_point(lane):
    length_between_points = [calculate_distance(lane[i], lane[i+1]) for i in range(len(lane) - 1)]

    center_len = sum(length_between_points) / 2

    sum_len = 0
    center_idx = 0
    center_sub_len = 0
    for i, length in enumerate(length_between_points):
        before_sum_len = sum_len
        sum_len += length
        if center_len < sum_len:
            center_idx = i
            center_sub_len = center_len - before_sum_len
            break

    center_rate = center_sub_len / length_between_points[center_idx]
    center = (lane[center_idx+1] - lane[center_idx]) * center_rate + lane[center_idx]

    return center

def lane_3_points(lanes):
    lanes_3p = []
    for lane in lanes:
        lane = np.array(lane)
        center = find_center_point(lane)
        lanes_3p.append([lane[0], center, lane[-1]])

    return lanes_3p