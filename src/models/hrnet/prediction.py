import os
import pickle
import itertools
from typing import Dict, Tuple, Optional
from collections import defaultdict

import cv2
import numpy as np

from baseline.camera import Camera
from src.datatools.ellipse import POINTS_LEFT, PITCH_POINTS, POINTS_RIGHT, \
    INTERSECTON_TO_PITCH_POINTS, get_homography
from src.datatools.intersections import LINE_INTERSECTIONS

top_gates = [0, 1, 24, 25]  # Points above the ground plane

# Points planes
point_sets = {
    'groundplane': [i for i in range(58) if i not in top_gates],
    'goal_left': [0, 1, 2, 3, 6, 7, 10, 11, 12, 13],
    'goal_right': [18, 19, 22, 23, 24, 25, 26, 27, 28, 29]

}
IMG_SIZE = (960, 540)
keep_points = list(range(29))
keep_points.extend([40, 41, 42, 44, 45, 48, 51, 52, 55])


def swap_z_y(point_3d):
    point_3d = point_3d.copy()
    point_3d[0] = point_3d[1]
    point_3d[1] = point_3d[2]
    point_3d[2] = 0.0
    return point_3d


sets_transforms = {
    'groundplane': lambda x: x,
    'goal_left': swap_z_y,
    'goal_right': swap_z_y
}


class CameraCreator:
    def __init__(self, pitch: Dict[str, np.ndarray],
                 img_size: Tuple[int, int] = (960, 540),
                 conf_thresh: float = 0.2,
                 algorithm: str = 'opencv_calibration',
                 lines_file: str | None = None,
                 **kwargs):
        """Base class for camera calibration and all relevant heuristics
            management.

        Algorithms:
            opencv_calibration - Perform cv2.calibrateCamera with all available
                points on the pitch surface.
            opencv_calibration_multiplane - Similar to opencv_calibration,
                but it uses two additional planes with the goalposts.
            voter - Select several points subsets and pick the most likely
                camera model by assessing predicted camera model reprojection
                error and plausibility of the predicted values.
            original_voter - Perform camera calibration with selected points.
                Refine camera parameters if possible or return the camera model
                from homography if other algorithms do not work for the sample.
            iterative_voter: Try 'original_voter' approach with
                conf_thresh=0.5. If the result is not positive, iterate over
                the confidence thresholds provided in conf_threshs until
                parameters which generate the camera calibration values are
                found.

        Note: Any algorithm may return None if calibration is not possible.

        Args:
            pitch (Dict[str, np.ndarray]): Dict, which describes the pitch
                model. Keys are points names, values are np.ndarray with 3
                elements for 3D coordinates of the point.
            img_size (Tuple[int, int], optional): Prediction image resolution
                (W, H). Defaults to (960, 540).
            conf_thresh (float, optional): Points prediction confidence
                threshold. Should be in [0..1]. Defaults to 0.2.
            algorithm (str, optional): One of available algorithms:
                [opencv_calibration, opencv_calibration_multiplane,
                voter, iterative_voter, original_voter]. Note: Same algoritms
                requires additional parameters provided via kwargs. Defaults to
                'opencv_calibration'.
            lines_file (str | None, optional): Optional path to the lines
                prediction file. Lines are ignored if None. Defaults to None.

        """
        algorithms = {
            'opencv_calibration': self.opencv_calibration,
            'opencv_calibration_multiplane': self.opencv_calibration_multiplane,
            'voter': self.voter,
            'iterative_voter': self.iterative_voter,
            'original_voter': self.original_voter
        }
        assert algorithm in algorithms, \
            f'Should be one of: {list(algorithms.keys())}'

        self.algorithm = algorithms[algorithm]
        self.conf_thresh = conf_thresh
        self.pitch = pitch
        self.img_size = img_size
        self.lines_data = {}
        if lines_file is not None:
            assert os.path.exists(lines_file), f'{lines_file} does not exist'
            with open(lines_file, 'rb') as f:
                lines_data = pickle.load(f)

            for img_name in lines_data.keys():
                points = {}
                pred = lines_data[img_name]['lines'][0]
                if 'Goal left post left' in pred:
                    pred['Goal left post left '] = pred['Goal left post left']
                    del pred['Goal left post left']

                for idx, pair in LINE_INTERSECTIONS.items():
                    if pair[0] in pred and pair[1] in pred:
                        intersection_point = line_eq_intersection(
                            pred[pair[0]], pred[pair[1]])
                        if intersection_point is not None:
                            points[idx] = intersection_point
                if len(points) > 0:
                    self.lines_data[img_name] = points
        for key, value in kwargs.items():
            setattr(self, key, value)
        self.stat = {'n': 0, 'frames_4': 0,
                     'frames_4_6': 0, 'frames_bad_cam': 0}

    def __call__(self, pred, name: str | None = None) -> Optional[Camera]:
        cam = None
        try:
            cam = self.algorithm(pred, name)
        except Exception as e:
            print(f'Camera initialization exc: {e}')
        return cam

    def opencv_calibration(self, pred, name: str | None) -> Optional[Camera]:
        cam = None
        camera_points = []
        world_points = []
        conf = pred[:, 2]
        for i, conf in enumerate(conf):
            # Exclude points, which are not on the pitch (goal crossbars)
            if i not in top_gates and conf > self.conf_thresh:
                x = float(pred[i, 0])
                y = float(pred[i, 1])
                camera_points.append((x, y))
                world_points.append(self.pitch[INTERSECTON_TO_PITCH_POINTS[i]])
        # cv2.calibrateCamera requires at least 6 points
        if len(world_points) > 5:
            flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
            flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2\
                | cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4\
                | cv2.CALIB_FIX_TANGENT_DIST
            _, mtx, dist, rvect, tvect = cv2.calibrateCamera(
                np.array([world_points], dtype=np.float32),
                np.array([camera_points], dtype=np.float32),
                self.img_size, None, None, flags=flags)
            cam = Camera(*self.img_size)
            cam.calibration = mtx
            cam.xfocal_length = mtx[0, 0]
            cam.yfocal_length = mtx[1, 1]
            cam.principal_point = (self.img_size[0] / 2.0,
                                   self.img_size[1] / 2.0)
            R, _ = cv2.Rodrigues(rvect[0])

            cam.rotation = R
            cam.position = (- np.transpose(cam.rotation) @ tvect[0]).T[0]
        return cam

    def opencv_calibration_multiplane(self, pred, name: str | None)\
            -> Optional[Camera]:
        cam = None
        camera_points = defaultdict(Tuple[float, float])
        world_points_sampled = []
        camera_points_sampled = []
        n_det_points = np.count_nonzero(pred[:, 2] > self.conf_thresh)
        for i in range(pred.shape[0]):
            conf = pred[i, 2]
            if conf > self.conf_thresh and (n_det_points < self.reliable_thresh
                                            or i in keep_points):
                x = float(pred[i, 0])
                y = float(pred[i, 1])
                camera_points[i] = (x, y)
        line_points = self._get_points_from_lines(name)
        if len(line_points) > 0:
            for i in line_points:
                if i not in camera_points:
                    if len(camera_points) <= self.min_points:
                        camera_points[i] = line_points[i]
                        print('Point added', name, i, line_points[i])

        for point_set in point_sets:
            world_p = []
            camera_p = []
            for i in point_sets[point_set]:
                if i in camera_points:
                    world_p.append(
                        sets_transforms[point_set](self.pitch[
                            INTERSECTON_TO_PITCH_POINTS[i]]))
                    camera_p.append(camera_points[i])
            if len(camera_p) > 0:
                world_points_sampled.append(world_p)
                camera_points_sampled.append(camera_p)

        world_points_sampled = [np.array(sample, dtype=np.float32)
                                for sample in world_points_sampled
                                if len(sample) >= self.min_points_per_plane]
        camera_points_sampled = [np.array(sample, dtype=np.float32)
                                 for sample in camera_points_sampled
                                 if len(sample) >= self.min_points_per_plane]

        if len(camera_points_sampled) > 0 and len(camera_points)\
                > self.min_points:
            flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
            flags = flags | cv2.CALIB_FIX_TANGENT_DIST |\
                cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY

            flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |\
                cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |\
                cv2.CALIB_FIX_K6
            rms, mtx, dist, rvect, tvect = cv2.calibrateCamera(
                world_points_sampled, camera_points_sampled, self.img_size,
                None, None, flags=flags)
            # print(rms, mtx)
            if mtx[0, 0] > self.min_focal_length:
                cam = Camera(*self.img_size)
                cam.calibration = mtx
                cam.xfocal_length = mtx[0, 0]
                cam.yfocal_length = mtx[1, 1]
                cam.principal_point = (self.img_size[0] / 2.0,
                                       self.img_size[1] / 2.0)
                R, _ = cv2.Rodrigues(rvect[0])

                cam.rotation = R
                cam.position = (- np.transpose(cam.rotation) @ tvect[0]).T[0]
                if len(camera_points) > self.min_points_for_refinement:
                    matched_points = [
                        (self.pitch[INTERSECTON_TO_PITCH_POINTS[i]],
                         camera_points[i]) for i in camera_points]
                    cam.refine_camera(matched_points)
        return cam

    def iterative_voter(self, pred, name: str | None) -> Optional[Camera]:
        self.conf_thresh = 0.5
        try:
            cam = self.original_voter(pred, name)
            if cam is not None:
                return cam
        except:
            pass
        for p in self.conf_threshs:
            self.conf_thresh = p
            cam = self.voter(pred, name)
            if cam is not None:
                return cam

    def voter(self, pred, name: str | None) -> Optional[Camera]:
        print(f'Processing {name}@{self.conf_thresh}')
        cam = None
        camera_points = defaultdict(Tuple[float, float])
        for i in range(pred.shape[0]):
            conf = pred[i, 2]
            if conf > self.conf_thresh:
                x = float(pred[i, 0])
                y = float(pred[i, 1])
                camera_points[i] = (x, y)

        line_points = self._get_points_from_lines(name)
        if len(line_points) > 0:
            for i in line_points:
                if i not in camera_points:
                    if sum(key in point_sets['groundplane']
                           for key in camera_points)\
                            < self.min_points_per_plane:
                        camera_points[i] = line_points[i]
                        print('Point added', name, i, line_points[i])
        # camera_points = resolve_ambiguous_points(camera_points)
        print('Camera_points:', camera_points)
        hom_cam = None
        hom_cam_rmse = 10000  # Just a big default value
        hom_cam_pred = get_camera_from_homography(camera_points)
        if hom_cam_pred is not None:
            hom_cam, hom_cam_rmse = hom_cam_pred
            print("Homography prediction", name, hom_cam_rmse)

        camera_all_points = get_camera_all_points(camera_points)
        camera_reliable_points = get_camera_reliable_points(camera_points)
        camera_accurate_points = get_camera_accurate_points(camera_points, 5.0)
        gamera_ground_points = get_camera_groundplane_points(camera_points)
        cameras = []
        if camera_reliable_points is not None:
            cam_reliable, cam_reliable_rmse = camera_reliable_points
            good_cam = is_good_camera(cam_reliable)
            print('camera_reliable_points', cam_reliable_rmse, good_cam)
            if good_cam:
                cameras.append((cam_reliable, cam_reliable_rmse, 'camera_rel'))
        if camera_accurate_points is not None:
            cam_acc, cam_acc_rmse = camera_accurate_points
            good_cam = is_good_camera(cam_acc)
            print('camera_accurate_points', cam_acc_rmse, good_cam)
            if good_cam:
                cameras.append((cam_acc, cam_acc_rmse, 'camera_acc'))
        if camera_all_points is not None:
            cam_all, cam_all_rmse = camera_all_points
            good_cam = is_good_camera(cam_all)
            print('cam_all_points', cam_all_rmse, good_cam)
            if good_cam:
                cameras.append((cam_all, cam_all_rmse, 'cam_all'))
        if gamera_ground_points is not None:
            cam_ground, cam_ground_rmse = gamera_ground_points
            good_cam = is_good_camera(cam_ground)
            print('cam_ground_points', cam_ground_rmse, good_cam)
            if good_cam:
                cameras.append((cam_ground, cam_ground_rmse, 'cam_ground'))
        if len(cameras) > 0:
            min_cam = max(cameras, key=lambda x: (
                x[2] == 'camera_rel' and x[1] < self.max_rmse_rel, 1/x[1]))
            if min_cam[1] < self.max_rmse:
                cam = min_cam[0]
                print(f'Selected camera: {min_cam[2]}, '
                      f'RMSE: {min_cam[1]}')
            else:
                print(f'Unable to select camera, best RMSE {min_cam[1]}'
                      f'for {min_cam[2]}')
        if cam is None and hom_cam is not None and hom_cam_rmse < self.max_rmse:
            cam = hom_cam  # Use the camera from homography as the last resort
            print('return hom_cam', name, hom_cam_rmse)
        return cam

    def _get_points_from_lines(self, name: str | None = None):
        points = {}
        if self.lines_data is not None and name is not None\
                and name in self.lines_data:
            points = self.lines_data[name]
        return points

    def original_voter(self, pred, name: str | None) -> Optional[Camera]:
        cam = None
        camera_points = defaultdict(Tuple[float, float])
        world_points_sampled = []
        camera_points_sampled = []
        n_points_on_groundpalane = 0
        n_det_points = np.count_nonzero(pred[:, 2] > self.conf_thresh)
        for i in range(pred.shape[0]):
            conf = pred[i, 2]
            if conf > self.conf_thresh and (n_det_points < self.reliable_thresh
                                            or i in keep_points):
                x = float(pred[i, 0])
                y = float(pred[i, 1])
                camera_points[i] = (x, y)
                if i not in top_gates:
                    n_points_on_groundpalane += 1

        line_points = self._get_points_from_lines(name)
        if len(line_points) > 0:
            for i in line_points:
                if i not in camera_points:
                    if n_points_on_groundpalane < self.min_points_per_plane\
                        or (0 <= line_points[i][0] <= self.img_size[0] and
                            0 <= line_points[i][1] <= self.img_size[1]):
                        camera_points[i] = line_points[i]
                        print('Point added', name, i, line_points[i])
        matched_points = get_matched_points(camera_points)
        print('Camera_points:', camera_points)
        hom_cam = None
        hom_cam_rmse = 10000
        hom_cam_pred = get_camera_from_homography(camera_points)
        if hom_cam_pred is not None:
            hom_cam, hom_cam_rmse = hom_cam_pred
            print("Homography prediction", name, hom_cam_rmse)

        points_dict = {}
        for point_set in point_sets:
            points_dict[point_set] = {'world_p': [],
                                      'camera_p': []}
            for i in point_sets[point_set]:
                if i in camera_points:
                    points_dict[point_set]['world_p'].append(
                        sets_transforms[point_set](self.pitch[
                            INTERSECTON_TO_PITCH_POINTS[i]]))
                    points_dict[point_set]['camera_p'].append(camera_points[i])
            if len(points_dict[point_set]['camera_p']) > 0:
                world_points_sampled.append(points_dict[point_set]['world_p'])
                camera_points_sampled.append(
                    points_dict[point_set]['camera_p'])

        world_points_sampled = [np.array(sample, dtype=np.float32)
                                for sample in world_points_sampled
                                if len(sample) >= self.min_points_per_plane]
        camera_points_sampled = [np.array(sample, dtype=np.float32)
                                 for sample in camera_points_sampled
                                 if len(sample) >= self.min_points_per_plane]

        if len(camera_points_sampled) > 0 and len(camera_points)\
                > self.min_points:
            flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
            flags = flags | cv2.CALIB_FIX_TANGENT_DIST |\
                cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY

            flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |\
                cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |\
                cv2.CALIB_FIX_K6
            print(world_points_sampled, camera_points_sampled)
            rms, mtx, dist, rvect, tvect = cv2.calibrateCamera(
                world_points_sampled, camera_points_sampled, self.img_size,
                None, None, flags=flags)
            print(rms, mtx)

            cam = Camera(*self.img_size)
            cam.calibration = mtx
            cam.xfocal_length = mtx[0, 0]
            cam.yfocal_length = mtx[1, 1]
            cam.principal_point = (self.img_size[0] / 2.0,
                                   self.img_size[1] / 2.0)
            R, _ = cv2.Rodrigues(rvect[0])

            cam.rotation = R
            cam.position = (- np.transpose(cam.rotation) @ tvect[0]).T[0]

            if len(points_dict['groundplane']['camera_p'])\
                    < self.min_points_per_plane:
                cam.solve_pnp(matched_points)
            if not good_camera(cam.calibration, cam.position):
                # self.stat['frames_bad_cam'] += 1
                print('frames_bad_cam', name,
                      cam.position, cam.calibration[0, 0])
                print(cam.to_json_parameters())
                cam = None
            else:
                if len(camera_points) > self.min_points_for_refinement:
                    cam.refine_camera(matched_points)
        if cam is None and hom_cam is not None and hom_cam_rmse < 26:
            cam = hom_cam  # Use the camera from homography as the last resort
            print('return hom_cam', name)
        return cam


def resolve_ambiguous_points(camera_points: Dict[int, Tuple[float, float]],
                             threshold: float = 4):
    n_left = sum(key in POINTS_LEFT for key in camera_points.keys())
    n_right = sum(key in POINTS_RIGHT for key in camera_points.keys())
    thresh = threshold**2
    selected_points = {k: v for k, v in camera_points.items()}
    for pair in itertools.product(
            [i for i in camera_points if i in POINTS_RIGHT],
            [i for i in camera_points if i in POINTS_LEFT]):
        p1 = camera_points[pair[0]]
        p2 = camera_points[pair[1]]
        dist = (p1[0]-p2[0])**2+(p1[1]-p2[1])**2
        if dist <= thresh:
            if n_left > n_right:
                if pair[0] in selected_points:
                    del selected_points[pair[0]]
                    print('Deleted point:', pair[0])
            else:
                if pair[1] in selected_points:
                    del selected_points[pair[1]]
                    print('Deleted point:', pair[1])
    return selected_points


def get_matched_points(camera_points):
    return [(PITCH_POINTS[INTERSECTON_TO_PITCH_POINTS[i]],
             camera_points[i]) for i in camera_points]


def good_camera(mtx, pos):
    # Camera seems feasible if its focal length and position are reasonable
    focus = mtx[0, 0] >= 10 and mtx[0, 0] <= 20000
    pox_x = -250 < pos[0] < 250
    pos_y = -250 < pos[1] < 250
    pos_z = -100 < pos[2] < 0
    return focus and pox_x and pos_y and pos_z


def is_good_camera(cam: Camera):
    focus = cam.calibration[0, 0] >= 10 and cam.calibration[0, 0] <= 20000
    pos = cam.position
    pox_x = -250 < pos[0] < 250
    pos_y = -250 < pos[1] < 250
    pos_z = -100 < pos[2] < 0
    return focus and pox_x and pos_y and pos_z


def get_camera_from_homography(camera_points)\
        -> Optional[Tuple[Camera, float]]:
    """Perform camera calibratrion based on homography.

    Args:
        camera_points(List[Tuple[float, float]]): Detected points list.

    Returns:
        Camera: Camera, initilized from homography. None - if calibration is
            not posble.
        float: RMSE of the model projections.

    """
    cam_hom = Camera()
    world_p = []
    camera_p = []
    for i in point_sets['groundplane']:
        if i in camera_points:
            world_p.append(sets_transforms['groundplane'](PITCH_POINTS[
                INTERSECTON_TO_PITCH_POINTS[i]]))
            camera_p.append(camera_points[i])

    world_p = np.array(world_p, dtype=np.float32)
    camera_p = np.array(camera_p, dtype=np.float32)
    if camera_p.shape[0] >= 4:
        hom = get_homography(world_p, camera_p, 10)
        if hom is not None:
            cam_hom.estimate_calibration_matrix_from_plane_homography(hom)
            matched_points = get_matched_points(camera_points)
            cam_hom.solve_pnp(matched_points)
            cam_hom.refine_camera(matched_points)
            rmse = cam_hom.projection_rmse(matched_points)
            return cam_hom, rmse
    return None


def get_camera_all_points(camera_points):
    # General camera calibration with all the given points
    points_dict = {}
    world_points_sampled = []
    camera_points_sampled = []
    for point_set, point_ids in point_sets.items():
        points_dict[point_set] = {'world_p': [],
                                  'camera_p': []}
        for i in point_ids:
            if i in camera_points:
                points_dict[point_set]['world_p'].append(
                    sets_transforms[point_set](PITCH_POINTS[
                        INTERSECTON_TO_PITCH_POINTS[i]]))
                points_dict[point_set]['camera_p'].append(camera_points[i])
            if len(points_dict[point_set]['camera_p']) > 0:
                world_points_sampled.append(points_dict[point_set]['world_p'])
                camera_points_sampled.append(
                    points_dict[point_set]['camera_p'])

    world_points_list = [np.array(sample, dtype=np.float32)
                         for sample in world_points_sampled
                         if len(sample) >= 6]
    camera_points_list = [np.array(sample, dtype=np.float32)
                          for sample in camera_points_sampled
                          if len(sample) >= 6]
    try:
        cam = get_camera_gen(world_points_list, camera_points_list,
                             get_matched_points(camera_points),
                             len(points_dict['groundplane']))
        return cam
    except Exception as e:
        print(f'Camera init exception {e}')
        return None


def get_camera_reliable_points(camera_points):
    # Select reliable points only - intersections and points with clear marking
    camera_points_selected = {k: v for k, v in camera_points.items() if k in
                              keep_points}
    return get_camera_all_points(camera_points_selected)


def get_camera_groundplane_points(camera_points):
    # Calibrate camera with points on the groundplane only.
    camera_points_selected = {k: v for k, v in camera_points.items() if k in
                              point_sets['groundplane']}
    return get_camera_all_points(camera_points_selected)


def get_camera_accurate_points(camera_points, threshold: float = 10.0):
    # Select points, which are in aggreament with RANSAC initialized homography
    # i.e. subsample points which can be predicted accurately by homography
    # reprojection and then perform camera calibration with the points.
    camera_points_selected = {}
    world_p = []
    camera_p = []
    idxs = []
    for i in point_sets['groundplane']:
        if i in camera_points:
            world_p.append(sets_transforms['groundplane'](PITCH_POINTS[
                INTERSECTON_TO_PITCH_POINTS[i]]))
            camera_p.append(camera_points[i])
            idxs.append(i)

    world_p = np.array(world_p, dtype=np.float32)
    camera_p = np.array(camera_p, dtype=np.float32)
    if camera_p.shape[0] >= 4:
        hom = get_homography(world_p, camera_p, threshold)
        if hom is not None:
            world_h = np.concatenate(
                (world_p[:, :2], np.ones((world_p.shape[0], 1))), axis=-1)
            img_proj = []
            for d in world_h:
                img_h = hom @ d
                img_proj.append(img_h[:2] / img_h[2])
            rmse = np.linalg.norm(np.array(img_proj) - camera_p, 2, axis=1)
            for i in range(len(rmse)):
                if rmse[i] < threshold:
                    camera_points_selected[idxs[i]] = camera_points[idxs[i]]
            for i in top_gates:
                if i in camera_points:
                    camera_points_selected[i] = camera_points[i]
            print(f'N accurate points: {len(camera_points_selected)}')
            return get_camera_all_points(camera_points_selected)


def get_camera_gen(world_points_list, camera_points_list, matched_points,
                   n_groundplane: int):

    if len(world_points_list) > 0 and sum(len(s)
                                          for s in camera_points_list) > 6:
        flags = cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_FIX_ASPECT_RATIO
        flags = flags | cv2.CALIB_FIX_TANGENT_DIST |\
            cv2.CALIB_FIX_S1_S2_S3_S4 | cv2.CALIB_FIX_TAUX_TAUY

        flags = flags | cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 |\
            cv2.CALIB_FIX_K3 | cv2.CALIB_FIX_K4 | cv2.CALIB_FIX_K5 |\
            cv2.CALIB_FIX_K6
        rms, mtx, dist, rvect, tvect = cv2.calibrateCamera(
            world_points_list, camera_points_list, IMG_SIZE,
            None, None, flags=flags)

        cam = Camera(*IMG_SIZE)
        cam.calibration = mtx
        cam.xfocal_length = mtx[0, 0]
        cam.yfocal_length = mtx[1, 1]
        cam.principal_point = (IMG_SIZE[0] / 2.0, IMG_SIZE[1] / 2.0)
        R, _ = cv2.Rodrigues(rvect[0])

        cam.rotation = R
        cam.position = (- np.transpose(cam.rotation) @ tvect[0]).T[0]

        if n_groundplane < 6:
            cam.solve_pnp(matched_points)
        if len(matched_points) > 6:
            cam.refine_camera(matched_points)

        return cam, cam.projection_rmse(matched_points)


def line_eq_intersection(line1: Tuple[float, float],
                         line2: Tuple[float, float])\
        -> Optional[Tuple[float, float]]:
    """Intersection point of two lines."""
    k1, b1 = line1
    k2, b2 = line2
    if abs(k1-k2) > 1e-4:
        x = (b2 - b1) / (k1-k2)
        y = k1 * x + b1
        return (x, y)
    return None
