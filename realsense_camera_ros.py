
"""
File: realsense_camera_ros.py
Author: Xiaomeng Fang
Description: Rospy (ROS1) script for realsense camera

History:
    - Version 0.0 (2024-01-26): careated

Dependencies:
    - rospy
"""

import rospy
import cv2
import numpy as np
from plyfile import PlyData, PlyElement
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
import matplotlib.pyplot as plt
from realsense2_camera.msg import Extrinsics
import open3d as o3d
from packaging import version

TIME_MAX = 10.0

class RealSenseCameraROS(object):
    
    def __init__(self, topic_prefix="/camera"):
        self.topic_prefix = topic_prefix    
        self.flange2camera = None
        self.depth_to_color_extrinsics = None
        self.color_camera_info = None

    def read_ply(self, file_name):
        """Read XYZ point cloud from filename PLY file"""
        ply_data = PlyData.read(file_name)
        pc = ply_data["vertex"].data
        pc_array = np.array([[x, y, z] for x, y, z in pc])
        return pc_array

    def write_ply(self, save_path, points, text=True):
        """
        save_path: path to save: '/yy/XX.ply'
        points: (N,3)
        """
        points = [
            (points[i, 0], points[i, 1], points[i, 2]) for i in range(points.shape[0])
        ]
        vertex = np.array(points, dtype=[("x", "f4"), ("y", "f4"), ("z", "f4")])
        el = PlyElement.describe(vertex, "vertex", comments=["verteces"])
        PlyData([el], text=text).write(save_path)
        
    def get_rgb_image(self):
        """Subscribe rgb image."""
        rgb = rospy.wait_for_message(f"{self.topic_prefix}/color/image_raw", Image, timeout=TIME_MAX)
        cb = CvBridge()
        if rgb is not None:
            rospy.loginfo("Received message from realsense camera.")
            rgb_frame = cb.imgmsg_to_cv2(rgb, "bgr8")
            rgb_frame = rgb_frame[:, :, [2, 1, 0]]
            return rgb_frame
        else:
            rospy.logwarn("One or more message are not received within the timeout.")

    def get_ir_and_rgb_image(self):
        """Subscribe rgb, left_ir, and right_ir."""
        rgb = rospy.wait_for_message(f"{self.topic_prefix}/color/image_raw", Image, timeout=TIME_MAX)
        ir_l = rospy.wait_for_message(f"{self.topic_prefix}/infra1/image_rect_raw", Image, timeout=TIME_MAX)
        ir_r = rospy.wait_for_message(f"{self.topic_prefix}/infra2/image_rect_raw", Image, timeout=TIME_MAX)
        cb = CvBridge()
        if rgb is not None and ir_l is not None and ir_r is not None:
            rospy.loginfo("Received message from realsense camera.")
            rgb_frame = cb.imgmsg_to_cv2(rgb, "bgr8")
            ir_l_frame = cb.imgmsg_to_cv2(ir_l, "bgr8")
            ir_r_frame = cb.imgmsg_to_cv2(ir_r, "bgr8")
            rgb_frame = rgb_frame[:, :, [2, 1, 0]]
            return rgb_frame, ir_l_frame, ir_r_frame
        else:
            rospy.logwarn("One or more message are not received within the timeout.")

    def get_rgbd_image(self, is_align:bool=False):
        """Get rgb and depth image, need to align."""
        if is_align:
            depth = rospy.wait_for_message(f"{self.topic_prefix}/aligned_depth_to_color/image_raw", Image, timeout=TIME_MAX)
        else:
            depth = rospy.wait_for_message(f"{self.topic_prefix}/depth/image_rect_raw", Image, timeout=TIME_MAX)
        rgb = rospy.wait_for_message(f"{self.topic_prefix}/color/image_raw", Image, timeout=TIME_MAX)
        cb = CvBridge()
        if rgb is not None and depth is not None:
            rospy.loginfo("Received message from realsense camera.")
            rgb_frame = cb.imgmsg_to_cv2(rgb, "bgr8")
            depth_frame = cb.imgmsg_to_cv2(depth, "passthrough")
            rgb_frame = rgb_frame[:, :, [2, 1, 0]]
            return rgb_frame, depth_frame
        else:
            rospy.logwarn("One or more message are not received within the timeout.")
             
    def get_rgbd_and_ir_image(self):
        """Subscribe rgbd, left_ir, and right_ir."""
        depth = rospy.wait_for_message(f"{self.topic_prefix}/depth/image_rect_raw", Image, timeout=TIME_MAX)
        rgb = rospy.wait_for_message(f"{self.topic_prefix}/color/image_raw", Image, timeout=TIME_MAX)
        ir_l = rospy.wait_for_message(f"{self.topic_prefix}/infra1/image_rect_raw", Image, timeout=TIME_MAX)
        ir_r = rospy.wait_for_message(f"{self.topic_prefix}/infra2/image_rect_raw", Image, timeout=TIME_MAX)
        cb = CvBridge()
        if rgb is not None and ir_l is not None and ir_r is not None and depth is not None:
            # print("Received message from realsense camera.")
            depth_frame = cb.imgmsg_to_cv2(depth, "passthrough")
            rgb_frame = cb.imgmsg_to_cv2(rgb, "bgr8")
            ir_l_frame = cb.imgmsg_to_cv2(ir_l, "bgr8")
            ir_r_frame = cb.imgmsg_to_cv2(ir_r, "bgr8")
            rgb_frame = rgb_frame[:, :, [2, 1, 0]]
            return ir_l_frame, ir_r_frame, rgb_frame, depth_frame
        else:
            # print("One or more message are not received within the timeout.")     
            return None, None, None, None   
    
    def get_depth_image(self):
        """Get depth image."""
        depth = rospy.wait_for_message(f"{self.topic_prefix}/aligned_depth_to_color/image_raw", Image, timeout=5.0)
        cb = CvBridge()
        if depth is not None:
            rospy.loginfo("Received message from realsense camera.")
            depth_frame = cb.imgmsg_to_cv2(depth, "passthrough")
            return depth_frame
        else:
            rospy.logwarn("One or more message are not received within the timeout.")
                    
    def is_point_in_mask(self, point: np.array, mask: np.array):
        """Check if a point is in a mask."""
        # kernal = np.ones((31, 31), np.uint8)
        # fig = plt.figure()
        # plt.imshow(mask)
        # plt.show()
        # mask = cv2.dilate(mask, kernal, iterations=1)
        # fig = plt.figure()
        # plt.imshow(mask)
        # plt.show()
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        camera_info = rospy.wait_for_message(
            f"{self.topic_prefix}/color/camera_info", CameraInfo, timeout=1.0
        )
        # K: [455.6827087402344, 0.0, 326.03302001953125, 0.0, 454.86822509765625, 180.9109649658203, 0.0, 0.0, 1.0]
        camera_matrix = np.array(camera_info.K).reshape((3, 3))
        dist_coeffs = np.array(camera_info.D)
        image_point, _ = cv2.projectPoints(point, rvec, tvec, camera_matrix, dist_coeffs)
        image_point = image_point.squeeze().astype(np.int)
        # y = image_point[0] if image_point[0] < camera_info.height else (image_point[0] - 10)
        # x = image_point[1] if image_point[1] < camera_info.width else (image_point[1] - 10)
        x = np.clip(image_point[0], 0, camera_info.width - 1)
        y = np.clip(image_point[1], 0, camera_info.height - 1)
        # print(image_point, x, y, camera_info.width, camera_info.height)
        # return mask[image_point[1], image_point[0]] == True
        # print(mask.shape)
        return mask[y, x] == True

    def translation_to_2d(self, translation):
        rvec = np.zeros((3, 1))
        tvec = np.zeros((3, 1))
        camera_info = rospy.wait_for_message(
            f"{self.topic_prefix}/color/camera_info", CameraInfo, timeout=1.0
        )
        camera_matrix = np.array(camera_info.K).reshape((3, 3))
        dist_coeffs = np.array(camera_info.D)
        image_point, _ = cv2.projectPoints(translation, rvec, tvec, camera_matrix, dist_coeffs)
        image_point = image_point.squeeze().astype(np.int)
        return image_point
    
    def pixel_to_point(self, pixel_pos, depth, intrinsics, extrinsics, scale=1.0):
        depth = depth / scale
        u_coords = pixel_pos[1] # x
        v_coords = pixel_pos[0] # y
        if depth != 0:
            fx = intrinsics[0, 0]
            fy = intrinsics[1, 1]
            cx = intrinsics[0, 2]
            cy = intrinsics[1, 2]
            points_camera = np.vstack([(u_coords - cx) * depth / fx, (v_coords - cy) * depth / fy, depth]).T
            
            rotation_matrix = extrinsics[:3, :3]
            translation_vector = extrinsics[:3, 3]
            point_world = rotation_matrix.dot(points_camera.T).T + translation_vector
        else:
            point_world = np.array([0, 0, 0])
        return point_world
    
    def depth_image_to_point_cloud(self, depth_image: np.ndarray, intrinsics: np.ndarray, extrinsics: np.ndarray):
        """
        create a point cloud from a depth image and a camera pose.
        """
        # get the height and width of the depth map
        height, width = depth_image.shape

        # get the focal length and the center of the camera
        fx, fy = intrinsics[0, 0], intrinsics[1, 1]
        cx, cy = intrinsics[0, 2], intrinsics[1, 2]

        # get the rotation matrix and the translation vector
        rotation_matrix = extrinsics[:3, :3]
        translation_vector = extrinsics[:3, 3]

        # calculate the coordinates of the points in the camera coordinate system
        u_coords, v_coords = np.meshgrid(np.arange(width), np.arange(height))
        u_coords = u_coords.flatten()
        v_coords = v_coords.flatten()
        depths = depth_image[v_coords, u_coords]

        points_camera = np.vstack([(u_coords - cx) * depths / fx, (v_coords - cy) * depths / fy, depths]).T

        # transform the points to the world coordinate system
        pcd = rotation_matrix.dot(points_camera.T).T + translation_vector

        return pcd
    
    def create_point_cloud(self, 
                           depth=None, 
                           rgb=None, 
                           mask=None, 
                           intrinsics=None, 
                           camera_pose=None, 
                           scale = 1000.0):
        """Create a point cloud from depth and RGB images

        Args:
            depth (np.ndarray, optional): depth image
            rgb (np.ndarray, optional): rgb image
            mask (np.ndarray, optional): mask image. Defaults to None.
            intrinsics (np.ndarray, optional): color intrinsics. Defaults to None. 3x3 matrix
            camera_pose (np.ndarray, optional): camera pose. Defaults to None. 4x4 matrix
            scale (float, optional): depth scale. Defaults to 1000.0.

        Returns:
            pcd (o3d.geometry.PointCloud()): point cloud
            points (np.ndarray): 3D points
            colors (np.ndarray): colors
        """
        if depth is None or rgb is None:
            rgb, depth = self.get_rgbd_image()
        depth = depth/scale
        if mask is not None:
            depth = depth * mask
        height, width = depth.shape
        if intrinsics is None:
            color_intrinsics, depth_intrinsics = self.get_intrinsics()
        fx, fy, cx, cy = color_intrinsics[0, 0], color_intrinsics[1, 1], color_intrinsics[0, 2], color_intrinsics[1, 2]

        # Create a meshgrid of pixel coordinates
        x = np.linspace(0, width - 1, width)
        y = np.linspace(0, height - 1, height)
        u, v = np.meshgrid(x, y)

        # Backproject depth to 3D points
        z = depth.flatten()
        x = (u.flatten() - cx) * z / fx
        y = (v.flatten() - cy) * z / fy
        points_camera = np.vstack((x, y, z)).T

        if camera_pose is None:
            points = points_camera
        else:
            rotation_matrix = camera_pose[:3, :3]
            translation_vector = camera_pose[:3, 3]
            points = rotation_matrix.dot(points_camera.T).T + translation_vector
        
        # Flatten the RGB image and combine with points
        colors = rgb.reshape(-1, 3) / 255  # Normalize colors

        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        return pcd, points, colors
    
    def get_point_cloud(self, 
                        depth=None, 
                        rgb=None, 
                        mask=None, 
                        camera_pose=None, 
                        intrinsics=None, 
                        scale = 1000.0):
        """Create a point cloud from depth and RGB images

        Args:
            depth (np.ndarray, optional): depth image
            rgb (np.ndarray, optional): rgb image
            mask (np.ndarray, optional): mask image. Defaults to None.
            intrinsics (np.ndarray, optional): color intrinsics. Defaults to None. 3x3 matrix
            camera_pose (np.ndarray, optional): camera pose. Defaults to None. 4x4 matrix
            scale (float, optional): depth scale. Defaults to 1000.0.

        Returns:
            pcd (o3d.geometry.PointCloud()): point cloud
        """
        pcd, _, _ = self.create_point_cloud(depth, rgb, mask, intrinsics, camera_pose, scale)
        return pcd

    def get_intrinsics(self):
        """Get intrinsics from camera."""
        camera_info = rospy.wait_for_message(
            f"{self.topic_prefix}/color/camera_info", CameraInfo, timeout=TIME_MAX
        )
        color_intrinsics = np.array(camera_info.K).reshape((3, 3))
        camera_info = rospy.wait_for_message(
            f"{self.topic_prefix}/depth/camera_info", CameraInfo, timeout=TIME_MAX
        )
        depth_intrinsics = np.array(camera_info.K).reshape((3, 3))
        return color_intrinsics, depth_intrinsics
    
    def get_extrinsics(self):
        """Get extrinsics from camera."""
        extrinsics = rospy.wait_for_message(
            f"{self.topic_prefix}/extrinsics/depth_to_color", Extrinsics, timeout=TIME_MAX
        )
        R = np.array(extrinsics.rotation).reshape((3, 3))
        t = np.array(extrinsics.translation)
        return R, t
    
    def align_depth_to_color(self):
        """Align depth image to color image."""
        
        color_intrinsics, depth_intrinsics = self.get_intrinsics()
        
        # depth intrinsics
        fx_d = depth_intrinsics[0, 0]
        fy_d = depth_intrinsics[1, 1]
        cx_d = depth_intrinsics[0, 2]
        cy_d = depth_intrinsics[1, 2]
        
        # color intrinsics 
        fx_rgb = color_intrinsics[0, 0]
        fy_rgb = color_intrinsics[1, 1]
        cx_rgb = color_intrinsics[0, 2]
        cy_rgb = color_intrinsics[1, 2]

        # camera extrinsics
        R, T = self.get_extrinsics()

        # get images
        color_img, depth_img = self.get_rgbd_image(is_align=False)
        height, width = depth_img.shape

        # create aligned depth image
        aligned_depth_img = np.zeros((height, width), dtype=np.uint16)

        for v in range(height):
            for u in range(width):
                depth = depth_img[v, u] / 1000.0  
                if depth > 0:
                    z = depth
                    x = (u - cx_d) * z / fx_d
                    y = (v - cy_d) * z / fy_d
                    
                    xyz_depth = np.array([x, y, z])
                    xyz_rgb = np.dot(R, xyz_depth) + T
                    
                    u_rgb = int(fx_rgb * xyz_rgb[0] / xyz_rgb[2] + cx_rgb)
                    v_rgb = int(fy_rgb * xyz_rgb[1] / xyz_rgb[2] + cy_rgb)
                    
                    if 0 <= u_rgb < width and 0 <= v_rgb < height:
                        aligned_depth_img[v_rgb, u_rgb] = depth_img[v, u]
        
        return aligned_depth_img, color_img

    def rotation_vector_to_quaternion(self, rvec):
        # Normalize the rotation vector
        norm = np.linalg.norm(rvec)
        if norm < 1e-5:
            return np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
        axis = rvec / norm
        
        # Compute sine and cosine of half the rotation angle
        half_angle = norm / 2.0
        sin_half = np.sin(half_angle)
        cos_half = np.cos(half_angle)
        
        # Compute quaternion components
        qw = cos_half
        qx = axis[0] * sin_half
        qy = axis[1] * sin_half
        qz = axis[2] * sin_half
        
        return np.array([qx, qy, qz, qw])
    
    def get_aruco_marker_pose(self, visualizer=False):
        """get aruco marker pose

        Args:
            visualizer (bool, optional): whether to visualize the process. Defaults to False.

        Returns:
            np.ndarray: pose of the aruco marker
        """
        import cv2
        from cv2 import aruco
        rgb = self.get_rgb_image()
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        color_intr, _ = self.get_intrinsics()
        intr_matrix = color_intr
        intr_coeffs = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        if version.parse(cv2.__version__) >= version.parse("4.7.0"):
            aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_ARUCO_ORIGINAL)
            parameters = aruco.DetectorParameters()
            detector = aruco.ArucoDetector(aruco_dict, parameters)
            corners, ids, _ = detector.detectMarkers(rgb)
        else:
            aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
            parameters = aruco.DetectorParameters_create()
            corners, ids, _ = aruco.detectMarkers(rgb, aruco_dict, parameters)
        if ids is None:
            return None
        
        marker_length = 0.1 # 10 cm, the length of the marker's side
        rvec, tvec, _ = aruco.estimatePoseSingleMarkers(corners, marker_length, intr_matrix, intr_coeffs)

        if visualizer:
            try:
                aruco.drawDetectedMarkers(rgb, corners)
                axis_length = 0.05
                result_img = cv2.drawFrameAxes(rgb, intr_matrix, intr_coeffs, rvec, tvec, axis_length)
                cv2.imshow('RGB image', result_img)
                cv2.waitKey(0)
            except Exception as e:
                rospy.logerr(f"Error: {e}")
            
        quat = self.rotation_vector_to_quaternion(np.array(rvec[0][0]))
        pose = np.concatenate((tvec[0][0], quat))
        
        return pose
        
if __name__ == "__main__":
    import imageio
    import matplotlib.pyplot as plt
    
    rospy.init_node("realsense_camera")
    RSCR = RealSenseCameraROS(topic_prefix = '/front_head_camera')
    rgb = RSCR.get_rgb_image()
    plt.imshow(rgb)
    plt.show()
    
    RSCR.get_aruco_marker_pose(visualizer=True)
