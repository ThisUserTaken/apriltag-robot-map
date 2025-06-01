import cv2
import numpy as np
from pupil_apriltags import Detector
from collections import deque
import json
import time
import threading
from networktables import NetworkTables
from utils import detect_blur
import logging

class AprilTagDetector:
    def __init__(self, config_file="config.json"):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("AprilTagDetector")
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading config: {e}")
            raise

        self.calibration_mode = self.config["calibration"]
        self.calibration_positions = [tuple(pos) for pos in self.config["calibration_positions"]]
        self.current_position_idx = 0
        self.checkerboard_size = (6, 8)
        self.calibration_data = {'obj_points': [], 'img_points': []}
        self.config_file = config_file

        self.tag_size = self.config["tag"]["size"]
        self.tag_family = self.config["tag"]["family"]
        try:
            self.detector = Detector(
                families=self.tag_family,
                nthreads=self.config["detector"]["nthreads"],
                quad_decimate=self.config["detector"]["quad_decimate"],
                quad_sigma=self.config["detector"]["quad_sigma"],
                refine_edges=self.config["detector"]["refine_edges"],
                decode_sharpening=self.config["detector"]["decode_sharpening"]
            )
        except Exception as e:
            self.logger.error(f"Error initializing detector: {e}")
            raise

        self.camera_params = self.config["camera_params"]
        self.camera_matrix = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        self.dist_coeffs = np.array([-0.1, 0.05, 0.001, 0.001])

        self.map_scale = self.config["map"]["scale"]
        self.map_width = self.config["map"]["width"]
        self.map_height = self.config["map"]["height"]
        self.tag_positions = {}
        self.tag_yaws = {}
        self.tag_locations = {int(k): v for k, v in self.config["tag_locations"].items()}
        self.tag_rotations = {int(k): float(v) for k, v in self.config.get("tag_rotations", {}).items()}
        self.robot_position = None
        self.robot_yaw = None
        self.robot_position_history = deque(maxlen=5)

        self.display_interval_s = self.config["pipeline"].get("display_interval_s", 0.5)
        self.display_frame_interval = self.config["pipeline"].get("display_frame_interval", 5)
        self.headless = self.config["pipeline"].get("headless", False)

        from networktables_publisher import NetworkTablesPublisher
        self.nt_publisher = NetworkTablesPublisher(
            server=self.config["networktables"]["server"],
            table_name=self.config["networktables"]["table"]
        )

        self.controls = self.config["controls"]
        self.key_bindings = {k: tuple(v) for k, v in self.config["key_bindings"].items()}
        self.manual_mode = False

        from camera_manager import CameraManager
        self.camera_manager = CameraManager(self.config)
        self.frame_count = 0
        self.fps = 0.0
        self.prev_detections = {}
        self.running = False
        self.thread = None
        self.frame_queue = deque(maxlen=1)

    def start(self):
        if not self.camera_manager.start():
            self.logger.error("Failed to start camera, aborting")
            return
        self.running = True
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join()
        self.camera_manager.stop()
        self.nt_publisher.shutdown()
        if not self.headless:
            cv2.destroyAllWindows()

    def process_frames(self):
        frame_times = deque(maxlen=10)
        while self.running:
            start_time = time.time()
            try:
                frame = self.camera_manager.get_frame()
                if frame is None:
                    self.logger.warning(f"Frame {self.frame_count}: No valid frame received")
                    self.nt_publisher.publish_robot_location(None, float('nan'), float('nan'), self.frame_count)
                    time.sleep(0.1)  # Avoid busy loop
                    continue

                self.frame_queue.clear()
                if not self.detect_blur(frame):
                    self.frame_queue.append(frame)
                    self.frame_count += 1
                    current_time = time.time()
                    frame_times.append(current_time)
                    if len(frame_times) > 1:
                        self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    self.logger.debug(f"Frame acquisition time: {(current_time - start_time)*1000:.1f}ms, FPS: {self.fps:.1f}")
                else:
                    self.logger.debug("Skipped blurry frame")
                    self.nt_publisher.publish_robot_location(None, float('nan'), float('nan'), self.frame_count)

                if self.frame_queue:
                    process_start = time.time()
                    if self.calibration_mode:
                        self.calibrate_camera(self.frame_queue[-1])
                    self.detect_tags(self.frame_queue[-1])
                    self.logger.debug(f"Frame processing time: {(time.time() - process_start)*1000:.1f}ms")

            except Exception as e:
                self.logger.error(f"Error processing frame {self.frame_count}: {e}")
                self.nt_publisher.publish_robot_location(None, float('nan'), float('nan'), self.frame_count)
                time.sleep(0.1)

    def detect_blur(self, frame):
        return detect_blur(frame)

    def calibrate_camera(self, frame):
        start_time = time.time()
        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            x1, y1, x2, y2 = self.calibration_positions[self.current_position_idx]

            ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            self.logger.debug(f"Checkerboard detection: {ret}, Corners: {len(corners) if ret else 0}, Time: {(time.time() - start_time)*1000:.1f}ms")
            if ret:
                corners_refined = cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), criteria=(
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001))
                
                corners_x = corners_refined[:, 0, 0]
                corners_y = corners_refined[:, 0, 1]
                if (np.all(corners_x >= x1) and np.all(corners_x <= x2) and
                    np.all(corners_y >= y1) and np.all(corners_y <= y2)):
                    objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
                    objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
                    self.calibration_data['obj_points'].append(objp)
                    self.calibration_data['img_points'].append(corners_refined)
                    self.logger.info(f"Captured calibration image at position {self.current_position_idx + 1}")
                    self.current_position_idx += 1

                    if self.current_position_idx >= len(self.calibration_positions):
                        ret, mtx, dist, _, _ = cv2.calibrateCamera(
                            self.calibration_data['obj_points'], self.calibration_data['img_points'],
                            gray.shape[::-1], None, None)
                        if ret < 1:
                            self.camera_params = {
                                'fx': mtx[0, 0], 'fy': mtx[1, 1], 'cx': mtx[0, 2], 'cy': mtx[1, 2],
                                'dist_coeffs': dist.tolist()
                            }
                            self.camera_matrix = mtx
                            self.config["camera_params"] = self.camera_params
                            with open(self.config_file, 'w') as f:
                                json.dump(self.config, f, indent=2)
                            self.logger.info(f"Calibration complete, reprojection error: {ret:.3f}, updated config.json")
                            self.calibration_mode = False
                        else:
                            self.logger.error(f"Calibration failed, reprojection error too high: {ret:.3f}")
                        self.current_position_idx = 0
                        self.calibration_data = {'obj_points': [], 'img_points': []}
                else:
                    self.logger.debug("Checkerboard not fully within rectangle")
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")

    def update_map_and_publish(self, detections, frame_count):
        half_size = self.tag_size / 2
        obj_points = np.array([
            [-half_size, half_size, 0],
            [half_size, half_size, 0],
            [half_size, -half_size, 0],
            [-half_size, -half_size, 0]
        ], dtype=np.float32)

        for det in detections:
            tag_id = det.tag_id
            pose_t = det.pose_t.flatten()
            x, z = pose_t[0], pose_t[2]
            self.tag_positions[tag_id] = (x, z)

            try:
                image_points = det.corners.astype(np.float32)
                success, rvec, tvec = cv2.solvePnP(
                    obj_points, image_points, self.camera_matrix, self.dist_coeffs,
                    flags=cv2.SOLVEPNP_IPPE_SQUARE
                )
                if success:
                    R, _ = cv2.Rodrigues(rvec)
                    yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
                    expected_yaw = self.tag_rotations.get(tag_id, 0.0)
                    adjusted_yaw = (yaw - expected_yaw) % 360
                    if adjusted_yaw > 180:
                        adjusted_yaw -= 360
                    elif adjusted_yaw < -180:
                        adjusted_yaw += 360
                    self.tag_yaws[tag_id] = adjusted_yaw
                else:
                    self.tag_yaws[tag_id] = 0.0
                    self.logger.warning(f"Tag {tag_id} solvePnP failed")
            except Exception as e:
                self.tag_yaws[tag_id] = 0.0
                self.logger.error(f"Tag {tag_id} solvePnP error: {e}")

        self.nt_publisher.publish_tag_data(detections, frame_count, self.tag_yaws)

        robot_positions = []
        yaws = []
        for det in detections:
            tag_id = det.tag_id
            prob = 1.0 - det.decision_margin / 100.0
            self.logger.debug(f"Tag {tag_id} prob: {prob:.2f}")
            self.logger.debug(f"Tag {tag_id} relative pose: x={pose_t[0]:.3f}m, z={pose_t[2]:.3f}m, yaw={self.tag_yaws.get(tag_id, 0.0):.1f}°")
            if tag_id in self.tag_locations and prob >= 0.2:
                pose_t = det.pose_t.flatten()
                rel_x, rel_z = pose_t[0], pose_t[2]
                global_coords = self.tag_locations[tag_id]
                global_x, global_z = global_coords[0], global_coords[1]
                robot_x = global_x - rel_x
                robot_z = global_z - rel_z
                robot_positions.append([robot_x, robot_z])
                if tag_id in self.tag_yaws:
                    yaws.append(self.tag_yaws[tag_id])

        if robot_positions:
            robot_pos = np.mean(robot_positions, axis=0)
            self.robot_position_history.append(robot_pos)
            smoothed_pos = np.mean(self.robot_position_history, axis=0)
            self.robot_position = (smoothed_pos[0], smoothed_pos[1])
            self.robot_yaw = np.mean(yaws) if yaws else 0.0
            self.logger.info(f"Robot position: x={self.robot_position[0]:.3f}m, z={self.robot_position[1]:.3f}m, yaw={self.robot_yaw:.1f}°")
        else:
            self.logger.info("No valid tag detections for robot position")
            self.robot_position = None
            self.robot_yaw = None

        self.nt_publisher.publish_robot_location(self.robot_position, self.robot_yaw, frame_count)
        self.logger.debug(f"Frame {frame_count}: NetworkTables connected: {NetworkTables.isConnected()}")

    def draw_map(self):
        if self.headless:
            return
        map_frame = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255
        center_x, center_z = self.map_width // 2, self.map_height // 2

        for tag_id, (x, z) in self.tag_positions.items():
            px = int(center_x + x * self.map_scale)
            pz = int(center_z - z * self.map_scale)
            if 0 <= px < self.map_width and 0 <= pz < self.map_height:
                cv2.circle(map_frame, (px, pz), 5, (0, 0, 255), -1)
                cv2.putText(map_frame, f"ID: {tag_id}", (px + 10, pz),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
                if tag_id in self.tag_yaws:
                    yaw = self.tag_yaws[tag_id]
                    angle_rad = np.radians(yaw)
                    arrow_length = 15
                    arrow_end_x = px + int(arrow_length * np.cos(angle_rad))
                    arrow_end_z = pz - int(arrow_length * np.sin(angle_rad))
                    cv2.arrowedLine(map_frame, (px, pz), (arrow_end_x, arrow_end_z), (255, 0, 0), 1)

        if self.robot_position is not None:
            rx, rz = self.robot_position
            px = int(center_x + rx * self.map_scale)
            pz = int(center_z - rz * self.map_scale)
            if 0 <= px < self.map_width and 0 <= pz < self.map_height:
                cv2.circle(map_frame, (px, pz), 8, (0, 255, 0), -1)
                if self.robot_yaw is not None:
                    angle_rad = np.radians(self.robot_yaw)
                    arrow_length = 20
                    arrow_end_x = px + int(arrow_length * np.cos(angle_rad))
                    arrow_end_z = pz - int(arrow_length * np.sin(angle_rad))
                    cv2.arrowedLine(map_frame, (px, pz), (arrow_end_x, arrow_end_z), (0, 255, 0), 2)

        cv2.line(map_frame, (center_x - 50, center_z), (center_x + 50, center_z), (0, 0, 0), 1)
        cv2.line(map_frame, (center_x, center_z - 50), (center_x, center_z + 50), (0, 0, 0), 1)
        cv2.putText(map_frame, "X", (center_x + 60, center_z), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(map_frame, "Z", (center_x, center_z - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("2D AprilTag Map", map_frame)

    def detect_tags(self, frame):
        start_time = time.time()
        display_frame = frame.copy() if not self.headless or self.calibration_mode else None
        self.logger.debug(f"Rendering frame {self.frame_count}, calibration mode: {self.calibration_mode}, position: {self.current_position_idx}")
        
        if self.calibration_mode:
            if not self.headless:
                x1, y1, x2, y2 = self.calibration_positions[self.current_position_idx]
                self.logger.debug(f"Drawing rectangle: ({x1}, {y1}, {x2}, {y2})")
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(display_frame, f"Position {self.current_position_idx + 1}/{len(self.calibration_positions)}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                cv2.putText(display_frame, "Place 6x8 checkerboard in rectangle", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            start_detect = time.time()
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                detections = self.detector.detect(
                    gray, estimate_tag_pose=True, camera_params=(
                        self.camera_params['fx'], self.camera_params['fy'],
                        self.camera_params['cx'], self.camera_params['cy']
                    ), tag_size=self.tag_size
                )
            except RuntimeError as e:
                self.logger.error(f"Detection error: {e}")
                detections = []
            self.logger.debug(f"Detection time: {(time.time() - start_detect)*1000:.1f}ms")

            current_detections = {}
            if detections:
                self.logger.debug(f"Detected tags: {[det.tag_id for det in detections]}")
                for det in detections:
                    current_detections[det.tag_id] = det
                    self.prev_detections[det.tag_id] = det
                self.update_map_and_publish(detections, self.frame_count)
            else:
                self.logger.debug("No tags detected in current frame")
                self.update_map_and_publish([], self.frame_count)

            if not self.headless and self.frame_count % self.display_frame_interval == 0:
                for tag_id, det in list(self.prev_detections.items()):
                    if tag_id in current_detections or (time.time() - getattr(det, 'last_seen', 0) < 0.5):
                        corners = det.corners.astype(int)
                        for i in range(4):
                            cv2.line(display_frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                        center = tuple(map(int, det.center))
                        cv2.circle(display_frame, (center[0], center[1]), 5, (0, 0, 255), -1)
                        prob = 1.0 - det.decision_margin / 100.0
                        depth = det.pose_t.flatten()[2]
                        cv2.putText(display_frame, f"ID: {tag_id}", (center[0] + 10, center[1]),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Prob: {prob*100:.0f}%", (center[0] + 10, center[1] + 15),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        cv2.putText(display_frame, f"Depth: {depth:.3f}m", (center[0] + 10, center[1] + 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                        pose_t = det.pose_t.flatten()
                        self.logger.debug(f"Tag ID: {tag_id}, X: {pose_t[0]:.3f}m, Z: {pose_t[2]:.3f}m, "
                                        f"Depth: {depth:.3f}m")
                    setattr(det, 'last_seen', time.time())

        if not self.headless and self.frame_count % self.display_frame_interval == 0:
            cv2.putText(display_frame, f"FPS: {self.fps:.1f}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            if not self.calibration_mode:
                control_text = [f"{k}: {'Auto' if v == 0 else v}" for k, v in self.controls.items()]
                binding_text = [f"{k}: {v[0]} {'+' if v[1] > 0 else '-'}" for k, v in self.key_bindings.items()]
                for i, text in enumerate(control_text + binding_text):
                    cv2.putText(display_frame, text, (10, 70 + i * 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("AprilTag Detection", display_frame)
            if not self.calibration_mode:
                self.draw_map()

        self.logger.debug(f"Total frame time: {(time.time() - start_time)*1000:.1f}ms")

        if not self.headless:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                self.stop()
            elif not self.calibration_mode and chr(key) in self.key_bindings:
                self.manual_mode = True
                control, delta = self.key_bindings[chr(key)]
                if control in ['exposure', 'iso']:
                    initial = 1000 if control == 'exposure' else 400
                    self.controls[control] = max(100, self.controls[control] + delta if self.controls[control] > 0 else initial)
                else:
                    self.controls[control] = max(-10, min(10, self.controls[control] + delta))
                self.logger.info(f"Applying control: {control} = {self.controls[control]}")
                self.update_camera_controls()

    def update_camera_controls(self):
        self.camera_manager.update_controls(self.controls, self.manual_mode)