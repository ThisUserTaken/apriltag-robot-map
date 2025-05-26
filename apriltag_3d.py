import cv2
import numpy as np
from depthai import Device, Pipeline, node, CameraBoardSocket, ColorCameraProperties, CameraControl
from pupil_apriltags import Detector
import time
import threading
from collections import deque
import json
import os

class AprilTagDetector:
    def __init__(self, config_file="config.json"):
        """Initialize AprilTag detector with Oak-D Pro W RGB camera."""
        try:
            with open(config_file, 'r') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            print(f"Error: {config_file} not found")
            exit(1)
        except json.JSONDecodeError:
            print(f"Error: Invalid JSON in {config_file}")
            exit(1)

        self.calibration_mode = self.config["calibration"]
        print(f"Calibration mode: {self.calibration_mode}")
        self.calibration_positions = [tuple(pos) for pos in self.config["calibration_positions"]]
        self.current_position_idx = 0
        self.checkerboard_size = (6, 8)
        self.calibration_data = {'obj_points': [], 'img_points': []}
        self.config_file = config_file

        self.tag_size = self.config["tag"]["size"]
        self.tag_family = self.config["tag"]["family"]
        self.detector = Detector(
            families=self.tag_family,
            nthreads=self.config["detector"]["nthreads"],
            quad_decimate=self.config["detector"]["quad_decimate"],
            quad_sigma=self.config["detector"]["quad_sigma"],
            refine_edges=self.config["detector"]["refine_edges"],
            decode_sharpening=self.config["detector"]["decode_sharpening"]
        )
        self.pipeline = Pipeline()
        self.device = None
        self.cam_node = None
        self.frame_queue = deque(maxlen=10)
        self.running = False
        self.frame_count = 0
        self.fps_target = self.config["pipeline"]["fps_target"]
        self.display_interval_s = 0.5
        self.manual_mode = False
        self.last_frame_time = time.time()
        self.fps = 0.0
        self.prev_detections = {}

        # Map settings
        self.map_scale = self.config["map"]["scale"]
        self.map_width = self.config["map"]["width"]
        self.map_height = self.config["map"]["height"]
        self.tag_positions = {}  # Store tag ID: (x, z) positions

        self.camera_params = self.config["camera_params"]
        self.camera_matrix = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ], dtype=np.float64)

        self.controls = self.config["controls"]
        self.key_bindings = {k: tuple(v) for k, v in self.config["key_bindings"].items()}

    def setup_pipeline(self):
        """Configure DepthAI pipeline for Oak-D Pro W RGB camera."""
        self.cam_node = self.pipeline.create(node.ColorCamera)
        xout = self.pipeline.create(node.XLinkOut)
        control_in = self.pipeline.create(node.XLinkIn)
        xout.setStreamName("video")
        control_in.setStreamName("control")

        self.cam_node.setBoardSocket(CameraBoardSocket.CAM_A)
        self.cam_node.setResolution(ColorCameraProperties.SensorResolution.THE_800_P)
        self.cam_node.setFps(self.fps_target)
        self.cam_node.setInterleaved(False)
        self.cam_node.setColorOrder(ColorCameraProperties.ColorOrder.BGR)
        self.cam_node.isp.link(xout.input)
        control_in.out.link(self.cam_node.inputControl)

        control = self.cam_node.initialControl
        control.setAutoExposureEnable()
        control.setAutoWhiteBalanceMode(CameraControl.AutoWhiteBalanceMode.AUTO)
        control.setAntiBandingMode(CameraControl.AntiBandingMode.AUTO)
        control.setManualExposure(5000, 400)

    def calibrate_camera(self, frame):
        """Run calibration with 6x8 checkerboard."""
        start_time = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        x1, y1, x2, y2 = self.calibration_positions[self.current_position_idx]

        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        print(f"Checkerboard detection: {ret}, Corners: {len(corners) if ret else 0}, Time: {(time.time() - start_time)*1000:.1f}ms")
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
                print(f"Captured calibration image at position {self.current_position_idx + 1}")
                self.current_position_idx += 1

                if self.current_position_idx >= len(self.calibration_positions):
                    ret, mtx, dist, _, _ = cv2.calibrateCamera(
                        self.calibration_data['obj_points'], self.calibration_data['img_points'],
                        gray.shape[::-1], None, None)
                    if ret:
                        self.camera_params = {
                            'fx': mtx[0, 0], 'fy': mtx[1, 1], 'cx': mtx[0, 2], 'cy': mtx[1, 2]
                        }
                        self.camera_matrix = mtx
                        self.config["camera_params"] = self.camera_params
                        with open(self.config_file, 'w') as f:
                            json.dump(self.config, f, indent=2)
                        print("Calibration complete, updated config.json")
                        self.calibration_mode = False
                    self.current_position_idx = 0
                    self.calibration_data = {'obj_points': [], 'img_points': []}
            else:
                print("Checkerboard not fully within rectangle")

    def detect_blur(self, frame):
        """Estimate blur using Laplacian variance."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()
        return variance < 100

    def update_map(self, detections):
        """Update 2D map with AprilTag positions."""
        for det in detections:
            tag_id = det.tag_id
            pose_t = det.pose_t.flatten()
            x, z = pose_t[0], pose_t[2]  # Use x, z for 2D map
            self.tag_positions[tag_id] = (x, z)

    def draw_map(self):
        """Draw 2D map of AprilTags."""
        map_frame = np.ones((self.map_height, self.map_width, 3), dtype=np.uint8) * 255  # White background
        center_x, center_z = self.map_width // 2, self.map_height // 2

        for tag_id, (x, z) in self.tag_positions.items():
            # Convert to pixel coordinates
            px = int(center_x + x * self.map_scale)
            pz = int(center_z - z * self.map_scale)  # Invert z for top-down view
            if 0 <= px < self.map_width and 0 <= pz < self.map_height:
                cv2.circle(map_frame, (px, pz), 5, (0, 0, 255), -1)  # Red dot for tag
                cv2.putText(map_frame, f"ID: {tag_id}", (px + 10, pz),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw axes
        cv2.line(map_frame, (center_x - 50, center_z), (center_x + 50, center_z), (0, 0, 0), 1)  # X-axis
        cv2.line(map_frame, (center_x, center_z - 50), (center_x, center_z + 50), (0, 0, 0), 1)  # Z-axis
        cv2.putText(map_frame, "X", (center_x + 60, center_z), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(map_frame, "Z", (center_x, center_z - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        cv2.imshow("2D AprilTag Map", map_frame)

    def start(self):
        """Start camera and detection thread."""
        self.device = Device(self.pipeline)
        self.running = True
        self.control_queue = self.device.getInputQueue("control")
        self.thread = threading.Thread(target=self.process_frames)
        self.thread.start()

    def stop(self):
        """Stop camera and thread."""
        self.running = False
        self.thread.join()
        self.device.close()

    def process_frames(self):
        """Process frames at target FPS."""
        video = self.device.getOutputQueue("video", maxSize=4, blocking=False)
        frame_times = deque(maxlen=10)
        last_display_time = 0
        while self.running:
            start_time = time.time()
            in_frame = video.tryGet()
            if in_frame is not None:
                frame = in_frame.getCvFrame()
                if frame.shape[2] != 3 or frame.dtype != np.uint8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                if not self.detect_blur(frame):
                    self.frame_queue.append(frame)
                    self.frame_count += 1
                    current_time = time.time()
                    frame_times.append(current_time)
                    if len(frame_times) > 1:
                        self.fps = len(frame_times) / (frame_times[-1] - frame_times[0])
                    print(f"Frame acquisition time: {(current_time - start_time)*1000:.1f}ms, FPS: {self.fps:.1f}")
                else:
                    print("Skipped blurry frame")

            if self.frame_queue and (time.time() - last_display_time) >= self.display_interval_s:
                process_start = time.time()
                if self.calibration_mode:
                    self.calibrate_camera(self.frame_queue[-1])
                self.detect_tags(self.frame_queue[-1])
                last_display_time = time.time()
                print(f"Frame processing time: {(last_display_time - process_start)*1000:.1f}ms")

    def detect_tags(self, frame):
        """Detect AprilTags or display calibration rectangle."""
        start_time = time.time()
        display_frame = frame.copy()
        print(f"Rendering frame, calibration mode: {self.calibration_mode}, position: {self.current_position_idx}")
        
        if self.calibration_mode:
            x1, y1, x2, y2 = self.calibration_positions[self.current_position_idx]
            print(f"Drawing rectangle: ({x1}, {y1}, {x2}, {y2})")
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(display_frame, f"Position {self.current_position_idx + 1}/{len(self.calibration_positions)}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            cv2.putText(display_frame, "Place 6x8 checkerboard in rectangle", (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        else:
            start_detect = time.time()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            try:
                detections = self.detector.detect(
                    gray, estimate_tag_pose=True, camera_params=(
                        self.camera_params['fx'], self.camera_params['fy'],
                        self.camera_params['cx'], self.camera_params['cy']
                    ), tag_size=self.tag_size
                )
            except RuntimeError as e:
                print(f"Detection error: {e}")
                detections = []
            print(f"Detection time: {(time.time() - start_detect)*1000:.1f}ms")

            current_detections = {}
            if detections:
                for det in detections:
                    current_detections[det.tag_id] = det
                    self.prev_detections[det.tag_id] = det
                self.update_map(detections)  # Update 2D map
            else:
                print("No tags detected in current frame")

            for tag_id, det in list(self.prev_detections.items()):
                if tag_id in current_detections or (time.time() - getattr(det, 'last_seen', 0) < 0.5):
                    corners = det.corners.astype(int)
                    for i in range(4):
                        cv2.line(display_frame, tuple(corners[i]), tuple(corners[(i + 1) % 4]), (0, 255, 0), 2)
                    center = tuple(map(int, det.center))
                    cv2.circle(display_frame, center, 5, (0, 0, 255), -1)
                    prob = 1.0 - det.decision_margin / 100.0
                    depth = det.pose_t.flatten()[2]
                    cv2.putText(display_frame, f"ID: {det.tag_id}", (center[0] + 10, center[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Prob: {prob*100:.0f}%", (center[0] + 10, center[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Depth: {depth:.3f}m", (center[0] + 10, center[1] + 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    pose_t = det.pose_t.flatten()
                    R = det.pose_R
                    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
                    yaw = np.arctan2(R[1, 0], R[0, 0])
                    roll = np.arctan2(R[2, 1], R[2, 2])
                    print(f"Tag ID: {det.tag_id}, X: {pose_t[0]:.3f}m, Y: {pose_t[1]:.3f}m, Z: {pose_t[2]:.3f}m, "
                          f"Depth: {depth:.3f}m, Pitch: {np.degrees(pitch):.1f}°, Yaw: {np.degrees(yaw):.1f}°, "
                          f"Roll: {np.degrees(roll):.1f}°")
                setattr(det, 'last_seen', time.time())

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
            self.draw_map()  # Draw 2D map
        print(f"Total frame time: {(time.time() - start_time)*1000:.1f}ms")

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
            print(f"Applying control: {control} = {self.controls[control]}")
            self.update_camera_controls()

    def update_camera_controls(self):
        """Update RGB camera settings via control queue."""
        if self.device and self.manual_mode and self.control_queue:
            control = CameraControl()
            if self.controls['exposure'] > 0 and self.controls['iso'] > 0:
                control.setManualExposure(self.controls['exposure'], self.controls['iso'])
            control.setBrightness(self.controls['brightness'])
            control.setContrast(self.controls['contrast'])
            control.setSaturation(self.controls['saturation'])
            control.setSharpness(self.controls['sharpness'])
            self.control_queue.send(control)

    def add_camera(self, camera_params):
        """Placeholder for adding additional cameras."""
        pass

if __name__ == "__main__":
    detector = AprilTagDetector()
    detector.setup_pipeline()
    detector.start()
    try:
        while detector.running:
            time.sleep(0.01)
    except KeyboardInterrupt:
        detector.stop()
    cv2.destroyAllWindows()