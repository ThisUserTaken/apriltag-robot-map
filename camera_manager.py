import numpy as np
import cv2
from depthai import Device, Pipeline, node, CameraBoardSocket, ColorCameraProperties, CameraControl
from collections import deque
import time
import logging

class CameraManager:
    def __init__(self, config):
        self.logger = logging.getLogger("CameraManager")
        self.pipeline = Pipeline()
        self.device = None
        self.cam_node = None
        self.control_queue = None
        self.video_queue = None
        self.frame_queue = deque(maxlen=1)
        self.fps_target = config["pipeline"]["fps_target"]
        self.camera_params = config["camera_params"]
        self.dist_coeffs = np.array([-0.1, 0.05, 0.001, 0.001])
        self.camera_matrix = np.array([
            [self.camera_params['fx'], 0, self.camera_params['cx']],
            [0, self.camera_params['fy'], self.camera_params['cy']],
            [0, 0, 1]
        ], dtype=np.float64)
        self.setup_pipeline()
        self.connected = False

    def setup_pipeline(self):
        self.cam_node = self.pipeline.create(node.ColorCamera)
        xout = self.pipeline.create(node.XLinkOut)
        control_in = self.pipeline.create(node.XLinkIn)
        xout.setStreamName("video")
        control_in.setStreamName("control")

        self.cam_node.setBoardSocket(CameraBoardSocket.CAM_A)
        self.cam_node.setResolution(ColorCameraProperties.SensorResolution.THE_720_P)
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

    def start(self, max_retries=3, retry_delay=2):
        """Initialize camera with retries."""
        for attempt in range(max_retries):
            try:
                self.device = Device(self.pipeline)
                self.control_queue = self.device.getInputQueue("control")
                self.video_queue = self.device.getOutputQueue("video", maxSize=4, blocking=False)
                self.connected = True
                self.logger.info("Camera initialized successfully")
                return True
            except Exception as e:
                self.logger.error(f"Camera initialization attempt {attempt + 1}/{max_retries} failed: {e}")
                self.connected = False
                if self.device:
                    self.device.close()
                time.sleep(retry_delay)
        self.logger.error("Failed to initialize camera after max retries")
        return False

    def stop(self):
        if self.device:
            self.device.close()
            self.device = None
            self.connected = False
            self.logger.info("Camera stopped")

    def get_frame(self):
        """Get a frame, handling disconnections."""
        if not self.connected:
            self.logger.warning("Camera not connected, attempting to reconnect")
            self.start()
            return None

        try:
            in_frame = self.video_queue.tryGet()
            if in_frame is not None:
                frame = in_frame.getCvFrame()
                if frame.shape[2] != 3 or frame.dtype != np.uint8:
                    frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
                frame = cv2.undistort(frame, self.camera_matrix, self.dist_coeffs)
                return frame
            return None
        except Exception as e:
            self.logger.error(f"Failed to get frame: {e}")
            self.connected = False
            self.stop()
            return None

    def update_controls(self, controls, manual_mode):
        if self.device and manual_mode and self.control_queue:
            try:
                control = CameraControl()
                if controls['exposure'] > 0 and controls['iso'] > 0:
                    control.setManualExposure(controls['exposure'], controls['iso'])
                control.setBrightness(controls['brightness'])
                control.setContrast(controls['contrast'])
                control.setSaturation(controls['saturation'])
                control.setSharpness(controls['sharpness'])
                self.control_queue.send(control)
            except Exception as e:
                self.logger.error(f"Failed to update camera controls: {e}")