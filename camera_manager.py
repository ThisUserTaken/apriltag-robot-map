import numpy as np  # Added import
from depthai import Device, Pipeline, node, CameraBoardSocket, ColorCameraProperties, CameraControl
from collections import deque

class CameraManager:
    def __init__(self, config):
        self.pipeline = Pipeline()
        self.device = None
        self.cam_node = None
        self.control_queue = None
        self.frame_queue = deque(maxlen=10)
        self.fps_target = config["pipeline"]["fps_target"]
        self.setup_pipeline()

    def setup_pipeline(self):
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

    def start(self):
        self.device = Device(self.pipeline)
        self.control_queue = self.device.getInputQueue("control")
        self.video_queue = self.device.getOutputQueue("video", maxSize=4, blocking=False)

    def stop(self):
        if self.device:
            self.device.close()

    def get_frame(self):
        in_frame = self.video_queue.tryGet()
        if in_frame is not None:
            frame = in_frame.getCvFrame()
            if frame.shape[2] != 3 or frame.dtype != np.uint8:
                frame = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_NV12)
            return frame
        return None

    def update_controls(self, controls, manual_mode):
        if self.device and manual_mode and self.control_queue:
            control = CameraControl()
            if controls['exposure'] > 0 and controls['iso'] > 0:
                control.setManualExposure(controls['exposure'], controls['iso'])
            control.setBrightness(controls['brightness'])
            control.setContrast(controls['contrast'])
            control.setSaturation(controls['saturation'])
            control.setSharpness(controls['sharpness'])
            self.control_queue.send(control)