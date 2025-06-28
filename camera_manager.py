import depthai as dai
import logging
import time
import numpy as np

class CameraManager:
    def __init__(self, config):
        """Initialize CameraManager with configuration."""
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("CameraManager")
        self.config = config
        self.pipeline = None
        self.device = None
        self.rgb_queue = None
        self.fps_target = self.config.get("pipeline", {}).get("fps_target", 30)
        self.controls = self.config.get("controls", {})
        self.running = False
        self.cam_rgb = None  # Store ColorCamera node for control updates

    def start(self):
        """Start the camera pipeline and initialize the device."""
        try:
            # Create pipeline
            self.pipeline = dai.Pipeline()

            # Define RGB camera
            self.cam_rgb = self.pipeline.create(dai.node.ColorCamera)
            self.cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
            self.cam_rgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_800_P)  # Use 1280x800
            self.cam_rgb.setFps(self.fps_target)
            self.cam_rgb.setVideoSize(1280, 800)

            # Apply initial camera controls
            self.apply_controls()

            # Create output
            xout_rgb = self.pipeline.create(dai.node.XLinkOut)
            xout_rgb.setStreamName("rgb")
            self.cam_rgb.video.link(xout_rgb.input)

            # Start the device
            self.device = dai.Device(self.pipeline)
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.running = True
            self.logger.info("Camera initialized successfully")
            return True
        except Exception as e:
            self.logger.error(f"Failed to start camera: {e}")
            return False

    def apply_controls(self):
        """Apply camera controls (exposure, ISO, etc.)."""
        if not self.cam_rgb:
            self.logger.warning("Cannot apply controls: ColorCamera node not initialized")
            return
        try:
            if self.controls.get("exposure", 0) > 0:
                self.cam_rgb.initialControl.setManualExposure(self.controls["exposure"], self.controls.get("iso", 400))
            else:
                self.cam_rgb.initialControl.setAutoExposureEnable()
            # Other controls (brightness, contrast, etc.) may require post-processing
            self.logger.debug(f"Applied controls: {self.controls}")
        except Exception as e:
            self.logger.error(f"Error applying controls: {e}")

    def update_controls(self, controls, manual_mode):
        """Update camera controls dynamically."""
        self.controls.update(controls)
        if manual_mode:
            self.apply_controls()
        else:
            # Reset to auto if manual mode is disabled
            self.controls = {k: 0 for k in self.controls}
            self.apply_controls()
        self.logger.info(f"Updated controls: {self.controls}, manual_mode={manual_mode}")

    def get_frame(self):
        """Retrieve the latest RGB frame."""
        if not self.running or not self.rgb_queue:
            self.logger.warning("Camera not running or queue not initialized")
            return None
        try:
            frame_data = self.rgb_queue.get()
            if frame_data:
                frame = frame_data.getCvFrame()
                return frame
            return None
        except Exception as e:
            self.logger.error(f"Error getting frame: {e}")
            return None

    def stop(self):
        """Stop the camera and clean up resources."""
        self.running = False
        if self.device:
            try:
                self.device.close()
                self.logger.info("Camera device closed")
            except Exception as e:
                self.logger.error(f"Error closing camera device: {e}")
        self.device = None
        self.rgb_queue = None
        self.pipeline = None
        self.cam_rgb = None