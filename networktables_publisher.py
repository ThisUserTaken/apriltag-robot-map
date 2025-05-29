import logging
import time
import numpy as np
from networktables import NetworkTables

class NetworkTablesPublisher:
    def __init__(self, server, table_name):
        logging.basicConfig(level=logging.DEBUG)
        self.logger = logging.getLogger("NetworkTablesPublisher")
        self.server = server
        self.table = None
        self.table_name = table_name
        self._initialize_networktables()

    def _initialize_networktables(self, max_retries=5, retry_delay=2):
        """Initialize NetworkTables with retries."""
        for attempt in range(max_retries):
            NetworkTables.initialize(server=self.server)
            time.sleep(1)
            if NetworkTables.isConnected():
                self.table = NetworkTables.getTable(self.table_name)
                self.logger.info(f"Initialized NetworkTables with server {self.server}, table {self.table_name}")
                return
            self.logger.warning(f"Connection attempt {attempt + 1}/{max_retries} failed")
            NetworkTables.shutdown()
            time.sleep(retry_delay)
        self.logger.error(f"Failed to connect to NetworkTables after {max_retries} attempts")
        self.table = NetworkTables.getTable(self.table_name)

    def publish_tag_data(self, detections, frame_count):
        """Publish detailed tag data to NetworkTables."""
        if not NetworkTables.isConnected():
            self.logger.warning(f"Frame {frame_count}: Not connected, skipping tag publish")
            return
        self.table.putNumberArray("TagIDs", [])
        for det in detections:
            tag_id = det.tag_id
            prob = 1.0 - det.decision_margin / 100.0
            pose_t = det.pose_t.flatten()
            depth = pose_t[2]
            corners = det.corners.flatten().tolist()
            R = det.pose_R
            pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
            yaw = np.arctan2(R[1, 0], R[0, 0])
            roll = np.arctan2(R[2, 1], R[2, 2])

            tag_prefix = f"Tag_{tag_id}"
            self.table.putNumber(f"{tag_prefix}/Probability", prob)
            self.table.putNumberArray(f"{tag_prefix}/Pose", pose_t.tolist())
            self.table.putNumber(f"{tag_prefix}/Depth", depth)
            self.table.putNumberArray(f"{tag_prefix}/Corners", corners)
            self.table.putNumber(f"{tag_prefix}/Roll", np.degrees(roll))
            self.table.putNumber(f"{tag_prefix}/Pitch", np.degrees(pitch))
            self.table.putNumber(f"{tag_prefix}/Yaw", np.degrees(yaw))
            self.logger.debug(f"Frame {frame_count}: Published Tag {tag_id}: Prob={prob:.2f}, Depth={depth:.3f}m")

        tag_ids = [det.tag_id for det in detections]
        self.table.putNumberArray("TagIDs", tag_ids)
        self.logger.debug(f"Frame {frame_count}: Published TagIDs: {tag_ids}")

    def publish_robot_location(self, position, roll, pitch, yaw, frame_count):
        """Publish robot position (2D: x, z) to NetworkTables."""
        if not NetworkTables.isConnected():
            self.logger.warning(f"Frame {frame_count}: Not connected, skipping robot publish")
            return False
        prefix = "RobotPose"
        try:
            if position is not None:
                robot_x, robot_z = position
                self.table.putNumber(f"{prefix}/x", float(robot_x))  # Ensure float
                self.table.putNumber(f"{prefix}/z", float(robot_z))
                self.logger.debug(f"Frame {frame_count}: Published RobotPose: x={robot_x:.3f}, z={robot_z:.3f}")
            else:
                self.table.putNumber(f"{prefix}/x", float('nan'))
                self.table.putNumber(f"{prefix}/z", float('nan'))
                self.logger.debug(f"Frame {frame_count}: Published RobotPose: No valid position")
            return True
        except Exception as e:
            self.logger.error(f"Frame {frame_count}: Failed to publish RobotPose: {e}")
            return False

    def shutdown(self):
        self.logger.info("Shutting down NetworkTables")
        NetworkTables.shutdown()