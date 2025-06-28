import time
import math
import logging
import numpy as np
from networktables_publisher import NetworkTablesPublisher

class PIDController:
    def __init__(self, kP, kI, kD):
        self.kP = kP
        self.kI = kI
        self.kD = kD
        self.prev_error = 0.0
        self.integral = 0.0

    def calculate(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
        output = self.kP * error + self.kI * self.integral + self.kD * derivative
        self.prev_error = error
        return output

    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0

def drive_to_coordinate(target_x, target_z, timeout=10.0):
    """
    Move robot to target (x, z) using PID controllers.
    Reads pose from /SmartDashboard/RobotPose/x, /z, /yaw (degrees).
    Publishes commands to /SmartDashboard/DriveCommand/forward, /rotation.
    Returns True if successful, False if timeout or failure.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("DriveToCoordinate")

    # Initialize NetworkTablesPublisher
    publisher = NetworkTablesPublisher(server="10.92.2.2", table_name="SmartDashboard")
    table = publisher.table
    pose_table = table.getSubTable("RobotPose")
    command_table = table.getSubTable("DriveCommand")

    # PID controllers
    distance_pid = PIDController(kP=0.5, kI=0.0, kD=0.01)  # Tune these
    angle_pid = PIDController(kP=1.0, kI=0.0, kD=0.02)
    distance_pid.reset()
    angle_pid.reset()

    # Tolerances
    DISTANCE_TOLERANCE = 0.05  # meters
    ANGLE_TOLERANCE = math.radians(2)  # radians

    start_time = time.time()
    prev_time = start_time

    try:
        while time.time() - start_time < timeout:
            # Read robot pose
            robot_x = pose_table.getEntry("x").getDouble(0.0)
            robot_z = pose_table.getEntry("z").getDouble(0.0)
            robot_yaw = math.radians(pose_table.getEntry("yaw").getDouble(0.0))

            # Compute errors
            dx = target_x - robot_x
            dz = target_z - robot_z
            # Transform to robot frame
            x_offset = math.cos(robot_yaw) * dx + math.sin(robot_yaw) * dz
            z_offset = -math.sin(robot_yaw) * dx + math.cos(robot_yaw) * dz
            distance_error = math.sqrt(x_offset**2 + z_offset**2)
            heading_error = math.atan2(z_offset, x_offset)
            # Normalize heading error to [-pi, pi]
            heading_error = ((heading_error + math.pi) % (2 * math.pi)) - math.pi

            logger.debug(f"Robot pose: x={robot_x:.3f}, z={robot_z:.3f}, yaw={math.degrees(robot_yaw):.1f}°")
            logger.debug(f"Target: x={target_x:.3f}, z={target_z:.3f}")
            logger.debug(f"Distance error: {distance_error:.3f}m, Heading error: {math.degrees(heading_error):.1f}°")

            # Check if within tolerance
            if distance_error < DISTANCE_TOLERANCE and abs(heading_error) < ANGLE_TOLERANCE:
                logger.info("Reached target coordinate")
                command_table.getEntry("forward").setDouble(0.0)
                command_table.getEntry("rotation").setDouble(0.0)
                return True

            # Compute PID outputs
            dt = time.time() - prev_time
            forward_speed = distance_pid.calculate(distance_error, dt)
            rotation_speed = angle_pid.calculate(heading_error, dt)
            prev_time = time.time()

            # Limit speeds
            forward_speed = max(-0.5, min(0.5, forward_speed))
            rotation_speed = max(-0.3, min(0.3, rotation_speed))

            # Publish commands
            command_table.getEntry("forward").setDouble(forward_speed)
            command_table.getEntry("rotation").setDouble(rotation_speed)
            logger.debug(f"Command: forward={forward_speed:.3f}, rotation={rotation_speed:.3f}")

            time.sleep(0.01)  # Avoid overloading

        logger.error("Timeout reached, failed to reach target")
        command_table.getEntry("forward").setDouble(0.0)
        command_table.getEntry("rotation").setDouble(0.0)
        return False

    finally:
        publisher.shutdown()

if __name__ == "__main__":
    # Example usage
    drive_to_coordinate(2.0, 1.0)