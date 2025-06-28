import time
import math
import logging
from networktables import NetworkTables
import numpy as np

class PIDController:
    def __init__(self, kp, ki, kd, tolerance):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.tolerance = tolerance
        self.prev_error = 0.0
        self.integral = 0.0
        self.last_time = time.time()

    def calculate(self, error):
        current_time = time.time()
        dt = current_time - self.last_time
        if dt == 0:
            dt = 1e-6  # Avoid division by zero

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        derivative = (error - self.prev_error) / dt
        d_term = self.kd * derivative

        # Update state
        self.prev_error = error
        self.last_time = current_time

        return p_term + i_term + d_term

def move_to_target(target_x, target_z, nt_table="SmartDashboard", max_speed=0.5, max_rotation=0.5, timeout=10.0):
    """
    Move robot to target (x, z) using PID controllers for heading and distance.
    Reads current pose and heading from NetworkTables and sends speed/rotation commands.
    
    Args:
        target_x (float): Target x-coordinate (meters).
        target_z (float): Target z-coordinate (meters).
        nt_table (str): NetworkTables table name (default: "SmartDashboard").
        max_speed (float): Maximum forward speed (0 to 1).
        max_rotation (float): Maximum rotation speed (0 to 1).
        timeout (float): Maximum time to run (seconds).
    
    Returns:
        bool: True if target reached within tolerance, False if timed out or failed.
    """
    logging.basicConfig(level=logging.DEBUG)
    logger = logging.getLogger("PIDController")
    
    # Initialize NetworkTables
    if not NetworkTables.isConnected():
        NetworkTables.initialize(server="10.92.2.2")
        time.sleep(1)
        if not NetworkTables.isConnected():
            logger.error("Failed to connect to NetworkTables")
            return False
    
    table = NetworkTables.getTable(nt_table)
    
    # PID controllers
    angle_pid = PIDController(kp=0.8, ki=0.0, kd=0.1, tolerance=0.05)  # Radians
    distance_pid = PIDController(kp=1.0, ki=0.0, kd=0.2, tolerance=0.05)  # Meters
    
    start_time = time.time()
    
    while time.time() - start_time < timeout:
        # Read current robot pose and heading
        robot_x = table.getNumber("RobotPose/x", float('nan'))
        robot_z = table.getNumber("RobotPose/z", float('nan'))
        robot_heading = math.radians(table.getNumber("RobotPose/yaw", float('nan')))
        
        if math.isnan(robot_x) or math.isnan(robot_z) or math.isnan(robot_heading):
            logger.warning("Invalid robot pose or heading, retrying")
            time.sleep(0.1)
            continue
        
        # Calculate differences
        dx = target_x - robot_x
        dz = target_z - robot_z
        
        # Transform to robot's frame
        x_offset = dx * math.cos(robot_heading) + dz * math.sin(robot_heading)
        z_offset = -dx * math.sin(robot_heading) + dz * math.cos(robot_heading)
        
        # Calculate errors
        distance_error = math.sqrt(x_offset**2 + z_offset**2)
        heading_error = math.atan2(x_offset, z_offset)
        
        # Normalize heading error to [-pi, pi]
        heading_error = (heading_error + math.pi) % (2 * math.pi) - math.pi
        
        # Check if within tolerance
        if abs(distance_error) < distance_pid.tolerance and abs(heading_error) < angle_pid.tolerance:
            logger.info(f"Target reached: x={target_x:.3f}, z={target_z:.3f}")
            table.putNumber("Drive/Speed", 0.0)
            table.putNumber("Drive/Rotation", 0.0)
            return True
        
        # Calculate PID outputs
        rotation = angle_pid.calculate(heading_error)
        speed = distance_pid.calculate(distance_error)
        
        # Clamp outputs
        rotation = max(min(rotation, max_rotation), -max_rotation)
        speed = max(min(speed, max_speed), -max_speed)
        
        # Send commands to NetworkTables
        table.putNumber("Drive/Speed", speed)
        table.putNumber("Drive/Rotation", rotation)
        
        logger.debug(f"Pose: x={robot_x:.3f}, z={robot_z:.3f}, heading={math.degrees(robot_heading):.1f}° | "
                     f"Target: x={target_x:.3f}, z={target_z:.3f} | "
                     f"Errors: dist={distance_error:.3f}m, heading={math.degrees(heading_error):.1f}° | "
                     f"Commands: speed={speed:.3f}, rotation={rotation:.3f}")
        
        time.sleep(0.02)  # 50 Hz control loop
    
    logger.error("Timeout: Failed to reach target within time limit")
    table.putNumber("Drive/Speed", 0.0)
    table.putNumber("Drive/Rotation", 0.0)
    return False

if __name__ == "__main__":
    # Example usage
    move_to_target(target_x=1.0, target_z=2.0)