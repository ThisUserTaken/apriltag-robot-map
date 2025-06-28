import smbus2
import time
import math
import logging

class Magnetometer:
    def __init__(self, config):
        """Initialize WT901 magnetometer with configuration."""
        self.logger = logging.getLogger("Magnetometer")
        self.I2C_BUS = config.get("magnetometer", {}).get("i2c_bus", 7)
        self.WT901_ADDR = config.get("magnetometer", {}).get("address", 0x50)
        self.REG_HX = 0x3a  # E-axis magnetic field
        self.REG_HY = 0x3b  # N-axis magnetic field
        self.REG_HZ = 0x3c  # Up-axis magnetic field
        self.REG_HXOFFSET = 0x0b
        self.REG_HYOFFSET = 0x0c
        self.REG_HZOFFSET = 0x0d
        self.REG_CALSW = 0x01
        self.REG_SAVE = 0x00
        self.MAG_SCALE = 1.0
        self.bus = None
        self.mag_offsets = {'x': 0, 'y': 0, 'z': 0}
        self.magnetometer_yaw = 0.0
        self.calibrate_on_start = config.get("magnetometer", {}).get("calibrate_on_start", False)

        try:
            self.bus = smbus2.SMBus(self.I2C_BUS)
            self.configure_wt901()
            if self.calibrate_on_start:
                self.calibrate_magnetometer()
            else:
                self.mag_offsets = self.read_mag_offsets()
        except Exception as e:
            self.logger.error(f"Failed to initialize WT901: {e}")
            self.bus = None

    def configure_wt901(self):
        """Configure WT901 for 50 Hz data output."""
        try:
            self.bus.write_i2c_block_data(self.WT901_ADDR, 0x03, [0x08, 0x00])  # 50 Hz
            time.sleep(0.01)
            self.bus.write_i2c_block_data(self.WT901_ADDR, 0x02, [0x1F, 0x00])  # Enable mag output
            time.sleep(0.01)
            self.bus.write_i2c_block_data(self.WT901_ADDR, self.REG_SAVE, [0x00, 0x00])  # Save config
            time.sleep(0.1)
            self.logger.info("WT901 configured successfully")
        except Exception as e:
            self.logger.error(f"WT901 configuration error: {e}")

    def read_sensor_data(self, reg):
        """Read 16-bit sensor data from WT901."""
        try:
            data = self.bus.read_i2c_block_data(self.WT901_ADDR, reg, 2)
            value = (data[1] << 8) | data[0]
            if value > 32767:
                value -= 65536
            return value
        except Exception as e:
            self.logger.error(f"Read error at reg {reg}: {e}")
            return 0

    def read_mag_offsets(self):
        """Read stored magnetometer offsets."""
        try:
            offsets = {
                'x': self.read_sensor_data(self.REG_HXOFFSET) * self.MAG_SCALE,
                'y': self.read_sensor_data(self.REG_HYOFFSET) * self.MAG_SCALE,
                'z': self.read_sensor_data(self.REG_HZOFFSET) * self.MAG_SCALE
            }
            self.logger.info(f"Loaded magnetometer offsets: E={offsets['x']:.2f}, N={offsets['y']:.2f}, Up={offsets['z']:.2f}")
            return offsets
        except Exception as e:
            self.logger.error(f"Error reading magnetometer offsets: {e}")
            return {'x': 0, 'y': 0, 'z': 0}

    def calibrate_magnetometer(self):
        """Calibrate WT901 magnetometer by rotating around each axis."""
        self.logger.info("Magnetometer calibration started. Keep away from magnetic interference.")
        self.logger.info("Rotate the WT901 360° around each axis (E, N, Up) multiple times.")
        
        try:
            self.bus.write_i2c_block_data(self.WT901_ADDR, self.REG_CALSW, [0x02, 0x00])
            time.sleep(0.1)
            
            mag_data = {'x': [], 'y': [], 'z': []}
            axes = ['East (E)', 'North (N)', 'Up (Z)']
            
            for axis in axes:
                self.logger.info(f"Rotating around {axis}-axis for 10 seconds.")
                start_time = time.time()
                while time.time() - start_time < 10:
                    x = self.read_sensor_data(self.REG_HX)
                    y = self.read_sensor_data(self.REG_HY)
                    z = self.read_sensor_data(self.REG_HZ)
                    mag_data['x'].append(x * self.MAG_SCALE)
                    mag_data['y'].append(y * self.MAG_SCALE)
                    mag_data['z'].append(z * self.MAG_SCALE)
                    time.sleep(0.05)
                self.logger.info(f"Completed rotation around {axis}-axis.")
            
            offsets = {
                'x': (min(mag_data['x']) + max(mag_data['x'])) / 2,
                'y': (min(mag_data['y']) + max(mag_data['y'])) / 2,
                'z': (min(mag_data['z']) + max(mag_data['z'])) / 2
            }
            
            for axis, reg in zip(['x', 'y', 'z'], [self.REG_HXOFFSET, self.REG_HYOFFSET, self.REG_HZOFFSET]):
                offset = int(offsets[axis] / self.MAG_SCALE)
                self.bus.write_i2c_block_data(self.WT901_ADDR, reg, [offset & 0xFF, (offset >> 8) & 0xFF])
                time.sleep(0.01)
            
            self.bus.write_i2c_block_data(self.WT901_ADDR, self.REG_CALSW, [0x00, 0x00])
            time.sleep(0.1)
            self.bus.write_i2c_block_data(self.WT901_ADDR, self.REG_SAVE, [0x00, 0x00])
            time.sleep(0.1)
            
            self.logger.info(f"Calibration complete. Offsets: E={offsets['x']:.2f}, N={offsets['y']:.2f}, Up={offsets['z']:.2f}")
            self.mag_offsets = offsets
        except Exception as e:
            self.logger.error(f"Calibration error: {e}")

    def get_yaw(self):
        """Compute yaw from magnetometer data."""
        try:
            mag_x = (self.read_sensor_data(self.REG_HX) * self.MAG_SCALE) - self.mag_offsets['x']
            mag_y = (self.read_sensor_data(self.REG_HY) * self.MAG_SCALE) - self.mag_offsets['y']
            heading = math.atan2(mag_y, mag_x) * 180 / math.pi
            if heading < 0:
                heading += 360
            self.magnetometer_yaw = heading
            return heading
        except Exception as e:
            self.logger.error(f"Magnetometer yaw error: {e}")
            return self.magnetometer_yaw  # Return last known yaw

    def shutdown(self):
        """Close I2C bus."""
        if self.bus:
            self.bus.close()
            self.logger.info("I2C bus closed")
            self.bus = None

if __name__ == "__main__":
    # Example usage for testing
    logging.basicConfig(level=logging.DEBUG)
    config = {
        "magnetometer": {
            "i2c_bus": 7,
            "address": 0x50,
            "calibrate_on_start": True
        }
    }
    mag = Magnetometer(config)
    try:
        while True:
            yaw = mag.get_yaw()
            print(f"Magnetometer Yaw: {yaw:.1f}°")
            time.sleep(0.02)
    except KeyboardInterrupt:
        mag.shutdown()