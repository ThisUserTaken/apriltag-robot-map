import logging
import time
import json
from magnetometer import Magnetometer

def load_config(config_file="config.json"):
    """Load configuration from JSON file."""
    try:
        with open(config_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def main():
    # Configure logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("MagnetometerTest")

    # Load configuration
    config = load_config()
    if not config:
        logger.error("Failed to load configuration, exiting")
        return

    # Initialize magnetometer
    magnetometer = None
    try:
        magnetometer = Magnetometer(config)
        logger.info("Magnetometer initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize magnetometer: {e}")
        return

    # Test loop
    try:
        logger.info("Starting magnetometer direction test. Press Ctrl+C to stop.")
        while True:
            yaw = magnetometer.get_yaw()
            logger.info(f"Magnetometer Direction: {yaw:.1f}°")
            print(f"Magnetometer Direction: {yaw:.1f}°")  # Also print to console
            time.sleep(0.02)  # 50 Hz, matching main system
    except KeyboardInterrupt:
        logger.info("Test terminated by user")
    finally:
        if magnetometer:
            magnetometer.shutdown()
            logger.info("Magnetometer shutdown complete")

if __name__ == "__main__":
    main()