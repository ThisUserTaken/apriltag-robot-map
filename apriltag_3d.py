import sys
import os
import time

# Add current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from apriltag_detector import AprilTagDetector
except ImportError as e:
    print(f"Failed to import AprilTagDetector: {e}")
    print(f"Current directory: {os.getcwd()}")
    print(f"Python path: {sys.path}")
    raise

def main():
    detector = None
    try:
        detector = AprilTagDetector(config_file="config.json")
        detector.start()
        while detector.running:
            time.sleep(0.01)
    except Exception as e:
        print(f"Error in main: {e}")
        raise
    finally:
        if detector is not None:
            detector.stop()

if __name__ == "__main__":
    main()