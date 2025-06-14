import time
from networktables import NetworkTables
import logging

logging.basicConfig(level=logging.DEBUG)
NetworkTables.initialize(server="10.92.2.2")
table = NetworkTables.getTable("SmartDashboard")

for i in range(10):
    table.putNumber("test_x", i)
    print(f"Published test_x: {i}, Connected: {NetworkTables.isConnected()}")
    time.sleep(1)

NetworkTables.shutdown()