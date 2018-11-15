import numpy as np
from darwin.darwin_utils import *
from dynamixel_sdk import *
import itertools
from dabs import *

PROTOCOL_VERSION = 1

if PROTOCOL_VERSION == 1:
    import motors.p1mx28 as mx28
else:
    import motors.p2mx28 as mx28

class BasicDarwin:
    def __init__(self):
        self.BAUD = 1000000
        self.dxl_ids = np.arange(1, 21).tolist()

        self.read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION)]

        self.write_attrs_torque_enable = [(mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE)]

        self.write_attrs_PID = [(mx28.ADDR_P_GAIN, mx28.LEN_P_GAIN),
                                (mx28.ADDR_I_GAIN, mx28.LEN_I_GAIN),
                                (mx28.ADDR_D_GAIN, mx28.LEN_D_GAIN)]

        self.write_attrs_goal = [(mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

        self.motor_reader = BulkMultiReader(port_handler, packet_handler, self.dxl_ids, self.read_attrs)
        self.torque_enable_writer = MultiWriter(port_handler, packet_handler, self.dxl_ids, self.write_attrs_torque_enable)
        self.pid_writer = MultiWriter(port_handler, packet_handler, self.dxl_ids, self.write_attrs_PID)
        self.motor_goal_writer = MultiWriter(port_handler, packet_handler, self.dxl_ids, self.write_attrs_goal)

    def connect(self):
        self.port_handler = PortHandler("/dev/ttyUSB0")
        if not port_handler.openPort():
            raise RuntimeError("Couldn't open port")
        if not port_handler.setBaudRate(BAUD):
            raise RuntimeError("Couldn't change baud rate")

        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

    def disconnect(self):
        self.port_handler.closePort()

    def read_motor_positions(self):
        return self.motor_reader.read()

    def write_torque_enable(self, enable):
        data = 1 if enable else 0
        self.torque_enable_writer.write([data] * 20)

    def write_pid(self, p_gain, i_gain, d_gain):
        self.pid_writer.write([p_gain, i_gain, d_gain] * 20)

    def write_motor_goal(self, goals):
        assert len(goals) == 20
        self.motor_goal_writer.write(goals)





















