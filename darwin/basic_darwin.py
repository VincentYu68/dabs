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

    def connect(self):
        self.port_handler = PortHandler("/dev/ttyUSB0")
        if not self.port_handler.openPort():
            raise RuntimeError("Couldn't open port")
        if not self.port_handler.setBaudRate(self.BAUD):
            raise RuntimeError("Couldn't change baud rate")

        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        self.motor_reader = BulkMultiReader(self.port_handler, self.packet_handler, self.dxl_ids, self.read_attrs)
        self.torque_enable_writer = MultiWriter(self.port_handler, self.packet_handler, self.dxl_ids,
                                                self.write_attrs_torque_enable)
        self.pid_writer = MultiWriter(self.port_handler, self.packet_handler, self.dxl_ids, self.write_attrs_PID)
        self.motor_goal_writer = MultiWriter(self.port_handler, self.packet_handler, self.dxl_ids,
                                             self.write_attrs_goal)

    def disconnect(self):
        self.port_handler.closePort()

    def read_motor_positions(self):
        return self.motor_reader.read()

    def write_torque_enable(self, enable):
        data = 1 if enable else 0
        self.torque_enable_writer.write(np.array([data] * 20, dtype=np.int))

    def write_pid(self, p_gain, i_gain, d_gain):
        self.pid_writer.write(np.array([p_gain, i_gain, d_gain] * 20, dtype=np.int))

    def write_motor_goal(self, goals):
        assert len(goals) == 20
        self.motor_goal_writer.write(goals.astype(int))




















