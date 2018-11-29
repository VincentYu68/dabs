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

        self.voltage_attrs = [(mx28.ADDR_PRESENT_VOLTAGE, mx28.LEN_PRESENT_VOLTAGE)]

        self.torque_limit_attrs = [(mx28.ADDR_TORQUE_LIMIT, mx28.LEN_TORQUE_LIMIT)]

        self.delay_attrs = [(mx28.ADDR_RETURN_DELAY_TIME, mx28.LEN_RETURN_DELAY_TIME)]

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
        #self.motor_goal_writer = MultiWriter(self.port_handler, self.packet_handler, self.dxl_ids,
        #                                     self.write_attrs_goal)
        self.motor_goal_writer = SyncMultiWriter(self.port_handler, self.packet_handler, self.dxl_ids,
                                             self.write_attrs_goal)

        self.delay_reader = BulkMultiReader(self.port_handler, self.packet_handler, self.dxl_ids, self.delay_attrs)
        self.delay_writer = MultiWriter(self.port_handler, self.packet_handler, self.dxl_ids, self.delay_attrs)

        self.torque_limit_writer = SyncMultiWriter(self.port_handler, self.packet_handler, self.dxl_ids,
                                                 self.torque_limit_attrs)

        self.voltage_reader = BulkMultiReader(self.port_handler, self.packet_handler, self.dxl_ids, self.voltage_attrs)

        self.write_motor_delay([0] * 20)

    def disconnect(self):
        self.port_handler.closePort()

    def read_motor_positions(self):
        return self.motor_reader.read()

    def read_motor_voltages(self):
        return self.voltage_reader.read()

    def write_torque_enable(self, enable):
        data = 1 if enable else 0
        self.torque_enable_writer.write(np.array([data] * 20, dtype=np.int))

    def write_torque_limit(self, limit):
        self.torque_limit_writer.write(np.array(limit, dtype=np.int))

    def write_pid(self, p_gain, i_gain, d_gain):
        self.pid_writer.write(np.array([p_gain, i_gain, d_gain] * 20, dtype=np.int))

    def write_motor_goal(self, goals):
        assert len(goals) == 20
        self.motor_goal_writer.write([int(g) for g in goals])

    def read_motor_delay(self):
        return self.delay_reader.read()

    def write_motor_delay(self, delay):
        assert len(delay) == 20
        self.delay_writer.write([int(g) for g in delay])




















