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
    def __init__(self, use_bno055 = False):
        self.BAUD = 1000000
        self.dxl_ids = np.arange(1, 21).tolist()

        self.read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION)]

        self.read_velocity_attrs = [(mx28.ADDR_PRESENT_SPEED, mx28.LEN_PRESENT_SPEED)]

        self.read_posvel_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION), (mx28.ADDR_PRESENT_SPEED, mx28.LEN_PRESENT_SPEED)]

        self.voltage_attrs = [(mx28.ADDR_PRESENT_VOLTAGE, mx28.LEN_PRESENT_VOLTAGE)]

        self.torque_limit_attrs = [(mx28.ADDR_TORQUE_LIMIT, mx28.LEN_TORQUE_LIMIT)]

        self.delay_attrs = [(mx28.ADDR_RETURN_DELAY_TIME, mx28.LEN_RETURN_DELAY_TIME)]

        self.write_attrs_torque_enable = [(mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE)]

        self.write_attrs_PID = [(mx28.ADDR_P_GAIN, mx28.LEN_P_GAIN),
                                (mx28.ADDR_I_GAIN, mx28.LEN_I_GAIN),
                                (mx28.ADDR_D_GAIN, mx28.LEN_D_GAIN)]

        self.write_attrs_goal = [(mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

        self.use_bno055 = use_bno055




    def connect(self):
        self.port_handler = PortHandler("/dev/ttyUSB0")
        if not self.port_handler.openPort():
            raise RuntimeError("Couldn't open port")
        if not self.port_handler.setBaudRate(self.BAUD):
            raise RuntimeError("Couldn't change baud rate")

        self.packet_handler = PacketHandler(PROTOCOL_VERSION)

        self.motor_reader = BulkMultiReader(self.port_handler, self.packet_handler, self.dxl_ids, self.read_attrs)
        self.motor_velocity_reader = BulkMultiReader(self.port_handler, self.packet_handler, self.dxl_ids, self.read_velocity_attrs)
        self.motor_position_velocity_reader = BulkMultiReader(self.port_handler, self.packet_handler, self.dxl_ids,
                                                     self.read_posvel_attrs)
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

        self.gyro_reader = BulkMultiReader(self.port_handler, self.packet_handler, [200], [(38, 2), (40, 2), (42, 2)])

        self.write_motor_delay([0] * 20)

        if self.use_bno055:
            from bno055_usb_stick_py import BnoUsbStick
            self.bno_usb_stick = BnoUsbStick()
            self.bno_usb_stick.activate_streaming()

    def disconnect(self):
        self.port_handler.closePort()

    def read_bno055_gyro(self):
        gyro_data = []

        packet = self.bno_usb_stick.recv_streaming_packet()
        euler = DEGREE2RAD(np.array(packet.euler))
        angvel = DEGREE2RAD(np.array(packet.g))
        if euler[0] > np.pi:
            euler[0] -= 2 * np.pi
        gyro_data.append(np.array([-euler[1], euler[2], -euler[0],   angvel[1], -angvel[0], angvel[2]]))
        return np.mean(gyro_data, axis=0)


    def read_motor_positions(self):
        return self.motor_reader.read()

    def read_motor_velocities(self):
        return self.motor_velocity_reader.read()

    def read_motor_positionvelocities(self):
        return self.motor_position_velocity_reader.read()

    def read_motor_voltages(self):
        return self.voltage_reader.read()

    def write_torque_enable(self, enable):
        data = 1 if enable else 0
        self.torque_enable_writer.write(np.array([data] * 20, dtype=np.int))

    def write_torque_limit(self, limit):
        self.torque_limit_writer.write([int(g) for g in limit])

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

    def read_gyro(self):
        return np.array(self.gyro_reader.read())



















