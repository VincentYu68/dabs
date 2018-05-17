# Import dxl stuff up here

import ctypes

from dynamixel_sdk import *

import itertools

def check_comm_error(port_num):

    dxl_comm_result = dxl.getLastTxRxResult(port_num, PROTOCOL_VERSION)
    dxl_error = dxl.getLastRxPacketError(port_num, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeWarning(dxl.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        raise RuntimeWarning(dxl.getRxPacketError(PROTOCOL_VERSION, dxl_error))


class MultiReader():

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.sync_packets = [None] * len(attrs)
        self.construct_packets()

    def construct_packets(self):
        for i, attr in enumerate(self.attrs):
            self.sync_packets[i] = GroupSyncRead(self.port_handler, self.packet_handler,
                                                 *attr)
            for motor_id in self.motor_ids:
                if not self.sync_packets[i].addParam(motor_id):
                    raise RuntimeError("Couldn't add parameter for motor %i, param %i",
                                       motor_id, self.attrs[i][0])

    def read(self):

        results = [None] * (len(self.motor_ids) * len(self.attrs))

        for packet in self.sync_packets:
            comm_result = packet.txRxPacket()
            if comm_result != COMM_SUCCESS:
                raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                if not self.bulk_packet.isAvailable(motor_id, *attr):
                    raise RuntimeError("Data unavailable for " + str((motor_id, attr)))

                data_location = len(self.attrs) * motor_index + attr_index
                results[data_location] = self.bulk_packet.getData(
                    motor_id, *attr)

        return results

class MultiWriter:

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.sync_packets = [None] * len(attrs)

        # Note: In SDK version 3.6.0 the changeParam function is literally the same
        # as addParam, so this is sorta pointless. Keep this in case it's implemented
        # more efficiently in the future though
        self.construct_packets()

    def construct_packets(self):

        self.sync_packets = [GroupSyncWrite(self.port_handler, self.packet_handler, *attr)
                             for attr in self.attrs]

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                data_location = (motor_index * len(self.attrs)) + attr_index
                if not self.sync_packets[attr_index].addParam(motor_index, [0] * attr[1]):
                    raise RuntimeError("Couldn't add parameter for motor %i, param %i",
                                       motor_id, self.attrs[i][0])


    def write(self, targets):

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                data_location = (motor_index * len(self.attrs)) + attr_index
                if not self.sync_packets[attr_index].changeParam(motor_index,
                                                                 targets[data_location].to_bytes(attr[1], "big")):
                    raise RuntimeError("Couldn't set value for motor %i, param %i",
                                       motor_id, self.attrs[i][0])

        for packet in self.sync_packets:
            packet.txPacket()

if __name__ == "__main__":

    import motors.p1ax18 as ax18

    PROTOCOL_VERSION = 2
    BAUD = 1000000
    dxl_ids = [12, 18]

    read_attrs = [(ax18.ADDR_PRESENT_POSITION, ax18.LEN_PRESENT_POSITION)]

    write_attrs = [(ax18.ADDR_GOAL_POSITION, ax18.LEN_GOAL_POSITION),
                   (ax18.ADDR_TORQUE_ENABLE, ax18.LEN_TORQUE_ENABLE)]


    port_handler = PortHandler("/dev/ttyUSB0")
    if not port_handler.openPort():
        raise RuntimeError("Couldn't open port")
    if not port_handler.setBaudRate(1000000):
        raise RuntimeError("Couldn't change baud rate")

    packet_handler = PacketHandler(PROTOCOL_VERSION)

    # reader = MultiReader(port_handler, packet_handler, dxl_ids, read_attrs)
    writer = MultiWriter(port_handler, packet_handler, dxl_ids, write_attrs)
    # print(reader.read())
    writer.write([0, 1, 0, 1])
