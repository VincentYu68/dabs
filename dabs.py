# Import dxl stuff up here

import ctypes

import motors.p2mx28 as mx28

from dynamixel_sdk import *

import itertools

def check_comm_error(port_num):

    dxl_comm_result = dxl.getLastTxRxResult(port_num, PROTOCOL_VERSION)
    dxl_error = dxl.getLastRxPacketError(port_num, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeWarning(dxl.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        raise RuntimeWarning(dxl.getRxPacketError(PROTOCOL_VERSION, dxl_error))


class IndirectMultiReader():

    def __init__(self, port_handler, packet_handler, motor_ids, attrs, indirect_root):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids

        self.attrs = attrs
        self.indirected_attrs = self.setup_indirects(indirect_root)

        self.sync_packet = self.construct_packet()

    def setup_indirects(self, indirect_root):

        indirected_attrs = [None]
        curr_addr = 2 * indirect_root + 168

        for attr_index, attr in enumerate(self.attrs):
            indirected_attrs[attr_index] = (curr_addr, attr[1])
            for offset in range(attr[1]):
                for dxl_id in self.motor_ids:

                    dxl_comm_result, dxl_error = self.packet_handler.write2ByteTxRx(self.port_handler,
                                                                                    dxl_id, curr_addr, attr[0] + offset)

                    if dxl_comm_result != COMM_SUCCESS:
                        raise RuntimeError("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
                    elif dxl_error != 0:
                        raise RuntimeError("%s" % self.packet_handler.getRxPacketError(dxl_error))
                curr_addr += 2

    def construct_packet(self):

        sync_packet = GroupSyncRead(self.port_handler, self.packet_handler,
                                         self.indirected_attrs[0][0],
                                         sum([attr[1] for attr in self.indirected_attrs]))

        for motor_id in self.motor_ids:
            if not self.sync_packets.addParam(motor_id):
                raise RuntimeError("Couldn't add parameter for motor %i",
                                   motor_id)

        return sync_packet

    def read(self):

        results = [None] * (len(self.motor_ids) * len(self.attrs))

        comm_result = self.sync_packet.txRxPacket()
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.indirected_attrs):
                if not self.sync_packet.isAvailable(motor_id, *attr):
                    raise RuntimeError("Data unavailable for " + str(motor_id) + ", attribute " + str(attr))

                data_location = len(self.attrs) * motor_index + attr_index
                results[data_location] = self.sync_packet.getData(
                    motor_id, *attr)

        return results

class MultiWriter:

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.sync_packets = None

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
                if not self.sync_packets[attr_index].addParam(motor_id, [0] * attr[1]):
                    raise RuntimeError("Couldn't add parameter for motor %i, param %i",
                                       motor_id, self.attrs[i][0])


    def write(self, targets):

        [packet.clearParam() for packet in self.sync_packets]

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                data_location = (motor_index * len(self.attrs)) + attr_index

                # TODO Seems that maybe big vs little endian doesn't matter? Certainly that's
                # not true...
                write_val = list(targets[data_location].to_bytes(attr[1], "little"))

                if not self.sync_packets[attr_index].addParam(motor_id, write_val):
                    raise RuntimeError("Couldn't set value for motor %i, param %i",
                                       motor_id, attr[0])

        for packet in self.sync_packets:

            dxl_comm_result = packet.txPacket()
            if dxl_comm_result != COMM_SUCCESS:
                raise RuntimeError("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))

if __name__ == "__main__":

    PROTOCOL_VERSION = 2
    BAUD = 1000000
    dxl_ids = [12]

    read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION)]

    write_attrs = [(mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE),
    (mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

    port_handler = PortHandler("/dev/ttyUSB0")
    if not port_handler.openPort():
        raise RuntimeError("Couldn't open port")
    if not port_handler.setBaudRate(BAUD):
        raise RuntimeError("Couldn't change baud rate")

    packet_handler = PacketHandler(PROTOCOL_VERSION)


    # TODO MX-28 Dependent
    dxl_comm_result, dxl_error = packet_handler.write1ByteTxRx(self.port_handler, dxl_ids[0], mx28.ADDR_TORQUE_ENABLE, 0)
    if dxl_comm_result != COMM_SUCCESS:
        print("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))
    elif dxl_error != 0:
        print("%s" % self.packet_handler.getRxPacketError(dxl_error))
    else:
        print("[ID:%03d] Dynamixel has been successfully connected" % dxl_id)

    reader = IndirectMultiReader(port_handler, packet_handler, dxl_ids, read_attrs, 0)
    # writer = MultiWriter(port_handler, packet_handler, dxl_ids, write_attrs)
    print(reader.read())
    # writer.write([1, 0])
