# Import dxl stuff up here

import ctypes

import motors.p2mx28 as mx28

from dynamixel_sdk import *

def setup_indirects(self, port_handler, packet_handler, motor_ids, attrs, indirect_root):
    """
    Given appropriate port constructs, list of motor ids, attributes, and
    an indirect index (from 0-55 on the mx28), set the indirect addresses
    on each mx28 so that the (indirected) attributes form a contiguous
    block of memory suitable for sync/bulk rw

    TODO There are two blocks of indirect addresses/data, each with a capacity
    of 28. This function ONLY sets up one contiguous block. It does not
    attempt to detect points where it should jump blocks and do so, or even
    fail when it should

    WARNING/TODO: Apparently, indirect addresses cannot be set while motor torque
    is enabled. Therefore, this function DISABLES TORQUE on all motors and
    makes no attempt to restore it to those motors which were enabled

    WARNING: This function is specifically designed for MX28 right now
    This link should clear up all the "magic numbers"
    http://emanual.robotis.com/docs/en/dxl/mx/mx-28/#control-table-data-address
    """

    # Array which will eventually store list of attributes in form
    # [(indirect_attr_1_address, attr1_len), ...]
    indirected_attrs = [None]

    # There are two blocks of indirect addresses/data, so it takes
    # a bit of logic to set the starting points right based on the indices
    # TODO Validate data lengths to make sure we have enough space given
    # the current attributes and indirect root
    indirect_addr = None
    data_addr = None
    if indirect_root <= 27:
        curr_addr = 2 * indirect_root + 168
        data_addr = 224 + indirect_root
    else:
        curr_addr = 2 * (indirect_root - 27) + 578
        data_addr = 634 + (indirect_root - 27)

    zero_torques()

    # Calculate and write appropriate addresses
    for attr_index, attr in enumerate(attrs):

        indirected_attrs[attr_index] = (data_addr, attr[1])
        data_addr += attr[1]

        for offset in range(attr[1]):
            for dxl_id in motor_ids:

                dxl_comm_result, dxl_error = packet_handler.write2ByteTxRx(port_handler,
                                                                                dxl_id, indirect_addr, attr[0] + offset)

                if dxl_comm_result != COMM_SUCCESS:
                    raise RuntimeError("Communication error on setting motor %i's address %i:\n%s"
                                       % dxl_id, attr[0],
                                       packet_handler.getTxRxResult(
                                           dxl_comm_result))
                elif dxl_error != 0:
                    raise RuntimeError("Hardware error on setting motor %i's address %i:\n%s" %
                                       dxl_id, attr[0],
                                       packet_handler.getRxPacketError
                                       (dxl_error))

            # Each address is more than one byte, as there are more than 256
            indirect_addr += 2

    return indirected_attrs


def zero_torques(port_handler, packet_handler, motor_ids):

    for dxl_id in self.motors_ids:

        # TODO Specific to MX28
        dxl_comm_result, dxl_error = self.packet_handler.write1ByteTxRx(self.port_handler, dxl_id, mx28.ADDR_TORQUE_ENABLE, 0)
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("Comm error while trying to disable motor %i:\n%s"
                               % dxl_id, self.packet_handler.getTxRxResult(dxl_comm_result))
        elif dxl_error != 0:
            raise RuntimeError("Hardware error while trying to disable motor %i:\n%s"
                               % dxl_id, self.packet_handler.getRxPacketError(dxl_error))


class BulkMultiReader():

    """
    Read multiple attributes via BulkRead
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs, indirect_root):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids

        self.attrs = attrs

        # Python compares tuples by entry going left to right, so the
        # following logic works
        self.block_start = max(self.attrs)[0]
        self.block_end = sum(min(self.attrs))
        self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()

    def construct_packet(self):

        packet = GroupBulkRead(self.port_handler, self.packet_handler)

        for motor_id in self.motor_ids:
            if not packet.addParam(motor_id, self.block_start, self.block_len):
                raise RuntimeError("Couldn't add parameter for motor %i",
                                   motor_id)

        return packet

    def read(self):

        results = [None] * (len(self.motor_ids) * len(self.attrs))

        comm_result = self.packet.txRxPacket()
        if comm_result != COMM_SUCCESS:
            raise RuntimeError(self.packet_handler.getTxRxResult(comm_result))

        for motor_index, motor_id in enumerate(self.motor_ids):
            for attr_index, attr in enumerate(self.attrs):
                if not self.packet.isAvailable(motor_id, *attr):
                    raise RuntimeError("Data unavailable for " + str(motor_id) + ", attribute " + str(attr))

                data_location = len(self.attrs) * motor_index + attr_index
                results[data_location] = self.packet.getData(
                    motor_id, *attr)

        return results

class SyncMultiWriter:

    """
    Write to the same contiguous block of memory across multiple motors

    WARNING: Using this with non-contiguous attributes will write anything
    that happens to be in between with 0
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.block_start = max(self.attrs)[0]
        self.block_end = sum(min(self.attrs))
        self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()

    def construct_packet(self):

        packet = GroupSyncWrite(self.port_handler, self.packet_handler,
                                     self.block_start, self.block_len)

        for motor_id in self.motor_ids:
            if not self.packets[attr_index].addParam(motor_id,
                                                     [0] * block_len):
                raise RuntimeError("Couldn't add any storage for motor %i, param %i" % motor_id)


    def write(self, targets):

        self.packet.clearParam()

        for motor_index, motor_id in enumerate(self.motor_ids):
            motor_data = [0] * self.block_len
            motor_targets = targets[motor_index * len(self.attrs)
                                    :(motor_index + 1) * len(self.attrs)]

            for attr_index in range(len(self.attrs)):
                attr = self.attrs[attr_index]

                # Replace the relevant subrange in the data array with the
                # byte list
                # TODO Big or little endian?
                motor_data[(attr[0] - self.block_start)
                           :sum(attr) - self.block_start] = list(targets[attr_index].to_bytes(attr[1], "little"))

            if not self.packet.addParam(motor_id, motor_data):
                raise RuntimeError("Couldn't set value for motor %i" % motor_id)

        dxl_comm_result = packet.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("%s" % self.packet_handler.getTxRxResult(dxl_comm_result))

if __name__ == "__main__":

    PROTOCOL_VERSION = 1
    BAUD = 1000000
    dxl_ids = [1]

    read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION)]

    write_attrs = [(mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE),
    (mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

    port_handler = PortHandler("/dev/ttyUSB0")
    if not port_handler.openPort():
        raise RuntimeError("Couldn't open port")
    if not port_handler.setBaudRate(BAUD):
        raise RuntimeError("Couldn't change baud rate")

    packet_handler = PacketHandler(PROTOCOL_VERSION)


    reader = BulkMultiReader(port_handler, packet_handler, dxl_ids, read_attrs, 0)
    # writer = MultiWriter(port_handler, packet_handler, dxl_ids, write_attrs)
    print(reader.read())
    # writer.write([1, 0])
