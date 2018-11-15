# Import dxl stuff up here

from dynamixel_sdk import *
import itertools

PROTOCOL_VERSION = 1

if PROTOCOL_VERSION == 1:
    import motors.p1mx28 as mx28
else:
    import motors.p2mx28 as mx28

def setup_indirects(self, port_handler, packet_handler, motor_ids, attrs,
                    indirect_root):
    """
    Given appropriate port constructs, list of motor ids, attributes, and
    an indirect index (from 0-55 on the mx28), set the indirect addresses
    on each mx28 so that the (indirected) attributes form a contiguous
    block of memory suitable for sync/bulk rw

    TODO There are two blocks of indirect addresses/data, each with a capacity
    of 28. This function ONLY sets up one contiguous block. It does not
    attempt to detect points where it should jump blocks and do so, or even
    fail when it should

    WARNING/TODO: Apparently, indirect addresses cannot be set while motor
    torque is enabled. Therefore, this function DISABLES TORQUE on all motors
    and makes no attempt to restore it to those motors which were enabled

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

    zero_torques(port_handler, packet_handler, motor_ids)

    # Calculate and set the indirect addresses appropriately
    for attr_index, attr in enumerate(attrs):

        indirected_attrs[attr_index] = (data_addr, attr[1])
        data_addr += attr[1]

        # Addresses / attributes may span multiple words, so there's a tiny bit
        # of logic to handle that correctly; ie if an attr spans four words,
        # make sure all four of those are being mirrored
        for offset in range(attr[1]):
            for dxl_id in motor_ids:

                dxl_comm_result,
                dxl_error = packet_handler.write2ByteTxRx(port_handler,
                                                          dxl_id,
                                                          indirect_addr,
                                                          attr[0] + offset)

                if dxl_comm_result != COMM_SUCCESS:
                    raise RuntimeError("Communication error on setting " \
                                       + "mtor %i's address %i:\n%s"
                                       % dxl_id, attr[0],
                                       packet_handler.getTxRxResult(
                                           dxl_comm_result))
                elif dxl_error != 0:
                    raise RuntimeError("Hardware error on setting motor %i's " \
                                       + "address %i:\n%s" %
                                       dxl_id, attr[0],
                                       packet_handler.getRxPacketError
                                       (dxl_error))

            # Each address is two bytes, as there are more than 256
            indirect_addr += 2

    return indirected_attrs


def zero_torques(port_handler, packet_handler, motor_ids):

    for dxl_id in motor_ids:

        # TODO Specific to MX28
        comm_result, error = packet_handler.write1ByteTxRx(
            port_handler, dxl_id, mx28.ADDR_TORQUE_ENABLE, 0)

        if comm_result != COMM_SUCCESS:
            raise RuntimeError("Comm error on trying to disable motor %i:\n%s"
                               % dxl_id,
                               self.packet_handler.getTxRxResult(comm_result))
        elif error != 0:
            raise RuntimeError("Hardware error on disabling motor %i:\n%s"
                               % dxl_id,
                               packet_handler.getRxPacketError(error))


class BulkMultiReader():

    """
    Read multiple attributes via BulkRead
    """

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids

        self.attrs = attrs

        # Python compares tuples by entry going left to right, so the
        # following logic works
        self.block_start = min(self.attrs)[0]
        self.block_end = sum(max(self.attrs))
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
                    raise RuntimeError("Data unavailable for " + str(motor_id)
                                       + ", attribute " + str(attr))

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

        self.block_start = min(self.attrs)[0]
        self.block_end = sum(max(self.attrs))
        self.block_len = self.block_end - self.block_start

        self.packet = self.construct_packet()

    def construct_packet(self):

        packet = GroupSyncWrite(self.port_handler, self.packet_handler,
                                     self.block_start, self.block_len)

        for motor_id in self.motor_ids:
            if not packet.addParam(motor_id,
                                   [0] * self.block_len):
                raise RuntimeError("Couldn't add any storage for motor %i, " \
                                   + "param %i" % motor_id)

        return packet


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
                motor_data[(attr[0] - self.block_start)
                           :sum(attr) - self.block_start] = \
                                            list(motor_targets[attr_index]
                                                 .to_bytes(attr[1], "little"))

            if not self.packet.addParam(motor_id, motor_data):
                raise RuntimeError("Couldn't set value for motor %i"
                                   % motor_id)

        dxl_comm_result = self.packet.txPacket()
        if dxl_comm_result != COMM_SUCCESS:
            raise RuntimeError("%s"
                               % self.packet_handler.getTxRxResult( \
                                                            dxl_comm_result))


class MultiReader():

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.identified_attrs = list(itertools.product(self.motor_ids,
                                                       self.attrs))

    def read(self):


        results = [None] * len(self.identified_attrs)

        for index, (id, attr) in enumerate(self.identified_attrs):

            if attr[1] == 1:
                val, comm, err = \
                        self.packet_handler.read1ByteTxRx(self.port_handler,
                                                          self.protocol_version,
                                                          id, attr[0])
            elif attr[1] == 2:
                val, comm, err = \
                        self.packet_handler.read2ByteTxRx(self.port_handler,
                                                          self.protocol_version,
                                                          id, attr[0])
            elif attr[2] == 4:
                val, comm, err = \
                        self.packet_handler.read4ByteTxRx(self.port_handler,
                                                          self.protocol_version,
                                                          id, attr[0])
            else:
                raise RuntimeError("Invalid data size")

            if comm != COMM_SUCCESS:
                raise RuntimeError("Comm error on reading motor %i, " \
                                   + "attribute %s"% id, attr)
            elif err != 0:
                raise RuntimeError("Hardware error on reading motor %i, " \
                                   + "attribute %s" % id, attr)

            results[index] = val
        return results

class MultiWriter:

    def __init__(self, port_handler, packet_handler, motor_ids, attrs):

        self.port_handler = port_handler
        self.packet_handler = packet_handler
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.identified_attrs = list(itertools.product(self.motor_ids,
                                                       self.attrs))

    def write(self, targets):

        for index, (id, attr) in enumerate(self.identified_attrs):
            if attr[1] == 1:
                comm, err = \
                    self.packet_handler.write1ByteTxRx(self.port_handler, id,
                                                       attr[0], targets[index])
            elif attr[1] == 2:
                comm, err = \
                    self.packet_handler.write2ByteTxRx(self.port_handler, id,
                                                       attr[0], targets[index])
            elif attr[2] == 4:
                comm, err = \
                    self.packet_handler.write4ByteTxRx(self.port_handler, id,
                                                       attr[0], targets[index])
            else:
                raise RuntimeError("Invalid data size")

            if comm != COMM_SUCCESS:
                raise RuntimeError("Comm error on writing motor %i," \
                                   + "attribute %s" % id, attr)
            elif err != 0:
                raise RuntimeError("Hardware error on writing motor %i, " \
                                   + "attribute %s" % id, attr)

if __name__ == "__main__":
    BAUD = 1000000
    dxl_ids = [1, 2]

    read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION),
                  (mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE)]

    write_attrs = [(mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE)]
    # (mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

    port_handler = PortHandler("/dev/ttyUSB0")
    if not port_handler.openPort():
        raise RuntimeError("Couldn't open port")
    if not port_handler.setBaudRate(BAUD):
        raise RuntimeError("Couldn't change baud rate")

    packet_handler = PacketHandler(PROTOCOL_VERSION)

    ###############
    # READ ME !!! #
    ###############

    # To take advantage of the indirect addressing, all you would need to do is
    # something like "iattrs = setup_indirects(port, packet, dxl_ids, attrs, 0)"
    # then you would be able to pass iattrs to the next two lines where
    # read_attrs and write_attrs are being passed now

    # Unfortunately protocol 2 doesn't seem to work on the robot motors :(
    # I opened up a github issue here
    # https://github.com/ROBOTIS-GIT/DynamixelSDK/issues/191
    # where you can communicate with the devs if you want to tackle it

    #############################
    # READER/WRITER DIFFERENCES #
    #############################

    # Most of this is explained elsewhere, but I'm consolidating a bit

    # All classes have the same interface so all you have to do to switch
    # between them is change the name wherever they're instantiated

    # MultiReader and MultiWriter are the simplest, and should *just work*
    # Unfortunately they're also slow (probably. I haven't tested speed yet)
    # since they send a packet per attribute per motor

    # BulkMultiReader reads a block of contiguous memory (which it
    # auto-generates) and extracts the relevant attributes. In practice,
    # there's probably no reason *not* to use this over MultiReader
    # (short version: use it)
    # TODO It could be made even faster with some slight modifications to make
    # use GroupSyncRead packets instead, but then it wouldn't work on protocol 1

    # SyncMultiWriter writes a block of contiguous memory, which will be great
    # once indirects actually work. Unfortunately, the current implementation
    # clobbers anything in the block of memory which isn't explicitly set
    # DONT USE IT until indirects work :(

    ##############
    # I/O FORMAT #
    ##############

    # Motor data is kept contiguous;
    # Also, there's no sorting of any kind performed on motor IDs or attribute
    # addresses

    # that is, if I set up a reader with args
    # dxl_ids = (2, 1) and attributes (a, b) then the read() function will
    # return an array formatted like (m2.a, m2.b, m1.a, m1.b)

    # Similarly, if I set up a writer as above then the write() function would
    # expect the data in the same format

    reader = BulkMultiReader(port_handler, packet_handler, dxl_ids, read_attrs)
    writer = MultiWriter(port_handler, packet_handler, dxl_ids, write_attrs)
    print(reader.read())
    writer.write([1, 1])
