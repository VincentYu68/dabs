# Import dxl stuff up here

import ctypes

import dynamixel_functions as dxl

import itertools

COMM_SUCCESS = 0
COMM_TX_FAIL = -1001

def check_comm_error(port_num):

    dxl_comm_result = dxl.getLastTxRxResult(port_num, PROTOCOL_VERSION)
    dxl_error = dxl.getLastRxPacketError(port_num, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeWarning(dxl.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        raise RuntimeWarning(dxl.getRxPacketError(PROTOCOL_VERSION, dxl_error))


class MultiReader():

    def __init__(self, port_num, protocol_version, motor_ids, attrs):

        self.port_num = port_num
        self.protocol_version = protocol_version
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.param_list = list(itertools.product(self.motor_ids, self.attrs))

    def read(self):

        # bulk_read_packet = dxl.groupBulkRead(port_num, PROTOCOL_VERSION)

        # Add parameters to the bulk packetHandler

        # for id, attr in read_list:
        #     add_result = ctypes.c_ubyte(dxl.groupBulkReadAddParam(
        #         bulk_read_packet, id, attr[0], attr[1])).value
        #     if add_result != 1:
        #         raise RuntimeError("Failed to add storage for motor " + str(id)
        #                            + " address " + attr[0])

        # dxl.groupBulkReadTxRxPacket(bulk_read_packet)
        # check_comm_error(port_num)

        results = [None] * len(self.param_list)

        for index, (id, attr) in enumerate(self.param_list):
            # getdata_result = ctypes.c_ubyte(
            #     dxl.groupBulkReadIsAvailable(bulk_read_packet, id,
            #                                        attr[0], attr[1])).value
            # if getdata_result != 1:
            #     raise RuntimeWarning("Failed reading motor " + str(id),
            #                          + " address " + attr[0])
            # else:
            #     results[index] = dxl.groupBulkReadGetData(bulk_read_packet,
            #                                                     id, attr[0],
            #                                                     attr[1])

            if attr[1] == 1:
                val = dxl.read1ByteTxRx(self.port_num, self.protocol_version, id, attr[0])
            elif attr[1] == 2:
                val = dxl.read2ByteTxRx(self.port_num, self.protocol_version, id, attr[0])
            elif attr[2] == 4:
                val = dxl.read4ByteTxRx(self.port_num, self.protocol_version, id, attr[0])
            else:
                raise RuntimeError("Invalid data size")
            check_comm_error(port_num)
            results[index] = val
        return results

class MultiWriter:

    def __init__(self, port_num, protocol_version, motor_ids, attrs):

        self.port_num = port_num
        self.protocol_version = protocol_version
        self.motor_ids = motor_ids
        self.attrs = attrs

        self.param_list = list(itertools.product(self.motor_ids, self.attrs))

    def write(self, targets):


        # bulk_write_packet = dxl.groupBulkWrite(port_num, PROTOCOL_VERSION)

        for index, (id, attr) in enumerate(self.param_list):
            # add_result = ctypes.c_ubyte(
            #     dxl.groupBulkWriteAddParam(bulk_write_packet, id,
            #                                      attr[0],
            #                                      attr[1],
            #                                      targets[index], attr[1])).value
            # if add_result != 1:
            #     raise RuntimeError("Failed to add instruction for motor " + str(id))
            if attr[1] == 1:
                dxl.write1ByteTxRx(self.port_num, self.protocol_version, id,
                                         attr[0], targets[index])
            elif attr[1] == 2:
                dxl.write2ByteTxRx(self.port_num, self.protocol_version, id,
                                         attr[0], targets[index])
            elif attr[2] == 4:
                dxl.write4ByteTxRx(self.port_num, self.protocol_version, id,
                                         attr[0], targets[index])
            else:
                raise RuntimeError("Invalid data size")
            check_comm_error(port_num)

        # dxl.groupBulkWriteTxPacket(bulk_write_packet)
        # check_comm_error(port_num)

        # Clear bulkwrite parameter storage
        # dxl.groupBulkWriteClearParam(bulk_write_packet)

def initialize_port():

    BAUDRATE = 1000000
    DEVICENAME = "/dev/ttyUSB0".encode('utf-8')
    port_num = dxl.portHandler(DEVICENAME)

    # INitialize porthandler sturcts
    packet_handler = dxl.packetHandler()

    if not dxl.openPort(port_num):
        raise RuntimeError("Failed to open port")

    if not dxl.setBaudRate(port_num, BAUDRATE):
        raise RuntimeError("Failed to set baudrate")

    return port_num, packet_handler

if __name__ == "__main__":

    import motors.p1mx28 as mx28

    PROTOCOL_VERSION = 1
    dxl_ids = [12, 18]

    read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION),
                  (mx28.ADDR_TORQUE_ENABLE, mx28.LEN_TORQUE_ENABLE)]

    write_attrs = [(mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

    port_num, phandler = initialize_port()
    reader = MultiReader(port_num, PROTOCOL_VERSION, dxl_ids, read_attrs)
    writer = MultiWriter(port_num, PROTOCOL_VERSION, dxl_ids, write_attrs)
    print(reader.read())
    writer.write([0, 400])
