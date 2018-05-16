# Import dynamixel stuff up here

import ctypes

import dynamixel_functions as dynamixel
import p1mx28 as mx28

import itertools


PROTOCOL_VERSION = 1
COMM_SUCCESS = 0
COMM_TX_FAIL = -1001

dxl_ids = [12]

read_attrs = [(mx28.ADDR_PRESENT_POSITION, mx28.LEN_PRESENT_POSITION),
              (mx28.ADDR_PRESENT_SPEED, mx28.LEN_PRESENT_SPEED)]

write_attrs = [(mx28.ADDR_GOAL_POSITION, mx28.LEN_GOAL_POSITION)]

read_list = list(itertools.product(dxl_ids, read_attrs))
write_list = list(itertools.product(dxl_ids, write_attrs))

def check_comm_error(port_num):

    dxl_comm_result = dynamixel.getLastTxRxResult(port_num, PROTOCOL_VERSION)
    dxl_error = dynamixel.getLastRxPacketError(port_num, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        raise RuntimeWarning(dynamixel.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        raise RuntimeWarning(dynamixel.getRxPacketError(PROTOCOL_VERSION, dxl_error))

def initialize_port():

    BAUDRATE = 1000000
    DEVICENAME = "/dev/ttyUSB0".encode('utf-8')
    port_num = dynamixel.portHandler(DEVICENAME)

    # INitialize porthandler sturcts
    packet_handler = dynamixel.packetHandler()

    if not dynamixel.openPort(port_num):
        raise RuntimeError("Failed to open port")

    if not dynamixel.setBaudRate(port_num, BAUDRATE):
        raise RuntimeError("Failed to set baudrate")

    return port_num, packet_handler

def read_state(port_num):

    bulk_read_packet = dynamixel.groupBulkRead(port_num, PROTOCOL_VERSION)

    # Add parameters to the bulk packetHandler

    for id, attr in read_list:
        add_result = ctypes.c_ubyte(dynamixel.groupBulkReadAddParam(
            bulk_read_packet, id, attr[0], attr[1])).value
        if add_result != 1:
            raise RuntimeError("Failed to add storage for motor " + str(id)
                               + " address " + attr[0])

    dynamixel.groupBulkReadTxRxPacket(bulk_read_packet)
    check_comm_error(port_num)

    results = [None] * len(read_list)

    for index, (id, attr) in read_list:
        getdata_result = ctypes.c_ubyte(
            dynamixel.groupBulkReadIsAvailable(bulk_read_packet, id,
                                               attr[0], attr[1])).value
        if getdata_result != 1:
            raise RuntimeWarning("Failed reading motor " + str(id),
                                 + " address " + attr[0])
        else:
            results[index] = dynamixel.groupBulkReadGetData(bulk_read_packet,
                                                            id, attr[0],
                                                            attr[1])

    return results

def write_positions(port_num, targets):

    bulk_write_packet = dynamixel.groupBulkWrite(port_num, PROTOCOL_VERSION)

    for index, (id, attr) in enumerate(write_list):
        add_result = ctypes.c_ubyte(
            dynamixel.groupBulkWriteAddParam(bulk_write_packet, id,
                                             attr[0],
                                             attr[1],
                                             targets[index], attr[1])).value
        if add_result != 1:
            raise RuntimeError("Failed to add instruction for motor " + str(id))

    dynamixel.groupBulkWriteTxPacket(bulk_write_packet)
    check_comm_error(port_num)

    # Clear bulkwrite parameter storage
    dynamixel.groupBulkWriteClearParam(bulk_write_packet)

if __name__ == "__main__":
    port_num, phandler = initialize_port()


    dxl_model_number = dynamixel.pingGetModelNum(port_num, PROTOCOL_VERSION, dxl_ids[0])
    dxl_comm_result = dynamixel.getLastTxRxResult(port_num, PROTOCOL_VERSION)
    dxl_error = dynamixel.getLastRxPacketError(port_num, PROTOCOL_VERSION)
    if dxl_comm_result != COMM_SUCCESS:
        print(dynamixel.getTxRxResult(PROTOCOL_VERSION, dxl_comm_result))
    elif dxl_error != 0:
        print(dynamixel.getRxPacketError(PROTOCOL_VERSION, dxl_error))
    else:
        print("ping Succeeded on " + str(dxl_ids[0]))

    for i in range(5):
        read_state(port_num)
