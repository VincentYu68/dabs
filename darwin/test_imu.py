from darwin.basic_darwin import *
from darwin.darwin_utils import *
import numpy as np
from bno055_usb_stick_py import BnoUsbStick
bno_usb_stick = BnoUsbStick()
bno_usb_stick.activate_streaming()


if __name__ == '__main__':
    darwin = BasicDarwin()

    darwin.connect()

    darwin.write_torque_enable(True)
    darwin.write_pid(32, 0, 16)

    motor_pose = darwin.read_motor_positions()

    darwin.write_motor_goal(RADIAN2VAL(np.zeros(20)))


    for packet in bno_usb_stick.recv_streaming_generator():
        print(DEGREE2RAD(list(packet.euler)))