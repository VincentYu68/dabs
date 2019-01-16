from darwin.darwin_utils import *
from bno055_usb_stick_py import BnoUsbStick
bno_usb_stick = BnoUsbStick()
bno_usb_stick.activate_streaming()
for packet in bno_usb_stick.recv_streaming_generator():
    print(DEGREE2RAD(packet.euler))