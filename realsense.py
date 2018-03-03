import logging
logging.basicConfig(level=logging.INFO)

import time
import numpy as np
import cv2
import pyrealsense as pyrs
import matplotlib.pyplot as plt
from pyrealsense.constants import rs_option
from pyrealsense.extstruct import rs_intrinsics
from pyrealsense.extstruct import rs_extrinsics

def get_cam():
    with pyrs.Service() as serv:
        with serv.Device() as dev:

            dev.apply_ivcam_preset(0)

            try:  # set custom gain/exposure values to obtain good depth image
                custom_options = [(rs_option.RS_OPTION_R200_LR_EXPOSURE, 30.0),
                                  (rs_option.RS_OPTION_R200_LR_GAIN, 100.0)]
                dev.set_device_options(*zip(*custom_options))
                print(dir(rs_intrinsics.width))
                print(rs_intrinsics.width.__getattribute__('size'),rs_intrinsics.height)
                print(rs_intrinsics.ppx,rs_intrinsics.ppy)
                print(rs_intrinsics.fx,rs_intrinsics.fy)
                print(rs_extrinsics.rotation, rs_extrinsics.translation)
            except pyrs.RealsenseError:
                pass  # options are not available on all devices

            cnt = 0
            last = time.time()
            smoothing = 0.9
            fps_smooth = 30

            while True:
                cnt += 1
                if (cnt % 10) == 0:
                    now = time.time()
                    dt = now - last
                    fps = 10/dt
                    fps_smooth = (fps_smooth * smoothing) + (fps * (1.0-smoothing))
                    last = now

                dev.wait_for_frames()
                c = dev.color
                c = cv2.cvtColor(c, cv2.COLOR_RGB2BGR)
                d = dev.depth * dev.depth_scale * 1000
                
                d = cv2.applyColorMap(d.astype(np.uint8), cv2.COLORMAP_RAINBOW)

                cd = np.concatenate((c, d), axis=1)

                cv2.putText(c, str(fps_smooth)[:4], (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0))
            
                cv2.imshow('',c)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                    


if __name__ == '__main__':
    get_cam()
