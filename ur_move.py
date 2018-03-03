from __future__ import absolute_import, division, print_function
import socket 
import time
import math
import cv2
import pyrealsense as pyrs
import matplotlib.pyplot as plt
from pyrealsense.constants import rs_option
from multiprocessing import Process, Queue
from camera import keep_cam
from copy import deepcopy
import numpy as np
import torch


import _init_paths
import os
import sys
import numpy as np
import argparse
import pprint
import time
import cv2
import torch
from torch.autograd import Variable
import copy
from scipy.misc import imread
from model.utils.config import cfg, cfg_from_file, cfg_from_list, get_output_dir
from model.rpn.bbox_transform import clip_boxes
from model.nms.nms_wrapper import nms
from model.rpn.bbox_transform import bbox_transform_inv
from model.utils.net_utils import save_net, load_net, vis_detections
from model.utils.blob import im_list_to_blob
from model.faster_rcnn.vgg16 import vgg16
from model.faster_rcnn.resnet import resnet
from util import get_box
def _get_image_blob(im):
  """Converts an image into a network input.
  Arguments:
    im (ndarray): a color image in BGR order
  Returns:
    blob (ndarray): a data blob holding an image pyramid
    im_scale_factors (list): list of image scales (relative to im) used
      in the image pyramid
  """
  im_orig = im.astype(np.float32, copy=True)
  im_orig -= cfg.PIXEL_MEANS

  im_shape = im_orig.shape
  im_size_min = np.min(im_shape[0:2])
  im_size_max = np.max(im_shape[0:2])

  processed_ims = []
  im_scale_factors = []

  for target_size in cfg.TEST.SCALES:
    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
      im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
            interpolation=cv2.INTER_LINEAR)
    im_scale_factors.append(im_scale)
    processed_ims.append(im)

  # Create a blob to hold the input images
  blob = im_list_to_blob(processed_ims)

  return blob, np.array(im_scale_factors)


class UR5():

    def __init__(self, radius, center):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.radius = radius
        self.center = center
        self.initpose = [-0.64153, -0.04586, -0.02667, 0, 0, 0]
        #self.joint = [-1.41,-86.25,152.52,-213.32,-95.8,8.37]
        self.joint = [-0.0246,-1.505,2.66,-3.72,-1.67,0.26]
        
        assert len(self.center)==3
        assert len(self.initpose)==6
        #print "endpose:", self.endpose, "  radius:", self.radius, "  center:", self.center
        
        self.connect()
        self.reset()

    def connect(self):
        HOST = "192.168.1.18"
        PORT = 30002
        self.s.connect((HOST, PORT))
        print('ur5 connected\n')

    def disconnect(self):
        self.s.close()
        print('ur5 disconnected\n')

    def moveWrist(self):
        self.s.send(("movej([-0.0246,-1.505,2.66,-3.72,-1.67,0.26], t=4)" + "\n").encode())
        time.sleep(5)


    def movel(self, endpose,time=1):
        assert len(endpose)==6
        self.endpose = endpose
        # self.s.send(("movel(p[" + str(self.endpose[0]) + "," + str(self.endpose[1]) + "," + str(self.endpose[2]) + "," + str(self.endpose[3]) + "," + str(self.endpose[4]) + "," + str(self.endpose[5])+ "])" + "\n").encode())
        assert len(self.endpose)==6
        self.s.send(("movel(p" + str(self.endpose) + ", t="+str(time)+")" + "\n").encode())
        #data = self.s.recv(4096)
        #parseData(data)
        #print "endpose:", self.endpose

    def moveSphere(self, theta):
        # count move
        # self.x = self.x + math.cos(theta * math.pi)
        # self.y = self.y + math.sin(theta * math.pi)
        #safe check
        if (self.endpose[5]-self.initpose[5]+ math.cos(theta * math.pi)* 0.0785)<-1.9 or (self.endpose[5]-self.initpose[5]+ math.cos(theta * math.pi)*0.0785)>1.9 or (self.endpose[4]-self.initpose[4]- math.sin(theta * math.pi)*0.0785)<-1.2 or (self.endpose[4]-self.initpose[4]- math.sin(theta * math.pi)*0.0785)>0.5:
            print(self.endpose[5]+ math.cos(theta * math.pi)*0.0785)
            print(self.endpose[4]- math.sin(theta * math.pi)*0.0785)
            #self.reset()
            print("Warning!!!")
            return True

        self.endpose[5] = self.endpose[5] + math.cos(theta * math.pi) * 0.0785#x
        self.endpose[4] = self.endpose[4] - math.sin(theta * math.pi) * 0.0785#y
        #if self.endpose[]
        self.movel(self.endpose)
        #data = self.s.recv(4096)
        #d=parseData(data,self)
        time.sleep(1)

        return False

    

    def set_radius(self, radius):
        self.radius = radius
        print("radius:", self.radius, "  center:", self.center)

    def set_center(self, center):
        self.center = center
        print("radius:", self.radius, "  center:", self.center)

    def reset(self):
        self.endpose = [-0.64153, -0.04586, -0.02667, 0, 0, 0]
        self.joint = []
        self.s.send(("movel(p" + str(self.endpose) + ", t=3)" + "\n").encode())
        self.x = 0
        self.y = 0
        time.sleep(4)
        print('\nrobot reset')


if __name__ == '__main__':
    q1 = Queue()
    q2 = Queue()
    cam = Process(target=keep_cam, args=(q1, q2))
    cam.start()

    pascal_classes = np.asarray(['__background__',
                     'cola', 'holder', 'tape', 'measure', 'screwdriver', 'eraser',
                     'glue', 'stapler', 'juice', 'yellowbox', 'toolbox', 'soda'])
    fasterRCNN = vgg16(pascal_classes, pretrained=False, class_agnostic=False)
    fasterRCNN.create_architecture()
    load_name='/home/shin/faster-rcnn.pytorch/trained_models/vgg16/pascal_voc/faster_rcnn_1_45_1253.pth'
    print("load checkpoint %s" % (load_name))
    checkpoint = torch.load(load_name)
    fasterRCNN.load_state_dict(checkpoint['model'])
    fasterRCNN.cuda()
    fasterRCNN.eval()

    if 'pooling_mode' in checkpoint.keys():
        cfg.POOLING_MODE = checkpoint['pooling_mode']

    ######################## initiate gripper######################
    # rospy.init_node("robotiq_c_ctrl_test")
    # gripper = RobotiqCGripper()
    # gripper.wait_for_connection()
    # if gripper.is_reset():
    #     gripper.reset()
    #     gripper.activate()
    # [-625.69,154.03,-466.9,0.0086,-0.287,-1.4199]

    ################## initialize ur5#####################
    epsilon=70
    center=[320,240]
    u = UR5(5, [1,2,3])
    u.moveSphere(1)
    q1.put(1)
    q2.get(True)
    u.moveSphere(0)
    q1.put(1)
    q2.get(True)
    u.moveSphere(0)
    q1.put(1)
    q2.get(True)
    print("Knock!!!")
    time.sleep(5)
    for i in range(8):
        q1.put(1)
        q2.get(True)
        print(i+1)
        u.moveSphere(0.5)
    q1.put(1)
    q2.get(True)
    success=False
    with open('trajectory.txt','w') as f:
        f.write(str(u.endpose))
        f.write('\n')
        ret=False
        d=0
        up=True
        while(1):
            q1.put(1)
            q2.get(True)
            im=np.array(imread('/home/jeremy/1/1.jpg'))
            if im.shape==(480,640,3):
                box = get_box(fasterRCNN,im,12)
                # for target in range(12):
                #     print(str(target)+": "+str(get_box(fasterRCNN,im,target+1)))
                # print('\n')
            if box is None:
                if ret:
                    u.moveSphere(0.5)
                    u.moveSphere(0.5)

                    print('up')
                    d=1-d
                    ret=u.moveSphere(d)
                    f.write(str(u.endpose))
                    f.write('\n')
                    continue
                else:
                    ret=u.moveSphere(d)
                    f.write(str(u.endpose))
                    f.write('\n')
                    continue
                # else:
                #     u.moveSphere(0)
                #     f.write(str(u.endpose))
                #     f.write('\n')
                #     continue


            bound=[(box[0,0]+box[0,2])/2,(box[0,1]+box[0,3])/2]
            print(bound)
            if abs(bound[0]-center[0])<epsilon and abs(bound[1]-center[1])<epsilon:
                print("success!!!")
                success=True
                continue
            elif bound[0]>center[0]:
                dir=math.atan( (center[1]-bound[1])/(bound[0]-center[0]) )/math.pi
            elif bound[0]==center[0]:
                if bound[1]>center[1]:
                    dir=-0.5
                else:
                    dir=0.5
            elif bound[1]>center[1]:
                dir=math.atan( (center[1]-bound[1])/(bound[0]-center[0]) )/math.pi -1
            else:
                dir=math.atan( (center[1]-bound[1])/(bound[0]-center[0]) )/math.pi + 1

            print("direction: ",dir)
            u.moveSphere(dir)
            f.write(str(u.endpose))
            f.write('\n')

        ############ find best grasp pose ################
        time.sleep(10)# wait for knocking down
        u.movel([],time=1)
        u.moveSphere()
        u.moveSphere()





        # print('ready')
        # u.endpose[5]=u.endpose[5]+0.04
        # u.movel(u.endpose,time=1)
        # time.sleep(1)
        # f.write(str(u.endpose))
        # f.write('\n')

        # u.endpose[5]=u.endpose[5]+0.04
        # u.movel(u.endpose,time=1)
        # time.sleep(1)
        # f.write(str(u.endpose))
        # f.write('\n')

        # u.endpose[5]=u.endpose[5]-0.04
        # u.movel(u.endpose,time=1)
        # time.sleep(1)
        # f.write(str(u.endpose))
        # f.write('\n')

        # u.endpose[5]=u.endpose[5]-0.04
        # u.movel(u.endpose,time=1)
        # time.sleep(1)
        # f.write(str(u.endpose))
        # f.write('\n')
        

        # # ################ time for grasp ##############
        # u.movel([-0.77418,-0.07322,-0.03538,0.0228,-0.0368,0.5894],time=3)     
        # print('grasp')
        # f.write(str(u.endpose))
        # f.write('\n')
        # time.sleep(6)



        # ################ end of route2 #############
        # u.movel([-0.77418,-0.07322,0.29430,0.0227,-0.0368,0.5894],time=3)
        # f.write(str(u.endpose))
        # f.write('\n')
        # time.sleep(3)

        # u.movel([-0.359,0.4304,0.01747,2.0757,4.4727,1.658] ,time=3)
        # f.write(str(u.endpose))
        # f.write('\n')
        # time.sleep(4)

        # u.movel([-0.359,0.4304,-0.1306,2.0757,4.4727,1.658] ,time=3)
        # f.write(str(u.endpose))
        # f.write('\n')
        # time.sleep(4)


########################## end of route1 ############################
        # u.movel([-0.77026,-0.08112,0.21664,0.0224,-0.0288,0.5315],time=3)
        # f.write(str(u.endpose))
        # f.write('\n')
        # time.sleep(4)

        # u.movel([-0.44785,0.45716,0.01284,2.2173,3.9679,2.5362] ,time=3)
        # f.write(str(u.endpose))
        # f.write('\n')
        # time.sleep(4)

        # u.movel([-0.44787,0.45717,-0.14942,2.2175,3.9678,2.536],time=3)
        # f.write(str(u.endpose))
        # f.write('\n')



############### recover trajectory#############
    # pose=[0,0,0,0,0,0]
    # u = UR5(5, [1,2,3])
    # with open('/home/jeremy/Data222/Route1/trajectory.txt','r') as f:
    #     for line in f.readlines():
    #         linestr=line.replace("\n","").replace("[","").replace("]","").split(",")
    #         for i in range(6):
    #             pose[i]=float(linestr[i])
    #         print(pose)
    #         q1.put(1)
    #         q2.get(True)
    #         u.movel(pose)
    #         time.sleep(1)
    #     q1.put(1)
    #     q2.get(True)
        
    #     u.movel([-0.64152,0.18973,-0.02664,0,0.2966,1.5273],time=3)
    #     time.sleep(3)
    #     time.sleep(2)
    #     q1.put(1)
    #     q2.get(True)
        
    #     u.movel([-0.64152,0.18973,-0.02664,0,0.2966,1.5273-0.05])
    #     time.sleep(1)
    #     q1.put(1)
    #     q2.get(True)
        
    #     u.movel([-0.64152,0.18973,-0.02664,0,0.2966,1.5273-0.01])
    #     time.sleep(1)
    #     time.sleep(2)
    #     q1.put(1)
    #     q2.get(True)

    #     u.movel([-0.7493,-0.24328,0.076,0.877,-1.4527,-4.4175],time=3)
    #     time.sleep(3)
    #     q1.put(1)
    #     q2.get(True)
    #     print("grasp")
    #     time.sleep(7)
        
    #     u.movel([-0.7493,-0.24328,0.42545,0.877,-1.4527,-4.4175],time=3)
    #     time.sleep(3)
    #     q1.put(1)
    #     q2.get(True)

    #     u.movel([-0.5462,-0.43474,0.05134,1.1588,-0.4372,2.3036],time=3)
    #     time.sleep(3)
    #     q1.put(1)
    #     q2.get(True)
        
    #     u.movel([-0.5462,-0.43474,-0.1159,1.1588,-0.4372,2.3036],time=3)
    #     time.sleep(3)
    #     q1.put(1)
    #     q2.get(True)        
# u.movel([-0.64153, -0.04586, -0.02667, 0, 0.2965590305440537, 1.5273398721704161-0.08],time=1)
        # time.sleep(1)
        # u.movel([-0.61267,-0.23107,-0.35129,0.999,-0.8976,-4.4722],time=1)
        # time.sleep(6)

        # u.movel([-0.61267,-0.23107,-0.05771,0.999,-0.8976,-4.4722],time=3)
        # time.sleep(3)
        # u.movel([-0.3, 0.3998, -0.411, 0.6165, -0.7228, 1.60] ,time=3)
        # time.sleep(4)
        # u.movel([-0.34, 0.4638, -0.58, 0.6817, -0.7441, 1.6] ,time=3)
