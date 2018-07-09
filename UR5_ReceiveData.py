import socket 
import numpy as np
import struct
import os
import sys
import time
import logging  
import random 
import cv2
from multiprocessing import Process, Queue

from util import setup_logger    
from ForceTorque import ForceTorque
from realsense2 import get_cam
from UR5_realtime import URRTMonitor
from UR_secmon import SecondaryMonitor

                                    

class UR5():

    def __init__(self,q1,q2):
        self.q1=q1
        self.q2=q2
        self.save_dir = "/home/newuhe/Data9/"
        self.test_dir = "/home/newuhe/Data9/Test/"
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        
        # self.rtmon = URRTMonitor("192.168.1.147")  # som information is only available on rt interface
        # self.rtmon.start()
        # self.rtmon.set_csys(None)
        self.secmon = SecondaryMonitor("192.168.1.147") 
        self.log = {}
        
        setup_logger('worker_log', r'worker_log')
        self.log['worker_log'] = logging.getLogger(
        'worker_log')

        self.test_log = {}
        setup_logger('test_log', r'test_log')
        self.test_log['test_log'] = logging.getLogger(
        'test_log')
        
        self.force = 18
        self.d = -0.3
        self.v = 0.0003

        ## usb
        #self.primitive = [-0.61306625, -0.05149676 , 0.0065 ,-2.8868024,   1.23261833,0.00922825]
        
        ## round
        # self.primitive = [-0.30288  ,0.38096 , 0.019 , 2.3236 , -2.1558 , -0.0266]
        #(up pose)self.primitive = [-0.3026  ,0.37825 , 0.01041 , 2.3236 , -2.1558 , -0.0266]
        #(new round)self.primitive = [-4.96312008e-01 , 5.53456907e-01  ,1.12701285e-03, -1.16724081e+00, .87952391e+00 , 1.91615209e-02]
        self.primitive=[-0.50454686 , 0.56308032 , 0.00392812,  1.19445411 ,-2.90431322 , 0.05488621]

        self.half_count = 1
        self.rcount = 3

        self.count = 0
        self.pcount = 0

        ## for orientation
        self.sequence = []
        self.rinit = -0.21
        self.interval = 0.07

        ## for translation
        self.xcount=3
        self.xhalf = (self.xcount-1)/2
        self.ycount=3
        self.yhalf = (self.ycount-1)/2
        
        ## round
        self.z = self.primitive[2]
        self.yinterval = 0.006 / self.xhalf
        self.xinterval = 0.006 / self.yhalf
        
        ## usb
        # self.z = 0.0065
        # self.yinterval = 0.003 / self.yhalf
        # self.xinterval = 0.006 / self.xhalf
       
        # for i in range(9):b
        #     self.xsequence.append(x)
        #     self.ysequence.append(y)
        #     x += self.xinterval
        #     y += self.yinterval
        # self.ysequence_reverse = list(reversed(self.ysequence))
        # self.sequence_reverse = list(reversed(self.sequence))

        # print(self.xsequence)
        # print(self.ysequence)
        # print(self.sequence)
        #self.log['worker_log'].info('x- = %0.5f, x+ = %0.5f, y- = %0.5f, y+ = %0.5f, r- = %0.5f, r+ = %0.5f' % (self.xsequence[0],self.xsequence[8],self.ysequence[0],self.ysequence[8],self.sequence[0],self.sequence[8]))

        self.connect()
        
        ## reset to ground truth      TCP:(0,0,0.225,0,0,0)
        self.movel(self.primitive,t=3)
        #self.movej([-1.20141346,-1.24530489,1.70199537,-2.01912338,-1.5878895,-0.38679821],t=3)
                
        ## take reference picture
        time.sleep(0.5)
        self.move_down(v=self.v)
        fz = 0
        z=0.1
        # (up pose)while( abs(fz) < self.force and z>0.001):
        #(new round)while( abs(fz) < self.force and z>-0.005):
        while( abs(fz) < self.force and z>-0.004):
            pose = self.secmon.get_cartesian_info(False)
            z = pose["Z"]
            f.getForce()
            fx, fy, fz, px, py, pz = f.parseData()
        self.stop()

        ## take picture
        self.q1.put(1)
        pic, depth = self.q2.get(True)
        cv2.imwrite("/home/newuhe/Data9/" + 'reference.jpg',pic)
        cv2.imwrite("/home/newuhe/Data9/Test/" + 'reference.jpg',pic)
        
                

        ## pose init
        self.yi = int(self.count/self.ycount)
        self.xi = self.count - self.ycount * self.yi
        self.rxi = int(self.pcount/self.rcount)
        self.ryi = self.pcount - (self.half_count * 2 + 1) * self.rxi
        if self.yi % 2 == 0:
            if self.xi % 2 ==0:
                #orientation1
                if self.rxi % 2 == 0:
                    self.move(self.primitive,[ self.xinterval*(self.xhalf-self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(self.half_count-self.rxi), self.interval*(self.half_count-self.ryi), 0])
                else:
                    self.move(self.primitive,[ self.xinterval*(self.xhalf-self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(self.half_count-self.rxi), self.interval*(-self.half_count+self.ryi), 0])
            else:
                #orientation2
                if self.rxi % 2 == 0:
                    self.move(self.primitive,[ self.xinterval*(self.xhalf-self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(-self.half_count+self.rxi), self.interval*(-self.half_count+self.ryi), 0])
                else:
                    self.move(self.primitive,[ self.xinterval*(self.xhalf-self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(-self.half_count+self.rxi), self.interval*(self.half_count-self.ryi), 0])
        else:
            if self.xi % 2 ==0:
                #orientation2
                if self.rxi % 2 == 0:
                    self.move(self.primitive,[ self.xinterval*(-self.xhalf+self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(-self.half_count+self.rxi), self.interval*(-self.half_count+self.ryi), 0])
                else:
                    self.move(self.primitive,[ self.xinterval*(-self.xhalf+self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(-self.half_count+self.rxi), self.interval*(self.half_count-self.ryi), 0])
            else:
                #orientation1
                if self.rxi % 2 == 0:
                    self.move(self.primitive,[ self.xinterval*(-self.xhalf+self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(self.half_count-self.rxi), self.interval*(self.half_count-self.ryi), 0])
                else:
                    self.move(self.primitive,[ self.xinterval*(-self.xhalf+self.xi), self.yinterval*(self.yhalf-self.yi), 0, self.interval*(self.half_count-self.rxi), self.interval*(-self.half_count+self.ryi), 0])
            
    def connect(self):
        HOST = "192.168.1.147"
        PORT = 30002
        self.s.connect((HOST, PORT))
        print('ur5 connected\n')

    def disconnect(self):
        self.s.close()
        print('ur5 disconnected\n')

    def move(self, pose1, pose2, t=1):
        self.s.send( ("movel(pose_trans(p"+ str(pose1) + ",p"+str(pose2)+"),t=" + str(t) + ")\n").encode() )
        time.sleep(t+0.2)

    def move_relative(self, pose, t=1):
        self.s.send( ("movel(pose_trans(get_actual_tcp_pose(),p"+str(pose)+")" + ",t=" + str(t) + ")"+"\n").encode() )
        time.sleep(t+0.2)

    def move_down(self, v=0.0005):
        self.s.send( ("movel(pose_trans(get_actual_tcp_pose(),p[0,0,0.05,0,0,0])" + ",v=" + str(v) + ")"+"\n").encode() )

    def movel(self, endpose, t=1):
        self.s.send( ("movel(p" + str(endpose) + ",t=" + str(t) + ")" + "\n").encode() )
        time.sleep(t+0.2)

    def movej(self, endpose, t=1):
        self.s.send( ("movej(" + str(endpose) + ",t=" + str(t) + ")" + "\n").encode() )
        time.sleep(t+0.2)

    def stop(self):
        self.s.send( ("stopl(500)" + "\n").encode() )

    def getTcpPose(self):
        #print(self.rtmon.getTCF(True))
        return self.rtmon.getTCF(True) 

    def getJoint(self):
        print(self.rtmon.getActual(True))

    def getForce(self,f):
        self.move_down(v=self.v)
        fz = 0
        while( abs(fz) < self.force ):
            f.getForce()
            fx, fy, fz, px, py, pz = f.parseData()
        print("force: ",fx,fy,fz,px,py,pz)

        #self.move_up()
        #self.getTcpPose()
        #self.stop()
        
        # fo = []
        # for i in range(3):
        #     self.move_down(v=self.v)
        #     fz = 0
        #     while( abs(fz) < self.force ):
        #         f.getForce()
        #         fx, fy, fz, px, py, pz = f.parseData()
        #     fo.append([fx, fy, fz, px, py, pz ])
        #     self.move_up()
        # #self.getTcpPose()
        # m=np.mean(np.array(fo),axis=0)
        # #print("force: ",fx,fy,fz,px,py,pz)
        # print("force: ",m[0],m[1],m[2],m[3],m[4],m[5])
        # self.stop()

    def orientation1(self,x,y): # -0.5 ~ 0.5
        for rx in range(self.rxi,self.rcount):
        # self.sequence[self.rxi:]:
            if self.rxi % 2 == 0:
                for ry in range(self.ryi,self.rcount):
                #for ry in self.sequence[self.ryi:]:
                    # self.pose[2] = self.z
                    # self.pose[3] = rx
                    # self.pose[4] = ry  
                    if x>=0:
                        self.move(self.primitive,[ self.xinterval*(self.xhalf-x), self.yinterval*(self.xhalf-y), 0, self.interval*(self.half_count-rx), self.interval*(self.half_count-ry), 0])
                    else: 
                        self.move(self.primitive,[ self.xinterval*(-self.xhalf-x), self.yinterval*(self.yhalf-y), 0, self.interval*(self.half_count-rx), self.interval*(self.half_count-ry), 0]) 
                    
                    ## get force torque
                    self.move_down(v=self.v)
                    fz = 0
                    z = 0.1
                    #while( abs(fz) < self.force and z>0.001 ):
                    #while( abs(fz) < self.force and z>-0.005):
                    while( abs(fz) < self.force and z>-0.004):
                        pose = self.secmon.get_cartesian_info(False)
                        z = pose["Z"]
                        f.getForce()
                        fx, fy, fz, px, py, pz = f.parseData()
                    self.stop()

                    #pose = self.getTcpPose()
                    #pose = self.secmon.get_cartesian_info(False)
                    pose = [pose["X"], pose["Y"], pose["Z"], pose["Rx"], pose["Ry"], pose["Rz"]]
                    
                    ## take picture
                    self.q1.put(1)
                    pic, depth = self.q2.get(True)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'.jpg',pic)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'_depth.jpg',depth)
                    np.savez(self.save_dir + str(self.count) + "/" + str(self.pcount) + ".npz", force = [fx,fy,fz,px,py,pz],pos = pose, pic = depth,count = [self.count,self.pcount])
                    self.log['worker_log'].info('x = %0.4f,y = %0.4f,z = %0.4f,rx = %0.4f,ry = %0.4f,rz = %0.4f,fx = %0.4f, fy = %0.4f,fz = %0.4f,px = %0.4f,py = %0.4f,pz = %0.4f, count = %d, pcount = %d' % (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],fx,fy,fz,px,py,pz, self.count,self.pcount))
                    
                    self.pcount += 1
                    self.ryi += 1
                self.rxi += 1
                self.ryi = 0

            else:
                for ry in range(self.ryi,self.rcount):
                #for ry in self.sequence_reverse[self.ryi:]:
   
                    if x>=0:
                        self.move(self.primitive,[ self.xinterval*(self.xhalf-x), self.yinterval*(self.yhalf-y), 0, self.interval*(self.half_count-rx), self.interval*(-self.half_count+ry), 0])
                    else:
                        self.move(self.primitive,[ self.xinterval*(-self.xhalf-x), self.yinterval*(self.yhalf-y), 0, self.interval*(self.half_count-rx), self.interval*(-self.half_count+ry), 0])

                    ## get force torque
                    self.move_down(v=self.v)
                    fz = 0
                    z=0.1
                    #while( abs(fz) < self.force and z>0.001 ):
                    #while( abs(fz) < self.force and z>-0.005):
                    while( abs(fz) < self.force and z>-0.004):
                        pose = self.secmon.get_cartesian_info(False)
                        z = pose["Z"]
                        f.getForce()
                        fx, fy, fz, px, py, pz = f.parseData()
                    #self.stop()

                    #pose = self.getTcpPose()
                    #pose = self.secmon.get_cartesian_info(False)
                    pose = [pose["X"], pose["Y"], pose["Z"], pose["Rx"], pose["Ry"], pose["Rz"]]

                    ## take picture
                    self.q1.put(1)
                    pic, depth = self.q2.get(True)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'.jpg',pic)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'_depth.jpg',depth)
                    np.savez(self.save_dir + str(self.count) + "/" + str(self.pcount) + ".npz", force = [fx,fy,fz,px,py,pz],pos = pose, pic = depth,count = [self.count,self.pcount])
                    self.log['worker_log'].info('x = %0.4f,y = %0.4f,z = %0.4f,rx = %0.4f,ry = %0.4f,rz = %0.4f,fx = %0.4f, fy = %0.4f,fz = %0.4f,px = %0.4f,py = %0.4f,pz = %0.4f, count = %d, pcount = %d' % (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],fx,fy,fz,px,py,pz, self.count,self.pcount))
                    
                    self.pcount += 1
                    self.ryi += 1
                self.rxi += 1
                self.ryi = 0

    def orientation2(self,x,y): # 0.5 ~ -0.5
        #for rx in self.sequence_reverse[self.rxi:]:
        for rx in range(self.rxi,self.rcount):
            if self.rxi % 2 == 0:
                #for ry in self.sequence_reverse[self.ryi:]:
                for ry in range(self.ryi,self.rcount):   
                    if x>=0:
                        self.move(self.primitive,[ self.xinterval*(self.xhalf-x), self.yinterval*(self.yhalf-y), 0, self.interval*(-self.half_count+rx), self.interval*(-self.half_count+ry), 0])
                    else:
                        self.move(self.primitive,[ self.xinterval*(-self.xhalf-x), self.yinterval*(self.yhalf-y), 0, self.interval*(-self.half_count+rx), self.interval*(-self.half_count+ry), 0])
                    
                    ## get force torque
                    self.move_down(v=self.v)
                    fz = 0
                    z=0.1
                    #while( abs(fz) < self.force and z>0.001 ):
                    #while( abs(fz) < self.force and z>-0.005):
                    while( abs(fz) < self.force and z>-0.004):
                        pose = self.secmon.get_cartesian_info(False)
                        z = pose["Z"]
                        f.getForce()
                        fx, fy, fz, px, py, pz = f.parseData()
                    self.stop()

                    #pose = self.getTcpPose()
                    #pose = self.secmon.get_cartesian_info(False)
                    pose = [pose["X"], pose["Y"], pose["Z"], pose["Rx"], pose["Ry"], pose["Rz"]]
                    
                    ## take picture
                    self.q1.put(1)
                    pic, depth = self.q2.get(True)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'.jpg',pic)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'_depth.jpg',depth)
                    np.savez(self.save_dir + str(self.count) + "/" + str(self.pcount) + ".npz", force = [fx,fy,fz,px,py,pz],pos = pose, pic = depth,count = [self.count,self.pcount])
                    self.log['worker_log'].info('x = %0.4f,y = %0.4f,z = %0.4f,rx = %0.4f,ry = %0.4f,rz = %0.4f,fx = %0.4f, fy = %0.4f,fz = %0.4f,px = %0.4f,py = %0.4f,pz = %0.4f, count = %d, pcount = %d' % (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],fx,fy,fz,px,py,pz, self.count,self.pcount))
                    
                    self.pcount += 1
                    self.ryi += 1
                self.rxi += 1
                self.ryi = 0

            else:
                #for ry in self.sequence[self.ryi:]:
                for ry in range(self.ryi,self.rcount):
                    if x>=0:
                        self.move(self.primitive,[ self.xinterval*(self.xhalf-x), self.yinterval*(self.xhalf-y), 0, self.interval*(-self.half_count+rx), self.interval*(self.half_count-ry), 0])
                    else:        
                        self.move(self.primitive,[ self.xinterval*(-self.xhalf-x), self.yinterval*(self.yhalf-y), 0, self.interval*(-self.half_count+rx), self.interval*(self.half_count-ry), 0])
                        

                    ## get force torque
                    self.move_down(v=self.v)
                    fz = 0
                    z=0.1
                    #while( abs(fz) < self.force and z>0.001 ):
                    #while( abs(fz) < self.force and z>-0.005):
                    while( abs(fz) < self.force and z>-0.004):
                        pose = self.secmon.get_cartesian_info(False)
                        z = pose["Z"]
                        f.getForce()
                        fx, fy, fz, px, py, pz = f.parseData()
                    self.stop()

                    
                    pose = [pose["X"], pose["Y"], pose["Z"], pose["Rx"], pose["Ry"], pose["Rz"]]
                    
                    ## take picture
                    self.q1.put(1)
                    pic, depth = self.q2.get(True)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'.jpg',pic)
                    cv2.imwrite(self.save_dir + str(self.count) + '/' + str(self.pcount) +'_depth.jpg',depth)
                    np.savez(self.save_dir + str(self.count) + "/" + str(self.pcount) + ".npz", force = [fx,fy,fz,px,py,pz],pos = pose, pic = depth,count = [self.count,self.pcount])
                    self.log['worker_log'].info('x = %0.4f,y = %0.4f,z = %0.4f,rx = %0.4f,ry = %0.4f,rz = %0.4f,fx = %0.4f, fy = %0.4f,fz = %0.4f,px = %0.4f,py = %0.4f,pz = %0.4f, count = %d, pcount = %d' % (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],fx,fy,fz,px,py,pz, self.count,self.pcount))
                    
                    self.pcount += 1
                    self.ryi += 1
                self.rxi += 1
                self.ryi = 0

    def traverse(self):
        for y in range(self.yi,self.ycount):
            if self.yi % 2 == 0:
                for x in range(self.xi,self.xcount):
                    # self.pose[0] = x
                    # self.pose[1] = y
                    #self.move(self.primitive,[x*self.xinterval, y*self.yinterval,0,0,0,0])

                    if not os.path.exists(self.save_dir+str(self.count)):
                        os.mkdir(self.save_dir+str(self.count))
                    
                    if self.xi % 2 ==0 :
                        self.orientation1(x,y)
                        self.pcount = 0
                        self.rxi = 0
                    else:
                        self.orientation2(x,y)
                        self.pcount = 0
                        self.rxi = 0
                    self.count += 1
                    self.xi += 1
                self.yi += 1
                self.xi = 0

            else:
                # for x in self.ysequence_reverse[self.xi:]:
                for x in range(self.xi, self.xcount):
                    # self.pose[0] = x
                    # self.pose[1] = y
                    #self.move(-x*,y*,0,0,0,0)

                    if not os.path.exists(self.save_dir+str(self.count)):
                        os.mkdir(self.save_dir+str(self.count))
                    
                    if self.xi % 2 == 0:
                        self.orientation2(-x,y)
                        self.pcount = 0
                        self.rxi = 0
                    else:
                        self.orientation1(-x,y)
                        self.pcount = 0
                        self.rxi = 0
                    self.count += 1
                    self.xi += 1
                self.yi += 1
                self.xi = 0

    def testData(self):
        test_count = 0
        sequence_count = 0
        for j in range(20):
            # x = random.uniform(-self.xinterval*self.xhalf, self.xinterval*self.xhalf)
            # y = random.uniform(-self.yinterval*self.yhalf, self.yinterval*self.yhalf)
            # rx = random.uniform(self.rinit, -self.rinit)
            # ry = random.uniform(self.rinit, -self.rinit)

            # self.move(self.primitive,[ x, y, 0, rx, ry, 0],t=2)

            test_count=0
            ## one sequence
            if not os.path.exists("/home/newuhe/Data9/Test/"+str(sequence_count)):
                os.mkdir("/home/newuhe/Data9/Test/"+str(sequence_count))
           
            for i in range(10):

                x = random.uniform(-self.xinterval*self.xhalf, self.xinterval*self.xhalf)
                y = random.uniform(-self.yinterval*self.yhalf, self.yinterval*self.yhalf)
                rx = random.uniform(self.rinit, -self.rinit)
                ry = random.uniform(self.rinit, -self.rinit)

                self.move(self.primitive,[ x, y, 0, rx, ry, 0])
                ## get force torque
                self.move_down(v=self.v)
                fz = 0
                z=0.1
                while( abs(fz) < self.force and z>0 ):
                    pose = self.secmon.get_cartesian_info(False)
                    z = pose["Z"]
                    f.getForce()
                    fx, fy, fz, px, py, pz = f.parseData()
                self.stop()

                #pose = self.getTcpPose()
                #pose = self.secmon.get_cartesian_info(False)
                pose = [pose["X"], pose["Y"], pose["Z"], pose["Rx"], pose["Ry"], pose["Rz"]]
                
                ## take picture
                self.q1.put(1)
                pic, depth = self.q2.get(True)
                cv2.imwrite(self.test_dir + str(sequence_count) + '/' + str(test_count) +'.jpg',pic)
                cv2.imwrite(self.test_dir + str(sequence_count) + '/' + str(test_count) +'_depth.jpg',depth)
                np.savez(self.test_dir + str(sequence_count) + "/" + str(test_count) + ".npz", force = [fx,fy,fz,px,py,pz],pos = pose, pic = depth,count = [self.count,self.pcount])
                self.test_log['test_log'].info('x = %0.4f,y = %0.4f,z = %0.4f,rx = %0.4f,ry = %0.4f,rz = %0.4f,fx = %0.4f, fy = %0.4f,fz = %0.4f,px = %0.4f,py = %0.4f,pz = %0.4f,sequencecount=%d, count = %d' % (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],fx,fy,fz,px,py,pz,sequence_count, test_count))
                test_count += 1

            sequence_count += 1
                
                # choice = random.randint(0,1)
                # ## translate
                # if choice == 0:
                #     flag=0
                #     while(flag==0):
                #         direction = random.randint(0,3)
                #         x=self.pose[0]
                #         y=self.pose[1]
                #         if direction==0:
                #             x -= self.xinterval
                #         elif direction==1:
                #             x += self.xinterval
                #         elif direction==2:
                #             y -= self.yinterval
                #         elif direction==3:
                #             y += self.yinterval

                #         if x<self.xsequence[0] or x>self.xsequence[8] or y<self.ysequence[0] or y>self.ysequence[8]:
                #             flag=0
                #         else:
                #             flag=1
                    
                #     self.pose[0] = x
                #     self.pose[1] = y
                #     self.pose[2] = self.z
                #     self.movel(self.pose,t=1)
                # ## rotate
                # else:
                #     flag=0
                #     while(flag==0):
                #         direction = random.randint(0,3)
                #         rx=self.pose[3]
                #         ry=self.pose[4]
                #         if direction==0:
                #             rx -= self.interval
                #         elif direction==1:
                #             rx += self.interval
                #         elif direction==2:
                #             ry -= self.interval
                #         elif direction==3:
                #             ry += self.interval

                #         if rx<self.sequence[0] or rx>self.sequence[8] or ry<self.sequence[0] or ry>self.sequence[8]:
                #             flag=0
                #         else:
                #             flag=1
                    
                #     self.pose[2] = self.z
                #     self.pose[3] = rx
                #     self.pose[4] = ry
                #     self.movel(self.pose,t=1)

                # print(self.pose)
                # ## get force torque
                # self.pose[2] = 0
                # self.movev(self.pose,v=self.v)
                # fz = 0
                # while( abs(fz) < self.force ):
                #     f.getForce()
                #     fx, fy, fz, px, py, pz = f.parseData()
                # self.stop()
                # time.sleep(0.5)
                # f.getForce()
                # fx, fy, fz, px, py, pz = f.parseData()
                # np.savez("/home/newuhe/Data/Test/"+str(sequence_count) + "/"+str(test_count) + ".npz", force = [fx,fy,fz,px,py,pz],pos = self.pose)
                
                # ## take picture
                # q1.put(1)
                # pic, depth = q2.get(True)
                # cv2.imwrite('/home/newuhe/Data/Test/'+str(sequence_count)+"/" + str(test_count) + '.jpg',pic)
                # cv2.imwrite('/home/newuhe/Data/Test/'+str(sequence_count)+"/" + str(test_count) + '_depth.jpg',depth)

            #     self.test_log['test_log'].info('x = %0.4f,y = %0.4f,z = %0.4f,rx = %0.4f,ry = %0.4f,rz = %0.4f,fx = %0.4f, fy = %0.4f,fz = %0.4f,px = %0.4f,py = %0.4f,pz = %0.4f, count = %d' % (pose[0],pose[1],pose[2],pose[3],pose[4],pose[5],fx,fy,fz,px,py,pz, test_count))
            #     test_count += 1

            # sequence_count+=1

    def reset(self):
        flag = input("Are you sure to reset? y or n\n")
        self.header = ""
        self.program = ""
        if flag=='y':
            self.endpose = [-0.64153, -0.04586, -0.02667, 0, 0, 0]
            self.joint = []
            self.s.send(("movel(p" + str(self.endpose) + ", t=10)" + "\n").encode())
            self.x = 0
            self.y = 0
            time.sleep(4)
            print('\nrobot reset')

    def getState(self):
        data = self.s.recv(4096)
        pose = self.parseData(data)
        return pose

    def parseData(self, data, verbose=True, robot=None):
        ## get total length of useful data
        totalLen, robotState = struct.unpack(">IB",data[:5])# unpack returns a tuple
        print('totalLen: ',totalLen,' Robot State: ',robotState)

        # assert robotState == 16, robotState
        # if robotState == 20:
        #     tmp = self._get_data(pdata, "!iB Qbb", ("size", "type", "timestamp", "source", "robotMessageType"))
        #     if tmp["robotMessageType"] == 3:
        #         allData["VersionMessage"] = self._get_data(pdata, "!iBQbb bAbBBiAb", ("size", "type", "timestamp", "source", "robotMessageType", "projectNameSize", "projectName", "majorVersion", "minorVersion", "svnRevision", "buildDate"))
        #     elif tmp["robotMessageType"] == 6:
        #         allData["robotCommMessage"] = self._get_data(pdata, "!iBQbb iiAc", ("size", "type", "timestamp", "source", "robotMessageType", "code", "argument", "messageText"))
        #     elif tmp["robotMessageType"] == 1:
        #         allData["labelMessage"] = self._get_data(pdata, "!iBQbb iAc", ("size", "type", "timestamp", "source", "robotMessageType", "id", "messageText"))
        #     elif tmp["robotMessageType"] == 2:
        #         allData["popupMessage"] = self._get_data(pdata, "!iBQbb ??BAcAc", ("size", "type", "timestamp", "source", "robotMessageType", "warning", "error", "titleSize", "messageTitle", "messageText"))
        #     elif tmp["robotMessageType"] == 0:
        #         allData["messageText"] = self._get_data(pdata, "!iBQbb Ac", ("size", "type", "timestamp", "source", "robotMessageType", "messageText"))
        #     elif tmp["robotMessageType"] == 8:
        #         allData["varMessage"] = self._get_data(pdata, "!iBQbb iiBAcAc", ("size", "type", "timestamp", "source", "robotMessageType", "code", "argument", "titleSize", "messageTitle", "messageText"))
        #     elif tmp["robotMessageType"] == 7:
        #         allData["keyMessage"] = self._get_data(pdata, "!iBQbb iiBAcAc", ("size", "type", "timestamp", "source", "robotMessageType", "code", "argument", "titleSize", "messageTitle", "messageText"))
        #     elif tmp["robotMessageType"] == 5:
        #         allData["keyMessage"] = self._get_data(pdata, "!iBQbb iiAc", ("size", "type", "timestamp", "source", "robotMessageType", "code", "argument", "messageText"))
        #     else:
        #         self.logger.debug("Message type parser not implemented %s", tmp)

        if len(data) < totalLen:
            return None
        ret = data[totalLen:]
        data = data[5:totalLen]
        ## parse sub package
        while len(data) > 0:
            subLen, packageType = struct.unpack(">IB", data[:5])
            if len(data) < subLen:
                return None
            ## print packageType, subLen
            #print('subLen: ',subLen,' packageType: ',packageType)
            if packageType == 0:
                # Robot Mode Data
                assert subLen == 46, subLen
                timestamp, connected, enabled, powerOn, emergencyStopped, protectiveStopped, programRunning, programPaused, \
                    robotMode, controlMode, targetSpeedFraction, speedScaling, targetSpeedFractionLimit \
                    = struct.unpack( ">QBBBBBBBBBddd", data[5:subLen] )
                if robot:
                    robot.timestamp = timestamp
                # if verbose:
                #     print('timestep: ',timestamp,' targetSpeedFraction: ', targetSpeedFraction,' speedScaling: ',speedScaling)
            elif packageType == 1:
                # Joint Data
                assert subLen == 251, subLen
                sumSpeed = 0
                joint = []
                for i in range(6):
                    position, target, speed, current, voltage, temperature, obsolete, mode = \
                            struct.unpack(">dddffffB", data[5+i*41:5+(i+1)*41])
                    # print i,speed
                    sumSpeed += abs(speed)
                    joint.append(position)
                    # 253 running mode
                # if verbose:
                #     print("sumSpeed", sumSpeed)
                #     print('jointPosition: ', joint)
                if robot:
                    robot.moving = (sumSpeed > 0.000111)
            elif packageType == 2:
                # Tool Data
                assert subLen == 37, subLen
                # if verbose:
                #     print("Tool", struct.unpack(">bbddfBffB", data[5:subLen] ))
            elif packageType == 3:
                # Masterboard Data
                assert subLen == 74, subLen
                # if verbose:
                #     print("Masterboard", [hex(x) for x in struct.unpack(">II", data[5:5+8] )])
                if robot:
                    robot.inputs, robot.outputs = struct.unpack(">II", data[5:5+8])
            elif packageType == 4:
                # Cartesian Info
                assert subLen == 101, subLen
                x,y,z, rx,ry,rz, offsetx, offsety, offsetz, offsetRX, offsetRY, offsetRZ = struct.unpack( ">dddddddddddd", data[5:subLen] )
                pose = [x, y, z, rx, ry, rz]
                if robot:
                    robot.pose = (x,y,z, rx,ry,rz)
                if verbose:
                    print('TcpPose: '+"(%.5f, %.5f, %.5f,    %.5f, %.5f, %.5f)" % (x,y,z, rx,ry,rz))
                    #print('TcpOffset: '+"(%.3f, %.3f, %.3f,    %.3f, %.3f, %.3f)" % (offsetx, offsety, offsetz, offsetRX, offsetRY, offsetRZ))
            data = data[subLen:]
        if verbose:
            print("------------")
        return pose



if __name__ == '__main__':

    ## init
    q1 = Queue()
    q2 = Queue()
    cam = Process(target=get_cam, args=(q1, q2))
    cam.start()
    f=ForceTorque(1)
    u = UR5(q1,q2)

    ## pre picture
    q1.put(1)
    pic, depth = q2.get(True)
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', depth)
    cv2.waitKey(1)
    
    u.traverse()
    u.testData()
    u.disconnect()
