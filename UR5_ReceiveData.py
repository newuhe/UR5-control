import socket 
import numpy as np
import struct
import os
import sys
import time

class UR5():

    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.initpose = [-0.64153, -0.04586, -0.02667, 0, 0, 0]
        #self.joint = [-1.41,-86.25,152.52,-213.32,-95.8,8.37]
        self.joint = [-0.0246,-1.505,2.66,-3.72,-1.67,0.26]
        
        #print "endpose:", self.endpose, "  radius:", self.radius, "  center:", self.center
        
        self.connect()
        #self.reset()

    def connect(self):
        HOST = "192.168.1.147"
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

    def getActualPose(self):
        self.s.send(("get_actual_tcp_pose()" + "\n").encode())
        data=self.s.recv(4096)
        l,state=struct.unpack(">IB",data[:5])
        if len(data)>l:
            print('totalLen: ',l,' Robot State: ',state)
        # timestamp,  = struct.unpack( ">Qcc", data[5:l] )
        x,y,z,rx,ry,rz=struct.unpack( ">dddddd", data[56-48:56] )
        print('actualPose: '+"(%.3f, %.3f, %.3f,    %.3f, %.3f, %.3f)" % (x,y,z, rx,ry,rz))
        print('\n')

    def getState(self):
        data = self.s.recv(4096)
        self.parseData(data)

    def parseData(self, data, verbose=True, robot=None):
        # get total length of useful data
        totalLen, robotState = struct.unpack(">IB",data[:5])# unpack returns a tuple
        print('totalLen: ',totalLen,' Robot State: ',robotState)
        assert robotState == 16, robotState
        if len(data) < totalLen:
            return None
        ret = data[totalLen:]
        data = data[5:totalLen]
        # parse sub package
        while len(data) > 0:
            subLen, packageType = struct.unpack(">IB", data[:5])
            if len(data) < subLen:
                return None
            # print packageType, subLen
            print('subLen: ',subLen,' packageType: ',packageType)
            if packageType == 0:
                # Robot Mode Data
                assert subLen == 46, subLen
                timestamp, connected, enabled, powerOn, emergencyStopped, protectiveStopped, programRunning, programPaused, \
                    robotMode, controlMode, targetSpeedFraction, speedScaling, targetSpeedFractionLimit \
                    = struct.unpack( ">QBBBBBBBBBddd", data[5:subLen] )
                if robot:
                    robot.timestamp = timestamp
                if verbose:
                    print('timestep: ',timestamp,' targetSpeedFraction: ', targetSpeedFraction,' speedScaling: ',speedScaling)
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
                if verbose:
                    print("sumSpeed", sumSpeed)
                    print('jointPosition: ', joint)
                if robot:
                    robot.moving = (sumSpeed > 0.000111)
            elif packageType == 2:
                # Tool Data
                assert subLen == 37, subLen
                if verbose:
                    print("Tool", struct.unpack(">bbddfBffB", data[5:subLen] ))
            elif packageType == 3:
                # Masterboard Data
                assert subLen == 74, subLen
                if verbose:
                    print("Masterboard", [hex(x) for x in struct.unpack(">II", data[5:5+8] )])
                if robot:
                    robot.inputs, robot.outputs = struct.unpack(">II", data[5:5+8])
            elif packageType == 4:
                # Cartesian Info
                assert subLen == 101, subLen
                x,y,z, rx,ry,rz, offsetx, offsety, offsetz, offsetRX, offsetRY, offsetRZ = struct.unpack( ">dddddddddddd", data[5:subLen] )
                if robot:
                    robot.pose = (x,y,z, rx,ry,rz)
                if verbose:
                    print('TcpPose: '+"(%.3f, %.3f, %.3f,    %.3f, %.3f, %.3f)" % (x,y,z, rx,ry,rz))
                    print('TcpOffset: '+"(%.3f, %.3f, %.3f,    %.3f, %.3f, %.3f)" % (offsetx, offsety, offsetz, offsetRX, offsetRY, offsetRZ))
            data = data[subLen:]
        if verbose:
            print("------------")
        return ret



    def getPos(self):
        self.s.send(("get_actual_tcp_pose()" + "\n").encode())
        data=self.s.receive(4096)

    def getJoint(self):
        self.s.send(("get_actual_joint_positions()" + "\n").encode())
        data=self.s.receive()

    def reset(self):
        self.endpose = [-0.64153, -0.04586, -0.02667, 0, 0, 0]
        self.joint = []
        self.s.send(("movel(p" + str(self.endpose) + ", t=3)" + "\n").encode())
        self.x = 0
        self.y = 0
        time.sleep(4)
        print('\nrobot reset')


u = UR5()
u.getActualPose()
u.getState()
u.disconnect()


# s.send(("get_actual_tcp_pose()" + "\n").encode())
# data=s.recv(4096)
# l,state=struct.unpack(">IB",data[:5])
# print(l,state)