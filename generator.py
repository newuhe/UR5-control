import numpy as np
import cv2
import copy
import random
from torch.autograd import Variable
from random import shuffle
import torch
from util import norm


class dataGenerator():

    def __init__(self):
        self.path = "/home/newuhe/Data/"
        self.test_path = "/home/newuhe/Data1/"
        self.batch_size = 10
        self.reference = cv2.imread("/home/newuhe/Data/0/0.jpg")
        self.reference_test = cv2.imread("/home/newuhe/Data1/0/0.jpg")

    def train_data(self):
        p = []
        f = []
        a = []
        t = []
        count = random.randint(0,80)
        pcount = random.randint(0,80)
        last_pos = np.load(self.path + str(count) + "/" + str(pcount) + ".npz")["pos"]

        for i in range(self.batch_size): 
            ## take action
            try_count = count
            try_pcount = pcount


            choice = random.randint(0,1)
            ## translate
            if choice == 0:
                if int(try_count/9)==0:
                    if try_count==0:
                        act = random.randint(0,1)
                        if act==0:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=17
                            try_pcount = 80 - try_pcount
                    elif try_count==8:
                        act = random.randint(0,1)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==2:
                            try_count=17-try_count
                            try_pcount = 80 - try_pcount
                elif int(try_count/9)==8:
                    if try_count==72:
                        act = random.randint(0,1)
                        if act==0:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                    elif try_count==80:
                        act = random.randint(0,1)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count-=17
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==2:
                            try_count=143-try_count
                            try_pcount = 80 - try_pcount
                elif int(try_count/9)%2==0:
                    if try_count==int(try_count/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 17
                            try_pcount = 80 - try_pcount
                    elif try_count==int(try_count/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 17
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count = ((int(try_count/9)*2+2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
                        elif act == 3:
                            try_count = ((int(try_count/9)*2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
                else:
                    if try_count==int(try_count/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 17
                            try_pcount = 80 - try_pcount
                    elif try_count==int(try_count/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 17
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count = ((int(try_count/9)*2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
                        elif act == 3:
                            try_count = ((int(try_count/9)*2+2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
            ## rotate
            else:
                if int(try_pcount/9)==0:
                    if try_pcount==0:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount+=1
                        if act==1:
                            try_pcount+=17
                    elif try_pcount==8:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount+=1
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount+=1
                        if act==2:
                            try_pcount=17-try_pcount
                elif int(try_pcount/9)==8:
                    if try_pcount==72:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount+=1
                        if act==1:
                            try_pcount+=1
                    elif try_pcount==80:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount-=17
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount+=1
                        if act==2:
                            try_pcount=143-try_pcount
                elif int(try_pcount/9)%2==0:
                    if try_pcount==int(try_pcount/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount += 17
                    elif try_pcount==int(try_pcount/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 17
                        elif act == 1:
                            try_pcount -= 1
                        elif act == 2:
                            try_pcount += 1
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount = ((int(try_pcount/9)*2+2)*9-1)-try_pcount
                        elif act == 3:
                            try_pcount = ((int(try_pcount/9)*2)*9-1)-try_pcount
                else:
                    if try_pcount==int(try_pcount/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount += 17
                    elif try_pcount==int(try_pcount/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 17
                        elif act == 1:
                            try_pcount -= 1
                        elif act == 2:
                            try_pcount += 1
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount = ((int(try_pcount/9)*2)*9-1)-try_pcount
                        elif act == 3:
                            try_pcount = ((int(try_pcount/9)*2+2)*9-1)-try_pcount                

            count = try_count
            pcount = try_pcount   
            #print(self.path + str(count) + "/" + str(pcount) + ".jpg") 
            pic =  norm(cv2.imread(self.path + str(count) + "/" + str(pcount) + ".jpg") - self.reference)
            pic = cv2.resize(pic, (160,120))
            force = np.array([np.load(self.path + str(count) + "/" + str(pcount) + ".npz")["force"]])
            pos = np.array([np.load(self.path + str(count) + "/" + str(pcount) + ".npz")["pos"]])
            action = pos - last_pos
            last_pos = copy.deepcopy(pos)
            target = pos
            p.append(copy.deepcopy(pic))
            f.append(copy.deepcopy(force))
            a.append(copy.deepcopy(action))
            t.append(copy.deepcopy(target))

        return p,f,a,t

    def test_data(self):
        p = []
        f = []
        a = []
        t = []
        count = random.randint(0,80)
        pcount = random.randint(0,80)
        last_pos = np.load(self.path + str(count) + "/" + str(pcount) + ".npz")["pos"]

        for i in range(self.batch_size): 
            ## take action
            try_count = count
            try_pcount = pcount


            choice = random.randint(0,1)
            ## translate
            if choice == 0:
                if int(try_count/9)==0:
                    if try_count==0:
                        act = random.randint(0,1)
                        if act==0:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=17
                            try_pcount = 80 - try_pcount
                    elif try_count==8:
                        act = random.randint(0,1)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==2:
                            try_count=17-try_count
                            try_pcount = 80 - try_pcount
                elif int(try_count/9)==8:
                    if try_count==72:
                        act = random.randint(0,1)
                        if act==0:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                    elif try_count==80:
                        act = random.randint(0,1)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count-=17
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_count-=1
                            try_pcount = 80 - try_pcount
                        if act==1:
                            try_count+=1
                            try_pcount = 80 - try_pcount
                        if act==2:
                            try_count=143-try_count
                            try_pcount = 80 - try_pcount
                elif int(try_count/9)%2==0:
                    if try_count==int(try_count/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 17
                            try_pcount = 80 - try_pcount
                    elif try_count==int(try_count/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 17
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count = ((int(try_count/9)*2+2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
                        elif act == 3:
                            try_count = ((int(try_count/9)*2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
                else:
                    if try_count==int(try_count/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 17
                            try_pcount = 80 - try_pcount
                    elif try_count==int(try_count/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_count -= 17
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_count -= 1
                            try_pcount = 80 - try_pcount
                        elif act == 1:
                            try_count += 1
                            try_pcount = 80 - try_pcount
                        elif act == 2:
                            try_count = ((int(try_count/9)*2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
                        elif act == 3:
                            try_count = ((int(try_count/9)*2+2)*9-1)-try_count
                            try_pcount = 80 - try_pcount
            ## rotate
            else:
                if int(try_pcount/9)==0:
                    if try_pcount==0:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount+=1
                        if act==1:
                            try_pcount+=17
                    elif try_pcount==8:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount+=1
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount+=1
                        if act==2:
                            try_pcount=17-try_pcount
                elif int(try_pcount/9)==8:
                    if try_pcount==72:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount+=1
                        if act==1:
                            try_pcount+=1
                    elif try_pcount==80:
                        act = random.randint(0,1)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount-=17
                    else:
                        act = random.randint(0,2)
                        if act==0:
                            try_pcount-=1
                        if act==1:
                            try_pcount+=1
                        if act==2:
                            try_pcount=143-try_pcount
                elif int(try_pcount/9)%2==0:
                    if try_pcount==int(try_pcount/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount += 17
                    elif try_pcount==int(try_pcount/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 17
                        elif act == 1:
                            try_pcount -= 1
                        elif act == 2:
                            try_pcount += 1
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount = ((int(try_pcount/9)*2+2)*9-1)-try_pcount
                        elif act == 3:
                            try_pcount = ((int(try_pcount/9)*2)*9-1)-try_pcount
                else:
                    if try_pcount==int(try_pcount/9)*9:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount += 17
                    elif try_pcount==int(try_pcount/9)*9+8:
                        act = random.randint(0, 2)
                        if act == 0:
                            try_pcount -= 17
                        elif act == 1:
                            try_pcount -= 1
                        elif act == 2:
                            try_pcount += 1
                    else:
                        act = random.randint(0, 3)
                        if act == 0:
                            try_pcount -= 1
                        elif act == 1:
                            try_pcount += 1
                        elif act == 2:
                            try_pcount = ((int(try_pcount/9)*2)*9-1)-try_pcount
                        elif act == 3:
                            try_pcount = ((int(try_pcount/9)*2+2)*9-1)-try_pcount                

            count = try_count
            pcount = try_pcount   
            #print(self.path + str(count) + "/" + str(pcount) + ".jpg") 
            pic =  norm(cv2.imread(self.test_path + str(count) + "/" + str(pcount) + ".jpg") - self.reference_test)
            pic = cv2.resize(pic, (160,120))
            force = np.array([np.load(self.test_path + str(count) + "/" + str(pcount) + ".npz")["force"]])
            pos = np.array([np.load(self.test_path + str(count) + "/" + str(pcount) + ".npz")["pos"]])
            action = pos - last_pos
            last_pos = copy.deepcopy(pos)
            target = pos
            p.append(copy.deepcopy(pic))
            f.append(copy.deepcopy(force))
            a.append(copy.deepcopy(action))
            t.append(copy.deepcopy(target))

        return p,f,a,t