import torch
from torch.autograd import Variable
import numpy as np
import cv2
import torch.optim as optim
from generator import dataGenerator
from lstm import locNet
from util import setup_logger    
import logging  
import torch.nn as nn
import argparse
import sys

parser = argparse.ArgumentParser(description='A3C')
parser.add_argument(
    '--gpu',
    default=True,
    metavar='L',
    help='train model')
parser.add_argument(
    '--load',
    default=False,
    metavar='G',
    help='train model')
parser.add_argument(
    '--load-model-dir',
    default='trained_model/',
    metavar='LMD',
    help='folder to load trained models from')
parser.add_argument(
    '--epoch',
    type=int,
    default=0,
    metavar='W',
    help='the num of epoch')

def train():
    args = parser.parse_args()
    print(args.gpu)
    log = {}
    setup_logger('train_log', r'train_log')
    log['train_log'] = logging.getLogger(
        'train_log')
    save_model_dir = "/home/newuhe/UR5_control/trained_model/"
    n_epochs = 1000000
    save_time = 500
    gpu = False

    generator = dataGenerator()
    model = locNet()
    if args.load is True:
        saved_state = torch.load(args.load_model_dir + str(args.epoch) + '.dat')
        model.load_state_dict(saved_state) 
        print("load succssfully")
    model.train()

    if args.gpu is True:
        print("gpu used")
        model = model.cuda()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    losses = np.zeros(n_epochs) 

    for epoch in range(args.epoch ,n_epochs):
        ## one sequence
        p, f, a, t = generator.train_data()
        pic = Variable(torch.from_numpy(np.array(p)).float())
        pic = pic.permute( 0, 3, 1, 2) # (10,3,120,160)
        force = Variable(torch.from_numpy(np.array(f)).float()) # (10,1,6)
        action = Variable(torch.from_numpy(np.array(a)).float())
        target = Variable(torch.from_numpy(np.array(t)).float())
        cx = Variable(torch.zeros(1, 1, 72))
        hx = Variable(torch.zeros(1, 1, 72))
        if args.gpu is True:
            hx = hx.cuda()
            cx = cx.cuda()
            pic = pic.cuda()
            force = force.cuda()
            action = action.cuda()
            target = target.cuda()
        model.zero_grad()

        # for pic, force, action, target in train_data:
        #     pic = Variable(torch.from_numpy(np.array([pic.tolist(),])).float())
        #     pic = pic.permute(0,3,1,2)
        #     force = Variable(torch.from_numpy(np.array([force,])).float())
        #     action = Variable(torch.from_numpy(np.array([action,])).float())
        #     target = Variable(torch.from_numpy(np.array([target,])).float())

        pos, (hx, cx) = model( (pic, force, action,(hx, cx)) )
        loss = criterion(pos, target)
        #loss = criterion(pos + action, target)
        # loss.backward(retain_graph=True)
        loss.backward()
        optimizer.step()
        if epoch % 500 == 0:
            log['train_log'].info('epoch = %d,train_loss = %0.5f' % (epoch, loss))

        if epoch % save_time == 0:
            ## save model and eval
            state_to_save = model.state_dict()
            torch.save(state_to_save, '{0}{1}.dat'.format(save_model_dir, epoch))
            print("get: ", pos)
            print("target: ", target)
            
            pt, ft, at, tt = generator.test_data()
            pict = Variable(torch.from_numpy(np.array(pt)).float())
            pict = pict.permute( 0, 3, 1, 2) # (10,3,120,160)
            forcet = Variable(torch.from_numpy(np.array(ft)).float()) # (10,1,6)
            actiont = Variable(torch.from_numpy(np.array(at)).float())
            targett = Variable(torch.from_numpy(np.array(tt)).float())
            cxt = Variable(torch.zeros(1, 1, 72))
            hxt = Variable(torch.zeros(1, 1, 72))
            if args.gpu is True:
                hxt = hxt.cuda()
                cxt = cxt.cuda()
                pict = pict.cuda()
                forcet = forcet.cuda()
                actiont = actiont.cuda()
                targett = targett.cuda()
            post, (hxt, cxt) = model( (pict, forcet, actiont,(hxt, cxt)) )
            losst = criterion(post, targett)

            # for pic, force, action, target in test_data:
            #     pic = Variable(torch.from_numpy(np.array([pic.tolist(),])).float())
            #     pic = pic.permute(0,3,1,2)
            #     force = Variable(torch.from_numpy(np.array([force,])).float())
            #     action = Variable(torch.from_numpy(np.array([action,])).float())
            #     target = Variable(torch.from_numpy(np.array([target,])).float())
            #     if args.gpu==True:
            #         pic = pic.cuda()
            #         force = force.cuda()
            #         ation = action.cuda()
            #         target = target.cuda()
            #     pos, (hx, cx) = model( (pic, force, action,(hx, cx)) )
                
            #     loss = criterion(pos, target)
            #     test_loss.append(loss.data[0])
            log['train_log'].info('epoch = %d,test_loss = %0.5f' % (epoch, losst))



if __name__ == '__main__':
    train()


