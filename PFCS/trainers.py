from __future__ import print_function, absolute_import
import time
from .utils.meters import AverageMeter
from PFCS.models.triplet import  SoftTripletLoss_vallia
import torch


class Trainer_teacher(object):
    def __init__(self, encoder,memory=None):
        super(Trainer_teacher, self).__init__()
        self.encoder = encoder
        self.memory = memory
        self.softTripletLoss = SoftTripletLoss_vallia(margin=0.0).cuda()


    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400,lamb=None,lamb2=None):
        self.encoder.train()
        self.lamb = lamb
        self.lamb2 = lamb2
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            # load data
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            # process inputs
            inputs, labels, indexes = self._parse_data(inputs)
            f_out, f_out_up, f_out_down,f_out_mix = self._forward(inputs)
            loss= self.memory(f_out, f_out_up, f_out_down, f_out_mix,labels, epoch)
            loss_softTriplet = self.softTripletLoss(f_out, f_out, labels)
            loss_softTriplet_up = self.softTripletLoss(f_out_up, f_out_up, labels)
            loss_softTriplet_down = self.softTripletLoss(f_out_down, f_out_down, labels)
            loss_softTriplet = (1-self.lamb2) * loss_softTriplet + self.lamb2 * loss_softTriplet_up + self.lamb2 * loss_softTriplet_down
            loss = self.lamb * loss +  (1 - self.lamb) * loss_softTriplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())
            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

class Trainer(object):
    def __init__(self, encoder, encoder_teacher, memory=None):
        super(Trainer, self).__init__()
        self.encoder = encoder
        self.encoder_teacher = encoder_teacher
        self.memory = memory
     

    def train(self, epoch, data_loader, optimizer, print_freq=10, train_iters=400,lamb=None,lamb2=None):
        self.encoder.train()
        self.encoder_teacher.train()
        self.lamb = lamb
        self.lamb2 = lamb2
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.softTripletLoss = SoftTripletLoss_vallia(margin=0.0).cuda()
        losses = AverageMeter()

        end = time.time()
        for i in range(train_iters):
            inputs = data_loader.next()
            data_time.update(time.time() - end)

            inputs, labels, indexes = self._parse_data(inputs)

            f_out, f_out_up, f_out_down,_ = self._forward(inputs)
            with torch.no_grad():
                f_out_teacher, f_out_up_teacher, f_out_down_teacher,_ = self.encoder_teacher(inputs)

            loss = self.memory(f_out, f_out_up, f_out_down, f_out_teacher, f_out_up_teacher, f_out_down_teacher, labels, epoch)
            loss_softTriplet = self.softTripletLoss(f_out, f_out, labels)
            loss_softTriplet_up = self.softTripletLoss(f_out_up, f_out_up, labels)
            loss_softTriplet_down = self.softTripletLoss(f_out_down, f_out_down, labels)
            loss_softTriplet = (1 - self.lamb2) * loss_softTriplet + self.lamb2 * loss_softTriplet_up + self.lamb2 * loss_softTriplet_down
            loss = self.lamb * loss + (1 - self.lamb) * loss_softTriplet

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses.update(loss.item())

            # print log
            batch_time.update(time.time() - end)
            end = time.time()

            if (i + 1) % print_freq == 0:
                print('Epoch: [{}][{}/{}]\t'
                      'Time {:.3f} ({:.3f})\t'
                      'Data {:.3f} ({:.3f})\t'
                      'Loss {:.3f} ({:.3f})'
                      .format(epoch, i + 1, len(data_loader),
                              batch_time.val, batch_time.avg,
                              data_time.val, data_time.avg,
                              losses.val, losses.avg))

    def _parse_data(self, inputs):
        imgs, _, pids, _, indexes = inputs
        return imgs.cuda(), pids.cuda(), indexes.cuda()

    def _forward(self, inputs):
        return self.encoder(inputs)

