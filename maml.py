import torch
from torch import optim
import os
from copy import deepcopy
import torch.nn.functional as F
from utils import TrainingArgs
import logging
from nn import Net
from torch import nn
from vat import VATLoss
import numpy as np

logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed_value = 123

config = TrainingArgs()


class MAML(nn.Module):
    """
    MAML meta learner
    """

    def __init__(self, args):
        super(MAML, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.update_lr = args.update_lr
        self.meta_lr = args.meta_lr
        self.update_step = args.update_step
        self.update_step_test = args.update_step_test
        self.bert_model = args.bert_model.to(self.device)
        self.alpha1 = args.alpha1
        self.alpha2 = args.alpha2
        self.alpha3 = args.alpha3
        self.xi = args.xi
        self.eps = args.eps

        self.net = Net(768, args.neurons).to(self.device)
        self.meta_optim = optim.Adam(self.net.parameters(), lr=self.meta_lr)

    def forward(self, x_spt, y_spt, x_qry, y_qry, vat_flag1=False, vat_flag2=False, vat_flag3=False, qry_num=5):
        task_num = 1
        querysz = qry_num

        losses_q = [0 for _ in range(self.update_step + 1)]  # losses_q[i] is the loss on step i
        corrects = [0 for _ in range(self.update_step + 1)]

        input_ids, attention_mask, _ = (elt.to(self.device) for elt in x_spt)

        with torch.no_grad():
            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            q_input_ids, q_attention_mask, _ = (elt.to(self.device) for elt in x_qry)

            q_input_ids, q_attention_mask = q_input_ids[:qry_num], q_attention_mask[:qry_num]

            x_q = self.bert_model(input_ids=q_input_ids, attention_mask=q_attention_mask)[1]

            vat_input_ids, vat_attention_mask = q_input_ids[:], q_attention_mask[:]
            x_vat = self.bert_model(input_ids=vat_input_ids, attention_mask=vat_attention_mask)[1]

            y_qry, y_vat = y_qry[:qry_num], y_qry[:]
            y_spt = y_spt.to(self.device)
            y_qry = y_qry.to(self.device)
            y_vat = y_vat.to(self.device)

        for i in range(task_num):

            if vat_flag1:
                vat_loss1 = VATLoss(xi=10.0, eps=1.0, ip=1)
                lds1 = vat_loss1(self.net, x)

            # 1. run the i-th task and compute loss for k=0
            logits = self.net(x, bn_training=True)

            loss = F.cross_entropy(logits, y_spt)

            if vat_flag1:
                loss += lds1 * self.alpha1

            grad = torch.autograd.grad(loss, self.net.parameters(), allow_unused=True)

            fast_weights = list(map(lambda p: (p[1] - self.update_lr * p[0]) if p[0] is not None else p[1],
                                    zip(grad, self.net.parameters())))

            # this is the loss and accuracy before first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_q, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry)
                losses_q[0] += loss_q

                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[0] = corrects[0] + correct

            for j, p in enumerate(self.net.parameters()):
                p.data = fast_weights[j]

            # this is the loss and accuracy after the first update
            with torch.no_grad():
                # [setsz, nway]
                logits_q = self.net(x_q, bn_training=True)
                loss_q = F.cross_entropy(logits_q, y_qry)
                losses_q[1] += loss_q
                # [setsz]
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()
                corrects[1] = corrects[1] + correct

            for k in range(1, self.update_step):
                if vat_flag2:
                    vat_loss2 = VATLoss(xi=10.0, eps=1.0, ip=1)
                    lds2 = vat_loss2(self.net, x)

                # 1. run the i-th task and compute loss for k=1~K-1
                logits = self.net(x, bn_training=True)
                loss = F.cross_entropy(logits, y_spt)

                if vat_flag2:
                    loss += lds2 * self.alpha2
                # 2. compute grad on theta_pi
                grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
                # 3. theta_pi = theta_pi - train_lr * grad
                fast_weights = list(map(lambda p: (p[1] - self.update_lr * p[0]) if p[0] is not None else p[1],
                                        zip(grad, fast_weights)))

                for j, p in enumerate(self.net.parameters()):
                    p.data = fast_weights[j]

                # VAT LOSS 3
                if vat_flag3:
                    vat_loss3 = VATLoss(xi=self.xi, eps=self.eps, ip=1)
                    lds3 = vat_loss3(self.net, x_vat)

                logits_q = self.net(x_q, bn_training=True)
                # loss_q will be overwritten and just keep the loss_q on last update step.
                loss_q = F.cross_entropy(logits_q, y_qry)
                if vat_flag3:
                    loss_q += lds3 * self.alpha3
                losses_q[k + 1] += loss_q

                with torch.no_grad():
                    pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                    correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                    corrects[k + 1] = corrects[k + 1] + correct

        # end of all tasks
        # sum over all losses on query set across all tasks
        loss_q = losses_q[-1] / task_num

        # optimize theta parameters
        self.meta_optim.zero_grad()
        loss_q.backward()
        self.meta_optim.step()

        accs = np.array(corrects) / (querysz * task_num)

        return accs

    def finetuning(self, x_spt, y_spt, x_qry, y_qry, vat_flag2=False, vat_flag3=False):
        """
        :param x_spt:   [setsz, c_, h, w]
        :param y_spt:   [setsz]
        :param x_qry:   [querysz, c_, h, w]
        :param y_qry:   [querysz]
        :return:
        """

        querysz = len(y_qry)

        corrects = [0 for _ in range(self.update_step_test + 1)]

        net = deepcopy(self.net)

        with torch.no_grad():
            input_ids, attention_mask, _ = (elt.squeeze(0).to(self.device) for elt in x_spt)
            x = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)[1]
            q_input_ids, q_attention_mask, _ = (elt.squeeze(0).to(self.device) for elt in x_qry)
            x_q = self.bert_model(input_ids=q_input_ids, attention_mask=q_attention_mask)[1]
            y_spt = y_spt.squeeze(0).to(self.device)
            y_qry = y_qry.squeeze(0).to(self.device)

        # 1. run the i-th task and compute loss for k=0
        logits = net(x)
        loss = F.cross_entropy(logits, y_spt)
        grad = torch.autograd.grad(loss, net.parameters(), allow_unused=True)
        fast_weights = list(
            map(lambda p: (p[1] - self.update_lr * p[0]) if p[0] is not None else p[1], zip(grad, net.parameters())))

        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_q, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[0] = corrects[0] + correct

        for i, p in enumerate(net.parameters()):
            p.data = fast_weights[i]

        with torch.no_grad():
            # [setsz, nway]
            logits_q = net(x_q, bn_training=True)
            # [setsz]
            pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
            # scalar
            correct = torch.eq(pred_q, y_qry).sum().item()
            corrects[1] = corrects[1] + correct

        for k in range(1, self.update_step_test):

            vat_loss2 = VATLoss(xi=self.xi, eps=self.eps, ip=1)
            lds2 = vat_loss2(net, x)

            vat_loss3 = VATLoss(xi=self.xi, eps=self.eps, ip=1)
            lds3 = vat_loss3(net, x_q)

            # 1. run the i-th task and compute loss for k=1~K-1
            logits = net(x, bn_training=True)
            loss = F.cross_entropy(logits, y_spt)

            if vat_flag3:
                loss += lds3 * self.alpha3

            if vat_flag2:
                loss += lds2 * self.alpha2

            # 2. compute grad on theta_pi
            grad = torch.autograd.grad(loss, fast_weights, allow_unused=True)
            # 3. theta_pi = theta_pi - train_lr * grad
            fast_weights = list(
                map(lambda p: (p[1] - self.update_lr * p[0]) if p[0] is not None else p[1], zip(grad, fast_weights)))

            for i, p in enumerate(net.parameters()):
                p.data = fast_weights[i]

            logits_q = net(x_q, bn_training=True)

            with torch.no_grad():
                pred_q = F.softmax(logits_q, dim=1).argmax(dim=1)
                correct = torch.eq(pred_q, y_qry).sum().item()  # convert to numpy
                corrects[k + 1] = corrects[k + 1] + correct

        del net

        accs = np.array(corrects) / querysz

        return accs, lds3
