#! -*- coding:utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import math
from collections import OrderedDict

import torch as th
import torch.nn as nn
from torch.autograd import Variable

from .container import CallbackContainer, RegularizerContainer, InitializerContainer, ConstraintContainer, MetricContainer
from .callback import History, TQDM

from tqdm import tqdm
from matplotlib import pyplot as plt
import numpy as np
import matplotlib

class ModuleTrainer(object):

    def __init__(self,  model):
        assert( isinstance(model, nn.Module) )
        self.model = model
        
    def compile(self,optimizer,
                    get_loss,
                    callbacks=[],
                    initializers=[],
                    regularizers=[],
                    constraints=[],
                    metrics=[]):
        
        self.optimizer    = optimizer
        self.get_loss     = get_loss
        
        self.cter_initializer = InitializerContainer(initializers)
        self.cter_initializer.apply(self.model)
        
        self.cter_regularizer = RegularizerContainer(regularizers)
        self.cter_regularizer.register_forward_hooks(self.model)
        
        self.cter_constraint = ConstraintContainer(constraints)
        self.cter_constraint.register_constraints(self.model)

        self.cter_metric = MetricContainer(metrics)
        
        self.history = History([metric.name for metric in metrics])
        self.cter_callback = CallbackContainer([self.history]+callbacks)


    def fit_loader(self,loader_train,
                        loader_valid=None,
                        epoch_start = 0,
                        epoch_total=100,
                        order_input=None,
                        order_target=None):
        """
        Fit a model on in-memory tensors using ModuleTrainer
        """    
        self.model.train(mode=True)
        batch_size = loader_train.batch_size
        num_batches = int(math.ceil(len(loader_train.dataset)/batch_size))
        
        with TQDM() as pbar:
            self.cter_callback.append(pbar)
            self.cter_callback.on_train_begin({ 'num_batches': num_batches,
                                                'batch_size': batch_size,
                                                'epoch_start':epoch_start,
                                                'num_epochs': epoch_total})

            for epoch in range(epoch_start, epoch_total, 1):                
                logs_epoch = {}
                
                self.cter_callback.on_epoch_begin(epoch, logs_epoch)
                
                logs_batch = {}
                self.cter_metric.reset()
                for batch_idx, (batch_input, batch_target) in enumerate(loader_train):
                    
                    self.cter_callback.on_batch_begin(batch_idx, logs_batch)
                    self.cter_regularizer.reset()
                    
                    
                    if isinstance(batch_input, dict):
                        assert(isinstance(order_input, list))
                        batch_input = [batch_input[key] for key in order_input]
                    if isinstance(batch_target, dict):
                        assert(isinstance(order_target, list))
                        batch_target = [batch_target[key] for key in order_target]

                    # ---------------------------------------------
                    self.optimizer.zero_grad()
                    batch_output = self.model(batch_input)
                    loss_getloss = self.get_loss(batch_output, batch_target)
                    loss_regularizer = self.cter_regularizer.get_value()
                    loss = loss_getloss+loss_regularizer
                    loss.backward()
                    self.optimizer.step()
                    # ---------------------------------------------
                    logs_batch['loss_all'] = loss.item()
                    logs_batch['loss_fn'] = loss_getloss.item()
                    logs_batch['loss_reg'] = loss_regularizer.item()
                    metriclog = self.cter_metric(batch_output, batch_target, '_train')
                    logs_batch.update(metriclog)
                    # -----------------------------------------------------
                    self.cter_constraint.apply_batch_constraints(batch_idx)
                    self.cter_callback.on_batch_end(batch_idx, logs_batch)
                logs_epoch.update(self.history.loss_batch)
                logs_epoch.update(metriclog)

                self.cter_metric.reset()
                if loader_valid is not None:
                    logs_valid = self.evaluate_loader(loader_valid, self.cter_metric, order_input, order_target, self.get_loss)
                    logs_epoch.update(logs_valid)

                self.cter_callback.on_epoch_end(epoch, logs_epoch)
                self.cter_constraint.apply_epoch_constraints(epoch)

        self.model.train(mode=False)
        return self.history


    def predict_loader(self, loader_test, batch_size, order_input=None, order_output=['']):
        self.model.train(mode=False)
        num_batches = int(math.ceil(len(loader_test.dataset)/batch_size))
        pbar = tqdm(range(num_batches))
        
        dict_predict = {key:[] for key in order_output}
        
        for batch_idx in range(num_batches):
            
            if loader_test.dataset.task_type=='cls' and loader_test.dataset.labels is not None:
                batch_input, _ = next(loader_test)
            elif loader_test.dataset.task_type=='seg' and loader_test.dataset.targets is not None:
                batch_input, _ = next(loader_test)
            else:
                batch_input = next(loader_test)
            
            if isinstance(batch_input, dict):
                assert(isinstance(order_input, list))
                batch_input = [batch_input[key] for key in order_input]

            batch_output = self.model(batch_input)
            if isinstance(batch_output, list):
                assert(len(order_output)==len(batch_output))
                for idx,key in enumerate(order_output):
                    dict_predict[key].append(batch_output[idx])

        return dict_predict[''] if len(order_output)==1 else dict_predict


    def evaluate_loader(self, loader_test, cter_metric, order_input=None, order_target=None, get_loss=None):
        self.model.train(mode=False)
        batch_size = loader_test.batch_size
        num_batches = int(math.ceil(len(loader_test.dataset)/batch_size))
#         pbar = tqdm(range(num_batches))

        logs_eval= {'loss_valid': []}
        cter_metric.reset()
        for batch_idx, (batch_input, batch_target) in enumerate(loader_test):
            if isinstance(batch_input, dict):
                assert(isinstance(order_input, list))
                batch_input = [batch_input[key] for key in order_input]
            if isinstance(batch_target, dict):
                assert(isinstance(order_target, list))
                batch_target = [batch_target[key] for key in order_target]
            else:
                batch_target = batch_target.cuda()
            
            self.optimizer.zero_grad()
            batch_output = self.model(batch_input)
            if get_loss is not None:
                loss_getloss = self.get_loss(batch_output, batch_target)
                logs_eval['loss_valid'].append(loss_getloss.item())
                
            logs_metric = cter_metric(batch_output, batch_target, '_valid')
        
        logs_eval['loss_valid'] = sum(logs_eval['loss_valid'])/len(logs_eval['loss_valid'])
        logs_eval.update(logs_metric)
        cter_metric.reset()
        return logs_eval

#     def summary(self, input_size):
#         def register_hook(module):
#             def hook(module, input, output):
#                 class_name = str(module.__class__).split('.')[-1].split("'")[0]
#                 module_idx = len(summary)
# 
#                 m_key = '%s-%i' % (class_name, module_idx+1)
#                 summary[m_key] = OrderedDict()
#                 summary[m_key]['input_shape'] = list(input[0].size())
#                 summary[m_key]['input_shape'][0] = -1
#                 summary[m_key]['output_shape'] = list(output.size())
#                 summary[m_key]['output_shape'][0] = -1
# 
#                 params = 0
#                 if hasattr(module, 'weight'):
#                     params += th.prod(th.LongTensor(list(module.weight.size())))
#                     if module.weight.requires_grad:
#                         summary[m_key]['trainable'] = True
#                     else:
#                         summary[m_key]['trainable'] = False
#                 if hasattr(module, 'bias'):
#                     params +=  th.prod(th.LongTensor(list(module.bias.size())))
#                 summary[m_key]['nb_params'] = params
# 
#             if not isinstance(module, nn.Sequential) and \
#                not isinstance(module, nn.ModuleList) and \
#                not (module == self.model):
#                 hooks.append(module.register_forward_hook(hook))
# 
#         # create properties
#         summary = OrderedDict()
#         hooks = []
#         # register forward hooks
#         self.model.apply(register_hook)
# 
#         if isinstance(input_size[0], (list, tuple)):
#             x = [Variable(th.rand(1,*in_size)) for in_size in input_size]
#             self.model(*x)
#         else:
#             x = Variable(th.rand(1,*input_size))
#             self.model(x)
# 
#         # remove these hooks
#         for h in hooks:
#             h.remove()
# 
#         return summary



# def get_loss_cls(model, criterion, input, label, is_grad=False):
#     output = get_out(model, input, is_grad)
#     label = torch.ByteTensor(label.astype(np.uint8))
#     label = Variable(label).cuda() if is_grad else Variable(label, volatile=True).cuda()
#     loss = criterion(output.float(), label.float())
#     correct = label.data.eq(output.data.max(1)[1]).cpu().sum()
#     return loss, correct
# 
# def get_loss_seg(model, criterion, ori, gro, is_grad=False):
#     pre = get_out(model, ori, is_grad)
#     gro = torch.ByteTensor(gro.astype(np.uint8))
#     gro = Variable(gro).cuda() if is_grad else Variable(gro, volatile=True).cuda()
#     loss = criterion(pre.float(), gro.float())
#     acc = gro.eq( pre.gt(0.5) ).sum() / (gro.size()[0]*gro.size()[2]*gro.size()[3])
#     return loss, acc


