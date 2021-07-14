#! -*- coding:utf-8 -*-
import datetime
from fnmatch import fnmatch
import torch as tc


class CallbackContainer(object):
    """
    Container holding a list of callbacks.
    """
    def __init__(self, callbacks=None):
        self.callbacks = callbacks or []

    def append(self, callback):
        self.callbacks.append(callback)

    def set_params(self, params):
        for callback in self.callbacks:
            callback.set_params(params)

    def set_trainer(self, trainer):
        self.trainer = trainer
        for callback in self.callbacks:
            callback.set_trainer(trainer)

    def on_epoch_begin(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        logs = logs or {}
        logs['start_time'] = datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")
        for callback in self.callbacks:
            callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        logs = logs or {}
        logs['final_loss'] = self.trainer.history.epoch_losses[-1],
        logs['best_loss'] = min(self.trainer.history.epoch_losses),
        logs['stop_time'] = datetime.datetime.now().strftime("%B %d, %Y - %I:%M%p")
        for callback in self.callbacks:
            callback.on_train_end(logs)



class ConstraintContainer(object):

    def __init__(self, constraints=None):
        self.constraints = constraints or []
        self.batch_constraints = [c for c in self.constraints if c.unit.upper() == 'BATCH']
        self.epoch_constraints = [c for c in self.constraints if c.unit.upper() == 'EPOCH']

    def register_constraints(self, model):
        """
        Grab pointers to the weights which will be modified by constraints so
        that we dont have to search through the entire network using `apply`
        each time
        """
        # get batch constraint pointers
        self._batch_c_ptrs = {}
        for c_idx, constraint in enumerate(self.batch_constraints):
            self._batch_c_ptrs[c_idx] = []
            for name, module in model.named_modules():
                if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                    self._batch_c_ptrs[c_idx].append(module)

        # get epoch constraint pointers
        self._epoch_c_ptrs = {}
        for c_idx, constraint in enumerate(self.epoch_constraints):
            self._epoch_c_ptrs[c_idx] = []
            for name, module in model.named_modules():
                if fnmatch(name, constraint.module_filter) and hasattr(module, 'weight'):
                    self._epoch_c_ptrs[c_idx].append(module)

    def apply_batch_constraints(self, batch_idx):
        for c_idx, modules in self._batch_c_ptrs.items():
            if (batch_idx+1) % self.constraints[c_idx].frequency == 0:
                for module in modules:
                    self.constraints[c_idx](module)

    def apply_epoch_constraints(self, epoch_idx):
        for c_idx, modules in self._epoch_c_ptrs.items():
            if (epoch_idx+1) % self.constraints[c_idx].frequency == 0:
                for module in modules:
                    self.constraints[c_idx](module)



class InitializerContainer(object):

    def __init__(self, initializers):
        self.initializers = initializers

    def apply(self, model):
        for initializer in self.initializers:
            model.apply(initializer)



class MetricContainer(object):
    
    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self):
        for metric in self.metrics:
            metric.reset()

    def __call__(self, output_batch, target_batch, tail):
        logs = {}
        for metric in self.metrics:
            logs[metric.name+tail] = metric(output_batch,target_batch) 
        return logs



class RegularizerContainer(object):

    def __init__(self, regularizers):
        self.regularizers = regularizers
        self.forward_hooks = []

    def register_forward_hooks(self, model):
        for regularizer in self.regularizers:
            for module_name, module in model.named_modules():
                if fnmatch(module_name, regularizer.module_filter) and hasattr(module, 'weight'):
                    hook = module.register_forward_hook(regularizer)
                    self.forward_hooks.append(hook)

    def unregister_forward_hooks(self):
        for hook in self.forward_hooks:
            hook.remove()

    def reset(self):
        for r in self.regularizers:
            r.reset()

    def get_value(self):
        if len(self.regularizers)==0:
            return tc.tensor(0.).type(tc.FloatTensor).cuda()
        else:
            value = sum([r.value for r in self.regularizers])
            self.current_value = value.item()
            return value.cuda()

    def __len__(self):
        return len(self.regularizers)


