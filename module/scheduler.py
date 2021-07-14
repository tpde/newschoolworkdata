class Scheduler:
    def __init__(self, n_itrs_per_epoch_d, n_itrs_per_epoch_g, schedules, init_lr):
        self.schedules=schedules
        self.init_dsteps=n_itrs_per_epoch_d
        self.init_gsteps=n_itrs_per_epoch_g
        self.init_lr=init_lr
        self.dsteps=self.init_dsteps
        self.gsteps=self.init_gsteps
        self.lr=self.init_lr

    def get_dsteps(self):
        return self.dsteps
    
    def get_gsteps(self):
        return self.gsteps
    
    def get_lr(self):
        return self.lr
        
    def update_steps(self, n_round):
        key=str(n_round)
        if key in self.schedules['lr_decay']:
            self.lr=self.init_lr*self.schedules['lr_decay'][key]
        if key in self.schedules['step_decay']:
            self.dsteps=max(int(self.init_dsteps*self.schedules['step_decay'][key]),1)
            self.gsteps=max(int(self.init_gsteps*self.schedules['step_decay'][key]),1)