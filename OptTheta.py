
# coding: utf-8

# In[3]:


import numpy as np
from tqdm import tqdm

import torch
from torch.autograd import Variable
import matplotlib.pylab as plt
import csv

from IPython.display import clear_output


# In[4]:


class OptTheta():
    
    def __init__(self, theta_1=0, theta_2=2, alpha=0.8, beta=0.2, phi=0.8, method='exp_mean', use_h = False, lr = 0.001, 
                 schedule = None, schedule_factor=0.9, verbose=5, is_sgd=False):
        '''
        theta_1 - theta of the 1st line, usually 0 < theta_1 < 1
        theta_2 - theta of the 2nd line, usually > 1
        alpha - coefficient for exponential mean
        method - method to extrapolate z's values, possible values: exp_mean, exp_trend, holts, damped
        h - use h for predicting or not (not for exp_mean)
        lr - learning rate
        schedule - function for schedule of the probability to add true next value or predicted value
        schedule_factor - parameter for the schedule
        verbose - frequency for verbose output
        sgd - if True - use stochastic gradient descent, else - compute full loss and then make gd step
        '''
        self.is_sgd = is_sgd
        
        self.logs = dict()
        self.logs['err_log'] = []
        self.logs['mean_err_log'] = []
        
        self.lr = lr
        self.verbose = verbose
        self.test_size = None
        
        self.theta_1 = Variable(torch.FloatTensor(np.array([theta_1])), requires_grad=True)
        self.theta_2 = Variable(torch.FloatTensor(np.array([theta_2])), requires_grad=True)
        self.logs['theta_1'] = [theta_1]
        self.logs['theta_2'] = [theta_2]
        
        self.using_method = method
        self.use_h = use_h
        
        self.b_cur = None
        self.l_cur = None
        
        if schedule is None:
            self.schedule = self.exp_schedule
        else:
            self.schedule = schedule
        self.schedule_factor = schedule_factor
        
        if method == 'exp_mean':
            self.method = self.exp_mean
            self.method_next = self.exp_mean_next
            
            self.alpha = Variable(torch.FloatTensor(np.array([alpha])), requires_grad=True)
            self.logs['alpha'] = [alpha]
            
        elif method == 'exp_trend':
            self.method = self.exp_trend
            self.method_next = self.exp_trend_next
            
            self.alpha = Variable(torch.FloatTensor(np.array([alpha])), requires_grad=True)
            self.beta = Variable(torch.FloatTensor(np.array([beta])), requires_grad=True)
            self.logs['alpha'] = [alpha]
            self.logs['beta'] = [beta]
            
        elif method == 'holts':
            self.method = self.holts
            self.method_next = self.holts_next
            
            self.alpha = Variable(torch.FloatTensor(np.array([alpha])), requires_grad=True)
            self.beta = Variable(torch.FloatTensor(np.array([beta])), requires_grad=True)
            self.logs['alpha'] = [alpha]
            self.logs['beta'] = [beta]
            
        elif method == 'damped':
            self.method = self.damped
            self.method_next = self.damped_next
            
            self.alpha = Variable(torch.FloatTensor(np.array([alpha])), requires_grad=True)
            self.beta = Variable(torch.FloatTensor(np.array([beta])), requires_grad=True)
            self.phi = Variable(torch.FloatTensor(np.array([phi])), requires_grad=True)
            self.logs['alpha'] = [alpha]
            self.logs['beta'] = [beta]
            self.logs['phi'] = [phi]
            
        else:
            print(':P')
            print('Wrong method')
        
    def exp_schedule(self, cur_prob, iteration):
        '''Schedule for the probabilty, exponential decay'''
        return cur_prob * self.schedule_factor
        
    def exp_mean(self, z):
        '''Compute exponential mean with coefficient alpha'''
        
        exp = Variable(torch.arange(z.shape[0] - 1, -1, -1))
        power = torch.pow(1 - self.alpha, exp)
        
        return torch.sum(self.alpha * power * z)
    
    def exp_mean_next(self, prev_z, next_z):
        return self.alpha * next_z + (1 - self.alpha) * prev_z

    def exp_trend(self, z):
        self.l_cur = z[0]
        self.b_cur = z[1] / z[0]
        cur_val = None
        

        for i in range(len(z)):
            cur_val = self.exp_trend_next(z[i])
        
        return cur_val
        
    def exp_trend_next(self, prev_z, next_z = None, l=None, b=None, h = 1):
        if l is None:
            l = self.l_cur
        if b is None:
            b = self.b_cur

        self.b_cur = self.beta * (self.alpha * prev_z + (1 - self.alpha) * l * b) / l + (1 - self.beta) * b
        self.l_cur = self.alpha * prev_z + (1 - self.alpha) * l * b

        return self.l_cur * torch.pow(self.b_cur, h)
        
    def holts(self, z):
        self.l_cur = z[0]
        self.b_cur = z[1] - z[0]
        cur_val = None
        
        for i in range(len(z)):
            cur_val = self.holts_next(z[i])
        
        return cur_val
        
    def holts_next(self, prev_z, next_z = None, l=None, b=None, h = 1):
        if l is None:
            l = self.l_cur
        if b is None:
            b = self.b_cur

        self.b_cur = self.beta * ((self.alpha * prev_z + (1 - self.alpha) * (l + b)) - l) + (1 - self.beta) * b
        self.l_cur = self.alpha * prev_z + (1 - self.alpha) * (l + b)

        return self.l_cur + self.b_cur * h

    def damped(self, z):
        self.l_cur = z[0]
        self.b_cur = z[1] - z[0]
        cur_val = None
        
        for i in range(len(z)):
            cur_val = self.damped_next(z[i])
        
        return cur_val
        
    def damped_next(self, prev_z, next_z = None, l=None, b=None, h = 1):
        if l is None:
            l = self.l_cur
        if b is None:
            b = self.b_cur

        self.b_cur = (self.beta * ((self.alpha * prev_z + (1 - self.alpha) * (l + self.phi * b)) - l) +
                     (1 - self.beta) * self.phi * b)
        
        self.l_cur = self.alpha * prev_z + (1 - self.alpha) * (l + self.phi * b)

        return self.l_cur + torch.cumsum(torch.pow(self.phi, h), -1) * self.b_cur
        
    def get_predict(self, em1, em2):
        '''Compute predicted value, formulas required, that 0 < theta_1 < 1, theta_2 > 1'''
        
        w1 = (self.theta_2 - 1) / (self.theta_2 - self.theta_1)
        w2 = (1 - self.theta_1) / (self.theta_2 - self.theta_1)
        
        return w1 * em1 + w2 * em2
    
    def mae(self, target, predict):
        '''Compute mean average error'''
        

        idx = torch.LongTensor(np.array(np.isnan(predict.data.numpy()), dtype=int))
        return torch.mean(torch.abs(predict[idx] - target[idx]))
    
    def logging(self):
        self.logs['theta_1'].append(self.theta_1.data.numpy()[0])
        self.logs['theta_2'].append(self.theta_2.data.numpy()[0])
        
        if 'alpha' in self.logs:
            self.logs['alpha'].append(self.alpha.data.numpy()[0])
        if 'beta' in self.logs:
            self.logs['beta'].append(self.beta.data.numpy()[0])
        if 'phi' in self.logs:
            self.logs['phi'].append(self.phi.data.numpy()[0])
            
    def print_graph(self):
        clear_output()
        plt.plot(np.arange(len(self.logs['theta_1'])), self.logs['theta_1'], label='theta_1')
        plt.plot(np.arange(len(self.logs['theta_2'])), self.logs['theta_2'], label='theta_2')
        plt.title('Current values: theta_1 = {}, theta_2 = {}'.format(self.theta_1.data.numpy()[0], self.theta_2.data.numpy()[0]))
        plt.legend(loc='best')
        plt.show()
        
        for name in ['alpha', 'beta', 'phi']:
            if name in self.logs:
                plt.plot(np.arange(len(self.logs[name])), self.logs[name], label=name)
                plt.title('Current value {} = {}'.format(name, self.logs[name][-1]))
                plt.legend(loc='best')
                plt.show()
                
        plt.plot(np.arange(len(self.logs['err_log'])), self.logs['err_log'], label='err_log', alpha=0.3)
        plt.plot(np.arange(len(self.logs['mean_err_log'])) * self.test_size, self.logs['mean_err_log'], label='mean_err_log', alpha=0.7)
        plt.title('Current values: err_log = {}, mean_err_log = {}'.format(self.logs['err_log'][-1], self.logs['mean_err_log'][-1]))
        plt.legend(loc='best')
        plt.show()
        
    def fit(self, data_base, data_train, n_iters, optimizer=torch.optim.Adam, loss_function=None, 
            constraints_1=True, constraints_2=True, lr=None):
        '''
        Find optimal theta_1, theta_2 and alpha with gradiend descent
        
        data_base - data to compute first n thetas
        data_train - data to optimize parameters with GD
        n_iters - number of epochs
        optimizer - which type of GD to use
        loss_function - function to compute error, should be writtein on pytorch, syntax: loss_function(target, predict),
        default value is mae
        constraints_1 - flag to constraint values of thetas
        constraints_2 - flag to constraint values of alpha, beta, phi
        '''
        
        if loss_function is None:
            loss_function = self.mae
        
        if lr is None:
            lr = self.lr
        
        #Simple preparing
        y_np = np.copy(data_base)
        n = len(y_np)
        
        t_np = np.arange(n) + 1
        t_var = Variable(torch.FloatTensor(t_np))
        y_var = Variable(torch.FloatTensor(y_np))
        
        #Compute coefficients A and B 
        B = 6 / (n ** 2 - 1) * (2 / n * np.sum(t_np * y_np) - (1 + n) / n * np.sum(y_np))
        A = 1 / n * np.sum(y_np) - (n + 1) / 2 * B
        
        B = Variable(torch.FloatTensor(np.array([B])))
        A = Variable(torch.FloatTensor(np.array([A])))
        
        #Compute sequence of z_t 
        z_var_1 = self.theta_1 * y_var + (1 - self.theta_1) * (A + B * t_var)
        z_var_2 = self.theta_2 * y_var + (1 - self.theta_2) * (A + B * t_var)

        list_of_params = [self.theta_1, self.theta_2, self.alpha]
        if self.using_method in ['exp_trend', 'holts', 'damped']:
            list_of_params.append(self.beta)
        if self.using_method == 'damped':
            list_of_params.append(self.phi)
            
        opt = optimizer(list_of_params, lr=lr)
        targets_np = np.copy(data_train)
        targets_var = Variable(torch.FloatTensor(targets_np))
        
        prob = 1
        self.test_size = targets_var.shape[0]
        

        for it in range(n_iters):

            if not self.is_sgd:
                full_predict = Variable(torch.FloatTensor(np.zeros(self.test_size)))
    
            if self.using_method == 'exp_mean' or not self.use_h:
                # Compute exponential mean to predict
                z1_predict = self.method(z_var_1)
                z2_predict = self.method(z_var_2)
                
                for i in range(targets_var.shape[0]):
                    #predict
                    predict = self.get_predict(z1_predict, z2_predict)

                    #Gradient step
                    if self.is_sgd:
                        opt.zero_grad()
                        loss = loss_function(predict, targets_var[i])
                        self.logs['err_log'].append(loss.data.numpy()[0])
                        loss.backward(retain_graph=True)
                        opt.step()
                    else:
                        full_predict[i] = predict

                    #Limiting values
                    if constraints_1:
                        if (self.theta_1 < 0).data.numpy():
                            self.theta_1 = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)
                        if (self.theta_1 > 1).data.numpy():
                            self.theta_1 = Variable(torch.FloatTensor(np.array([1.0])), requires_grad=True)
                        if (self.theta_2 < 1).data.numpy():
                            self.theta_2 = Variable(torch.FloatTensor(np.array([2.0])), requires_grad=True)
                    
                    if constraints_2:    
                        if (self.alpha > 1).data.numpy():
                            self.alpha = Variable(torch.FloatTensor(np.array([1.0])), requires_grad=True)
                        if (self.alpha < 0).data.numpy():
                            self.alpha = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)

                        if 'beta' in self.logs and (self.beta > 1).data.numpy():
                            self.beta = Variable(torch.FloatTensor(np.array([1.0])), requires_grad=True)
                        if 'beta' in self.logs and (self.beta < 0).data.numpy():
                            self.beta = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)

                        if 'phi' in self.logs and (self.phi > 1).data.numpy():
                            self.phi = Variable(torch.FloatTensor(np.array([0.98])), requires_grad=True)
                        if 'phi' in self.logs and (self.phi < 0).data.numpy():
                            self.phi = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)
                            
                    self.logging()
                    
                    # Remember last values
                    prev_z1 = z1_predict 
                    prev_z2 = z2_predict

                    # Compute next value in z
                    if np.random.binomial(n=1, p=prob, size=1) == 0:
                        next_z1 = self.theta_1 * predict + (1 - self.theta_1) * (A + B * (n + i + 1))
                        next_z2 = self.theta_2 * predict + (1 - self.theta_2) * (A + B * (n + i + 1))
                    else:
                        next_z1 = self.theta_1 * targets_var[i] + (1 - self.theta_1) * (A + B * (n + i + 1))
                        next_z2 = self.theta_2 * targets_var[i] + (1 - self.theta_2) * (A + B * (n + i + 1))

                    #Compute next exp mean
                    z1_predict = self.method_next(prev_z1, next_z1)
                    z2_predict = self.method_next(prev_z2, next_z2)
                
                
                if not self.is_sgd:
                    opt.zero_grad()
                    loss = loss_function(full_predict, targets_var)
                    self.logs['err_log'].append(loss.data.numpy()[0])
                    self.logs['mean_err_log'].append(loss.data.numpy()[0])
                    loss.backward(retain_graph=True)
                    opt.step()
                
                #decrease prob
                prob = self.schedule(prob, i)
                if self.is_sgd:
                    self.logs['mean_err_log'].append(np.mean(self.logs['err_log'][-self.test_size : ]))

            else:
                predicts = Variable(torch.FloatTensor(np.zeros(len(targets_var))))            
                z1_predict = self.method(z_var_1)
                z2_predict = self.method(z_var_2)

                predicts[0] = self.get_predict(z1_predict, z2_predict)
                
                h = Variable(torch.FloatTensor(np.arange(1, len(targets_var))))
                z1_predict = self.method_next(z1_predict, h = h)
                z2_predict = self.method_next(z2_predict, h = h)
                predicts[1:] = self.get_predict(z1_predict, z2_predict)

                opt.zero_grad()
                loss = loss_function(targets_var, predicts)
                self.logs['err_log'].append(loss.data.numpy()[0])
                self.logs['mean_err_log'].append(loss.data.numpy()[0])
                loss.backward(retain_graph=True)
                opt.step()
                
                if constraints_1:
                    if (self.theta_1 < 0).data.numpy():
                        self.theta_1 = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)
                    if (self.theta_1 > 1).data.numpy():
                        self.theta_1 = Variable(torch.FloatTensor(np.array([1.0])), requires_grad=True)
                    if (self.theta_2 < 1).data.numpy():
                        self.theta_2 = Variable(torch.FloatTensor(np.array([2.0])), requires_grad=True)
                
                if constraints_2:
                    if (self.alpha > 1).data.numpy():
                        self.alpha = Variable(torch.FloatTensor(np.array([1.0])), requires_grad=True)
                    if (self.alpha < 0).data.numpy():
                        self.alpha = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)
                        
                    if 'beta' in self.logs and (self.beta > 1).data.numpy():
                        self.beta = Variable(torch.FloatTensor(np.array([1.0])), requires_grad=True)
                    if 'beta' in self.logs and (self.beta < 0).data.numpy():
                        self.beta = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)
                        
                    if 'phi' in self.logs and (self.phi > 1).data.numpy():
                        self.phi = Variable(torch.FloatTensor(np.array([0.98])), requires_grad=True)
                    if 'phi' in self.logs and (self.phi < 0).data.numpy():
                        self.phi = Variable(torch.FloatTensor(np.array([0.0])), requires_grad=True)
            
                self.logging()
              ### for printing progress of optimization  
#             if it % self.verbose == 0 or it == n_iters - 1:
#                 self.print_graph()
                    
        return self.logs['err_log']
                
    def predict(self, data_base, n_iters, use_h = None):
        '''
        Predict n_iters values
        n_iters - number of values to predict
        use_h - use h for prediction or predict iteratively
        '''
        
        if use_h is None:
            use_h = self.use_h
        
        y_np = np.copy(data_base)
        n = len(y_np)

        t_np = np.arange(n) + 1
        t_var = Variable(torch.FloatTensor(t_np))
        y_var = Variable(torch.FloatTensor(y_np))

        B = 6 / (n ** 2 - 1) * (2 / n * np.sum(t_np * y_np) - (1 + n) / n * np.sum(y_np))
        A = 1 / n * np.sum(y_np) - (n + 1) / 2 * B

        B = Variable(torch.FloatTensor(np.array([B])))
        A = Variable(torch.FloatTensor(np.array([A])))

        z_var_1 = self.theta_1 * y_var + (1 - self.theta_1) * (A + B * t_var)
        z_var_2 = self.theta_2 * y_var + (1 - self.theta_2) * (A + B * t_var)

        z1_predict = self.method(z_var_1)
        z2_predict = self.method(z_var_2)

        predicted_values = []

        
        if self.using_method == 'exp_mean' or not use_h:
            for i in range(n_iters):
                predict = self.get_predict(z1_predict, z2_predict)
            
                predicted_values.append(predict.data.numpy()[0])
                # Remember last values
                prev_z1 = z1_predict 
                prev_z2 = z2_predict

                # Compute next value in z

                next_z1 = self.theta_1 * predict + (1 - self.theta_1) * (A + B * (n + i + 1))
                next_z2 = self.theta_2 * predict + (1 - self.theta_2) * (A + B * (n + i + 1))

                #Compute next exp mean
                z1_predict = self.method_next(prev_z1, next_z1)
                z2_predict = self.method_next(prev_z2, next_z2)

            return np.array(predicted_values)
        
        else:
            predicts = np.zeros(n_iters)
            z1_predict = self.method(z_var_1)
            z2_predict = self.method(z_var_2)
            predicts[0] = self.get_predict(z1_predict, z2_predict).data.numpy()
            
            h = Variable(torch.FloatTensor(np.arange(1, n_iters)))
            z1_predict = self.method_next(z1_predict, h = h)
            z2_predict = self.method_next(z2_predict, h = h)
            predicts[1:] = self.get_predict(z1_predict, z2_predict).data.numpy()
            
            return predicts

