import csv
import numpy as np
from sklearn import datasets
import time

class SVM(object):
    # This class is the hard margin SVM and it is the parent
    # class of KernelSVM and SoftMarginSVM.
    # Please add any function to the class if it is needed.
    def __init__(self, sample, label):
    # This function is an constructor and shouldn't be modified.
    # The 'self.w' represents the director vector and should be
    # in the form of numpy.array 
    # The 'self.b' is the displacement of the SVM and it should
    # be a float64 variable.
        self.sample = sample
        self.label = label
        
        self.label=self.label.reshape(-1,1)
        self.sample_num = len(sample)
        self.sample_var_num = self.sample.shape[1]
        self.sample=self.sample.reshape(self.sample_num,self.sample_var_num)

        self.sample_train = self.sample
        self.label_train=self.label

        self.w = np.zeros((self.sample_var_num, 1),dtype='float64')
        self.alpha = np.zeros((len(self.sample_train), 1),dtype='float64')
        self.b = 0.0
        self.C = float('inf')# regularization parameter
        self.tol=0.00001 # numerical tolerance

    # Won't change
    def calc_ker_val(self, x_mat, x_vec, kernel_type,par):
        num_samp=x_mat.shape[0]
        ker_val = np.mat(np.zeros((num_samp, 1),dtype='float64'))

        if kernel_type == 'Linear':
            ker_val = x_mat * x_vec.T
        elif kernel_type == 'Polynomial':
            for i in range(num_samp):
                ker_val[i] = (x_mat[i,:] * x_vec.T)**par
        elif kernel_type == 'Gaussian':
            for i in range(num_samp):
                minus = x_mat[i,:] - x_vec
                dis_sqr=float(minus * minus.T)
                ker_val[i] = np.exp(-dis_sqr / (2.0 * (par ** 2)))
        elif kernel_type == 'Laplace':
            for i in range(num_samp):
                minus = x_mat[i,:] - x_vec
                dis=float(minus * minus.T)**0.5
                ker_val[i] = np.exp(-dis/par)
        return ker_val

    # Won't change
    def calc_ker(self, x_mat, kernel_type, par):
        num_samp = x_mat.shape[0]
        x_mat=np.mat(x_mat)
        mat_ker = np.mat(np.zeros((num_samp,num_samp),dtype='float64'))
        for i in range(num_samp):
            mat_ker[:, i] = self.calc_ker_val(x_mat, x_mat[i,:], kernel_type,par)
        return mat_ker

    # Will change
    def calc_err(self, index, ker_mat):
        alpha = np.mat(self.alpha)
        label_train = np.mat(self.label_train)
        output = float(np.multiply(alpha, label_train).T * ker_mat[:, index] + self.b)
        err_val = output - float(label_train[index,0])
        return err_val

    def training(self):
    # Implement this function by yourself and do not modifiy 
    # the parameter.
        ker_mat=self.calc_ker(self.sample_train,'Linear',1)
        max_pass = int(1000)
        cur_pass = 0
        samp = np.mat(self.sample_train)
        labl = np.mat(self.label_train)
        alp = np.mat(self.alpha)
        C=self.C
        while (cur_pass < max_pass):
            num_chg_a = 0
            for i in range(len(samp)):
                E_i = self.calc_err(i, ker_mat)
                if (((labl[i,0] * E_i < -self.tol) and (alp[i,0] < self.C)) or (labl[i,0] * E_i > self.tol) and (alp[i,0] > 0)):
                    
                    j = np.random.choice(len(samp), 1, replace=False)[0]
                    while (j == i):
                        j = np.random.choice(len(samp), 1, replace=False)[0]
                    E_j = self.calc_err(j, ker_mat)

                    alp_old_i = alp[i,0]
                    alp_old_j = alp[j,0]
                    L = 0
                    H = C
                    if labl[i,0] != labl[j,0]:
                        L = max(0, alp[j,0] - alp[i,0])
                        H = min(C, C + alp[j,0] - alp[i,0])
                    else:
                        L = max(0, alp[j,0] + alp[i,0] - C)
                        H = min(C, alp[j,0] + alp[i,0])
                    if (L == H):
                        continue

                    eta = 2.0 * ker_mat[i, j] - ker_mat[i, i] - ker_mat[j, j]
                    if (eta >= 0):
                        continue

                    alp[j,0] -= labl[j,0] * (E_i - E_j) / eta

                    # step 5: clip alpha j
                    if alp[j,0] > H:
                        alp[j,0] = H
                    if alp[j,0] < L:
                        alp[j,0] = L

                    if (abs(alp[j,0] - alp_old_j) < 0.00001):
                        continue

                    alp[i,0] += labl[i,0] *labl[j,0]* (alp_old_j - alp[j,0])

                    b1 = float(self.b) -  E_i - labl[i,0] * (alp[i,0] - alp_old_i) * ker_mat[i, i] - labl[j,0] * (alp[j,0] - alp_old_j) * ker_mat[i, j]
                    
                    b2 = float(self.b) - E_j - labl[i,0] * (alp[i,0] - alp_old_i) * ker_mat[i, j] - labl[j,0] * (alp[j,0] - alp_old_j) * ker_mat[j, j]

                    if (alp[i,0]>0) and (alp[i,0] < C):
                        self.b = b1
                    elif (alp[j,0]>0) and (alp[j,0] < C):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_chg_a += 1
            if (num_chg_a == 0):
                cur_pass += 1
            else:
                cur_pass = 0
        
        self.alpha = np.array(alp)
        w = np.multiply(alp, labl).T * samp
        
        self.w = np.array(w)
        self.b=float(self.b)

    def testing(self, test_sample, test_lable):
    # This function should return the accuracy 
    # of the input test_sample in float64, e.g 0.932
        
        test_num = test_sample.shape[0]
        corr = 0

        test_var_num = test_sample.shape[1]
        test_samp = np.mat(test_sample.reshape(test_num, test_var_num))

        for i in range(test_num):
            x_mat=np.mat(self.sample_train)
            ker=self.calc_ker_val(x_mat,test_samp[i,:],'Linear',1)
            output = float(np.multiply(np.mat(self.alpha), np.mat(self.label_train)).T * ker + self.b)

            if ((output)*(test_lable[i])>0):
                    corr += 1
        
        return float(corr)/test_num

    def parameter_w(self):
    # This function is used to return the parameter w of the SVM.
    # The result is supposed to be an np.array
    # This functin shouldn't be modified.
        return self.w 
    def parameter_b(self):
    # This function is used to return the parameter b of the SVM.
    # The result is supposed to be an real number.
    # This functin shouldn't be modified.
        return self.b  


class KernelSVM(SVM):
    # This class is the kernel SVM.
    # Please add any function to the class if it is needed.
    def training(self, kernel = 'Linear', parameter = 1):
    # Specifics:
    #   For the parameter of 'kernel':
    #   1. The default kernel function is 'Linear'.
    #      The parameter is 1 by default.
    #   2. Gaussian kernel function is 'Gaussian'.
    #      The parameter is the Gaussian bandwidth.
    #   3. Laplace kernel funciton is 'Laplace'.
    #   4. Polynomial kernel functino is 'Polynomial'.
    #      The parameter is the exponential of polynomial.
        self.kernel = kernel
        self.parameter = parameter
        ker_mat=self.calc_ker(self.sample_train,kernel,parameter)
        max_pass = int(1000)
        cur_pass = 0
        samp = np.mat(self.sample_train)
        labl = np.mat(self.label_train)
        alp = np.mat(self.alpha)
        C=self.C
        while (cur_pass < max_pass):
            num_chg_a = 0
            for i in range(len(samp)):
                E_i = self.calc_err(i, ker_mat)
                if (((labl[i,0] * E_i < -self.tol) and (alp[i,0] < self.C)) or (labl[i,0] * E_i > self.tol) and (alp[i,0] > 0)):
                    
                    j = np.random.choice(len(samp), 1, replace=False)[0]
                    while (j == i):
                        j = np.random.choice(len(samp), 1, replace=False)[0]
                    E_j = self.calc_err(j, ker_mat)

                    alp_old_i = alp[i,0].copy()
                    alp_old_j = alp[j,0].copy()
                    L = 0
                    H = C
                    if labl[i,0] != labl[j,0]:
                        L = max(0, alp[j,0] - alp[i,0])
                        H = min(C, C + alp[j,0] - alp[i,0])
                    else:
                        L = max(0, alp[j,0] + alp[i,0] - C)
                        H = min(C, alp[j,0] + alp[i,0])
                    if (L == H):
                        continue

                    eta = 2.0 * ker_mat[i, j] - ker_mat[i, i] - ker_mat[j, j]
                    if (eta >= 0):
                        continue

                    alp[j,0] -= labl[j,0] * (E_i - E_j) / eta

                    # step 5: clip alpha j
                    if alp[j,0] > H:
                        alp[j,0] = H
                    if alp[j,0] < L:
                        alp[j,0] = L

                    if (abs(alp[j,0] - alp_old_j) < 0.00001):
                        continue

                    alp[i,0] += labl[i,0] *labl[j,0]* (alp_old_j - alp[j,0])

                    b1 = float(self.b) -  E_i - labl[i,0] * (alp[i,0] - alp_old_i) * ker_mat[i, i] - labl[j,0] * (alp[j,0] - alp_old_j) * ker_mat[i, j]
                    
                    b2 = float(self.b) - E_j - labl[i,0] * (alp[i,0] - alp_old_i) * ker_mat[i, j] - labl[j,0] * (alp[j,0] - alp_old_j) * ker_mat[j, j]

                    if (alp[i,0]>0) and (alp[i,0] < C):
                        self.b = b1
                    elif (alp[j,0]>0) and (alp[j,0] < C):
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2.0

                    num_chg_a += 1
            if (num_chg_a == 0):
                cur_pass += 1
            else:
                cur_pass = 0
        
        self.alpha = np.array(alp)
        w = np.multiply(alp, labl).T * samp
        
        self.w = np.array(w)
        self.b=float(self.b)

    def testing(self, test_sample, test_lable, kernel='Linear', parameter = 1):
        # This function should return the accuracy 
        # of the input test_sample in float64, e.g 0.932
            
            test_num = test_sample.shape[0]
            corr = 0

            test_var_num = test_sample.shape[1]
            test_samp = np.mat(test_sample.reshape(test_num, test_var_num))

            for i in range(test_num):
                x_mat=np.mat(self.sample_train)
                ker=self.calc_ker_val(x_mat,test_samp[i,:],kernel,parameter)
                output = float(np.multiply(np.mat(self.alpha), np.mat(self.label_train)).T * ker + self.b)

                if ((output)*(test_lable[i])>0):
                    corr += 1
            
            return float(corr)/test_num

class SoftMarginSVM(KernelSVM):
    # This class is the soft margin SVM and inherits
    # the kernel SVM to expand to both linear Non-seperable and
    # soft margin problem.
    # Please add any function to the class if it is needed.
    def __init__(self, sample, label, C, tol):
        KernelSVM.__init__(self, sample, label)
        self.C = C
        self.tol=tol

    
# Sample Usage


iris = datasets.load_iris()

x_vals = np.array([[x[0], x[3]] for x in iris.data],float)
y_vals = np.array([1 if y == 0 else -1 for y in iris.target],float)

# Set train samples and test samples
train_indices = np.random.choice(len(x_vals), int(0.8*len(x_vals)), replace=False)
sample_train = x_vals[train_indices]
label_train = y_vals[train_indices]
test_indices = np.array(list(set(range(len(x_vals))) - set(train_indices)))
sample_test = x_vals[test_indices]
label_test = y_vals[test_indices]

test = KernelSVM(sample_train, label_train)
start=time.time()
test.training('Laplace')
t=float(time.time()-start)
print('Training Accuracy is', test.testing(sample_test, label_test,'Laplace'))
print('w is', test.parameter_w())
print('b is', test.parameter_b())
print('Training time is', t, 's')