# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 11:52:41 2018

@author: huang

求Y=X1^2+2*X2^2的最小值
化成二次型X.T A X+2 b.T X
则A=[[1,0],[0,2]],b=[0,0]

"""
import numpy as np

def gradient_method_quadratic(A,b,x0,epsilon):
    x = x0
    iter_num = 0
    # 梯度,对二次型求导得到
    grad = 2*(A.dot(x)+b)
    while(np.linalg.norm(grad)>epsilon):
        iter_num=iter_num+1
        # t为步长,是对f(x+t*d)求导得到的，d为取的每次下降的梯度，此处取d=-grad
        t = np.linalg.norm(grad)**2/(2*grad.T.dot(A).dot(grad))
        # 梯度下降
        x = x-t*grad
        grad = 2*(A.dot(x)+b)
        fun_val = x.T.dot(A).dot(x)+2*b.T.dot(x)
        print('iter_number = %3d norm_grad = %2.6f fun_val = %2.6f \n'%(iter_num,np.linalg.norm(grad),fun_val))

A = np.array([[1,0],[0,2]])
b = np.array([0,0])
x0 = np.array([2,1])
epsilon=1e-5
gradient_method_quadratic(A,b,x0,epsilon)
