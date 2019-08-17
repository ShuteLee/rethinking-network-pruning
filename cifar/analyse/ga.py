# -*- coding: utf-8 -*-
import numpy as np
import geatpy as ea # import geatpy
from pruning import *
import xlwt
import argparse
# from MyProblem import MyProblem # 导入自定义问题接口

parser = argparse.ArgumentParser(description='PyTorch Slimming CIFAR training')
parser.add_argument('--ptarget', type=int, default=32, help='pruning target')
parser.add_argument('--ptimes', type=int, default=5, help='pruning times')
parser.add_argument('--save_name', type=str, default='data.xls', help='savig to excel')
parser.add_argument('--nind', type=int, default=5, help='nind')
args = parser.parse_args()

workbook = xlwt.Workbook()
sheet = workbook.add_sheet('data')

os.environ['CUDA_VISIBLE_DEVICES'] = "0"
class MyProblem(ea.Problem): # 继承Problem父类
    generate_num = 0
    filter_scale = [1, 2, 3, 3, 2]
    def __init__(self):
        name = 'MyProblem' # 初始化name（函数名称，可以随意设置）
        M = 1 # 初始化M（目标维数）
        maxormins = [-1] # 初始化maxormins（目标最小最大化标记列表，1：最小化该目标；-1：最大化该目标）
        Dim = args.ptimes # 初始化Dim（决策变量维数）
        varTypes = [1] * Dim # 初始化varTypes（决策变量的类型，元素为0表示对应的变量是连续的；1表示是离散的）
        lb = np.zeros(Dim, dtype=np.int).tolist() # 决策变量下界
        ub = (np.ones(Dim, dtype=np.int)*args.ptarget).tolist() # 决策变量上界
        lbin = [1] * Dim # 决策变量下边界
        ubin = [1] * Dim # 决策变量上边界
        # 调用父类构造方法完成实例化
        ea.Problem.__init__(self, name, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)
    
    def aimFunc(self, pop): # 目标函数
        # print('generation:{}'.format(self.generate_num))
        x = pop.Phen # 得到决策变量矩阵
        x = x[:] * (1 / np.sum(x, 1) * args.ptarget).reshape(-1, 1)
        x = x.astype(int)
        objv = []
        for i in range(x.shape[0]):
            print('generation:{}, individual:{}'.format(self.generate_num, i))
            left = args.ptarget
            num_list = []
            for item in x[i][:-1]:
                if left > 0:
                    num_list.append(np.min([left, item]))
                    left = left - np.min([left, item])
                else:
                    num_list.append(0)
            num_list.append(left)
            print('pruning num of each step is:{}'.format(num_list))
            pruner = Pruner()
            acc_p, acc_f = pruner.pruning_list(num_list)
            objv.append(acc_f)
            for j in range(len(num_list)):
                sheet.write(self.generate_num*(args.nind+2)+i+1, j+1, int(num_list[j]))
                sheet.write(self.generate_num*(args.nind+2)+i+1, j+args.ptimes+2, acc_p[j])
            sheet.write(self.generate_num*(args.nind+2)+i+1, j+args.ptimes+4, acc_f)
            # objv.append((num_list[0] - 5)*2+(num_list[0] - 3)*2+(num_list[0] - 1)*2+(num_list[0] +1)*2+(num_list[0] +3)*2)
        objv = np.array(objv,dtype=np.float)
        pop.ObjV = objv.reshape(-1, 1)
        self.generate_num += 1


"""==================================实例化问题对象================================"""
problem = MyProblem() # 生成问题对象
"""==================================种群设置================================"""
Encoding = 'BG'       # 编码方式
NIND = args.nind             # 种群规模
Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders) # 创建区域描述器
population = ea.Population(Encoding, Field, NIND) # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
"""==================================算法参数设置================================"""
myAlgorithm = ea.soea_EGA_templet(problem, population) # 实例化一个算法模板对象
myAlgorithm.MAXGEN = 10 # 最大进化代数
"""=======================调用算法模板进行种群进化=============================="""
[population, obj_trace, var_trace] = myAlgorithm.run() # 执行算法模板
population.save() # 把最后一代种群的信息保存到文件中
# 输出结果
best_gen = np.argmax(obj_trace[:, 1]) # 记录最优种群是在哪一代
best_ObjV = obj_trace[best_gen, 1]
print('最优的目标函数值为：%s'%(best_ObjV))
print('最优的控制变量值为：')
for i in range(var_trace.shape[1]):
    print(var_trace[best_gen, i])
print('有效进化代数：%s'%(obj_trace.shape[0]))
print('最优的一代是第 %s 代'%(best_gen + 1))
print('评价次数：%s'%(myAlgorithm.evalsNum))
workbook.save(args.save_name)
print('时间已过 %s 秒'%(myAlgorithm.passTime))



