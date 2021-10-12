import numpy
import random
import copy
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt

class Particles:
    def __init__(self):
        self.X = []  # 其中一个基本粒子
        self.S = []   # 其中一个粒子的适应度函数值，数组大小取决于目标个数，这里采取两个适应度函数
        self.C = 0      # 其中一个粒子的拥挤值
        self.XX = []     # 存放所有基本粒子以及其适应度函数值以及拥挤度 XX【i】= 【X，S，C】
        self.XX_child = []  # 存放所有子代粒子以及其适应度函数值以及拥挤度 XX【i】= 【X，S，C】
        self.np = []      # 存放所有基本粒子的被支配个数
        self.sp = []      # 存放所有基本粒子支配的个体集合（编号）
        self.XX2 = []   # 父子代混合基本粒子
        self.np2 = []      # 父子代混合被支配个体数
        self.sp2 = []      # 父子代混合支配的个体的集合
        self.pareto = []    # 记录各个阶级的个体，其中pareto【0】记录了最优的粒子编号(第0层），长度为阶级数
        self.rank = []      # 记录每个粒子的阶级，长度为粒子数
        self.rank2 = []  # 记录父子粒子的阶级，长度为粒子数
        self.cr = 0.4   # 交叉概率
        self.mu = 0.4   # 变异概率

    def init(self, number, length):
        """
        初始化种群
        :param number: 种群大小
        :param length: 粒子长度
        :return:
        """
        # 随机生成粒子
        self.S = numpy.zeros(2)
        self.X = numpy.zeros(length)
        for i in range(number):
            for j in range(length):     # 粒子的取值在0，1之间
                self.X[j] = random.random()
            self.S = self.cal_obj(self.X)
            temp_x = copy.deepcopy(self.X)
            self.XX.append([temp_x, self.S, self.C])

    def cal_obj(self, x):  # 计算一个个体的多目标函数值 f1,f2 最小值,函数值越小视为越优
        f1 = x[0]
        f = 0
        for i in range(len(self.X) - 1):
            f += 9 * (x[i + 1] / (len(self.X) - 1))
        g = 1 + f
        f2 = g * (1 - numpy.square(f1 / g))
        return [f1, f2]

    def quick_sort(self):   # 快速非支配排序
        #   算出每个粒子的np和sp
        self.np = numpy.zeros(len(self.XX))
        self.sp = []
        self.pareto = []
        self.rank = numpy.zeros(len(self.XX))
        # 1.计算出种群每个粒子的np和sp
        for i in range(len(self.XX)):
            temp = []   # 存放第i个粒子支配的粒子集合(编号）
            for j in range(len(self.XX)):
                if i != j:
                    if self.XX[i][1][0] <= self.XX[j][1][0] and self.XX[i][1][1] <= self.XX[j][1][1]:     # j被i支配
                        temp.append(j)
                    elif self.XX[i][1][0] >= self.XX[j][1][0] and self.XX[i][1][1] >= self.XX[j][1][1]:     # i被j支配
                        self.np[i] += 1
            self.sp.append(temp)
        # 2.算出第0阶级的粒子
        temp = []  # 处于同一个阶级的粒子列表
        for i in range(len(self.XX)):
            if self.np[i] == 0:
                self.rank[i] = 0
                temp.append(i)
        self.pareto.append(temp)  # 记录这个阶级的粒子
        # 3.循环计算每个阶级的粒子
        index = 0   # 代表当前循环的阶级数
        while len(self.pareto[index]) > 0:
            temp = []   # 处于同一个阶级的粒子列表
            for i in self.pareto[index]:    # 对于该阶级的第i个粒子,i为该粒子编号
                for j in self.sp[i]:    # 对于该阶级的第i个粒子的第j个支配粒子,j为该粒子编号
                    self.np[j] -= 1
                    if self.np[j] == 0:
                        self.rank[j] = index + 1    # 该粒子进入阶级
                        temp.append(j)
            self.pareto.append(temp)
            index += 1
        self.pareto.pop()

    def quick_sort_2(self):   # 快速非支配排序
        #   算出父子代粒子的np和sp
        self.np2 = numpy.zeros(len(self.XX2))
        self.sp2 = []
        self.pareto = []
        self.rank2 = numpy.zeros(len(self.XX2))
        # 1.计算出种群每个粒子的np和sp
        for i in range(len(self.XX2)):
            temp = []   # 存放第i个粒子支配的粒子集合(编号）
            for j in range(len(self.XX2)):
                if i != j:
                    if self.XX2[i][1][0] <= self.XX2[j][1][0] and self.XX2[i][1][1] <= self.XX2[j][1][1]:     # j被i支配
                        temp.append(j)
                    elif self.XX2[i][1][0] >= self.XX2[j][1][0] and self.XX2[i][1][1] >= self.XX2[j][1][1]:     # i被j支配
                        self.np2[i] += 1
            self.sp2.append(temp)
        # 2.算出第0阶级的粒子
        temp = []  # 处于同一个阶级的粒子列表
        for i in range(len(self.XX2)):
            if self.np2[i] == 0:
                self.rank2[i] = 0
                temp.append(i)
        self.pareto.append(temp)  # 记录这个阶级的粒子
        # 3.循环计算每个阶级的粒子
        index = 0   # 代表当前循环的阶级数
        while len(self.pareto[index]) > 0:
            temp = []   # 处于同一个阶级的粒子列表
            for i in self.pareto[index]:    # 对于该阶级的第i个粒子,i为该粒子编号
                for j in self.sp2[i]:    # 对于该阶级的第i个粒子的第j个支配粒子,j为该粒子编号
                    self.np2[j] -= 1
                    if self.np2[j] == 0:
                        self.rank2[j] = index + 1    # 该粒子进入阶级
                        temp.append(j)
            self.pareto.append(temp)
            index += 1
        self.pareto.pop()

    def select(self):   # 轮盘赌选择法,阶级越高的粒子越容易被选中，0为最高
        temp_xx = []
        p = numpy.zeros(len(self.XX))  # 存放各个粒子的被选中概率值
        q = numpy.zeros(len(self.XX) + 1)   # 存放各个粒子的被选中概率值的累计值
        total = 0     # 求选中概率时的分母
        for i in range(len(self.pareto)):
            total += len(self.pareto[i]) / (i + 1)
        for i in range(len(self.XX)):
            p[i] = (1 / (self.rank[i] + 1)) / total
        for i in range(len(self.XX) + 1):   # q[0]为0
            for j in range(i):  # 求q[i+1] = p[0]+...+p[i]
                q[i] += p[j]
        # 进行选择
        for i in range(len(self.XX)):   # i轮选择i个粒子入围
            random_x = random.random()
            for j in range(len(self.XX)):
                if q[j + 1] > random_x >= q[j]:
                    temp_x = copy.deepcopy(self.XX[j])
                    temp_xx.append(temp_x)
        self.XX_child = temp_xx


    def cross(self):
        for i in range(len(self.XX_child)):
            if self.cr > random.random():     # 进行交叉
                temp_index = random.randint(0, len(self.XX) - 1)    # 选择交叉的粒子
                if temp_index != i:
                    temp_x = random.random()
                    for j in range(len(self.X)):
                        x = temp_x * self.XX_child[temp_index][0][j] + (1 - temp_x) * self.XX_child[i][0][j]
                        self.XX_child[temp_index][0][j] = temp_x * self.XX_child[i][0][j] + (1 - temp_x) * self.XX_child[temp_index][0][j]
                        self.XX_child[i][0][j] = x
                # print(i, "交叉", temp_index)

    def mutation(self):
        for i in range(len(self.XX_child)):
            for j in range(len(self.X)):
                if self.mu > random.random():  # 进行变异
                    self.XX_child[i][0][j] = self.XX_child[i][0][j] - 0.1 + numpy.random.random() * 0.2
                    if self.XX_child[i][0][j] < 0:
                        self.XX_child[i][0][j] = 0  # 最小值0
                    if self.XX_child[i][0][j] > 1:
                        self.XX_child[i][0][j] = 1  # 最大值1

    def update_s_child(self):   # 更新子代粒子适应度值
        for i in range(len(self.XX_child)):
            self.XX_child[i][1] = self.cal_obj(self.XX_child[i][0])

    def be_the_one(self):   # 合并父子粒子
        self.XX2 = []
        for i in range(len(self.XX)):
            temp = copy.deepcopy(self.XX[i])
            self.XX2.append(temp)
            temp = copy.deepcopy(self.XX_child[i])
            self.XX2.append(temp)

    def crowd(self):    # 拥挤度计算
        #   根据函数1对粒子进行排序
        temp1 = []  # temp1存放函数1的适应度值以及对应粒子的编号
        for i in range(len(self.XX2)):
            temp1.append([self.XX2[i][1][0], i])
        temp1.sort(key=lambda x: x[0])
        temp2 = []  # temp2存放函数2的适应度值以及对应粒子的编号
        for i in range(len(self.XX2)):
            temp2.append([self.XX2[i][1][1], i])
        temp2.sort(key=lambda x: x[0])
        #   计算所有粒子的拥挤度
        for i in range(len(self.XX2)):
            self.XX2[i][2] = 0
        for i in range(len(self.XX2)):  # 计算粒子i的拥挤度
            for j in range(len(temp1)):
                if i == temp1[j][1]:
                    if j == 0 or j == len(self.XX2) - 1:     # 该粒子在边界
                        self.XX2[i][2] = float('inf')
                    else:
                        self.XX2[i][2] += (temp1[j+1][0] - temp1[j-1][0])
            for j in range(len(temp2)):
                if i == temp2[j][1]:
                    if j == 0 or j == len(self.XX2) - 1:     # 该粒子在边界
                        self.XX2[i][2] = float('inf')
                    else:
                        self.XX2[i][2] += (temp2[j+1][0] - temp2[j-1][0])
        # for i in range(len(self.XX2)):
        #     print(i, self.XX2[i])

    def choose(self):    # 选择新父代
        self.XX = []    # 初始化父代
        # 从pareto最优面的前几阶级塞粒子进父代，直到某一阶级的粒子不能全放入新的父代为止
        rank_temp = 0   # 现在进行遍历第rank_temp阶级的粒子
        while (len(self.XX_child) - len(self.XX)) > len(self.pareto[rank_temp]):
            for i in self.pareto[rank_temp]:
                temp = copy.deepcopy(self.XX2[i])
                self.XX.append(temp)
            rank_temp += 1
        # 当某一阶级的粒子不能全放入新父代时，选择拥挤度高的粒子放入新父代
        sort_temp = []  # 把要排序的粒子放入此处
        for i in self.pareto[rank_temp]:
            temp = copy.deepcopy(self.XX2[i])
            sort_temp.append(temp)
        sort_temp.sort(key=lambda x: x[2], reverse=True)
        for i in range(len(self.XX_child) - len(self.XX)):
            self.XX.append(sort_temp[i])
        # print("拥挤度排序")
        # for i in range(len(sort_temp)):
        #     print("该粒子编号", i, "适应度函数值为", sort_temp[i][1][0], sort_temp[i][1][1], "拥挤度值为", sort_temp[i][2])

    def draw(self):
        x = []  # 要绘制点的x坐标，即适应度函数1的值
        y = []
        for i in self.pareto[0]:
            x.append(self.XX[i][1][0])
            y.append(self.XX[i][1][1])
        ax = plt.subplot(111)
        plt.scatter(x, y)  # ,marker='+')#self.objectives[:][0],self.objectives[:][1]) #?
        # plt.plot(,'--',label='')
        plt.axis([0.0, 1.0, 0.0, 1.1])
        xmajorLocator = MultipleLocator(0.1)
        ymajorLocator = MultipleLocator(0.1)
        ax.xaxis.set_major_locator(xmajorLocator)
        ax.yaxis.set_major_locator(ymajorLocator)
        plt.xlabel('f1')
        plt.ylabel('f2')
        plt.title('Pareto Front')
        plt.grid()
        # plt.show()
        plt.savefig('Pareto Front 2.png')


if __name__ == '__main__':
    gen = 500
    particles = Particles()
    particles.init(100, 30)
    particles.quick_sort()
    while gen > 0:
        particles.select()
        particles.cross()
        particles.mutation()
        particles.update_s_child()
        particles.be_the_one()
        particles.quick_sort_2()
        particles.crowd()
        particles.choose()
        particles.quick_sort()
        # for i in range(len(particles.pareto)):
        #     print("正在输出第", i, "阶级")
        #     for j in particles.pareto[i]:
        #         print("该粒子编号", j, "适应度函数值为", particles.XX[j][1][0], particles.XX[j][1][1], "拥挤度值为", particles.XX[j][2])
        gen -= 1

    particles.draw()