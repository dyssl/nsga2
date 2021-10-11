import numpy
import random
import copy


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
        self.cr = 0.1   # 交叉概率
        self.mu = 0.1   # 变异概率

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

    @staticmethod
    def cal_obj(x):  # 计算一个个体的多目标函数值 f1,f2 最小值,函数值越小视为越优
        f1 = x[0]
        f2 = 0
        for i in range(len(x)):
            f2 += x[i]
        f2 /= len(x)
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
                    if self.XX[i][1][0] < self.XX[j][1][0] and self.XX[i][1][1] < self.XX[j][1][1]:     # j被i支配
                        temp.append(j)
                    elif self.XX[i][1][0] > self.XX[j][1][0] and self.XX[i][1][1] > self.XX[j][1][1]:     # i被j支配
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
                    if self.XX2[i][1][0] < self.XX2[j][1][0] and self.XX2[i][1][1] < self.XX2[j][1][1]:     # j被i支配
                        temp.append(j)
                    elif self.XX2[i][1][0] > self.XX2[j][1][0] and self.XX2[i][1][1] > self.XX2[j][1][1]:     # i被j支配
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
                        self.XX_child[i][0][j] = temp_x * self.XX_child[temp_index][0][j] + (1 - temp_x) * self.XX_child[i][0][j]
                        self.XX_child[temp_index][0][j] = temp_x * self.XX_child[i][0][j] + (1 - temp_x) * self.XX_child[temp_index][0][j]
                # print(i, "交叉", temp_index)

    def mutation(self):
        for i in range(len(self.XX_child)):
            if self.mu > random.random():     # 进行变异
                temp_index = random.randint(0, (len(self.X) - 1))     # 选择变异的位置
                self.XX_child[i][0][temp_index] = random.random()
                # print(i, "变异")

    def update_s_child(self):   # 更新子代粒子适应度值
        for i in range(len(self.XX_child)):
            self.XX_child[i][1] = self.cal_obj(self.XX_child[i][0])

    def be_the_one(self):   # 合并父子粒子
        for i in range(len(self.XX)):
            self.XX2.append(self.XX[i])
            self.XX2.append(self.XX_child[i])

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
        print(temp1)
        print(temp2)
        #   计算所有粒子的拥挤度
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
                        print(i, "2")
                    else:
                        self.XX2[i][2] += (temp2[j+1][0] - temp2[j-1][0])
        for i in range(len(self.XX2)):
            print(i, self.XX2[i])

    def choose(self):    # 选择新父代
        pass


if __name__ == '__main__':
    gen = 1
    particles = Particles()
    particles.init(30, 5)
    particles.quick_sort()
    # for i in range(len(particles.pareto)):
    #     print("正在输出第", i, "阶级")
    #     for j in particles.pareto[i]:
    #         print("该粒子编号", j, "适应度函数值为", particles.XX[j][1][0], particles.XX[j][1][1])
    while gen > 0:
        particles.select()
        particles.cross()
        particles.mutation()
        particles.update_s_child()
        particles.be_the_one()
        particles.quick_sort_2()
        particles.crowd()
        particles.choose()
        gen -= 1

