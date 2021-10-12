import random
import numpy as np
from matplotlib.ticker import MultipleLocator
import matplotlib.pyplot as plt


class NSGA2():
    def __init__(self, dim, pop, max_iter):  # 维度，群体数量，迭代次数
        self.pc = 0.4  # 交叉概率
        self.pm = 0.4  # 变异概率
        self.dim = dim  # 搜索维度
        self.pop = pop  # 粒子数量
        self.max_iter = max_iter  # 迭代次数
        self.population = []  # 父代种群
        self.new_popu = []  # 选择算子操作过后的新种群
        # self.children = []                     #子代种群
        self.popu_child = []  # 合并后的父代与子代种群
        self.fronts = []  # Pareto前沿面
        self.rank = []  # np.zeros(self.pop)       #非支配排序等级
        self.crowding_distance = []  # 个体拥挤度
        self.objectives = []  # 目标函数值,pop行 2列
        self.set = []  # 个体 i的支配解集
        self.np = []  # 该个体被支配的数目

    def init_Population(self):  # 初始化种群
        self.population = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            for j in range(self.dim):
                self.population[i][j] = random.random()

    def children_parent(self):  # 父代种群和子代种群合并,pop*2
        self.popu_child = np.zeros((2 * self.pop, self.dim))  # self.population
        for i in range(self.pop):
            for j in range(self.dim):
                self.popu_child[i][j] = self.population[i][j]
                self.popu_child[i + self.pop][j] = self.new_popu[i][j]

    def select_newparent(self):  # 根据排序和拥挤度计算，选取新的父代种群 pop*2 到 pop*1
        # self.non_donminate2()
        # self.crowd_distance()
        self.population = np.zeros((self.pop, self.dim))  # 选取新的种群
        a = len(self.fronts[0])  # Pareto前沿面第一层 个体的个数
        if a >= self.pop:
            for i in range(self.pop):
                self.population[i] = self.popu_child[self.fronts[0][i]]
        else:
            d = []  # 用于存放前b层个体
            i = 1
            while a < self.pop:
                c = a  # 新种群内 已经存放的个体数目    *列
                a += len(self.fronts[i])
                for j in range(len(self.fronts[i - 1])):
                    d.append(self.fronts[i - 1][j])
                    # while d < self.dim:
                    # self.population[j][d] = self.popu_child[self.fronts[i-1][j]][d]
                    # d += 1
                b = i  # 第b层不能放，超过种群数目了    *行
                i = i + 1
            # 把前c个放进去
            for j in range(c):
                self.population[j] = self.popu_child[d[j]]
            temp = np.zeros((len(self.fronts[b]), 2))  # 存放拥挤度和个体序号
            for i in range(len(self.fronts[b])):
                temp[i][0] = self.crowding_distance[self.fronts[b][i]]
                temp[i][1] = self.fronts[b][i]
            temp = sorted(temp.tolist())  # 拥挤距离由小到大排序
            for i in range(self.pop - c):
                self.population[c + i] = self.popu_child[int(temp[len(temp) - i - 1][1])]
                # 按拥挤距离由大到小填充直到种群数量达到 pop

    def cal_obj(self, position):  # 计算一个个体的多目标函数值 f1,f2 最小值
        f1 = position[0]
        f = 0
        for i in range(self.dim - 1):
            f += 9 * (position[i + 1] / (self.dim - 1))
        g = 1 + f
        f2 = g * (1 - np.square(f1 / g))
        return [f1, f2]

    def non_donminate2(self):  # pop*2行
        self.fronts = []  # Pareto前沿面
        self.fronts.append([])
        self.set = []
        self.objectives = []  # np.zeros((2*self.pop,2))
        self.np = np.zeros(2 * self.pop)
        self.rank = np.zeros(2 * self.pop)
        position = []
        for i in range(2 * self.pop):  # 越界处理
            for j in range(self.dim):
                if self.popu_child[i][j] < 0:
                    self.popu_child[i][j] = 0  # 最小值0
                if self.popu_child[i][j] > 1:
                    self.popu_child[i][j] = 1  # 最大值1
        for i in range(2 * self.pop):
            position = self.popu_child[i]
            # self.cal_obj(position)
            self.objectives.append(self.cal_obj(position))  # [i][0] = f1          #将 f1,f2赋到目标函数值矩阵里
            # self.objectives[i][1] = f2
        for i in range(2 * self.pop):
            temp = []
            for j in range(2 * self.pop):
                # temp=[]
                if j != i:
                    if self.objectives[i][0] >= self.objectives[j][0] and self.objectives[i][1] >= self.objectives[j][1]:
                        self.np[i] += 1  # j支配 i，np+1
                    if self.objectives[j][0] >= self.objectives[i][0] and self.objectives[j][1] >= self.objectives[i][
                        1]:
                        temp.append(j)
            self.set.append(temp)  # i支配 j，将 j 加入 i 的支配解集里
            if self.np[i] == 0:
                self.fronts[0].append(i)  # 个体序号
                self.rank[i] = 1  # Pareto前沿面 第一层级
        i = 0
        while len(self.fronts[i]) > 0:
            temp = []
            for j in range(len(self.fronts[i])):
                a = 0
                while a < len(self.set[self.fronts[i][j]]):
                    self.np[self.set[self.fronts[i][j]][a]] -= 1
                    if self.np[self.set[self.fronts[i][j]][a]] == 0:
                        self.rank[self.set[self.fronts[i][j]][a]] = i + 2  # 第二层级
                        temp.append(self.set[self.fronts[i][j]][a])
                    a = a + 1
            i = i + 1
            self.fronts.append(temp)

    def non_donminate1(self):  # pop行 快速非支配排序
        self.fronts = []  # Pareto前沿面
        self.fronts.append([])
        self.set = []
        self.objectives = []  # np.zeros((self.pop,2))
        self.np = np.zeros(self.pop)
        self.rank = np.zeros(self.pop)
        position = []
        for i in range(self.pop):  # 越界处理
            for j in range(self.dim):
                if self.population[i][j] < 0:
                    self.population[i][j] = 0  # 最小值0
                if self.population[i][j] > 1:
                    self.population[i][j] = 1  # 最大值1
        for i in range(self.pop):
            position = self.population[i]
            # self.cal_obj(position)
            self.objectives.append(self.cal_obj(position))  # [i][0] = f1          #将 f1,f2赋到目标函数值矩阵里
            # self.objectives[i][1] = f2
        for i in range(self.pop):
            temp = []
            for j in range(self.pop):
                # temp=[]
                if j != i:
                    if self.objectives[i][0] >= self.objectives[j][0] and self.objectives[i][1] >= self.objectives[j][
                        1]:
                        self.np[i] += 1  # j支配 i，np+1
                    if self.objectives[j][0] >= self.objectives[i][0] and self.objectives[j][1] >= self.objectives[i][
                        1]:
                        temp.append(j)
            self.set.append(temp)  # i支配 j，将 j 加入 i 的支配解集里
            if self.np[i] == 0:
                self.fronts[0].append(i)  # 个体序号
                self.rank[i] = 1  # Pareto前沿面 第一层级
        i = 0
        while len(self.fronts[i]) > 0:
            temp = []
            for j in range(len(self.fronts[i])):
                a = 0
                while a < len(self.set[self.fronts[i][j]]):
                    self.np[self.set[self.fronts[i][j]][a]] -= 1
                    if self.np[self.set[self.fronts[i][j]][a]] == 0:
                        self.rank[self.set[self.fronts[i][j]][a]] = i + 2  # 第二层级
                        temp.append(self.set[self.fronts[i][j]][a])
                    a = a + 1
            i = i + 1
            self.fronts.append(temp)

    def selection(self):  # 轮盘赌选择
        self.non_donminate1()  # 非支配排序,获得Pareto前沿面
        pi = np.zeros(self.pop)  # 个体的概率
        qi = np.zeros(self.pop + 1)  # 个体的累积概率
        P = 0
        for i in range(len(self.fronts)):
            # for j in range(len(self.fronts[i])):
            P += (1 / (i + 1)) * (len(self.fronts[i]))  # 累积适应度
        for i in range(len(self.fronts)):
            for j in range(len(self.fronts[i])):
                pi[self.fronts[i][j]] = (1 / (i + 1)) / P  # 个体遗传到下一代的概率，层数越低越容易被选中
        for i in range(self.pop):
            qi[0] = 0
            qi[i + 1] = np.sum(pi[0:i + 1])  # 累积概率
        self.new_popu = np.zeros((self.pop, self.dim))
        for i in range(self.pop):
            r = random.random()  # 生成随机数，
            a = 0
            for j in range(self.pop):
                if r > qi[j] and r < qi[j + 1]:
                    while a < self.dim:
                        self.new_popu[i][a] = self.population[j][a]     # 深拷贝
                        a += 1
                j += 1

    def crossover(self):  # 交叉,SBX交叉
        for i in range(self.pop - 1):
            # temp1 = []
            # temp2 = []
            if random.random() < self.pc:
                # pc_point = random.randint(0,self.dim-1)        #生成交叉点
                # temp1.append(self.population[i][pc_point:self.dim])
                # temp2.append(self.population[i+1][pc_point:self.dim])
                # self.population[i][pc_point:self.dim] = temp2
                # self.population[i+1][pc_point:self.dim] = temp1
                a = random.random()
                for j in range(self.dim):
                    self.new_popu[i][j] = a * self.new_popu[i][j] + (1 - a) * self.new_popu[i + 1][j]
                    self.new_popu[i + 1][j] = a * self.new_popu[i + 1][j] + (1 - a) * self.new_popu[i][j]
            i += 2

    def mutation(self):  # 变异
        for i in range(self.pop):
            for j in range(self.dim):
                if random.random() < self.pm:
                    self.new_popu[i][j] = self.new_popu[i][j] - 0.1 + np.random.random() * 0.2
                    if self.new_popu[i][j] < 0:
                        self.new_popu[i][j] = 0  # 最小值0
                    if self.new_popu[i][j] > 1:
                        self.new_popu[i][j] = 1  # 最大值1

    def crowd_distance(self):  # 拥挤度计算，前沿面每个个体的拥挤度
        self.crowding_distance = np.zeros(2 * self.pop)
        for i in range(len(self.fronts) - 1):  # fronts最后一行为空集
            temp1 = np.zeros((len(self.fronts[i]), 2))
            temp2 = np.zeros((len(self.fronts[i]), 2))
            for j in range(len(self.fronts[i])):
                temp1[j][0] = self.objectives[self.fronts[i][j]][0]  # f1赋值
                temp1[j][1] = self.fronts[i][j]
                temp2[j][0] = self.objectives[self.fronts[i][j]][1]  # f2赋值
                temp2[j][1] = self.fronts[i][j]
            # temp3 = temp1.tolist()
            # temp4 = temp2.tolist()
            temp1 = sorted(temp1.tolist())  # f1排序,按照适应度的值
            temp2 = sorted(temp2.tolist())  # f2排序
            self.crowding_distance[int(temp1[0][1])] = float('inf')
            self.crowding_distance[int(temp1[len(self.fronts[i]) - 1][1])] = float('inf')
            f1_min = temp1[0][0]
            f1_max = temp1[len(self.fronts[i]) - 1][0]
            f2_max = temp2[len(self.fronts[i]) - 1][0]
            f2_min = temp2[0][0]
            a = 1
            while a < len(self.fronts[i]) - 1:
                self.crowding_distance[int(temp1[a][1])] = (temp1[a + 1][0] - temp1[a - 1][0]) / (f1_max - f1_min) + (
                            temp2[a + 1][0] - temp2[a - 1][0]) / (f2_max - f2_min)  # 个体i的拥挤度等于 f1 + f2
                a += 1

    def draw(self):  # 画图
        self.objectives = []  # np.zeros((self.pop,2))
        position = []
        for i in range(self.pop):  # 越界处理
            for j in range(self.dim):
                if self.population[i][j] < 0:
                    self.population[i][j] = 0  # 最小值0
                if self.population[i][j] > 1:
                    self.population[i][j] = 1  # 最大值1
        self.non_donminate1()
        for i in range(len(self.fronts[0])):
            position = self.population[self.fronts[0][i]]
            self.objectives.append(self.cal_obj(position))
        # for i in range(self.pop):
        # position = self.population[i]
        # self.objectives.append(self.cal_obj(position))#[i][0] = f1          #将 f1,f2赋到目标函数值矩阵里
        x = []
        y = []
        for i in range(self.pop):
            x.append(self.objectives[i][0])
            y.append(self.objectives[i][1])
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
        plt.title('ZDT2 Pareto Front')
        plt.grid()
        # plt.show()
        plt.savefig('ZDT2 Pareto Front 2.png')

    def run(self):  # 主程序
        self.init_Population()  # 初始化种群，选择交叉变异，生成子代种群
        # self.selection()
        # self.crossover()
        # self.mutation()
        for i in range(self.max_iter):
            self.selection()
            self.crossover()
            self.mutation()
            self.children_parent()  # 父代与子代种群合并，快速非支配排序和拥挤度计算
            self.non_donminate2()
            self.crowd_distance()
            self.select_newparent()  # 根据Pareto等级和拥挤度选取新的父代种群，选择交叉变异
            self.non_donminate1()
            for j in range(len(self.fronts)):
                print("阶级", j)
                for i in range(len(self.fronts[j])):
                    print(self.cal_obj(self.population[self.fronts[j][i]]))
        self.draw()
        # print(self.fronts)
        # print(self.population)
        # print(self.new_popu)
        # print(self.popu_child)
        # print(self.objectives)
        # print()


def main():
    NSGA = NSGA2(30, 100, 500)
    NSGA.run()


if __name__ == '__main__':
    main()
