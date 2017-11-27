import random
import time

class Individual:
    def __init__(self, chromosome, fitness=None):
        # 初始化函数
        self.chromosome = chromosome
        self.fitness = fitness

    def mutate(self, mutation): # 基因突变函数，参数 mutation 是基因变异概率
        for i in self.chromosome: # 遍历每个遗传因子，准备变异
            if mutation > (random.randint(0, 100) / 100): # 大于变异概率就进行变异
                self.chromosome[i] = random.randint(0, 1)

    def fit(self):
        self.fitness = sum(self.chromosome) / len(self.chromosome) # 适应值为1的百分比，最好为100%

def create_individual(chr_size): # 创建个体, 返回一个个体类
    return Individual([random.randint(0, 1) for i in range(chr_size)])

def select_individual(population, elite_size): # 选择个体,参数为种群和精英个数，返回精英和种群两个集合
    population = sorted(population, reverse=True, key=lambda i: i.fitness)     # 根据个体的适应度进行排序
    elite = [population.pop(0) for i in range(elite_size)] # 根据精英个数从种群中选出精英，剩下小种群
    random.shuffle(population) # 重新打乱种群的顺序
    return elite, population

def crossover(a, b): # 两个个体交叉操作
    # Two-point crossover
    a_chr = a.chromosome # 取出a的染色体
    b_chr = b.chromosome # 取出b的染色体
    
    size = len(a_chr) # 取出染色体的长度

    f = random.randint(0, size) # 选取两个基因点，准备交叉
    s = random.randint(f, size)

    _a = a_chr[:f] + b_chr[f:s] + a_chr[s:]
    _b = b_chr[:f] + a_chr[f:s] + b_chr[s:]
# 交叉完毕，返回两个新的个体
    return [Individual(_a), Individual(_b)]

def mutate(population, individual_mutation, gene_mutation): # 变异函数，参数为种群，个体变异率，gene_mutation
    for i in range(len(population)): # 执行种群个体数量的次数
        if individual_mutation > (random.randint(0, 100) / 100): # 当个体变异率大于一定值时，变异该个体
            population[i].mutate(gene_mutation) # 调用该个体的变异函数
    return population # 返回种群

random.seed(64) # 随机种子设置

CHROMOSOME_SIZE = 1000 # 染色体尺寸
GENE_MUTATION = 0.05 # 基因变异率
INDIVIDUAL_MUTATION = 0.2 # 个体变异率
CROSSOVER = 0.5
POPULATION_SIZE = 300 # 种群数量
ELITE_PERCENTAGE = 0.5
#ELITE_SIZE = int(POPULATION_SIZE * ELITE_PERCENTAGE)
ELITE_SIZE = 10
GENERATION_MAX = 10000 # 最大迭代次数


if __name__ == '__main__':
    
    start = time.clock() # 开始计时

    # Population of the first generation
    population = [create_individual(CHROMOSOME_SIZE) for i in range(POPULATION_SIZE)] # 创建初始种群
    
    # Start
    print("-- START Evaluation! --\n")
    
    for generation in range(GENERATION_MAX): # 执行设定的迭代次数

        # Fit 计算每个个体的适应度
        [population[i].fit() for i in range(POPULATION_SIZE)]
        
        # Result 提取出所有的适应度
        result = [p.fitness for p in population]
        print("{0} Generation ---".format(generation + 1))
        print("\tMIN: {0}".format(min(result)))
        print("\tMAX: {0}".format(max(result)))
        print("\tAVG: {0}".format(round(sum(result) / len(result), 3)), "\n")

        # Select 选择出精英者，和剩下的种群
        selected_individual, population = select_individual(population, ELITE_SIZE)
        
        # Crossover 
        crossover_individual = []
        for i in range(len(population)-1): # 遍历所有的剩余种群
            crossover_individual.extend(crossover(population[i], population[i+1])) # 进行两两交叉，得出新的子代
        random.shuffle(crossover_individual) # 进行混洗，打乱子代

        # Mutate, Generate offspring
        offspring = mutate(crossover_individual[:POPULATION_SIZE-ELITE_SIZE], INDIVIDUAL_MUTATION, GENE_MUTATION)
        offspring.extend(selected_individual)

        # Update population
        population = offspring

    print("-"*30, "\nResult:")
    print(selected_individual[0].chromosome)
    
    elapsed = (time.clock() - start)
    print("Time used:%d Seconds",elapsed)
    m, s = divmod(elapsed, 60)
    h, m = divmod(m, 60)
    print ("Transfer to Hour&Min&Sec is : %02d:%02d:%02d" % (h, m, s))
