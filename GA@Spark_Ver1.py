import random
import sys
import time

from pyspark import SparkConf
from pyspark import SparkContext
conf = SparkConf().setMaster("local").setAppName("OneMaxPython")
sc = SparkContext(conf = conf)
# Import Parts
#-----------------------------------------------
# Functions Parts

def MakePop(pop_size, ind_size): 
    # Create Populations for initialization
    lisPop = []
    for i in range(pop_size):
        lisPop.append([[random.randint(0,1) for i in range(ind_size)], 0])
    return lisPop

def EvaForEachInd(ele):
    # Fitness
    ele[1] = sum(ele[0])/len(ele[0])
    return ele

#选择精英者和剩下的群体
def Select(fitRDD, POPULATION_SIZE, ELITE_SIZE):
# 对评价过的种群按适应度从大到小进行排序
    sortedPopRDD = fitRDD.sortBy((lambda x: x[1]), False)
# 取出精英并创建精英RDD
    lisElite = sortedPopRDD.take(ELITE_SIZE)
    eliteRDD = sc.parallelize(lisElite)
# 取出剩下的种群
    RemainPop = fitRDD.sortBy(lambda x: x[1]).take(POPULATION_SIZE - ELITE_SIZE)
    random.shuffle(RemainPop)
    RemainPopRDD = sc.parallelize(RemainPop)
    return eliteRDD, RemainPopRDD

# 交叉

def CROSSOVER_BAK(ele): # RDD层面的交叉操作
    def crossover(a_chr, b_chr): # 两个个体交叉操作
        size = len(a_chr) # 取出染色体的长度

        f = random.randint(0, size) # 选取两个基因点，准备交叉
        s = random.randint(f, size)

        _a = a_chr[:f] + b_chr[f:s] + a_chr[s:]
        _b = b_chr[:f] + a_chr[f:s] + b_chr[s:]
    
        return _a, _b

    a_Chromo, b_Chromo = crossover(ele[0][0],ele[1][0])
    ele[0][0] = a_Chromo
    ele[0][1] = 0
    ele[1][0] = b_Chromo
    ele[1][1] = 0
    return ele[0],ele[1]

def CROSSOVER(ele): # RDD层面的交叉操作
    
    a_Chr = ele[0][0][0]
    b_chr = ele[0][1][0]
    size = len(a_chr) # 取出染色体的长度

    f = random.randint(0, size) # 选取两个基因点，准备交叉
    s = random.randint(f, size)

    _a = a_chr[:f] + b_chr[f:s] + a_chr[s:]
    _b = b_chr[:f] + a_chr[f:s] + b_chr[s:]
    
    ele[0][0][0] = _a
    ele[0][1][0] = _b     
    ele[0][0][1] = 0
    ele[0][1][1] = 0
    return ele[0][0],ele[0][1]

# 变异部分
           
def Mutation(ele):  # 选择变异的个体，RDD 层面    
    def MutationForInd(gene): # 基因变异
        global GENE_MUTATION
        for i in gene:
            if GENE_MUTATION > (random.randint(0, 100) / 100):
                i = random.randint(0,1)
        return gene

    global INDIVIDUAL_MUTATION
    if INDIVIDUAL_MUTATION > (random.randint(0, 100) / 100):
        ele[0] = MutationForInd(ele[0])
    return ele


#---------------------------------------------
#        MAIN()
#---------------------------------------------

random.seed(64) # 随机种子设置
#---------------------------------------
# Constant Variables
CHROMOSOME_SIZE = 10 # 染色体尺寸
GENE_MUTATION = 0.05 # 基因变异率
INDIVIDUAL_MUTATION = 0.2 # 个体变异率
CROSSOVER = 0.5
POPULATION_SIZE = 40 # 种群数量
ELITE_PERCENTAGE = 0.5
#ELITE_SIZE = int(POPULATION_SIZE * ELITE_PERCENTAGE)
ELITE_SIZE = 2 
GENERATION_MAX = 100 # 最大迭代次数
#------------------------------------------------
# starts
start = time.clock() # 开始计时
population = MakePop(POPULATION_SIZE, CHROMOSOME_SIZE) # initial population
popRDD = sc.parallelize(population) 
fitRDD = popRDD.map(EvaForEachInd)
fitValues = [ele[1] for ele in fitRDD.collect()]
#--------------------------------------------------
# Fitness statistics 
print("{0} Generation ---".format(1))
print("\tMIN: {0}".format(min(fitValues)))
print("\tMAX: {0}".format(max(fitValues)))
print("\tAVG: {0}".format(round(sum(fitValues) / len(fitValues), 3)), "\n")

# ------------------------------------------------------
# select elites and remained populations
eliteRDD, RemainPopRDD = Select(fitRDD, POPULATION_SIZE, ELITE_SIZE)
#------------------------------------------------
# crossover
RemainPopList = RemainPopRDD.collect()
PairRDD = sc.parallelize(RemainPopList, int(len(RemainPopList)/2)).glom()
print(PairRDD.take(1))
CrossedRDD = PairRDD.flatMap(CROSSOVER)
CrossedList = CrossedRDD.collect()
random.shuffle(CrossedList)
CrossedRDD = sc.parallelize(CrossedList)
#--------------------------------------------------
# mutations
MutatedRDD = CrossedRDD.map(Mutation)
# -------------------------------------------------
# get new generations
MutatedList = MutatedRDD.collect()
eliteList = eliteRDD.collect()
population = MutatedList + eliteList
# new generation population RDD
popRDD = sc.parallelize(population)

print(popRDD.collect())






