import numpy.random
from matplotlib import pyplot as plt
from numpy import random
def fixed_param_sim():
     x1 = random.poisson(lam = 1, size = 1000)
     x2 = random.poisson(lam = 2, size = 1000)
     x3 = random.poisson(lam = 5, size = 1000)
     x4 = random.poisson(lam = 10, size = 1000)
     plt.hist(x1)
     plt.title('Lambda 1')
     plt.show()
     plt.hist(x2)
     plt.title('Lambda 2')
     plt.show()
     plt.hist(x3)
     plt.title('Lambda 5')
     plt.show()
     plt.hist(x4)
     plt.title('Lambda 10')
     plt.show()

def random_param_sim():
    x = []
    for i in range(1000):
        params = [1, 2, 5, 10]
        paramin = random.randint(0, 3)
        param = params[paramin]
        x.append(param)
    x1 = numpy.random.poisson(lam=x, size=2)
    plt.hist(x1)
    plt.title('Randomized Lambda')
    plt.show()

fixed_param_sim()
random_param_sim()


