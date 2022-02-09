# Back-Propagation Neural Networks
#
import math
import random

random.seed(0)

# calculate a random number where:  a <= rand < b
def rand(a, b):
    return (b-a)*random.random() + a

# Make a matrix (we could use NumPy to speed this up)
def makeMatrix(I, J, fill=0.0):
    m = []
    for i in range(I):
        m.append([fill]*J)
    return m

def sigmoid(x):
    return x # math.tanh(x)

# derivative
def dsigmoid(y):
    return -1 # 1.0 - y**2

class NN:
    def __init__(self, ni, nh, no):
        # number of input, hidden, and output nodes
        self.ni = ni + 1 # +1 for bias node
        self.nh = nh
        self.no = no

        # activations for nodes
        self.ai = [0]*self.ni
        self.ah = [0]*self.nh
        self.ao = [0]*self.no

        # create weights
        self.wi = makeMatrix(self.ni, self.nh)
        self.wo = makeMatrix(self.nh, self.no)
        # set them to random vaules
        for i in range(self.ni):
            for j in range(self.nh):
                self.wi[i][j] = rand(-0.2, 0.2)
        for j in range(self.nh):
            for k in range(self.no):
                self.wo[j][k] = rand(-2.0, 2.0)

        # last change in weights for momentum
        self.ci = makeMatrix(self.ni, self.nh)
        self.co = makeMatrix(self.nh, self.no)

    def update(self, inputs):
        if len(inputs) != self.ni-1:
            raise ValueError('wrong number of inputs')

        temp_in = [0] * self.ni
        # input activations
        for i in range(self.ni-1):
            self.ai[i] += inputs[i]
            temp_in[i] = inputs[i]

        temp_hidden = [0] * self.nh
        # hidden activations
        for j in range(self.nh):
            sum = 0.0
            for i in range(self.ni):
                sum = sum + temp_in[i] * self.wi[i][j]
            self.ah[j] += sigmoid(sum)
            temp_hidden[j] = sigmoid(sum)

        temp_output = [0] * self.no
        # output activations
        for k in range(self.no):
            sum = 0.0
            for j in range(self.nh):
                sum = sum + temp_hidden[j] * self.wo[j][k]
            self.ao[k] += sigmoid(sum)
            temp_output[k] = sigmoid(sum)

        return temp_output


    def backPropagate(self, targets, N, M, samples):
        if len(targets) != self.no:
            raise ValueError('wrong number of target values')

        # calculate error terms for output
        output_deltas = [0.0] * self.no
        for k in range(self.no):
            error = (targets[k]-self.ao[k])**2/2*samples #MSE
            output_deltas[k] = dsigmoid(self.ao[k]/samples) * error

        # calculate error terms for hidden
        hidden_deltas = [0.0] * self.nh
        for j in range(self.nh):
            error = 0.0
            for k in range(self.no):
                error = error + output_deltas[k]*self.wo[j][k]
            hidden_deltas[j] = dsigmoid(self.ah[j]/samples) * error

        # update output weights
        for j in range(self.nh):
            for k in range(self.no):
                change = output_deltas[k]*(self.ah[j]/samples)
                self.wo[j][k] = self.wo[j][k] + N*change + M*self.co[j][k]
                self.co[j][k] = change
                #print N*change, M*self.co[j][k]

        # update input weights
        for i in range(self.ni):
            for j in range(self.nh):
                change = hidden_deltas[j]*(self.ai[i]/samples)
                self.wi[i][j] = self.wi[i][j] + N*change + M*self.ci[i][j]
                self.ci[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            error = error + (targets[k]-self.ao[k])

        #zeroes grads
        self.ao[0] = 0
        self.ah = [0]*len(self.ah)
        self.ai = [0]*len(self.ai)

        return error/samples


    def test(self, patterns):
        for p in patterns:
            print(p[0], '->', self.update(p[0]))

    def weights(self):
        print('Input weights:')
        for i in range(self.ni):
            print(self.wi[i])
        print()
        print('Output weights:')
        for j in range(self.nh):
            print(self.wo[j])

    def train(self, patterns, iterations=1000, N=0.5, M=0.1, batches=2):
        # N: learning rate
        # M: momentum factor
        for i in range(iterations):
            error = 0.0
            targets = [0] * len(patterns[0][1])
            for p in random.sample(patterns,batches):
                inputs = p[0]
                for j in range(len(p[1])):
                    targets[j] += p[1][j]
                self.update(inputs)
            error = error + self.backPropagate(targets, N, M, batches)
            if i % 1 == 0:
                print('error %-.5f' % error)


def demo():
    # Teach network addition function
    pat = [
        [[0,0], [0]],
        [[0,1], [1]],
        [[1,0], [1]],
        [[1,1], [2]],
        [[3,2], [5]]
    ]
    pat2 = [
        [[0,0], [0]],
        [[3,1], [4]],
        [[1,2], [3]],
        [[2,1], [3]]
    ]

    # create a network with two input, two hidden, and one output nodes
    n = NN(2, 3, 1)
    # train it with some patterns
    n.train(pat, 1000, 0.00005, 0, 5)
    # test it
    n.test(pat2)



if __name__ == '__main__':
    demo()