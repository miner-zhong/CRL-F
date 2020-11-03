import numpy as np


class OGD:
    def __init__(self, dim, proj, eta=0.01, args=None):
        self.dim = dim
        self.proj = proj
        self.eta = eta

        if(args.pretrained_theta=='n/a'):
            self.thetas = [np.zeros(dim)]
        else:
            self.thetas = [np.zeros(dim)]

        self.velocities = [np.zeros(dim)]
        self.loss_v = []
        self.loss = []
        self.iter = 1.0
        self.mass = 0.9
        self.power = 0.9

    def reset(self):
        self.iter = 1.0

    def step_linear(self, loss_vector):
        theta = self.thetas[-1]
        loss = np.dot(theta, loss_vector)
        self.loss.append(loss)
        self.loss_v.append(loss_vector)

        update = (self.eta/np.sqrt(self.iter))*loss_vector

        # Momentum
        velocity = self.mass*self.velocities[-1] - (1.0-self.mass)*update

        theta_n = self.proj(theta + velocity)
        self.thetas.append(theta_n)
        self.velocities.append(velocity)
        self.iter += 1

    def get_theta(self):
        return self.thetas[-1]


