import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import constants as si
from random import seed
import json
from random import randint

with open('planets.json') as f:
  data = json.load(f)


v_0 = np.array([0,0,0]) # probably best to keep them all at rest from the start
a_0 = np.zeros_like(v_0) # starting acceleration is kinda useless but needed non the less

class Body(object):
    number_of_bodys = 0
    body_color = []

    def __init__(self, pos, vel, accl, mass, color,radius=0.85e-15):
        self.pos = pos
        self.vel = vel
        self.accl = accl
        self.mass = mass
        self.radius = radius
        self.__class__.body_color.append(color)
        Body.number_of_bodys += 1

    def _r(self, P2):
        r = np.linalg.norm(self.pos - P2.pos)
        if r <= 8.7e-16: # Using the mass of the proton to simulate a collision to avoid a dividing by zero
            return 8.7e-16
        return r

    def r_hat(self, P2):
        return (self.pos - P2.pos) / self._r(P2)

    def force(self, P2):
        return -self.mass * P2.mass * si.G * 1e-9 * self.r_hat(P2) / (self._r(P2)) ** 2

    def update_accl(self,force):
        self.accl = force / self.mass

    def update_pos(self, dt):
        self.pos = self.pos + self.vel * dt + 1 / 2 * self.accl * dt ** 2
        self.vel = self.vel + self.accl * dt



class Simulation():
    def __init__(self, bodys):
        self.bodys = bodys

    def step(self, dt=1):
        coords = None
        for p1 in self.bodys:
            sum_of_forces = np.zeros_like(p1.pos)
            for p2 in self.bodys:

                if p1 == p2:
                    continue
                sum_of_forces = np.add(sum_of_forces, p1.force(p2))
            p1.update_accl(sum_of_forces)

        for p1 in self.bodys:
            p1.update_pos(dt)


            coords = np.vstack((coords, p1.pos)) if coords is not None else \
            p1.pos
        return coords


def main():
    body_array=[]
    shift = data[-1]['distanceFromSun']
    body_array.append(Body(np.array([0, 0, 0]), v_0, a_0, 1.989e21, "red", 696340e-9))
    for k in range(len(data)):
        vx = randint(0, 2)
        vy = randint(0,2)

        r = (data[k]['distanceFromSun'])
        x = randint(0,int(r))
        y= np.sqrt(r**2-x**2)
        pos = np.array([x ,y, 0])
        mass = data[k]['mass']*1e15
        radius = data[k]['diameter'] * 1e-6 / 2
        body_array.append(Body(pos,np.array([vx,vy,0]), a_0, mass, "blue",radius))

    s = Simulation(body_array)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    plt.xlabel('x')
    plt.ylabel('y')

#    ax.set_xlim3d(0, CubeSize)
#    ax.set_ylim3d(0, CubeSize)
#    ax.set_zlim3d(0, CubeSize)
    i = s.step()
    graph = ax.scatter(i[..., 0], i[..., 1], i[..., 2], '.', c = Body.body_color)

    def update(frame_number):
        i = s.step()
        graph._offsets3d = (i[..., 0], i[..., 1], i[..., 2])
        return graph,

    animation = FuncAnimation(fig, update, interval=1, blit=False)
    plt.show()


if __name__ == "__main__":
    main()