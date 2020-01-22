import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from matplotlib.animation import FuncAnimation
from scipy import constants as si

Nbodies = 1


class Particle(object):
    number_of_particles = 0
    particle_color = []
    CubeSize = 5

    def __init__(self, pos, vel, accl, radius=0.85e-15):
        self.pos = pos
        self.vel = vel
        self.accl = accl
        self.radius = radius
        Particle.number_of_particles += 1

    def _r(self, P2):
        r = np.linalg.norm(self.pos - P2.pos)
        if r == 0:
            return 1e-3
        return r

    def r_hat(self, P2):
        return (self.pos - P2.pos) / self._r(P2)

    def force_columbs(self, P2):
        k = 1 / (4 * si.pi * si.epsilon_0)
        return self.charge * P2.charge * k * self.r_hat(P2) / (self._r(P2)) ** 2

    def force_magneto(self, P2):
        k = si.mu_0 / (4 * si.pi)
        B = np.cross(P2.vel, P2.r_hat(self)) / (P2._r(self)) ** 2
        return self.charge * np.cross(self.vel, B)

    def force(self, P2):
        return self.force_columbs(P2) + self.force_magneto(P2)

    def boundary_cross_check_and_update(self):  # checks to see if the x ,y z, positions are in the box of dimensionals maxsize
        if (self.pos[0] < 0 or self.pos[0] > Particle.CubeSize):
            self.pos[0] = Particle.CubeSize - (self.pos[0] % Particle.CubeSize)
            self.vel[0] = -self.vel[0]

        if (self.pos[1] < 0 or self.pos[1] > Particle.CubeSize):
            self.pos[1] = Particle.CubeSize - (self.pos[1] % Particle.CubeSize)
            self.vel[1] = -self.vel[1]

        if (self.pos[2] < 0 or self.pos[2] > Particle.CubeSize):
            self.pos[2] = Particle.CubeSize - (self.pos[2] % Particle.CubeSize)
            self.vel[2] = -self.vel[2]

    def update_pos(self, force, dt):
        self.accl = force / self.mass
        self.pos = self.pos + self.vel * dt + 1 / 2 * self.accl * dt ** 2
        self.vel = self.vel + self.accl * dt
        self.boundary_cross_check_and_update()
        if self.vel[0] > si.c or self.vel[1] > si.c or self.vel[2] > si.c:
            print("EINSTEIN WAS WRONG")


class Electron(Particle):
    def __init__(self, pos, vel, accl):
        super(Electron, self).__init__(pos, vel, accl)
        self.type = "Electron"
        self.mass = si.m_e
        self.charge = -1 * si.e
        self.color = "Blue"
        self.__class__.particle_color.append(self.color)

class Proton(Particle):
    def __init__(self, pos, vel, accl,):
        super(Proton, self).__init__(pos, vel, accl)
        self.type = "Proton"
        self.mass = si.m_p
        self.charge = si.e
        self.color = "Red"
        self.__class__.particle_color.append(self.color)






class Simulation():#AB
    def __init__(self, particles):
        self.particles = particles
        self.temp = particles

    def step(self, dt=.002):
        coords = None#AB
        sum_of_forces_final = []
        for p1 in self.particles:
            for p2 in self.particles:
                sum_of_forces = np.zeros_like(p1.pos)
                if p1 == p2:
                    continue
                sum_of_forces = sum_of_forces + p1.force(p2)
            p1.update_pos(sum_of_forces, dt)


            coords = np.vstack((coords, p1.pos)) if coords is not None else \
            p1.pos
        return coords


def main():
    particle_array=[]
    particle_array.append(Proton(np.array([Particle.CubeSize, Particle.CubeSize, Particle.CubeSize]), np.array([0, 1, 0]), np.array([0, 0, 0])))
    particle_array.append(Electron(np.array([0, 0, 0 ]), np.array([10, 10, 10]), np.array([0, 0, 0])))
    particle_array.append(Electron(np.array([0,  Particle.CubeSize/2, 0]), np.array([0, 0, 0]), np.array([0, 0, 0])))
    #Proton(np.array([Particle.CubeSize / 4, Particle.CubeSize / 4, Particle.CubeSize / 4]), np.array([1, 0, 1]), np.array([0, 0, 0]))

    s = Simulation(particle_array)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xticks([]), ax.set_yticks([]), ax.set_zticks([])
    plt.xlabel('x')
    plt.ylabel('y')

    ax.set_xlim3d(0, Particle.CubeSize)
    ax.set_ylim3d(0, Particle.CubeSize)
    ax.set_zlim3d(0, Particle.CubeSize)
    i = s.step()
    graph = ax.scatter(i[..., 0], i[..., 1], i[..., 2], '.', c=Particle.particle_color)

    def update(frame_number):
        i = s.step()
        graph._offsets3d = (i[..., 0], i[..., 1], i[..., 2])
        return graph,

    animation = FuncAnimation(fig, update, interval=1, blit=False)
    plt.show()


if __name__ == "__main__":
    main()
