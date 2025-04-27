import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

class ParticleFilter:
    def __init__(self, num_particles, init_state, init_cov, smooth_alpha_mean=0.9, smooth_alpha_cov=0.75):
        self.num_particles = num_particles
        self.particles = np.random.multivariate_normal(init_state, init_cov, num_particles)
        self.weights = np.ones(num_particles) / num_particles
        self.state_estimates = np.array([init_state])
        
        # smoothing parameters
        self.smooth_alpha_mean = smooth_alpha_mean
        self.smooth_alpha_cov = smooth_alpha_cov
        self.smoothed_state = np.array(init_state, dtype=float)
        self.smoothed_cov   = init_cov.copy()
        self.smoothed_state_estimates = [self.smoothed_state.copy()]

        #state derivatives
        self.prev_state = np.array([init_state])
        self.prev_cov   = np.zeros((2, 2))
        self.velocity = np.array([0.0, 0.0])
        self.dt = 1.0
        

    def predict(self, motion_model, dt=1.0):
        self.particles = np.array([motion_model(p, dt) for p in self.particles])
        
    def update(self, measurement, sigma, sensor_type='cam'):
        if measurement is None or (sensor_type == 'RF' and np.allclose(measurement, [0, 0])):
            return
        inv_sigma = np.linalg.inv(sigma)
        diff = self.particles - measurement
        exponents = -0.5 * np.sum(diff @ inv_sigma * diff, axis=1)
        likelihood = np.exp(exponents - exponents.max())
        self.weights *= likelihood
        wsum = self.weights.sum()
        if wsum == 0:
            self.weights.fill(1.0 / self.num_particles)
        else:
            self.weights /= wsum

    def neff(self):
        return 1.0 / np.sum(self.weights**2)

    def resample(self):
        positions = (np.arange(self.num_particles) + np.random.rand()) / self.num_particles
        cumsum = np.cumsum(self.weights)
        idx = np.searchsorted(cumsum, positions)
        self.particles = self.particles[idx]
        self.weights.fill(1.0 / self.num_particles)

    def estimate(self):
        mean = np.average(self.particles, weights=self.weights, axis=0)
        diff = self.particles - mean
        raw_cov = np.cov(diff.T, aweights=self.weights)

        self.state_estimates = np.vstack((self.state_estimates, mean))
        
        # exponential smoothing 
        self.smoothed_state = (
            self.smooth_alpha_mean * self.smoothed_state +
            (1 - self.smooth_alpha_mean) * mean
        )
        # exponential smoothing on the covariance
        self.smoothed_cov = (
            self.smooth_alpha_cov*self.smoothed_cov
            + (1-self.smooth_alpha_cov)*raw_cov
        )

        # update state derivative and the previous state and covariance
        self.velocity = ((self.smoothed_state - self.prev_state) / self.dt).reshape(2)
        self.prev_state = self.smoothed_state.copy()
        self.prev_cov   = self.smoothed_cov.copy()

        # store for plotting
        self.smoothed_state_estimates.append(self.smoothed_state.copy())
        
        return mean, raw_cov

    def plot(self, ax):
        ax.clear()
        ax.set_title('Particle Filter State Estimate (Smoothed)')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_xlim(-4, 8)
        ax.set_ylim(-1, 6)

        # draw particles
        ax.scatter(self.particles[:, 0], self.particles[:, 1],
                   color='gray', alpha=0.2, s=2)

        mean, cov = self.estimate()
        print(f"Estimated Position: ({mean[0]:2f}, {mean[1]:2f}), Covariance diag: (({cov[0,0]:2f}, 0.0), (0.0, {cov[1,1]:2f})) \n")


        # plot smoothed path
        path = np.array(self.smoothed_state_estimates)
        ax.plot(path[:, 0], path[:, 1], c='red', linewidth=2, label='Smoothed Path', alpha=0.8)

        # plot current state estimate
        ax.scatter(self.smoothed_state[0], self.smoothed_state[1],
                   color='red', s=30, marker='*', label='Smoothed Estimate')

        # covariance ellipse
        eigvals, eigvecs = np.linalg.eig(self.smoothed_cov)
        angle = np.arctan2(eigvecs[1, 0], eigvecs[0, 0])
        width, height = 2 * np.sqrt(eigvals)
        ell = Ellipse(self.smoothed_state, width, height, angle=np.degrees(angle), color='r', alpha=0.2)
        ax.add_patch(ell)
        plt.pause(0.01)
        ax.clear()

    def motion_model(self, state, dt):
        Q = np.array([[0.5, 0.0], [0.0, 0.5]])
        # return state + np.random.multivariate_normal([self.velocity[0]*dt, self.velocity[1]*dt], Q)
        return state + np.random.multivariate_normal([0, 0], Q)

def main():
    cam_data = np.loadtxt(sys.argv[1])
    rf_data  = np.loadtxt(sys.argv[2])

    sigma_cam = np.eye(2)
    sigma_rf  = np.eye(2) * 0.1

    pf = ParticleFilter(num_particles=500,
                        init_state=cam_data[0],
                        init_cov=np.eye(2)*1.0,
                        smooth_alpha_mean=0.9,
                        smooth_alpha_cov=0.9)

    plt.ion()
    _, ax = plt.subplots()

    for i, cam in enumerate(cam_data):
        pf.predict(pf.motion_model)
        pf.update(cam, sigma_cam, 'cam')
        if i % 20 == 0:
            pf.update(tuple(rf_data[i]), sigma_rf, 'RF')
        pf.resample()
        pf.plot(ax)

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    main()
