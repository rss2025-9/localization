import numpy.typing as npt
import numpy as np

class MotionModel:

    def __init__(self, node:str, noise:float=0.1, deterministic:bool=False):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.deterministic: bool = deterministic
        self.std: float = noise
        self.node: str = node

        ####################################

    def evaluate(self, particles: npt.NDArray, odometry: npt.NDArray) -> npt.NDArray:
        """
        Update the particles to reflect probable
        future states given the odometry data.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            odometry: A 3-vector [dx dy dtheta]

        returns:
            particles: An updated matrix of the
                same size
        """

        ####################################
        # Gets number of particles
        N: int = particles.shape[0]
        # Calculates the rotation matrix for the particles.
        sins: npt.NDArray = np.sin(particles[:, 2])
        coss: npt.NDArray = np.cos(particles[:, 2])

        # Generates all odometry noise for all particles.
        if self.deterministic:
            particles[:, 0] += odometry[0] * coss - odometry[1] * sins
            particles[:, 1] += odometry[0] * sins + odometry[1] * coss
            particles[:, 2] += odometry[2]
        else:
            noisy_odom: npt.NDArray = np.random.normal(
                odometry[:2], [self.std, self.std/2], size=(N, 2)
            )
            particles[:, 0] += noisy_odom[:, 0] * coss - noisy_odom[:, 1] * sins
            particles[:, 1] += noisy_odom[:, 0] * sins + noisy_odom[:, 1] * coss
            particles[:, 2] += np.random.normal(odometry[2], self.std/2, size=N)
        # Normalize the angles to be between -pi and pi
        particles[:, 2] = np.arctan2(np.sin(particles[:, 2]), np.cos(particles[:, 2]))
        return particles

        ####################################
