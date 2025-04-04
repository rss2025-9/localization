import numpy.typing as npt
import numpy as np

class MotionModel:

    def __init__(self, node:str, noise:float=0.1, deterministic:bool=False):
        ####################################
        # TODO
        # Do any precomputation for the motion
        # model here.

        self.deterministic: bool = deterministic
        self.noise: float = noise
        self.std: float = np.sqrt(noise)
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
        N: int = particles.shape[0]
        
        # Generate noise for each particle. Noise has shape (N, 3).
        noise: npt.NDArray = (
            np.zeros((N, 3)) if self.deterministic else 
            np.random.normal(0, self.std, size=particles.shape)
        )
        
        # Extract the current orientations for all particles.
        theta = particles[:, 2]
        cos_theta = np.cos(theta)
        sin_theta = np.sin(theta)
        
        # Apply the odometry translation (with noise) to all particles.
        dx = odometry[0] + noise[:, 0]
        dy = odometry[1] + noise[:, 1]

        # Update positions by rotating the odometry translation.
        particles[:, 0] += cos_theta * dx - sin_theta * dy
        particles[:, 1] += sin_theta * dx + cos_theta * dy
        
        # Update orientations.
        particles[:, 2] += odometry[2] + noise[:, 2]
        
        return particles
