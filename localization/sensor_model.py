import numpy as np
from scan_simulator_2d import PyScanSimulator2D
# Try to change to just `from scan_simulator_2d import PyScanSimulator2D` 
# if any error re: scan_simulator_2d occurs

from tf_transformations import euler_from_quaternion

from nav_msgs.msg import OccupancyGrid

import sys

np.set_printoptions(threshold=sys.maxsize)


class SensorModel:

    def __init__(self, node):
        node.declare_parameter('map_topic', "default")
        node.declare_parameter('num_beams_per_particle', 1)
        node.declare_parameter('scan_theta_discretization', 1.0)
        node.declare_parameter('scan_field_of_view', 1.0)
        node.declare_parameter('lidar_scale_to_map_scale', 1.0)

        self.map_topic = node.get_parameter('map_topic').get_parameter_value().string_value
        self.num_beams_per_particle = node.get_parameter('num_beams_per_particle').get_parameter_value().integer_value
        self.scan_theta_discretization = node.get_parameter(
            'scan_theta_discretization').get_parameter_value().double_value
        self.scan_field_of_view = node.get_parameter('scan_field_of_view').get_parameter_value().double_value
        self.lidar_scale_to_map_scale = node.get_parameter(
            'lidar_scale_to_map_scale').get_parameter_value().double_value

        ####################################
        # Adjust these parameters - initialized to part a values for now 
        self.alpha_hit: float = 0.74
        self.alpha_short: float = 0.07
        self.alpha_max: float = 0.07
        self.alpha_rand: float = 0.12
        self.sigma_hit: float = 0.5

        # Your sensor table will be a `table_width` x `table_width` np array:
        self.table_width: int = 201
        ####################################

        node.get_logger().info("%s" % self.map_topic)
        node.get_logger().info("%s" % self.num_beams_per_particle)
        node.get_logger().info("%s" % self.scan_theta_discretization)
        node.get_logger().info("%s" % self.scan_field_of_view)

        # Precompute the sensor model table
        self.sensor_model_table = np.empty((self.table_width, self.table_width))
        self.precompute_sensor_model()

        # Create a simulated laser scan
        self.scan_sim = PyScanSimulator2D(
            self.num_beams_per_particle,
            self.scan_field_of_view,
            0,  # This is not the simulator, don't add noise
            0.01,  # This is used as an epsilon
            self.scan_theta_discretization)

        # Subscribe to the map
        self.map = None
        self.map_set = False
        self.map_subscriber = node.create_subscription(
            OccupancyGrid,
            self.map_topic,
            self.map_callback,
            1)

    def precompute_sensor_model(self):
        """
        Generate and store a table which represents the sensor model.

        For each discrete computed range value, this provides the probability of 
        measuring any (discrete) range. This table is indexed by the sensor model
        at runtime by discretizing the measurements and computed ranges from
        RangeLibc.
        This table must be implemented as a numpy 2D array.

        Compute the table based on class parameters alpha_hit, alpha_short,
        alpha_max, alpha_rand, sigma_hit, and table_width.

        args:
            N/A

        returns:
            No return type. Directly modify `self.sensor_model_table`.
        """


        #LIKE PART A OF WRITTEN HW
        z_max=self.table_width-1
        #given in question
        #epsilon=0.1

        # p_rand is the probability of a random measurement and is approx.
        # a uniform distribution across all possible measurements.
        p_rand: float = 1.0/z_max
        for d in range(self.table_width): #LIKE d IN PART A
            # p_hit is the probability of a hit at distance d and needs to be
            # normalized as the Gaussian is not probabilistic due to cutoffs.
            p_hit_d = np.array([
                np.exp(-0.5*((z_k-d)**2.0/self.sigma_hit**2.0))
                for z_k in range(self.table_width)
            ])
            p_hit_d = p_hit_d / np.sum(p_hit_d) # normalize p_hit across z_k

            z_k: float
            for z_k in range(self.table_width): # LIKE z_k IN PART A
                p_short: float = 0.0 
                if 0 <= z_k <= d and d > 0:
                    p_short = (2.0 / float(d)) * (1.0 - (z_k / float(d)))

                p_max: float = 1.0 if z_k == z_max else 0.0
                
                #do weighted sum as given based on alphas, put into table
                self.sensor_model_table[z_k, d] = (
                    self.alpha_hit*p_hit_d[z_k] + self.alpha_short*p_short + 
                    self.alpha_max*p_max + self.alpha_rand*p_rand
                )
        
        #normal cols to sum to one (each col. is a d val)
        self.sensor_model_table=self.sensor_model_table/(self.sensor_model_table.sum(axis=0, keepdims=True))

    def evaluate(self, particles, observation):
        """
        Evaluate how likely each particle is given
        the observed scan.

        args:
            particles: An Nx3 matrix of the form:

                [x0 y0 theta0]
                [x1 y0 theta1]
                [    ...     ]

            observation: A vector of lidar data measured
                from the actual lidar. THIS IS Z_K. Each range in Z_K is Z_K^i

        returns:
           probabilities: A vector of length N representing
               the probability of each particle existing
               given the observation and the map.
        """

        if not self.map_set:
            return

        ####################################
        # TODO
        # Evaluate the sensor model here!
        #
        # You will probably want to use this function
        # to perform ray tracing from all the particles.
        # This produces a matrix of size N x num_beams_per_particle 

        scans=self.scan_sim.scan(particles)
        
        #distances to pixels via scaling factor
        scale=self.resolution*self.lidar_scale_to_map_scale

        observation=np.clip(observation/scale, 0, self.table_width-1).astype(int)
        scans=np.clip(scans/scale, 0, self.table_width-1).astype(int)
        
        #gets probs
        probs=self.sensor_model_table[scans, observation]
        
        #multiply probs across all beams per particle
        return np.prod(probs, axis=1)

        ####################################

    def map_callback(self, map_msg):
        # Convert the map to a numpy array
        self.map = np.array(map_msg.data, np.double) / 100.
        self.map = np.clip(self.map, 0, 1)

        self.resolution = map_msg.info.resolution

        # Convert the origin to a tuple
        origin_p = map_msg.info.origin.position
        origin_o = map_msg.info.origin.orientation
        origin_o = euler_from_quaternion((
            origin_o.x,
            origin_o.y,
            origin_o.z,
            origin_o.w))
        origin = (origin_p.x, origin_p.y, origin_o[2])

        # Initialize a map with the laser scan
        self.scan_sim.set_map(
            self.map,
            map_msg.info.height,
            map_msg.info.width,
            map_msg.info.resolution,
            origin,
            0.5)  # Consider anything < 0.5 to be free

        # Make the map set
        self.map_set = True

        print("Map initialized")













