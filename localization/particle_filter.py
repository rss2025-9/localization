from localization.sensor_model import SensorModel
from localization.motion_model import MotionModel

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped, PoseArray, Pose

import numpy as np 

from rclpy.node import Node
import rclpy

assert rclpy

from sensor_msgs.msg import LaserScan

import threading 


class ParticleFilter(Node):

    def __init__(self):
        super().__init__("particle_filter")

        self.declare_parameter('particle_filter_frame', "default")
        self.particle_filter_frame = self.get_parameter('particle_filter_frame').get_parameter_value().string_value

        #  *Important Note #1:* It is critical for your particle
        #     filter to obtain the following topic names from the
        #     parameters for the autograder to work correctly. Note
        #     that while the Odometry message contains both a pose and
        #     a twist component, you will only be provided with the
        #     twist component, so you should rely only on that
        #     information, and *not* use the pose component.
        
        self.declare_parameter('odom_topic', "/odom")
        self.declare_parameter('scan_topic', "/scan")

        scan_topic = self.get_parameter("scan_topic").get_parameter_value().string_value
        odom_topic = self.get_parameter("odom_topic").get_parameter_value().string_value

        self.laser_sub = self.create_subscription(LaserScan, scan_topic,
                                                  self.laser_callback,
                                                  1)

        self.odom_sub = self.create_subscription(Odometry, odom_topic,
                                                 self.odom_callback,
                                                 1)

        #  *Important Note #2:* You must respond to pose
        #     initialization requests sent to the /initialpose
        #     topic. You can test that this works properly using the
        #     "Pose Estimate" feature in RViz, which publishes to
        #     /initialpose.

        self.pose_sub = self.create_subscription(PoseWithCovarianceStamped, "/initialpose",
                                                 self.pose_callback,
                                                 1)

        #  *Important Note #3:* You must publish your pose estimate to
        #     the following topic. In particular, you must use the
        #     pose field of the Odometry message. You do not need to
        #     provide the twist part of the Odometry message. The
        #     odometry you publish here should be with respect to the
        #     "/map" frame.

        self.odom_pub = self.create_publisher(Odometry, "/pf/pose/odom", 1)

        # Initialize the models
        self.motion_model = MotionModel(self)
        self.sensor_model = SensorModel(self)

        self.get_logger().info("=============+READY+=============")

        # Implement the MCL algorithm
        # using the sensor model and the motion model
        #
        # Make sure you include some way to initialize
        # your particles, ideally with some sort
        # of interactive interface in rviz
        #
        # Publish a transformation frame between the map
        # and the particle_filter_frame.

        # visualizing particles
        self.particles_pub = self.create_publisher(PoseArray, "/particles", 1)

        self.prev_time = None

        # initializing number of particles, particles, + their weights 
        self.declare_parameter('num_particles', 50) 
        self.num_particles = self.get_parameter('num_particles').get_parameter_value().integer_value

        self.particles = np.zeros((self.num_particles, 3)) # (x, y, theta)  
        self.weights = np.ones(self.num_particles)/self.num_particles 

        self.lock = threading.Lock()

        self.simulation = False

    def pose_callback(self, pose: PoseWithCovarianceStamped): 
        """
        use rviz /initial_pose for initializing the particles and a guess of where the robot's location is 
        with a random spread of particles around a clicked point or pose. 
        """

        # getting x, y, theta from the pose 
        x = pose.pose.pose.position.x 
        y = pose.pose.pose.position.y
        theta = np.arctan2(pose.pose.pose.orientation.z, pose.pose.pose.orientation.w) * 2 # calcuting yaw angle 
        
        # intialize particles around this with gaussian 
        if self.weights is not None: 
            with self.lock: 
                self.particles[:, 0] = x + np.random.normal(0, 0.5, self.num_particles)
                self.particles[:, 1] = y + np.random.normal(0, 0.5, self.num_particles)
                self.particles[:, 2] = theta + np.random.normal(0, 0.5, self.num_particles)

                self.weights.fill(1 / self.num_particles) # weights set uniformly across all particles for initialization 

    def odom_callback(self, odometry: Odometry): 
        """
        process odometry data to pass into motion model
        only rely on twist and not pose 
        """
        
        # calculate the change in time (dt)
        curr_time = odometry.header.stamp.sec + odometry.header.stamp.nanosec * 1e-9

        if self.prev_time is None:
            self.prev_time = curr_time
            return
        
        dt = curr_time - self.prev_time
        self.prev_time = curr_time

        # get odometry data 
        if self.simulation: 
            vx = odometry.twist.twist.linear.x
            vy = odometry.twist.twist.linear.y
            theta = odometry.twist.twist.angular.z  # yaw once again
        else:
            vx = -odometry.twist.twist.linear.x
            vy = -odometry.twist.twist.linear.y
            theta = -odometry.twist.twist.angular.z  # yaw once again

        # calculate dx, dy, dtheta
        dx = vx * dt
        dy = vy * dt
        dtheta = theta * dt
        odometry_data = np.array([dx, dy, dtheta])

        # evaluate through motion model and update particles
        if self.weights is not None: 
            with self.lock: 
                self.particles = self.motion_model.evaluate(self.particles, odometry_data)
                self.publish_avg_pose()

    def laser_callback(self, scan: LaserScan): 
        """
        process scan data to pass into sensor model 
        and resample particles based on sensor model probabilities, numpy.random.choice can be useful 
        """
        
        scan_ranges = np.array(scan.ranges)

        # self.get_logger().info("before lock")
        with self.lock: 
            # get probabilities for each particle by passing scans into the sensor model and update weights 
            self.weights = self.sensor_model.evaluate(self.particles, scan_ranges)

            # self.get_logger().info("before weights")
            if self.weights is None: 
                self.get_logger().info("no weights")
                return # no weights  
            
            # self.get_logger().info("weights found")

            # self.weights += 1e-10 # to prevent dividing by 0 
            if np.sum(self.weights) != 0:
                self.weights /= np.sum(self.weights) # normalize all the weights 

            # resample particles 
            self.particles = self.particles[np.random.choice(self.particles.shape[0], size = self.particles.shape[0], p = self.weights, replace = True)]
            
            self.publish_avg_pose()
            
            self.weights.fill(1 / self.num_particles)   # resetting the weights for all particles

    def publish_avg_pose(self):
        # publish msg
        # determine "Average" particle pose and publish 
        # publishes estimated pose as a transformation between the /map frame and a frame for the expected car's base link 
        # --> publish to /base_link_pf for simulator 
        # --> publish to /base_link for real car 

        # weighted means for x and y, circular mean for theta 
        mean_x = np.sum(self.particles[:, 0] * self.weights)
        mean_y = np.sum(self.particles[:, 1] * self.weights)
        mean_theta = np.arctan2(np.sum(np.sin(self.particles[:, 2])), np.sum(np.cos(self.particles[:,2]))) 

        # publish estimated pose 
        msg = Odometry() 
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "map"
        if self.simulation: 
            msg.child_frame_id = "base_link_pf"
        else: 
            msg.child_frame_id = "base_link"
        msg.pose.pose.position.x = mean_x
        msg.pose.pose.position.y = mean_y 
        msg.pose.pose.orientation.z = np.sin(mean_theta / 2)
        msg.pose.pose.orientation.w = np.cos(mean_theta / 2)

        self.odom_pub.publish(msg)

        self.publish_particles()

    def publish_particles(self):
        pose_array = PoseArray()
        pose_array.header.stamp = self.get_clock().now().to_msg()
        pose_array.header.frame_id = "map"
        for i in range(self.num_particles):
            p = Pose()
            p.position.x = self.particles[i, 0]
            p.position.y = self.particles[i, 1]
            p.orientation.z = np.sin(self.particles[i, 2] / 2)
            p.orientation.w = np.cos(self.particles[i, 2] / 2)
            pose_array.poses.append(p)
        self.particles_pub.publish(pose_array)

def main(args=None):
    rclpy.init(args=args)
    pf = ParticleFilter()
    rclpy.spin(pf)
    rclpy.shutdown()
