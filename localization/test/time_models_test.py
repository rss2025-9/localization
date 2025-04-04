import rclpy

from localization.test.motion_model_test import MotionModelTest
from localization.test.sensor_model_test import SensorModelTest


def time_motion_model(n: int = 10):
    rclpy.init(args=None)
    start_time = rclpy.clock.Clock().now()
    for _ in range(n):
        motion_model_test = MotionModelTest()
        motion_model_test.test_evaluate_motion_model()
        rclpy.spin(motion_model_test)
    end_time = rclpy.clock.Clock().now()
    elapsed_time = end_time - start_time
    print(f"Time taken for {n} iterations of motion model test: {elapsed_time.nanoseconds / 1e6} ms")
    rclpy.shutdown()


def time_sensor_model(n: int = 10):
    rclpy.init(args=None)
    start_time = rclpy.clock.Clock().now()
    for _ in range(n):
        sensor_model_test = SensorModelTest()
        sensor_model_test.test_evaluate_sensor_model()
        rclpy.spin(sensor_model_test)
    end_time = rclpy.clock.Clock().now()
    elapsed_time = end_time - start_time
    print(f"Time taken for {n} iterations of sensor model test: {elapsed_time.nanoseconds / 1e6} ms")
    rclpy.shutdown()


def main(args=None):
    n = 10
    time_motion_model(n)
    time_sensor_model(n)