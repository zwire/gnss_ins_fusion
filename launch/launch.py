import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
  return LaunchDescription([
    Node(
      package="gnss_ins_fusion",
      executable="exe",
      output="screen",
      emulate_tty=True,
      parameters=[os.path.join(get_package_share_directory("gnss_ins_fusion"), "config", "params.yaml")]
    )
  ])