/// 
/// ROS1 style demo program
/// 

#include <iostream>
#include <fstream>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/nav_sat_fix.hpp>
#include <sensor_msgs/msg/imu.hpp>
#include <geodesy/utm.h>
#include "sequential_bayesian_filter.h"

// state
//   x:  position  (m)
//   y:  position  (m)
//   r:  yaw angle (rad)
// input
//   vx: speed x   (m/s)
//   vy: speed y   (m/s)
//   vr: yaw rate  (rad/s)
// observation
//   x:  position  (m)
//   y:  position  (m)

// transition function (input, current state -> next state)
//   x += (cos(r) * vx - sin(r) * vy) * dt
//   y += (sin(r) * vx + cos(r) * vy) * dt
//   r += vr * dt
// observation function (state -> observervation)
//   x = x
//   y = y

int main(int argc, char* argv[])
{

  // launch ROS node
  rclcpp::init(argc, argv);
  auto node = rclcpp::Node::make_shared("gnss_ins_fusion_node");

  // load parameters
  RCLCPP_INFO(node->get_logger(), "Configuring Node...");
  node->declare_parameter("log_path", "");
  node->declare_parameter("noise_std_dev", 0.0);
  node->declare_parameter("noise_interval", 0);
  node->declare_parameter("particles", 0);
  node->declare_parameter("type", "");
  node->declare_parameter("std_x", 0.0);
  node->declare_parameter("std_y", 0.0);
  node->declare_parameter("std_yaw", 0.0);
  node->declare_parameter("speed", 0.0);
  node->declare_parameter("yaw", 0.0);

  ofstream ofs(node->get_parameter("log_path").as_string());
  ofs << "predicted,,observed,,,error" << endl;
  ofs << "x,y,x,y,yawrate,m" << endl;

  auto noise_std_dev = node->get_parameter("noise_std_dev").as_double();  // arbitrary addition of noise
  auto noise_interval = node->get_parameter("noise_interval").as_int();   // iteration

  auto particles = node->get_parameter("particles").as_int();  // applied only when using particle filter
  auto std_x = node->get_parameter("std_x").as_double();       // longitudinal position standard deviation in transition (m)
  auto std_y = node->get_parameter("std_y").as_double();       // lateral position standard deviation in transition (m)
  auto std_yaw = node->get_parameter("std_yaw").as_double();   // angle standard deviation in transition (rad)
  
  auto vx = node->get_parameter("speed").as_double();          // speed (m/s) (assuming it is constant so far)
  auto vy = 0.0;

  // create filter instance and upcast to abstract type
  shared_ptr<SequentialBayesianFilter> f;
  Vector3d x0(0, 0, node->get_parameter("yaw").as_double());   // yaw angle (rad));
  int y_size = 2;
  int u_size = 3;
  auto type = node->get_parameter("type").as_string();
  if (type == "ekf")
    f = make_shared<ExtendedKalmanFilter>(x0, y_size, u_size);
  else if (type == "ukf")
    f = make_shared<UnscentedKalmanFilter>(x0, y_size, u_size);
  else if (type == "pf")
    f = make_shared<ParticleFilter>(x0, y_size, u_size, particles);
  else
    throw logic_error("specified invalid filter type.");

  // initialize member variables / functions
  f->transition_func = [&](const TransitionVariables& v)
  {
    auto x = v.X(0) + (cos(v.X(2)) * v.U(0) - sin(v.X(2)) * v.U(1)) * v.dt;
    auto y = v.X(1) + (sin(v.X(2)) * v.U(0) + cos(v.X(2)) * v.U(1)) * v.dt;
    auto r = v.X(2) + v.U(2) * v.dt;
    return Vector3d(x, y, r);
  };
  f->observation_func = [&](const ObservationVariables& v)
  {
    return Vector2d(v.X(0), v.X(1));
  };
  f->P *= 0;
  f->Q(0, 0) = pow(std_x, 2);
  f->Q(1, 1) = pow(std_y, 2);
  f->Q(2, 2) = pow(std_yaw, 2);

  RCLCPP_INFO(node->get_logger(), "Configuring Done.");
    
  double prev_sec = 0.0;  // for laptime
  Vector2d y0(0, 0);      // initial observed position
  Vector2d y = y0;        // current observed position
  Vector3d x = x0;        // current state

  // intentional noise description
  int counter = 0;
  double bias_x = 0.0;
  double bias_y = 0.0;
  auto add_noise_on_observation = [&]()
  {
    random_device seed;
    default_random_engine engine(seed());
    if (counter % (noise_interval) == 0)
    {
      normal_distribution<> dist(0, 1);
      bias_x = dist(engine);
      bias_y = dist(engine);
    }
    Vector2d yn = y;
    if (counter / (noise_interval) % 2 == 1)
    {
      normal_distribution<> dist(0, noise_std_dev - 1);
      yn(0) += bias_x + dist(engine);
      yn(1) += bias_y + dist(engine);
      f->R(0, 0) = pow(noise_std_dev, 2);
      f->R(1, 1) = pow(noise_std_dev, 2);
    }
    counter++;
    return yn;
  };

  // INS subscription
  auto ins = node->create_subscription<sensor_msgs::msg::Imu>(
    "/vectornav/IMU", 
    rclcpp::QoS(10), 
    [&](const sensor_msgs::msg::Imu::SharedPtr msg)
    {
      // calculate dt from msg header
      auto sec = msg->header.stamp.sec + msg->header.stamp.nanosec * 1e-9;
      if (prev_sec == 0) { prev_sec = sec; return; }
      f->dt = sec - prev_sec;
      prev_sec = sec;
      // modify speed variance depending on current yaw angle
      auto qx = cos(x(2)) * std_x - sin(x(2)) * std_y;
      auto qy = sin(x(2)) * std_x + cos(x(2)) * std_y;
      f->Q(0, 0) = qx * qx;
      f->Q(0, 1) = qx * qy;
      f->Q(1, 0) = qx * qy;
      f->Q(1, 1) = qy * qy;
      auto yn = add_noise_on_observation();
      // Jacobian of transition (for EKF)
      f->A(0, 2) = (-vx * sin(x(2)) - vy * cos(x(2))) * f->dt;
      f->A(1, 2) = (+vx * cos(x(2)) - vy * sin(x(2))) * f->dt;
      // caluclate error (compared to observation without noise)
      auto error = sqrt(pow(y(0) - x(0), 2) + pow(y(1) - x(1), 2));
      // update filter
      auto yawrate = -msg->angular_velocity.z;
      x = f->predict(Vector3d(vx, vy, yawrate));
      RCLCPP_INFO(node->get_logger(), "predicted: (%.2f %.2f), observed: (%.2f %.2f), error: %.3f", x(0), x(1), yn(0), yn(1), error);
      ofs << x(0) << "," << x(1) << "," << yn(0) << "," << yn(1) << "," << yawrate << "," << error << endl;
    }
  );

  // GNSS subscription
  auto gnss = node->create_subscription<sensor_msgs::msg::NavSatFix>(
    "/fix", 
    rclcpp::QoS(10),
    [&](const sensor_msgs::msg::NavSatFix::SharedPtr msg)
    {
      // WGS84 -> UTM
      geographic_msgs::msg::GeoPoint wgs;
      wgs.latitude = msg->latitude;
      wgs.longitude = msg->longitude;
      geodesy::UTMPoint utm(wgs);
      Vector2d p(utm.easting, utm.northing);
      if (y0.x() == 0) y0 = p;
      y = p - y0;
      // 2D position covariance -> observation noise covariance
      f->R(0, 0) = msg->position_covariance[0];
      f->R(1, 1) = msg->position_covariance[4];
      // update filter
      f->update(y);
    }
  );

  rclcpp::spin(node);

  ofs.close();
  ins = nullptr;
  gnss = nullptr;
  RCLCPP_INFO(node->get_logger(), "Shutdown.");
  rclcpp::shutdown();
  return 0;
}