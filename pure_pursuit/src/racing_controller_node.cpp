#include <rclcpp/rclcpp.hpp>
#include <nav_msgs/msg/path.hpp>
#include <nav_msgs/msg/odometry.hpp>
#include <ackermann_msgs/msg/ackermann_drive_stamped.hpp>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>
#include <cmath>
#include <algorithm>

class RacingController : public rclcpp::Node {
public:
  RacingController() : Node("racing_controller") {

    lookahead_ = declare_parameter<double>("lookahead", 2.2);
    wheelbase_ = declare_parameter<double>("wheelbase", 0.34);
    v_min_ = declare_parameter<double>("speed_min", 2.0);
    v_max_ = declare_parameter<double>("speed_max", 6.0);
    k_speed_ = declare_parameter<double>("k_speed", 2.5);
    k_accel_ = declare_parameter<double>("k_accel", 1.2);
    a_min_ = declare_parameter<double>("accel_min", -6.0);
    a_max_ = declare_parameter<double>("accel_max", 10.0);

    drive_topic_ = declare_parameter<std::string>("drive_topic", "ackermann_cmd0");
    center_topic_ = declare_parameter<std::string>("center_path_topic", "center_path");
    left_topic_ = declare_parameter<std::string>("left_boundary", "left_boundary");
    right_topic_ = declare_parameter<std::string>("right_boundary", "right_boundary");
    odom_topic_ = declare_parameter<std::string>("odom_topic", "odom0");

    auto map_qos = rclcpp::QoS(1).reliable().transient_local();
    auto drive_qos = rclcpp::QoS(1).reliable();
    auto odom_qos = rclcpp::QoS(1).best_effort();

    drive_pub_ = create_publisher<ackermann_msgs::msg::AckermannDriveStamped>(
        drive_topic_, drive_qos);

    sub_center_ = create_subscription<nav_msgs::msg::Path>(
        center_topic_, map_qos,
        [this](nav_msgs::msg::Path::SharedPtr msg){
          center_path_ = *msg;
          got_center_ = true;
          tryMakeRacingLine();
        });

    sub_left_ = create_subscription<nav_msgs::msg::Path>(
        left_topic_, map_qos,
        [this](nav_msgs::msg::Path::SharedPtr msg){
          left_path_ = *msg;
          got_left_ = true;
          tryMakeRacingLine();
        });

    sub_right_ = create_subscription<nav_msgs::msg::Path>(
        right_topic_, map_qos,
        [this](nav_msgs::msg::Path::SharedPtr msg){
          right_path_ = *msg;
          got_right_ = true;
          tryMakeRacingLine();
        });

    sub_odom_ = create_subscription<nav_msgs::msg::Odometry>(
        odom_topic_, odom_qos,
        [this](nav_msgs::msg::Odometry::SharedPtr msg){
          odom_ = *msg;
          has_odom_ = true;
        });

    timer_ = create_wall_timer(std::chrono::milliseconds(10),
                           std::bind(&RacingController::onTimer, this));


    RCLCPP_INFO(get_logger(), "RacingController started!");
  }

private:

  // Try compute once
  void tryMakeRacingLine(){
    if(!got_center_ || !got_left_ || !got_right_) return;
    if(racing_done_) return;
    makeRacingLine();
    racing_done_ = true;
  }

  void makeRacingLine(){
    racing_path_ = center_path_; // baseline

    RCLCPP_INFO(get_logger(), "Racing Path fixed, poses=%zu",
                racing_path_.poses.size());
  }

  void onTimer(){
    if(!racing_done_ || !has_odom_) return;
    if(racing_path_.poses.empty()) return;

    // Pose
    const auto &p = odom_.pose.pose.position;
    const auto &q = odom_.pose.pose.orientation;
    double roll,pitch,yaw;
    tf2::Quaternion tq(q.x,q.y,q.z,q.w);
    tf2::Matrix3x3(tq).getRPY(roll,pitch,yaw);

    double x=p.x, y=p.y;

    int idx = findLookaheadIndex(x,y,lookahead_);
    if(idx<0) return;
    auto &tp = racing_path_.poses[idx].pose.position;

    double dx = tp.x-x;
    double dy = tp.y-y;
    double xL = cos(yaw)*dx + sin(yaw)*dy;
    double yL =-sin(yaw)*dx + cos(yaw)*dy;
    if(xL<=0.01) return;

    double Ld = hypot(xL,yL);
    double curvature = 2.0*yL/(Ld*Ld);
    double steer = atan(wheelbase_*curvature);

    double v_ref = std::max(v_min_, v_max_ - k_speed_*fabs(curvature));
    double v = odom_.twist.twist.linear.x;

    double a_cmd = k_accel_*(v_ref-v);
    a_cmd = std::clamp(a_cmd, a_min_, a_max_);

    ackermann_msgs::msg::AckermannDriveStamped cmd;
    cmd.header.stamp = now();
    cmd.drive.steering_angle = steer;
    cmd.drive.acceleration = a_cmd;

    drive_pub_->publish(cmd);
  }

  int findLookaheadIndex(double x,double y,double Ld){
    const size_t N = racing_path_.poses.size();
    if(N<2) return -1;
    size_t closest = 0;
    double best = 1e18;
    for(size_t i=0;i<N;i++){
      auto &pt = racing_path_.poses[i].pose.position;
      double d2=(pt.x-x)*(pt.x-x)+(pt.y-y)*(pt.y-y);
      if(d2<best){best=d2;closest=i;}
    }
    double accum=0.0;
    for(size_t i=0;i<N;i++){
      size_t j=(closest+i)%N;
      size_t jn=(j+1)%N;
      auto &a=racing_path_.poses[j].pose.position;
      auto &b=racing_path_.poses[jn].pose.position;
      accum+=hypot(b.x-a.x,b.y-a.y);
      if(accum>=Ld) return (int)jn;
    }
    return (int)closest;
  }

  rclcpp::TimerBase::SharedPtr timer_;
  rclcpp::Publisher<ackermann_msgs::msg::AckermannDriveStamped>::SharedPtr drive_pub_;
  rclcpp::Subscription<nav_msgs::msg::Path>::SharedPtr sub_center_,sub_left_,sub_right_;
  rclcpp::Subscription<nav_msgs::msg::Odometry>::SharedPtr sub_odom_;

  nav_msgs::msg::Path center_path_, left_path_, right_path_, racing_path_;
  nav_msgs::msg::Odometry odom_;

  bool got_center_{false},got_left_{false},got_right_{false};
  bool has_odom_{false};
  bool racing_done_{false};

  double lookahead_, wheelbase_;
  double v_min_, v_max_, k_speed_;
  double k_accel_, a_min_, a_max_;
  std::string drive_topic_, center_topic_, left_topic_, right_topic_, odom_topic_;
};

int main(int argc,char**argv){
  rclcpp::init(argc,argv);
  rclcpp::spin(std::make_shared<RacingController>());
  rclcpp::shutdown();
  return 0;
}

