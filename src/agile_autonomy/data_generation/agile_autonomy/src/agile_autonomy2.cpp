#include "agile_autonomy/agile_autonomy.h"

namespace agile_autonomy2 {
Agile_Autonomy2::Agile_Autonomy2(const ros::NodeHandle& nh,
                             const ros::NodeHandle& pnh)
          : nh_(nh), pnh_(pnh){
  //使用nh_句柄，在话题前加命名空间/hummingbird2/
  //使用pnh_句柄，在话题前加命名空间和节点名 /hummingbird2/agile_autonomy2/
  odometry_sub2_ = nh_.subscribe("ground_truth/odometry", 1,
                              &Agile_Autonomy2::odometryCallback2, this,
                              ros::TransportHints().tcpNoDelay());
  ref_traj_sub2_ = pnh_.subscribe("reference_trajectory",1,
                              &Agile_Autonomy2::refTrajCallback2, this,
                              ros::TransportHints().tcpNoDelay());
  off_sub2_ = nh_.subscribe("autopilot/off", 1,
                              &Agile_Autonomy2::offCallback2, this,
                              ros::TransportHints().tcpNoDelay());

  // Publishers
  control_command_pub2_ = nh_.advertise<quadrotor_msgs::ControlCommand>(
      "autopilot/control_command_input", 1);
  // Saving timer 以一定频率调用saveLoop2函数，
  save_timer2_ = nh_.createTimer(ros::Duration(1.0 / save_freq2_),
                                &Agile_Autonomy2::saveLoop2, this);
  // flightmare_bridge2_ =
  //     std::make_shared<flightmare_bridge::FlightmareBridge>(nh_, pnh_);                               
}

Agile_Autonomy2::~Agile_Autonomy2()
{
}

void Agile_Autonomy2::offCallback2(const std_msgs::EmptyConstPtr& msg) {
  ROS_INFO("UAV2 Received off command, stopping maneuver execution!");
  ROS_INFO("UAV2 Switching to kOff");
  // reference_progress_abs_ = 0;
  // network_prediction_.clear();
  // visualizer_->clearBuffers();
  // setup_done_ = false;
  base_controller2_.off();
  state_machine2_ = StateMachine::kOff;
}

quadrotor_common::QuadStateEstimate Agile_Autonomy2::getPredictedStateEstimate2(
    const ros::Time& time, const state_predictor::StatePredictor* predictor) {
  return predictor->predictState(time);
}

//为2号机新增refTrajCallback2,负责订阅1号机的reference_trajectory话题，生成2号机的控制命令
void Agile_Autonomy2::refTrajCallback2(const quadrotor_msgs::TrajectoryConstPtr &msg){
    quadrotor_common::Trajectory reference_trajectory2 = *msg;
    ros::Time time_now = ros::Time::now();
    ros::Time cmd_execution_time2 = time_now + ros::Duration(ctrl_cmd_delay2_);
    quadrotor_common::ControlCommand control_cmd2;
    //如果还用base_controller_，就会报错system_error。但用base_controller2_，不会有之前的报错，但飞机会乱飞。
    control_cmd2 = base_controller2_.run2(predicted_state2, reference_trajectory2,base_controller_params2_);
    control_cmd2.timestamp = time_now;
    control_cmd2.expected_execution_time = cmd_execution_time2;
    publishControlCommand2(control_cmd2);//发布2号机的控制命令
}

void Agile_Autonomy2::odometryCallback2(const nav_msgs::OdometryConstPtr& msg) {
  quadrotor_common::QuadStateEstimate state_estimate2;
  {
    std::lock_guard<std::mutex> guard(odom_mtx2_);
    received_state_est2_ = *msg;
    received_state_est2_.transformVelocityToWorldFrame();
    state_estimate2 = received_state_est2_;
  }
  
  // Push received state estimate into predictor
  state_predictor2_.updateWithStateEstimate(state_estimate2);
  ros::Time time_now = ros::Time::now();

  //如果状态机处在kExecteExpert或kNetwork模式下，则创建控制命令，
  //对2号机而言，暂时不处理状态机的问题，先取消掉
  ros::Time cmd_execution_time2 = time_now + ros::Duration(ctrl_cmd_delay2_);
  predicted_state2 = getPredictedStateEstimate2(cmd_execution_time2, &state_predictor2_);
  
}

void Agile_Autonomy2::saveLoop2(const ros::TimerEvent& time) {
  // if inputs are ok and unity is ready, we compute & publish depth
  // 利用2号机自身的状态输入，去在flightmare中计算出2号机的深度话题并发布
  agile_autonomy::AgileAutonomy *leader2;//必须得用指针，否则会有类未定义的错误
  
  printf("成功进入saveLoop函数\n"); 
  if (leader2->unity_is_ready_ && received_state_est2_.isValid()) {
    quadrotor_common::QuadStateEstimate temp_state_estimate2;
    {
      std::lock_guard<std::mutex> guard(odom_mtx2_);
      temp_state_estimate2 = received_state_est2_;
    }
    if (state_machine2_ == StateMachine::kExecuteExpert ||
        state_machine2_ == StateMachine::kNetwork) {
      // 在这加上二号机的getImages，以在flightmare中使用二号机真实的自身位姿，并发布二号机的rgb和深度话题
      // curr_data_dir_是用来保存图像等数据的路径
      printf("成功运行到if里\n"); 
      leader2->flightmare_bridge_->getImages2(temp_state_estimate2, curr_data_dir2_,
                                    frame_counter2_);
      frame_counter2_ += 1;
    } 
  }
}

void Agile_Autonomy2::publishControlCommand2(
    const quadrotor_common::ControlCommand& control_command) {
    quadrotor_msgs::ControlCommand control_cmd_msg;
    // 把控制指令通过ROS消息发布出去,同时把控制指令也传送到状态估计器 state_predictor2_中，用于估计无人机的实时位姿 odometry 。
    control_cmd_msg = control_command.toRosMessage();
    control_command_pub2_.publish(control_cmd_msg);
    // std::cout << "成功发布/hummingbird2/autopilot/control_command_input" << std::endl;
    //这里用的是2号机的state_predictor_,应该加一个uav_id判断用哪号无人机的状态估计器
    state_predictor2_.pushCommandToQueue(control_command);
}

}// namespace agile_autonomy2

//leader2 飞机控制节点测试
int main(int argc, char** argv) {
  ros::init(argc, argv, "agile_autonomy2");
  agile_autonomy2::Agile_Autonomy2 agile_autonomy2;

  ros::MultiThreadedSpinner spinner(4);
  spinner.spin();

  return 0;
} 