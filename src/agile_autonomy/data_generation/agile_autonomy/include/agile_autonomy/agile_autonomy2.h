#pragma once
#include "ros/ros.h"
#include "rpg_mpc/mpc_controller.h"
#include "quadrotor_common/trajectory.h"
#include "quadrotor_msgs/Trajectory.h"
#include "state_predictor/state_predictor.h"
#include "agile_autonomy/agile_autonomy.h"

namespace agile_autonomy2 {

class Agile_Autonomy2
{
public:
    Agile_Autonomy2(const ros::NodeHandle& nh, const ros::NodeHandle& pnh);
    
    Agile_Autonomy2() : Agile_Autonomy2(ros::NodeHandle(), ros::NodeHandle("~")) {}//委托构造函数
    
    virtual ~Agile_Autonomy2();

private:
  enum class StateMachine {
    kOff,
    kAutopilot,
    kExecuteExpert,
    kNetwork,
    kComputeLabels
  };
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber odometry_sub2_;
    ros::Subscriber ref_traj_sub2_;
    ros::Subscriber off_sub2_;

    ros::Publisher control_command_pub2_;
    ros::Timer save_timer2_;

    quadrotor_common::QuadStateEstimate getPredictedStateEstimate2(
      const ros::Time& time, const state_predictor::StatePredictor* predictor);
    void odometryCallback2(const nav_msgs::OdometryConstPtr& msg);
    void refTrajCallback2(const quadrotor_msgs::TrajectoryConstPtr &msg);
    void publishControlCommand2(const quadrotor_common::ControlCommand& control_command);
    void offCallback2(const std_msgs::EmptyConstPtr& msg);
    void saveLoop2(const ros::TimerEvent& time);
    
    rpg_mpc::MpcController<double> base_controller2_ =
      rpg_mpc::MpcController<double>(ros::NodeHandle(),ros::NodeHandle("~"),"/hummingbird2/test2");
    rpg_mpc::MpcParams<double> base_controller_params2_;

    StateMachine state_machine2_ = StateMachine::kAutopilot;
    state_predictor::StatePredictor state_predictor2_;

    quadrotor_common::QuadStateEstimate predicted_state2;
    quadrotor_common::QuadStateEstimate received_state_est2_;
    std::shared_ptr<flightmare_bridge::FlightmareBridge> flightmare_bridge2_;
    std::string curr_data_dir2_ = "/home/pc205/agile_autonomy_ws/catkin_aa/src/agile_autonomy/data_generation/data_leader2";
    double ctrl_cmd_delay2_;
      
    double save_freq2_ = 15.0; // save images & odometry at this frequency;
    // Data generation
    int frame_counter2_ = 0;
    bool unity_is_ready_ = false;
};

}//namespace agile_autonomy2    