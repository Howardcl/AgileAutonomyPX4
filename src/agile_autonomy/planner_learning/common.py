import os
import time
from geometry_msgs.msg import PoseStamped
from gazebo_msgs.srv import *
from std_msgs.msg import Bool, Empty, Float32
from nav_msgs.msg import Odometry
import rospy
import numpy as np
from pyquaternion import Quaternion


class MessageHandler():
    def __init__(self):
        self.autopilot_off = rospy.Publisher("/hummingbird/autopilot/off", Empty,
                                             queue_size=1)
        self.autopilot_off2 = rospy.Publisher("/hummingbird2/autopilot/off", Empty,
                                             queue_size=1)
        self.arm_bridge = rospy.Publisher("/hummingbird/bridge/arm", Bool,
                                          queue_size=1)
        self.arm_bridge2 = rospy.Publisher("/hummingbird2/bridge/arm", Bool,
                                          queue_size=1)
        self.autopilot_start = rospy.Publisher("/hummingbird/autopilot/start", Empty,
                                               queue_size=1)
        self.autopilot_start2 = rospy.Publisher("/hummingbird2/autopilot/start", Empty,
                                               queue_size=1)
        self.autopilot_pose_cmd = rospy.Publisher("/hummingbird/autopilot/pose_command",
                                                  PoseStamped, queue_size=1)
        self.tree_spacing_cmd = rospy.Publisher("/hummingbird/tree_spacing",
                                                Float32, queue_size=1)
        self.obj_spacing_cmd = rospy.Publisher("/hummingbird/object_spacing",
                                               Float32, queue_size=1)
        self.reset_exp_pub = rospy.Publisher("/success_reset",
                                             Empty, queue_size=1)
        self.save_pc_pub = rospy.Publisher("/hummingbird/save_pc",
                                           Bool, queue_size=1)
    #改动，为2号机也发布这个话题
    def publish_autopilot_off(self):
        msg = Empty()
        self.autopilot_off.publish(msg)
        self.autopilot_off2.publish(msg)#不是2号机的koff状态导致跟踪轨迹乱飞的
        time.sleep(1)

    def publish_reset(self):
        msg = Empty()
        self.reset_exp_pub.publish(msg)

    def publish_tree_spacing(self, spacing):
        msg = Float32()
        msg.data = spacing
        print("Setting Tree Spacing to {}".format(msg.data))
        self.tree_spacing_cmd.publish(msg)

    def publish_obj_spacing(self, spacing):
        msg = Float32()
        msg.data = spacing
        print("Setting Object Spacing to {}".format(msg.data))
        self.obj_spacing_cmd.publish(msg)
    #改动，为2号机也发布这个话题
    def publish_arm_bridge(self):
        msg = Bool()
        msg.data = True
        self.arm_bridge.publish(msg)

    def publish_save_pc(self):
        msg = Bool()
        msg.data = True
        self.save_pc_pub.publish(msg)
    #改动，为2号机也发布这个话题
    def publish_autopilot_start(self):
        msg = Empty()
        self.autopilot_start.publish(msg)
        self.autopilot_start2.publish(msg)
        time.sleep(25)

    def publish_goto_pose(self, pose=[0, 0, 3.]):
        msg = PoseStamped()
        msg.header.frame_id = "world"
        msg.pose.position.x = pose[0]
        msg.pose.position.y = pose[1]
        msg.pose.position.z = pose[2]
        msg.pose.orientation.x = 0.0
        msg.pose.orientation.y = 0.0
        msg.pose.orientation.z = 0.0
        msg.pose.orientation.w = 1.0
        self.autopilot_pose_cmd.publish(msg)
        time.sleep(3)

def setup_sim(msg_handler, config):
    print("==========================")
    print("     RESET SIMULATION     ")
    print("==========================")

    # after this message, autopilot will automatically go to 'BREAKING' and 'HOVER' state since
    # no control_command_inputs are published any more
    #system函数可以将字符串转化成命令在服务器上运行；
    #其原理是每一条system函数执行时，其会创建一个子进程在系统上执行命令行，子进程的执行结果无法影响主进程；
    # 暂停仿真，仿真时间停止，对象变静态，但gazebo内部更新循环仍在继续，但仿真时间没有改变，因此任何受仿真时间限制的内容都不会更新
    os.system("rosservice call /gazebo/pause_physics")
    # 取消暂停
    print("Unpausing Physics...")
    os.system("rosservice call /gazebo/unpause_physics")

    print("Placing quadrotor...")
    #发布"/hummingbird/autopilot/off"话题，当进入AgileAutonomy::offCallback回调函数后，1号机切换到kOff状态
    #发布"/hummingbird2/autopilot2/off"话题，当进入Agile_Autonomy2::offCallback回调函数后，2号机切换到kOff状态
    msg_handler.publish_autopilot_off() 
    # get a position  设置无人机初始位置
    pos_choice = np.random.choice(len(config.unity_start_pos))
    position = np.array(config.unity_start_pos[pos_choice])
    # No yawing possible for trajectory generation  设置无人机初始方向
    start_quaternion = Quaternion(axis=[0,0,1], angle=position[-1]).elements

    #向服务端发送请求，设置无人机状态,包括位置和姿态。
    start_string = "rosservice call /gazebo/set_model_state " + \
     "'{model_state: { model_name: hummingbird, pose: { position: { x: %f, y: %f ,z: %f }, " % (position[0],position[1],position[2]) + \
     "orientation: {x: %f, y: %f, z: %f, w: %f}}, " % (start_quaternion[1],start_quaternion[2],start_quaternion[3],start_quaternion[0]) + \
     "twist:{ linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 }}, " + \
     "reference_frame: world } }'"
    os.system(start_string)

    rospy.wait_for_service('/gazebo/set_model_state')
    set_state_service = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)
    objstate = SetModelStateRequest()  # Create an object of type SetModelStateRequest
    objstate.model_state.model_name = "hummingbird2"
    objstate.model_state.pose.position.x = -20.0
    objstate.model_state.pose.position.y = 21.0
    objstate.model_state.pose.position.z = 0.06
    objstate.model_state.pose.orientation.w = 1
    objstate.model_state.pose.orientation.x = 0
    objstate.model_state.pose.orientation.y = 0
    objstate.model_state.pose.orientation.z = 0
    objstate.model_state.twist.linear.x = 0.0
    objstate.model_state.twist.linear.y = 0.0
    objstate.model_state.twist.linear.z = 0.0
    objstate.model_state.twist.angular.x = 0.0
    objstate.model_state.twist.angular.y = 0.0
    objstate.model_state.twist.angular.z = 0.0
    objstate.model_state.reference_frame = "world"
    result = set_state_service(objstate)
    # start_string_2 = "rosservice call /gazebo/set_model_state " + \
    #  "'{model_state: { model_name: hummingbird2, pose: { position: { x: %f, y: %f ,z: %f }, " % (-20,21,0.06) + \
    #  "orientation: {x: %f, y: %f, z: %f, w: %f}}, " % (0,0,0,1) + \
    #  "twist:{ linear: {x: 0.0 , y: 0 ,z: 0 } , angular: { x: 0.0 , y: 0 , z: 0.0 }}, " + \
    #  "reference_frame: world } }'"
    # os.system(start_string_2)
    
    #返回无人机当前设置的位置值
    return position


def place_quad_at_start(msg_handler):
    '''
    start position: a tuple, array, or list with [x,y,z] representing the start position.
    '''
    # Make sure to use GT odometry in this step
    msg_handler.publish_autopilot_off()#发布"/hummingbird/autopilot/off"话题，当进入AgileAutonomy::offCallback回调函数后，1号机切换到kOff状态
    # reset quad to initial position
    msg_handler.publish_arm_bridge()#发布"/hummingbird/bridge/arm"话题
    msg_handler.publish_autopilot_start()#发布"/hummingbird/autopilot/start"话题
    return
