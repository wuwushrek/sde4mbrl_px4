#include "geometric_controller.h"

using namespace Eigen;
using namespace std;

// Constructor
geometricCtrl::geometricCtrl(const ros::NodeHandle &nh)
    :   nh_(nh)
{

    // Get the parameters according to the attribute of this class
    nh_.param<double>("drag_dx", dx_, 0.0);
    nh_.param<double>("drag_dy", dy_, 0.0);
    nh_.param<double>("drag_dz", dz_, 0.0);
    nh_.param<double>("attctrl_tau", attctrl_tau_, 0.1);
    nh_.param<double>("norm_thrust_const", norm_thrust_const_, 0.05);  // 1 / max acceleration
    nh_.param<double>("norm_thrust_offset", norm_thrust_offset_, 0.1);    // 1 / max acceleration
    nh_.param<double>("Kp_x", Kpos_x_, 8.0);
    nh_.param<double>("Kp_y", Kpos_y_, 8.0);
    nh_.param<double>("Kp_z", Kpos_z_, 10.0);
    nh_.param<double>("Kv_x", Kvel_x_, 1.5);
    nh_.param<double>("Kv_y", Kvel_y_, 1.5);
    nh_.param<double>("Kv_z", Kvel_z_, 3.3);
    nh_.param<double>("gravity", gravity_, 9.8);
    nh_.param<double>("max_acc", max_fb_acc_, 9.0);
    nh_.param<int>("ctrl_mode", ctrl_mode_, ERROR_QUATERNION);
    nh_.param<bool>("feedthrough_enable", feedthrough_enable_, false);
    last_time_ = ros::Time::now();
    trajec_time_ = -1.0;
    current_stage_ = 0;
    run_trajectory_ = false;

    // Initialize array version of some of these parameters
    g_ << 0.0, 0.0, -gravity_;
    Kpos_ << -Kpos_x_, -Kpos_y_, -Kpos_z_;
    Kvel_ << -Kvel_x_, -Kvel_y_, -Kvel_z_;
    D_ << dx_, dy_, dz_;

    // Initialize the subscribers
    mav_full_state_sub_ = nh_.subscribe("mavros/mpc_full_state/state", 1, &geometricCtrl::mavStateCallback, this);
    
    // Publishers
    cmdPub_ = nh_.advertise<mavros_msgs::AttitudeTarget>("mavros/setpoint_raw/attitude", 1);
    setpointPub_ = nh_.advertise<geometry_msgs::PoseStamped>("mavros/desired_setpoint", 10);

    // Starting the required services
    setTrajectorySrv_ = nh_.advertiseService("set_trajectory_and_params", &geometricCtrl::setTrajectoryAndParams, this);
    startTrajectorySrv_ = nh_.advertiseService("start_trajectory", &geometricCtrl::triggerTrajectory, this);
}


geometricCtrl::~geometricCtrl() {
  // Destructor
}

bool geometricCtrl::setTrajectoryAndParams(mpc4px4::LoadTrajAndParams::Request &req, mpc4px4::LoadTrajAndParams::Response &res)
{
    // Check if traj_dir_csv is an empty string
    if (!req.traj_dir_csv.empty())
    {
        // First load the trajectory
        if (loadTrajectory(req.traj_dir_csv) == false)
        {
            res.success = false;
            ROS_ERROR("Failed to load the trajectory");
            return false;
        }
    }
    if (!req.controller_param_yaml.empty())
    {
        // Then load the parameters
        if (loadParameters(req.controller_param_yaml) == false)
        {
            res.success = false;
            ROS_ERROR("Failed to load the parameters");
            return false;
        }
    }
    res.success = true;
    return true;
}

bool geometricCtrl::triggerTrajectory(mpc4px4::FollowTraj::Request &req, mpc4px4::FollowTraj::Response &res)
{
    startTrajectory(req.state_controller);
    res.success = true;
    return true;
}

void geometricCtrl::mavStateCallback(const mavros_msgs::MPCFullStateConstPtr &msg) {
    // Extract the position convert to double
    pos_ << (double) msg->x, (double) msg->y, (double) msg->z;
    // Extract the velocity
    vel_ << (double) msg->vx, (double) msg->vy, (double) msg->vz;
    // Extract the quaternion
    quat_ << (double) msg->qw, (double) msg->qx, (double) msg->qy, (double) msg->qz;
    // Extract the angular velocity
    omega_ << (double) msg->wx, (double) msg->wy, (double) msg->wz;

    // TODO: Maybe in a spinonce type of approach
    // Do the control
    controlLoopBody();
}

void geometricCtrl::controlLoopBody()
{
    // Get the current time
    ros::Time now = ros::Time::now();
    double dt = (now - last_time_).toSec();
    last_time_ = now;

    // // In case no trajectory is received don't do anything
    // if (trajec_time_ < 0.0) return;

    // Get the next setpoint from the trajectory
    Eigen::Vector3d targetPos, targetVel, targetAcc;
    double targetYaw;
    Eigen::Vector4d q_des;

    bool success = extractSetpointFromTrajectory(dt, targetPos, targetVel, targetAcc, targetYaw);
    
    // // warn if the trajectory is not being followed
    // if (!success) {
    //     ROS_WARN("Trajectory not being followed");
    // }

    if (!success) return;

    // // Print current time
    // ROS_WARN("Current time: %f", trajec_time_);
    // // Ros warn the current target position
    // ROS_WARN("Target Position: %f, %f, %f", targetPos(0), targetPos(1), targetPos(2));
    // // velocity
    // ROS_WARN("Target Velocity: %f, %f, %f", targetVel(0), targetVel(1), targetVel(2));
    // // acceleration
    // ROS_WARN("Target Acceleration: %f, %f, %f", targetAcc(0), targetAcc(1), targetAcc(2));
    // // yaw
    // ROS_WARN("Target Yaw: %f", targetYaw);


    Eigen::Vector3d desired_acc;
    if (feedthrough_enable_) {
        desired_acc = targetAcc;
    } else {
        desired_acc = controlPosition(targetPos, targetVel, targetAcc, targetYaw);
    }

    // Compute the desired body rate
    Eigen::Vector4d cmd_value = computeBodyRateCmd(desired_acc, targetYaw, q_des);

    // // Warn the desired body rate
    // ROS_WARN("Desired Body Rate: Mx = %f, My = %f, Mz = %f, Thrust = %f", cmd_value(0), cmd_value(1), cmd_value(2), cmd_value(3));

    // Publish the command values
    pubRateCommands(now, cmd_value, q_des);

    // Publish the desired setpoint. The time is now + dt
    // TODO: Check if this is the correct way to do this -> it is now + dt or now
    pubReferencePose(now+ros::Duration(dt), targetPos, q_des);
}


bool geometricCtrl::extractSetpointFromTrajectory(const double dt, 
    Eigen::Vector3d &targetPos, 
    Eigen::Vector3d &targetVel, 
    Eigen::Vector3d &targetAcc,
    double &targetYaw)
{
    // Check if the trajectory is given
    if (trajec_time_ < 0.0) return false;
    // Current time
    double current_time_val = trajec_time_;
    // Next time step
    double next_time_val = current_time_val + dt;

    // // Print next time
    // ROS_WARN("Next time: %f", next_time_val);

    // Check if the next time step is out of the trajectory and return the last setpoint
    if (next_time_val >= target_time_.back()) {
        targetPos = target_pos_.back();
        targetVel = target_vel_.back();
        targetAcc = target_acc_.back();
        // Set the current time to the end of the trajectory
        trajec_time_ = target_time_.back();
        // Update the stage of the trajectory
        current_stage_ = target_time_.size() - 1;
        return true;
    }
    // Check if run_trajectory_ is true
    if (!run_trajectory_){
        // Use the first setpoint
        targetPos = target_pos_.front();
        targetVel = target_vel_.front();
        targetAcc = target_acc_.front();
        // Set the current stage to the first stage
        current_stage_ = 0;
        trajec_time_ = 0.0;
        return true;
    }

    // Find the index where the next time step is located from the current stage
    auto it = std::upper_bound(target_time_.begin() + current_stage_, target_time_.end(), next_time_val);
    // Get the time value corresponding to the index 
    double time_val = *it;
    // Get the index
    int index = std::distance(target_time_.begin(), it);
    // Get the time value corresponding to the previous index
    double prev_time_val = *(it - 1);
    // Get the index of the previous time step
    int prev_index = index - 1;
    // Extrapolate the position, velocity and acceleration between these two time steps
    double alpha = (next_time_val - prev_time_val) / (time_val - prev_time_val);
    targetPos = target_pos_[prev_index] + alpha * (target_pos_[index] - target_pos_[prev_index]);
    targetVel = target_vel_[prev_index] + alpha * (target_vel_[index] - target_vel_[prev_index]);
    targetAcc = target_acc_[prev_index] + alpha * (target_acc_[index] - target_acc_[prev_index]);
    targetYaw = target_yaw_[prev_index] + alpha * (target_yaw_[index] - target_yaw_[prev_index]);
    // Update the current stage
    current_stage_ = prev_index; // TODO: Maybe index-1
    // Update the current time
    trajec_time_ = next_time_val;
    return true;
}


void geometricCtrl::pubReferencePose(const ros::Time &now, const Eigen::Vector3d &target_position, const Eigen::Vector4d &target_attitude) {
    geometry_msgs::PoseStamped msg;
    msg.header.stamp = now;
    msg.header.frame_id = "map";
    msg.pose.position.x = target_position(0);
    msg.pose.position.y = target_position(1);
    msg.pose.position.z = target_position(2);
    msg.pose.orientation.w = target_attitude(0);
    msg.pose.orientation.x = target_attitude(1);
    msg.pose.orientation.y = target_attitude(2);
    msg.pose.orientation.z = target_attitude(3);
    setpointPub_.publish(msg);
}


void geometricCtrl::pubRateCommands(const ros::Time &now, const Eigen::Vector4d &cmd, const Eigen::Vector4d &target_attitude) {
    mavros_msgs::AttitudeTarget msg;
    msg.header.stamp = now;
    msg.header.frame_id = "map";
    msg.body_rate.x = cmd(0);
    msg.body_rate.y = cmd(1);
    msg.body_rate.z = cmd(2);
    msg.type_mask = 128;  // Ignore orientation messages
    msg.orientation.w = target_attitude(0);
    msg.orientation.x = target_attitude(1);
    msg.orientation.y = target_attitude(2);
    msg.orientation.z = target_attitude(3);
    msg.thrust = cmd(3);
    cmdPub_.publish(msg);
}


geometry_msgs::PoseStamped geometricCtrl::vector3d2PoseStampedMsg(Eigen::Vector3d &position,
                                                                  Eigen::Vector4d &orientation) {
    geometry_msgs::PoseStamped encode_msg;
    encode_msg.header.stamp = ros::Time::now();
    encode_msg.header.frame_id = "map";
    encode_msg.pose.orientation.w = orientation(0);
    encode_msg.pose.orientation.x = orientation(1);
    encode_msg.pose.orientation.y = orientation(2);
    encode_msg.pose.orientation.z = orientation(3);
    encode_msg.pose.position.x = position(0);
    encode_msg.pose.position.y = position(1);
    encode_msg.pose.position.z = position(2);
    return encode_msg;
}

Eigen::Vector3d geometricCtrl::controlPosition(
        const Eigen::Vector3d &target_pos, 
        const Eigen::Vector3d &target_vel,
        const Eigen::Vector3d &target_acc,
        const double yaw_des) {
    /// Compute BodyRate commands using differential flatness
    /// Controller based on Faessler 2017
    const Eigen::Vector3d a_ref = target_acc;
    const Eigen::Vector4d q_ref = acc2quaternion(a_ref - g_, yaw_des);
    const Eigen::Matrix3d R_ref = quat2RotMatrix(q_ref);
    const Eigen::Vector3d pos_error = pos_ - target_pos;
    const Eigen::Vector3d vel_error = vel_ - target_vel;
    // Position Controller
    const Eigen::Vector3d a_fb = poscontroller(pos_error, vel_error);
    // Rotor Drag compensation
    const Eigen::Vector3d a_rd = R_ref * D_.asDiagonal() * R_ref.transpose() * target_vel;  // Rotor drag
    // Reference acceleration
    const Eigen::Vector3d a_des = a_fb + a_ref - a_rd - g_;
    return a_des;
}

Eigen::Vector4d  geometricCtrl::computeBodyRateCmd(
    const Eigen::Vector3d &a_des, 
    const double &yaw_des,
    Eigen::Vector4d &q_des) 
{
    Eigen::Vector4d bodyrate_cmd;
    q_des = acc2quaternion(a_des, yaw_des);
    if (ctrl_mode_ == ERROR_GEOMETRIC) {
        bodyrate_cmd = geometric_attcontroller(q_des, a_des, quat_);  // Calculate BodyRate
    } else {
        bodyrate_cmd = attcontroller(q_des, a_des, quat_);  // Calculate BodyRate
    }
    
    return bodyrate_cmd;
}

Eigen::Vector3d geometricCtrl::poscontroller(
    const Eigen::Vector3d &pos_error, 
    const Eigen::Vector3d &vel_error) 
{
    Eigen::Vector3d a_fb =
        Kpos_.asDiagonal() * pos_error + Kvel_.asDiagonal() * vel_error;  // feedforward term for trajectory error

    if (a_fb.norm() > max_fb_acc_)
        a_fb = (max_fb_acc_ / a_fb.norm()) * a_fb;  // Clip acceleration if reference is too large

    return a_fb;
}

Eigen::Vector4d geometricCtrl::attcontroller(
        const Eigen::Vector4d &ref_att, 
        const Eigen::Vector3d &ref_acc,
        const Eigen::Vector4d &curr_att) 
{
    // Geometric attitude controller
    // Attitude error is defined as in Brescianini, Dario, Markus Hehn, and Raffaello D'Andrea. Nonlinear quadrocopter
    // attitude control: Technical report. ETH Zurich, 2013.

    Eigen::Vector4d ratecmd;

    const Eigen::Vector4d inverse(1.0, -1.0, -1.0, -1.0);
    const Eigen::Vector4d q_inv = inverse.asDiagonal() * curr_att;
    const Eigen::Vector4d qe = quatMultiplication(q_inv, ref_att);
    ratecmd(0) = (2.0 / attctrl_tau_) * std::copysign(1.0, qe(0)) * qe(1);
    ratecmd(1) = (2.0 / attctrl_tau_) * std::copysign(1.0, qe(0)) * qe(2);
    ratecmd(2) = (2.0 / attctrl_tau_) * std::copysign(1.0, qe(0)) * qe(3);
    // TODO clean quat_ and curr_att
    const Eigen::Matrix3d rotmat = quat2RotMatrix(quat_);
    const Eigen::Vector3d zb = rotmat.col(2);
    ratecmd(3) =
        std::max(0.0, std::min(1.0, norm_thrust_const_ * ref_acc.dot(zb) + norm_thrust_offset_));  // Calculate thrust

    return ratecmd;
}


Eigen::Vector4d geometricCtrl::geometric_attcontroller(
                const Eigen::Vector4d &ref_att, 
                const Eigen::Vector3d &ref_acc,
                const Eigen::Vector4d &curr_att) 
{
    // Geometric attitude controller
    // Attitude error is defined as in Lee, Taeyoung, Melvin Leok, and N. Harris McClamroch. "Geometric tracking control
    // of a quadrotor UAV on SE (3)." 49th IEEE conference on decision and control (CDC). IEEE, 2010.
    // The original paper inputs moment commands, but for offboard control, angular rate commands are sent

    Eigen::Vector4d ratecmd;
    Eigen::Matrix3d rotmat;    // Rotation matrix of current attitude
    Eigen::Matrix3d rotmat_d;  // Rotation matrix of desired attitude
    Eigen::Vector3d error_att;

    rotmat = quat2RotMatrix(curr_att);
    rotmat_d = quat2RotMatrix(ref_att);

    error_att = 0.5 * matrix_hat_inv(rotmat_d.transpose() * rotmat - rotmat.transpose() * rotmat_d);
    ratecmd.head(3) = (2.0 / attctrl_tau_) * error_att;
    // TODO clean quat_ and curr_att
    rotmat = quat2RotMatrix(quat_);
    const Eigen::Vector3d zb = rotmat.col(2);
    ratecmd(3) =
        std::max(0.0, std::min(1.0, norm_thrust_const_ * ref_acc.dot(zb) + norm_thrust_offset_));  // Calculate thrust

    return ratecmd;
}


bool geometricCtrl::loadTrajectory(const std::string &filename) {
    // Cannot update trajectory if already running
    if (trajec_time_ >= 0.0){
        ROS_WARN("Cannot update trajectory while running or idling");
        ROS_WARN("Ignoring the update !!!!");
        return true;
    }

    // Load the csv file
    std::ifstream file(filename);
    if (!(file.is_open() && file.good())) {
        ROS_ERROR("Could not open trajectory file: %s", filename.c_str());
        return false;
    }
    // Read the column names first
    std::string line, colnames;
    std::getline(file, colnames);
    std::stringstream ss(colnames);
    std::vector<std::string> colnames_vec;
    while (std::getline(ss, line, ',')) {
        colnames_vec.push_back(line);
    }

    // ros warn  in a single line the column names
    std::string colnames_str;
    for (auto &colname : colnames_vec) {
        colnames_str += colname + " ";
    }
    ROS_WARN("Loaded trajectory with columns: %s", colnames_str.c_str());

    // Check if the quantity of interest are present
    std::string target_names[] = {"t", "x", "y", "z", "vx", "vy", "vz", "ax", "ay", "az", "yaw"};
    // Find the number of element in target_names
    int num_target_names = sizeof(target_names) / sizeof(target_names[0]);
    // Check if the quantity of interest are present and store the index
    std::vector<int> target_idx;
    // TODO specify the number of element in target_names
    for (int i = 0; i < num_target_names; i++) {
        auto it = std::find(colnames_vec.begin(), colnames_vec.end(), target_names[i]);
        if (it == colnames_vec.end()) {
            ROS_ERROR("Could not find %s in the trajectory file", target_names[i].c_str());
            return false;
        }
        target_idx.push_back(std::distance(colnames_vec.begin(), it));
    }

    // Clear the data
    target_time_.clear();
    target_pos_.clear();
    target_vel_.clear();
    target_acc_.clear();

    // Read the data and store it into target_time_, target_pos_, target_vel_, target_acc_
    while (std::getline(file, line)) {
        std::stringstream _ss(line);
        std::vector<double> row;
        while (std::getline(_ss, line, ',')) {
            // If the string is empty, push a quiet NaN
            if (line.empty()) {
                row.push_back(std::numeric_limits<double>::quiet_NaN());
            }else{
                row.push_back(std::stod(line));
            }
        }
        // If the row is empty, skip it
        if (row.empty()) {
            continue;
        }
        // Check if the row is complete
        if (row.size() != colnames_vec.size() and row.size() != colnames_vec.size()-1) {
            ROS_ERROR("Incomplete row in the trajectory file");
            return false;
        }
        // Store the data
        target_time_.push_back(row[target_idx[0]]);
        target_pos_.push_back(Eigen::Vector3d(row[target_idx[1]], row[target_idx[2]], row[target_idx[3]]));
        target_vel_.push_back(Eigen::Vector3d(row[target_idx[4]], row[target_idx[5]], row[target_idx[6]]));
        target_acc_.push_back(Eigen::Vector3d(row[target_idx[7]], row[target_idx[8]], row[target_idx[9]]));
        target_yaw_.push_back(row[target_idx[10]]);
        // // ros warn time, position, and yaw
        // ROS_WARN("time: %f, position: %f %f %f, yaw: %f", target_time_.back(), target_pos_.back()(0), target_pos_.back()(1), target_pos_.back()(2), target_yaw_.back());
    }
    // Warn trajectory load and number of points
    ROS_WARN("Loaded trajectory with %d points", (int)target_time_.size());
    return true;
}

bool geometricCtrl::loadParameters(const std::string &filename)
{
    std::ifstream file(filename);
    if (!(file.is_open() && file.good())) {
        ROS_ERROR("Could not open parameters file: %s", filename.c_str());
        return false;
    }else{
        ROS_INFO("Opened parameter file: %s", filename.c_str());
    }
    // The parameters are saved as a yaml file
    // Load the yaml file
    YAML::Node config = YAML::LoadFile(filename);
    if (config.IsNull()) {
        ROS_ERROR("Could not open parameter file: %s", filename.c_str());
        return false;
    }
    // Update the attributes of attctrl_tau_, norm_thrust_const_, norm_thrust_offset_, max_fb_acc_, gravity_,
    if (config["attctrl_tau"]) {
        attctrl_tau_ = config["attctrl_tau"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated attctrl_tau to %f", attctrl_tau_);
    }
    if (config["norm_thrust_const"]) {
        norm_thrust_const_ = config["norm_thrust_const"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated norm_thrust_const to %f", norm_thrust_const_);
    }
    if (config["norm_thrust_offset"]) {
        norm_thrust_offset_ = config["norm_thrust_offset"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated norm_thrust_offset to %f", norm_thrust_offset_);
    }
    if (config["max_acc"]) {
        max_fb_acc_ = config["max_acc"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated max_acc to %f", max_fb_acc_);
    }
    if (config["gravity"]) {
        gravity_ = config["gravity"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated gravity to %f", gravity_);
    }
    // drag_dx
    if (config["drag_dx"]) {
        dx_ = config["drag_dx"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated drag_dx to %f", dx_);
    }
    // drag_dy
    if (config["drag_dy"]) {
        dy_ = config["drag_dy"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated drag_dy to %f", dy_);
    }
    // drag_dz
    if (config["drag_dz"]) {
        dz_ = config["drag_dz"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated drag_dz to %f", dz_);
    }
    // Kp_x
    if (config["Kp_x"]) {
        Kpos_x_ = config["Kp_x"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated Kp_x to %f", Kpos_x_);
    }
    // Kp_y
    if (config["Kp_y"]) {
        Kpos_y_ = config["Kp_y"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated Kp_y to %f", Kpos_y_);
    }
    // Kp_z
    if (config["Kp_z"]) {
        Kpos_z_ = config["Kp_z"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated Kp_z to %f", Kpos_z_);
    }
    // Kv_x
    if (config["Kv_x"]) {
        Kvel_x_ = config["Kv_x"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated Kv_x to %f", Kvel_x_);
    }
    // Kv_y
    if (config["Kv_y"]) {
        Kvel_y_ = config["Kv_y"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated Kv_y to %f", Kvel_y_);
    }
    // Kv_z
    if (config["Kv_z"]) {
        Kvel_z_ = config["Kv_z"].as<double>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated Kv_z to %f", Kvel_z_);
    }
    // ctrl_mode
    if (config["ctrl_mode"]) {
        ctrl_mode_ = config["ctrl_mode"].as<int>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated ctrl_mode to %d", ctrl_mode_);
    }
    // feedthrough_enable_
    if (config["feedthrough_enable"]) {
        feedthrough_enable_ = config["feedthrough_enable"].as<bool>();
        // Warn the user that the parameter is updated
        ROS_WARN("Updated feedthrough_enable to %d", feedthrough_enable_);
    }

    // Initialize array version of some of these parameters
    g_ << 0.0, 0.0, -gravity_;
    Kpos_ << -Kpos_x_, -Kpos_y_, -Kpos_z_;
    Kvel_ << -Kvel_x_, -Kvel_y_, -Kvel_z_;
    D_ << dx_, dy_, dz_;

    // last_time_ = ros::Time::now();
    if (trajec_time_ >= 0) return true;
    
    trajec_time_ = -1.0;
    current_stage_ = 0;
    run_trajectory_ = false;

    return true;
}

void geometricCtrl::startTrajectory(int start)
{
    // Check if target_time_, target_pos_, target_vel_, target_acc_ are non empty
    if (target_time_.size() == 0 || target_pos_.size() == 0 || target_vel_.size() == 0 || target_acc_.size() == 0){
        ROS_ERROR("Target trajectory is empty");
        return;
    }
    if (run_trajectory_ && start == 1) {
        ROS_WARN("Trajectory already running");
        return;
    }

    run_trajectory_ = start == 1;
    trajec_time_ = start >= 1 ? 0.0 : -1.0;
    current_stage_ = 0;
    // Warn print run_trajectory_, trajec_time_, current_stage_
    ROS_WARN("run_trajectory_ = %d, trajec_time_ = %f, current_stage_ = %d", run_trajectory_, trajec_time_, current_stage_);

}

int main(int argc, char** argv) {
    ros::init(argc, argv, "geometric_controller");
    ros::NodeHandle nh;

    geometricCtrl controller(nh);
    ros::spin();
    // controller.loadTrajectory("/home/franckdjeumou/catkin_ws/src/mpc4px4/mpc4px4/trajectory_generation/my_trajs/test_traj_gen.csv");
    // controller.loadParameters("/home/franckdjeumou/catkin_ws/src/mpc4px4/launch/gm_iris.yaml");

    // controller.startTrajectory(0);
    // controller.controlLoopBody();
    return 0;
}