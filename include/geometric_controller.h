/****************************************************************************
 *
 *   Copyright (c) 2018-2022 Franck Djeumou. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in
 *    the documentation and/or other materials provided with the
 *    distribution.
 * 3. Neither the name PX4 nor the names of its contributors may be
 *    used to endorse or promote products derived from this software
 *    without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
 * FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
 * COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
 * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
 * BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 * OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED
 * AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
 * ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 *
 ****************************************************************************/
/**
 * Modified from @author Jaeyoung Lim <jalim@ethz.ch>
 * Geometric controller
 *
 * @author Franck Djeumou
 */

#ifndef GEOMETRIC_CONTROLLER_H
#define GEOMETRIC_CONTROLLER_H

#include <ros/ros.h>
#include <ros/subscribe_options.h>
#include <tf/transform_broadcaster.h>


#include <stdio.h>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <sstream>
#include <string>

#include <mavros_msgs/AttitudeTarget.h>
#include <mavros_msgs/CommandBool.h>
#include <mavros_msgs/MPCFullState.h>
#include <mavros_msgs/CompanionProcessStatus.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <std_msgs/Float32.h>

#include <std_srvs/SetBool.h>

#include <geometry_msgs/PoseStamped.h>
#include <geometry_msgs/Twist.h>
#include <geometry_msgs/TwistStamped.h>
#include <Eigen/Dense>

// Import YAML
#include <yaml-cpp/yaml.h>

#include <mpc4px4/LoadTrajAndParams.h>
#include <mpc4px4/FollowTraj.h>


#define ERROR_QUATERNION  1
#define ERROR_GEOMETRIC   2

static Eigen::Matrix3d matrix_hat(const Eigen::Vector3d &v) {
    Eigen::Matrix3d m;
    // Sanity checks on M
    m << 0.0, -v(2), v(1), v(2), 0.0, -v(0), -v(1), v(0), 0.0;
    return m;
}

static Eigen::Vector3d matrix_hat_inv(const Eigen::Matrix3d &m) {
    Eigen::Vector3d v;
    // TODO: Sanity checks if m is skew symmetric
    v << m(7), m(2), m(3);
    return v;
}

Eigen::Vector3d toEigen(const geometry_msgs::Point &p) {
    Eigen::Vector3d ev3(p.x, p.y, p.z);
    return ev3;
}

inline Eigen::Vector3d toEigen(const geometry_msgs::Vector3 &v3) {
    Eigen::Vector3d ev3(v3.x, v3.y, v3.z);
    return ev3;
}

Eigen::Vector4d quatMultiplication(const Eigen::Vector4d &q, const Eigen::Vector4d &p) {
    Eigen::Vector4d quat;
    quat << p(0) * q(0) - p(1) * q(1) - p(2) * q(2) - p(3) * q(3), p(0) * q(1) + p(1) * q(0) - p(2) * q(3) + p(3) * q(2),
        p(0) * q(2) + p(1) * q(3) + p(2) * q(0) - p(3) * q(1), p(0) * q(3) - p(1) * q(2) + p(2) * q(1) + p(3) * q(0);
  return quat;
}

Eigen::Matrix3d quat2RotMatrix(const Eigen::Vector4d &q) {
    Eigen::Matrix3d rotmat;
    rotmat << q(0) * q(0) + q(1) * q(1) - q(2) * q(2) - q(3) * q(3), 2 * q(1) * q(2) - 2 * q(0) * q(3),
        2 * q(0) * q(2) + 2 * q(1) * q(3),

        2 * q(0) * q(3) + 2 * q(1) * q(2), q(0) * q(0) - q(1) * q(1) + q(2) * q(2) - q(3) * q(3),
        2 * q(2) * q(3) - 2 * q(0) * q(1),

        2 * q(1) * q(3) - 2 * q(0) * q(2), 2 * q(0) * q(1) + 2 * q(2) * q(3),
        q(0) * q(0) - q(1) * q(1) - q(2) * q(2) + q(3) * q(3);
    return rotmat;
}

Eigen::Vector4d rot2Quaternion(const Eigen::Matrix3d &R) {
    Eigen::Vector4d quat;
    double tr = R.trace();
    if (tr > 0.0) {
        double S = sqrt(tr + 1.0) * 2.0;  // S=4*qw
        quat(0) = 0.25 * S;
        quat(1) = (R(2, 1) - R(1, 2)) / S;
        quat(2) = (R(0, 2) - R(2, 0)) / S;
        quat(3) = (R(1, 0) - R(0, 1)) / S;
    } else if ((R(0, 0) > R(1, 1)) & (R(0, 0) > R(2, 2))) {
        double S = sqrt(1.0 + R(0, 0) - R(1, 1) - R(2, 2)) * 2.0;  // S=4*qx
        quat(0) = (R(2, 1) - R(1, 2)) / S;
        quat(1) = 0.25 * S;
        quat(2) = (R(0, 1) + R(1, 0)) / S;
        quat(3) = (R(0, 2) + R(2, 0)) / S;
    } else if (R(1, 1) > R(2, 2)) {
        double S = sqrt(1.0 + R(1, 1) - R(0, 0) - R(2, 2)) * 2.0;  // S=4*qy
        quat(0) = (R(0, 2) - R(2, 0)) / S;
        quat(1) = (R(0, 1) + R(1, 0)) / S;
        quat(2) = 0.25 * S;
        quat(3) = (R(1, 2) + R(2, 1)) / S;
    } else {
        double S = sqrt(1.0 + R(2, 2) - R(0, 0) - R(1, 1)) * 2.0;  // S=4*qz
        quat(0) = (R(1, 0) - R(0, 1)) / S;
        quat(1) = (R(0, 2) + R(2, 0)) / S;
        quat(2) = (R(1, 2) + R(2, 1)) / S;
        quat(3) = 0.25 * S;
    }
    return quat;
}

Eigen::Vector4d acc2quaternion(
    const Eigen::Vector3d &vector_acc, 
    const double &yaw) 
{
    Eigen::Vector4d quat;
    Eigen::Vector3d zb_des, yb_des, xb_des, proj_xb_des;
    Eigen::Matrix3d rotmat;

    proj_xb_des << std::cos(yaw), std::sin(yaw), 0.0;

    zb_des = vector_acc / vector_acc.norm();
    yb_des = zb_des.cross(proj_xb_des) / (zb_des.cross(proj_xb_des)).norm();
    xb_des = yb_des.cross(zb_des) / (yb_des.cross(zb_des)).norm();

    rotmat << xb_des(0), yb_des(0), zb_des(0), xb_des(1), yb_des(1), zb_des(1), xb_des(2), yb_des(2), zb_des(2);
    quat = rot2Quaternion(rotmat);
    return quat;
}


using namespace std;
using namespace Eigen;

class geometricCtrl
{
    private:
        ros::NodeHandle nh_;

        ros::Subscriber mav_full_state_sub_;

        ros::Publisher cmdPub_;
        ros::Publisher setpointPub_;

        // Server to set the trajectories and gains
        ros::ServiceServer setTrajectorySrv_;

        // Server to start the trajectory
        ros::ServiceServer startTrajectorySrv_;

        // TODO: find a way to store the csv trajectory and parse it
        std::vector<double> target_time_; // Time in seconds
        std::vector<Eigen::Vector3d> target_pos_; // trajectory in ENU
        std::vector<Eigen::Vector3d> target_vel_; // trajectory in ENU
        std::vector<Eigen::Vector3d> target_acc_; // trajectory in ENU
        std::vector<double> target_yaw_; // YAW

        // Store the trajectory path to find in the configuration file
        std::string trajectory_path_;
        // Store the configuration file for the controller
        std::string config_dir_;
        std::string config_name_;

        // Desired pose setpoint if not in trajectory mode
        geometry_msgs::PoseStamped target_sp_;
        bool pose_ctrl_;

        double trajec_time_; // time of the trajectory in seconds
        int current_stage_; // current stage of the trajectory
        bool run_trajectory_; // flag to run the trajectory

        // Current state of the MAV
        Eigen::Vector3d pos_; // ENU
        Eigen::Vector3d vel_; // ENU local
        Eigen::Vector4d quat_; // FLU to ENU
        Eigen::Vector3d omega_; // ENU local
        ros::Time last_time_;

        // Parameters of the controller
        double attctrl_tau_;
        double norm_thrust_const_, norm_thrust_offset_;
        double max_fb_acc_;
        double gravity_;
        double dx_, dy_, dz_;
        double Kpos_x_, Kpos_y_, Kpos_z_, Kvel_x_, Kvel_y_, Kvel_z_;
        int ctrl_mode_;
        bool feedthrough_enable_;

        Eigen::Vector3d Kpos_, Kvel_, D_;
        Eigen::Vector3d g_;

        // Publisher functions
        void pubRateCommands(const ros::Time &now, const Eigen::Vector4d &cmd, const Eigen::Vector4d &target_attitude);
        void pubReferencePose(const ros::Time &now, const Eigen::Vector3d &target_position, const Eigen::Vector4d &target_attitude);

        // Subscriber function
        void mavStateCallback(const mavros_msgs::MPCFullState::ConstPtr &msg);


        // Controller sub-functions
        Eigen::Vector4d geometric_attcontroller(const Eigen::Vector4d &ref_att, const Eigen::Vector3d &ref_acc, const Eigen::Vector4d &curr_att);
        Eigen::Vector4d attcontroller(const Eigen::Vector4d &ref_att, const Eigen::Vector3d &ref_acc, const Eigen::Vector4d &curr_att);
        Eigen::Vector3d poscontroller(const Eigen::Vector3d &pos_error, const Eigen::Vector3d &vel_error);
        Eigen::Vector4d computeBodyRateCmd(const Eigen::Vector3d &target_acc, const double &yaw_des, Eigen::Vector4d &q_des);
        Eigen::Vector3d controlPosition(const Eigen::Vector3d &target_pos, const Eigen::Vector3d &target_vel, const Eigen::Vector3d &target_acc, const double yaw_des);


        // Utility functions
        geometry_msgs::PoseStamped vector3d2PoseStampedMsg(Eigen::Vector3d &position, Eigen::Vector4d &orientation);
        inline Eigen::Vector3d toEigen(const geometry_msgs::Point& p) {
            Eigen::Vector3d ev3(p.x, p.y, p.z);
            return ev3;
        }
        inline Eigen::Vector3d toEigen(const geometry_msgs::Vector3& v3) {
            Eigen::Vector3d ev3(v3.x, v3.y, v3.z);
            return ev3;
        }

        bool extractSetpointFromTrajectory(const double dt, Eigen::Vector3d &pos, Eigen::Vector3d &vel, Eigen::Vector3d &acc, double &targetYaw);

        // Functions foe the two services
        bool setTrajectoryAndParams(mpc4px4::LoadTrajAndParams::Request &req, mpc4px4::LoadTrajAndParams::Response &res);
        bool triggerTrajectory(mpc4px4::FollowTraj::Request &req, mpc4px4::FollowTraj::Response &res);

    public:
        geometricCtrl(const ros::NodeHandle& nh);
        virtual ~ geometricCtrl();

        bool loadTrajectory(const std::string &filename);
        bool loadParameters(const std::string &filename);

        // Start the trajectory
        void startTrajectory(int start);
        void controlLoopBody();

};

#endif // GEOMETRIC_CONTROLLER_H