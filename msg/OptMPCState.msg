# Publish on a ROS topic information related to the MPC-based Accelerated Gradient Descent Solver
# The ROS time at which the message has been published
time stamp

# The average linesearch for learning rate computation
float32 avg_linesearch
# The average stepsize used
float32 avg_stepsize

# The latest stepsize used
float32 stepsize

# The norm of the gradient
float32 grad_norm

# The initial cost value
float32 cost_init
# The cost value after the descent steps
float32 opt_cost

# The number of steps until convergence
uint16 num_steps
# The compute time until convergence
float32 solve_time

# The time it takes to process one state measurement and return the control input
float32 callback_dt

# The time it takes between each state messages
float32 state_dt

# The state of the MPC controller
string ctrl_state

# Which index is sent from the set of solution returned by the MPC
uint8 mpc_indx