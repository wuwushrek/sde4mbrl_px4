import jax
import jax.numpy as jnp

from mpc4px4.helpers import quatmult as quatmult_gen
from mpc4px4.helpers import quat_rotatevector as quat_rotatevector_gen
from mpc4px4.helpers import quat_rotatevectorinv as quat_rotatevectorinv_gen

quatmult = lambda a, b: quatmult_gen(a, b, jnp)
quat_rotatevector = lambda q, v: quat_rotatevector_gen(q, v, jnp)
quat_rotatevectorinv = lambda q, v: quat_rotatevectorinv_gen(q, v, jnp)

import haiku as hk

def get_pos(state):
    """Get the position from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The pose of the drone.
    """
    return state[0:3]

def set_pos(state, pos):
    """Set the position from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The pose of the drone.
    """
    return state.at[0:3].set(pos)

def get_vel(state):
    """Get the velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The velocity of the drone.
    """
    return state[3:6]

def set_vel(state, vel):
    """Set the velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The velocity of the drone.
    """
    return state.at[3:6].set(vel)

def get_quat(state):
    """Get the quaternion from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The quaternion of the drone.
    """
    return state[6:10]

def set_quat(state, q):
    """Set the quaternion from the state.
       The convention is q0 + q1*i + q2*j + q3*k
    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The quaternion of the drone.
    """
    return state.at[6:10].set(q)

def get_ang_vel(state):
    """Get the angular velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The angular velocity of the drone.
    """
    return state[10:13]

def set_ang_vel(state, w):
    """Set the angular velocity from the state.

    Args:
        state (jax.numpy.ndarray): The state of the drone.

    Returns:
        jax.numpy.ndarray: The angular velocity of the drone.
    """
    return state.at[10:13].set(w)


# TODO: Later on make it derive from  ControlledSDE class instead
# and override the init method accordingly
class QuadDynamics(hk.Module):
    """The nonlinear dynamics of the quadrotor.
        All parameters of the model are assumed to be nonnegative
    """
    def __init__(self, init_params={'gravity': 9.81, 'motor_model' : 'quadratic'}, name=None):

        super().__init__(name=name)
        # A dictionary cpntaining Initial parameters estimation of the quadrotor
        # such as mass, inertia, etc.
        self.init_params = init_params
        assert 'motor_model' in self.init_params, "The motor model must be specified"
    
    def get_param(self, param_name):
        """Get the value of the parameter with the given name.

        Args:
            param_name (str): The name of the parameter.

        Returns:
            float: The value of the parameter.
        """
        init_value = self.init_params.get(param_name, 1e-4) # Small non-zero value
        return hk.get_parameter(param_name, shape=(), init=hk.initializers.Constant(init_value))
    
    
    def translational_dynamics(self, x, Fz, Fres=None):
        """
        Given the state x and the thrust and residual forces not explained by model, 
        This is ENU coordinate frame with z up.
        return the derivative of the state
        """
        # Get the mass
        m = self.get_param('mass')
        # Get the gravity
        g = self.init_params.get('gravity', 9.81)

        # Get the total thrust
        F = jnp.array([0., 0., Fz])
        if Fres is not None:
            F += Fres
        
        # Rotate the thrust into the inertial frame
        q = get_quat(x)
        F = quat_rotatevector(q, F)

        # Compute the acceleration
        a = (F / m) + jnp.array([0., 0., -g])

        # Get the velocity
        pos_dot = get_vel(x)

        # Return the derivative of the state
        return pos_dot, a


    def rotational_dynamics(self, x, Mxyz, Mres=None):
        """
        Given the state x and the moments and residual moments not explained by model,
        This is in the body frame
        return the derivative of the state
        """
        # Get the inertia
        I = jnp.array([self.get_param('Ixx'), self.get_param('Iyy'), self.get_param('Izz')])

        # Get the total moment
        M = Mxyz
        if Mres is not None:
            assert Mres.shape[0] == 3 or Mres.shape[0] == 6, "The residual moment must be a 3D vector or a 6D vector"
            if Mres.shape[0] == 6:
                M *= Mres[3:]
            M += Mres[:3]
        
        # Get the angular velocity
        ang_vel = get_ang_vel(x)

        # Compute the cross product between the angular velocity and the inertia and the angular velocity
        ang_vel_cross = jnp.cross(ang_vel, I * ang_vel)

        # Update the total moment
        M -= ang_vel_cross

        # Compute the angular acceleration
        ang_acc = M / I

        # Get the quaternion derivative
        quat_dot = 0.5 * quatmult(get_quat(x), jnp.array([0., ang_vel[0], ang_vel[1], ang_vel[2]]))

        return quat_dot, ang_acc
    
    def body_thrust_from_actuator(self, w_vec, ext_thrust=jnp.array([1., 1., 1., 1.])):
        """Compute the thrust in the body frame from the motor speed.

        Args:
            w_vec (jax.numpy.ndarray): A vector of rotational speed of each motor.
            We assume that the motors follow an X configuration as follows:
            1 : Front right
            3 : Front left
            2 : Back left
            4 : Back right

        Returns:
            jax.numpy.ndarray: The thrust in the body frame.
        """
        # Get the thrust coefficient
        Ct = self.get_param('Ct')
        return Ct * jnp.sum(jnp.square(w_vec) * ext_thrust)
    
    def body_moment_from_actuator(self, w_vec, x=None, ext_thrust=jnp.array([1., 1., 1., 1.])):
        """Compute the moment in the body frame from the motor speed.

        Args:
            w_vec (jax.numpy.ndarray): A vector of rotational speed of each motor.
            We assume that the motors follow an X configuration as follows:
            1 : Front right
            3 : Front left
            2 : Back left
            4 : Back right

        Returns:
            jax.numpy.ndarray: The moment in the body frame.
        """
        # Get the thrust coefficient
        Ct = self.get_param('Ct')
        # Get the drag coefficient
        Cd = self.get_param('Cd')
        # Get the lengths Dm_x and Dm_y
        Dm_x = self.get_param('Dm_x')
        Dm_y = self.get_param('Dm_y')
        # Square of the motor speed
        w1_2, w2_2, w3_2, w4_2 = jnp.square(w_vec)
        # Compute the moment along z
        Mz = (w3_2 + w4_2 - w1_2 - w2_2) * Cd
        # Include the external thrust constraints
        w1_2, w2_2, w3_2, w4_2 = w1_2 * ext_thrust[0], w2_2 * ext_thrust[1], w3_2 * ext_thrust[2], w4_2 * ext_thrust[3]
        # Compute the moment along x
        Mx = (w2_2 + w3_2 - w1_2 - w4_2) * Dm_y * Ct
        # Compute the moment along y
        My = (w2_2 + w4_2 - w1_2 - w3_2) * Dm_x * Ct

        # Extra terms due to the angular velocity
        if x is not None:
            w1, w2, w3, w4 = w_vec * self.init_params['max_rotor_vel']
            ang_vel = get_ang_vel(x)
            Iz = self.get_param('Izz')
            ang_vel_x = ang_vel[0]
            ang_vel_y = ang_vel[1]
            gyro_effect = Iz * (w3 - w1 + w4 - w2)
            Mx_extra = ang_vel_y * gyro_effect
            My_extra = -ang_vel_x * gyro_effect
            Mx += Mx_extra
            My += My_extra
        
        return jnp.array([Mx, My, Mz])
    
    def get_single_motor_vel(self, pwm_in):
        return self.single_motor_model(pwm_in) * self.init_params['max_rotor_vel']
    
    def single_motor_model(self, pwm_in):
        """Compute the rotational speed of a motor given the PWM input.

        Args:
            pwm_in (jax.numpy.ndarray): The PWM input already normalized between 0 and 1.

        Returns:
            jax.numpy.ndarray: The rotational speed of the motor.
        """
        # Get the prior on the motor speed model
        motor_model = self.init_params.get('motor_model', 'linear')
        if motor_model == 'linear':
            # Model parameters
            c1 = hk.get_parameter('lin_c1', shape=(), init=hk.initializers.Constant(0.001))
            return c1 * pwm_in
        
        if motor_model == 'quadratic':
            # Model parameters
            c1 = hk.get_parameter('quad_c1', shape=(), init=hk.initializers.Constant(0.001))
            c2_eps = hk.get_parameter('quad_c2', shape=(), init=hk.initializers.Constant(0.))
            # We need to make sure that the function is increasing
            # For a quadratic function _ + c1 *x + c2 * x^2, 
            #  Since pwm_in is between 0 and 1, we need c1 non-negative and c2 > -c1 / 2
            c2 = -0.5 * c1 + c2_eps
            return c1 * pwm_in + c2 * jnp.square(pwm_in)
        
        if motor_model == 'cubic':
            # Model parameters
            c1 = hk.get_parameter('cubic_c1', shape=(), init=hk.initializers.Constant(0.001))
            c2_eps = hk.get_parameter('cubic_c2', shape=(), init=hk.initializers.Constant(0.))
            c3 = hk.get_parameter('cubic_c3', shape=(), init=hk.initializers.Constant(0.))
            # We need to make sure that the function is increasing
            # For a cubic function _ + c1 *x + c2 * x^2 + c3 * x^3,
            c2 = jnp.sqrt(4*c1*c3 +1.) - c2_eps # To avoid differentiability issues
            return c1 * pwm_in + c2 * jnp.square(pwm_in) + c3 * jnp.power(pwm_in, 3)
        
        if motor_model == 'sigmoid_quad':
            # Model parameters
            c1 = hk.get_parameter('sig_c1', shape=(), init=hk.initializers.Constant(0.001))
            a1_eps = hk.get_parameter('sig_a1', shape=(), init=hk.initializers.Constant(0.))
            a2 = hk.get_parameter('sig_a2', shape=(), init=hk.initializers.Constant(0.001))
            a3 = hk.get_parameter('sig_a3', shape=(), init=hk.initializers.Constant(0.))
            # Second order polynomial
            a1 = -0.5 * a2 + a1_eps
            poly_v = a1 * jnp.square(pwm_in) + a2 * pwm_in - a3
            return c1 * jax.nn.sigmoid(poly_v)
        
        if motor_model == "sigmoid_linear":
            # Model parameters
            c1 = hk.get_parameter('sig_c1', shape=(), init=hk.initializers.Constant(0.001))
            a1 = hk.get_parameter('sig_a1', shape=(), init=hk.initializers.Constant(0.001))
            a2 = hk.get_parameter('sig_a2', shape=(), init=hk.initializers.Constant(0.))
            # Linear function
            lin_v = a1 * pwm_in - a2
            return c1 * jax.nn.sigmoid(lin_v)
        
        if motor_model == "sigmoid_cubic":
            c1 = hk.get_parameter('cubic_c1', shape=(), init=hk.initializers.Constant(0.001))
            c2_eps = hk.get_parameter('cubic_c2', shape=(), init=hk.initializers.Constant(0.))
            c3 = hk.get_parameter('cubic_c3', shape=(), init=hk.initializers.Constant(0.))
            c4 = hk.get_parameter('cubic_c4', shape=(), init=hk.initializers.Constant(0.))
            # We need to make sure that the function is increasing
            # For a cubic function _ + c1 *x + c2 * x^2 + c3 * x^3,
            c2 = jnp.sqrt(4*c1*c3 +1.0) - c2_eps # To avoid differentiability issues
            return 1.1 * jax.nn.sigmoid(c1 * pwm_in + c2 * jnp.square(pwm_in) + c3 * jnp.power(pwm_in, 3) - c4)
        
        if motor_model == "crazy":
            # Create 2 quadratic functions, a linear function and a cubic function, and multiply them
            # Model parameters
            c1 = hk.get_parameter('crazy_c1', shape=(), init=hk.initializers.Constant(0.001))
            c2_eps = hk.get_parameter('crazy_c2', shape=(), init=hk.initializers.Constant(0.))
            c3 = hk.get_parameter('crazy_c3', shape=(), init=hk.initializers.Constant(0.))
            c4 = hk.get_parameter('crazy_c4', shape=(), init=hk.initializers.Constant(0.))
            c5 = hk.get_parameter('crazy_c5', shape=(), init=hk.initializers.Constant(0.))
            c6 = hk.get_parameter('crazy_c6', shape=(), init=hk.initializers.Constant(0.))
            c7 = hk.get_parameter('crazy_c7', shape=(), init=hk.initializers.Constant(0.))
            c8 = hk.get_parameter('crazy_c8', shape=(), init=hk.initializers.Constant(0.))
            c9 = hk.get_parameter('crazy_c9', shape=(), init=hk.initializers.Constant(0.))
            c10 = hk.get_parameter('crazy_c10', shape=(), init=hk.initializers.Constant(0.))
            c11 = hk.get_parameter('crazy_c11', shape=(), init=hk.initializers.Constant(0.))
            c12 = hk.get_parameter('crazy_c12', shape=(), init=hk.initializers.Constant(0.))
            # Create the quadratic functions
            c2 = -0.5 * c1 + c2_eps
            c6 = -0.5 * c5 + c6
            quad_1 = c1 * pwm_in + c2 * jnp.square(pwm_in) + c3
            quad_2 = c5 * pwm_in + c6 * jnp.square(pwm_in) + c4
            # Create the linear function
            lin = c7 * pwm_in + c8
            # Create the cubic function
            c10 = jnp.sqrt(4*c9*c11 +1.0) - c10 # To avoid differentiability issues
            cubic = c9 * pwm_in + c10 * jnp.square(pwm_in) + c11 * jnp.power(pwm_in, 3) + c12
            return quad_1 * quad_2 * lin * cubic
        
        if motor_model == "fourrier":
            # Create 8 coefficients for a fourrier series
            # Model parameters
            c1 = hk.get_parameter('fourrier_c1', shape=(), init=hk.initializers.Constant(0.001))
            c2 = hk.get_parameter('fourrier_c2', shape=(), init=hk.initializers.Constant(0.))
            c3 = hk.get_parameter('fourrier_c3', shape=(), init=hk.initializers.Constant(0.))
            c4 = hk.get_parameter('fourrier_c4', shape=(), init=hk.initializers.Constant(0.))
            c5 = hk.get_parameter('fourrier_c5', shape=(), init=hk.initializers.Constant(0.))
            c6 = hk.get_parameter('fourrier_c6', shape=(), init=hk.initializers.Constant(0.))
            c7 = hk.get_parameter('fourrier_c7', shape=(), init=hk.initializers.Constant(0.))
            c8 = hk.get_parameter('fourrier_c8', shape=(), init=hk.initializers.Constant(0.))
            c9 = hk.get_parameter('fourrier_c9', shape=(), init=hk.initializers.Constant(0.))
            # Create the fourrier series
            return (c2-c1)*jnp.sin(c3*pwm_in) + (c5-c4)*jnp.sin(c6*pwm_in) + (c8-c7)*jnp.sin(c9*pwm_in)

        
        # If we reach this point, we have an unknown model
        raise ValueError(f"Unknown motor model {motor_model}")
    
    def motor_model_constraints(self):
        """Get the constraints on the motor model parameters.

        Returns:
            dict: The constraints on the motor model parameters.
        """
        # # Get the prior on the motor speed model
        # motor_model = self.init_params.get('motor_model', 'linear')
        # if motor_model in ['linear', 'quadratic', 'cubic']:
        #     return jnp.array(0.0)
        # # For S-shape like or converging functions,
        # # we minimize the distance to the extreme
        # # Assuming 100 is a good "large enough" value
        # return self.single_motor_model(1.0) - self.single_motor_model(100.0)
        return jnp.square(1.0 - self.single_motor_model(1.0)) + jnp.square(self.single_motor_model(0.0))
    
    def vector_field(self, x, pwm_in, Fres=None, Mres=None, ext_thrust=jnp.array([1.0, 1.0, 1.0, 1.0])):
        """Compute the dynamics of the quadrotor.

        Args:
            x (jax.numpy.ndarray): The state of the quadrotor.
            u (jax.numpy.ndarray): The input of the quadrotor.
            Fres (jax.numpy.ndarray, optional): The residual force. Defaults to None.
            Mres (jax.numpy.ndarray, optional): The residual moment. Defaults to None.
            ext_thrust (jax.numpy.ndarray, optional): The external thrust, e.g. ground effect. 
                Defaults to jnp.array([1.0, 1.0, 1.0, 1.0]). This value must be between 0 and 1.

        Returns:
            jax.numpy.ndarray: The derivative of the state.
        """
        assert pwm_in.shape[0] == 4, "The input should be a 4D vector"
        # Rotational speed -> Adimensional quantities (They are assumed to be divied by the max_rotor_vel)
        w1 = self.single_motor_model(pwm_in[0])
        w2 = self.single_motor_model(pwm_in[1])
        w3 = self.single_motor_model(pwm_in[2])
        w4 = self.single_motor_model(pwm_in[3])
        w_vect = jnp.array([w1, w2, w3, w4])
        
        # _Fres = None
        # # Residual force and moments
        # if Fres is not None:
        #     _Fres = Fres(x, w_vect)
        
        # _Mres = None
        # if Mres is not None:
        #     _Mres = Mres(x, w_vect)

        # Compute the thrust
        thrust = self.body_thrust_from_actuator(w_vect, ext_thrust=ext_thrust)
        # Compute the moments and consider gyro effects if enable
        moment = self.body_moment_from_actuator(w_vect, x if self.init_params.get('gyro_effect', False) else None, ext_thrust=ext_thrust)
        # Compute the translational dynamics
        pos_dot, v_dot = self.translational_dynamics(x, thrust, Fres)
        # Compute the rotational dynamics
        quat_dot, omega_dot = self.rotational_dynamics(x, moment, Mres)
        # Return the derivative
        return jnp.concatenate((pos_dot, v_dot, quat_dot, omega_dot))
    
    def param_deviation_from_init(self):
        """Compute the deviation of the parameters from the initial values.

        Returns:
            float: square sum of the deviations
        """
        dev_value = jnp.array(0.0)
        dev_value += self.init_params['conf_mass'] * (self.init_params['mass'] - self.get_param('mass')) ** 2
        dev_value += self.init_params['conf_Dm'] * (self.init_params['Dm_x'] - self.get_param('Dm_x')) ** 2
        dev_value += self.init_params['conf_Dm'] * (self.init_params['Dm_y'] - self.get_param('Dm_y')) ** 2
        dev_value += self.init_params['conf_Ct'] * (self.init_params['Ct'] - self.get_param('Ct')) ** 2
        dev_value += self.init_params['conf_Cd'] * (self.init_params['Cd'] - self.get_param('Cd')) ** 2

        dev_value += self.init_params['conf_I'] * jnp.sum((self.init_params['Ixx'] - self.get_param('Ixx')) ** 2)
        dev_value += self.init_params['conf_I'] * jnp.sum((self.init_params['Iyy'] - self.get_param('Iyy')) ** 2)
        dev_value += self.init_params['conf_I'] * jnp.sum((self.init_params['Izz'] - self.get_param('Izz')) ** 2)
        # # Additional constraints on the Inertia
        # # Ixx + Iyy >= Izz, Ixx + Izz >= Iyy, Iyy + Izz >= Ixx
        # Ixx, Iyy, Izz = self.get_param('Ixx'), self.get_param('Iyy'), self.get_param('Izz')
        # dev_value += 1e3 * jnp.where(Izz > Ixx + Iyy,  Izz - Ixx - Iyy, jnp.array(0.0))
        # dev_value += 1e3 * jnp.where(Iyy > Ixx + Izz,  Iyy - Ixx - Izz, jnp.array(0.0))
        # dev_value += 1e3 * jnp.where(Ixx > Iyy + Izz,  Ixx - Iyy - Izz, jnp.array(0.0))
        return dev_value