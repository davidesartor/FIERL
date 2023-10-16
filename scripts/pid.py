import numpy as np

class PID: 
    """
    Implement decoupled digital PID control for multidimensional systems. 
    u(k) = kp * [e(k) + Ts/Ti * sum(e(i)) + Td/Ts * (e(k)-e(k-1))]
    ki = kp * Ts/Ti
    kd = kp * Td/Ts
    """

    def __init__(self, kp:np.ndarray, ki:np.ndarray, kd:np.ndarray, input_dim:int, output_dim:int): 
        """
        Args: 
            kp (np.ndarray): vector of shape (n,) with proportional gains in discrete-time
            ki (np.ndarray): vector of shape (n,) with integral gains in discrete-time
            kd (np.ndarray): vector of shape (n,) with derivative gains in discrete-time
        """
        self.kp = np.diag(kp)[:input_dim, :output_dim]
        self.ki = np.diag(ki)[:input_dim, :output_dim]
        self.kd = np.diag(kd)[:input_dim, :output_dim]

        self._proportional = np.zeros((self.kp.shape[1],1))
        self._integral = np.zeros((self.kp.shape[1],1))
        self._derivative = np.zeros((self.kp.shape[1],1))

    def step(self, setpoint:np.ndarray, current_value:np.ndarray): 
        """
        Args: 
            setpoint (np.ndarray): vector of shape (n,) with setpoint
            current_value (np.ndarray): vector of shape (n,) with current outputs
        """
        error = setpoint - current_value 
        self._integral += error 
        self._derivative = error - self._proportional
        self._proportional = error

        return self.kp @ self._proportional + self.ki @ self._integral + self.kd @ self._derivative
