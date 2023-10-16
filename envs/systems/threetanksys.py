import numpy as np
from envs.systems.systems import *

class ThreeTankSystem(FaultyActuatorNoisySystem):
    def __init__(self, 
        sample_time = 0.1,
        h_target=(0.489, 0.2332, 0.3611),
        outflow_coeff=(0.46, 0.48, 0.58),
        tank_cross_section=1.54e-2,
        connector_cross_section=5e-5,
        gravitational_constant=9.81, 
        state_noise_std=1e-3,
        output_noise_std=1e-3,
        min_input = -0.002, 
        max_input = 0.02
        ):

        g = gravitational_constant
        sn,sa = connector_cross_section, tank_cross_section
        az1,az2,az3 = outflow_coeff
        hstar1,hstar2,hstar3 = h_target

        k = g*sn/(sa*np.sqrt(2*g))
        a1 = az1*k/np.sqrt(hstar1-hstar3)
        a2 = az3*k/np.sqrt(hstar3-hstar2)
        a3 = az2*k/np.sqrt(hstar2)
        
        A = np.array([
            [-a1, 0., a1],
            [0., -a2-a3, a2], 
            [a1, a2, -a1-a2]
        ])

        B = np.array([
            [1., 0.,],
            [0., 1.,], 
            [0., 0.,]
        ])/sa

        C = np.array([
            [1., 0., 0.],
            [0., 1., 0.], 
        ])

        D = np.array([
            [0., 0.,],
            [0., 0.,], 
        ])

        super().__init__(A, B, C, D, sample_time, 'continuous', min_input = min_input, max_input = max_input, state_noise_std=state_noise_std, output_noise_std=output_noise_std)