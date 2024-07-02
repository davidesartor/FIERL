import numpy as np
import matplotlib.pyplot as plt
import os

colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

class ObserverLogger: 
    def __init__(self): 
        #time log
        self.time_log = []

        #observer logs
        self.mean_state_log = []
        self.cov_state_log = []
        self.mean_fault_log = []
        self.cov_fault_log = []

        #system logs
        self.state_log = []
        self.fault_log = []
        self.output_log = []

        # control input log 
        self.control_input_log = []
        
        # reference log 
        self.reference_log = []
    
    def log(self, system, observer = None, control_input = None, reference = None): 

        # log time 
        if self.time_log: # if not empty 
            self.time_log.append(self.time_log[-1] + system.dt)
        else: 
            self.time_log.append(0)

        # log system state, fault, output and control input if available
        self.state_log.append(system.state)
        self.fault_log.append(system.fault)
        self.output_log.append(system.output)
        if control_input is not None: self.control_input_log.append(control_input)
        if reference is not None: self.reference_log.append(reference)

        if observer is not None: 
            state_estimate, fault_estimate = observer.split()
            # log observer state and parameters
            self.mean_state_log.append(state_estimate.mean)
            self.cov_state_log.append(state_estimate.cov)
            self.mean_fault_log.append(fault_estimate.mean)
            self.cov_fault_log.append(fault_estimate.cov)
        
    def plot_estimate_evolution(self, times, mu_log, cov_log, labels = None):
        autocorr = np.array([np.sqrt(np.diag(cov)) for cov in cov_log]).T
        if labels is None:
            labels = [None for _ in autocorr]       
        for i,(x,dx,lab,c) in enumerate(zip(np.hstack(mu_log), autocorr, labels, colors)):
            plt.plot(times, x, ":",alpha=0.3, color=c, label=lab)
            plt.fill_between(times, x-dx, x+dx, alpha=0.2, color=c)
        plt.xlim(left=0, right = self.time_log[-1])
        plt.legend(loc='lower right')
        plt.xlabel('time (s)')

    def plot_true_evolutions(self, times, mu_true, labels=None, step_visual:bool = True):
        x = np.hstack([x.reshape(-1,1) for x in mu_true])
        if labels is None:
            labels = [None for _ in x]
        
        for i,(x,lab,c) in enumerate(zip(x, labels, colors)):
            if step_visual: 
                plt.step(times, x, '-', markersize = 3, linewidth = 1, alpha=1, color=c, label=lab) #, where='mid')
            else:
                plt.plot(times, x, 'o-', markersize = 3, linewidth = 2, alpha=0.5, color=c, label=lab)
        plt.legend(loc='lower right')
        plt.xlabel('time (s)')
    
    def reset(self): 
        self.time_log = []
        self.mean_state_log = []
        self.cov_state_log = []
        self.mean_fault_log = []
        self.cov_fault_log = []
        self.state_log = []
        self.fault_log = []
        self.output_log = []
        self.control_input_log = []
        self.reference_log = []
    
    def plot(self, save:bool = False, save_path:str = '', save_name:str = '', title_prefix:str = '', track_threshold:float = 0.0):
        """
        Plot and save four figures. 
            1. True state and estimated state evolutions
            2. True fault and estimated fault evolutions 
            3. Output evolution
            4. Control input evolution (if available)
        
        Args: 
            save (bool): if True save the figure 
            save_name (str): n
        Plot 4 figures: the state and state estimate evolution, the fault and fault estimate evolution, 
        the output evolution and if available the control input evolution.
      
        Args: 
            save (bool): if True, save the figure
            save_path (str, optional): the path where to save the figure
            save_name (str, optional): name of the figure to save
            title_prefix (str, optional): prefix of the figure title
        """

        
        plot_dir = os.path.join(save_path, 'plot')
        if not os.path.exists(plot_dir):
            os.makedirs(plot_dir)
    
        # State and estimated state 
        plt.figure(figsize=(6, 3.25))
        plt.title('State')
        labels = [f'$x^{{true}}_{i}$' for i in range(len(self.state_log[0]))]
        self.plot_true_evolutions(times=self.time_log, mu_true=self.state_log, labels=labels)
        plt.xlim(left=0, right = self.time_log[-1])

        if len(self.mean_state_log) > 0:
            labels = [f'$x^{{est}}_{i}$' for i in range(len(self.mean_state_log[0]))]
            self.plot_estimate_evolution(self.time_log, self.mean_state_log, self.cov_state_log, labels=labels)
            plt.xlim(left=0, right = self.time_log[-1])
            plt.title('State and State Estimate')
        
        if save: 
            save_name_state = save_path + '/plot/State_estimate_' + save_name + '.pdf' 
            plt.savefig(save_name_state, bbox_inches='tight')
        
        
        # Fault and estimated fault
        plt.figure(figsize=(6, 3.25))
        plt.title(title_prefix + 'Fault')
        labels = [f'$z^{{true}}_{i}$' for i in range(len(self.fault_log[0]))]
        self.plot_true_evolutions(self.time_log, self.fault_log, labels=labels)
        plt.xlim(left=0, right = self.time_log[-1])

        if len(self.mean_fault_log) > 0:
            plt.title(title_prefix + 'Fault Estimate')
            labels = [f'$z^{{est}}_{i}$' for i in range(len(self.mean_fault_log[0]))]
            self.plot_estimate_evolution(self.time_log, self.mean_fault_log, self.cov_fault_log, labels=labels)
            plt.xlim(left=0, right = self.time_log[-1])
            plt.title('Fault and Fault Estimate')

        if save: 
            save_name_fault = save_path + '/plot/Fault_estimate_' + save_name + '.pdf'
            plt.savefig(save_name_fault, bbox_inches='tight')
        

        # plot output
        decomposed_ref = list(zip(*self.reference_log)) if len(self.reference_log) > 0 else [None, None]
   
        plt.figure(figsize=(6, 3.25))
        plt.title(title_prefix + 'Output')
        self.plot_true_evolutions(self.time_log, self.output_log, labels=[f'$y_{i}$' for i in range(len(self.output_log[0]))], step_visual=True)
            

        for i in range(len(decomposed_ref)):
            ref_top = [a - track_threshold for a in decomposed_ref[i]]
            ref_down = [a + track_threshold for a in decomposed_ref[i]]
            plt.plot(self.time_log, decomposed_ref[i], '--', color = colors[i])

            for j, s in enumerate(zip(ref_top, ref_down)):
                t, d = s
                try:
                    time = [self.time_log[j], self.time_log[j+1]]
                except: 
                    time = self.time_log[j]
                plt.fill_between(time, t, d, color = colors[i], linewidth=0.0, alpha=0.1, label='$y_{%d}^{ref} \pm \Delta y^{max}_{i}$' % i, step='mid')
                          
        plt.xlim(left=0, right = self.time_log[-1])
        plt.ylim(bottom = 0.0)
        plt.plot()

        if save:
            save_name_output = save_path + '/plot/Output_' + save_name + '.pdf'
            plt.savefig(save_name_output, bbox_inches='tight')
    
        # if all the input are store
        if len(self.control_input_log) == len(self.time_log[1:]):
            plt.figure(figsize=(6, 3.25))
            plt.title(title_prefix + 'Control Input')
            labels = [f'$a_{i}$' for i in range(len(self.control_input_log[0]))]
            self.plot_true_evolutions(self.time_log[1:], self.control_input_log, labels=labels, step_visual=True)
            plt.plot(self.time_log, [0]*len(self.time_log), '-', linewidth = 1., alpha=0.8, color="tab:gray")
            plt.xlim(left=0, right = self.time_log[-1])
            plt.ylabel('Cntrol input')
            
            if save:
                save_name_control = save_path + '/plot/Control_input_' + save_name + '.pdf'
                plt.savefig(save_name_control, bbox_inches='tight')