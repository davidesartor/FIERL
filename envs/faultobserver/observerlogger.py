import numpy as np
import matplotlib.pyplot as plt

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
    
    def log(self, system, observer = None, control_input = None): 

        # log time 
        try: self.time_log.append(self.time_log[-1] + system.dt)
        except: self.time_log.append(0)

        # log system state, fault, output and control input if available
        self.state_log.append(system.state)
        self.fault_log.append(system.fault)
        self.output_log.append(system.output)
        if control_input is not None: self.control_input_log.append(control_input)

        if observer is not None: 
            # log observer state and parameters
            self.mean_state_log.append(observer.state_estimate.mean)
            self.cov_state_log.append(observer.state_estimate.cov)
            self.mean_fault_log.append(observer.fault_estimate.mean)
            self.cov_fault_log.append(observer.fault_estimate.cov)
        
    def plot_estimate_evolution(self, times, mu_log, cov_log, labels = None):
        autocorr = np.array([np.sqrt(np.diag(cov)) for cov in cov_log]).T
        if labels is None:
            labels = [None for _ in autocorr]       
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"] 
        for i,(x,dx,lab,c) in enumerate(zip(np.hstack(mu_log), autocorr, labels, colors)):
            plt.plot(times, x, ":",alpha=0.3, color=c, label=lab)
            plt.fill_between(times, x-dx, x+dx, alpha=0.2, color=c)
        plt.xlim(left=0)
        plt.legend(loc='lower right')
        plt.xlabel('time (s)')

    def plot_true_evolutions(self, times, mu_true, labels=None, step_visual:bool = True):
        x = np.hstack([x.reshape(-1,1) for x in mu_true])
        if labels is None:
            labels = [None for _ in x]
        colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]
        for i,(x,lab,c) in enumerate(zip(x, labels, colors)):
            if step_visual: 
                plt.step(times, x, '-', markersize = 3, linewidth = 1, alpha=1, color=c, label=lab)
            else:
                plt.plot(times, x, 'o-', markersize = 3, linewidth = 2, alpha=0.5, color=c, label=lab)
        plt.legend(loc='lower right')
        plt.xlabel('time (s)')
    
    def plot(self, save:bool = False, save_name:str = None, title_prefix:str = None):
        """
        Plot 4 figures: the state and state estimate evolution, the fault and fault estimate evolution, 
        the output evolution and if available the control input evolution.
      
        Args: 
            save (bool): if True, save the figure
            save_name (str): name of the figure to save
            title_prefix (str, optional): prefix of the figure title
        """

        title_prefix = '' if title_prefix is None else title_prefix

        # plot state and state estimate 
        plt.figure(figsize=(6, 3.25))
        plt.title(title_prefix + 'True State')
        labels = [f'$x^{{true}}_{i}$' for i in range(len(self.state_log[0]))]
        self.plot_true_evolutions(times=self.time_log, mu_true=self.state_log, labels=labels)

        if len(self.mean_state_log) > 0:
            labels = [f'$x^{{est}}_{i}$' for i in range(len(self.mean_state_log[0]))]
            self.plot_estimate_evolution(self.time_log, self.mean_state_log, self.cov_state_log, labels=labels)
        
        if save: 
            save_name_state = 'State_estimate.png' if save_name is None else 'State_estimate_' + save_name
            dpi = 400 if save_name_state.endswith(".png") else None
            plt.savefig(save_name_state, dpi = dpi, bbox_inches='tight')
        
        # plot fault and fault estimate
        plt.figure(figsize=(6, 3.25))
        plt.title(title_prefix + 'True Fault')
        labels = [f'$z^{{true}}_{i}$' for i in range(len(self.fault_log[0]))]
        self.plot_true_evolutions(self.time_log, self.fault_log, labels=labels)

        if len(self.mean_fault_log) > 0:
            plt.title(title_prefix + 'Fault Estimate')
            labels = [f'$z^{{est}}_{i}$' for i in range(len(self.mean_fault_log[0]))]
            self.plot_estimate_evolution(self.time_log, self.mean_fault_log, self.cov_fault_log, labels=labels)

        if save: 
            save_name_fault = 'Fault_estimate.png' if save_name is None else 'Fault_estimate_' + save_name
            dpi = 400 if save_name_fault.endswith(".png") else None
            plt.savefig(save_name_fault, dpi = dpi, bbox_inches='tight')
        
        # plot output
        plt.figure(figsize=(6, 3.25))
        plt.title(title_prefix + 'Output')
        labels = [f'$y_{i}$' for i in range(len(self.output_log[0]))]
        self.plot_true_evolutions(self.time_log, self.output_log, labels=labels, step_visual=True)

        if save:
            save_name_output = 'Output.png' if save_name is None else 'Output_' + save_name
            dpi = 400 if save_name_output.endswith(".png") else None
            plt.savefig(save_name_output, dpi = dpi, bbox_inches='tight')
    
        # if all the input are store
        if len(self.control_input_log) == len(self.time_log[1:]):
            plt.figure(figsize=(6, 3.25))
            plt.title(title_prefix + 'Control Input')
            labels = [f'$a_{i}$' for i in range(len(self.control_input_log[0]))]
            self.plot_true_evolutions(self.time_log[1:], self.control_input_log, labels=labels, step_visual=True)
            
            if save:
                save_name_control = 'Control_input.png' if save_name is None else 'Control_input_' + save_name
                dpi = 400 if save_name_control.endswith(".png") else None
                plt.savefig(save_name_control, dpi = dpi, bbox_inches='tight')
    

def plot_comparison(ObserverLogger1:ObserverLogger, ObserverLogger2:ObserverLogger, data_type:str, labels:list = None, save:bool = False, save_name:str = None) -> None: 
    """
    Plot n plots, where n is the dimension of the data_type. i-th plot displays the i-th component 
    of the data_type of both ObserverLogger1 and ObserverLogger2.

    Args: 
        ObserverLogger1 (ObserverLogger): 
            first observer logger

        ObserverLogger2 (ObserverLogger):
            second observer logger

        data_type (str):
            type of data to plot. Can be 'state', 'fault', 'output' or 'control_input'
        
        labels (list, optional): 
            list of prefix to add to the labels of the plots. Example: ['RL', 'P']. 
            If not given, it will be ['Obs1', 'Obs2']
        
        save (bool):
            if True, save the figure
        
        save_name (str):
            name of the figure to save
    """

    if labels is None or len(labels) != 2: 
        labels = ['Obs1', 'Obs2']
    
    if data_type == 'state': 
        true_data = ObserverLogger1.state_log
        data1 = ObserverLogger1.mean_state_log
        data2 = ObserverLogger2.mean_state_log
        corr1 = ObserverLogger1.cov_state_log
        corr2 = ObserverLogger2.cov_state_log
        title = 'State Estimate Evolution'
    elif data_type == 'fault': 
        true_data = ObserverLogger1.fault_log
        data1 = ObserverLogger1.mean_fault_log
        data2 = ObserverLogger2.mean_fault_log
        corr1 = ObserverLogger1.cov_fault_log
        corr2 = ObserverLogger2.cov_fault_log
        title = 'Fault Estimate Evolution'
    elif data_type == 'output': 
        data1 = ObserverLogger1.output_log
        data2 = ObserverLogger2.output_log
        title = 'System Output Envolution'
    elif data_type == 'control_input': 
        data1 = ObserverLogger1.control_input_log
        data2 = ObserverLogger2.control_input_log
        title = 'Policy Evolution'
    else: 
        raise ValueError(f"Unknown data type {data_type}. Must be 'state', 'fault', 'output' or 'control_input'")
    
    if len(data1) != len(data2): 
        raise ValueError('data1 and data2 must have the same length')

    colors = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7"]

    if data_type == 'state' or data_type == 'fault': 
        true_data = np.hstack([x.reshape(-1,1) for x in true_data])
        autocorr1 = np.array([np.sqrt(np.diag(cov)) for cov in corr1]).T
        autocorr2 = np.array([np.sqrt(np.diag(cov)) for cov in corr2]).T
    
    data1 = np.hstack([x.reshape(-1,1) for x in data1])
    data2 = np.hstack([x.reshape(-1,1) for x in data2])

 
    for i,(data1_c,data2_c) in enumerate(zip(data1,data2)): 
        plt.figure(figsize=(6, 3.25))
        if data_type == 'state' or data_type == 'fault': 
            if data_type == 'state': 
                prefix = 'State_estimate_'
                labs = [f'$x^{{true}}_{i}$', f'$x^{{est}}_{i}$' + labels[0] , f'$x^{{est}}_{i}$' + labels[1]]
            if data_type == 'fault': 
                prefix = 'Fault_estimate_'
                labs = [f'$z^{{true}}_{i}$', f'$z^{{est}}_{i}$' + labels[0] , f'$z^{{est}}_{i}$' + labels[1]]
            plt.plot(ObserverLogger1.time_log, true_data[i,:], '-', linewidth = 1, alpha=0.5, color='k', label=labs[0])
            plt.plot(ObserverLogger1.time_log, data1_c, '-', linewidth = 1.5, alpha=1, color=colors[i], label=labs[1])
            plt.plot(ObserverLogger2.time_log, data2_c, '--', linewidth = 1, alpha=1, color="tab:gray", label=labs[2])
            plt.fill_between(ObserverLogger1.time_log, data1_c-autocorr1[i], data1_c+autocorr1[i], alpha=0.2, color=colors[i])
            plt.fill_between(ObserverLogger2.time_log, data2_c-autocorr2[i], data2_c+autocorr2[i], alpha=0.2, color="tab:gray")
        elif data_type == 'output':
            prefix = 'Output_'
            labs = [f'$y_{i}$'+labels[0], f'$y_{i}$'+labels[1]]
            plt.step(ObserverLogger1.time_log, data1_c, '-', linewidth = 1.5, alpha=1, color=colors[i], label=labs[0])
            plt.step(ObserverLogger2.time_log, data2_c, '-', linewidth = 1, alpha=1, color="tab:gray", label=labs[1])
        else: 
            prefix = 'Control_input_'
            labs = [f'$a_{i}$'+labels[0], f'$a_{i}$'+labels[1]]
            plt.step(ObserverLogger1.time_log[1:], data1_c, '-', linewidth = 1.5, alpha=1, color=colors[i], label=labs[0])
            plt.step(ObserverLogger2.time_log[1:], data2_c, '-', linewidth = 1, alpha=1, color="tab:gray", label=labs[1])
        
        plt.title(title + ' - component' +  f' {i}')
        plt.xlim(left = 0)
        plt.legend(loc='best')

        if save: 
            save_name_ = prefix +  str(i) + save_name if save_name is not None else prefix + str(i) + '.pdf'
            dpi = 400 if save_name_.endswith(".png") else None
            plt.savefig(save_name_, dpi = dpi, bbox_inches='tight')


