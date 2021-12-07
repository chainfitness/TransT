from got10k_toolkit.toolkit.experiments import ExperimentGOT10k
from got10k_toolkit.toolkit.trackers.identity_tracker import IdentityTracker
from got10k_toolkit.toolkit.trackers.net_wrappers import NetWithBackbone

#Specify the path
net_path = "/home/rzh/win-pycharm/Trans-T/pytracking/networks/transt_N2.pth" #Absolute path of the model
dataset_root= "/home/rzh/win-pycharm/Trans-T/datasets/got10k" #Absolute path of the datasets

#TransT
net = NetWithBackbone(net_path=net_path, use_gpu=True)
tracker = IdentityTracker(name='transt', net=net, window_penalty=0.49, exemplar_size=128, instance_size=256)

#Test
experiment = ExperimentGOT10k(
    root_dir=dataset_root,  # GOT-10k's root directory
    subset='test',  # 'train' | 'val' | 'test'
    result_dir='results',  # where to store tracking results
    report_dir='reports'  # where to store evaluation reports
)
experiment.run(tracker, visualize=False)
