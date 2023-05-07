from nni.retiarii.evaluator import FunctionalEvaluator
from nni.experiment import RemoteMachineConfig
from nni.retiarii.experiment.pytorch import RetiariiExperiment, RetiariiExeConfig


class Experiment:

    def __init__(self, model_space, evaluator, search_strategy):
        self.evaluator = FunctionalEvaluator(evaluator.evaluate_model)
        self.exp = RetiariiExperiment(model_space, self.evaluator, [], search_strategy)
        self._config()

    def _config(self):
        self.exp_config = RetiariiExeConfig('remote')
        self.exp_config.experiment_name = 'cifar10_search'
        self.exp_config.max_trial_number = 96
        self.exp_config.trial_concurrency = 4
        self.exp_config.trial_gpu_number = 4
        self.exp_config.nni_manager_ip = '34.160.111.145'

        rm_conf = RemoteMachineConfig()
        rm_conf.host = '35.221.22.168'
        rm_conf.user = 'jan'
        #rm_conf.password = '1'
        rm_conf.ssh_key_file = '~/.ssh/google_compute_engine'
        rm_conf.ssh_passphrase = ''
        rm_conf.port = 22
        rm_conf.python_path = '/opt/conda/bin/python3.7'
        rm_conf.gpu_indices = [0, 1, 2, 3]
        rm_conf.use_active_gpu = True
        rm_conf.max_trial_number_per_gpu = 1

        self.exp_config.training_service.machine_list = [rm_conf]

    def run(self):
        self.exp.run(self.exp_config, 8083)

    def export_top(self):
        for model_dict in self.exp.export_top_models(formatter='dict'):
            print(model_dict)
