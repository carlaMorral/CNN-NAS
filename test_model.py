import json
from nni.nas.execution.pytorch.simplified import *


# Load the parameter.cfg file
path = 'parameter.cfg'
with open(path, 'r') as f:
    parameter_cfg = json.load(f)

# Access the values in the dictionary
cls = parameter_cfg['class']
parameters = parameter_cfg['parameters']
init_parameters = parameters['init_parameters']
mutation = parameters['mutation']
evaluator = Evaluator._load(parameters['evaluator'])
evaluator.num_epochs = 50

graph_data = PythonGraphData.load(cls, init_parameters, mutation, evaluator)

def _model():
    return graph_data.class_(**graph_data.init_parameters)

with ContextStack('fixed', graph_data.mutation):
    graph_data.evaluator._execute(_model)

