import json
import base64
import pickle
from nni.retiarii.execution.python import *


# Load the parameter.cfg file
path = 'parameters.cfg'
with open(path, 'r') as f:
    parameter_cfg = json.load(f)

# Access the values in the dictionary
parameters = parameter_cfg['parameters']
cls = parameters['class']
init_parameters = parameters['init_parameters']
mutation = parameters['mutation']

# Decode the function bytes and load the pickled function
func_bytes = base64.b64decode(parameters['evaluator']['function']['__nni_type__'][6:])
func = pickle.loads(func_bytes)

# Instantiate the FunctionalEvaluator class with the loaded function
class_path = parameters['evaluator']['type']['__nni_type__'][5:]
module_path, class_name = class_path.rsplit('.', 1)
module = __import__(module_path, fromlist=[class_name])
EvaluatorClass = getattr(module, class_name)
evaluator = EvaluatorClass(func)

graph_data = PythonGraphData(cls, init_parameters, mutation, evaluator)
evaluator.num_epochs = 50

def _model():
    return graph_data.class_(**graph_data.init_parameters)

with ContextStack('fixed', graph_data.mutation):
    graph_data.evaluator._execute(_model)

