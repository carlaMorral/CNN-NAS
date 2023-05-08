import json
from test_evaluator import *
from mobilenetv3_mod import MobileNetV3Space
from nni.retiarii.execution.python import *


path = 'parameters.cfg'
with open(path, 'r') as f:
    parameter_cfg = json.load(f)

parameters = parameter_cfg['parameters']
cls = parameters['class']
init_parameters = parameters['init_parameters']
mutation = parameters['mutation']

evaluator = TestEvaluator()
model = MobileNetV3Space.load_searched_model(mutation)
evaluator.evaluate_model(model)
