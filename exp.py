from sacred import Experiment
import datetime

ex = Experiment('neural_map' + datetime.datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"), save_git_info=False)
