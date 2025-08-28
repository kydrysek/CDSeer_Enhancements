from river import metrics
# from capymoa.evaluation import ClassificationWindowedEvaluator, ClassificationEvaluator
from collections import deque
import pandas as pd
# from capymoa.instance import LabeledInstance

from copy import deepcopy

class Telemetry():
    def __init__(self):
        
        self.outcomes = deque()   

        self.inspector_log = dict()    

        self.metrics_capymoa = {
            # "ClassificationWindowedEvaluator": ClassificationWindowedEvaluator(schema=stream_schema, window_size=interval_size),
            # "ClassificationEvaluator": ClassificationEvaluator(schema=stream_schema, window_size=interval_size)
            }
        
        self.metrics_river = {
            "Accuracy":metrics.Accuracy(),
            # "MicroPrecision":metrics.MicroPrecision(),
            # "MicroRecall":metrics.MicroRecall(),
            # "MicroF1": metrics.MicroF1(),   
            "BalancedAccuracy":metrics.BalancedAccuracy(),
            "MacroF1":metrics.MacroF1(),
            "MacroPrecision":metrics.MacroPrecision(),
            "MacroRecall":metrics.MacroRecall(),
            "Precision":metrics.Precision(),
            "Recall":metrics.Recall(),
            "F1": metrics.F1()
        }
        
    def update_metrics(self,ground_truth,pred_model):
        for metric in self.metrics_river.values():
            metric.update(ground_truth,pred_model)
        for metric in self.metrics_capymoa.values():
            metric.update(ground_truth,pred_model)
   
    def get_metrics(self):
        return self.metrics_river | self.metrics_capymoa
    
    def get_metrics_capymoa(self):
        print(type(self.metrics_capymoa))
        print(self.metrics_capymoa)
        return self.metrics_capymoa
    
    def get_metrics_river(self):
        return self.metrics_river
    
    
    def log_debug(self, step_number, ground_truth, pred_model, pred_inspector, drift_detected, data_as_numpy):
        # # self.outcomes.append((step_number, ground_truth, pred_model,instance.y_index,instance.y_label,data[step_count,-1],pred_inspector,drift_detected))
        # # print("Piggedy pog")
        # # print(stream_instance)
        # print(type(data_as_numpy))
        # print(data_as_numpy.shape)
        log_entry = (step_number, ground_truth, pred_model,pred_inspector, drift_detected, data_as_numpy[step_number,-1])
        # # print("Biggety bog")
        # # print()
        self.outcomes.append(log_entry)

    def log_inspector(self,step,inspector):
        # self.inspector_log[step]=deepcopy(inspector)
        pass
    
    def get_debug_as_df(self):
            outcomes_columns=["step","GT","pred","inspector","drift","inst_label_from_array"]
            outcome_df = pd.DataFrame(self.outcomes,columns=outcomes_columns)
            return outcome_df
    
    def get_drift_points(self):
        outcome_df = self.get_debug_as_df()
        if len(outcome_df):
            drift_series = outcome_df[outcome_df["drift"]]["step"]
        else:
            drift_series = []
        return list(drift_series)
    
    def get_inspector_log(self):
        return self.inspector_log

    
        