from Oracle import Oracle
# from Inspector import Inspector
from Inspector import Inspector
from Telemetry import Telemetry

from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline

# from capymoa.stream import NumpyStream
# from capymoa.instance import LabeledInstance

import river
from river.drift.binary import DDM
from river.drift import ADWIN
from river import stream
from statistics import mean

import pandas as pd
import numpy as np
import time
import pickle
import random
import shap
import yaml
import json
import os

from tqdm import tqdm
from typing import Optional
from itertools import product

LIMIT_DATASET_SIZE = 100000
DATE_TIME_FMT = '%d-%m-%Y %H:%M:%S'

DRIFT_GOLDEN_SOURCE = {
    'NOAA.csv':[1299, 8973, 17045],
    'Jiaolong_DSMS_V2.csv':[2899, 8359, 10618, 15524, 19023, 29770], 
    'elec2.csv':[1345, 3019, 5887, 7723, 10527, 11738, 14010, 16422, 18940, 20249, 22541, 24163, 25724, 27394, 28894, 31376, 33059, 34695, 36994, 38924, 43338],
    'harv_mixed_0101_gradual.csv':[9500,20000,30500],
    'harv_mixed_1010_gradual.csv':[9500,20000,30500],
    'harv_sea_0123_gradual_noise_0.2.csv':[9500,20000,30500],
    'harv_sea_3210_gradual_noise_0.2.csv':[9500,20000,30500],
    'harv_sine_0123_gradual.csv':[9500,20000,30500],
    'own_sine_paper_balanced_noise_230.csv':[3000,10000],
    'own_sine_paper_balanced_nonoise_230.csv':[3000,10000],
    'own_sine_paper_imbalanced_noise_230.csv':[3000,10000],
    'own_sine_paper_imbalanced_nonoise_230.csv':[3000,10000],

    'own_sine_paper_balanced_noise_230_stretched.csv':[3000,10000],
    'own_sine_paper_balanced_nonoise_230_stretched.csv':[3000,10000],
    'own_sine_paper_imbalanced_noise_230_stretched.csv':[3000,10000],
    'own_sine_paper_imbalanced_nonoise_230_stretched.csv':[3000,10000],

    'own_sine_paper_balanced_noise_230_stretched_x.csv':[3000,10000],
    'own_sine_paper_balanced_nonoise_230_stretched_x.csv':[3000,10000],
    'own_sine_paper_imbalanced_noise_230_stretched_x.csv':[3000,10000],
    'own_sine_paper_imbalanced_nonoise_230_stretched_x.csv':[3000,10000],

    'nonlinear_abrupt_chocolaterotation_binary.csv':[5000,10000,15000,20000],
    'nonlinear_abrupt_chocolaterotation_multi.csv':[5000,10000,15000,20000],
    'nonlinear_abrupt_chocolaterotation_noise_and_redunce_binary.csv':[4800,9600,14400,19200],
    'nonlinear_abrupt_chocolaterotation_noise_and_redunce_multi.csv':[4800,9600,14400,19200],
    'nonlinear_abrupt_chocolaterotation_noise_binary.csv':[4800,9600,14400,19200],
    'nonlinear_abrupt_chocolaterotation_noise_multi.csv':[4800,9600,14400,19200],
    'nonlinear_abrupt_chocolaterotation_redunce_binary.csv':[4800,9600,14400,19200],
    'nonlinear_abrupt_chocolaterotation_redunce_multi.csv':[4800,9600,14400,19200],
    'nonlinear_sudden_chocolaterotation_binary.csv':[5000,10000,15000,20000],
    'nonlinear_sudden_chocolaterotation_multi.csv':[5000,10000,15000,20000],
    'nonlinear_sudden_chocolaterotation_noise_and_redunce_binary.csv':[4800,9600,14400,19200],
    'nonlinear_sudden_chocolaterotation_noise_and_redunce_multi.csv':[4800,9600,14400,19200],
    'nonlinear_sudden_chocolaterotation_noise_binary.csv':[4800,9600,14400,19200],
    'nonlinear_sudden_chocolaterotation_noise_multi.csv':[4800,9600,14400,19200],
    'nonlinear_sudden_chocolaterotation_redunce_binary.csv':[4800,9600,14400,19200],
    'nonlinear_sudden_chocolaterotation_redunce_multi.csv':[4800,9600,14400,19200],

    'own_sea_extra_cols.csv':[3000,10000],
    'own_sea_no_extra_cols.csv':[3000,10000],
    'own_sea_noise_extra_cols_012_2percent.csv':[3000,10000],
    'own_sea_noise_extra_cols_0123_2percent.csv':[3000,10000,16000],
    'own_sea_noise_no_extra_cols_012_2percent.csv':[3000,10000],
    'own_sea_noise_no_extra_cols_0123_2percent.csv':[3000,10000,16000],

    'own_sea_extra_cols_stretched.csv':[3000,10000],
    'own_sea_no_extra_cols_stretched.csv':[3000,10000],
    'own_sea_noise_extra_cols_012_2percent_stretched.csv':[3000,10000],
    'own_sea_noise_extra_cols_0123_2percent_stretched.csv':[3000,10000,16000],
    'own_sea_noise_no_extra_cols_012_2percent_stretched.csv':[3000,10000],
    'own_sea_noise_no_extra_cols_0123_2percent_stretched.csv':[3000,10000,16000],

    'own_sea_extra_cols_stretched_y.csv':[3000,10000],
    'own_sea_no_extra_cols_stretched_y.csv':[3000,10000],
    'own_sea_noise_extra_cols_012_2percent_stretched_y.csv':[3000,10000],
    'own_sea_noise_extra_cols_0123_2percent_stretched_y.csv':[3000,10000,16000],
    'own_sea_noise_no_extra_cols_012_2percent_stretched_y.csv':[3000,10000],
    'own_sea_noise_no_extra_cols_0123_2percent_stretched_y.csv':[3000,10000,16000]
}


def retrain_model(model, oracle:Oracle, training_size:int):
    """ Retrains the reference model on the latest training_size points with ground truth labels from the oracle. """
    # 1. Get the latest training_size points and their labels from the oracle
    xs_train, ys_train = oracle.latest_points_labels(training_size,record_it=False)
    
    # 2. Retrain the model
    model.fit(np.array(xs_train), np.array(ys_train))

    return model, xs_train



def initialise_model_and_inspector(model_class:object,oracle:Oracle, inspector_curiosity_factor:float, training_size:int, memory_size:int,
                                   insp_window_size:int, cluster_method:Optional[str],random_state:int, inspector_model,
                                   clustering_params,log_me=False, keep_known_labels=True,initialise_inspector=True,model_object:object=None):    

    # 1. Train reference model
    # main_model = Pipeline([("scaler",MinMaxScaler()),("model",model_class(random_state=RANDOM_SEED,n_jobs=N_JOBS))])
    # main_model = Pipeline([("scaler",MinMaxScaler()),("model",model_class(random_state=random_state))])
    if model_object is None:
        main_model = Pipeline([("model",model_class(random_state=42))])
    else:
        main_model = model_object
    main_model, model_train_data = retrain_model(main_model, oracle, training_size)
    

    # # 2. Create a suprvised drift detector object
    # # drift_detector = capymoa.drift.detectors.ADWIN(delta=.02)  # Initialize the drift detector
    # drift_detector = capymoa.drift.detectors.DDM(min_n_instances=100,out_control_level=3)  # Initialize the drift detector

    # 3. Create an Inspector object - inspector also needs  epsilon for DBSCAN clustering, so calculate this     
    config={"random_seed":random_state, "log_extras":log_me, "cluster_method":cluster_method, "clustering_params":clustering_params}
        #     "perform_clustering":perform_clustering,"scale_for_clustering":scale_for_clustering,"use_model_pred":use_model_pred}

        
        # self.clustering_params = config.get("clustering_params",default_clustering_params)
    # FIXME
    if initialise_inspector:
        inspector = Inspector(oracle,inspector_curiosity_factor, memory_size, insp_window_size, inspector_model=inspector_model,ref_model=main_model, 
                                        ref_model_train_data=model_train_data, keep_known_labels=keep_known_labels,config=config)
    else:
        inspector = None


    return main_model, inspector 


def initialise_data_stream(file_path:str, max_size:int,skip_initial:int):
    data = pd.read_csv(file_path)
    data = data.iloc[skip_initial:max_size]
    data_stream = stream.iter_pandas(X=data.iloc[:,:-1],y=data.iloc[:,-1])

    feature_count = data.shape[1]-1
        
    return data_stream, np.array(data), feature_count # data array is returned only for logging 


def get_experiment_results(telemetry:Telemetry)->dict:    
    results = {
        "drift_record":telemetry.get_drift_points(),
        "metrics_river":telemetry.get_metrics_river(),        
        # "windowed_evaluator":telemetry.get_metrics_capymoa()["ClassificationWindowedEvaluator"],
        # "classic_evaluator":telemetry.get_metrics_capymoa()["ClassificationEvaluator"],
        "outcome_df":telemetry.get_debug_as_df()
    }
    return results



def serialisedfilename_suffix_from_repr(repr:dict)->str:
    config=repr["run_config"]    
    time_end=time.mktime(time.strptime(repr["ended_at"],DATE_TIME_FMT))
    file=config["file_path"].split(".")[0].split("/")[-1]
    repr_str = f"{file}__MS_{config['memory_size']}__IWS_{config['insp_window_size']}__CM_{config['cluster_method']}"    
    repr_str += f"__DD_{str(config['drift_detector'])}__ICF_{config['inspector_curiosity_factor']}"
    repr_str += f"__ORC_{config['oracle_compliance']}__END_{int(time_end)}"
    return repr_str

def serialise_run_outcomes(run_config:dict,oracle:Oracle,telemetry:Telemetry,time_start:float,time_end:float,prefix:str)->str:
    time_end_str = time.strftime(DATE_TIME_FMT,time.strptime(time.ctime(time_end)))
    repr = {"run_config":run_config, "oracle":oracle,"telemetry":telemetry,"time_taken":time_end-time_start,"ended_at":time_end_str}    


    save_file=f"run_outcomes/{prefix}__{serialisedfilename_suffix_from_repr(repr)}.pickle"

    with open(save_file,"wb",buffering=0) as fb:
        pickle.dump(repr,fb)
    return save_file


def serialise_aggregate_results(run_config:dict,aggregate_result:dict,prefix:str)->str:
    current_time = time.ctime()
    time_str = time.strftime(DATE_TIME_FMT,time.strptime(current_time))
    repr = {"run_config":run_config, "aggregate_results":aggregate_result,"ended_at":time_str}    

    save_file=f"run_outcomes_aggregate/{prefix}__{serialisedfilename_suffix_from_repr(repr)}.pickle"

    with open(save_file,"wb",buffering=0) as fb:        
        pickle.dump(repr,fb)
    return save_file


def reconstruct_run_outcomes(save_file:str)->dict:
    with open(save_file,"rb",buffering=0) as fb:
        repr_reconstructed = pickle.load(fb)
    return repr_reconstructed


# def single_run(file_path:str,run_config:dict,max_size, skip_initial):
def single_run(run_config:dict,max_size, skip_initial,prefix:str, random_state,save):
    print("Starting on: ",run_config["file_path"], " seed = ",random_state)
    time_start = time.time()
    oracle, telemetry = run_experiment(max_size=max_size,skip_initial=skip_initial,random_state=random_state,**run_config)
    time_end = time.time()
    saved_file = serialise_run_outcomes(run_config,oracle,telemetry,time_start,time_end,prefix) if save else "Results not serialised"

    # print("Config:",run_config)
    print("Run results saved as ",saved_file)
    print(f"Drifts detected: {telemetry.get_drift_points()}")
    print(f"Oracle provided {len(oracle.labels_given)} labels, approximately {oracle.percentage_of_labels_provided():.8%} - training points included in denominator")
    metrics:dict = telemetry.get_metrics_river()
    acc,f1,prec,rec = metrics["Accuracy"].get(),metrics["F1"].get(),metrics["Precision"].get(),metrics["Recall"].get()
    print(f"Metrics: accuracy={acc} f1={f1} recall={rec} precision={prec}")
    print()
    
    return saved_file, oracle, telemetry

def combined_run(run_config:dict,max_size, skip_initial,prefix:str,repetitions, random_states,save=True):
    outcome = [single_run(run_config=run_config,max_size=max_size,prefix=f"run{i}_{prefix}",skip_initial=skip_initial,save=save,random_state=random_states[i]) for i in range(repetitions)]
    return outcome


def aggregate_results_to_df_JUSTMACRO_NORESCORE(aggregate_files, save_as_name=None):
    
    agg_res = []
    for file in aggregate_files:
        # try:
        if True:
            res = reconstruct_run_outcomes(file)
                     
            vals = res["aggregate_results"]            
            agg_res += [(file,vals)]
        # except Exception as e:
        #     print(f"An error occurred: {e}")                                                               

    out = []
    for file, vals in agg_res:
        vals["aggregate_file"]=file        
        out += [vals]
        print(vals["label_count"])

    df = pd.DataFrame(out).drop(columns=['label_percentage_med','label_percentage_low'])
    df = df[list(df.columns[:-3])+[df.columns[-1]]+list(df.columns[-3:-1])]
    cols_order = ["aggregate_file","MacroPrecision",'MacroRecall','MacroF1',"BalancedAccuracy"]
    df = df[cols_order]
    if save_as_name is not None:
        df.to_csv(save_as_name,index=False)
    return df




def aggregate_results_to_df(aggregate_files, save_as_name=None,rescore_drift=False):

    agg_res = []
    for file in aggregate_files:
        try:
            res = reconstruct_run_outcomes(file)
            constituent_files = res['aggregate_results']["files"]
            interim = [(f,reconstruct_run_outcomes(f)) for f in constituent_files]
            outcomes = [(f,outcome["oracle"],outcome["telemetry"]) for f, outcome in interim]
            run_config = interim[0][1]["run_config"]
            base_file = os.path.basename(run_config["file_path"])               

            if not rescore_drift:        
                vals = res["aggregate_results"]
            else:            
                golden_drift = DRIFT_GOLDEN_SOURCE.get(base_file)                
                vals = score_outcomes(outcomes,golden_drift,skip_initial=0,train_forward=run_config["train_forward"],train_size=run_config["training_size"],mem_size=run_config["memory_size"])
                vals["files"] = constituent_files
            agg_res += [(file,vals,base_file)]
        except Exception as e:
            print(f"An error occurred: {e}")                                                               

    out = []
    for file, vals, base_file in agg_res:
        vals["aggregate_file"]=file
        vals["base_file"]=base_file
        out += [vals]
        print(vals["label_count"])

    df = pd.DataFrame(out).drop(columns=['label_percentage_med','label_percentage_low'])
    df = df[list(df.columns[:-3])+[df.columns[-1]]+list(df.columns[-3:-1])]
    cols_order = ["label_percentage_high","label_count","drift_count","drift_precision_1","drift_recall_1","tp_1","fp_1","fn_1","drift_precision_2",
              "drift_recall_2","tp_2","fp_2","fn_2","drift_precision_3","drift_recall_3","tp_3","fp_3","fn_3","drift_precision_4","drift_recall_4",
              "tp_4","fp_4","fn_4","drift_precision_1_2sided","drift_recall_1_2sided","tp_1_2sided","fp_1_2sided","fn_1_2sided","drift_precision_2_2sided",
              "drift_recall_2_2sided","tp_2_2sided","fp_2_2sided","fn_2_2sided","drift_precision_3_2sided","drift_recall_3_2sided","tp_3_2sided",
              "fp_3_2sided","fn_3_2sided","Accuracy","Precision","Recall","F1","aggregate_file","drift_point_list","files",
              "MacroPrecision",'MacroRecall','MacroF1',"BalancedAccuracy","base_file"]
    df = df[cols_order]
    if save_as_name is not None:
        df.to_csv(save_as_name,index=False)
    return df


def score_outcomes(outcomes:list[tuple],drift_golden_source,skip_initial,train_forward,train_size,mem_size):

    scores = []
    drifts = []
    for saved_file, oracle, telemetry in outcomes:
        scores.append(score_outcome(oracle, telemetry,drift_golden_source,skip_initial,train_forward,train_size,mem_size))        
        drifts.append(telemetry.get_drift_points())

    # # res = dict()     
    # for key in scores[0]:
    #     print(f"{key}={[pp[key] for pp in scores]}")
    # print(scores[0])        
    # print(scores)        
    res = {key: mean([pp[key] for pp in scores]) for key in scores[0]}
    res["drift_point_list"] = drifts

    return res
    



def score_outcome(oracle:Oracle, telemetry:Telemetry, drift_golden_source,skip_initial,train_forward,train_size,mem_size):
    metrics = telemetry.get_metrics_river()
    drift_points = telemetry.get_drift_points()
    # print(f"skip_initial={skip_initial} train_size={train_size} mem_size={mem_size}")
    points_to_ignore = skip_initial+(1 if not train_forward else 1+ len(drift_points))*(train_size-mem_size)
    # print(f"skip_initial={skip_initial} train_size={train_size} mem_size={mem_size} points_to_ignore={points_to_ignore}")
    label_percentage_high = oracle.percentage_of_labels_provided(points_to_ignore)
    label_percentage_med = oracle.percentage_of_labels_provided(985)
    label_percentage_low = oracle.percentage_of_labels_provided(0)
    label_count = len(oracle.labels_given)
    
    score = {"label_percentage_high":label_percentage_high,"label_percentage_med":label_percentage_med,
             "label_percentage_low":label_percentage_low, "label_count":label_count, "drift_count":len(drift_points)}
    score.update({key:metrics[key].get() for key in metrics})
    score = add_drift_res_to_score(drift_points,drift_golden_source,score)
    # print(score)
    return score

def add_drift_res_to_score(drift_points,drift_golden_source,score):
    drift_precision_1, drift_recall_1,tp_1,fp_1,fn_1 = score_drift(drift_points,drift_golden_source,tolerance=500,twosided=False)
    drift_precision_2, drift_recall_2,tp_2,fp_2,fn_2 = score_drift(drift_points,drift_golden_source,tolerance=1000,twosided=False)
    drift_precision_3, drift_recall_3,tp_3,fp_3,fn_3 = score_drift(drift_points,drift_golden_source,tolerance=2000,twosided=False)
    drift_precision_4, drift_recall_4,tp_4,fp_4,fn_4 = score_drift(drift_points,drift_golden_source,tolerance=3000,twosided=False)
    drift_precision_1_2sided, drift_recall_1_2sided,tp_1_2sided,fp_1_2sided,fn_1_2sided = score_drift(drift_points,drift_golden_source,tolerance=500,twosided=True)
    drift_precision_2_2sided, drift_recall_2_2sided,tp_2_2sided,fp_2_2sided,fn_2_2sided = score_drift(drift_points,drift_golden_source,tolerance=1000,twosided=True)
    drift_precision_3_2sided, drift_recall_3_2sided,tp_3_2sided,fp_3_2sided,fn_3_2sided = score_drift(drift_points,drift_golden_source,tolerance=2000,twosided=True)
    score.update({"drift_count":len(drift_points),
             "drift_precision_1":drift_precision_1, "drift_recall_1":drift_recall_1, "tp_1":tp_1, "fp_1":fp_1, "fn_1":fn_1,
             "drift_precision_2":drift_precision_2, "drift_recall_2":drift_recall_2, "tp_2":tp_2, "fp_2":fp_2, "fn_2":fn_2,
             "drift_precision_3":drift_precision_3, "drift_recall_3":drift_recall_3, "tp_3":tp_3, "fp_3":fp_3, "fn_3":fn_3,
             "drift_precision_4":drift_precision_4, "drift_recall_4":drift_recall_4, "tp_4":tp_4, "fp_4":fp_4, "fn_4":fn_4,
             "drift_precision_1_2sided":drift_precision_1_2sided, "drift_recall_1_2sided":drift_recall_1_2sided, "tp_1_2sided":tp_1_2sided, "fp_1_2sided":fp_1_2sided, "fn_1_2sided":fn_1_2sided,
             "drift_precision_2_2sided":drift_precision_2_2sided, "drift_recall_2_2sided":drift_recall_2_2sided, "tp_2_2sided":tp_2_2sided, "fp_2_2sided":fp_2_2sided, "fn_2_2sided":fn_2_2sided,
             "drift_precision_3_2sided":drift_precision_3_2sided, "drift_recall_3_2sided":drift_recall_3_2sided, "tp_3_2sided":tp_3_2sided, "fp_3_2sided":fp_3_2sided, "fn_3_2sided":fn_3_2sided
             })
    return score
    


def divide_helper(n,d):
    return n/d if d else 0

def score_drift(drift_points, drift_golden_source, tolerance, twosided):
    tp = {k for k in drift_points for p in drift_golden_source if p <= k and k <= p + tolerance}
    if twosided:
        tp.update({k for k in drift_points for p in drift_golden_source if p >= k and k >= p - tolerance})

    fp = {k for k in drift_points if k not in tp}

    identified_golden_source = {p for k in drift_points for p in drift_golden_source if p <= k and k <= p + tolerance}
    if twosided:
        identified_golden_source.update({p for k in drift_points for p in drift_golden_source if p >= k and k >= p - tolerance})
    fn = {k for k in drift_golden_source if k not in identified_golden_source}
    
    precision = divide_helper(len(tp),(len(tp)+len(fp)))
    recall=divide_helper(len(tp),(len(tp)+len(fn)))
    return precision, recall, len(tp), len(fp), len(fn)

_, _, tp, fp, tn = score_drift([11500,21000,24000],[2000,12000,22000],500,twosided=True)
assert (tp, fp, tn) == (1, 2, 2)
_, _, tp, fp, tn = score_drift([11500,21000,24000],[2000,12000,22000],1000,twosided=True)
assert (tp, fp, tn) == (2, 1, 1)
_, _, tp, fp, tn = score_drift([11500,21000,24000],[2000,12000,22000],2000,twosided=True)
assert (tp, fp, tn) == (3, 0, 1)
_, _, tp, fp, tn = score_drift([11500,21000,24000],[2000,12000,22000],500,twosided=False)
assert (tp, fp, tn) == (0, 3, 3)
_, _, tp, fp, tn = score_drift([11500,21000,24000],[2000,12000,22000],1000,twosided=False)
assert (tp, fp, tn) == (0, 3, 3)
_, _, tp, fp, tn = score_drift([11500,21000,24000],[2000,12000,22000],2000,twosided=False)
assert (tp, fp, tn) == (1, 2, 2)


def step_count(instance_count,initial_skipped):
    return instance_count+initial_skipped

def run_experiment(file_path, max_size,inspector_curiosity_factor,oracle_compliance:float,drift_detector,
                   training_size, memory_size,insp_window_size,keep_known_labels, inspector_model, clustering_params_list,
                   train_forward=True,cluster_method=None,skip_initial=0,random_state=42,run_quiet=False, run_inspector=True, 
                   dont_propagate=False,model_class:object=None,model_object:object=None):
    """ Runs the experiment on the given file path with the given model class and parameters. """
    
    # 1. Cap any size requested by the global maximum
    max_size = min(max_size,LIMIT_DATASET_SIZE)

    # 2. Initiate the input data stream. Return also as whole array for logging purposes - to be removed later
    data_stream, data, feature_count = initialise_data_stream(file_path,max_size,skip_initial)
    max_size = min(max_size,len(data))

    # 3. Label oracle is part of the experiment, simulating a human expert who can label (some) unlabeled samples.
    #   It will witness the points with labels, and provide to the inspector and to model (during retrainining) only the labels
    #   they are entitled to have  
    oracle_memory_size = max(insp_window_size,training_size,memory_size)
    oracle = Oracle(compliance_factor=oracle_compliance,points_to_hold=oracle_memory_size,feature_count = feature_count,random_state=random_state) 

    inspector = None 

    # if drift_detector is None: drift_detector = capymoa.drift.detectors.ADWIN(delta=.02)  # Initialize the drift detector
    if drift_detector is None: drift_detector = DDM(warm_start=30)  # Initialize the drift detector
    
    
    # 4. Initialise object for capturing of metrics and debug
    telemetry = Telemetry()    


    # 4. Process stream element by element. 
    #    - the first phase is about accumulating points that the reference model can be trained on, to be able to execute the experiment
    #    - once the reference model is actually available, the main part of experiment - focused on CDSeer - begins
    
    step_number = skip_initial -1 # instance/step count is useful for debug info, and for initial training
    model = None # or could use the flag that it's offline
    
    for instance in tqdm(data_stream,total=max_size-skip_initial): 
        
        # A. instance comes with point info (x) and label (y). Label will only be visible to oracle and to metrics/debug functionality  
        # x,y = instance.x, instance.y_index
        x,y=instance
        x=np.array(list(x.values()))

        # print("Front")
        # print(instance)
        # print("Back")
        # print()
        
        # inst_count += 1
        # step_number = step_count(inst_count,skip_initial)
        step_number += 1        
        
                    
        # B. The oracle to know the ground truth labels, so that it can serve them up to the inspector (with some probability), and to the main model
        # once the drift is detected and there is a need to retrain the model. Note - main model may not be an online learning one!
        oracle.add_point_and_label(x, y, step_number)

        # C. Accumulate initial data for reference model/oracle. The control will skip to next loop
        #    until enough data is accumulated and the reference model is trained.        
        if model is None:
            # If there is enough data to (re)train the model, do so and in next step the assessment will start
            if oracle.number_of_available_points() >= training_size:
                initialise_inspector = (inspector is None) and run_inspector                
                model, inspector_tmp = initialise_model_and_inspector(model_class,oracle,inspector_curiosity_factor,training_size,memory_size, insp_window_size, 
                                                                  cluster_method,random_state=random_state,inspector_model=inspector_model, keep_known_labels=keep_known_labels, 
                                                                  clustering_params=clustering_params_list, initialise_inspector=initialise_inspector,
                                                                  model_object=model_object)
                if initialise_inspector: inspector = inspector_tmp

                telemetry.log_inspector(step_number,inspector)            

            continue # assessing starts on the next step after the model is initialised
        

        
        # D. The nuts and bolts of evaluation of the custom detector algorithm


        # D.1. Main model just predicts, and is maybe later retrained if there is drift
        pred_model = model.predict(x.reshape(1, -1))  # Reshape x to be a 2D array with one sample
        pred_model = int(pred_model[0])

        telemetry.update_metrics(y,pred_model)

        if not dont_propagate:

            # D.2. Get prediction from inspector. The call looks simple, but there's a lot happening in 
            #      the inspetor as part of this call - including clustering, label spreading and training
            if run_inspector:
                pred_inspector = inspector.predict(x) #FIXME #TODO
                pred_inspector = int(pred_inspector[0])
            else:
                pred_inspector = int(y)
                    
            # D.3. CDSeer relies on standard supervised detector model, fed by non-standard shadow/recency model      
            # drift_detector.add_element(int(not(pred_model == pred_inspector)))        
            drift_detector.update(int(not(pred_model == pred_inspector)))      
            is_drifting = drift_detector.drift_detected
        else:
            is_drifting = False
            gt_points, gt_labels = inspector.predict_knownonly(x)
            
            for label_x, label_y in zip(gt_points, gt_labels):
                label_model_pred = model.predict(label_x.reshape(1, -1))  # Reshape x to be a 2D array with one sample
                label_model_pred = int(label_model_pred[0])

                drift_detector.update(int(not(label_model_pred == int(label_y))))      
                is_drifting |= drift_detector.drift_detected
                                     
        # if (is_drifting := drift_detector.detected_change()):                                                    
        if (is_drifting):                                                    
            
            drift_detector = drift_detector.clone()   
            if train_forward:
                model = None # need to retrain
                inspector = None
                oracle.truncate_points(keep_last=1) # Also need a signal to oracle to purge past points - if a shift was detected then old points are likely misleading                
            else:
                model, _ = initialise_model_and_inspector(model_class,oracle,inspector_curiosity_factor,training_size,memory_size, insp_window_size, 
                                                                  cluster_method,random_state=random_state,inspector_model=inspector_model, keep_known_labels=keep_known_labels, 
                                                                  clustering_params=clustering_params_list, initialise_inspector=False, model_object=model_object)
            

            if not run_quiet: print(f"Drift detected at step {step_number} with model prediction {pred_model}"+("" if dont_propagate else f" and inspector prediction {pred_inspector}"))

            # # The reference model is retrained, to allow for more comprehensive evaluation
            # model = retrain_model(model, oracle, training_size) #retraining on latest 1000 points with ground truth labels.
            # # drift_detector.reset()            


        # D.4. Capture statistics and useful info about the experiment

        
        telemetry.log_debug(step_number, y, pred_model,(np.nan if dont_propagate else pred_inspector),is_drifting, data)
        telemetry.log_inspector(step_number,inspector)

        # outcomes.append((step_count, y, pred_model,instance.y_index,instance.y_label,data[step_count,-1],pred_inspector,is_drifting))

        # TODO FIXME record metrics; record drift
    
    return oracle, telemetry


def get_scenarios_to_calc(hyperparam_config:dict,calc_all=False,sample_frac = .4):
        # generate scenarios - in effect reimplements grid search and random search
        hyperparam_combinations = list(product(*hyperparam_config.values()))
        count_to_calc = int(np.ceil((sample_frac if calc_all else 1)*len(hyperparam_combinations)))
        scenarios_to_calc = random.sample(population=hyperparam_combinations,k=count_to_calc) 
        scenarios_to_calc_header = list(hyperparam_config.keys())
        scenarios_chosen=[dict(zip(scenarios_to_calc_header,scen)) for scen in scenarios_to_calc]
        return scenarios_chosen
