from dataclasses import dataclass
from enum import Enum, auto

class ClusteringParamsTypes(Enum):
    DBSCAN_noScale_Model = auto()
    NoCluster_noScale = auto()
    DBSCAN_Scale_noModel = auto()
    DBSCAN_noScale_Model = auto()
    DBSCAN_Scale_Model = auto()

    SHAP_noScale_noModel= auto()
    SHAP_Scale_noModel= auto()
    SHAP_noScale_Model= auto()
    SHAP_Scale_Model= auto()

    SHAP_noScale_noModel= auto()
    SHAP_Scale_noModel= auto()
    SHAP_noScale_Model= auto()
    SHAP_Scale_Model= auto()

@dataclass
class ClusteringParams:
    weight:int #1,
    use_SHAP_to_cluster:bool 
    combine_SHAP_with_point:bool
    perform_clustering:bool
    scale_for_clustering:bool
    use_model_pred:bool


@dataclass
class RunConfig:
    file_path: str
    training_size: int #WARM_UP_WINDOW_SIZE,
    memory_size:int #MEMORY_SIZE,
    insp_window_size:int #INSPECTOR_WINDOW_SIZE,
    cluster_method: object #None,
    model_object: object #model_object, 
    drift_detector:object #drift_detector, 
    keep_known_labels: bool #False,
    inspector_curiosity_factor:float #inspector_curiosity,
    oracle_compliance: float #oracle_compliance, 
    inspector_model:object #RandomForestClassifier(n_estimators=100,random_state=42),#KNeighborsClassifier()#
    clustering_params:ClusteringParams #clustering_params_list,
    train_forward:bool #train_forward,
    run_quiet: bool #False, 
    dont_propagate:bool #False



    