import numpy as np
import numpy_indexed as npi
from Oracle import Oracle
from collections import deque
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import DBSCAN, KMeans
from sklearn.semi_supervised import LabelSpreading
from sklearn.metrics import pairwise_distances
from kneefinder import KneeFinder
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import FunctionTransformer

import shap


class Inspector():
    
    
    def points_representation_full(self,points):
        points = np.array(points)
        if not self.perform_clustering: return points
        
        repr = points
        if self.scale_for_clustering: repr = self.scaler.transform(repr)
        if self.use_SHAP_to_cluster: repr_shap = self.explainer(np.array(repr),silent=True).values
        if self.combine_SHAP_with_point: repr = np.concatenate([repr,repr_shap],axis=1)

        if self.use_ref_model:
            pred = self.ref_model.predict(points)
            repr = np.concatenate([repr,pred.reshape(-1,1)],axis=1)

        return repr


    def point_representation_reduced(self,point):
        repr = self.points_representation_full([point])
        repr = self.reducer.transform(repr)
        return repr[0]
        

    
    """ Inspector class that will be used to inspect the model and its performance. """
    def __init__(self, oracle:Oracle,inspector_curiosity_factor, memory_size, insp_window_size, inspector_model, ref_model, ref_model_train_data,keep_known_labels=True,config:dict={}):
        self.feature_threshold = 10
        self.random_seed = config.get("random_seed",1900)
        self.n_jobs = config.get("n_jobs",1)
        self.log_extras = config.get("log_extras",False)    
        self.cluster_method = config.get("cluster_method",None)
        self.window_size = max(insp_window_size, memory_size)
        self.keep_known_labels = keep_known_labels


        self.inspector_model_template = inspector_model
        self.memory_queue = deque(maxlen=memory_size)
        self.memory_queue_keys = set()

        # default_clustering_params = [{"weight":1,"use_SHAP_to_cluster":False,"combine_SHAP_with_point":False,
        #                           "perform_clustering":True,"scale_for_clustering":True,"use_model_pred":False}]
        dict_params = config.get("clustering_params")
        
        self.use_SHAP_to_cluster=dict_params.get("use_SHAP_to_cluster",1)
        self.combine_SHAP_with_point=dict_params.get("combine_SHAP_with_point",1)
        self.perform_clustering=dict_params.get("perform_clustering",1)
        self.use_ref_model=dict_params.get("use_model_pred",1)
        self.scale_for_clustering=dict_params.get("scale_for_clustering",1)


        self.explainer:shap.KernerlExplainer = shap.KernelExplainer(ref_model.predict,data=ref_model_train_data) if self.use_SHAP_to_cluster else None        
        self.ref_model = ref_model if self.use_ref_model else None
        self.scaler = MinMaxScaler().fit(ref_model_train_data) if self.scale_for_clustering else None
        # self.reducer = TruncatedSVD(n_components=self.feature_threshold,random_state=self.random_seed).fit()

        
        init_points, init_labels = oracle.latest_points_labels(self.window_size)        
        # self.inspector_window, self.inspector_window_representation, self.inspector_window_labels
        # A. 
        self.window = deque(iterable=init_points,maxlen=self.window_size)
        # B. 
        init_labels_maybe = init_labels if self.keep_known_labels else np.ones_like(init_points[:,0])*np.nan        
        self.window_labels = deque(iterable=init_labels_maybe,maxlen=self.window_size)
        # C. 
        tmp_window_representation = self.points_representation_full(self.window)
        # D.
        if tmp_window_representation[0].shape[0] > self.feature_threshold:
            self.reducer = TruncatedSVD(n_components=self.feature_threshold,random_state=self.random_seed)
        else:
            self.reducer = FunctionTransformer(func = lambda x:x, inverse_func= lambda x:x, feature_names_out="one-to-one")

        tmp_window_representation = self.reducer.fit_transform(X=np.array(tmp_window_representation))
        self.window_representation = deque(iterable=tmp_window_representation,maxlen=self.window_size)
            

        # Epsilon will be estimated on all data that are being held by the oracle.
        # By custom, as inspector only initialised at original model training, this means only training data is used
        self.epsilon = self.__class__.__estimate_epsilon(oracle.points)
        self.cluster_width = insp_window_size // 100  # As per the paper
        self.curiosity_factor = inspector_curiosity_factor
        self.oracle = oracle

        self.initializeLabelMemory()


    def initializeLabelMemory(self):
        clustering = self.clusterWindow()

        control=0
        while (len(self.memory_queue) < self.memory_queue.maxlen) and control < 100:
            points_to_request_labels_on = self.samplePointsFromCluster(clustering,curiosity_factor=.000001)
            new_GT_points, new_GT_labels = self.oracle.label_points(points_to_request_labels_on)
            if self.log_extras: print(f"Requesting labels for {len(points_to_request_labels_on)} points, got {len(new_GT_points)} new labels.")

            self.incorporateLabelsIntoMemory(new_GT_points, new_GT_labels)
            control+=1
            if control >=100: print("!!! THIS SHOULD NOT BE HAPPENING !!!")
        return self.memory_queue



    def addPointToWindowAndLabelsToMemory(self, x):
        
        self.window.append(x)
        self.window_labels.append(np.nan)
        
        self.window_representation.append(self.point_representation_reduced(x))

        clustering = self.clusterWindow()

        points_to_request_labels_on = self.samplePointsFromCluster(clustering,curiosity_factor=self.curiosity_factor)
        new_GT_points, new_GT_labels = self.oracle.label_or_ignore_points(points_to_request_labels_on)
        if self.log_extras: print(f"Requesting labels for {len(points_to_request_labels_on)} points, got {len(new_GT_points)} new labels.")

        self.incorporateLabelsIntoMemory(new_GT_points, new_GT_labels)
        
        return new_GT_points, new_GT_labels


    def predict(self, point:np.ndarray):
        repr = self.point_representation_reduced(point)
        _ = self.addPointToWindowAndLabelsToMemory(point)
        inspector_model = self.deriveLabelsForWindowAndTrainInspector()
        y_insp = inspector_model.predict(repr.reshape(1, -1))        
        return y_insp
    

    def predict_knownonly(self, point:np.ndarray):
        repr = self.point_representation_reduced(point)
        new_labels_coord, new_labels_y = self.addPointToWindowAndLabelsToMemory(point)
        return new_labels_coord, new_labels_y


    def deriveLabelsForWindowAndTrainInspector(self):
        
        idx_window_nan = [i for i,y in enumerate(self.window_labels) if np.isnan(y)] 
        idx_window_nonnan = [i for i,y in enumerate(self.window_labels) if not np.isnan(y)] 
        # window_nan = np.array([self.inspector_window[i] for i in idx_window_nan])
        # window_nonnan = np.array([self.inspector_window[i] for i in idx_window_nonnan])
        
        # print(f"idx_window_nan = {idx_window_nan}")
        # print(f"len(self.window_representation)={len(self.window_representation)}")
        # print(f"self.window_representation={self.window_representation}")

        # for i in idx_window_nan:
        #     print(i, self.window_representation[i])

        window_repr_nan = np.array([self.window_representation[i] for i in idx_window_nan])
        window_repr_nonnan = np.array([self.window_representation[i] for i in idx_window_nonnan])

        label_nonnan = np.array([self.window_labels[i] for i in idx_window_nonnan])

        repr_points_to_spread_to, pseudo_labels_for_window = self.perform_label_spreading(self.memory_queue,window_repr_nan)


        if len(repr_points_to_spread_to) == 0:
            window_repr = window_repr_nonnan
            labels_for_window = label_nonnan
        elif len(window_repr_nonnan) == 0:
            window_repr = repr_points_to_spread_to
            labels_for_window = pseudo_labels_for_window
        else:
            window_repr = np.concatenate([repr_points_to_spread_to,window_repr_nonnan],axis=0)
            labels_for_window = np.concatenate([pseudo_labels_for_window,label_nonnan],axis=0)
        
        # inspector_model = Pipeline([("scaler",MinMaxScaler()),("model",RandomForestClassifier(random_state=self.random_seed,n_jobs=self.n_jobs,n_estimators=10))])
        # inspector_model = Pipeline([("model",RandomForestClassifier(random_state=self.random_seed,n_jobs=self.n_jobs))])
        # inspector_model = Pipeline([("model",RandomForestClassifier(random_state=self.random_seed,n_jobs=self.n_jobs,n_estimators=100))])
        inspector_model = clone(self.inspector_model_template)
        inspector_model.fit(window_repr,labels_for_window)       

        return inspector_model
    

    def perform_label_spreading(self,memory_queue,repr_points_to_spread_to,**spreading_kwargs):
        _ , repr_mem, y_mem = zip(*memory_queue) #first part, x_mem, to be ignored
        repr_mem, y_mem =  np.array(repr_mem), np.array(y_mem)
        repr_points_to_spread_to = np.array(repr_points_to_spread_to)
        point_count = len(repr_points_to_spread_to)

        
        if len(repr_points_to_spread_to):
            # # print(f"x_mem.shape={x_mem.shape} points_to_spread_to.shape={points_to_spread_to.shape}")
            # print(f"repr_mem.shape={repr_mem.shape}")
            # print(f"repr_mem={repr_mem}")
            # print(f"repr_points_to_spread_to.shape={repr_points_to_spread_to.shape}")
            x_comb = np.vstack((repr_mem,repr_points_to_spread_to))
            y_comb = np.concatenate([y_mem,[-1]*point_count])

            # deduplicate points
            x_comb_dedup, idx_dedup, idx_redup = np.unique(x_comb,return_index=True, return_inverse=True,axis=0)
            y_comb_dedup= y_comb[idx_dedup]
            # x_comb_dedup = x_comb
            # y_comb_dedup= y_comb

            # label_extrapol = LabelSpreading(kernel="knn",alpha=.8)
            label_extrapol = LabelSpreading(**spreading_kwargs,n_jobs=self.n_jobs)
            label_extrapol.fit(x_comb_dedup,y_comb_dedup)

            y_pseudo_window = label_extrapol.transduction_[idx_redup][-point_count:]
        else:
            repr_points_to_spread_to = np.empty(shape=(0,repr_mem.shape[1]))
            y_pseudo_window = np.empty(shape=(0,))
        # y_pseudo_window = label_extrapol.transduction_[-point_count:]
        return repr_points_to_spread_to,y_pseudo_window
    
    
    def incorporateLabelsIntoMemory(self,new_labeled_points, new_labels):        
        
        for x,y in zip(new_labeled_points,new_labels):                            
            # remove instances from the short term memory queue
            if tuple(x) in self.memory_queue_keys:                
                existing_idxs = [i for i,val in enumerate(self.memory_queue) if (val[0]==x).all()]
                self.memory_queue_keys.remove(tuple(x))
                for idx in existing_idxs[::-1]:           
                    del self.memory_queue[idx]

            # update instances in the (bigger) inspector window
            if self.keep_known_labels:
                window_idxs = [i for i,val in enumerate(self.window) if (val==x).all()]
                for idx in window_idxs:
                    self.window_labels[idx]=y
                

        for x,y in zip(new_labeled_points,new_labels):
            repr = self.point_representation_reduced(x)
            self.memory_queue.append((x,repr,y))
            self.memory_queue_keys.add(tuple(x))

        return self.memory_queue
        
    

    
    def clusterWindow(self):
        methods_to_try = ([self.cluster_method.lower()] if self.cluster_method is not None else {'dbscan','kmeans'})        

        window = self.window_representation

        clustering = None
        cluster_model = None
        if self.perform_clustering:
            # print("A")
            inspector_window_scaled= (MinMaxScaler().fit_transform(window) if self.scale_for_clustering else window)

            if 'dbscan' in methods_to_try:                
                cluster_model = DBSCAN(eps=self.epsilon, min_samples=self.cluster_width,n_jobs=self.n_jobs).fit(inspector_window_scaled)
                clustering = cluster_model.labels_
            
            if (clustering is None or not __class__.__isClusteringSufficient(clustering,log_extras=self.log_extras)) and 'kmeans' in methods_to_try:
                # Try with KMeans instead
                cluster_model = KMeans(n_clusters = self.cluster_width, random_state=self.random_seed).fit(inspector_window_scaled)
                clustering = cluster_model.labels_
            window = inspector_window_scaled
        else:
            # print("B")
            clustering = np.array([0]*len(window))
            window = window

        clustering = np.array(clustering)

        if self.log_extras:
            print(f"Ended up with clustering model {cluster_model}")
            print(clustering)

        return clustering

    

    # def clusterWindow(self):  
    #     methods_to_try = ([self.cluster_method.lower()] if self.cluster_method is not None else {'dbscan','kmeans'})        

    #     clustering = None
    #     if 'dbscan' in methods_to_try:                
    #         cluster_model = DBSCAN(eps=self.epsilon, min_samples=self.cluster_width,n_jobs=N_JOBS).fit(self.inspector_window)
    #         clustering = cluster_model.labels_
        
    #     if (clustering is None or not self.isClusteringSufficient(clustering)) and 'kmeans' in methods_to_try:
    #         # Try with KMeans instead
    #         cluster_model = KMeans(n_clusters = self.cluster_width, random_state=RANDOM_SEED).fit(self.inspector_window)
    #         clustering = cluster_model.labels_

    #     if LOG_ME: print(f"Ended up with clustering model {cluster_model}")
    #     return clustering #, cluster_model
    
    @staticmethod
    def __isClusteringSufficient(clustering, max_noise_ratio=.2, min_cluster_ratio=.01,max_cluster_ratio=.2,log_extras=False):
        total_points=len(clustering)
        
        noise_points_ratio = len(clustering[clustering == -1]) / total_points
        cluster_ratio = len(np.unique(clustering[clustering != -1])) / total_points

        if log_extras: print(f"Assessing goodness of clustering: noise points = {noise_points_ratio:.2%}, cluster ratio = {cluster_ratio:.2%}")

        if noise_points_ratio > max_noise_ratio: return False
        if cluster_ratio > max_cluster_ratio or cluster_ratio < min_cluster_ratio: return False
        
        return True

    # def isClusteringSufficient(self,clustering, max_noise_ratio=.2, min_cluster_ratio=.01,max_cluster_ratio=.2):
    #     total_points=len(clustering)
        
    #     noise_points_ratio = len(clustering[clustering == -1]) / total_points
    #     cluster_ratio = len(np.unique(clustering[clustering != -1])) / total_points

    #     if LOG_ME: print(f"Assessing goodness of clustering: noise points = {noise_points_ratio:.2%}, cluster ratio = {cluster_ratio:.2%}")

    #     if noise_points_ratio > max_noise_ratio: return False
    #     if cluster_ratio > max_cluster_ratio or cluster_ratio < min_cluster_ratio: return False
        
    #     return True

    # @staticmethod
    # def __samplePointsFromClusters(clustering, inspector_window, curiosity_factor):
    #     window_tmp=np.array(inspector_window)
    #     window_tmp =window_tmp[clustering != -1]
    #     clustering_tmp = clustering[clustering != -1]
        
    #     # cluster_count = len(np.unique(clustering_tmp))
    #     tgt_ratio =min(max(curiosity_factor,0),1)
    #     # selected_points, _ = train_test_split(window_tmp,train_size=tgt_ratio,stratify=clustering_tmp)
    #     selected_points = __class__.__stratified_sample(window_tmp, clustering_tmp, frac=tgt_ratio)
    #     return selected_points

    # def samplePointsFromClusters(self,clustering, curiosity_factor):
    #     window_tmp=np.array(self.inspector_window)
    #     window_tmp =window_tmp[clustering != -1]
    #     clustering_tmp = clustering[clustering != -1]
        
    #     # cluster_count = len(np.unique(clustering_tmp))
    #     tgt_ratio =min(max(curiosity_factor,0),1)
    #     # selected_points, _ = train_test_split(window_tmp,train_size=tgt_ratio,stratify=clustering_tmp)
    #     selected_points = __class__.__stratified_sample(window_tmp, clustering_tmp, frac=tgt_ratio)
    #     return selected_points

    # def samplePointsFromClusters_multiple(self,clusterings, curiosity_factor):
    #     selected_points_all = []
    #     for weight, clustering in clusterings:
    #         curiosity_factor_chgd = curiosity_factor*weight
    #         window_tmp=np.array(self.window)
    #         window_tmp =window_tmp[clustering != -1]
    #         clustering_tmp = clustering[clustering != -1]
            
    #         # cluster_count = len(np.unique(clustering_tmp))
    #         tgt_ratio =min(max(curiosity_factor_chgd,0),1)
    #         # selected_points, _ = train_test_split(window_tmp,train_size=tgt_ratio,stratify=clustering_tmp)
    #         selected_points = __class__.__stratified_sample(window_tmp, clustering_tmp, frac=tgt_ratio)
    #         selected_points_all.append(selected_points)
    #     return np.concatenate(selected_points_all,axis=0)
    

    def samplePointsFromCluster(self,clustering, curiosity_factor):

        curiosity_factor_chgd = curiosity_factor
        window_tmp=np.array(self.window)

        window_tmp =window_tmp[clustering != -1]
        
        clustering_tmp = clustering[clustering != -1]
        
        # cluster_count = len(np.unique(clustering_tmp))
        tgt_ratio =min(max(curiosity_factor_chgd,0),1)
        # selected_points, _ = train_test_split(window_tmp,train_size=tgt_ratio,stratify=clustering_tmp)

        selected_points = __class__.__stratified_sample(window_tmp, clustering_tmp, frac=tgt_ratio)
            
        return np.concatenate([selected_points],axis=0)


    @staticmethod
    def __stratified_sample(population_to_split, strata_vector, frac=.05):
        """ It will always try to get at least one point from each stratum, and then sample the rest of the points from the stratum."""
        
        strata = npi.group_by(strata_vector).split(population_to_split)
        selected_indices = [np.random.choice(len(s), size=int(max(1,len(s)*frac)), replace=False) for s in strata]
        selected_points = [s[indices] for s,indices in zip(strata,selected_indices)]
        
        return np.concatenate(selected_points,axis=0)
    
    # def samplePointsFromClusters(self,clustering):
    #     window_tmp=np.array(self.inspector_window)
    #     window_tmp =window_tmp[clustering != -1]
    #     clustering_tmp = clustering[clustering != -1]
        
    #     # cluster_count = len(np.unique(clustering_tmp))
    #     tgt_ratio =min(max(self.curiosity_factor,0),1)
    #     # selected_points, _ = train_test_split(window_tmp,train_size=tgt_ratio,stratify=clustering_tmp)
    #     selected_points = stratified_sample(window_tmp, clustering_tmp, frac=tgt_ratio)
    #     return selected_points
    

    # @staticmethod
    # def estimate_epsilon(data):
    #     return __class__.__estimate_epsilon(data)

    @staticmethod
    def __estimate_epsilon(data):
        
        # For now just calculate curvature through one-sided differences, though central differences might be more appropriate
        dists = pairwise_distances(data)
        dists = dists+np.diag(np.ones(dists.shape[0])*np.inf)  # Add infinity to the diagonal to avoid self-distances
        dists = dists.min(axis=0)        
        
        sorted_dists = np.sort(dists)
        
        kf = KneeFinder(range(len(sorted_dists)),sorted_dists)
        _ , knee_val = kf.find_knee()
        # kf.plot()
 
        return knee_val      
