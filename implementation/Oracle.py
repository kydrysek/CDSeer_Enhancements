import numpy as np


class Oracle():
    
    """ It would be nice to just extend a LookupTable classifier, but one doesn't seem to exist in sklearn, river of capymoa.
    Also Oracle will comply only with a percentage of requests, but this could have been wrapped around."""
    def __init__(self,compliance_factor:float, points_to_hold:int, feature_count:int,random_state):
        # Cannot initialise with actual objects as oracle needs an instance of a stream point to know the shape        
        self.points = np.empty(shape=(0,feature_count))
        self.labels, self.idxs =  np.empty(shape=(0)), np.empty(shape=(0)) 
                
        self.compliance_factor = compliance_factor
        self.labels_given = set()
        self.points_to_hold = points_to_hold
        np.random.seed(random_state)        

            
    def number_of_available_points(self):        
        return (0 if self.points is None else len(self.points))
    
    def truncate_points(self,keep_last=-1):
        keep_last = self.points_to_hold if keep_last < 0 else keep_last
        boundary = - keep_last if keep_last > 0 else len(self.points)
        
        self.points = self.points[boundary:]
        self.labels = self.labels[boundary:]
        self.idxs = self.idxs[boundary:]
        

    def add_point_and_label(self,point,label,idx):
        """ Adds a point and its label to the oracle. """        
        # self.labeled_data[tuple(point)] = label
        # if self.points is None:
        #     self.points, self.labels, self.idxs = np.empty(shape=(0,len(point))), np.empty(shape=(0)), np.empty(shape=(0))                   
        if isinstance(point,dict):
            point = np.array(list(point.values()))
        point=point.reshape((1,-1))
        self.points = np.append(arr=self.points, values=point, axis=0)
        self.labels = np.append(arr=self.labels, values=[label], axis=0)
        self.idxs = np.append(arr=self.idxs, values=[idx], axis=0)        

        self.truncate_points(self.points_to_hold)
        
    
    def percentage_of_labels_provided(self,points_to_ignore=0):
        return len(self.labels_given)/(max(self.idxs)-points_to_ignore)

    def get_label(self,point,record_it):
        rev_points = self.points[::-1]
        mask = (rev_points == point).all(axis=1)
        if not mask.any(axis=0):
            print("Houston, a problem")
            print(point)
            print(len(self.points), len(self.labels), self.points[0],self.points[1],"...", self.points[-2],self.points[-1])
        label = self.labels[::-1][mask][0]
        idx = self.idxs[::-1][mask][0]

        if record_it: self.labels_given.add(idx)

        return label
        # return self.labeled_data.get(tuple(point), None)
    
    def get_label_or_ignore(self,point,record_it):
        """ Returns the label of the point if it exists, otherwise returns None. """
        if np.random.rand() < self.compliance_factor:
            return self.get_label(point,record_it)
        else:
            return None
        
    # def get_label_multiple(self,points):
    #     """ Returns the labels of the points if they exist, otherwise returns None. """
    #     labels = [self.get_label(point) for point in points]
    #     return labels
    
    # def label_or_ignore_points(self,points,record_it=True):
    
    def label_points(self,points):
        """ Returns the labels of the points if they exist, otherwise returns None. """
        record_it=False
        labels = [self.get_label(point,record_it) for point in points]        
        return points, labels
    
    def label_or_ignore_points(self,points):
        """ Returns the labels of the points if they exist, otherwise returns None. """
        record_it=True
        labels = [self.get_label_or_ignore(point,record_it) for point in points]
        ret_points = [point for point, label in zip(points, labels) if label is not None]
        ret_lables = [label for label in labels if label is not None]
        return ret_points, ret_lables

    def latest_points_labels(self,size,record_it=True):        
        points = self.points[-size:]
        labels = self.labels[-size:]
        if record_it: self.labels_given.update(labels)
        return points, labels

    def latest_points(self,size):        
        points, _ = self.latest_points_labels(size,record_it=False)        
        return points
    

    
