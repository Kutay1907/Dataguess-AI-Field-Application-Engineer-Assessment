import numpy as np
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment
import collections

class STrack:
    """
    Single Track object.
    Holds state, history, and properties of a tracked object.
    """
    def __init__(self, tlwh, score, class_id):
        # tlwh: top-left width height
        self._tlwh = np.asarray(tlwh, dtype=np.float32)
        self.kalman_filter = None
        self.mean = None
        self.covariance = None
        self.is_activated = False
        self.score = score
        self.class_id = class_id
        self.track_id = 0
        self.state = 0 # 0: New, 1: Tracked, 2: Lost, 3: Removed
        self.frame_id = 0
        self.start_frame = 0

    @property
    def tlwh(self):
        if self.mean is None:
            return self._tlwh.copy()
        ret = self.mean[:4].copy()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2
        return ret
    
    @property
    def tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]
        return ret

    def activate(self, kalman_filter, frame_id):
        """Start a new tracklet"""
        self.kalman_filter = kalman_filter
        self.track_id = self.next_id()
        self.mean, self.covariance = self.kalman_filter.initiate(self.xyah)
        self.frame_id = frame_id
        self.start_frame = frame_id
        self.state = 1 # Tracked
        self.is_activated = True

    def re_activate(self, new_track, frame_id, new_id=False):
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )
        self.frame_id = frame_id
        self.state = 1 # Tracked
        self.is_activated = True
        self.score = new_track.score

    def update(self, new_track, frame_id):
        self.frame_id = frame_id
        self.state = 1 # Tracked
        self.is_activated = True
        self.score = new_track.score
        self.mean, self.covariance = self.kalman_filter.update(
            self.mean, self.covariance, new_track.xyah
        )

    @property
    def xyah(self):
        """Convert bounding box to format `(center x, center y, aspect ratio, height)`"""
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]
        return ret

    @staticmethod
    def next_id():
        STrack._count += 1
        return STrack._count

    _count = 0


class KalmanFilterXYAH:
    """
    Simple Kalman Filter for bounding box tracking.
    State: [xc, yc, ratio, h, vxc, vyc, vratio, vh]
    Observation: [xc, yc, ratio, h]
    """
    def __init__(self):
        ndim, dt = 4, 1.

        # Create Kalman Filter model matrices
        self._motion_mat = np.eye(2 * ndim, 2 * ndim)
        for i in range(ndim):
            self._motion_mat[i, ndim + i] = dt
            
        self._update_mat = np.eye(ndim, 2 * ndim)
        self._std_weight_position = 1. / 20
        self._std_weight_velocity = 1. / 160

    def initiate(self, measurement):
        mean_pos = measurement
        mean_vel = np.zeros_like(mean_pos)
        mean = np.r_[mean_pos, mean_vel]

        std = [
            2 * self._std_weight_position * measurement[3],
            2 * self._std_weight_position * measurement[3],
            1e-2,
            2 * self._std_weight_position * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            10 * self._std_weight_velocity * measurement[3],
            1e-5,
            10 * self._std_weight_velocity * measurement[3]
        ]
        covariance = np.diag(np.square(std))
        return mean, covariance

    def predict(self, mean, covariance):
        std_pos = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-2,
            self._std_weight_position * mean[3]
        ]
        std_vel = [
            self._std_weight_velocity * mean[3],
            self._std_weight_velocity * mean[3],
            1e-5,
            self._std_weight_velocity * mean[3]
        ]
        motion_cov = np.diag(np.square(np.r_[std_pos, std_vel]))
        mean = np.dot(self._motion_mat, mean)
        covariance = np.linalg.multi_dot((
            self._motion_mat, covariance, self._motion_mat.T
        )) + motion_cov
        return mean, covariance

    def update(self, mean, covariance, measurement):
        projected_mean, projected_cov = self.project(mean, covariance)
        chol_factor, lower = scipy_linalg_cholesky(projected_cov, lower=True)
        kalman_gain = scipy_linalg_cho_solve((chol_factor, lower), np.dot(covariance, self._update_mat.T).T).T
        
        innovation = measurement - projected_mean
        new_mean = mean + np.dot(innovation, kalman_gain.T)
        new_covariance = covariance - np.linalg.multi_dot((
            kalman_gain, projected_cov, kalman_gain.T
        ))
        return new_mean, new_covariance

    def project(self, mean, covariance):
        std = [
            self._std_weight_position * mean[3],
            self._std_weight_position * mean[3],
            1e-1,
            self._std_weight_position * mean[3]
        ]
        innovation_cov = np.diag(np.square(std))
        mean = np.dot(self._update_mat, mean)
        covariance = np.linalg.multi_dot((
            self._update_mat, covariance, self._update_mat.T
        )) + innovation_cov
        return mean, covariance

# Helper functions for Linear Algebra (since we stripped scipy dependency logic slightly for purity)
import scipy.linalg
def scipy_linalg_cholesky(a, lower=True):
    return scipy.linalg.cho_factor(a, lower=lower)
def scipy_linalg_cho_solve(c, b):
    return scipy.linalg.cho_solve(c, b)


def iou_batch(atlbrs, btlbrs):
    """
    Compute Cost Matrix using IoU
    """
    atlbrs = np.ascontiguousarray(atlbrs, dtype=np.float64)
    btlbrs = np.ascontiguousarray(btlbrs, dtype=np.float64)
    
    an = len(atlbrs)
    bn = len(btlbrs)
    
    ious = np.zeros((an, bn))
    if an == 0 or bn == 0:
        return ious
        
    area_a = (atlbrs[:, 2] - atlbrs[:, 0]) * (atlbrs[:, 3] - atlbrs[:, 1])
    area_b = (btlbrs[:, 2] - btlbrs[:, 0]) * (btlbrs[:, 3] - btlbrs[:, 1])
    
    for i in range(an):
        # Broadcasting
        xx1 = np.maximum(atlbrs[i, 0], btlbrs[:, 0])
        yy1 = np.maximum(atlbrs[i, 1], btlbrs[:, 1])
        xx2 = np.minimum(atlbrs[i, 2], btlbrs[:, 2])
        yy2 = np.minimum(atlbrs[i, 3], btlbrs[:, 3])
        
        w = np.maximum(0., xx2 - xx1)
        h = np.maximum(0., yy2 - yy1)
        inter = w * h
        union = area_a[i] + area_b - inter
        ious[i, :] = inter / union
    return ious


class ByteTracker:
    def __init__(self, track_thresh=0.5, track_buffer=30, match_thresh=0.8):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.kalman_filter = KalmanFilterXYAH()
        
        self.tracked_stracks = []  # type: list[STrack]
        self.lost_stracks = []     # type: list[STrack]
        self.removed_stracks = []  # type: list[STrack]
        
        self.frame_id = 0

    def update(self, output_results):
        """
        Update tracker with new detections.
        output_results: tensor/array [[x1, y1, x2, y2, score, class], ...]
        """
        self.frame_id += 1
        activated_stracks = []
        refind_stracks = []
        lost_stracks = []
        removed_stracks = []
        
        # 1. Separate detections
        if isinstance(output_results, np.ndarray):
            output_results = output_results
        else:
            output_results = output_results.cpu().numpy()
            
        scores = output_results[:, 4]
        bboxes = output_results[:, :4] # x1y1x2y2
        classes = output_results[:, 5]
        
        remain_inds = scores > self.track_thresh
        inds_low = scores > 0.1
        inds_high = scores < self.track_thresh
        inds_second = np.logical_and(inds_low, inds_high)
        
        dets_first = []
        dets_second = []
        
        # High confidence detections
        for i in np.where(remain_inds)[0]:
            # Convert x1y1x2y2 to tlwh
            box = bboxes[i]
            tlwh = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            dets_first.append(STrack(tlwh, scores[i], classes[i]))
            
        # Low confidence detections
        for i in np.where(inds_second)[0]:
            box = bboxes[i]
            tlwh = [box[0], box[1], box[2]-box[0], box[3]-box[1]]
            dets_second.append(STrack(tlwh, scores[i], classes[i]))
            
        # 2. Predict Track States
        strack_pool = join_stracks(self.tracked_stracks, self.lost_stracks)
        self.predict_tracks(strack_pool)
        
        # 3. First Association (High Conf Dets <-> Tracked/Lost Tracks)
        dists = iou_distance(strack_pool, dets_first)
        matches, u_track, u_detection = linear_assignment(dists, thresh=self.match_thresh)
        
        for itracked, idet in matches:
            track = strack_pool[itracked]
            det = dets_first[idet]
            if track.state == 1: # Tracked
                track.update(det, self.frame_id)
                activated_stracks.append(track)
            else: # Re-activated
                track.re_activate(det, self.frame_id, new_id=False)
                refind_stracks.append(track)
                
        # 4. Second Association (Low Conf Dets <-> Unmatched Tracks)
        # Only associate with tracks that are currently tracked (not lost)
        r_tracked_stracks = [strack_pool[i] for i in u_track if strack_pool[i].state == 1]
        
        dists = iou_distance(r_tracked_stracks, dets_second)
        matches, u_track, u_detection_second = linear_assignment(dists, thresh=0.5)
        
        for itracked, idet in matches:
            track = r_tracked_stracks[itracked]
            det = dets_second[idet]
            if track.state == 1:
                track.update(det, self.frame_id)
                activated_stracks.append(track)
                
        # 5. Deal with Unmatched Tracks
        for it in u_track:
            track = r_tracked_stracks[it]
            if not track.state == 2: # If not already lost
                track.state = 2 # Lost
                lost_stracks.append(track)
                
        # 6. Deal with Unmatched High-Conf Detections (New Tracks)
        for inew in u_detection:
            track = dets_first[inew]
            if track.score < self.track_thresh:
                continue
            track.activate(self.kalman_filter, self.frame_id)
            activated_stracks.append(track)
            
        # 7. Update States
        self.tracked_stracks = [t for t in self.tracked_stracks if t.state == 1]
        self.tracked_stracks = join_stracks(self.tracked_stracks, activated_stracks)
        self.tracked_stracks = join_stracks(self.tracked_stracks, refind_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
        self.lost_stracks.extend(lost_stracks)
        self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
        self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(self.tracked_stracks, self.lost_stracks)
        
        # Filter dead tracks
        val_lost_tracks = []
        for track in self.lost_stracks:
            if self.frame_id - track.frame_id > self.track_buffer:
                self.removed_stracks.append(track)
            else:
                val_lost_tracks.append(track)
        self.lost_stracks = val_lost_tracks
        
        return [t for t in self.tracked_stracks if t.is_activated]

    def predict_tracks(self, stracks):
        for t in stracks:
            t.mean, t.covariance = self.kalman_filter.predict(t.mean, t.covariance)


def join_stracks(tlista, tlistb):
    exists = {}
    res = []
    for t in tlista:
        exists[t.track_id] = 1
        res.append(t)
    for t in tlistb:
        tid = t.track_id
        if not exists.get(tid, 0):
            exists[tid] = 1
            res.append(t)
    return res

def sub_stracks(tlista, tlistb):
    stracks = {}
    for t in tlista:
        stracks[t.track_id] = t
    for t in tlistb:
        tid = t.track_id
        if stracks.get(tid, 0):
            del stracks[tid]
    return list(stracks.values())

def remove_duplicate_stracks(stracksa, stracksb):
    # pdist = iou_distance(stracksa, stracksb)
    # For simplicitly, assume strict ID checks in join/sub covers most. 
    # Proper ByteTrack does IoU check for duplicates.
    return stracksa, stracksb

def iou_distance(atracks, btracks):
    if (len(atracks) == 0 or len(btracks) == 0):
        return np.zeros((len(atracks), len(btracks)), dtype=np.float32)
    atlbrs = [track.tlbr for track in atracks]
    btlbrs = [track.tlbr for track in btracks]
    ious = iou_batch(np.asarray(atlbrs), np.asarray(btlbrs))
    cost_matrix = 1 - ious
    return cost_matrix

def linear_assignment(cost_matrix, thresh):
    if cost_matrix.size == 0:
        return np.empty((0, 2), dtype=int), tuple(range(cost_matrix.shape[0])), tuple(range(cost_matrix.shape[1]))
    
    matches, unmatched_a, unmatched_b = [], [], []
    x, y = linear_sum_assignment(cost_matrix)
    
    for i in range(len(x)):
        if cost_matrix[x[i], y[i]] > thresh:
            unmatched_a.append(x[i])
            unmatched_b.append(y[i])
        else:
            matches.append([x[i], y[i]])
            
    unmatched_a = list(set(range(cost_matrix.shape[0])) - set([m[0] for m in matches]))
    unmatched_b = list(set(range(cost_matrix.shape[1])) - set([m[1] for m in matches]))
            
    return np.array(matches), unmatched_a, unmatched_b
