"""
Utility functions for the GM-PHD-Tracker.
"""
import copy
import numpy as np
import yaml

from munkres import Munkres
from track import Track

Hungarian = Munkres()

# Load config. Look at the comments in the config.yaml for more information about each variable or parameter.
with open('config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

motion_model_type = config['tracking']['motion_model_type']
use_iou = config['tracking']['use_iou']
reid_threshold = config['tracking']['reid_threshold']


def xcycwh_to_x1y1x2y2(xc, yc, w, h):
    x1 = xc - w / 2.0  # x1 = x
    y1 = yc - h / 2.0  # y1 = y
    x2 = x1 + w        # x2 = X
    y2 = y1 + h        # x2 = Y
    return x1, y1, x2, y2


def cosine_distance(features1, features2):
    dot_product_f1f2 = np.dot(features1, features2.T)
    norm_feat1 = np.linalg.norm(features1)
    norm_feat2 = np.linalg.norm(features2)
    cosine_dist = dot_product_f1f2 / (norm_feat1 * norm_feat2)

    return cosine_dist


def appearance_likelihood(features1, features2):
    cosine_dist = cosine_distance(features1, features2)
    app_lik = np.exp(cosine_dist)/(np.exp(cosine_dist) + np.exp(-cosine_dist))

    return app_lik


def match_features_reid(feature1, feature2, threshold):
    cosine_dist = cosine_distance(feature1, feature2)
    dist = (1 - cosine_dist) / 2.0
    if dist <= threshold:  # threshold = 0.3 or 0.2
        return True, dist
    else:
        return False, dist


def bb_iou(boxA, boxB):   # boxA = np.array([x, y, X, Y]) where X = x + w, Y = y + h
    # determine the (x, y)-coordinates of the intersection rectangle.
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # compute the area of intersection rectangle
    intersection_area = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # compute the area of both the prediction and ground-truth
    # rectangles
    boxA_area = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxB_area = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the intersection area
    iou = intersection_area / float(boxA_area + boxB_area - intersection_area)

    # return the intersection over union value
    return iou


def nms_score_mot(dets, overlap_thresh, score_thresh):
    """
    Apply non-maximum suppression (NMS) to avoid detecting too many overlapping bounding boxes for a given object.
    Args:
        dets: (numpy) The boxes and scores for the image, Shape: [num_detections,5].
        overlap_thresh: (float) The overlap threshold for suppressing unnecessary boxes.
        score_thresh: (float) score threshold below which to discard.

        Note: For MOT test,only take confident detections (threshold at 0) and overlap of 0.3 (default overlap is 0.5).
        Basically, use overlap of 0.3 for MOT16 (uses DPM detector) and MOT17-DPM but default overlap of 0.5 may be
        better for MOT17-FRCNN and SDP.
    Return:
        The indices of the kept boxes with respect to detections.
        detections after NMS and score thresholding.
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)  # overlap

        inds = np.where(ovr <= overlap_thresh)[0]
        order = order[inds + 1]

    dets = dets[keep]  # NMS
    dets_out = dets[np.where(dets[:, 4] >= score_thresh)]  # Score threshold

    return keep, dets_out


def constrain_detections_inFrame(image, detections):
    h, w, c = image.shape

    for i in range(detections.shape[0]):
        # constrain points to within frame
        if detections[i, 0] < 0:  # left
            detections[i, 0] = 0
        if detections[i, 1] < 0:  # top
            detections[i, 1] = 0
        if detections[i, 2] > w:  # right
            detections[i, 2] = w
        if detections[i, 3] > h:  # bottom
            detections[i, 3] = h

    return detections


def compute_associations(image, active_tracks, estimates_m, estimates_feat, include_appearance, appearance_weight):

    im_H, im_W, C = image.shape

    # Compute a matrix of distances
    len_tracks = len(list(active_tracks))
    len_estimates = len(estimates_m)
    distance_matrix = np.zeros((len_estimates, len_tracks))  # matrix of distances
    if len_tracks == 0:
        distance_matrix = None

    track_list = list(active_tracks)  # active_tracks.keys()
    for track_index in range(len(track_list)):
        active_tracks[track_list[track_index]].touched = False

        for estId in range(len_estimates):
            # Compute motion distance
            xc_tra = active_tracks[track_list[track_index]].m_k[0][0]
            yc_tra = active_tracks[track_list[track_index]].m_k[1][0]
            xc_est = estimates_m[estId][0][0]
            yc_est = estimates_m[estId][1][0]

            if use_iou:
                if motion_model_type == 'cv':  # CV
                    w_tra = active_tracks[track_list[track_index]].m_k[4][0]
                    h_tra = active_tracks[track_list[track_index]].m_k[5][0]
                    w_est = estimates_m[estId][4][0]
                    h_est = estimates_m[estId][5][0]
                else:   # RW
                    w_tra = active_tracks[track_list[track_index]].m_k[2][0]
                    h_tra = active_tracks[track_list[track_index]].m_k[3][0]
                    w_est = estimates_m[estId][2][0]
                    h_est = estimates_m[estId][3][0]
                x_tra, y_tra, X_tra, Y_tra = xcycwh_to_x1y1x2y2(xc_tra, yc_tra, w_tra, h_tra)
                x_est, y_est, X_est, Y_est = xcycwh_to_x1y1x2y2(xc_est, yc_est, w_est, h_est)
                bb_tra = np.array([x_tra, y_tra, X_tra, Y_tra])
                bb_est = np.array([x_est, y_est, X_est, Y_est])
                iou = bb_iou(bb_tra, bb_est)
                dist_i = 1.0 - iou  # Jaccard distance = 1.0 - iou
            else:
                # Normalized Euclidean distance (Norm 2)
                dist_i = np.sqrt(((xc_tra - xc_est)/im_W)**2 + ((yc_tra - yc_est)/im_H)**2)  # better

                # # Normalized absolute distance (Norm 1)
                # distX = (abs(xc_tra - xc_est) / im_W)
                # distY = (abs(yc_tra - yc_est) / im_H)  # 1 means a full frame away, 0 means in the same place
                # dist_i = ((distX + distY) / 2.0 + 1e-6)

            if include_appearance:
                # Compute cosine distance
                track_feats = active_tracks[track_list[track_index]].mean_features
                estimate_feats = estimates_feat[estId]
                cosine_dist = cosine_distance(track_feats, estimate_feats)
                dist = (1 - cosine_dist)  # / 2.0  # In this case 0 means identical and 1 means different. Don't divide
                # by 2.0 if you use a normalized Euclidean distance (Norm 2)!

                dist *= appearance_weight  # appearance feature information part
                dist_i *= (1 - appearance_weight)  # motion information part

                dist_total = dist + dist_i
                distance_matrix[estId, track_index] = dist_total
            else:
                distance_matrix[estId, track_index] = dist_i

    # Apply Hungarian algorithm
    if distance_matrix is None:
        # associations = None
        return None, None

    if distance_matrix.shape[0] > 0 and distance_matrix.shape[1] > 0:
        # Column matrices are not supported by munkres. Transpose to row and transpose back.
        if distance_matrix.shape[0] > distance_matrix.shape[1]:
            associations = Hungarian.compute(distance_matrix.transpose().copy())

            # assoc=[list(x).reverse() for x in associations]
            # The above line is wrong - reverse acts in place on the object and returns nothing
            assoc = list()
            for x in associations:
                lst = list(x)
                lst.reverse()
                assoc.append(lst)
            associations = [tuple(x) for x in assoc]
        else:
            associations = Hungarian.compute(distance_matrix.copy())
        return associations, distance_matrix

    return None, None


def create_new_track(frame, active_tracks, global_tracks, archived_tracks, estimates_w_m_P_score_feat, model,
                     include_appearance, include_ReId, window_ReId):

    colour = tuple(np.random.choice(range(256), size=3))
    track_id = len(list(global_tracks))
    track = Track(estimates_w_m_P_score_feat[0], estimates_w_m_P_score_feat[1], estimates_w_m_P_score_feat[2],
                  estimates_w_m_P_score_feat[3], estimates_w_m_P_score_feat[4], model['F_k'], model['Q_k'], colour,
                  track_id, frame, motion_model_type)

    # Do ReId here
    if include_ReId and include_appearance:
        match_reid = False
        matches = []
        k_del = []
        for k, v in archived_tracks.items():
            if (v.end_frame - v.start_frame >= 1) and (frame - v.end_frame <= window_ReId):
                bool_m, dist = match_features_reid(v.mean_features, track.mean_features, reid_threshold)
                if bool_m:
                    matches.append((k, v, dist))
            else:
                k_del.append(k)

        # Delete out of window tracks and no update at least once tracks (hasattr(v, 'endTime'))
        for i in k_del:
            del archived_tracks[i]

        # Find the best match
        if len(matches) > 0:
            if len(matches) == 1:
                first_match = matches[0]
            else:
                sorted_by_dist = sorted(matches, key=lambda m: m[2])  # sort by the distance
                first_match = sorted_by_dist[0]
            k_m = first_match[0]
            track.uniqueid = k_m
            track.colour = archived_tracks[k_m].colour  # self.global_track_dict_reid
            track.features = track.features + archived_tracks[k_m].features

            track.update_mean_features()
            track.reid_status = 1  # ReId happened.
            active_tracks[k_m] = track
            del archived_tracks[k_m]
            match_reid = True

        if not match_reid:
            active_tracks[track_id] = track
            global_tracks[track_id] = track

    else:
        active_tracks[track_id] = track
        global_tracks[track_id] = track

    return active_tracks, global_tracks, archived_tracks


def addOn_prediction(track):

    out_track = copy.deepcopy(track)
    out_track.m_k = track.F_k.dot(track.m_k)
    out_track.P_k = track.Q_k + track.F_k.dot(track.P_k).dot(np.transpose(track.F_k))

    return out_track


def multi_cmc(predicted_intensity, H=np.eye(2, 3)):

    if len(predicted_intensity['m']) == 0:
        return predicted_intensity

    multi_mean = predicted_intensity['m']
    multi_covariance = predicted_intensity['P']

    R = H[:2, :2]
    R6x6 = np.kron(np.eye(3, dtype=float), R)
    t = H[:2, 2]

    for i in range(len(multi_mean)):
        mean = R6x6.dot(multi_mean[i])
        mean[:2] += t.reshape(2,1)
        cov = multi_covariance[i]
        cov = R6x6.dot(cov).dot(R6x6.transpose())

        predicted_intensity['m'][i] = mean
        predicted_intensity['P'][i] = cov

    return predicted_intensity




