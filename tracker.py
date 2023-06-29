import argparse
import copy
from timeit import default_timer as timer

import cv2
import numpy as np
import os
import yaml

import torch
from ultralytics import YOLO

from GM_PHD_Filter import GM_PHD_Filter
from reid_feature_extractor import FeatureExtractor
from track_utility import nms_score_mot, constrain_detections_inFrame, create_new_track, compute_associations, \
    addOn_prediction

np.random.seed(5)  # For reproducibility


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking pipeline for GM-PHD-Tracker.')
    parser.add_argument('--base_data', type=str, default='./data',
                        help="Path to base tracking data folder.")
    parser.add_argument('--base_result', type=str, default='./result',
                        help='Path to base tracking result folder to be saved to.')
    parser.add_argument('--reid_path', type=str, default='./model/reid_resnet34.pth',
                        help='Path to base tracking result folder to be saved to.')

    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"  # Check how many GPUs are there using
    # torch.cuda.device_count()

    # Load config. Look at the comments in the config.yaml for more information about each variable or parameter.
    with open('config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)

    score_thresh = config['detection']['score_thresh']
    overlap_thresh = config['detection']['overlap_thresh']
    display_detections = config['detection']['display_detections']
    motion_model_type = config['tracking']['motion_model_type']
    MOT_data_type = config['tracking']['MOT_data_type']
    TrainTest_type = config['tracking']['TrainTest_type']
    similarity_threshold = config['tracking']['similarity_threshold']
    prediction_time_threshold = config['tracking']['prediction_time_threshold']
    track_kill_time_threshold = config['tracking']['track_kill_time_threshold']
    track_min_length = config['tracking']['track_min_length']
    appearance_weight = config['tracking']['appearance_weight']
    window_ReId = config['tracking']['window_ReId']
    detections_type = config['tracking']['detections_type']
    feature_extraction_stage_type = config['tracking']['feature_extraction_stage_type']
    use_Jmax = config['tracking']['use_Jmax']
    is_AddOn_prediction = config['tracking']['is_AddOn_prediction']
    include_appearance = config['tracking']['include_appearance']
    include_ReId = config['tracking']['include_ReId']
    display_tracks = config['tracking']['display_tracks']
    display_trajectories = config['tracking']['display_trajectories']
    base_data = args.base_data
    base_result = args.base_result
    reid_path = args.reid_path

    # Object detector and feature extractor
    if not detections_type:
        detector = YOLO('yolov8l.pt')  # YOLOv8: https://github.com/ultralytics/ultralytics
        detector.to(device)
    feat_extractor = FeatureExtractor(reid_path)  # Features extraction using pre-trained ResNet34.

    if MOT_data_type == 1:
        if TrainTest_type == 1:
            phase = 'train'
            sequences = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
        else:
            phase = 'test'
            sequences = ['MOT16-01', 'MOT16-03', 'MOT16-06', 'MOT16-07', 'MOT16-08', 'MOT16-12', 'MOT16-14']
    elif MOT_data_type == 2:
        if TrainTest_type == 1:
            phase = 'train'
            sequences = ['MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',
                         'MOT17-13-DPM']
        else:
            phase = 'test'
            sequences = ['MOT17-01-DPM', 'MOT17-03-DPM', 'MOT17-06-DPM', 'MOT17-07-DPM', 'MOT17-08-DPM', 'MOT17-12-DPM',
                         'MOT17-14-DPM']
    elif MOT_data_type == 3:
        if TrainTest_type == 1:
            phase = 'train'
            sequences = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',
                         'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
        else:
            phase = 'test'
            sequences = ['MOT17-01-FRCNN', 'MOT17-03-FRCNN', 'MOT17-06-FRCNN', 'MOT17-07-FRCNN', 'MOT17-08-FRCNN',
                         'MOT17-12-FRCNN', 'MOT17-14-FRCNN']
        overlap_thresh = 0.5
    elif MOT_data_type == 4:
        if TrainTest_type == 1:
            phase = 'train'
            sequences = ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP',
                         'MOT17-13-SDP']
        else:
            phase = 'test'
            sequences = ['MOT17-01-SDP', 'MOT17-03-SDP', 'MOT17-06-SDP', 'MOT17-07-SDP', 'MOT17-08-SDP', 'MOT17-12-SDP',
                         'MOT17-14-SDP']
        overlap_thresh = 0.5
    elif MOT_data_type == 5:
        if TrainTest_type == 1:
            phase = 'train'
            sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
        else:
            phase = 'test'
            sequences = ['MOT20-04', 'MOT20-06', 'MOT20-07', 'MOT20-08']
        overlap_thresh = 0.5
    elif MOT_data_type == 6:
        if TrainTest_type == 1:
            phase = 'train'
            sequences = ['1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4', '6.MP4', '7.mp4', '8.mp4', '9.mp4', '10.MOV',
                         '11.mp4', '12.mp4', '13.mp4', '14.mp4', '15.mp4', '16.mp4', '17.mp4', '18.MOV', '19.mp4']
        else:
            phase = 'test'
            sequences = ['20.mp4', '21.mp4', '22.mp4', '23.mp4', '24.mp4', '25.MOV', '26.mp4', '27.mp4', '28.mp4',
                         '29.mp4', '30.mp4', '31.mp4', '32.mp4']
        overlap_thresh = 0.5
    else:
        print('\nError: Set to correct MOT dataset: Set to 1 for MOT16, to 2 for MOT17-DPM, to 3 for MOT17-FRCNN, to 4 '
              'for MOT17-SDP, to 5 for MOT20 or to 6 for HiEve.')
        exit()

    # Tracking starts here.
    for seq in sequences:

        print('Sequence:', seq)

        if MOT_data_type == 1:
            data_folder = os.path.join(base_data, 'MOT16/%s/%s/det/det.txt')
            seq_dets = np.loadtxt(data_folder % (phase, seq), delimiter=',')  # load detections
            result_folder = os.path.join(base_result, 'MOT16')
            if not os.path.isdir(result_folder):
                os.mkdir(result_folder)
            result_file = result_folder + '/' + seq + '.txt'
        elif MOT_data_type == 5:
            data_folder = os.path.join(base_data, 'MOT20/%s/%s/det/det.txt')
            seq_dets = np.loadtxt(data_folder % (phase, seq), delimiter=',')  # load detections
            result_folder = os.path.join(base_result, 'MOT20')
            if not os.path.isdir(result_folder):
                os.mkdir(result_folder)
            result_file = result_folder + '/' + seq + '.txt'
        elif MOT_data_type == 6:  # HiEve
            seq_name = seq.split('.')[0]
            if phase == 'train':
                video_filename = os.path.join(base_data, 'HiEve/HIE20/videos', seq)
                seq_dets = np.loadtxt(os.path.join(base_data, 'HiEve/public_detection_train/train', seq_name + '.txt'),
                                      delimiter=',')  # load detections
            else:  # if phase == 'test'
                video_filename = os.path.join(base_data, 'HiEve/HIE20test/test', seq)
                seq_dets = np.loadtxt(os.path.join(base_data, 'HiEve/public_detection_test/test', seq_name + '.txt'),
                                      delimiter=',')  # load detections
            result_folder = os.path.join(base_result, 'HiEve')
            if not os.path.isdir(result_folder):
                os.mkdir(result_folder)
            result_file = result_folder + '/' + seq + '.txt'
            cap = cv2.VideoCapture(video_filename)
        else:  # MOT_data_type == 2, 3 or 4
            data_folder = os.path.join(base_data, 'MOT17/%s/%s/det/det.txt')
            seq_dets = np.loadtxt(data_folder % (phase, seq), delimiter=',')  # load detections
            result_folder = os.path.join(base_result, 'MOT17')
            if not os.path.isdir(result_folder):
                os.mkdir(result_folder)
            result_file = result_folder + '/' + seq + '.txt'

        # Initialize pruned_intensity for GM-PHD-Filter.
        pruned_intensity = dict()
        pruned_intensity['w'] = []
        pruned_intensity['m'] = []
        pruned_intensity['P'] = []
        pruned_intensity['score'] = []
        pruned_intensity['feat'] = []

        active_tracks = {}  # Only active tracks
        global_tracks = {}  # Both dead and active tracks
        archived_tracks = {}  # Only dead tracks

        N_k = 0  # Number of observations at time k
        M_k_1 = 0  # Number of extracted states at time k-1

        output_file = open(result_file, 'w')
        start_time = timer()

        for frame in range(int(seq_dets[:, 0].max())):
            frame += 1  # detection and frame numbers begin at 1.
            print('frame: ', frame)

            if MOT_data_type == 1:
                fn = os.path.join(base_data, 'MOT16/%s/%s/img1/%06d.jpg') % (phase, seq, frame)
                image = cv2.imread(fn)
            elif MOT_data_type == 5:
                fn = os.path.join(base_data, 'MOT20/%s/%s/img1/%06d.jpg') % (phase, seq, frame)
                image = cv2.imread(fn)
            elif MOT_data_type == 6:  # HiEve
                ret, image = cap.read()
            else:  # MOT_data_type == 2, 3 or 4
                fn = os.path.join(base_data, 'MOT17/%s/%s/img1/%06d.jpg') % (phase, seq, frame)
                image = cv2.imread(fn)

            image_track = copy.deepcopy(image)

            if detections_type:             # Use MOT challenge public detections
                detections = seq_dets[seq_dets[:, 0] == frame, 2:7]
                if MOT_data_type == 1:
                    if seq == 'MOT16-03' or seq == 'MOT16-04':  # For MOT16-03 and MOT16-04, particularly for DPM
                        # detection
                        detections = detections[np.where(detections[:, 3] < 280.0)]
                if MOT_data_type == 2:
                    if seq == 'MOT17-03-DPM' or seq == 'MOT17-04-DPM':  # For MOT17-03-DPM and MOT17-04-DPM,
                        # particularly for DPM detection
                        detections = detections[np.where(detections[:, 3] < 280.0)]

                detections = constrain_detections_inFrame(image, detections)
                detections[:, 2:4] += detections[:, 0:2]  # convert [x1,y1,w,h] to [x1,y1,x2,y2]

                # Filter detections based on score
                kept, dets = nms_score_mot(detections, overlap_thresh, score_thresh)  # ids of ones to keep after NMS.

                # Extract features from cropped images using observation bounding boxes.
                Z_k_feats = []
                features_all = []
                if feature_extraction_stage_type:
                    dets_cp = copy.deepcopy(dets)[:, 0:5]
                    for i in range(dets_cp.shape[0]):
                        features_all.append(0.0)
                    dets_cp[:, 2:4] -= dets_cp[:, 0:2]  # convert [x1,y1,x2,y2] to [x1,y1,w,h]
                    dets_cp[:, 0:2] += dets_cp[:, 2:4] / 2.0  # convert [x1,y1,w,h] to [xc,yc,w,h]

                    Z_k = copy.deepcopy(dets_cp).T

                    # Combine Z_k and features
                    Z_k_feats.append(Z_k)
                    Z_k_feats.append(features_all)

                else:
                    tlbrs = []
                    crops = []
                    dets_cp = copy.deepcopy(dets)[:, 0:5]
                    dets_cp_Z = copy.deepcopy(dets)[:, 0:5]
                    for i in range(dets_cp.shape[0]):
                        x = int(dets_cp[i, 0])
                        y = int(dets_cp[i, 1])
                        X = int(dets_cp[i, 2])
                        Y = int(dets_cp[i, 3])
                        crop = image[y:Y, x:X, :]

                        # # Extract from each crop
                        # feats = feat_extractor.extract_features_image(crop)
                        # features_all.append(feats)

                        # Or extract from batch of crops. If this fails due to memory issue, try the above one
                        # (extracting from each crop), particularly for MOT20!
                        crops.append(crop)
                    features_all = feat_extractor.extract_features_batch(crops)

                    dets_cp_Z[:, 2:4] -= dets_cp_Z[:, 0:2]  # convert [x1,y1,x2,y2] to [x1,y1,w,h]
                    dets_cp_Z[:, 0:2] += dets_cp_Z[:, 2:4]/2.0  # convert [x1,y1,w,h] to [xc,yc,w,h]

                    Z_k = copy.deepcopy(dets_cp_Z).T

                    # Combine Z_k and features
                    Z_k_feats.append(Z_k)
                    Z_k_feats.append(features_all)

            else:                               # Use detections from custom detectors. e.g. YOLOv8.
                Z_k_feats = []
                features_all = []
                crops = []

                detection_results = detector(image)  # YOLOv8
                detection_results = detection_results[0].boxes.data.cpu().numpy()
                detection_results = detection_results[detection_results[:, 5] == 0]  # 0 - Person class only,
                # [x1, y1, x2, y2, conf, class]

                dets_tm = np.zeros((len(detection_results), 5))

                if feature_extraction_stage_type:
                    for i in range(len(detection_results)):
                        dets_tm[i, :] = detection_results[i][0:5]
                        features_all.append(0.0)

                    dets_tm[:, 2:4] -= dets_tm[:, 0:2]  # convert [x1,y1,x2,y2] to [x1,y1,w,h]
                    dets_tm[:, 0:2] += dets_tm[:, 2:4]/2.0  # convert [x1,y1,w,h] to [xc,yc,w,h]

                    Z_k = copy.deepcopy(dets_tm).T

                    # Combine Z_k and features
                    Z_k_feats.append(Z_k)
                    Z_k_feats.append(features_all)

                else:
                    for i in range(len(detection_results)):
                        dets_tm[i, :] = detection_results[i][0:5]
                        x = int(dets_tm[i, 0])
                        y = int(dets_tm[i, 1])
                        X = int(dets_tm[i, 2])
                        Y = int(dets_tm[i, 3])
                        crop = image[y:Y, x:X, :]

                        # # Extract from each crop
                        # feats = feat_extractor.extract_features_image(crop)
                        # features_all.append(feats)

                        # Or extract from batch of crops. If this fails due to memory issue, try the above one
                        # (extracting from each crop), particularly for MOT20!
                        crops.append(crop)
                    features_all = feat_extractor.extract_features_batch(crops)

                    dets_tm[:, 2:4] -= dets_tm[:, 0:2]  # convert [x1,y1,x2,y2] to [x1,y1,w,h]
                    dets_tm[:, 0:2] += dets_tm[:, 2:4] / 2.0  # convert [x1,y1,w,h] to [xc,yc,w,h]

                    Z_k = copy.deepcopy(dets_tm).T

                    # Combine Z_k and features
                    Z_k_feats.append(Z_k)
                    Z_k_feats.append(features_all)

            # Apply GM-PHD filter
            N_k = Z_k.shape[1]
            # J_max = N_k + M_k_1  # Maximum allowable number of Gaussian components
            J_max = max(M_k_1, N_k, np.random.poisson(M_k_1))  # Maximum allowable number of Gaussian components
            im_height, im_width, C = image.shape
            Filter = GM_PHD_Filter(im_width, im_height, motion_model_type, feature_extraction_stage_type)
            predicted_intensity, model = Filter.predict(Z_k_feats, pruned_intensity)  # model is returned here
            updated_intensity = Filter.update(Z_k_feats, predicted_intensity)
            pruned_intensity, all_comp = Filter.prune_and_merge(updated_intensity, J_max, use_Jmax)
            estimates = Filter.extract_states(pruned_intensity)  # extracting estimates from the pruned intensity this
            # gives better result than extracting them from the updated intensity!

            M_k_1 = len(estimates['w'])

            # # Analysis of number of Gaussian components
            # print('Number of observations:', Z_k.shape[1])
            # print('Number of predicted Gaussian components:', len(predicted_intensity['w']))
            # print('Number of updated Gaussian components:', len(updated_intensity['w']))
            # print('Number of pruned Gaussian components:', len(pruned_intensity['w']))
            # print('Number of extracted states:', len(estimates['w']))

            # Extract features from cropped images using their estimated bounding boxes
            if feature_extraction_stage_type:
                tlbrs = []
                estimates['feat'] = []
                crops = []
                for i in range(len(estimates['m'])):
                    xc = estimates['m'][i][0][0]
                    yc = estimates['m'][i][1][0]
                    if motion_model_type == 'cv':
                        w = estimates['m'][i][4][0]
                        h = estimates['m'][i][5][0]
                    else:
                        w = estimates['m'][i][2][0]
                        h = estimates['m'][i][3][0]
                    x = int(xc - w/2.0)
                    y = int(yc - h/2.0)
                    X = int(x + w)
                    Y = int(y + h)
                    if x < 0:
                        x = 0
                    if y < 0:
                        y = 0
                    if X > im_width:
                        X = im_width
                    if Y > im_height:
                        Y = im_height
                    crop = image[y:Y, x:X, :]

                    # # Extract from each crop
                    # features = feat_extractor.extract_features_image(crop)
                    # estimates['feat'].append(features)

                    # Or extract from batch of crops. If this fails due to memory issue, try the above one
                    # (extracting from each crop), particularly for MOT20!
                    crops.append(crop)
                features_all = feat_extractor.extract_features_batch(crops)

                for f in range(len(estimates['m'])):
                    estimates['feat'].append(features_all[f])

            # Estimate-to-track association
            associations, distance_matrix = compute_associations(image, active_tracks, estimates['m'],
                                                                 estimates['feat'], include_appearance,
                                                                 appearance_weight)
            if associations is None:
                for i in range(len(estimates['m'])):
                    estimates_w_m_P_score_feat = [estimates['w'][i], estimates['m'][i], estimates['P'][i],
                                                  estimates['score'][i], estimates['feat'][i]]
                    active_tracks, global_tracks, archived_tracks = create_new_track(frame, active_tracks,
                                                                                     global_tracks,
                                                                                     archived_tracks,
                                                                                     estimates_w_m_P_score_feat, model,
                                                                                     include_ReId, window_ReId)
            else:
                matched = {}
                track_keys = list(active_tracks)
                for ind in associations:
                    est_indx, track_indx = ind[0], ind[1]
                    track = active_tracks[track_keys[track_indx]]
                    match_score = distance_matrix[est_indx, track_indx]
                    match_score = 1 - 2 * match_score
                    if match_score >= similarity_threshold:
                        match = True
                    else:
                        match = False

                    if match:
                        # Do this correctly for single track
                        active_tracks[track_keys[track_indx]].w_k.append(estimates['w'][est_indx])
                        active_tracks[track_keys[track_indx]].m_k = estimates['m'][est_indx]
                        if motion_model_type == 'cv':
                            active_tracks[track_keys[track_indx]].m_k_store.append(np.array(
                                [estimates['m'][est_indx][0] - estimates['m'][est_indx][4]/2,
                                 estimates['m'][est_indx][1] - estimates['m'][est_indx][5]/2,
                                 estimates['m'][est_indx][0], estimates['m'][est_indx][1] +
                                 estimates['m'][est_indx][5]/2]))
                        else:
                            active_tracks[track_keys[track_indx]].m_k_store.append(np.array(
                                [estimates['m'][est_indx][0] - estimates['m'][est_indx][2]/2,
                                 estimates['m'][est_indx][1] - estimates['m'][est_indx][3]/2,
                                 estimates['m'][est_indx][0], estimates['m'][est_indx][1] +
                                 estimates['m'][est_indx][3]/2]))

                        active_tracks[track_keys[track_indx]].P_k = estimates['P'][est_indx]
                        active_tracks[track_keys[track_indx]].features.append(estimates['feat'][est_indx])
                        active_tracks[track_keys[track_indx]].update_mean_features()  # Keep on running mean of the
                        # features
                        active_tracks[track_keys[track_indx]].scores.append(estimates['score'][est_indx])
                        active_tracks[track_keys[track_indx]].touched = True
                        active_tracks[track_keys[track_indx]].prediction_time = 0
                        active_tracks[track_keys[track_indx]].increment_true_length()
                        active_tracks[track_keys[track_indx]].predict_status = 0
                        # active_tracks[track_keys[track_indx]].reid_status = 0
                        active_tracks[track_keys[track_indx]].end_frame = frame
                        global_tracks[track_keys[track_indx]] = active_tracks[track_keys[track_indx]]  # Check this!
                    else:
                        # Create a new track
                        estimates_w_m_P_score_feat = [estimates['w'][est_indx], estimates['m'][est_indx],
                                                      estimates['P'][est_indx], estimates['score'][est_indx],
                                                      estimates['feat'][est_indx]]
                        active_tracks, global_tracks, archived_tracks = create_new_track(frame, active_tracks,
                                                                                         global_tracks, archived_tracks,
                                                                                         estimates_w_m_P_score_feat,
                                                                                         model, include_ReId,
                                                                                         window_ReId)
                    matched[est_indx] = 1

                # Next deal with any estimated states not associated with tracks
                if distance_matrix.shape[0] > len(associations):
                    for indx in range(distance_matrix.shape[0]):
                        if indx not in matched:
                            estimates_w_m_P_score_feat = [estimates['w'][indx], estimates['m'][indx],
                                                          estimates['P'][indx], estimates['score'][indx],
                                                          estimates['feat'][indx]]
                            # Create a new track
                            active_tracks, global_tracks, archived_tracks = create_new_track(frame, active_tracks,
                                                                                             global_tracks,
                                                                                             archived_tracks,
                                                                                             estimates_w_m_P_score_feat,
                                                                                             model, include_ReId,
                                                                                             window_ReId)

                # Next deal with any tracks not associated with current estimated states. i.e. - Just predict till
                # prediction <= prediction_time_threshold; after that, kill the tracks.
                if distance_matrix.shape[1] > len(associations):

                    # Predict the tracks for prediction_time_threshold times before you kill the tracks.
                    if is_AddOn_prediction:
                        for track_index in range(len(track_keys)):
                            key = track_keys[track_index]
                            track = active_tracks[key]
                            if not track.touched:  # and track.prediction_time < prediction_time_threshold
                                track_pred = addOn_prediction(track)  # Predict the track using the motion model.
                                track_pred.predict_status = 1
                                track_pred.end_frame = frame
                                # track_pred.touched = True     # Check this!
                                track_pred.prediction_time += 1
                                active_tracks[key] = track_pred
                                global_tracks[key] = track_pred

                            # Kill the track
                            if active_tracks[key].prediction_time > track_kill_time_threshold:  # prediction_time_threshold:
                                archived_tracks[key] = active_tracks[key]
                                del active_tracks[key]

                    else:   # Kill the track
                        for track_index in range(len(track_keys)):
                            key = track_keys[track_index]
                            track = active_tracks[key]
                            if not track.touched:
                                archived_tracks[key] = active_tracks[key]
                                del active_tracks[key]

                    # Kill the tracks out of the scene
                    del_keys = []
                    for ky in active_tracks.keys():
                        track_ky = active_tracks[ky]
                        xc = track_ky.m_k[0][0]
                        yc = track_ky.m_k[1][0]
                        if motion_model_type == 'cv':
                            w = track_ky.m_k[4][0]
                            h = track_ky.m_k[5][0]
                        else:
                            w = track_ky.m_k[2][0]
                            h = track_ky.m_k[3][0]
                        x1 = int(xc - w / 2.0)
                        y1 = int(yc - h / 2.0)
                        x2 = int(x1 + w)
                        y2 = int(y1 + h)

                        if x1 > im_width or y1 > im_height or x2 < 0 or y2 < 0:  # If the track bounding box is out of
                            # the scene.
                            archived_tracks[ky] = active_tracks[ky]
                            del_keys.append(ky)  # Store keys of active tracks to be deleted.
                    for k in del_keys:
                        del active_tracks[k]

            # For displaying detections
            if display_detections:
                for i in range(dets.shape[0]):
                    x1 = int(dets[i, 0])
                    y1 = int(dets[i, 1])
                    x2 = int(dets[i, 2])
                    y2 = int(dets[i, 3])
                    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

                cv2.imshow('Detections', image)

            # For tracking
            for trackId in active_tracks.keys():
                if is_AddOn_prediction:
                    if active_tracks[trackId].true_length < track_min_length or \
                            active_tracks[trackId].prediction_time > prediction_time_threshold:
                        continue
                isPredicted = active_tracks[trackId].predict_status  # 0 for False and 1 for True
                isReId = active_tracks[trackId].reid_status  # 0 for False and 1 for True
                xc = active_tracks[trackId].m_k[0][0]
                yc = active_tracks[trackId].m_k[1][0]
                if motion_model_type == 'cv':
                    w = active_tracks[trackId].m_k[4][0]
                    h = active_tracks[trackId].m_k[5][0]
                    vcx = active_tracks[trackId].m_k[2][0]
                    vcy = active_tracks[trackId].m_k[3][0]
                else:
                    w = active_tracks[trackId].m_k[2][0]
                    h = active_tracks[trackId].m_k[3][0]
                    vcx = 0.0
                    vcy = 0.0
                x1 = int(xc - w/2.0)
                y1 = int(yc - h/2.0)
                x2 = int(x1 + w)
                y2 = int(y1 + h)
                conf = active_tracks[trackId].scores[-1]  # Detection confidence, take the last one!

                col = active_tracks[trackId].colour
                cv2.rectangle(image_track, (x1, y1), (x2, y2), (int(col[0]), int(col[1]), int(col[2])), 6)

                # Display trajectories
                if display_trajectories:
                    if len(active_tracks[trackId].m_k_store) <= 20:
                        for t in range(len(active_tracks[trackId].m_k_store)):
                            cv2.circle(image_track, (int(active_tracks[trackId].m_k_store[-t][2][0]),
                                                     int(active_tracks[trackId].m_k_store[-t][3][0])), 6,
                                       (int(col[0]), int(col[1]), int(col[2])), -1)
                    else:
                        for t in range(20):  # Limit to 20 last frames to display trajectory
                            if t != 0:
                                cv2.circle(image_track, (int(active_tracks[trackId].m_k_store[-t][2][0]),
                                                         int(active_tracks[trackId].m_k_store[-t][3][0])), 6,
                                           (int(col[0]), int(col[1]), int(col[2])), -1)

                # Save the tracking results
                if MOT_data_type == 6:  # For HiEve Challenge submission, it starts from frame 0!
                    output_file.write("{},{},{},{},{},{},{},-1,-1,-1\n".format(frame-1, trackId, x1, y1, x2 - x1,
                                                                               y2 - y1, conf))
                else:    # For MOT Challenge submission
                    output_file.write("{},{},{},{},{},{},-1,-1,-1,-1\n".format(frame, trackId, x1, y1, x2 - x1,
                                                                               y2 - y1))

            # Display tracks
            if display_tracks:
                cv2.imshow('Tracking', image_track)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    cv2.destroyAllWindows()
                    break

        output_file.close()
        print("ID limit: {}".format(len(list(global_tracks))))
        end_time = timer()
        full_time = end_time - start_time
        fps = frame / full_time
        print('fps={0}'.format(fps))
