"""
This code evaluates a Multi-Object Tracking (MOT) performance.
"""
import argparse
from operator import attrgetter
from os import path
import yaml

import motmetrics as mm
import motmetrics.distances as mm_d
import numpy as np


class DetectionWithID:
    def __init__(self, p_id, frame_num, left, top, right, bottom, score=None):
        self.p_id = p_id
        self.frame_num = int(frame_num)
        self.left = int(left)
        self.top = int(top)
        self.right = int(right)
        self.bottom = int(bottom)
        self.score = score

    def __repr__(self):
        return "{},{},{},{},{},{},{}".format(self.p_id, self.frame_num, self.left, self.top, self.right, self.bottom,
                                             self.score)


class Frame:
    def __init__(self, num):
        self.num = num
        self.dets = []

    def __repr__(self):
        return "{}, dets: {}".format(self.num, len(self.dets))


class detections_to_frames:

    def __init__(self, dets):

        self.dets = dets
        self.frames = []
        self.factor = 0
        self.sort_frames = sorted(self.dets, key=attrgetter('frame_num'))
        self.it_frames = self.factor - 1  # Need this or the frames[-1] does not append!

    def setting_up_frame(self):

        for det in self.sort_frames:
            self.determine_frames(det)

        return self.frames

    def determine_frames(self, det):

        """
        det: detections in each frame. This will be checked to compare between the detection frame number and it_frames
        which is used to describe frames. If there are no detections, append all frames between it_frame and det.frame
        for no detections. After this, check this detection and append the detection, and it continues on.
        """

        if det.frame_num == self.it_frames:
            self.frames[-1].dets.append(det)
        elif det.frame_num == self.it_frames + 1:
            self.it_frames += 1
            self.frames.append(Frame(self.it_frames))
            self.frames[-1].dets.append(det)
        elif det.frame_num > self.it_frames + 1:
            diff_frame = det.frame_num - self.it_frames
            for i in range(diff_frame):
                self.frames.append(Frame(self.it_frames))
                self.it_frames += 1
            self.determine_frames(det)


def detections_from_ground_truth(f_name, data_type):
    """
    f_name: name of ground truth
    data_type: type of tracking dataset
    """
    detections = []
    with open(f_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(',')
            # class person, certainty 1, visibility >= 0.25 (visibility threshold default = 0.0)
            if int(data[7]) == 1 and int(data[6]) == 1 and float(data[8]) >= 0.0 and data_type != 'HiEVe':
                p_id = data[1]
                frame_num = data[0]
                left = float(data[2])
                top = float(data[3])
                right = float(data[4]) + left
                bottom = float(data[5]) + top
                detections.append(DetectionWithID(p_id, frame_num, left, top, right, bottom, data[6]))
            else:  # For HiEve
                p_id = data[1]
                frame_num = data[0]
                left = float(data[2])
                top = float(data[3])
                right = float(data[4]) + left
                bottom = float(data[5]) + top
                detections.append(DetectionWithID(p_id, frame_num, left, top, right, bottom, data[6]))

        return detections


def detections_from_output_tracks(f_name):
    """
    f_name is name of tracks output.
    """
    detections = []
    with open(f_name, 'r') as f:
        lines = f.readlines()
        for line in lines:
            data = line.split(',')
            p_id = data[1]
            frame_num = data[0]
            left = float(data[2])
            top = float(data[3])
            right = float(data[4]) + left
            bottom = float(data[5]) + top
            detections.append(DetectionWithID(p_id, frame_num, left, top, right, bottom, data[6]))

        return detections


# Distance matrix based on full box (4 corners)
def get_distance_matrix_box(gt_frame, track_frame):
    track_matrix = np.zeros((len(track_frame.dets), 4))  # 4 is x, y, w, h
    gt_matrix = np.zeros((len(gt_frame.dets), 4))

    for i, det in enumerate(track_frame.dets):
        track_matrix[i][0] = det.left
        track_matrix[i][1] = det.top
        track_matrix[i][2] = det.right - det.left
        track_matrix[i][3] = det.bottom - det.top

    for i, det in enumerate(gt_frame.dets):
        gt_matrix[i][0] = det.left
        gt_matrix[i][1] = det.top
        gt_matrix[i][2] = det.right - det.left
        gt_matrix[i][3] = det.bottom - det.top

    dist = mm_d.iou_matrix(gt_matrix, track_matrix, max_iou=0.5)  # default is 0.5

    return dist


# Calculate the MOT metrics
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Tracking pipeline for GM-PHD-Tracker.')
    parser.add_argument('--base_data', type=str, default='./data',
                        help="Path to base tracking data folder.")
    parser.add_argument('--base_result', type=str, default='./result',
                        help='Path to base tracking result folder to be saved to.')

    args = parser.parse_args()

    # Load config. Look at the comments in the config.yaml for more information about each variable or parameter.
    with open('config.yaml', 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    MOT_data_type = config['tracking']['MOT_data_type']  # We can only evaluate train sequence of each data type.
    base_data = args.base_data
    base_result = args.base_result

    if MOT_data_type == 1:
        data_type = 'MOT16'
        phase = 'train'
        sequences = ['MOT16-02', 'MOT16-04', 'MOT16-05', 'MOT16-09', 'MOT16-10', 'MOT16-11', 'MOT16-13']
    elif MOT_data_type == 2:
        data_type = 'MOT17'
        phase = 'train'
        sequences = ['MOT17-02-DPM', 'MOT17-04-DPM', 'MOT17-05-DPM', 'MOT17-09-DPM', 'MOT17-10-DPM', 'MOT17-11-DPM',
                     'MOT17-13-DPM']
    elif MOT_data_type == 3:
        data_type = 'MOT17'
        phase = 'train'
        sequences = ['MOT17-02-FRCNN', 'MOT17-04-FRCNN', 'MOT17-05-FRCNN', 'MOT17-09-FRCNN', 'MOT17-10-FRCNN',
                     'MOT17-11-FRCNN', 'MOT17-13-FRCNN']
    elif MOT_data_type == 4:
        data_type = 'MOT17'
        phase = 'train'
        sequences = ['MOT17-02-SDP', 'MOT17-04-SDP', 'MOT17-05-SDP', 'MOT17-09-SDP', 'MOT17-10-SDP', 'MOT17-11-SDP',
                     'MOT17-13-SDP']
    elif MOT_data_type == 5:
        data_type = 'MOT20'
        phase = 'train'
        sequences = ['MOT20-01', 'MOT20-02', 'MOT20-03', 'MOT20-05']
    elif MOT_data_type == 6:
        data_type = 'HiEve'
        phase = 'train'
        sequences = ['1.mp4', '2.mp4', '3.mp4', '4.mp4', '5.mp4', '6.MP4', '7.mp4', '8.mp4', '9.mp4', '10.MOV',
                     '11.mp4', '12.mp4', '13.mp4', '14.mp4', '15.mp4', '16.mp4', '17.mp4', '18.MOV', '19.mp4']
    else:
        print('\nError: Set to correct MOT dataset: Set to 1 for MOT16, to 2 for MOT17-DPM, to 3 for MOT17-FRCNN, to 4 '
              'for MOT17-SDP, to 5 for MOT20 or to 6 for HiEve.')
        exit()

    fn_use_gt = detections_from_ground_truth
    fn_use_output_tracks = detections_from_output_tracks

    mot_accum = []

    for sequence_name in sequences:
        tracks_file = path.join(base_result + '/' + data_type, sequence_name + '.txt')   # tracks output file

        if path.isfile(tracks_file):
            if MOT_data_type == 6:  # HiEve
                gt_file = path.join(base_data + '/' + data_type + '/HIE20/labels/track1',
                                    sequence_name.split('.')[0] + '.txt')  # ground-truth file
            else:
                gt_file = path.join(base_data + '/' + data_type + '/' + phase,
                                    sequence_name + '/gt/gt.txt')  # ground-truth file

            gt_dets = fn_use_gt(gt_file, data_type)
            output_track_dets = fn_use_output_tracks(tracks_file)

            gt_frames_1 = detections_to_frames(gt_dets)
            gt_frames = gt_frames_1.setting_up_frame()
            track_frames_1 = detections_to_frames(output_track_dets)
            track_frames = track_frames_1.setting_up_frame()

            acc = mm.MOTAccumulator(auto_id=True)

            if len(track_frames) != len(gt_frames):
                print("WARNING: gt frame count does not match track frame count")
            print('gt frames {0}, track frames {1}'.format(len(gt_frames), len(track_frames)))
            frame_count = min(len(track_frames), len(gt_frames))

            for i in range(frame_count):
                gt_det_ids = [det.p_id for det in gt_frames[i].dets]  # gt_ids
                track_det_ids = [det.p_id for det in track_frames[i].dets]  # track_ids
                dists = get_distance_matrix_box(gt_frames[i], track_frames[i])
                acc.update(gt_det_ids, track_det_ids, dists)

            mot_accum.append(acc)

    mh = mm.metrics.create()

    summary = mh.compute_many(
        mot_accum,
        metrics=mm.metrics.motchallenge_metrics,
        names=sequences,
        generate_overall=True, )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names, )
    print(str_summary)
