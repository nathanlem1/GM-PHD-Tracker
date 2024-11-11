import numpy as np


class Track:
    def __init__(self, w_k, m_k, P_k, score, features, F_k, Q_k, colour, track_id, start_frame, motion_model_type):
        self.w_k = [w_k]
        self.m_k = m_k
        self.P_k = P_k
        self.scores = [score]
        self.features = [features]
        self.mean_features = features
        self.F_k = F_k
        self.Q_k = Q_k
        self.colour = colour
        self.uniqueid = track_id
        self.start_frame = start_frame
        self.end_frame = start_frame
        self.true_length = 0
        self.prediction_time = 0
        self.touched = True
        self.predict_status = 0   # Set to 0 for no add-on prediction or to 1 for add-on prediction.
        self.reid_status = 0  # Set to 0 if no ReId happens or to 1 if ReId happens.
        self.motion_model_type = motion_model_type
        if self.motion_model_type == 'cv':
            self.m_k_store = [np.array([m_k[0] - m_k[4] / 2, m_k[1] - m_k[5] / 2, m_k[0], m_k[1] + m_k[5] / 2])]
        else:  # 'rw'
            self.m_k_store = [np.array([m_k[0] - m_k[2] / 2, m_k[1] - m_k[3] / 2, m_k[0], m_k[1] + m_k[3] / 2])]

    def increment_true_length(self):
        self.true_length += 1

    def update_mean_features(self):

        # # All features have equal weight for averaging
        # self.mean_features = np.mean(self.features, axis=0)

        # # Features are weighted by detection scores
        # all_feats_scoresWeighted = []
        # for i in range(len(self.scores)):
        #     all_feats_scoresWeighted.append(self.scores[i]*self.features[i])
        # self.mean_features = np.mean(all_feats_scoresWeighted, axis=0)

        # # Features are weighted by weights of Gaussian components (of extracted states)
        # all_feats_weightsWeighted = []
        # for i in range(len(self.w_k)):
        #     all_feats_weightsWeighted.append(self.w_k[i]*self.features[i])
        # self.mean_features = np.mean(all_feats_weightsWeighted,axis=0)

        # Exponential Moving Average (EMA)
        eta = 0.9
        self.mean_features = eta * self.mean_features + (1.0 - eta) * self.features[-1]  # This exponential moving
        # average improves the data association!
