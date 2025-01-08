"""
GM-PHD-Filter implementation for multi-target visual tracking. For understanding GH-PHD Filter using its simulation, you
can look at: https://github.com/nathanlem1/MTF-Lib/tree/master/GM-PHD-Filter/GM-PHD-Filter-Python
"""

import copy
import numpy as np
from scipy.stats import multivariate_normal
import yaml

from tracking_utils import appearance_likelihood

# Load config
with open('config.yaml', 'r') as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)

motion_model_type = config['tracking']['motion_model_type']


def set_model(im_width, im_height, motion_model_type):
    """
    Set motion model i.e. based on either constant velocity (cv) or random walk (rw).
    """
    model = dict()

    if motion_model_type == 'cv':  # Constant velocity (CV) model

        # Dynamic model parameters
        model['F_k'] = np.eye(6)  # state transition model
        T = 1.0
        I = T*np.eye(2, dtype=np.float64)
        model['F_k'][0:2, 2:4] = I

        sigma_v = 5
        Q1 = np.array([[T ** 4 / 4, T ** 3 / 2], [T ** 3 / 2, T ** 2]])
        Q = np.zeros((6, 6))
        Q[np.ix_([0, 2], [0, 2])] = Q1
        Q[np.ix_([1, 3], [1, 3])] = Q1
        Q[4, 4] = 1.0
        Q[5, 5] = 1.0
        model['Q_k'] = sigma_v ** 2 * Q   # covariance of process noise

        # Observation model parameters
        model['H_k'] = np.array(
            [[1., 0, 0, 0, 0, 0], [0, 1., 0, 0, 0, 0], [0, 0, 0, 0, 1., 0], [0, 0, 0, 0, 0, 1.]])  # observation model
        sigma_r = 6
        model['R_k'] = sigma_r ** 2 * np.eye(4)  # the covariance of observation noise (can it change with the size of
        # detection?)

        # Initial state covariance
        model['P_k'] = np.diag([200, 200, 100, 100, 50, 50])

    elif motion_model_type == 'rw':  # Random walk(RW) model

        # Motion model parameters
        model['F_k'] = np.eye(4)  # state transition matrix (for the centroid and size)

        # Initialize the process covariance
        sigma_v = 12
        model['Q_k'] = sigma_v ** 2 * np.eye(4)

        # Initialize the state covariance - This measures the uncertainty in the Kalman predictions
        model['P_k'] = np.diag([400, 400, 100, 100])

        # Measurement model parameters
        model['H_k'] = np.eye(4)  # Measurement matrix for xc,yc,w, and h of the bounding boxes

        # Initialize the measurement covariance
        sigma_r = 1.5
        model['R_k'] = sigma_r ** 2 * np.eye(4)

    else:
        print('\nError: please set the motion model type to be used: 0 for CV or 1 for RW.')
        exit()

    # Other important parameters
    model['w_birthsum'] = 0.1  # 0.0001 #0.02 # The total weight of birth targets. It is chosen depending on handling
    # false positives.

    model['p_D'] = 0.95  # Probability of target detection,
    model['p_S'] = 0.99  # Probability of target survival (prob_death = 1 - prob_survival)
    model['T'] = 10**-5  # Pruning weight threshold.
    model['U'] = 4  # Merge distance threshold.
    model['w_thresh'] = 0.5  # State extraction weight threshold

    # Compute clutter intensity
    lambda_t = np.random.poisson(10)  # Poisson average rate of uniform clutter (per scan); lambda_t = lambda_c*A
    A = im_width*im_height
    pdf_c = 1.0/A
    clutter_intensity = lambda_t * pdf_c   # Generate clutter intensity.
    model['clutterIntensity'] = clutter_intensity

    return model


# The probability density function (pdf) of the d-dimensional multivariate normal distribution
def mvnpdf(x, mean, covariance):

    # x = np.array(x, dtype=np.float64)
    # mean = np.array(mean, dtype=np.float64)
    # covariance = np.array(covariance, dtype=np.float64)

    d = mean.shape[0]
    delta_m = x - mean
    pdf_res = 1.0/(np.sqrt((2*np.pi)**d * np.linalg.det(covariance))) * \
              np.exp(-0.5 * np.transpose(delta_m).dot(np.linalg.inv(covariance)).dot(delta_m))[0][0]
    # pdf_res = 1.0 / (np.sqrt((2 * np.pi) ** d * np.linalg.det(covariance))) *
    # math.exp(-0.5 * np.transpose(delta_m).dot(np.linalg.inv(covariance)).dot(delta_m))

    return pdf_res


class GM_PHD_Filter:
    def __init__(self, im_width, im_height, motion_model_type, feature_extraction_estimates, include_appearance):
        self.motion_model_type = motion_model_type
        self.model = set_model(im_width, im_height, motion_model_type)
        self.feature_extraction_estimates = feature_extraction_estimates
        self.include_appearance = include_appearance

    # Step 1 and 2: Birth new targets and predict existing targets (Gaussian components) (According to the original
    # paper)
    def predict(self, Z_k_feat, pruned_intensity):

        Z_k = Z_k_feat[0]
        features = Z_k_feat[1]

        # An intensity (Probability Hypothesis Density - PHD) is described using weight, mean and covariance
        w = []  # weight of a Gaussian component
        m = []  # mean of a Gaussian component
        P = []  # Covariance of a Gausssian component
        score = []
        feat = []
        v_init = [0.0, 0.0]  # initial velocity (Pixel per frame!)
        w_birthsum = self.model['w_birthsum']

        # Birth new targets
        for i in range(Z_k.shape[1]):
            z_k = copy.deepcopy(Z_k[:, i]).reshape(-1, 1)
            w.append(w_birthsum)  # (w_birthsum / len(Z_k)
            if self.motion_model_type == 'cv':
                m.append(np.array([z_k[0][0], z_k[1][0], v_init[0], v_init[1], z_k[2][0],
                                   z_k[3][0]]).reshape(-1, 1).astype('float64'))  # Targets are born here with [x, y,
                # vx, vy, width, height] state format
            else:
                m.append(np.array([z_k[0], z_k[1], z_k[2], z_k[3]]).reshape(-1, 1).astype('float64'))  # Targets are
                # born here with [x, y, vx, vy, width, height] state format

            P.append(self.model['P_k'].astype('float64'))
            score.append(z_k[4][0].astype('float64'))
            feat.append(features[i])

        # Predict existing targets
        num_targets_Jk_minus_1 = len(pruned_intensity['w'])  # Number of Gaussian components after the pruning and
        # merging step

        for i in range(num_targets_Jk_minus_1):
            w.append(self.model['p_S'] * pruned_intensity['w'][i])
            m.append(self.model['F_k'].dot(pruned_intensity['m'][i]).astype('float64'))
            P.append(self.model['Q_k'] +
                     self.model['F_k'].dot(pruned_intensity['P'][i]).dot(np.transpose(self.model['F_k'])))
            score.append(pruned_intensity['score'][i])
            feat.append(pruned_intensity['feat'][i])

        predicted_intensity = dict()
        predicted_intensity['w'] = w
        predicted_intensity['m'] = m
        predicted_intensity['P'] = P
        predicted_intensity['score'] = score
        predicted_intensity['feat'] = feat

        return predicted_intensity, self.model

    # Step 3 and 4: Construct PHD update components and doing observation update (According to the orignal paper)
    def update(self, Z_k_feat, predicted_intensity):

        Z_k = Z_k_feat[0]
        features = Z_k_feat[1]

        # Construct PHD update components
        eta = []
        S = []
        K = []
        P = []
        for i in range(len(predicted_intensity['w'])):
            eta.append(self.model['H_k'].dot(predicted_intensity['m'][i]).astype('float64'))
            S.append(self.model['R_k'] + self.model['H_k'].dot(predicted_intensity['P'][i]).dot(np.transpose(
                self.model['H_k'])).astype('float64'))
            Si = copy.deepcopy(S[i])
            invSi = np.linalg.inv(np.array(Si, dtype=np.float64))  # Using normal inverse function
            # Vs = np.linalg.cholesky(np.array(Si, dtype=np.float64)); inv_sqrt_S = np.linalg.inv(Vs);
            # invSi = inv_sqrt_S.dot(np.transpose(inv_sqrt_S))  # Using Cholesky method
            K.append(predicted_intensity['P'][i].dot(np.transpose(self.model['H_k'])).dot(invSi).astype('float64'))
            P.append(predicted_intensity['P'][i] -
                     K[i].dot(self.model['H_k']).dot(predicted_intensity['P'][i]).astype('float64'))

        construct_update_intensity = dict()
        construct_update_intensity['eta'] = eta
        construct_update_intensity['S'] = S
        construct_update_intensity['K'] = K
        construct_update_intensity['P'] = P

        # Miss-detection part of GM-PHD update
        # We scale all weights by probability of missed detection (1 - p_D)
        w = []
        m = []
        P = []
        score = []
        feat = []
        for i in range(len(predicted_intensity['w'])):
            w.append((1.0 - self.model['p_D']) * predicted_intensity['w'][i])
            m.append(predicted_intensity['m'][i])
            P.append(predicted_intensity['P'][i])
            score.append(predicted_intensity['score'][i])
            feat.append(predicted_intensity['feat'][i])

        # Detection part of GM-PHD update
        # Every observation updates every target
        num_targets_Jk_k_minus_1 = len(predicted_intensity['w'])  # Number of Gaussian components after the prediction
        # step
        l = 0
        for z in range(Z_k.shape[1]):
            l = l + 1
            for j in range(num_targets_Jk_k_minus_1):
                z_k = copy.deepcopy(Z_k[:, z]).reshape(-1, 1)
                if not self.feature_extraction_estimates and self.include_appearance:
                    app_lik = appearance_likelihood(features[z], predicted_intensity['feat'][j])  # Appearance
                    # likelihood
                else:
                    app_lik = 1.0  # No appearance likelihood i.e. if features are to be extracted from estimates!
                w.append(app_lik * self.model['p_D'] * predicted_intensity['w'][j] *
                         mvnpdf(z_k[0:2], construct_update_intensity['eta'][j][0:2],
                                construct_update_intensity['S'][j][0:2, 0:2]))  # Hoping multivariate_normal.pdf is the
                # right one to use. Use only the [x,y] i.e. not use width and height for weight computation as it
                # doesn't give stable results!
                m.append(predicted_intensity['m'][j] +
                         construct_update_intensity['K'][j].dot(z_k[0:4] -
                                                                construct_update_intensity['eta'][j]).astype('float64'))
                P.append(construct_update_intensity['P'][j])
                score.append(z_k[4][0])
                feat.append(features[z])

            total_w_d = 0.0
            for j in range(num_targets_Jk_k_minus_1):
                total_w_d = total_w_d + w[l * num_targets_Jk_k_minus_1 + j]

            for j in range(num_targets_Jk_k_minus_1):
                k_k = self.model['clutterIntensity']
                w[l * num_targets_Jk_k_minus_1 + j] = w[l * num_targets_Jk_k_minus_1 + j] / (k_k + total_w_d)  # Updated
                # weight

        # Combine both miss-detection and detection part of the GM-PHD update
        updated_intensity = dict()
        updated_intensity['w'] = w
        updated_intensity['m'] = m
        updated_intensity['P'] = P
        updated_intensity['score'] = score
        updated_intensity['feat'] = feat

        return updated_intensity

    # Step 5: Pruning and Merging
    def prune_and_merge(self, updated_intensity, J_max, use_Jmax):
        w = []
        m = []
        P = []
        score = []
        feat = []
        all_comp = []

        # Prune out the low-weighted components
        I = [index for index, value in enumerate(updated_intensity['w']) if value > self.model['T']]  # Indices of large
        # enough weights
        # I = np.where(np.array(updated_intensity['w']) >= self.model['T'])[0]

        # Merge the close-together components
        while len(I) > 0:
            high_weights = np.array(updated_intensity['w'], dtype=object)[I]
            j = np.argmax(high_weights)
            j = I[j]
            # Find all points with Mahalanobis distance less than U from point updated_intensity['m'][j]
            L = []  # A vector of indices of merged Gaussians.
            for iterI in range(len(I)):
                thisI = copy.deepcopy(I[iterI])
                delta_m = updated_intensity['m'][thisI] - updated_intensity['m'][j]
                mahal_dist = np.transpose(delta_m).dot(np.linalg.inv(np.array(updated_intensity['P'][thisI],
                                                                              dtype=np.float64))).dot(delta_m)
                if mahal_dist <= self.model['U']:
                    L.append(thisI)  # Indices of merged Gaussians

            # The new weight of the resulted merged Guassian is the summation of the weights of the Gaussian components.
            w_bar = sum(np.array(updated_intensity['w'], dtype=object)[L])
            w.append(w_bar)

            # The new mean of the merged Gaussian is the weighted average of the merged means of Gaussian components.
            m_val = []
            score_val = []
            feat_val = []
            for i in range(len(L)):
                thisI = copy.deepcopy(L[i])
                m_val.append(updated_intensity['w'][thisI] * updated_intensity['m'][thisI])
                score_val.append(updated_intensity['w'][thisI] * updated_intensity['score'][thisI])
                feat_val.append(updated_intensity['w'][thisI] * updated_intensity['feat'][thisI])
            m_bar = sum(m_val) / w_bar
            m.append(m_bar.astype('float64'))
            score_bar = sum(score_val) / w_bar
            score.append(score_bar.astype('float64'))
            feat_bar = sum(feat_val) / w_bar
            feat.append(feat_bar.astype('float64'))

            # Calculating covariance P_bar is a bit trickier
            P_val = []
            for i in range(len(L)):
                thisI = copy.deepcopy(L[i])
                delta_m = m_bar - updated_intensity['m'][thisI]
                P_val.append(updated_intensity['w'][thisI] * (updated_intensity['P'][thisI] +
                                                              delta_m.dot(np.transpose(delta_m))))
            P_bar = sum(P_val) / w_bar
            P.append(P_bar.astype('float64'))

            # Combine all parts together
            all_comp.append((w_bar, m_bar.astype('float64'), P_bar.astype('float64'), score_bar.astype('float64'),
                             feat_bar.astype('float64')))

            # Now delete the elements in L from I
            for i in L:
                I.remove(i)

        # Limit the number of Gaussian components to the maximum allowable number of Gaussian components (J_max)
        if use_Jmax:
            if len(w) > J_max:
                sortedByWeight = sorted(all_comp, key=lambda m: m[0], reverse=True)
                sortedByWeight_Jmax = sortedByWeight[0:J_max]
                w, m, P, score, feat = [], [], [], [], []
                for c in range(len(sortedByWeight_Jmax)):
                    w.append(sortedByWeight_Jmax[c][0])
                    m.append(sortedByWeight_Jmax[c][1])
                    P.append(sortedByWeight_Jmax[c][2])
                    score.append(sortedByWeight_Jmax[c][3])
                    feat.append(sortedByWeight_Jmax[c][4])

                pruned_merged_intensity = dict()
                pruned_merged_intensity['w'] = w
                pruned_merged_intensity['m'] = m
                pruned_merged_intensity['P'] = P
                pruned_merged_intensity['score'] = score
                pruned_merged_intensity['feat'] = feat
            else:
                pruned_merged_intensity = dict()
                pruned_merged_intensity['w'] = w
                pruned_merged_intensity['m'] = m
                pruned_merged_intensity['P'] = P
                pruned_merged_intensity['score'] = score
                pruned_merged_intensity['feat'] = feat
        else:
            pruned_merged_intensity = dict()
            pruned_merged_intensity['w'] = w
            pruned_merged_intensity['m'] = m
            pruned_merged_intensity['P'] = P
            pruned_merged_intensity['score'] = score
            pruned_merged_intensity['feat'] = feat

        return pruned_merged_intensity, all_comp

    # Step 6: extracting estimated states
    def extract_states(self, pruned_and_merged):
        w = []
        m = []
        P = []
        score = []
        feat = []
        # pruned_and_merged = self.prune_and_merge()
        for i in range(len(pruned_and_merged['w'])):
            if pruned_and_merged['w'][i] > self.model['w_thresh']:
                # for j in range(int(round(pruned_and_merged['w'][i]))):  # If a target has a rounded weight greater
                # than 1, output it multiple times. Is this necessary for visual tracking?
                    w.append(pruned_and_merged['w'][i])
                    m.append(pruned_and_merged['m'][i])
                    P.append(pruned_and_merged['P'][i])
                    score.append(pruned_and_merged['score'][i])
                    feat.append((pruned_and_merged['feat'][i]))

        extracted_states = dict()
        extracted_states['w'] = w
        extracted_states['m'] = m
        extracted_states['P'] = P
        extracted_states['score'] = score
        extracted_states['feat'] = feat

        return extracted_states
