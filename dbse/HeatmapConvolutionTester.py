#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import warnings
from copy import deepcopy
warnings.simplefilter('ignore')
from dbse.ClusterTraining import ClusterTraining
from dbse.HeatmapConvolutionTrainer import HeatmapConvolutionTrainer
import traceback
from bisect import bisect_left
from dbse.tools.Visual import *
import math
import numpy
from scipy.signal.signaltools import medfilt
import copy


class HeatmapConvolutionTester(object):
    '''
    This class contains the testing phase of the prediction model.
    For each Object the RUL is estimated based on the trained model and a new field predicted_rul is added
    The model works as follows:
        1. load test sequence
        2. assign a cluster to the current object (scale, pca then cluster predictor)
        3. remove empty or constant features
        4. then per object and per feature and per time element of a feature - take y value of this feature and cut the heat map
           horizontally to get a heat curve, for delayed time elements do the same but then shift by the time difference to the current
           element
           Then sum up all curves per feature and per time element of each feature
        5. Iterate over object ids and there over all RULs (i.e. first use 1 then 2 then 3 RULs etc.)
           add the result
    '''
    PROCESS_INSTANCE= 0

    def __init__(self,
                 visualize_summed_curve = True,
                 visualize_per_feature_curve = True,
                 visualize_pre_post_risk = False,
                 enable_all_print = True,
                 write_csv = False,
                 csv_path = r"tested_result.csv",
                 test_mode = False,
                 optimize_via_average = False,
                 whole_data_set = False,
                 field_in_test_t = "test_t",
                field_in_test_crit = "test_crit",
                field_in_test_X = "test_X",
                field_in_test_id = "test_id",
                field_in_test_rul = "test_rul",
                field_in_train_cluster_stored = "train_cluster_stored",
                field_in_meta_dataset_name = "meta_dataset_name",
                field_in_train_model_grid_area = "train_model_grid_area",
                field_in_train_model_trained_scalers = "train_model_trained_scalers",
                field_in_train_model = "train_model",
                field_in_train_risc = "train_risc",
                field_in_train_rul = "train_rul",
                field_out_predicted_rul = "predicted_rul",
                smooth_per_feature = False,
                smoothing_side = 81,
                percentage_side_fine = 0.1,
                feature_selection_on = True):
        """
        Constructor
        :param whole_data_set: if true the evaluation is performed not only on critical parts but on the whole dataset
        :param optimize_via_average: this optimization finds the average of all feature favorites, then removes the ones most distant (i.e. outlier removal) and keeps the rest
        :param visualize_summed_curve: visualize sum of all heat curves
        :param visualize_per_feature_curve: visualize sum of heats per feature
        :param visualize_pre_post_risk: visualize heat prior to and after shift
        :param write_csv: if true then write directly to CSV per result retrieved
        :param field_in_test_t: time stamp of the test data
        :param field_in_test_crit: 1 if row is in critical area and 0 otherwise
        :param field_in_test_X: Features of the training data
        :param field_in_test_id: object identifier of test object
        :param field_in_test_rul: RUL of test data - used for evaluation purpose only!
        :param field_in_train_cluster_stored: cluster identifier assigned to each object (numeric from 0 to numberOfClusters-1)
        :param field_in_meta_dataset_name: name of the current data set (for feature extraction in clustering)
        :param field_in_train_model_grid_area: grid specifics for the heat map
        :param field_in_train_model_trained_scalers: scaling factors of the training data
        :param field_in_train_model: heat map of the trained model
        :param field_in_train_risc: risk values of the training data (used to get rul, risk ratio)
        :param field_in_train_rul: rul values of the training data (used to get rul, risk ratio)
        """
        # parameters
        self.percentage_side_fine = percentage_side_fine
        self._smooth_per_feature = smooth_per_feature
        self._visualize_summed_curve = visualize_summed_curve
        self._visualize_per_feature_curve = visualize_per_feature_curve
        self._visualize_pre_post_risk = visualize_pre_post_risk
        self._test_mode = test_mode
        self._optimize_via_average = optimize_via_average
        self._write_csv = write_csv
        self._csv_path = csv_path
        self._whole_data_set = whole_data_set
        self._enable_all_print = enable_all_print
        self.skip_never = False
        self.skip_never_whole = False
        self.deviation_dict = {}
        self.feature_selection_on = feature_selection_on

        # fields
        self.smoothing_side = smoothing_side
        self._field_in_test_t = field_in_test_t
        self._field_in_test_crit = field_in_test_crit
        self._field_in_test_X = field_in_test_X
        self._field_in_test_id = field_in_test_id
        self._field_in_test_rul = field_in_test_rul
        self._field_in_train_cluster_stored = field_in_train_cluster_stored
        self._field_in_meta_dataset_name = field_in_meta_dataset_name
        self._field_in_train_model_grid_area = field_in_train_model_grid_area
        self._field_in_train_model_trained_scalers = field_in_train_model_trained_scalers
        self._field_in_train_model = field_in_train_model
        self._field_in_train_risc = field_in_train_risc
        self._field_in_train_rul = field_in_train_rul
        super().__init__()

    def _predict_RUL(self, data, current_test_df, cluster_id, all_feature_sum, features_to_skip, timestamp_gap, expected_rul, test=False, fine=-1, dev_dict_path = ""):
        '''
        based on the trained model extract the RUL for the given test dataframe
        :param data_in data dictionary as passed by the main execution chain
        :param current_test_df: Dataframe to use for prediction
        :return column: predicted_rul - remaining useful life as determined by the predictor
        '''

        # 1. Preprocess
        grid_area, trained_scalers, dimensions, train_df, m = self._extract_prediction_information(data, cluster_id, test)

        # 2. Run prediction
        predicted_risk, predicted_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = self._do_prediction(current_test_df, grid_area, trained_scalers, dimensions, train_df, m, all_feature_sum, features_to_skip, timestamp_gap, expected_rul, fine, dev_dict_path)

        return predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict

    def _initialize_feature(self, grid_area, col):
        """
        Initialize an empty grid area
        """
        summ = np.empty(grid_area)
        summ.fill(0)
        all_good = True
        return summ, all_good

    def find_integrals(self, xi, yi):
        """
        Computation of integrals for input
        """
        all_integrals = []
        # integral 0 bis 0.5
        fst = numpy.where(xi <= 0.5)
        x1 = xi[fst]
        y1 = yi[fst]
        all_integrals.append(numpy.trapz(y1, x1))

        # integral 0.25 bis 0.75
        scd = numpy.where(xi>0.25)
        xtmp = xi[scd]
        ytmp = yi[scd]
        scd = numpy.where(xtmp <= 0.75)
        x2 = xtmp[scd]
        y2 = ytmp[scd]
        all_integrals.append(numpy.trapz(y2, x2))

        # integral 0.5 bis 1.0
        thrd = numpy.where(xi>0.5)
        xtmp = xi[thrd]
        ytmp = yi[thrd]
        thrd = numpy.where(xtmp <= 1.0)
        x3 = xtmp[thrd]
        y3 = ytmp[thrd]
        all_integrals.append(numpy.trapz(y3, x3))

        return all_integrals

    def _do_whole_model_prediction(self, cur_test_df, grid_area, dimensions, m, all_feature_sum, features_to_skip, timestamp_gap, dev_dict_path):
        """
        Runs the prediction of the model
        """

        # 1. Initialize
        feature_favorites, per_feature_sum, found_risk, weight_b, feature_favs_dict = [], {}, -1, True, {}

        # 2. Do prediction per feature
        for col in cur_test_df.columns:

            # 2.1. Initialize
            if HeatmapConvolutionTrainer.scaled_relevant_columns(col) or col in features_to_skip:  continue
            summ, all_good = self._initialize_feature(grid_area, col)

            try:
                l_iter_e = len(cur_test_df[col])
                l_iter_s = 0
                if l_iter_e > 6:
                    l_iter_s = len(cur_test_df[col]) - 6
                    l_iter_e = len(cur_test_df[col])

                # 2.2. Per Sample of Training data determine heat curve and add it shifted to most recent
                this_feature_tops = []
                for cur_row in list(range(l_iter_s, l_iter_e)):

                    # 2.3. Skip invalid features
                    try:
                        cur_heat_values, xi = self._get_current_heat(cur_test_df, dimensions, cur_row, m, grid_area, col)
                    except:
                        continue
                    if len(numpy.where(cur_heat_values != 0)[0]) == 0:
                        continue

                    # 2.4. do smoothing if defined
                    if self._smooth_per_feature:
                        cur_heat_values = medfilt(cur_heat_values, 41)
                    summ = summ + cur_heat_values

                    # 2.5 Extract resulting inputs
                    ten_per_idx =  math.floor(len(cur_heat_values)*0.1)
                    ind = self.largest_indices(cur_heat_values, ten_per_idx)
                    weight = cur_heat_values[ind]
                    max_val = numpy.max(weight)
                    normalized_weight = weight/max_val
                    weighted_res = numpy.sum(xi[ind] * normalized_weight)/numpy.sum(normalized_weight)
                    this_risk = weighted_res
                    this_feature_tops.append(this_risk)

                # 2.5. Compute average
                avg = numpy.average(this_feature_tops)
                if summ.argmax() != 0:
                    feature_favs_dict[col] = avg

                # 3. Close Timestamp gap by shifting heatmaps "onto" each other
                summ, xi = self._shift_array(xi, summ, timestamp_gap, m, grid_area)
                per_feature_sum[col] = deepcopy([xi, cur_heat_values])
                all_feature_sum = all_feature_sum + summ
                try:
                    two_max = numpy.array(self.find_integrals(xi, summ)).argsort()[-2:][::-1]
                    two_max = two_max.tolist()
                    if 1 in two_max:
                        # vote
                        two_max.remove(1)
                        self.voting[two_max[0]] += 2
                        self.voting[1] += 1
                    else:
                        self.voting[two_max[0]] += 1
                    self.integral_sum = self.find_integrals(xi, all_feature_sum)
                except:
                    print(traceback.format_exc())

                # 4. Compute average of current heats without outliers
                found_risk = avg
                if not math.isnan(found_risk):
                    feature_favorites.append(found_risk)
            except:
                found_risk = -1

        # 5. Return results
        found_rul = (found_risk - 1)/m
        return found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict

    def _do_prediction(self, cur_test_df, grid_area, trained_scalers, dimensions, train_df, m, all_feature_sum, features_to_skip, timestamp_gap, expected_rul, fine = -1, dev_dict_path = ""):
        """
        Do the prediction
        """

        # 1. initialize
        try:
            if not all_feature_sum:
                all_feature_sum = np.empty(grid_area)
                all_feature_sum.fill(0)
        except:
            pass

        # 2. Scale features
        cur_test_df = self._scale_with_scaler(cur_test_df, trained_scalers)
        feature_favs_dict = {}

        # 3. Apply prediction of main model
        if fine == -1:
            found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = self._do_whole_model_prediction( cur_test_df, grid_area, dimensions, m, all_feature_sum, features_to_skip, timestamp_gap, dev_dict_path)
            return found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict

        # 4. Apply prediction of sidemodel
        if fine != -1:
            weight_b, feature_favorites, per_feature_sum  = True, [], {}
            if self.skip_never: features_to_skip = []

            for col in cur_test_df.columns:
                # 3.1. Initialize
                info_key = "fine_" + str(fine) + "_" + col
                if HeatmapConvolutionTrainer.scaled_relevant_columns(col) or col in features_to_skip:  continue
                summ, all_good = self._initialize_feature(grid_area, col)
                cur_dev = 0
                try:
                    l_iter_e = len(cur_test_df[col])
                    l_iter_s = 0
                    if l_iter_e > 6:
                        l_iter_s = len(cur_test_df[col]) - 6
                        l_iter_e = len(cur_test_df[col])

                    # 3.2. iterate over rows
                    for cur_row in list(range(l_iter_s, l_iter_e)):
                        try:
                            cur_heat_values, xi = self._get_current_heat(cur_test_df, dimensions, cur_row, m, grid_area, col, info_key)
                        except:
                            continue
                        if self._smooth_per_feature: cur_heat_values = medfilt(cur_heat_values, self.smoothing_side)
                        summ = summ + cur_heat_values

                    # 3.3. Shift by bias that was determined
                    bias = cur_dev/m
                    summ, xi  = self._shift_array(xi, summ, bias, m, grid_area)

                    # 3.4. Timestamp gap and store
                    summ, xi  = self._shift_array(xi, summ, timestamp_gap, m, grid_area)
                    per_feature_sum[col] = deepcopy([xi, cur_heat_values])
                    all_feature_sum = all_feature_sum + summ
                    if self._visualize_per_feature_curve: self._visualize_per_feature_curve_m(xi, summ)
                    if summ.argmax() != 0:
                        feature_favs_dict[col] = xi[np.unravel_index(summ.argmax(), summ.shape)[0]] # WIRD ANDERS BESTIMMT IM ENDERGEBNIS -> WAHL SOLLTE NACH TOP 10 Erfolgen
                    else:
                        pass

                    # 3.5. Determine top 10 % weighted average
                    ten_per_idx =  math.floor(len(all_feature_sum)*self.percentage_side_fine)
                    ind = self.largest_indices(all_feature_sum, ten_per_idx)
                    weight = all_feature_sum[ind]
                    max_val = numpy.max(weight)
                    normalized_weight = weight/max_val
                    found_risk = [xi[ind], normalized_weight]

                    # 4.6. Determine maximum of curve
                    feature_favorites.append(xi[np.unravel_index(summ.argmax(), summ.shape)[0]])
                except:
                    found_risk = -1

            # 5. Return results
            found_rul = -1
            return found_risk, found_rul, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict

    def largest_indices(self, ary, n):
        """
        Returns the n largest indices from a numpy array.
        """
        flat = ary.flatten()
        indices = numpy.argpartition(flat, -n)[-n:]
        indices = indices[numpy.argsort(-flat[indices])]
        return numpy.unravel_index(indices, ary.shape)

    def _get_current_heat(self, cur_test_df, dimensions, cur_row, m, grid_area, col, dim_info = False):
        '''
        compute the heat values of the current row and shift it if required

        :param cur_test_df: input test dataframe
        :param dimensions: trained model with heat map
        :param cur_row: index of current row in test dataframe
        :param grid_area: size in x and y of the heatmap grid
        :return heat value curve as determined
        '''

        # 1. Initialize
        if not dim_info:
            dim_info = col
        cur_model = dimensions[dim_info][0]
        xi = dimensions[dim_info][1]
        yi = dimensions[dim_info][2]

        # 2. Get feature row - assume yi to be sorted and get index with y closest to our value
        y_val = list(cur_test_df[col])[cur_row]
        y_rel_row_idx = bisect_left(yi, y_val)

        # 3. Shift curves
        rul_shift = (-1) * (cur_test_df["TS"].max() - cur_test_df["TS"].iloc[cur_row])
        shift = m*rul_shift
        avg_abstand_x = (max(xi)-min(xi))/grid_area
        idx_shifts = math.floor((shift/avg_abstand_x)+0.499)

        # 4. Extract heat curve from model - shift right
        if y_rel_row_idx == 1000: y_rel_row_idx = 999
        cur_heat_values = deepcopy(cur_model[:][y_rel_row_idx])
        if idx_shifts >0:
            if idx_shifts > len(cur_heat_values): # shifting to far
                arr = np.empty(len(cur_heat_values)) # fill idx_shifts new values to the right
                arr.fill(0)
                cur_heat_values = arr
            else:
                cur_heat_values = cur_heat_values[:-idx_shifts] # cut at end
                arr = np.empty(idx_shifts) # fill idx_shifts new values to the left
                arr.fill(0)
                cur_heat_values = np.concatenate([arr, cur_heat_values])

        # shift left
        elif idx_shifts <0:
            if idx_shifts< -len(cur_heat_values): # shifting too far
                arr = np.empty(len(cur_heat_values))# fill new values to right 0000 val val val
                arr.fill(0)
                cur_heat_values = arr
            else:
                idx_shifts = (-1)*idx_shifts
                cur_heat_values = cur_heat_values[:-idx_shifts] # cut end
                arr = np.empty(idx_shifts)# fshift left to get  0000 val val val
                arr.fill(0)
                cur_heat_values = np.concatenate([cur_heat_values, arr])
        return cur_heat_values, xi

    def _shift_array(self, xi, target_array, rul_shift, m, grid_area):

        # 1. Shift curves
        shift = numpy.abs(m * rul_shift)
        avg_abstand_x = (max(xi)-min(xi))/grid_area
        idx_shifts = math.floor((shift/avg_abstand_x)+0.499)

        # 2. shift left
        if rul_shift <0:
            target_array = target_array[idx_shifts:] # cut end
            arr = np.empty(idx_shifts) # fill left to get 0000 val val val
            arr.fill(0)
            cur_heat_values = np.concatenate([target_array, arr])

        # 3. shift nach rechts
        elif rul_shift >0:
            target_array = target_array[:-idx_shifts] # cut end
            arr = np.empty(idx_shifts)# fill right to get 0000 val val val
            arr.fill(0)
            cur_heat_values = np.concatenate([arr, target_array])
        else:
            cur_heat_values = target_array

        return cur_heat_values, xi

    def _scale_with_scaler(self, cur_test_df, trained_scalers):
        '''
        to make the test data comparable to the training data a scaling needs to be performed
        in the same way
        :param cur_test_df: Test data frame that has the features that are to be scaled
        :param trained_scalers: Dictionary mapping ids of features to the scaler used in training
        :return test_df: tested dataframe with additional column scaled_FEATURE_XX
        '''
        for col in cur_test_df.columns:
            if HeatmapConvolutionTrainer.relevant_columns(col):
                continue
            try:
                cur_test_df['scaled_'+col] = trained_scalers[col].transform(cur_test_df[col])
            except:
                cur_test_df['scaled_'+col] = trained_scalers[col].transform(cur_test_df[col].values.reshape(-1,1))
        return cur_test_df

    def _extract_prediction_information(self, data, cluster_id, test = False):
        '''
        for the prediction data needs to be stored during training, the stored values
        are loaded here
        :param data: data dictionary as passed by the main execution chain
        :return grid_area: size of the grid of the heat map (grid_area x grid_area)
        :return trained_scalers: scaling of each feature
        :return dimensions: the heat map
        :return train_df: dataframe of the training data
        :return m: slope of ratio rul to risk i.e. risk = rul * m
        '''

        if test:
            grid_area = data["test_CL_" + str(cluster_id) + "_" + "train_model_grid_area"]
            trained_scalers = data["test_CL_" + str(cluster_id) + "_" + "train_model_trained_scalers"]
            dimensions = data["test_CL_" + str(cluster_id) + "_" + "train_model"]
        else:
            grid_area = data["CL_" + str(cluster_id) + "_" + "train_model_grid_area"]
            trained_scalers = data["CL_" + str(cluster_id) + "_" + "train_model_trained_scalers"]
            dimensions = data["CL_" + str(cluster_id) + "_" + "train_model"]

        # Train Frame
        train_df = data["train_risc"].to_frame()
        if 'rul' in (train_df.columns):
            train_df = train_df.rename(columns={'rul': "RISK"})
        else:
            train_df = train_df.rename(columns={"RUL": "RISK"})
        train_df["RUL"] = data["train_rul"].to_frame()

        # Determine linear mapping rul to risk
        train_df = train_df.dropna().sort_values("RUL").drop_duplicates()
        m = (train_df.iloc[5]["RISK"] - train_df.iloc[10]["RISK"])/(train_df.iloc[5]["RUL"] - train_df.iloc[10]["RUL"])

        return grid_area, trained_scalers, dimensions, train_df, m

    def _extract_dataframe(self, test_df, object_id, rul_thr_min, rul_thr_max, cluster_id):
        ''' From the test dataframe extract the part with RULs between min and max and with
            the given object id, as well as the specified cluster_id

        :param test_df: Dataframe  with testdata
        :param object_id: Id of the current object to be extracted
        :param rul_thr_min: Only RUL values bigger or equal than this are extracted
        :param rul_thr_max: Only RUL values smaller or equal than this are extracted
        :param cluster_id: Identifier of the cluster to be extracted (that is assigned to the object_id)
        :return test_df: Dataframe satisfying given conditions
        '''

        test_df = test_df[test_df['cluster_id']==cluster_id]
        test_df = test_df.sort_values("TS")
        df1 = test_df[test_df["id"]==object_id]
        test_df_first = df1[(df1["RUL"] >= rul_thr_min) & (df1["RUL"] <= rul_thr_max)] # FOR Evaluation only - in reality not needed

        return test_df_first

    def _assign_cluster(self, data, test_df):
        '''
        From the clustering algorithm performed in the training phase
        get the cluster id by first scaling each row of the data then applying
        a pca to it and then using the cluster predictor of the training phase
        :param data dictionary as passed by the main execution chain
        :param test dataframe with all features
        :return test dataframe with additional column cluster_id indicating the number of the assigned cluster
        '''
        try:
            test_df["cluster_id"] = data["test_cluster_id"]
        except:
            test_df["cluster_id"] = 0
            print("No cluster assigned")
        return test_df

        # 1. Determine Cluster
        cluster_scaler = data["train_cluster_stored"][0]
        cluster_pca = data["train_cluster_stored"][1]
        cluster_predictor = data["train_cluster_stored"][2]

        # 2. add cluster ids
        feature_id_df = test_df[test_df.columns[pd.Series(test_df.columns).str.startswith('FEATURE_')]]
        feature_id_df = feature_id_df.join(test_df["id"])
        grouped_feature_id_df  = feature_id_df.groupby("id")

        # 3. Map to selected features per id
        dataset = data['meta_dataset_name']
        prep_features_df = grouped_feature_id_df.apply(ClusterTraining.feature_extraction, dataset)
        prep_features_df = prep_features_df[prep_features_df.columns[pd.Series(prep_features_df.columns).str.startswith('CL_FEATURE')]]

        # 4. apply scale, pca and cluster
        c_data = cluster_scaler.transform(prep_features_df.as_matrix())
        c_data = cluster_pca.transform(c_data)
        c_data = cluster_predictor.predict(c_data)
        test_df["cluster_id"] = c_data

        return test_df

    def _extract_critical_data_frame(self, data):
        '''
        extracts the critical part of the testing dataframe
        :param data dictionary as passed by the main execution chain
        :return dataframe concatenated with features, ruls, ids etc.
        '''
        test_df = data["test_t"].to_frame().join(data["test_X"])
        test_df['test_crit'] = 1#data['test_crit']#.to_frame().join(test_df)

        test_df = test_df.rename(columns={"RUL": "CRIT"})
        test_df = test_df.rename(columns={"RUL": "RISK"})

        test_df = data['test_id'].to_frame().join(test_df)
        test_df = data['test_rul'].to_frame().join(test_df)
        test_df["id"] = data['test_id']
        if not self._whole_data_set:
            test_df = test_df[test_df["CRIT"]>0]
        test_df["TS"] = data["test_ts"]
        return test_df

    def run_initialize(self, object_id, test_df,  test_mode):
        """
        Do the initialization for this run
        """
        all_feature_sum = False
        cur_df1 = test_df[test_df['id'] == object_id]
        timestamp_gap = 0 # PER Cluster need to shift incoming data else I cannot sum it up
        last_ts = 0
        expected_rul = 99999999
        all_feature_favorites = []
        predicted_risk, predicted_rul = -1, -1
        cnt = 4
        return all_feature_sum, cur_df1, timestamp_gap, last_ts, expected_rul, all_feature_favorites, predicted_risk, predicted_rul, cnt

    def get_cluster_values(self, cur_df1, cluster_id):
        """
        Get value of the cluser
        """
        cur_df2 = cur_df1[cur_df1['cluster_id'] == cluster_id]
        cur_df3 = cur_df2.sort_values("RUL", ascending=False)

        # skip features if specified per cluster
        skip_features = {}
        if not self.skip_never_whole and self.feature_selection_on:
            try:
                skip_features = self.skip_features[0]
                print("Skip WHOLE")
                print(str(skip_features))
            except:
                pass

        # per object predict only the maximum
        first = True
        return cur_df3, skip_features, first

    def get_skip_feats(self, feature_skipping, skip_features, cluster_id):
        """
        Read the features the where set to be skipped
        """
        if not feature_skipping: return []
        try:
            skip_features = self.skip_features[0]
        except:
            pass
        try:
            skip = skip_features[int(cluster_id)]
        except:
            skip = []
        return skip

    def whole_model_prediction(self, cur_df1,  expected_rul, predicted_risk, predicted_rul, all_feature_sum, timestamp_gap, data, all_feature_favorites, only_cluster_id, only_object_id, last_ts, dev_dict_path1, test_mode = False):
        """
        Run prediction for the whole model
        """

        # 1. Initialize
        segmentation_shortening = False
        feature_skipping = True
        whole_model_features = []
        m = -1
        all_votes = []
        self.integral_sum = [0,0,0]

        # 2. Per cluster run test
        for cluster_id in list(cur_df1["cluster_id"].unique()):
            if test_mode and not (cluster_id == 3):
                continue
            if only_object_id and cluster_id not in only_cluster_id: continue
            current_test_df, skip_features, first = self.get_cluster_values(cur_df1, cluster_id)
            self.voting = [0.0, 0.0, 0.0]

            # 2.1. Optimization
            if segmentation_shortening: current_test_df = self.shorten_segment(current_test_df)
            if feature_skipping: skip = self.get_skip_feats(feature_skipping, skip_features, cluster_id)
            if self.skip_never_whole: skip = []

            # 2.2. shift the input curve to align with the one processed next
            if last_ts != 0:
                cur_ts = current_test_df["TS"].max()
                timestamp_gap = cur_ts - last_ts
            if current_test_df["RUL"].min() < expected_rul:  # store last Timestamp for shifting if it is more urgent
                expected_rul = current_test_df["RUL"].min()

            # 2.3. run prediction
            prev_risk, prev_rul = predicted_risk, predicted_rul
            predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = self._predict_RUL(data, current_test_df, cluster_id, all_feature_sum, skip, timestamp_gap, expected_rul, dev_dict_path = dev_dict_path1)

            # 2.4. Count in timestamp bias
            risk_gap = numpy.absolute(timestamp_gap*m)
            if timestamp_gap > 0:
                for p in range(len(whole_model_features)):
                    for f in whole_model_features[p][1]:
                        whole_model_features[p][1][f] += risk_gap
            if timestamp_gap < 0:
                for f in feature_favs_dict:
                    feature_favs_dict[f] += risk_gap
            whole_model_features.append([cluster_id,  feature_favs_dict])

            # 2.5. Deterimne final prediction
            try:
                if predicted_risk == -1: predicted_risk, predicted_rul = prev_risk, prev_rul
                f_in = [f for f in feature_favorites if f >= 0]
                all_feature_favorites += f_in
                predicted_risk = numpy.average(all_feature_favorites)
            except:
                predicted_risk, predicted_rul = prev_risk, prev_rul
            all_votes.append(copy.deepcopy(self.voting))

        # 3. Compute prediction based on voting
        fst = sum([a[0] for a in all_votes])
        scd = sum([a[1] for a in all_votes])
        thr = sum([a[2] for a in all_votes])
        two_max = numpy.array([fst, scd, thr]).argsort()[-2:][::-1].tolist()
        model = sorted(two_max)[-1]
        fine_indices = [model]
        if numpy.argmax(numpy.array(self.integral_sum)) == 1:
            fine_indices = [2]

        return fine_indices, predicted_risk, whole_model_features, m, expected_rul