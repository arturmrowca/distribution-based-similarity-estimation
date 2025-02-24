#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
from dbse.HeatmapConvolutionTester import HeatmapConvolutionTester
from contextlib import contextmanager
import sys, os
import numpy
import time
import math

@contextmanager
def suppress_stdout():
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


class Parameter(object):

    # TRAINING
    read_last_training = False
    tr_grid_area = 1000
    tr_interpol_method = 'cubic'
    tr_kernel_size = 11 # alle Experimente bis jetzt mit 51 gemacht!
    tr_std_gaus = 1
    tr_outlier_window_number = 20
    tr_outlier_iterations = 3
    tr_remove_outliers = True

    # TESTING
    te_csv_path = r"tested_result.csv"
    te_smooth_per_feat = True
    te_smoothing_side = 81
    te_percentage_side_fine = 0.1
    te_feature_selection_on = True


class Tester(object):

    result_collector = []
    skip_features = dict()

    def predict(self, trained_model, test_data, test = False, test_dff = None, no_skip = False,  test_mode = False, only_whole_model = False , smooth_per_feat = True, csv_pathh = "results.csv"):

        # 0. Join input information
        data = {**trained_model, **test_data}

        # 1. Initialize & extract data
        Tester.result_collector = []
        skip_features = Tester.skip_features
        only_object_id = []
        only_cluster_id = []
        tester  = HeatmapConvolutionTester(smooth_per_feature = smooth_per_feat, test_mode = test_mode, whole_data_set= True, visualize_summed_curve = True, percentage_side_fine = Parameter.te_percentage_side_fine, visualize_per_feature_curve = False, smoothing_side = Parameter.te_smoothing_side,  visualize_pre_post_risk = False,  write_csv = True, csv_path = csv_pathh, feature_selection_on = Parameter.te_feature_selection_on)
        tester.skip_features = skip_features
        tester.skip_never = no_skip
        tester.skip_never_whole = only_whole_model
        test_df = tester._extract_critical_data_frame(data)
        test_df = tester._assign_cluster(data, test_df)

        # 2. Get maximum RUL
        abs_max_rul = test_df["RUL"].max() # 217
        segment_thrshld = 0.33 *abs_max_rul

        # 3. Do prediction for each input object
        all_feats_dicts = []
        all_whole_model_features = []
        first_m = True
        llll = len(list(test_df["id"].unique())[::-1])
        pop = 0
        aa_stp = 0.25
        aa_cur = 0
        for object_id in list(test_df["id"].unique()):

            # 3.1. Initialize prediction
            print("--------------------------------------------------\nCurrent Object ID\t"+str(object_id))
            pop += 1;ttp = float(pop)/float(llll)
            if ttp >= aa_cur:
                aa_cur += aa_stp
            if only_object_id and object_id not in only_object_id: continue
            test_start_time = time.time()
            all_feature_sum, cur_df1, timestamp_gap, last_ts, expected_rul, all_feature_favorites, predicted_risk, predicted_rul, cnt = tester.run_initialize(object_id, test_df, test_mode)

            # 3.2. Stage 1 prediction of whole Model
            all_fine, predicted_risk_whole, whole_model_features, m_in, expected_rul  = tester.whole_model_prediction(cur_df1, expected_rul, predicted_risk, predicted_rul, all_feature_sum, timestamp_gap, data, all_feature_favorites, only_cluster_id, only_object_id, last_ts, None, test_mode = test_mode)
            if first_m:
                m = m_in
                first_m = False
            risk = 1 + m * expected_rul
            all_whole_model_features.append([risk, whole_model_features])
            if only_whole_model: continue

            # 3.3. Stage 2 Prediction of Submodel
            results = []
            feature_favs_of_all = []
            already = []
            for finetuner_index in all_fine:
                if not finetuner_index in already:
                    already.append(finetuner_index)
                else:
                    continue

                # 3.3.1. Initialize Finetuning
                all_feature_sum, timestamp_gap, last_ts, expected_rul, all_feature_favorites = False, 0,0, 99999999, []
                feat_favs = {}
                for cluster_id in list(cur_df1["cluster_id"].unique()):
                    if test_mode and not (cluster_id == 3):
                        continue
                    if only_object_id and cluster_id not in only_cluster_id: continue
                    cur_df2 = cur_df1[cur_df1['cluster_id'] == cluster_id]
                    current_test_df = cur_df2.sort_values("RUL", ascending=False)

                    # 3.3.2. Shorten and skip if needed
                    dist = current_test_df["RUL"].max()- current_test_df["RUL"].min()
                    try: skip = skip_features[0][int(cluster_id)]
                    except: skip = []
                    if dist > segment_thrshld:
                        thrshld = current_test_df["RUL"].min() + segment_thrshld
                        current_test_df = current_test_df[current_test_df["RUL"] < thrshld]

                    # 3.3.3. shift the input curve to align with the one processed next
                    if last_ts != 0:
                        cur_ts = current_test_df["TS"].max()
                        timestamp_gap = cur_ts - last_ts

                    # 3.3.4. store last Timestamp for shifting if it is more urgent
                    if current_test_df["RUL"].min() < expected_rul:
                        expected_rul = current_test_df["RUL"].min()

                    # 3.3.5. Do the RUL prediction
                    prev_risk, prev_rul = predicted_risk, predicted_rul
                    if no_skip: skip =[]
                    predicted_risk, predicted_rul, m, all_feature_sum, per_feature_sum, feature_favorites, feature_favs_dict = tester._predict_RUL(data, current_test_df, cluster_id, all_feature_sum, skip, timestamp_gap, expected_rul, fine=finetuner_index)
                    last_ts = current_test_df["TS"].max()

                    # 3.3.6. Use last prediction if current update did not optimize the value
                    if predicted_risk==None:
                        predicted_risk = prev_risk
                        predicted_rul = prev_rul

                    # 3.3.7. Shift all feature favorites by timestamp_gap
                    risk_gap = numpy.absolute(timestamp_gap*m)
                    if timestamp_gap > 0:
                        all_feature_favorites = [f + risk_gap for f in all_feature_favorites]
                        for i in feat_favs:
                            for f in feat_favs[i]:
                                feat_favs[i][f] += risk_gap
                    if timestamp_gap < 0:
                        feature_favorites = [f + risk_gap for f in feature_favorites]
                        for f in feature_favs_dict:
                            feature_favs_dict[f] += risk_gap
                    feat_favs[cluster_id] = [finetuner_index, feature_favs_dict]

                    # 3.3.8. Add all features
                    try:
                        if predicted_risk == -1: predicted_risk, predicted_rul = prev_risk, prev_rul
                        f_in = [f for f in feature_favorites if f >= 0]
                        all_feature_favorites += f_in
                    except:
                        predicted_risk, predicted_rul = prev_risk, prev_rul
                    if cluster_id == list(cur_df1["cluster_id"].unique())[-1] and predicted_risk != -1:
                        feature_favs_of_all += all_feature_favorites
                    results.append(predicted_risk)
                    risk = 1 + m * expected_rul
                    all_feats_dicts.append([finetuner_index, risk, feat_favs])
                    feat_favs ={}

            # 4. Try estimation using weighted average of last stage results
            try:
                res_x = results[-1][0]
                res_w = results[-1][1]
                weighted_res = numpy.sum(res_x * res_w)/numpy.sum(res_w)
                predicted_risk = weighted_res
            except:
                continue

            # 5. do estimation using average of all feature favorites
            feature_favs_of_all = [value for value in feature_favs_of_all if not math.isnan(value)]
            # Outlier via average
            avg = numpy.average(feature_favs_of_all)
            thr = 0.15
            favs_no_outies =[f for f in feature_favs_of_all if (avg+thr+0.15)>f and (avg - thr)<f]
            favs_no_outies = [value for value in favs_no_outies if not math.isnan(value)]
            predicted_risk_m = numpy.average(favs_no_outies)
            testing_time = time.time() - test_start_time

            # 6. Output
            predicted_rul_m = (predicted_risk_m - 1)/m
            predicted_rul = (predicted_risk - 1)/m
            print("Predicted RUL\t\t" + str(predicted_rul) + "\nPredicted Risk\t\t"+ str(predicted_risk)+"\n--------------------------------------------------\n")

            # 7. Collect results
            cnt += 1
            object_id = str(object_id)
            rul = expected_rul
            risk = 1 + m * expected_rul
            all_feats_dicts.append([all_fine[-1], risk, feat_favs])
            predicted_risk =predicted_risk
            predicted_rul = predicted_rul
            cluster_id = str(cluster_id)
            try:
                if numpy.count_nonzero(all_feature_sum) == 0:
                    Tester.result_collector.append(["NO_VALUE", cluster_id, rul, risk, "NO_VALUE", "NO_VALUE", testing_time])
                else:
                    Tester.result_collector.append([object_id, cluster_id, rul, risk, predicted_rul, predicted_risk, predicted_rul_m, predicted_risk_m, testing_time])
            except:
                    Tester.result_collector.append([object_id, cluster_id, rul, risk, predicted_rul, predicted_risk, predicted_rul_m, predicted_risk_m, testing_time])
        return Tester.result_collector

    def load_model(self, filepath):
        import dill
        with open(filepath, "rb") as f:
            loaded_model = dill.load(f)
        return loaded_model