#!/usr/bin/env python
# -*- coding: iso-8859-1 -*-
import warnings; warnings.simplefilter('ignore')
from dbse.tools.api import Parameter as P, Tester
import time
from dbse.tools.data import DataSource, DataStructure
from dbse.HeatmapConvolutionTrainer import HeatmapConvolutionTrainer
import pandas as pd


def set_features_to_skip():
    '''
    Note: any feature that is to be skipped requires the prefix scaled_
    A dictionary is passed where the key indicates the cluster id and the value is a list of features to
    skip during test
    '''

    skip_features = {}
    skip_features[0] = dict() # main model features
    skip_features[1] = dict() # fine tuner model left features
    skip_features[2] = dict() # fine tuner model mid features
    skip_features[3] = dict() # fine tuner model right features

    # one per cluster e.g.
    feats = dict()
    feats[0] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[1] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[2] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[3] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']
    feats[4] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_4','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_14','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21']
    feats[5] = ['scaled_FEATURE_2','scaled_FEATURE_3','scaled_FEATURE_7','scaled_FEATURE_8','scaled_FEATURE_9','scaled_FEATURE_11','scaled_FEATURE_13','scaled_FEATURE_15','scaled_FEATURE_17','scaled_FEATURE_20','scaled_FEATURE_21','scaled_FEATURE_12']

    # here use same features for all models
    skip_features[0] = feats
    skip_features[1] = feats
    skip_features[2] = feats
    skip_features[3] = feats

    return skip_features


if __name__ == "__main__":

    # -----------------------------------------------------------------------------------------------------
    # Parameters
    # -----------------------------------------------------------------------------------------------------
    # Algorithm and Model
    P.te_feature_selection_on = True # If true features are selected according to the features given
    P.tr_remove_outliers = True # If true outliers are removed per feature heat map
    P.te_smoothing_side = 51 # smoothing parameter
    P.tr_kernel_size = 11 # Kernel size
    P.te_percentage_side_fine = 0.1 # Percentage
    visualize_heatmap = False # If true for each class the heatmaps of each feature are shown
    # Application
    model_file = "trained_model.h66"
    run_training = True # if True run Training
    run_testing = True  # if True run Testing

    # -----------------------------------------------------------------------------------------------------
    #   Load Dataset
    # -----------------------------------------------------------------------------------------------------
    # Training Data
    df = pd.read_csv(r"data/trainset.csv", index_col=None)
    df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
    ti = DataStructure()
    ti.X, ti.t, ti.rul, ti.objectid, ti.cluster, ti.ts = DataSource.load_clean(df)

    # Test Data
    df = pd.read_csv(r"data/testset.csv", index_col=None)
    df = df.sort_values(["OBJECTID", "TS"], ascending=[True, True])
    ts = DataStructure()
    ts.X, ts.t, ts.rul, ts.objectid, ts.cluster, ts.ts = DataSource.load_clean(df)

    # Convert to Ingest set
    data = DataSource.to_ingest_set(ti, ts)
    train_data = {key: data[key] for key in ['train_t', 'train_X', 'train_id', 'train_rul', 'train_cluster_id', 'train_ts', 'train_crit', "train_risc"] if key in data}
    test_data = {key: data[key] for key in ['test_t', 'test_X', 'test_id', 'test_rul', 'test_cluster_id', 'test_ts'] if key in data}

    # -----------------------------------------------------------------------------------------------------
    #   Run Training and Test
    # -----------------------------------------------------------------------------------------------------
    # -----------------------------------------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------------------------------------
    if run_training:
        trainer = HeatmapConvolutionTrainer(test_mode = False, whole_data_set= True, remove_outliers=P.tr_remove_outliers, visualize_heatmap = visualize_heatmap, grid_area=P.tr_grid_area, interpol_method=P.tr_interpol_method, kernel_size = P.tr_kernel_size, std_gaus = P.tr_std_gaus)
        t = time.time()
        trained_model, _ = trainer.run(train_data)
        print("Training Time: " + str(time.time() - t))
        trainer.store_model(trained_model, model_file)

    # -----------------------------------------------------------------------------------------------------
    # Testing
    # -----------------------------------------------------------------------------------------------------
    if run_testing:
        # Load
        tester = Tester()
        model = tester.load_model(model_file) # load trained Model
        tester.skip_features = set_features_to_skip() # optionally can pass features to skip e.g. found via feature selection

        # Inference
        t = time.time()
        list_of_results = tester.predict(model, test_data, smooth_per_feat = P.te_smooth_per_feat, csv_pathh = P.te_csv_path, test_mode=False)
        testing_time = time.time() - t
        print("Total Testing Time: " + str(testing_time) + "\n------------------------------")

        # Print Evaluation Results
        y_pred = pd.DataFrame(list_of_results, columns=["object_id", "cluster_id", "rul", "risk", "predicted_rul", "predicted_risk", "invalid", "invalid2", "testing_time"])

        for result in list_of_results:
            print("\n\nObject Id: %s" % str(result[0]))
            print("Cluster Id: %s" % str(result[1]))
            print("Real RUL: %s" % str(result[2]))
            print("Real Risk: %s" % str(result[3]))
            print("Predicted RUL: %s" % str(result[4]))
            print("Predicted Risk: %s" % str(result[5]))