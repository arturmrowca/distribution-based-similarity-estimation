import pandas as pd
import numpy


class DataStructure:

    X = pd.DataFrame()         # the features
    y = pd.DataFrame()         # the labels
    t = pd.DataFrame()         # the t values (timesteps until failure are negative)
    rul = pd.DataFrame()       # the rul of each sample
    objectid = pd.DataFrame()  # the objectid of all samples
    cluster = pd.DataFrame()   # the cluster, to which the sample belongs
    ts = pd.DataFrame()        # the timestamp of each sample


class DataSource:

    @staticmethod
    def to_ingest_set(train, test):
        data = {}
        data["train_t"] = train.t
        data["train_X"] = train.X
        data["train_id"] = train.objectid
        data["train_rul"] = train.rul
        data["train_cluster_id"] = train.cluster
        data["train_ts"] = train.ts
        data["test_t"] = test.t
        data["test_X"] = test.X
        data["test_id"] = test.objectid
        data["test_rul"] = test.rul
        data["test_cluster_id"] = test.cluster
        data["test_ts"] = test.ts

        # add other datasets here
        data["train_crit"] = numpy.array([1]* len(data["train_X"]))
        data["test_crit"] = numpy.array([1]* len(data["test_X"]))

        # define  risk based on linear mapping
        # training
        rul_max = data["train_rul"].max()
        m = 1.0/float(rul_max)
        data["train_risc"] = 1 - m* data["train_rul"]

        # test
        data["test_risc"] = 1 - m* data["test_rul"]

        return data

    @staticmethod
    def load_clean(X):

        X = pd.DataFrame(X)
        X = X.sort_values(["OBJECTID", "TS"], ascending=[True, True])

        t = X['T']
        rul = X['RUL']
        objectid = X['OBJECTID']
        cluster = X['OPERATIONAL_SETTING']
        ts = X['TS']

        del X['T']
        del X['RUL']
        del X['OBJECTID']
        del X['OPERATIONAL_SETTING']
        del X['TS']

        return X, t, rul, objectid, cluster, ts
