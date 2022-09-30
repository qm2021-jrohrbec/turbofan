import numpy
from sklearn import cluster
from fcmeans import FCM

class Classification:

    def __init__(self) -> None:
        pass
    
    # unsupervised:
    def fit_kmeans(self, rul_df, rul_df_test = None, cols = ['set1', 'set2', 'set3'], n = 6, colname = 'kmeans_settings') -> cluster.KMeans:
        '''
        Method for k-means on cols for RUL.
        Inputs: data        |   RULFrame type
                cols        |   column names classification based on
                n           |   number of clusters
        '''
        X = numpy.asarray(rul_df.df[cols])
        kmeans = cluster.KMeans(n_clusters = n, random_state = 0)
        kmeans.fit(X)
        if not colname:
            colname = 'kmeans'
        if colname in rul_df.class_cols:
            print(f'Warning in Classification.fit_kmeans: {colname} already in use before. Will overwrite old column.')
        else:
            rul_df.class_cols.append(colname)
        rul_df.df[colname] = kmeans.labels_

        if rul_df_test:
            rul_df_test.df[colname] = kmeans.predict(numpy.asarray(rul_df_test.df[cols]))
            rul_df_test.class_cols.append(colname)

        return kmeans

    def get_fault_modes(self, rul_df, sensors = ['s7', 's12', 's15', 's20', 's21'], colname = 'kmeans_fault_modes', individual = False):
        if not individual:
            last_values = rul_df.df.groupby(rul_df.id_col)[sensors].tail(1)
            kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(numpy.asarray(last_values))
            label = kmeans.labels_
        else:
            labels = []
            for s in sensors:
                last_values = rul_df.df.groupby(rul_df.id_col)[s].tail(1)
                kmeans = cluster.KMeans(n_clusters=2, random_state=0).fit(numpy.asarray(last_values).reshape(-1, 1))
                #make sure first label is same, such that if classigication is the same, the values of kmeans dont matter
                if kmeans.labels_[0] == 0:
                    labels.append((kmeans.labels_ - 1 )* -1)
                else:
                    labels.append(kmeans.labels_)
            for i in range(len(sensors) - 1):
                if not all(labels[i] == labels[i + 1]):
                    print('Not all spreading sensors coincide')
                    return
            label = labels[0]
        if colname in rul_df.class_cols:
            print(f'Warning in Classification.get_fault_modes: {colname} already in use before. Will overwrite old column.')
        else:
            rul_df.class_cols.append(colname)
        rul_df.df[colname] = numpy.array(rul_df.df[rul_df.id_col])
        for i in range(len(label)):
            rul_df.df.loc[rul_df.df[rul_df.id_col] == i + 1, colname] = label[i]

    def fit_fuzzy_cmeans(self, rul_df, rul_df_test = None, cols = ['set1', 'set2', 'set3'], n = 6, colname = None) -> FCM:
        '''
        Method for fuzzy-c-means on cols for RUL.
        Inputs: data        |   RULFrame type
                cols        |   column names classification based on
                n           |   number of clusters
        '''
        X = numpy.asarray(rul_df.df[cols])
        if not colname:
            colname = 'fcmeans'
        if colname in rul_df.class_cols:
            print(f'Warning in Classification.fit_fuzzy_cmeans: {colname} already in use. Will overwrite old column.')
        else:
            rul_df.class_cols.append(colname)
        fcmeans = FCM(n_clusters = n)
        fcmeans.fit(X)
        rul_df.df[colname] = fcmeans.predict(X)

        if rul_df_test:
            rul_df_test.df[colname] = fcmeans.predict(numpy.asarray(rul_df_test.df[cols]))
            rul_df_test.class_cols.append(colname)
        
        return fcmeans