import numpy
import pandas
from scipy import stats
import dcor
from sklearn.decomposition import PCA

class Labeling():
    # 'shallow' labeling
    def __init__(self) -> None:
        pass
    
    def linear(self, rul_df) -> None:
        # add linear label    
        reverse_column = lambda c: c[::-1]
        rul_df.df['linear'] = rul_df.df[[rul_df.id_col, rul_df.time_col]].groupby(rul_df.id_col).transform(reverse_column) - rul_df.time_0
        rul_df.label_cols.append('linear')

    def frac(self, rul_df) -> None:
        # linear label scaled (0,1)
        remove_linear = False
        if 'linear' not in rul_df.label_cols:
            self.linear(rul_df)
            remove_linear = True
        fraction = lambda c: c / c.max()
        rul_df.df['frac'] = rul_df.df[[rul_df.id_col, 'linear']].groupby(rul_df.id_col).transform(fraction)
        rul_df.label_cols.append('frac')
        if remove_linear:
            del rul_df.df['linear']
            rul_df.label_cols.remove('linear')

    def piecewise(self, rul_df, max_rul = 125) -> None:
        # linear piecewise, i.e. max
        remove_linear = False
        if 'linear' not in rul_df.label_cols:
            self.linear(rul_df)
            remove_linear = True
        rul_df.df['piecewise'] = rul_df.df['linear']
        rul_df.df.loc[rul_df.df['piecewise'] > max_rul, 'piecewise'] = max_rul
        if not 'piecewise' in rul_df.label_cols:
            rul_df.label_cols.append('piecewise')
        if remove_linear:
            del rul_df.df['linear']
            rul_df.label_cols.remove('linear')

    def piecewise_optimized(self, rul_df, optimized_max_rul):
        # linear piecewise, optimized maxfor every trajectory
        if len(optimized_max_rul) == len(rul_df.df[rul_df.id_col].unique()):
            remove_linear = False
            if 'linear' not in rul_df.label_cols:
                self.linear(rul_df)
                remove_linear = True
            rul_df.df['piecewise_optimized'] = rul_df.df['linear']
            for i in range(len(optimized_max_rul)):
                id_mask = rul_df.df[rul_df.id_col] == i + 1
                value_mask = rul_df.df['linear'] > optimized_max_rul[i]
                mask = numpy.logical_and(id_mask, value_mask)
                rul_df.df.loc[mask, 'piecewise_optimized'] = optimized_max_rul[i]
            rul_df.label_cols.append('piecewise_optimized')
            if remove_linear:
                del rul_df.df['linear']
                rul_df.label_cols.remove('linear')
        else:
            print('Error: Scaling does not match ids.')

    def set_min_scale(self, optimized_max_rul, min_scale = 70):
        for i in range(len(optimized_max_rul)):
            if optimized_max_rul[i] < min_scale:
                optimized_max_rul[i] = min_scale
        return optimized_max_rul

    def scale_optimized(self, rul_df, col, optimized_max_rul):
        # scale label (col) with optimized max rul/fault step
        if len(optimized_max_rul) == len(rul_df.df[rul_df.id_col].unique()):
            rul_df.df[f'{col}_scaled'] = rul_df.df[col]
            for i in range(len(optimized_max_rul)):
                rul_df.df.loc[rul_df.df[rul_df.id_col] == i + 1, f'{col}_scaled'] = rul_df.df.loc[rul_df.df[rul_df.id_col] == i + 1, col + '_scaled'] * optimized_max_rul[i]
            rul_df.label_cols.append(f'{col}_scaled')
        else:
            print('Error: Scaling does not match ids.')

    def descriptive(self, rul_df, cols):
        # descriptive labeling, i.e. difference to mean failure values
        last_values = df = pandas.DataFrame(columns=cols)
        for i in rul_df.df[rul_df.id_col].unique():
            last_values = pandas.concat([last_values, rul_df.df.loc[rul_df.df[rul_df.id_col] == i, cols].tail(1)])
        mean_fails = numpy.mean(last_values, axis = 0)
        # 2. calculate differences
        diffs = df = pandas.DataFrame(columns=cols)
        for col in cols:
            diffs[col] = rul_df.df[col] - mean_fails[col]
            diffs[col] = diffs[col] * diffs[col]
        rul_df.df['descriptive'] = diffs.sum(axis = 1)**.5
        rul_df.label_cols.append('descriptive')
     
    def spearman(self, rul_df, rul_df_test = None, min_c = 0.66) -> None:
        # correlation based (spearman)

        # TODO: add cols argument

        remove_linear = False
        if 'linear' not in rul_df.label_cols:
            self.linear(rul_df)
            remove_linear = True
        # a. drop sensors with zero variance, get relevant columns
        rel_cols = rul_df.df[rul_df.data_cols].loc[:, abs(rul_df.df[rul_df.data_cols].var()) > 0.001].columns.to_list()
        # b. spearman rank correlation
        corr = stats.spearmanr(rul_df.df[rel_cols], rul_df.df['linear']).correlation[:,-1]
        corr = corr[:-1]
        roh = pandas.DataFrame(data = numpy.expand_dims(corr, axis = 0), index = [0], columns = rel_cols)
        roh = roh.abs().sort_values(by = 0, axis = 1)
        # get relevant columns according to minimal correlation value
        rel_cols = roh.loc[:, roh.loc[0,:] > min_c].columns.to_list()
        scale_col = lambda x: (x - x.min()) / (x.max() - x.min())
        if len(rel_cols) != 0:
            # c. PCA
            X = numpy.asarray(rul_df.df[rel_cols])
            pca = PCA(n_components = 1)
            rul_df.df['spearman'] = (-1) * pca.fit_transform(X)
            rul_df.df['spearman'] = rul_df.df[[rul_df.id_col, 'spearman']].groupby(rul_df.id_col).transform(scale_col)
            rul_df.label_cols.append('spearman')
            if remove_linear:
                del rul_df.df['linear']
                rul_df.label_cols.remove('linear')
            if rul_df_test:
                X = numpy.asarray(rul_df_test.df[rel_cols])
                rul_df_test.df['spearman'] = (-1) * pca.transform(X)
                rul_df_test.df['spearman'] = rul_df_test.df[[rul_df_test.id_col, 'spearman']].groupby(rul_df_test.id_col).transform(scale_col)
                rul_df_test.label_cols.append('spearman')
        else:
            print('Error: Labeling Failed. No sensor trajecories with correlation constraints. Select lower min_c.')

    def kendall(self, rul_df, rul_df_test = None, min_c = 0.5) -> None:
        # correlation based (kendall)
        remove_linear = False
        if 'linear' not in rul_df.label_cols:
            self.linear(rul_df)
            remove_linear = True
        # a. drop sensors with zero variance, get relevant cols
        rel_cols = rul_df.df[rul_df.data_cols].loc[:, abs(rul_df.df[rul_df.data_cols].var()) > 0.001].columns.to_list()
        # b. kendall tau rank correlation
        corr = []
        for s in rel_cols:
            corr.append(stats.kendalltau(rul_df.df[s], rul_df.df['linear']).correlation)
        roh = pandas.DataFrame(data = numpy.expand_dims(corr, axis = 0), index = [0], columns = rel_cols)
        roh = roh.abs().sort_values(by = 0, axis = 1)
        rel_cols = roh.loc[:, roh.loc[0,:] > min_c].columns.to_list()
        if len(rel_cols) != 0:
            # c. PCA
            X = numpy.asarray(rul_df.df[rel_cols])
            pca = PCA(n_components = 1)
            rul_df.df['kendall'] = (-1) * pca.fit_transform(X)
            scale_col = lambda x: (x - x.min()) / (x.max() - x.min())
            rul_df.df['kendall'] = rul_df.df[[rul_df.id_col, 'kendall']].groupby(rul_df.id_col).transform(scale_col)
            rul_df.label_cols.append('kendall')
            if remove_linear:
                del rul_df.df['linear']
                rul_df.label_cols.remove('linear')
            if rul_df_test:
                X = numpy.asarray(rul_df_test.df[rel_cols])
                rul_df_test.df['kendall'] = (-1) * pca.transform(X)
                rul_df_test.df['kendall'] = rul_df_test.df[[rul_df_test.id_col, 'kendall']].groupby(rul_df_test.id_col).transform(scale_col)
                rul_df_test.label_cols.append('kendall')
        else:
            print('Error: Labeling Failed. No sensor trajecories with correlation constraints. Select lower min_c.')  

    def pearson(self, rul_df, rul_df_test = None, min_c = 0.7) -> None:
        # correlation based (pearson)
        remove_linear = False
        if 'linear' not in rul_df.label_cols:
            self.linear(rul_df)
            remove_linear = True
        # a. drop sensors with zero variance, get relevant columns
        rel_cols = rul_df.df[rul_df.data_cols].loc[:, abs(rul_df.df[rul_df.data_cols].var()) > 0.001].columns.to_list()
        # b. pearson correlation
        corr = []
        for s in rel_cols:
            corr.append(stats.pearsonr(rul_df.df[s], rul_df.df['linear'])[0])
        roh = pandas.DataFrame(data = numpy.expand_dims(corr, axis = 0), index = [0], columns = rel_cols)
        roh = roh.abs().sort_values(by = 0, axis = 1)
        rel_cols = roh.loc[:, roh.loc[0,:] > min_c].columns.to_list()
        if len(rel_cols) != 0:
            # c. PCA
            X = numpy.asarray(rul_df.df[rel_cols])
            pca = PCA(n_components = 1)
            rul_df.df['pearson'] = (-1) * pca.fit_transform(X)
            scale_col = lambda x: (x - x.min()) / (x.max() - x.min())
            rul_df.df['pearson'] = rul_df.df[[rul_df.id_col, 'pearson']].groupby(rul_df.id_col).transform(scale_col)
            rul_df.label_cols.append('pearson')
            if remove_linear:
                del rul_df.df['linear']
                rul_df.label_cols.remove('linear')
            if rul_df_test:
                X = numpy.asarray(rul_df_test.df[rel_cols])
                rul_df_test.df['pearson'] = (-1) * pca.transform(X)
                rul_df_test.df['pearson'] = rul_df_test.df[[rul_df_test.id_col,'pearson']].groupby(rul_df_test.id_col).transform(scale_col)
                rul_df_test.label_cols.append('pearson')
        else:
            print('Error: Labeling Failed. No sensor trajecories with correlation constraints. Select lower min_c.') 

    def distance(self, rul_df, rul_df_test = None, min_c = 0.7) -> None:
        # correlation based (distance)
        remove_linear = False
        if 'linear' not in rul_df.label_cols:
            self.linear(rul_df)
            remove_linear = True
        # a. drop sensors with zero variance, get relevant columns
        rel_cols = rul_df.df[rul_df.data_cols].loc[:, abs(rul_df.df[rul_df.data_cols].var()) > 0.001].columns.to_list()
        # b. distance correlation
        corr = []
        for s in rel_cols:
            corr.append(dcor.distance_correlation(rul_df.df[s].astype(float), rul_df.df['linear'].astype(float)))
        roh = pandas.DataFrame(data = numpy.expand_dims(corr, axis = 0), index = [0], columns = rel_cols)
        roh = roh.abs().sort_values(by = 0, axis = 1)
        rel_cols = roh.loc[:, roh.loc[0,:] > min_c].columns.to_list()
        if len(rel_cols) != 0:
            # c. PCA
            X = numpy.asarray(rul_df.df[rel_cols])
            pca = PCA(n_components = 1)
            rul_df.df['distance'] = (-1) * pca.fit_transform(X)
            scale_col = lambda x: (x - x.min()) / (x.max() - x.min())
            rul_df.df['distance'] = rul_df.df[[rul_df.id_col, 'distance']].groupby(rul_df.id_col).transform(scale_col)
            rul_df.label_cols.append('distance')
            if remove_linear:
                del rul_df.df['linear']
                rul_df.label_cols.remove('linear')
            if rul_df_test:
                X = numpy.asarray(rul_df_test.df[rel_cols])
                rul_df_test.df['distance'] = (-1) * pca.transform(X)
                rul_df_test.df['distance'] = rul_df_test.df[[rul_df_test.id_col,'distance']].groupby(rul_df_test.id_col).transform(scale_col)
                rul_df_test.label_cols.append('distance')
        else:
            print('Error: Labeling Failed. No sensor trajecories with correlation constraints. Select lower min_c.')