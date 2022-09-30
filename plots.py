import matplotlib.pyplot as plt
import numpy
from rul_dataframe import RUL_DataFrame


class Plots:
    '''
    Prefeined plots for RUL.
    '''
    def __init__(self, rul_df) -> None:
        if not isinstance(rul_df, RUL_DataFrame):
            raise Exception('rul_df not RUL_DataFrame')
        self.rul_df = rul_df

    def timeplot_column(self, col, mod = 10) -> None:
        assets = self.rul_df.df[self.rul_df.id_col].unique()
        max_time = self.rul_df.df[self.rul_df.time_col].max()
        plt.figure(figsize=(15,5))
        for i in assets:
            if i % mod == 0:
                plt.plot(self.rul_df.time_col, col, data = self.rul_df.df.loc[self.rul_df.df[self.rul_df.id_col] == i])
        plt.xlim(-10, max_time + 20)
        plt.xticks(numpy.arange(0, max_time, 25))
        plt.ylabel(col)
        plt.xlabel('Time')
        plt.show()

    def reversetimeplot_column(self, col, mod = 10) -> None:
        remove_linear = False
        if 'linear' not in self.rul_df.label_cols:
            func_help = lambda c: c[::-1]
            self.rul_df.df['linear'] = self.rul_df.df[[self.rul_df.id_col, self.rul_df.time_col]].groupby(self.rul_df.id_col).transform(func_help) - self.rul_df.time_0
        assets = self.rul_df.df[self.rul_df.id_col].unique()
        max_time = self.rul_df.df[self.rul_df.time_col].max()        
        plt.figure(figsize=(15,5))
        for i in assets:
            if i % mod == 0:
                plt.plot('linear', col, data = self.rul_df.df.loc[self.rul_df.df[self.rul_df.id_col] == i])
        plt.xlim(max_time + 20, -10)
        plt.xticks(numpy.arange(0, max_time, 25))
        plt.ylabel(col)
        plt.xlabel('RUL')
        plt.show()
        if remove_linear:
            del self.rul_df.df['linear']
            self.rul_df.label_cols.remove('linear')

    def scatter_sensors(self, sensor1, sensor2):
        x = self.rul_df.df.loc[:, sensor1]
        y = self.rul_df.df.loc[:, sensor2]
        plt.figure(figsize=(15,5))
        plt.scatter(x, y)
        plt.xlabel(sensor1)
        plt.ylabel(sensor2)
        plt.show()

    def lagplot_sensor_asset(self, sensor, asset, lag = 1) -> None:
        plt.figure(figsize=(15,5))
        plt.scatter(self.rul_df.df.loc[self.rul_df.df[self.rul_df.id_col] == asset, sensor].head(-lag), self.rul_df.df.loc[self.rul_df.df[self.rul_df.id_col] == asset, sensor].tail(-lag))
        plt.xlabel(sensor)
        plt.ylabel(sensor + ' lag ' + str(lag))
        plt.show()

    def __autocorrelation_sensor_asset(self, rul_df, sen, ass, lag) -> float:
        col = rul_df.df.loc[rul_df.df[rul_df.id_col] == ass, sen]
        mean = col.mean()
        n = (col - mean).pow(2).sum()
        z = ((col.tail(-lag) - mean) * (col.head(-lag) - mean)).sum()
        return z / n

    def autocorrelationplot_sensor(self, sensor, max_lag = 50) -> None:
        assets = self.rul_df.df[self.rul_df.id_col].unique()
        n_assets = len(assets)
        r =[]
        for l in range(1, max_lag + 1):
            rl = 0
            for i in assets:
                rl = rl + self.__autocorrelation_sensor_asset(rul_df = self.rul_df, sen = sensor, ass = i, lag = l)
            r.append(rl / n_assets)
        plt.figure(figsize=(15,5))
        plt.bar(numpy.arange(max_lag), r)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()

    def autocorrelationplot_sensor_asset(self, sensor, asset, max_lag = 50) -> None:
        r =[]
        for l in range(1, max_lag + 1):
            r.append(self.__autocorrelation_sensor_asset(rul_df = self.rul_df, sen = sensor, ass = asset, lag = l))
        plt.figure(figsize=(15,5))
        plt.bar(numpy.arange(max_lag), r)
        plt.xlabel('Lag')
        plt.ylabel('Autocorrelation')
        plt.show()

    def heatmap_correlation(self) -> None:
        corrmat = self.rul_df.df[self.rul_df.data_cols].corr()
        plt.figure(figsize = (15 , 15))
        plt.imshow(corrmat)
        plt.show()

    def visualize_fault_step(self, col, asset, faulsteps, mod = 10) -> None:
        max_time = self.rul_df.df[self.rul_df.time_col].max()
        plt.figure(figsize=(15,5))
        plt.plot(self.rul_df.time_col, col, data = self.rul_df.df.loc[self.rul_df.df[self.rul_df.id_col] == asset])
        plt.axvline(x = faulsteps[asset-1])
        plt.xlim(-10, max_time + 20)
        plt.xticks(numpy.arange(0, max_time, 25))
        plt.ylabel(col)
        plt.xlabel('Time')
        plt.show()