from platform import java_ver
from tracemalloc import start
import numpy
import pandas
import typing
from typing import Optional, Union
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


class Preprocessing:
    '''
    Preprocessing Class for RUL. More or less preprocessing as in sklearn 
    but adapted to RUL/Timeseries data.

    Methods: lag_data, categorize, cat_scale, moving_average,
    exponential_smooting, holt_smooting, differences, one_hot_encode
    '''
    def __init__(self) -> None:
        pass

    def lag_data(self, rul_df, cols: list[str], lag: int = 5, dropna: bool = True, categories = False) -> None:
        # create lagged data.
        tempframe = []
        first = True
        names_og = rul_df.df.columns.copy()
        cols_og = cols.copy()
        for i, d in rul_df.df.groupby(rul_df.id_col):
            temp = {} # new dict
            for name in names_og:
                temp[name] = d[name]
                if name in cols:
                    for j in range(lag):
                        temp[f'{name}_lg_{j + 1}'] = d[name].shift(j + 1)
                        if first:
                            if name in rul_df.data_cols:
                                rul_df.data_cols.append(f'{name}_lg_{j + 1}')
                            if categories and name in rul_df.categ_cols:
                                rul_df.categ_cols.append(f'{name}_lg_{j + 1}')
            tempframe.append(pandas.DataFrame(temp, index = d.index))
            first = False
        df = pandas.concat(tempframe)

        if dropna:
            df = df.dropna()
            df.index = range(len(df))
        rul_df.df = df

    def one_hot_encode(self, rul_df, rul_df_test = None, c_col: str = 'kmeans_settings') -> None:
        # one hot encoder
        if c_col not in rul_df.df.columns:
            print(f'{c_col} not in data.')
            return
        for i in rul_df.df[c_col].unique():
            rul_df.df[f'{c_col}_{i}'] = 0
            rul_df.df.loc[rul_df.df[c_col] == i, f'{c_col}_{i}'] = 1
            rul_df.categ_cols.append(f'{c_col}_{i}')
        if rul_df_test:
            for i in rul_df_test.df[c_col].unique():
                rul_df_test.df[f'{c_col}_{i}'] = 0
                rul_df_test.df.loc[rul_df.df[c_col] == i, f'{c_col}_{i}'] = 1
                rul_df_test.categ_cols.append(f'{c_col}_{i}')
    
    def delete_first_n(self, rul_df, n: int) -> None:
        rul_df.df = rul_df.df.drop(rul_df.df.groupby(rul_df.id_col).head(n).index, axis=0)
        rul_df.df = rul_df.df.reset_index(drop = True)


    def moving_average(self, rul_df, shift: int = 5, dropna: bool = True,
                        cols: Optional[list[str]] = None, mp: Optional[int] = None) -> None:
        # moving average
        if not mp:
            mp = shift
        elif mp > shift:
            mp = shift
        if not cols:
            rul_df.df[rul_df.data_cols] = rul_df.df.groupby(rul_df.id_col)[rul_df.data_cols].rolling(shift, min_periods = mp).mean().droplevel(0)
            if dropna:
                rul_df.df = rul_df.df.dropna()
                rul_df.df = rul_df.df.reset_index(drop=True)
        else:
            if not all([col in rul_df.data_cols for col in cols]):
                print('Warning: Operation not on data columns')
            rul_df.df[cols] = rul_df.df.groupby(rul_df.id_col)[cols].rolling(shift, min_periods = mp).mean().droplevel(0)
            if dropna:
                rul_df.df = rul_df.df.dropna()
                rul_df.df = rul_df.df.reset_index(drop=True)

    def weighted_average(self, rul_df, kernel: list[float] = [0.1, 0.2, 0.3, 0.4],
                        dropna: bool = True, cols: Optional[list[str]] = None, mp: Optional[int] = None) -> None:
        # weighted moving average
        if not mp:
            mp = shift
        elif mp > shift:
            mp = shift
        shift = len(kernel)
        if not cols:  
            rul_df.df[rul_df.data_cols] = rul_df.df.groupby(rul_df.id_col)[rul_df.data_cols].rolling(shift, min_periods = mp).apply(lambda x: numpy.mean(kernel * x)).droplevel(0)
            if dropna:
                rul_df.df = rul_df.df.dropna()
                rul_df.df = rul_df.df.reset_index(drop=True)
        else:
            if not all([col in rul_df.data_cols for col in cols]):
                print('Warning: Operation not on data columns')
            rul_df.df[cols] = rul_df.df.groupby(rul_df.id_col)[cols].rolling(shift, min_periods = mp).apply(lambda x: numpy.mean(kernel * x)).droplevel(0)
            if dropna:
                rul_df.df = rul_df.df.dropna()
                rul_df.df = rul_df.df.reset_index(drop=True)

    def exponential_smooth(self, rul_df, alpha: float = 0.1,
                    cols: Optional[list[str]] = None) -> None:
        # exponentially weighted moving average
        if not cols:
            rul_df.df[rul_df.data_cols] = rul_df.df.groupby(rul_df.id_col)[rul_df.data_cols].transform(lambda x: x.ewm(alpha = alpha).mean())
        else:
            if not all([col in rul_df.data_cols for col in cols]):
                print('Warning: Operation not on data columns')
            rul_df.df[cols] = rul_df.df.groupby(rul_df.id_col)[cols].transform(lambda x: x.ewm(alpha = alpha).mean())

    def polynomial_features(self, rul_df, degree: int = 2, cols: Optional[list[str]] = None) -> None:
        poly = PolynomialFeatures(degree = degree)
        if not cols:
            rul_df.df[poly.get_feature_names_out(rul_df.data_cols)] = poly.fit_transform(rul_df.df[rul_df.data_cols])
            rul_df.data_cols = poly.get_feature_names_out(rul_df.data_cols)
        else:
            if not all([col in rul_df.data_cols for col in cols]):
                print('Warning: Operation not on data columns')
            rul_df.df[poly.get_feature_names_out(cols)] = poly.fit_transform(rul_df.df[cols])
            rul_df.data_cols = list(set(rul_df.data_cols + list(poly.get_feature_names_out(cols))))

    def differences(self, rul_df, delta: int = 2, dropna: bool = True,
                    cols: Optional[list[str]] = None) -> None:
        tempframe = []
        first = True
        names_og = rul_df.df.columns.copy()
        data_cols_og = rul_df.data_cols.copy()
        rul_df.data_cols = []
        for i, d in rul_df.df.groupby(rul_df.id_col):
            temp = {} # new dict
            for name in names_og:
                temp[name] = d[name]
                if name in data_cols_og:
                    if first:
                        rul_df.data_cols.append(name)
                    if not cols or name in cols:
                        if first:
                            rul_df.data_cols.append(f'{name}_diff_{delta}')
                        temp[f'{name}_diff_{delta}'] = d[name].shift(delta) - d[name]
            tempframe.append(pandas.DataFrame(temp, index = d.index))
            first = False
        df = pandas.concat(tempframe)
        if dropna:
            df = df.dropna()
            df.index = range(len(df))
        rul_df.df = df

    def drop_zero_variance(self, rul_df, rul_df_test = None, eps: float = 0.00001) -> None:
        zero_cols = rul_df.df[rul_df.data_cols].loc[:, abs(rul_df.df[rul_df.data_cols].var()) < eps].columns.to_list()
        rul_df.df = rul_df.df.drop(columns = zero_cols)
        for col in zero_cols:
            rul_df.data_cols.remove(col)
        if rul_df_test:
            rul_df_test.df = rul_df_test.df.drop(columns = zero_cols)
            for col in zero_cols:
                rul_df_test.data_cols.remove(col)

    def c_drop_zero_variance(self, rul_df, c_cols: list[str],
                    rul_df_test = None, eps: float = 0.00001) -> None:
        # categorical drop zero variance
        zero_cols = []
        for c in c_cols:
            X = rul_df.df.loc[rul_df.df[c] == 1, :]
            zero_cols.append(X[rul_df.data_cols].loc[: , abs(X[rul_df.data_cols].var()) < eps].columns.to_list())
        for col in rul_df.data_cols:
            s = sum([col in l for l in zero_cols])
            if s == len(c_cols):
                rul_df.df = rul_df.df.drop(columns = col)
                rul_df.data_cols.remove(col)
                if rul_df_test and col in rul_df_test.data_cols:
                    rul_df_test.df = rul_df_test.df.drop(columns = col)
                    rul_df_test.data_cols.remove(col)

    def scale(self, rul_df, rul_df_test = None, scale: str = 'std',
                    cols: Optional[list[str]] = None) -> None:
        if scale == 'std':
            scaler = StandardScaler()
        elif scale == 'minmax':
            scaler = MinMaxScaler()
        else:
            return

        if not cols:
            cols = rul_df.data_cols
        X = rul_df.df[cols]
        scaler.fit(X)  
        rul_df.df[cols] = scaler.transform(rul_df.df[cols])
        if rul_df_test:
            rul_df_test.df[cols] = scaler.transform(rul_df_test.df[cols])

    def scale_1_0(self, rul_df, col: str):
        # scaling for labels 1 -> 0
        fl_scaler = lambda col: ((col - col.iloc[-1]) / (col.iloc[0] - col.iloc[-1]))
        rul_df.df[col] = rul_df.df[[rul_df.id_col, col]].groupby(rul_df.id_col).transform(fl_scaler)

    def scale_0_1(self, rul_df, col: str):
        # scaling for labels 0 -> 1
        fl_scaler = lambda col: ((col - col.iloc[0]) / (col.iloc[-1] - col.iloc[0]))
        rul_df.df[col] = rul_df.df[[rul_df.id_col, col]].groupby(rul_df.id_col).transform(fl_scaler)

    def c_scale(self, rul_df, c_cols: list[str],
                    rul_df_test = None, scale: str = 'std',
                    cols: Optional[list[str]] = None) -> None:
        # categorical scaling
        if scale == 'std':
            scaler = StandardScaler()
        elif scale == 'minmax':
            scaler = MinMaxScaler()
        else:
            return

        if not cols:
            cols = rul_df.data_cols
        for c in c_cols:
            X = rul_df.df.loc[rul_df.df[c] == 1, cols]
            scaler.fit(X)  
            rul_df.df.loc[rul_df.df[c] == 1, cols] = scaler.transform(rul_df.df.loc[rul_df.df[c] == 1, cols])
            if rul_df_test:
                rul_df_test.df.loc[rul_df_test.df[c] == 1, cols] = scaler.transform(rul_df_test.df.loc[rul_df_test.df[c] == 1, cols])

    def of_scale(self, rul_df, o_cols: list[str], f_cols,
                    rul_df_test = None, scale: str = 'std',
                    cols: Optional[list[str]] = None) -> None:
        # categorical scaling: settings and fault modes (doesn't make much sense but experiment)
        if scale == 'std':
            scaler = StandardScaler()
        elif scale == 'minmax':
            scaler = MinMaxScaler()
        else:
            return

        if not cols:
            cols = rul_df.data_cols
        for f in f_cols:
            f_mask = rul_df.df[f] == 1
            for c in o_cols:
                o_mask = rul_df.df[c] == 1
                mask = numpy.logical_and(o_mask, f_mask)
                X = rul_df.df.loc[mask, cols]
                scaler.fit(X)  
                rul_df.df.loc[rul_df.df[c] == 1, cols] = scaler.transform(rul_df.df.loc[rul_df.df[c] == 1, cols])
                if rul_df_test:
                    rul_df_test.df.loc[rul_df_test.df[c] == 1, cols] = scaler.transform(rul_df_test.df.loc[rul_df_test.df[c] == 1, cols])

    def cummulate_columns(self, rul_df, cols: list[str], rul_df_test = None) -> None:
        if all([col in rul_df.data_cols for col in cols]):
            print('Warning: Operation not on data columns')
        for col in cols:
            rul_df.df[col] = rul_df.df[[rul_df.id_col, col]].groupby(rul_df.id_col).cumsum()
            if rul_df_test:
                rul_df_test.df[col] = rul_df_test.df[[rul_df_test.id_col, col]].groupby(rul_df_test.id_col).cumsum()

    def fourier_denoise_clumns(self, rul_df, psd_threshold,
                    cols: Optional[list[str]] = None, trend: bool = True) -> None:
        if not cols:
            cols = rul_df.data_cols
        if not all([col in rul_df.data_cols for col in cols]):
            print('Warning: Operation not on data columns')
        for col in cols:
            for i in rul_df.df[rul_df.id_col].unique():
                x = numpy.asarray(rul_df.df.loc[rul_df.df[rul_df.id_col] == i, col])
                d = len(x)
                if trend:
                    slope = ((x[d-1] - x[0])/d)
                    x = x - numpy.arange(d) * slope
                ft = numpy.fft.fft(x, d)
                psd = ft * numpy.conj(ft) / d
                psd_ids = psd > psd_threshold
                ft_clean = ft * psd_ids
                x_clean = numpy.fft.ifft(ft_clean)
                if trend:
                    x_clean = x_clean + numpy.arange(d) * slope
                rul_df.df.loc[rul_df.df[rul_df.id_col] == i, col] = x_clean.real

    def rolling_mean_outlier_detection(self, rul_df, cols: Optional[list[str]], std_n: int = 3, window: int = 5):
        # rolling mean +- std_n * std condidered outlier. replaced with mean
        if not cols:
            cols = rul_df.data_cols
        if not all([col in rul_df.data_cols for col in cols]):
            print('Warning: Operation not on data columns')
        for col in cols:
            means = rul_df.df.groupby(rul_df.id_col)[col].rolling(window, min_periods=1).mean()
            stds = rul_df.df.groupby(rul_df.id_col)[col].rolling(window, min_periods=1).std()
            min_bound = means - std_n * stds
            max_bound = means + std_n * stds
            mask1 = min_bound.values > rul_df.df[col].values
            mask2 = rul_df.df[col].values > max_bound.values
            mask = numpy.logical_or(mask1, mask2)
            rul_df.df[col] = rul_df.df[col] * numpy.logical_not(mask) + means.reset_index()[col] * mask

    def highest_acceleration(self, rul_df, col: str, p: float = 0.033):
        fault_index = []
        fault_timestep =[]
        scale = []
        for i, d in rul_df.df.groupby(rul_df.id_col):
            len_d = d.shape[0]
            w = int(len_d * p)
            a_max = 0
            t_fault_index = 0
            for t in range(len_d - 5*w):
                s1 = d[col].iloc[range(t,t+w)].mean()
                s2 = d[col].iloc[range(t+2*w,t+3*w)].mean()
                s3 = d[col].iloc[range(t+4*w,t+5*w)].mean()
                v1 = s1 - s2
                v2 = s2 - s3
                a = v1 - v2
                if a > a_max:
                    t_fault_index = int(t+2.5*w) # approximate index of highest acceleration
                    t_fault = d[rul_df.time_col].iloc[t_fault_index]
                    a_max = a
            fault_index.append(t_fault_index)
            fault_timestep.append(t_fault)
            scale.append(d[rul_df.time_col].iloc[len_d - 1] - t_fault) # time of failure - fault step time
        return scale, fault_timestep, fault_index

    def get_classy_change_step(self, rul_df, col: str):
        fault_index = []
        fault_timestep =[]
        scale = []
        for i, d in rul_df.df.groupby(rul_df.id_col):
            len_d = d.shape[0]
            start_value = d[col].iloc[0]
            t_fault_index = 0
            for j in range(len_d):
                if not d[col].iloc[j] == start_value:
                    t_fault_index = j
                    t_fault = d[rul_df.time_col].iloc[t_fault_index]
                    break
            fault_index.append(t_fault_index)
            fault_timestep.append(t_fault)
            scale.append(d[rul_df.time_col].iloc[len_d - 1] - t_fault)
        return scale, fault_timestep, fault_index

    def acceleration_threshold(self, rul_df, col: str, p: float = 0.05, q: float = 0.25, s: float = 1.15):
        fault_index = []
        fault_timestep =[]
        scale = []
        for i, d in rul_df.df.groupby(rul_df.id_col):
            len_d = d.shape[0]
            normal_time = int(len_d * q) + 1
            w = int(len_d * p)
            a_norm = 0
            t_fault_index = 0
            for t in range(len_d - 5*w):
                s1 = d[col].iloc[range(t,t+w)].mean()
                s2 = d[col].iloc[range(t+2*w,t+3*w)].mean()
                s3 = d[col].iloc[range(t+4*w,t+5*w)].mean()
                v1 = s1 - s2
                v2 = s2 - s3
                a = v1 - v2
                if t+2.5*w <= normal_time and a > a_norm and v1 < 0:
                    a_norm = a
                elif t+2.5*w > normal_time and a > (a_norm * s) and v1 < 0:
                    t_fault_index = int(t+2.5*w) # approximate index of highest acceleration
                    t_fault = d[rul_df.time_col].iloc[t_fault_index]
                    break
            fault_index.append(t_fault_index)
            fault_timestep.append(t_fault)
            scale.append(d[rul_df.time_col].iloc[len_d - 1] - t_fault) # time of failure - fault step time
        return scale, fault_timestep, fault_index

    def poly_fit(self, rul_df, col: str, deg: int = 2, replace = True) -> None:
        a = numpy.empty(0)
        for i, d in rul_df.df.groupby(rul_df.id_col):
            x = numpy.arange(d.shape[0])
            y = numpy.asarray(d[col])
            p = numpy.polyfit(x, y, deg)
            temp = numpy.repeat(p[deg], d.shape[0])
            if deg > 1:
                for j in range(0, deg):
                    temp = temp + numpy.array(p[j] * x**(deg-j))
            a = numpy.append(a,temp)
        if replace:
            rul_df.df[col] = a
        else:
            rul_df.df[f'{col}_poly_{deg}'] = a
            if not f'{col}_poly_{deg}' in rul_df.label_cols:
                rul_df.label_cols.append(f'{col}_poly_{deg}')
            else:
                print(f'Warning, {col}_poly_{deg} already exists in label cols')