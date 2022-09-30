import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def getTFData(path = "dataset", sets = [1], rulfrac = False, maxrul = None, rulmethod = 'linear'):
    """
    Function for loading the NASA Turbofan Dataset. Selected datasets are concatenated. If rulfrac
    selected remaining useful life is scaled to a fraction from 1 to 0.

    Returns pandas dataframes Xtrain, Ytrain, Xtest, Ytest.
    """

    colnames = ['id', 'dt',
                'set1', 'set2', 'set3',
                's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 
                's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21']
    rulcol = 'rul' if not rulfrac else 'rulfrac'
    
    if 1 in sets:
        train1 = pd.read_table(f"{path}/train_FD001.txt", header=None, delim_whitespace=True)
        test1 = pd.read_table(f"{path}/test_FD001.txt", header=None, delim_whitespace=True)
        rul1 = pd.read_table(f"{path}/RUL_FD001.txt", header=None, delim_whitespace=True)
    if 2 in sets:
        train2 = pd.read_table(f"{path}/train_FD002.txt", header=None, delim_whitespace=True)
        test2 = pd.read_table(f"{path}/test_FD002.txt", header=None, delim_whitespace=True)
        rul2 = pd.read_table(f"{path}/RUL_FD002.txt", header=None, delim_whitespace=True)
    if 3 in sets: 
        train3 = pd.read_table(f"{path}/train_FD003.txt", header=None, delim_whitespace=True)
        test3 = pd.read_table(f"{path}/test_FD003.txt", header=None, delim_whitespace=True)
        rul3 = pd.read_table(f"{path}/RUL_FD003.txt", header=None, delim_whitespace=True)
    if 4 in sets:
        train4 = pd.read_table(f"{path}/train_FD004.txt", header=None, delim_whitespace=True)
        test4 = pd.read_table(f"{path}/test_FD004.txt", header=None, delim_whitespace=True)
        rul4 = pd.read_table(f"{path}/RUL_FD004.txt", header=None, delim_whitespace=True)
    
    getRul = lambda col: col[::-1] - 1 # reverses order, - 1 so that it ends with 0 
    getFrac = lambda col: col / col.max() # calculates fraction

    trainframes = []
    testframes = []
    
    maxidtr = 0
    maxidte = 0
    lentr = 0
    lente = 0

    for (train, test, rul) in [('train' + str(i), 'test' + str(i), 'rul' + str(i)) for i in sets]:
        
        # training data
        train = eval(train)
        train.columns = colnames
        
        # create rul for training
        train['rul'] = train[['id', 'dt']].groupby('id').transform(getRul)
        if maxrul is not None:
            train.loc[train.rul > maxrul, 'rul'] = maxrul
        train['rulfrac'] = train[['id','rul']].groupby('id').transform(getFrac)
        
        # update index and make id unique
        train['id'] = train['id'] + maxidtr
        maxidtr = train['id'].max()
        
        train.index = range(lentr, lentr + len(train))
        lentr = lentr + len(train)
        
        trainframes.append(train)
        
        # testing data
        test = eval(test)
        rul = eval(rul)
        test.columns = colnames
            
        # create rul for testing
        test['rul'] = test[['id', 'dt']].groupby('id').transform(getRul)
        for j in test['id'].unique():
            test.loc[test['id'] == j, 'rul'] = test.loc[test['id'] == j, 'rul'] + rul.iloc[j - 1, 0]
        test['rulfrac'] = test[['id','rul']].groupby('id').transform(getFrac)
            
        # update index and make id unique
        test['id'] = test['id'] + maxidte
        maxidte = test['id'].max()
            
        test.index = range(lente, lente + len(test))
        lente = lente + len(test)
            
        testframes.append(test)
        
    train = pd.concat(trainframes)
    test = pd.concat(testframes)

    return train, test

# add lagged data
def lagData(data, lagsize = 5, dropna = False):
    """
    Create lagged data frame.
    """
    
    colnames = data.columns
    tempframe = []
    
    for i, d in data.groupby('id'):
        temp = {} # new dict
        for name in colnames:
            temp[name] = d[name]
            if name not in ['id','dt','rul','rulfrac']:
                for j in range(lagsize):
                    temp['%s_lg_%d' %(name, j + 1)] = d[name].shift(j + 1)
        tempframe.append(pd.DataFrame(temp, index = d.index))            
    df = pd.concat(tempframe)
    if dropna:
        df = df.dropna()
        df.index = range(len(df))
    return df

# categories from settings
def addSettings(data):
    # from prior explorataion
    setting_names = ['set1', 'set2', 'set3']
    settings_df = data[setting_names].copy()
    settings_df['set1'] = settings_df['set1'].round()
    settings_df['set2'] = settings_df['set2'].round(decimals=2)
    
    data['c1'] = 0
    data['c2'] = 0
    data['c3'] = 0
    data['c4'] = 0
    data['c5'] = 0
    data['c6'] = 0

    c = 0
    for i, d in settings_df.groupby(by = ['set1','set2','set3']):
        c += 1
        data.loc[d.index,['c' + str(c)]] = 1
    return data

# standard scaler categorical
def cScale(dftrain, dftest, sensors):

    scaler = StandardScaler()
    
    for c in ['c1', 'c2', 'c3', 'c4', 'c5', 'c6']:
        scaler.fit(dftrain.loc[dftrain[c] == 1, sensors])
        
        dftrain.loc[dftrain[c] == 1, sensors] = scaler.transform(dftrain.loc[dftrain[c] == 1, sensors])
        
        dftest.loc[dftest[c] == 1, sensors] = scaler.transform(dftest.loc[dftest[c] == 1, sensors])
    
    return dftrain, dftest

