import pandas as pd

class RUL_DataFrame():
    '''
    Pandas DataFrame 'extention' for RUL. In essence, this is a time 
    series DataFrame type.
    '''
    def __init__(self,
                df,
                data_cols = ['s1', 's2', 's3', 's4', 's5', 's6', 's7', 
                            's8', 's9', 's10', 's11', 's12', 's13', 
                            's14', 's15', 's16', 's17', 's18', 's19', 
                            's20', 's21'],
                option_cols = ['set1', 'set2', 'set3'],
                label_cols = [],
                class_cols = [],
                categ_cols = [],
                id_col = 'id',
                time_col = 'dt',
                time_0 = 1) -> None:
        if not isinstance(df, pd.DataFrame):
            raise Exception('df not pandas data frame')
        self.df = df.copy()

        for lst in [data_cols, option_cols, label_cols, 
                    class_cols, categ_cols]:
            if not isinstance(lst, list):
                raise Exception('_cols not list')
        self.data_cols = data_cols.copy()
        self.option_cols = option_cols.copy()
        self.label_cols = label_cols.copy()
        self.class_cols = class_cols.copy()
        self.categ_cols = categ_cols.copy()

        for st in [id_col, time_col]:
            if not isinstance(st, str):
                raise Exception('_col not string')
        self.id_col = id_col
        self.time_col = time_col

        if not isinstance(time_0, int):
            raise Exception('time_0 not int')
        self.time_0 = time_0 

    def remove_cols(self, cols) -> None:
        for col in cols:
            if col in self.data_cols:
                self.data_cols.remove(col)
            elif col in self.option_cols:
                self.option_cols.remove(col)
            elif col in self.label_cols:
                self.label_cols.remove(col)
            elif col in self.class_cols:
                self.class_cols.remove(col)
            elif col in self.categ_cols:
                self.categ_cols.remove(col)
            else:
                print(f'Cannot remove that column {col}')
                continue
                return None
            del self.df[col]
    
    def deep_copy(self):
        new_df = self.df.copy()
        new_data_cols = self.data_cols.copy()
        new_option_cols = self.option_cols.copy()
        new_label_cols = self.label_cols.copy()
        new_class_cols = self.class_cols.copy()
        new_categ_cols = self.categ_cols.copy()
        new_id_col = self.id_col
        new_time_col = self.time_col
        new_time_0 = self.time_0
        new_rul_df = RUL_DataFrame(new_df, data_cols = new_data_cols, option_cols = new_option_cols,
            label_cols = new_label_cols, class_cols = new_class_cols, categ_cols = new_categ_cols, id_col =  new_id_col,
                time_col = new_time_col, time_0 = new_time_0)
        return new_rul_df