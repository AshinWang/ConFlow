import pandas as pd
import numpy as np

from sklearn.utils import Bunch
from torch.utils.data import Dataset
from typing import  Optional, Any
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from data_pre import Preprocessor

from torch.utils.data import Dataset
from typing import  Optional, Any

class WideDeepDataset(Dataset):
        def __init__(
            self,
            X_tab: Optional[np.ndarray] = None,
            target: Optional[np.ndarray] = None,
        ):
            super(WideDeepDataset, self).__init__()
            self.X_tab = X_tab
            self.Y = target

        def __getitem__(self, idx: int): 
            X = Bunch()
            if self.X_tab is not None:
                X = self.X_tab[idx]
            if self.Y is not None:
                y = self.Y[idx]
                return X, y
            else:
                return X

        def __len__(self):
            if self.X_tab is not None:
                return len(self.X_tab)

def read_data(X_train, X_test, y_train, y_test, valid_state=False, valid_size=0.2): 

    print(X_train.shape, X_test.shape)
    print((X_test.shape[0])/(X_test.shape[0] + X_train.shape[0]))


    data_df = pd.concat([X_train, X_test])
    
    cat_cols, cont_cols = [], []
    for col in data_df.columns:
        # 50 is just a random number I choose here for this example
        if data_df[col].dtype == "object" or data_df[col].nunique() <= 50:
            cat_cols.append(col)
        else:
            cont_cols.append(col)
    
    tab_preprocessor = Preprocessor(embed_cols=cat_cols, continuous_cols=cont_cols)

    X_train = tab_preprocessor.fit_transform(X_train)
    X_test = tab_preprocessor.transform(X_test)

    column_idx=tab_preprocessor.column_idx
    embed_input=tab_preprocessor.embeddings_input
    continuous_cols=tab_preprocessor.continuous_cols


    if valid_state:
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, 
                                                            shuffle = True, 
                                                            random_state = 2022,
                                                            stratify=y_train, 
                                                            test_size=valid_size)    
    else:
        _, X_valid, _, y_valid = train_test_split(X_train, y_train, 
                                                            shuffle = True, 
                                                            random_state = 2022,
                                                            stratify=y_train, 
                                                            test_size=valid_size)                                                     
    y_train = y_train.to_numpy()
    y_valid = y_valid.to_numpy()
    y_test = y_test.to_numpy()
    
    train_dataset = WideDeepDataset(X_tab = X_train, target = y_train)
    valid_dataset = WideDeepDataset(X_tab = X_valid, target = y_valid)
    test_dataset = WideDeepDataset(X_tab = X_test, target = y_test)
    return column_idx, embed_input, continuous_cols, train_dataset, valid_dataset, test_dataset