import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.exceptions import NotFittedError
from typing import Union, Tuple, List, Optional
import warnings
warnings.filterwarnings("ignore")

__all__ = ["LabelEncoder"]

class LabelEncoder:
    def __init__(
        self,
        columns_to_encode: Optional[List[str]] = None,
        for_transformer: bool = False,
        shared_embed: bool = False,
    ):
        self.columns_to_encode = columns_to_encode

        self.shared_embed = shared_embed
        self.for_transformer = for_transformer

        self.reset_embed_idx = not self.for_transformer or self.shared_embed

    def fit(self, df: pd.DataFrame) -> "LabelEncoder":
        """Creates encoding attributes"""

        df_inp = df.copy()

        if self.columns_to_encode is None:
            self.columns_to_encode = list(
                df_inp.select_dtypes(include=["object"]).columns
            )
        else:
            # sanity check to make sure all categorical columns are in an adequate
            # format
            for col in self.columns_to_encode:
                df_inp[col] = df_inp[col].astype("O")

        unique_column_vals = dict()
        for c in self.columns_to_encode:
            unique_column_vals[c] = df_inp[c].unique()

        self.encoding_dict = dict()
        if "cls_token" in unique_column_vals and self.shared_embed:
            self.encoding_dict["cls_token"] = {"[CLS]": 0}
            del unique_column_vals["cls_token"]
        # leave 0 for padding/"unseen" categories
        idx = 1
        for k, v in unique_column_vals.items():
            self.encoding_dict[k] = {
                o: i + idx for i, o in enumerate(unique_column_vals[k])
            }
            idx = 1 if self.reset_embed_idx else idx + len(unique_column_vals[k])

        self.inverse_encoding_dict = dict()
        for c in self.encoding_dict:
            self.inverse_encoding_dict[c] = {
                v: k for k, v in self.encoding_dict[c].items()
            }
            self.inverse_encoding_dict[c][0] = "unseen"

        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Label Encoded the categories in ``columns_to_encode``"""
        try:
            self.encoding_dict
        except AttributeError:
            raise NotFittedError(
                "This LabelEncoder instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this LabelEncoder."
            )

        df_inp = df.copy()
        # sanity check to make sure all categorical columns are in an adequate
        # format
        for col in self.columns_to_encode:  # type: ignore
            df_inp[col] = df_inp[col].astype("O")

        for k, v in self.encoding_dict.items():
            df_inp[k] = df_inp[k].apply(lambda x: v[x] if x in v.keys() else 0)

        return df_inp

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Combines ``fit`` and ``transform``

        Examples
        --------

        >>> import pandas as pd
        >>> from pytorch_widedeep.utils import LabelEncoder
        >>> df = pd.DataFrame({'col1': [1,2,3], 'col2': ['me', 'you', 'him']})
        >>> columns_to_encode = ['col2']
        >>> encoder = LabelEncoder(columns_to_encode)
        >>> encoder.fit_transform(df)
           col1  col2
        0     1     1
        1     2     2
        2     3     3
        >>> encoder.encoding_dict
        {'col2': {'me': 1, 'you': 2, 'him': 3}}
        """
        return self.fit(df).transform(df)

    def inverse_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Returns the original categories

        Examples
        --------

        >>> import pandas as pd
        >>> from pytorch_widedeep.utils import LabelEncoder
        >>> df = pd.DataFrame({'col1': [1,2,3], 'col2': ['me', 'you', 'him']})
        >>> columns_to_encode = ['col2']
        >>> encoder = LabelEncoder(columns_to_encode)
        >>> df_enc = encoder.fit_transform(df)
        >>> encoder.inverse_transform(df_enc)
           col1 col2
        0     1   me
        1     2  you
        2     3  him
        """
        for k, v in self.inverse_encoding_dict.items():
            df[k] = df[k].apply(lambda x: v[x])
        return df

class BasePreprocessor:
    """Base Class of All Preprocessors."""

    def __init__(self, *args):
        pass

    def fit(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

    def fit_transform(self, df: pd.DataFrame):
        raise NotImplementedError("Preprocessor must implement this method")

class Preprocessor(BasePreprocessor):
    def __init__(
        self,
        embed_cols: Union[List[str], List[Tuple[str, int]]] = None,
        continuous_cols: List[str] = None,
        scale: bool = True,
        auto_embed_dim: bool = True,
        embedding_rule: str = "fastai_new",
        default_embed_dim: int = 16,
        already_standard: List[str] = None,
        for_transformer: bool = False,
        with_cls_token: bool = False,
        shared_embed: bool = False,
        verbose: int = 1,
    ):
        super(Preprocessor, self).__init__()

        self.embed_cols = embed_cols
        self.continuous_cols = continuous_cols
        self.scale = scale
        self.auto_embed_dim = auto_embed_dim
        self.embedding_rule = embedding_rule
        self.default_embed_dim = default_embed_dim
        self.already_standard = already_standard
        self.for_transformer = for_transformer
        self.with_cls_token = with_cls_token
        self.shared_embed = shared_embed
        self.verbose = verbose

        self.is_fitted = False

        if (self.embed_cols is None) and (self.continuous_cols is None):
            raise ValueError(
                "'embed_cols' and 'continuous_cols' are 'None'. Please, define at least one of the two."
            )

        transformer_error_message = (
            "If for_transformer is 'True' embed_cols must be a list "
            " of strings with the columns to be encoded as embeddings."
        )
        if self.for_transformer and self.embed_cols is None:
            raise ValueError(transformer_error_message)
        if self.for_transformer and isinstance(self.embed_cols[0], tuple):  # type: ignore[index]
            raise ValueError(transformer_error_message)

    def fit(self, df: pd.DataFrame) -> BasePreprocessor:
        """Fits the Preprocessor and creates required attributes"""
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            self.label_encoder = LabelEncoder(
                columns_to_encode=df_emb.columns.tolist(),
                shared_embed=self.shared_embed,
                for_transformer=self.for_transformer,
            )
            self.label_encoder.fit(df_emb)
            self.embeddings_input: List = []
            for k, v in self.label_encoder.encoding_dict.items():
                if self.for_transformer:
                    self.embeddings_input.append((k, len(v)))
                else:
                    self.embeddings_input.append((k, len(v), self.embed_dim[k]))
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                self.scaler = StandardScaler().fit(df_std.values)
            elif self.verbose:
                warnings.warn("Continuous columns will not be normalised")
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Returns the processed ``dataframe`` as a np.ndarray"""
        check_is_fitted(self, condition=self.is_fitted)
        if self.embed_cols is not None:
            df_emb = self._prepare_embed(df)
            df_emb = self.label_encoder.transform(df_emb)
        if self.continuous_cols is not None:
            df_cont = self._prepare_continuous(df)
            if self.scale:
                df_std = df_cont[self.standardize_cols]
                df_cont[self.standardize_cols] = self.scaler.transform(df_std.values)
        try:
            df_deep = pd.concat([df_emb, df_cont], axis=1)
        except NameError:
            try:
                df_deep = df_emb.copy()
            except NameError:
                df_deep = df_cont.copy()
        self.column_idx = {k: v for v, k in enumerate(df_deep.columns)}
        return df_deep.values

    def inverse_transform(self, encoded: np.ndarray) -> pd.DataFrame:
        r"""Takes as input the output from the ``transform`` method and it will
        return the original values.

        Parameters
        ----------
        encoded: np.ndarray
            array with the output of the ``transform`` method
        """
        decoded = pd.DataFrame(encoded, columns=self.column_idx.keys())
        # embeddings back to original category
        if self.embed_cols is not None:
            if isinstance(self.embed_cols[0], tuple):
                emb_c: List = [c[0] for c in self.embed_cols]
            else:
                emb_c = self.embed_cols.copy()
            for c in emb_c:
                decoded[c] = decoded[c].map(self.label_encoder.inverse_encoding_dict[c])
        # continuous_cols back to non-standarised
        try:
            decoded[self.continuous_cols] = self.scaler.inverse_transform(
                decoded[self.continuous_cols]
            )
        except AttributeError:
            pass

        return decoded

    def fit_transform(self, df: pd.DataFrame) -> np.ndarray:
        """Combines ``fit`` and ``transform``"""
        return self.fit(df).transform(df)

    def _prepare_embed(self, df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(self.embed_cols[0], tuple):
            self.embed_dim = dict(self.embed_cols)  # type: ignore
            embed_colname = [emb[0] for emb in self.embed_cols]
        elif self.auto_embed_dim:
            n_cats = {col: df[col].nunique() for col in self.embed_cols}
            self.embed_dim = {
                col: embed_sz_rule(n_cat, self.embedding_rule)  # type: ignore[misc]
                for col, n_cat in n_cats.items()
            }
            embed_colname = self.embed_cols  # type: ignore
        else:
            self.embed_dim = {e: self.default_embed_dim for e in self.embed_cols}  # type: ignore
            embed_colname = self.embed_cols  # type: ignore
        return df.copy()[embed_colname]

    def _prepare_continuous(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.scale:
            if self.already_standard is not None:
                self.standardize_cols = [
                    c for c in self.continuous_cols if c not in self.already_standard
                ]
            else:
                self.standardize_cols = self.continuous_cols
        return df.copy()[self.continuous_cols]

def check_is_fitted(
    estimator: BasePreprocessor,
    attributes: List[str] = None,
    all_or_any: str = "all",
    condition: bool = True,):
    r"""Checks if an estimator is fitted

    Parameters
    ----------
    estimator: ``BasePreprocessor``,
        An object of type ``BasePreprocessor``
    attributes: List, default = None
        List of strings with the attributes to check for
    all_or_any: str, default = "all"
        whether all or any of the attributes in the list must be present
    condition: bool, default = True,
        If not attribute list is passed, this condition that must be True for
        the estimator to be considered as fitted
    """

    estimator_name: str = estimator.__class__.__name__
    error_msg = (
        "This {} instance is not fitted yet. Call 'fit' with appropriate "
        "arguments before using this estimator.".format(estimator_name)
    )
    if attributes is not None and all_or_any == "all":
        if not all([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif attributes is not None and all_or_any == "any":
        if not any([hasattr(estimator, attr) for attr in attributes]):
            raise NotFittedError(error_msg)
    elif not condition:
        raise NotFittedError(error_msg)

def embed_sz_rule(n_cat: int, embedding_rule: str = "fastai_new") -> int:
    r"""Rule of thumb to pick embedding size corresponding to ``n_cat``. Default rule is taken
    from recent fastai's Tabular API. The function also includes previously used rule by fastai
    and rule included in the Google's Tensorflow documentation

    Parameters
    ----------
    n_cat: int
        number of unique categorical values in a feature
    embedding_rule: str, default = fastai_old
        rule of thumb to be used for embedding vector size
    """
    if embedding_rule == "google":
        return int(round(n_cat ** 0.25))
    elif embedding_rule == "fastai_old":
        return int(min(50, (n_cat // 2) + 1))
    else:
        return int(min(600, round(1.6 * n_cat ** 0.56)))
