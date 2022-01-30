import logging
import typing

import pandas as pd
import numpy as np

from dask import compute
from dask.delayed import Delayed, delayed
from dask.distributed import Client as DaskClient
import dask.dataframe
import dask.array


from hamilton import node
from hamilton import base


logger = logging.getLogger(__name__)


class DaskExecutor(base.HamiltonExecutor):
    """Class representing what's required to make Hamilton run on Dask"""

    def __init__(self, dask_client: DaskClient, result_builder: base.ResultMixin = None, visualize: bool = True):
        """Constructor

        :param dask_client: the dask client -- we don't do anything with it, but thought that it would be useful
            to wire through here.
        :param result_builder: The function that will build the result. Optional, defaults to pandas dataframe.
        :param visualize: whether we want to visualize what Dask wants to execute.
        """
        self.client = dask_client
        self.result_builder = result_builder if result_builder else base.PandasDataFrameResult()
        self.visualize = visualize

    def check_input_type(self, node: node.Node, input: typing.Any) -> bool:
        # NOTE: the type of dask Delayed is unknown until they are computed
        if isinstance(input, Delayed):
            return True
        elif node.type == pd.Series and isinstance(input, dask.dataframe.Series):
            return True
        elif node.type == np.array and isinstance(input, dask.array.Array):
            return True

        return node.type == typing.Any or isinstance(input, node.type)

    def execute_node(self, node: node.Node, kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        return delayed(node.callable)(**kwargs)

    def build_result(self, columns: typing.Dict[str, typing.Any]) -> pd.DataFrame:
        for k, v in columns.items():
            logger.info(f'Got column {k}, with type [{type(v)}].')
        delayed_combine = delayed(self.result_builder.build_result)(**columns)
        if self.visualize:
            delayed_combine.visualize()
        df, = compute(delayed_combine)
        return df
