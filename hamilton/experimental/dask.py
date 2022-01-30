import typing

import pandas as pd

from dask import compute
from dask.delayed import Delayed, delayed

from hamilton import node
from hamilton import base


class DaskExecutor(base.HamiltonExecutor):

    def check_input_type(self, node: node.Node, input: typing.Any) -> bool:
        # NOTE: the type of dask Delayed is unknown until they are computed
        if isinstance(input, Delayed):
            return True

        return node.type == typing.Any or isinstance(input, node.type)

    def execute_node(self, node: node.Node, kwargs: typing.Dict[str, typing.Any]) -> typing.Any:
        return delayed(node.callable)(**kwargs)

    def build_result(self, columns: typing.Dict[str, typing.Any]) -> pd.DataFrame:
        columns, = compute(columns)
        return pd.DataFrame(columns)
