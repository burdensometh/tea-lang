import attr
import pandas as pd
import os
import csv
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
import requests

from tea.runtimeDataStructures.covariateJoin import (
    AnalysisSnapshot,
    CovariateJoinSpec,
    JoinEvidence,
    as_of_join,
    make_snapshot,
)

BASE_PATH = os.getcwd()


def _dir_exists(path):
    return os.path.isdir(path) and os.path.exists(path)

def absolute_path(p: str) -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), p)

@attr.s(hash=True)
class Dataset(object):
    dfile = attr.ib()  # path name
    variables = attr.ib()  # list of Variable objects <-- TODO: may not need this in new implementation....
    pid_col_name = attr.ib()  # name of column in pandas DataFrame that has participant ids
    row_pids = attr.ib(init=False, eq=False)  # list of unique participant ids
    data = attr.ib(init=False, eq=False)  # pandas DataFrame
    evidence = attr.ib(init=False, eq=False, repr=False, default=None)  # JoinEvidence or None
    snapshot = attr.ib(init=False, eq=False, repr=False, default=None)  # AnalysisSnapshot
    
    @staticmethod
    def load(path: str, name):
        assert(isinstance(path, str))
        home = Path.home()
        tea_path = home / '.tea'
        if not _dir_exists(tea_path):
            os.mkdir(tea_path)
        data_path = tea_path / 'data'
        if not _dir_exists(data_path):
            os.mkdir(data_path)
        
        url = urlparse(path)
        csv_name = name if '.csv' in name else str(name + '.csv')
        csv_path = data_path / csv_name

        # URL
        if url.scheme != '':
            data = requests.get(path)

            with open(csv_path, 'w') as f:
                writer = csv.writer(f)
                reader = csv.reader(data.text.splitlines())

                for row in reader:
                    writer.writerow(row)
        else: 
            with open(path, 'r') as readfile: 
                reader = csv.reader(readfile)
                with open(csv_path, 'w') as writefile:
                    writer = csv.writer(writefile)

                    for row in reader:
                        writer.writerow(row)

        return csv_path

    def __attrs_post_init__(self):
        if isinstance(self.dfile, (str, Path)):
            # import pdb;pdb.set_trace()
            self.data = pd.read_csv(self.dfile)
        elif isinstance(self.dfile, pd.DataFrame):
            self.data = self.dfile
        else:
            raise ValueError(f"Dataset expected DataFrame, str, or Path. Received: {self.dfile}")

        self.row_pids = self._compute_row_pids()
        self.snapshot = make_snapshot(self.data, self.variables)

        # if self.pid_col_name:
        #     # Reindex DataFrame indices to be pids
        #     self.data.set_index(self.pid_col_name, inplace=True)
        # else:
            # Treat each row as a unique observation

    def _compute_row_pids(self):
        if self.pid_col_name is not None and self.pid_col_name in self.data.columns:
            return list(pd.unique(self.data[self.pid_col_name]))
        return list(self.data.index)

    # -- External temporal covariates ---------------------------------------

    def join_covariates(
        self,
        history,
        spec: CovariateJoinSpec,
        *,
        inplace: bool = False,
    ) -> "Dataset":
        """Attach participant-scoped history attributes as of each event.

        The analysis rows keep their identity (order and index); every event
        gets at most one history record, the one in force at its event time.
        Future records are never used to backfill earlier events, and no
        one-to-many merge can inflate the sample size.

        ``spec`` declares the event-time column, the effective-time column,
        the optional expiry column and the participant key, together with the
        policies for overlapping versions, duplicate keys and missing matches.
        See :class:`tea.runtimeDataStructures.covariateJoin.CovariateJoinSpec`.

        Returns a new ``Dataset`` (or ``self`` when ``inplace=True``) whose
        ``data``, ``variables`` schema and ``snapshot`` all describe the same
        post-join frame.  ``history`` and this dataset's original frame are
        never mutated.

        New attributes must also be declared via ``define_variables`` before
        ``hypothesize`` can use them; they are validated against that schema
        here, at join time.
        """
        if isinstance(history, Dataset):
            history_frame = history.data
        elif isinstance(history, pd.DataFrame):
            history_frame = history
        elif isinstance(history, (str, Path)):
            history_frame = pd.read_csv(history)
        else:
            raise ValueError(
                "history must be a DataFrame, path, or Dataset. "
                f"Received: {type(history)!r}"
            )

        joined, evidence = as_of_join(
            self.data, history_frame, spec, self.variables
        )
        snapshot = make_snapshot(
            joined, self.variables, is_joined=True, evidence=evidence
        )

        if inplace:
            self.data = joined
            self.evidence = evidence
            self.snapshot = snapshot
            self.row_pids = self._compute_row_pids()
            return self

        result = Dataset(joined, self.variables, self.pid_col_name)
        result.evidence = evidence
        result.snapshot = snapshot
        return result

    @property
    def join_evidence(self) -> Optional[JoinEvidence]:
        return self.evidence

    def assert_snapshot_current(self) -> None:
        """Assert ``data`` still matches the snapshot taken at load/join time."""
        if self.snapshot is not None:
            self.snapshot.assert_current()

    def __getitem__(self, var_name: str):
        for v in self.variables:  # checks that the Variable is known to the Dataset object
            if v.name == var_name: 
                return self.data[var_name]  # returns the data, not the variable object

    # Returns variable object -- may want to altogether replace the __getitem__
    def get_variable(self, var_name: str): 
        for v in self.variables: 
            if v.name == var_name: 
                return v

    def get_variable_data(self, var_name: str):
        for v in self.variables: 
            if v.name == var_name:
                return {'dtype': v.dtype,
                        'categories': v.categories} 

    # SQL style select
    def select(self, col: str, where: list = None):

            # Add back ticks to variable names that have spaces in them 
        def format_where(vars: list, where: list): 
            formatted_where = []
            for clause in where: 
                bt_var_name = '`' # opening back tick
                for var in vars: 
                    # Is the variable in the where clause? 
                    if var.name in clause: 
                        # Does the variable name have a space in it? 
                        if ' ' in var.name: 
                            bt_var_name += var.name 
                            bt_var_name += '`' # closing back tick
                            split_clause = clause.split(var.name)
                            assert(len(split_clause) == 2)
                            assert(len(split_clause[0]) == 0)
                            formatted_where.append(bt_var_name + split_clause[1])
                        else: 
                            formatted_where.append(clause)
            
            formatted_where_str = "&".join(formatted_where)
            return formatted_where_str            

        df = self.data
        if where: # not None
            # query = build_query(self.variables, where)
            query = format_where(self.variables, where)
            # import pdb; pdb.set_trace()
            res = df.query(query)[col] # makes a copy
            # import pdb; pdb.set_trace()
        else: 
            res = df[col]

        return res
    
    def has_empty_data(self):
        assert(self.data is not None)
        assert(isinstance(self.data, pd.DataFrame))

        return self.data.empty