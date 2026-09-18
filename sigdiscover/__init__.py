import importlib.metadata

try:
    __version__ = importlib.metadata.version("sigdiscover")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.2.0"

from sigdiscover.assignment.similarity import assign_to_cosmic
from sigdiscover.extraction.nmf import nmf_mutational_signatures
from sigdiscover.extraction.rank_selection import select_optimal_rank
from sigdiscover.matrices.dbs import build_dbs78_matrix
from sigdiscover.matrices.id import build_id83_matrix
from sigdiscover.matrices.sbs import build_sbs96_matrix

__all__ = [
    "assign_to_cosmic",
    "build_dbs78_matrix",
    "build_id83_matrix",
    "build_sbs96_matrix",
    "nmf_mutational_signatures",
    "select_optimal_rank"
]
