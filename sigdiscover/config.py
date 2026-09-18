from dataclasses import dataclass, field
from typing import List, Union
from pathlib import Path
from sigdiscover.utils.io import load_yaml

@dataclass
class ProjectConfig:
    name: str = "SigDiscover Run"
    seed: int = 42

@dataclass
class DataConfig:
    cosmic_version: Union[str, float] = 3.4
    genome_build: str = "GRCh37"
    mutation_types: List[str] = field(default_factory=lambda: ["SBS96", "DBS78", "ID83"])

@dataclass
class ExtractionConfig:
    min_signatures: int = 1
    max_signatures: int = 10
    n_replicates: int = 100
    n_iterations: int = 1000000
    tolerance: float = 1.0e-15
    init_method: str = "random"

@dataclass
class AssignmentConfig:
    cosine_threshold: float = 0.8
    use_sigprofiler: bool = True

@dataclass
class BenchmarkConfig:
    synthetic_n_samples: int = 100
    synthetic_n_mutations: int = 5000
    synthetic_n_replicates: int = 10
    noise_levels: List[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

@dataclass
class VisualizationConfig:
    dpi: int = 300
    format: str = "png"

@dataclass
class Config:
    project: ProjectConfig = field(default_factory=ProjectConfig)
    data: DataConfig = field(default_factory=DataConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    assignment: AssignmentConfig = field(default_factory=AssignmentConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "Config":
        raw_config = load_yaml(path)
        return cls(
            project=ProjectConfig(**raw_config.get("project", {})),
            data=DataConfig(**raw_config.get("data", {})),
            extraction=ExtractionConfig(**raw_config.get("extraction", {})),
            assignment=AssignmentConfig(**raw_config.get("assignment", {})),
            benchmark=BenchmarkConfig(**raw_config.get("benchmark", {})),
            visualization=VisualizationConfig(**raw_config.get("visualization", {}))
        )