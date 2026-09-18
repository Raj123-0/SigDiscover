from dataclasses import dataclass, field
from pathlib import Path

from sigdiscover.utils.io import load_yaml


@dataclass
class ProjectConfig:
    name: str = "SigDiscover Run"
    seed: int = 42

@dataclass
class DataConfig:
    cosmic_version: str = "3.4"
    genome_build: str = "GRCh37"
    mutation_types: list[str] = field(default_factory=lambda: ["SBS96", "DBS78", "ID83"])

@dataclass
class ExtractionConfig:
    min_signatures: int = 1
    max_signatures: int = 10
    n_replicates: int = 30
    n_iterations: int = 2000
    tolerance: float = 1.0e-6
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
    noise_levels: list[float] = field(default_factory=lambda: [0.1, 0.2, 0.3])

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
    def from_yaml(cls, path: str | Path) -> "Config":
        raw_config = load_yaml(path)
        if raw_config is None:
            raw_config = {}

        # Known sections
        known_sections = {"project", "data", "extraction", "assignment", "benchmark", "visualization"}
        unknown_sections = set(raw_config.keys()) - known_sections
        if unknown_sections:
            raise ValueError(f"Unknown config sections: {unknown_sections}")

        # Helper to get dict and treat None as empty
        def get_section(name):
            section = raw_config.get(name, {})
            return section if section is not None else {}

        project_raw = get_section("project")
        data_raw = get_section("data")
        extraction_raw = get_section("extraction")
        assignment_raw = get_section("assignment")
        benchmark_raw = get_section("benchmark")
        visualization_raw = get_section("visualization")

        # Check unknown keys per section
        def check_keys(raw_dict, dc):
            valid_keys = {f.name for f in dc.__dataclass_fields__.values()}
            unknown_keys = set(raw_dict.keys()) - valid_keys
            if unknown_keys:
                raise ValueError(f"Unknown keys in {dc.__name__}: {unknown_keys}")

        check_keys(project_raw, ProjectConfig)
        check_keys(data_raw, DataConfig)
        check_keys(extraction_raw, ExtractionConfig)
        check_keys(assignment_raw, AssignmentConfig)
        check_keys(benchmark_raw, BenchmarkConfig)
        check_keys(visualization_raw, VisualizationConfig)

        return cls(
            project=ProjectConfig(**project_raw),
            data=DataConfig(**data_raw),
            extraction=ExtractionConfig(**extraction_raw),
            assignment=AssignmentConfig(**assignment_raw),
            benchmark=BenchmarkConfig(**benchmark_raw),
            visualization=VisualizationConfig(**visualization_raw)
        )