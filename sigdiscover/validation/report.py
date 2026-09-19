import os
from jinja2 import Template
from sigdiscover.utils.io import ensure_dir

def generate_validation_report(benchmark_results: dict, output_path: str) -> None:
    ensure_dir(os.path.dirname(output_path))
    template_str = """
    <!DOCTYPE html>
    <html lang="en"><body><h1>SigDiscover Validation Report</h1><p>Generated on: {{ timestamp }}</p>
    <h2>1. Synthetic Recovery Benchmark</h2>
    <table><tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Precision</td><td>{{ benchmarks.synthetic_recovery.precision | round(4) }}</td></tr>
    <tr><td>Recall</td><td>{{ benchmarks.synthetic_recovery.recall | round(4) }}</td></tr>
    <tr><td>F1 Score</td><td>{{ benchmarks.synthetic_recovery.f1 | round(4) }}</td></tr>
    <tr><td>Mean Cosine Similarity</td><td>{{ benchmarks.synthetic_recovery.mean_cosine | round(4) }}</td></tr>
    <tr><td>Exposure Correlation</td><td>{{ benchmarks.synthetic_recovery.exposure_r | round(4) }}</td></tr></table>
    <h2>2. Cross-Validation Benchmark</h2>
    <table><tr><th>Metric</th><th>Value</th></tr>
    <tr><td>Test Reconstruction R²</td><td>{{ benchmarks.cross_validation.test_reconstruction_r2 | round(4) }}</td></tr></table>
    <h2>3. Stability Under Perturbation</h2>
    <table><tr><th>Noise Level</th><th>Mean Cosine Similarity to Baseline</th></tr>
    {% for noise, sim in benchmarks.noise_stability.items() %}<tr><td>{{ noise }}</td><td>{{ sim | round(4) }}</td></tr>{% endfor %}</table>
    <h2>4. SigProfilerExtractor Agreement</h2>
    <table><tr><th>Metric</th><th>Value</th></tr><tr><td>Agreement Rate</td><td>{{ benchmarks.sigprofiler_agreement | round(4) }}</td></tr></table>
    </body></html>"""
    with open(output_path, 'w') as f:
        f.write(Template(template_str).render(**benchmark_results))