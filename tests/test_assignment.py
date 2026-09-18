import pytest
import numpy as np
from sigdiscover.assignment.similarity import assign_to_cosmic

def test_assign_to_cosmic():
    np.random.seed(42)
    cosmic_sigs = np.random.dirichlet(np.ones(96), size=2)
    cosmic_names = ['SBS1', 'SBS2']
    disc_sigs = np.zeros((3, 96))
    disc_sigs[0] = cosmic_sigs[0]
    disc_sigs[1] = cosmic_sigs[1] * 0.9 + np.random.uniform(0, 0.01, size=96)
    disc_sigs[2] = np.random.dirichlet(np.ones(96))
    assignments = assign_to_cosmic(disc_sigs, cosmic_sigs, cosmic_names, threshold=0.8)
    assert assignments.iloc[0]['best_cosmic_match'] == 'SBS1'
    assert assignments.iloc[0]['cosine_similarity'] > 0.99
    assert assignments.iloc[1]['best_cosmic_match'] == 'SBS2'
    assert assignments.iloc[2]['is_novel'] == True
    assert 'Novel' in assignments.iloc[2]['best_cosmic_match']
