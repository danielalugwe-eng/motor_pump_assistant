import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.rag.llm import build_rag_context, route_query
from src.features.extract import extract_features
import numpy as np

assert route_query('Will this pump break next week?') == 'predictor'
assert route_query('How do I maintain this pump?') == 'rag'
assert build_rag_context({'documents': [['First chunk', 'Second chunk']]}) == 'First chunk\n\nSecond chunk'
window = np.sin(np.linspace(0, 4 * np.pi, 2048))
features = extract_features(window)
expected = {
    'mean',
    'std',
    'rms',
    'skew',
    'kurtosis',
    'ptp',
    'crest_factor',
    'dominant_freq',
    'spectral_energy',
    'spectral_entropy',
}
assert set(features.keys()) == expected
print('smoke tests passed')
