import sys
from pathlib import Path


def _setup() -> Path:
     
    notebook_dir = Path(__file__).resolve().parent   
    project_root = notebook_dir.parent                

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    return project_root


PROJECT_ROOT = _setup()

RAW_DATA_PATH = PROJECT_ROOT / 'data' / 'raw' / 'btc_usd.csv'
PROCESSED_DATA_PATH = PROJECT_ROOT / 'data' / 'processed' / 'data_processed.csv'
PREPROCESSED_MODELS_PATH = PROJECT_ROOT / 'results' / 'models'
PLOTS_PATH = PROJECT_ROOT / 'reports' / 'image'

def ensure_paths():
    """Tüm gerekli klasörleri oluştur"""
    PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSED_MODELS_PATH.mkdir(parents=True, exist_ok=True)
    PLOTS_PATH.mkdir(parents=True, exist_ok=True)

ensure_paths()