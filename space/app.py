"""
Hugging Face Space entry point for the Ajami Lens.

The trained models live in a PRIVATE model repository (named by the
MODEL_REPO variable); this app downloads them with a read token stored as the
HF_TOKEN secret, so the public can use the Lens without being able to
download the weights.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

import torch                                            # noqa: E402
from huggingface_hub import snapshot_download          # noqa: E402

torch.set_num_threads(max(1, os.cpu_count() or 1))

from lens import Reader, build_app                     # noqa: E402

models = snapshot_download(os.environ["MODEL_REPO"], token=os.environ.get("HF_TOKEN"))
reader = Reader(models)
demo = build_app(reader, kraken_bin=None)
try:
    demo.queue(default_concurrency_limit=1, max_size=20)
except TypeError:                                      # older Gradio
    demo.queue()
demo.launch()
