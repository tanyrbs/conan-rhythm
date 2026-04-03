from tasks.tts.dataset_utils import FastSpeechDataset
from tasks.Conan.rhythm.dataset_mixin import RhythmConanDatasetMixin


class ConanDataset(RhythmConanDatasetMixin, FastSpeechDataset):
    """Thin dataset entrypoint; rhythm-heavy logic lives in RhythmConanDatasetMixin."""

    pass
