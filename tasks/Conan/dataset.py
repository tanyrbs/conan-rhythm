from tasks.tts.dataset_utils import FastSpeechDataset
from tasks.Conan.rhythm.v1_dataset_mixin import RhythmV1DatasetMixin


class ConanDataset(RhythmV1DatasetMixin, FastSpeechDataset):
    """Thin dataset entrypoint; rhythm-heavy logic lives in RhythmV1DatasetMixin."""

    pass
