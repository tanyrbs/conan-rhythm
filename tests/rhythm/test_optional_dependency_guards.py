from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.voice_encoder_optional import build_voice_encoder
from utils.audio import librosa_wav2spec
from utils.audio import align as audio_align
from utils.audio import cwt as audio_cwt
from utils.audio import vad as audio_vad
from utils.audio import pitch_utils
from utils.extract_f0_rmvpe import F0Extractor, _require_pyworld, _require_rmvpe_cls


class OptionalDependencyGuardTests(unittest.TestCase):
    def test_librosa_wav2spec_requires_pyloudnorm_only_when_loud_norm_enabled(self) -> None:
        with mock.patch("utils.audio.pyln", None):
            with self.assertRaisesRegex(ImportError, "pyloudnorm is required"):
                librosa_wav2spec(np.zeros(320, dtype=np.float32), sample_rate=16000, loud_norm=True)

    def test_get_mel2ph_requires_textgrid_only_on_use(self) -> None:
        with mock.patch.object(audio_align, "TextGrid", None):
            with self.assertRaisesRegex(ImportError, "textgrid is required"):
                audio_align.get_mel2ph("dummy.TextGrid", "a", np.zeros((4, 80), dtype=np.float32), 320, 16000)

    def test_get_lf0_cwt_requires_pycwt_only_on_use(self) -> None:
        with mock.patch.object(audio_cwt, "wavelet", None):
            with self.assertRaisesRegex(ImportError, "pycwt is required"):
                audio_cwt.get_lf0_cwt(np.zeros((8,), dtype=np.float32))

    def test_build_voice_encoder_requires_resemblyzer_only_on_use(self) -> None:
        with mock.patch(
            "data_gen.voice_encoder_optional.importlib.import_module",
            side_effect=ModuleNotFoundError("No module named 'resemblyzer'"),
        ):
            with self.assertRaisesRegex(ImportError, "Resemblyzer is required"):
                build_voice_encoder()

    def test_save_midi_requires_pretty_midi_only_on_use(self) -> None:
        with mock.patch.object(pitch_utils, "_PRETTY_MIDI", None):
            with mock.patch(
                "utils.audio.pitch_utils.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'pretty_midi'"),
            ):
                with self.assertRaisesRegex(ImportError, "pretty_midi is required"):
                    pitch_utils.save_midi(np.asarray([60]), np.asarray([[0.0, 1.0]]), "dummy.mid")

    def test_require_rmvpe_cls_raises_runtime_error_only_on_use(self) -> None:
        with mock.patch("utils.extract_f0_rmvpe._RMVPE_CLS", None):
            with mock.patch(
                "utils.extract_f0_rmvpe.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'torchaudio'"),
            ):
                with self.assertRaisesRegex(ImportError, "RMVPE pitch extraction requires"):
                    _require_rmvpe_cls()

    def test_require_pyworld_raises_runtime_error_only_on_use(self) -> None:
        with mock.patch("utils.extract_f0_rmvpe._PYWORLD", None):
            with mock.patch(
                "utils.extract_f0_rmvpe.importlib.import_module",
                side_effect=ModuleNotFoundError("No module named 'pyworld'"),
            ):
                with self.assertRaisesRegex(ImportError, "pyworld is required"):
                    _require_pyworld()

    def test_to_lf0_does_not_modify_numpy_input(self) -> None:
        source = np.asarray([0.0, 100.0], dtype=np.float32)
        source_copy = source.copy()
        result = pitch_utils.to_lf0(source)
        self.assertTrue(np.allclose(source, source_copy))
        self.assertLess(result[0], -1.0e9)

    def test_trim_long_silences_requires_scipy_skimage_only_on_use(self) -> None:
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name in {"skimage.transform", "scipy.ndimage"}:
                raise ModuleNotFoundError(f"No module named '{name}'")
            return real_import(name, globals, locals, fromlist, level)

        with mock.patch("builtins.__import__", side_effect=fake_import):
            with self.assertRaisesRegex(ImportError, "scikit-image and scipy"):
                audio_vad.trim_long_silences("dummy.wav")

    def test_generate_batch_requires_pe_ckpt_for_rmvpe(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "requires hparams\\['pe_ckpt'\\]"):
            F0Extractor.generate_batch({}, {"pe": "rmvpe"}, {}, device="cpu")


if __name__ == "__main__":
    unittest.main()
