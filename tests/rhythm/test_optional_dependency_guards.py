from __future__ import annotations

import importlib
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


def _pop_modules(prefix: str) -> dict[str, object]:
    removed = {}
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            removed[name] = sys.modules.pop(name)
    return removed


def _restore_modules(prefix: str, removed: dict[str, object]) -> None:
    for name in list(sys.modules):
        if name == prefix or name.startswith(f"{prefix}."):
            sys.modules.pop(name, None)
    sys.modules.update(removed)


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

    def test_require_rmvpe_cls_does_not_require_pyworld_at_import_time(self) -> None:
        removed = _pop_modules("modules.pe.rmvpe")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "pyworld":
                raise ModuleNotFoundError("No module named 'pyworld'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                with mock.patch("utils.extract_f0_rmvpe._RMVPE_CLS", None):
                    cls = _require_rmvpe_cls()
            self.assertEqual(cls.__name__, "RMVPE")
        finally:
            _restore_modules("modules.pe.rmvpe", removed)

    def test_to_lf0_does_not_modify_numpy_input(self) -> None:
        source = np.asarray([0.0, 100.0], dtype=np.float32)
        source_copy = source.copy()
        result = pitch_utils.to_lf0(source)
        self.assertTrue(np.allclose(source, source_copy))
        self.assertLess(result[0], -1.0e9)

    def test_trim_long_silences_requires_webrtcvad_only_on_use(self) -> None:
        waveform = np.zeros(4800, dtype=np.float32)
        with mock.patch.object(audio_vad, "webrtcvad", None):
            with mock.patch.object(audio_vad.librosa.core, "load", return_value=(waveform, 16000)):
                with mock.patch.object(audio_vad.librosa, "resample", side_effect=lambda wav, *args, **kwargs: wav):
                    with self.assertRaisesRegex(ImportError, "webrtcvad is required"):
                        audio_vad.trim_long_silences("dummy.wav", sr=16000, norm=False)

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

    def test_vocoder_infer_tolerates_optional_nsf_binary_failures(self) -> None:
        removed = _pop_modules("tasks.tts.vocoder_infer")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "tasks.tts.vocoder_infer.hifigan_nsf":
                raise OSError("bad native extension")
            if level == 1 and fromlist and "hifigan_nsf" in fromlist:
                raise OSError("bad native extension")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                module = importlib.import_module("tasks.tts.vocoder_infer")
            self.assertIsNone(module.hifigan_nsf)
            self.assertIsInstance(module._HIFIGAN_NSF_IMPORT_ERROR, OSError)
        finally:
            _restore_modules("tasks.tts.vocoder_infer", removed)


if __name__ == "__main__":
    unittest.main()
