from __future__ import annotations

import importlib
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_gen.voice_encoder_optional import build_voice_encoder
from data_gen.tts.txt_processors import base_text_processor as txt_base_module
from data_gen.tts.txt_processors.base_text_processor import get_txt_processor_cls
from utils.audio import librosa_wav2spec
from utils.audio import align as audio_align
from utils.audio import cwt as audio_cwt
from utils.audio import vad as audio_vad
from utils.commons import base_task as base_task_module
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

    def test_require_rmvpe_cls_does_not_require_torchaudio_at_import_time(self) -> None:
        removed = _pop_modules("modules.pe.rmvpe")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torchaudio.transforms" or name.startswith("torchaudio"):
                raise OSError("bad torchaudio native extension")
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

    def test_generate_batch_saves_returned_f0_files_without_private_rmvpe_kwargs(self) -> None:
        class FakeRMVPE:
            def __init__(self, model_path, device=None):
                self.model_path = model_path
                self.device = device

            def get_pitch_batch(
                self,
                audios,
                sample_rate,
                hop_size,
                lengths,
                interp_uv=False,
                fmin=50,
                fmax=1000,
            ):
                f0s = [np.linspace(100.0, 120.0, num=length, dtype=np.float32) for length in lengths]
                uvs = [np.zeros(length, dtype=np.bool_) for length in lengths]
                return f0s, uvs

        with tempfile.TemporaryDirectory() as tmp:
            wav_dir = Path(tmp) / "speaker"
            wav_dir.mkdir(parents=True, exist_ok=True)
            wav_fn = wav_dir / "demo.wav"
            items = {
                "demo": {
                    "wav_fn": str(wav_fn),
                    "duration": 0.04,
                }
            }
            hparams = {
                "pe": "rmvpe",
                "pe_ckpt": "dummy.ckpt",
                "audio_sample_rate": 16000,
                "hop_size": 160,
                "f0_max": 800,
                "f0_min": 50,
            }
            wav = np.zeros(640, dtype=np.float32)
            expected_path = Path(str(wav_dir) + "_f0") / "demo_f0.npy"

            with mock.patch("utils.extract_f0_rmvpe._require_rmvpe_cls", return_value=FakeRMVPE):
                with mock.patch("utils.extract_f0_rmvpe.librosa.core.load", return_value=(wav, 16000)):
                    F0Extractor.generate_batch(items, hparams, {}, device="cpu", bsz=1, max_tokens=1000)

            self.assertTrue(expected_path.exists())
            saved = np.load(expected_path, allow_pickle=False)
            self.assertEqual(saved.shape[0], 4)
            self.assertTrue(np.allclose(saved, np.linspace(100.0, 120.0, num=4, dtype=np.float32)))

    def test_generate_batch_rejects_non_rmvpe_batch_mode(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "supports only pe=rmvpe"):
            F0Extractor.generate_batch({}, {"pe": "pw"}, {}, device="cpu")

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

    def test_conan_module_import_does_not_require_torchdyn_at_import_time(self) -> None:
        removed_conan = _pop_modules("modules.Conan.Conan")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torchdyn.core" or name.startswith("torchdyn"):
                raise ModuleNotFoundError("No module named 'torchdyn'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                module = importlib.import_module("modules.Conan.Conan")
            self.assertTrue(hasattr(module, "Conan"))
            self.assertTrue(hasattr(module, "ConanPostnet"))
        finally:
            _restore_modules("modules.Conan.Conan", removed_conan)

    def test_conan_flow_helpers_raise_clear_error_when_torchdyn_is_missing(self) -> None:
        removed_conan = _pop_modules("modules.Conan.Conan")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "torchdyn.core" or name.startswith("torchdyn"):
                raise ModuleNotFoundError("No module named 'torchdyn'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                module = importlib.import_module("modules.Conan.Conan")
                with self.assertRaisesRegex(ImportError, "requires torchdyn"):
                    module._require_flow_mel()
                with self.assertRaisesRegex(ImportError, "requires torchdyn"):
                    module._require_reflow_f0()
        finally:
            _restore_modules("modules.Conan.Conan", removed_conan)

    def test_txt_processor_package_import_does_not_require_g2p_en_at_import_time(self) -> None:
        removed = _pop_modules("data_gen.tts.txt_processors")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "g2p_en" or name.startswith("g2p_en."):
                raise ModuleNotFoundError("No module named 'g2p_en'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                module = importlib.import_module("data_gen.tts.txt_processors")
            self.assertTrue(hasattr(module, "__doc__"))
        finally:
            _restore_modules("data_gen.tts.txt_processors", removed)

    def test_task_mixin_import_does_not_require_g2p_en_at_import_time(self) -> None:
        removed_task_mixin = _pop_modules("tasks.Conan.rhythm.task_mixin")
        removed_base_task = _pop_modules("tasks.Conan.base_gen_task")
        removed_txt = _pop_modules("data_gen.tts.txt_processors")
        real_import = __import__

        def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
            if name == "g2p_en" or name.startswith("g2p_en."):
                raise ModuleNotFoundError("No module named 'g2p_en'")
            return real_import(name, globals, locals, fromlist, level)

        try:
            with mock.patch("builtins.__import__", side_effect=fake_import):
                module = importlib.import_module("tasks.Conan.rhythm.task_mixin")
            self.assertTrue(hasattr(module, "RhythmConanTaskMixin"))
        finally:
            _restore_modules("data_gen.tts.txt_processors", removed_txt)
            _restore_modules("tasks.Conan.base_gen_task", removed_base_task)
            _restore_modules("tasks.Conan.rhythm.task_mixin", removed_task_mixin)

    def test_get_txt_processor_cls_raises_clear_error_for_missing_optional_dependency(self) -> None:
        missing = ModuleNotFoundError("No module named 'g2p_en'")
        missing.name = "g2p_en"
        with mock.patch.dict(txt_base_module.REGISTERED_TEXT_PROCESSORS, {}, clear=True):
            with mock.patch(
                "data_gen.tts.txt_processors.base_text_processor.importlib.import_module",
                side_effect=missing,
            ):
                with self.assertRaisesRegex(ImportError, "optional dependency 'g2p_en'"):
                    get_txt_processor_cls("en")

    def test_get_txt_processor_cls_returns_none_for_missing_processor_module(self) -> None:
        missing = ModuleNotFoundError("No module named 'data_gen.tts.txt_processors.zz'")
        missing.name = "data_gen.tts.txt_processors.zz"
        with mock.patch.dict(txt_base_module.REGISTERED_TEXT_PROCESSORS, {}, clear=True):
            with mock.patch(
                "data_gen.tts.txt_processors.base_text_processor.importlib.import_module",
                side_effect=missing,
            ):
                self.assertIsNone(get_txt_processor_cls("zz"))

    def test_build_tensorboard_falls_back_to_noop_writer_without_tensorboard(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            task = base_task_module.BaseTask.__new__(base_task_module.BaseTask)
            with mock.patch.object(base_task_module, "_TorchSummaryWriter", None):
                with mock.patch.object(base_task_module, "SummaryWriter", base_task_module._NullSummaryWriter):
                    task.build_tensorboard(tmp, "tb_logs")
            self.assertIsInstance(task.logger, base_task_module._NullSummaryWriter)
            self.assertIsNone(task.logger.add_scalar("demo", 1.0, 0))


if __name__ == "__main__":
    unittest.main()
