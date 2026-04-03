from resemblyzer import VoiceEncoder
from utils.audio import librosa_wav2spec
import shutil
import random, os, json
import traceback
from copy import deepcopy
import logging
from utils.commons.hparams import hparams
from utils.commons.indexed_datasets import IndexedDatasetBuilder
from utils.commons.multiprocess_utils import multiprocess_run_tqdm
from functools import partial
import numpy as np
from tqdm import tqdm
from utils.audio.align import get_mel2ph, mel2token_to_dur
from utils.text.text_encoder import build_token_encoder
from utils.audio.pitch.utils import f0_to_coarse
from modules.Conan.rhythm.supervision import build_item_rhythm_bundle, normalize_teacher_target_source
np.seterr(divide='ignore', invalid='ignore')


def _rhythm_teacher_kwargs_from_hparams():
    return {
        'rate_scale_min': float(hparams.get('rhythm_teacher_rate_scale_min', 0.55)),
        'rate_scale_max': float(hparams.get('rhythm_teacher_rate_scale_max', 1.95)),
        'local_rate_strength': float(hparams.get('rhythm_teacher_local_rate_strength', 0.45)),
        'segment_bias_strength': float(hparams.get('rhythm_teacher_segment_bias_strength', 0.30)),
        'pause_strength': float(hparams.get('rhythm_teacher_pause_strength', 1.10)),
        'boundary_strength': float(hparams.get('rhythm_teacher_boundary_strength', 1.50)),
        'pause_budget_ratio_cap': float(hparams.get('rhythm_teacher_pause_budget_ratio_cap', 0.80)),
        'speech_smooth_kernel': int(hparams.get('rhythm_teacher_speech_smooth_kernel', 3)),
        'pause_topk_ratio': float(hparams.get('rhythm_teacher_pause_topk_ratio', 0.30)),
    }


def _resolve_rhythm_teacher_target_source() -> str:
    return normalize_teacher_target_source(hparams.get('rhythm_teacher_target_source', 'algorithmic'))


def _parse_hubert_tokens(raw_tokens, *, item_name: str) -> list[int]:
    if isinstance(raw_tokens, str):
        pieces = [piece for piece in raw_tokens.strip().split() if piece]
    else:
        pieces = np.asarray(raw_tokens).reshape(-1).tolist()
    tokens = []
    for piece in pieces:
        try:
            tokens.append(int(float(piece)))
        except (TypeError, ValueError) as exc:
            raise BinarizationError(
                f"Invalid HuBERT token {piece!r} in item '{item_name}'."
            ) from exc
    if len(tokens) == 0:
        raise BinarizationError(f"Empty HuBERT token sequence for item '{item_name}'.")
    return tokens


def _resolve_teacher_bundle_override(item: dict, *, prefix: str | None = None) -> dict | None:
    teacher_source = _resolve_rhythm_teacher_target_source()
    if teacher_source != 'learned_offline':
        return None
    teacher_path = item.get('rhythm_teacher_npz_fn') or item.get('teacher_npz_fn')
    candidate_paths = []
    if teacher_path:
        teacher_path = str(teacher_path)
        if os.path.isabs(teacher_path):
            candidate_paths.append(teacher_path)
        else:
            candidate_paths.append(os.path.join(hparams.get('processed_data_dir', ''), teacher_path))
            teacher_target_dir = hparams.get('rhythm_teacher_target_dir', '')
            if teacher_target_dir:
                if prefix:
                    candidate_paths.append(os.path.join(teacher_target_dir, prefix, teacher_path))
                candidate_paths.append(os.path.join(teacher_target_dir, teacher_path))
    teacher_target_dir = str(hparams.get('rhythm_teacher_target_dir', '') or '').strip()
    item_name = str(item.get('item_name', '') or '')
    if teacher_target_dir and item_name:
        if prefix:
            candidate_paths.extend([
                os.path.join(teacher_target_dir, prefix, f'{item_name}.npz'),
                os.path.join(teacher_target_dir, prefix, f'{item_name}.teacher.npz'),
            ])
        candidate_paths.extend([
            os.path.join(teacher_target_dir, f'{item_name}.npz'),
            os.path.join(teacher_target_dir, f'{item_name}.teacher.npz'),
        ])
    seen = set()
    dedup_paths = []
    for path in candidate_paths:
        if not path or path in seen:
            continue
        seen.add(path)
        dedup_paths.append(path)
    for path in dedup_paths:
        if not os.path.exists(path):
            continue
        payload = np.load(path, allow_pickle=False)
        try:
            return {key: payload[key] for key in payload.files}
        finally:
            payload.close()
    searched = ', '.join(dedup_paths) if dedup_paths else '<none>'
    raise BinarizationError(
        f"Missing learned_offline teacher targets for item '{item_name}'. "
        f"Set rhythm_teacher_target_dir or rhythm_teacher_npz_fn. searched={searched}. "
        "Export them first with scripts/export_rhythm_teacher_targets.py from a teacher_offline checkpoint."
    )


class BinarizationError(Exception):
    pass

class BaseBinarizer:
    def __init__(self, processed_data_dir=None):
        if processed_data_dir is None:
            processed_data_dir = hparams['processed_data_dir']
        self.processed_data_dir = processed_data_dir
        self.binarization_args = hparams['binarization_args']
        self.items = {}
        self.item_names = []

    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)

    @property
    def train_item_names(self):
        range_ = self._convert_range(self.binarization_args['train_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def valid_item_names(self):
        range_ = self._convert_range(self.binarization_args['valid_range'])
        return self.item_names[range_[0]:range_[1]]

    @property
    def test_item_names(self):
        range_ = self._convert_range(self.binarization_args['test_range'])
        return self.item_names[range_[0]:range_[1]]

    def _convert_range(self, range_):
        if range_[1] == -1:
            range_[1] = len(self.item_names)
        return range_

    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    # def process_data(self, prefix):
    #     data_dir = hparams['binary_data_dir']
    #     builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')
    #     meta_data = list(self.meta_data(prefix))
    #     process_item = partial(self.process_item, binarization_args=self.binarization_args)
    #     ph_lengths = []
    #     mel_lengths = []
    #     total_sec = 0
    #     items = []
    #     args = [{'item': item} for item in meta_data]
    #     for item_id, item in multiprocess_run_tqdm(process_item, args, desc='Processing data'):
    #         if item is not None:
    #             items.append(item)
    #     if self.binarization_args['with_spk_embed']:
    #         args = [{'wav': item['wav']} for item in items]
    #         for item_id, spk_embed in multiprocess_run_tqdm(
    #                 self.get_spk_embed, args,
    #                 init_ctx_func=lambda wid: {'voice_encoder': VoiceEncoder().cuda()}, num_workers=4,
    #                 desc='Extracting spk embed'):
    #             items[item_id]['spk_embed'] = spk_embed

    #     for item in items:
    #         if not self.binarization_args['with_wav'] and 'wav' in item:
    #             del item['wav']
    #         builder.add_item(item)
    #         mel_lengths.append(item['len'])
    #         assert item['len'] > 0, (item['item_name'], item['txt'], item['mel2ph'])
    #         if 'ph_len' in item:
    #             ph_lengths.append(item['ph_len'])
    #         total_sec += item['sec']
    #     builder.finalize()
    #     np.save(f'{data_dir}/{prefix}_lengths.npy', mel_lengths)
    #     if len(ph_lengths) > 0:
    #         np.save(f'{data_dir}/{prefix}_ph_lengths.npy', ph_lengths)
    #     print(f"| {prefix} total duration: {total_sec:.3f}s")

    @classmethod
    def process_item(cls, item, binarization_args, prefix=None):
        item['ph_len'] = len(item['ph_token'])
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        try:
            n_bos_frames, n_eos_frames = 0, 0
            if binarization_args['with_align']:
                tg_fn = f"{hparams['processed_data_dir']}/mfa_outputs/{item_name}.TextGrid"
                item['tg_fn'] = tg_fn
                cls.process_align(tg_fn, item)
                if binarization_args['trim_eos_bos']:
                    n_bos_frames = item['dur'][0]
                    n_eos_frames = item['dur'][-1]
                    T = len(mel)
                    item['mel'] = mel[n_bos_frames:T - n_eos_frames]
                    item['mel2ph'] = item['mel2ph'][n_bos_frames:T - n_eos_frames]
                    item['mel2word'] = item['mel2word'][n_bos_frames:T - n_eos_frames]
                    item['dur'] = item['dur'][1:-1]
                    item['dur_word'] = item['dur_word'][1:-1]
                    item['len'] = item['mel'].shape[0]
                    item['wav'] = wav[n_bos_frames * hparams['hop_size']:len(wav) - n_eos_frames * hparams['hop_size']]
            if binarization_args['with_f0']:
                cls.process_pitch(item, n_bos_frames, n_eos_frames)
        except BinarizationError as e:
            print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        except Exception as e:
            traceback.print_exc()
            print(f"| Skip item. item_name: {item_name}, wav_fn: {wav_fn}")
            return None
        return item

    @classmethod
    def process_audio(cls, wav_fn, res, binarization_args):
        wav2spec_dict = librosa_wav2spec(
            wav_fn,
            fft_size=hparams['fft_size'],
            hop_size=hparams['hop_size'],
            win_length=hparams['win_size'],
            num_mels=hparams['audio_num_mel_bins'],
            fmin=hparams['fmin'],
            fmax=hparams['fmax'],
            sample_rate=hparams['audio_sample_rate'],
            loud_norm=hparams['loud_norm'])
        mel = wav2spec_dict['mel']
        wav = wav2spec_dict['wav'].astype(np.float16)
        # wav = wav2spec_dict['wav']
        if binarization_args['with_linear']:
            res['linear'] = wav2spec_dict['linear']
        res.update({'mel': mel, 'wav': wav, 'sec': len(wav) / hparams['audio_sample_rate'], 'len': mel.shape[0]})
        return wav, mel

    @staticmethod
    def process_align(tg_fn, item):
        ph = item['ph']
        mel = item['mel']
        ph_token = item['ph_token']
        if tg_fn is not None and os.path.exists(tg_fn):
            mel2ph, dur = get_mel2ph(tg_fn, ph, mel, hparams['hop_size'], hparams['audio_sample_rate'],
                                     hparams['binarization_args']['min_sil_duration'])
        else:
            raise BinarizationError(f"Align not found")
        if np.array(mel2ph).max() - 1 >= len(ph_token):
            raise BinarizationError(
                f"Align does not match: mel2ph.max() - 1: {np.array(mel2ph).max() - 1}, len(phone_encoded): {len(ph_token)}")
        item['mel2ph'] = mel2ph
        item['dur'] = dur

        ph2word = item['ph2word']
        mel2word = [ph2word[p - 1] for p in item['mel2ph']]
        item['mel2word'] = mel2word  # [T_mel]
        dur_word = mel2token_to_dur(mel2word, len(item['word_token']))
        item['dur_word'] = dur_word.tolist()  # [T_word]

    @staticmethod
    def process_pitch(item, n_bos_frames, n_eos_frames):
        wav, mel = item['wav'], item['mel']
        f0 = extract_pitch_simple(item['wav'])
        if sum(f0) == 0:
            raise BinarizationError("Empty f0")
        assert len(mel) == len(f0), (len(mel), len(f0))
        pitch_coarse = f0_to_coarse(f0)
        item['f0'] = f0
        item['pitch'] = pitch_coarse
        if hparams['binarization_args']['with_f0cwt']:
            uv, cont_lf0_lpf = get_cont_lf0(f0)
            logf0s_mean_org, logf0s_std_org = np.mean(cont_lf0_lpf), np.std(cont_lf0_lpf)
            cont_lf0_lpf_norm = (cont_lf0_lpf - logf0s_mean_org) / logf0s_std_org
            cwt_spec, scales = get_lf0_cwt(cont_lf0_lpf_norm)
            item['cwt_spec'] = cwt_spec
            item['cwt_mean'] = logf0s_mean_org
            item['cwt_std'] = logf0s_std_org

    @staticmethod
    def get_spk_embed(wav, ctx):
        return ctx['voice_encoder'].embed_utterance(wav.astype(float))

    @property
    def num_workers(self):
        return int(os.getenv('N_PROC', hparams.get('N_PROC', os.cpu_count())))
    
    def _word_encoder(self):
        fn = f"{hparams['binary_data_dir']}/word_set.json"
        word_set = []
        if self.binarization_args['reset_word_dict']:
            for word_sent in self.item2txt.values():
                word_set += [x for x in word_sent.split(' ') if x != '']
            word_set = Counter(word_set)
            total_words = sum(word_set.values())
            word_set = word_set.most_common(hparams['word_size'])
            num_unk_words = total_words - sum([x[1] for x in word_set])
            word_set = [x[0] for x in word_set]
            json.dump(word_set, open(fn, 'w'))
            print(f"| Build word set. Size: {len(word_set)}, #total words: {total_words},"
                  f" #unk_words: {num_unk_words}, word_set[:10]:, {word_set[:10]}.")
        else:
            word_set = json.load(open(fn, 'r'))
            print("| Load word set. Size: ", len(word_set), word_set[:10])
        return TokenTextEncoder(None, vocab_list=word_set, replace_oov='<UNK>')

class VCBinarizer(BaseBinarizer):
    _spker_map_cache = None
    _spker_map_cache_path = None

    def __init__(self, processed_data_dir=None):
        super().__init__(processed_data_dir=processed_data_dir)

    @classmethod
    def _get_spker_map(cls, processed_data_dir=None):
        processed_data_dir = processed_data_dir or hparams['processed_data_dir']
        spker_path = os.path.join(processed_data_dir, 'spker_set.json')
        if cls._spker_map_cache is None or cls._spker_map_cache_path != spker_path:
            if not os.path.exists(spker_path):
                raise BinarizationError(f"Speaker map not found: {spker_path}")
            with open(spker_path) as f:
                cls._spker_map_cache = json.load(f)
            cls._spker_map_cache_path = spker_path
        return cls._spker_map_cache

    @classmethod
    def _resolve_spk_id(cls, item_name: str, processed_data_dir=None) -> int:
        speaker_key = str(item_name).split('_', 1)[0]
        spker_map = cls._get_spker_map(processed_data_dir=processed_data_dir)
        if speaker_key not in spker_map:
            raise BinarizationError(
                f"Speaker '{speaker_key}' from item '{item_name}' is missing in spker_set.json."
            )
        return int(spker_map[speaker_key])

    def split_train_test_set(self, item_names):
        item_names = deepcopy(item_names)

        # first find test/validation sets
        test_item_names  = [x for x in item_names if any(ts in x for ts in hparams['test_prefixes'])]
        valid_item_names = [x for x in item_names if any(ts in x for ts in hparams['valid_prefixes'])]

        # ⚠️ Key: construct set only once
        test_set = set(test_item_names)
        valid_set = set(valid_item_names)

        # training set = neither in test set nor validation set
        train_item_names = [x for x in item_names if x not in test_set and x not in valid_set]

        logging.info(f"train {len(train_item_names)}")
        logging.info(f"valid {len(valid_item_names)}")
        logging.info(f"test  {len(test_item_names)}")
        return train_item_names, test_item_names, valid_item_names


    def load_meta_data(self):
        processed_data_dir = self.processed_data_dir
        items_list = json.load(open(f"{processed_data_dir}/metadata_vctk_librittsr_gt.json"))
        for r in tqdm(items_list, desc='Loading meta data.'):
            item_name = r['item_name']
            self.items[item_name] = r
            self.item_names.append(item_name)
        if self.binarization_args['shuffle']:
            random.seed(1234)
            random.shuffle(self.item_names)
        self._train_item_names, self._test_item_names,self._valid_item_names = self.split_train_test_set(self.item_names)

    @property
    def train_item_names(self):
        return self._train_item_names

    @property
    def valid_item_names(self):
        return self._valid_item_names

    @property
    def test_item_names(self):
        return self._test_item_names

    def process(self):
        self.load_meta_data()
        self.word_encoder = None
        os.makedirs(hparams['binary_data_dir'], exist_ok=True)
        # ph_set_fn = f"{hparams['binary_data_dir']}/phone_set.json"
        # if not os.path.exists(ph_set_fn):
        #     oldp=os.path.join(hparams["processed_data_dir"], "phone_set.json")
        #     newp=hparams['binary_data_dir']
            # shutil.copy(oldp,newp)
        self.process_data('valid')
        self.process_data('test')
        self.process_data('train')


    def meta_data(self, prefix):
        if prefix == 'valid':
            item_names = self.valid_item_names
        elif prefix == 'test':
            item_names = self.test_item_names
        else:
            item_names = self.train_item_names
        for item_name in item_names:
            yield self.items[item_name]

    def process_data(self, prefix):
        data_dir = hparams['binary_data_dir']
        builder = IndexedDatasetBuilder(f'{data_dir}/{prefix}')

        lengths, spk_ids = [], []
        total_sec = 0.0
        meta_data = list(self.meta_data(prefix))
        process_item = partial(self.process_item,
                            binarization_args=self.binarization_args,
                            prefix=prefix,
                            processed_data_dir=self.processed_data_dir)

        args = [{'item': it} for it in meta_data]

        if self.binarization_args['with_spk_embed']:
            logging.warning(
                "with_spk_embed=True, but inline speaker embedding extraction is disabled in "
                "data_gen/conan_binarizer.py. Precompute spk_embed separately if the downstream model uses it."
            )

        for item_id, item in multiprocess_run_tqdm(
                process_item, args, desc=f'Processing {prefix}'):
            # item['spk_embed'] = voice_encoder.embed_utterance(item['wav']) \
            #     if self.binarization_args['with_spk_embed'] else None
            if item is None:
                continue
            builder.add_item(item)          # spk_id is already included in item
            lengths.append(item['len'])
            spk_ids.append(item['spk_id'])  # 👈 collect spk_id in consistent order
            total_sec += item['sec']


        builder.finalize()
        np.save(f'{data_dir}/{prefix}_lengths.npy', np.array(lengths,  np.int32))
        np.save(f'{data_dir}/{prefix}_spk_ids.npy', np.array(spk_ids, np.int32))  # ✅ newly added
        print(f"| {prefix} total duration: {total_sec:.2f}s, #items: {len(lengths)}")


    # @classmethod
    # def process_item(cls, item, binarization_args):
    #     item_name = item['item_name']
    #     wav_fn = item['wav_fn']
    #     wav, mel = cls.process_audio(wav_fn, item, binarization_args)
    #     # item['spk_embed'] = np.load(wav_fn.replace(".wav", "_spk.npy"))
    #     try:
    #         cls.process_pitch(item, 0, 0)
    #         cls.process_align(item["tg_fn"], item)
    #     except BinarizationError as e:
    #         print(f"| Skip item ({e}). item_name: {item_name}, wav_fn: {wav_fn}")
    #         return None
    #     return item
    
class ConanBinarizer(VCBinarizer):
    # ph_encoder = build_token_encoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
    @classmethod
    def process_item(cls, item, binarization_args, prefix=None, processed_data_dir=None):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        mel=item['mel']
        wav=item['wav']
        content = _parse_hubert_tokens(item.get('hubert'), item_name=item_name)
        
        # item["ph_token"] = cls.ph_encoder.encode(' '.join(item["ph"]))
        item["spk_id"] = cls._resolve_spk_id(item["item_name"], processed_data_dir=processed_data_dir)
        # item['txt']=" ".join(item['txt'])
        
        # try:
        require_f0 = bool(binarization_args.get('with_f0', hparams.get('use_pitch_embed', False)))
        if require_f0:
            f0_path = os.path.join(
                os.path.dirname(wav_fn) + "_f0",
                os.path.basename(wav_fn).replace(".wav", "_f0.npy")
            )
            if not os.path.exists(f0_path):
                raise BinarizationError(f"Missing f0 file for {item_name}: {f0_path}")
            f0 = np.load(f0_path).reshape(-1)[:mel.shape[0]]
            min_length = min(len(content), len(mel), len(f0))
            item["f0"] = f0 = f0[:min_length]
        else:
            min_length = min(len(content), len(mel))
        if min_length <= 0:
            raise BinarizationError(f"Empty aligned sample after trimming: {item_name}")
        item['mel'] = mel = mel[:min_length]
        item['wav'] = wav = wav[:min_length * hparams['hop_size']]
        item['hubert'] = content = np.asarray(content[:min_length], dtype=np.int32)
        item['len'] = min_length
        if binarization_args.get('with_rhythm_cache', hparams.get('rhythm_enable_v2', False)):
            need_teacher_bundle = bool(hparams.get('rhythm_binarize_teacher_targets', False)) or str(hparams.get('rhythm_binarize_retimed_mel_source', 'guidance') or 'guidance').strip().lower() == 'teacher'
            teacher_bundle_override = _resolve_teacher_bundle_override(item, prefix=prefix) if need_teacher_bundle else None
            item.update(
                build_item_rhythm_bundle(
                    content_tokens=item['hubert'],
                    mel=item['mel'],
                    silent_token=hparams.get('silent_token', 57),
                    separator_aware=bool(hparams.get('rhythm_separator_aware', True)),
                    tail_open_units=int(hparams.get('rhythm_tail_open_units', 1)),
                    trace_bins=int(hparams.get('rhythm_trace_bins', 24)),
                    trace_horizon=float(hparams.get('rhythm_trace_horizon', 0.35)),
                    trace_smooth_kernel=int(hparams.get('rhythm_trace_smooth_kernel', 5)),
                    slow_topk=int(hparams.get('rhythm_slow_topk', 6)),
                    selector_cell_size=int(hparams.get('rhythm_selector_cell_size', 3)),
                    source_phrase_threshold=float(hparams.get('rhythm_source_phrase_threshold', 0.55)),
                    include_self_targets=bool(hparams.get('rhythm_binarize_self_targets', True)),
                    include_teacher_targets=bool(hparams.get('rhythm_binarize_teacher_targets', False)),
                    include_retimed_mel_target=bool(hparams.get('rhythm_binarize_retimed_mel_targets', False)),
                    retimed_mel_target_source=str(hparams.get('rhythm_binarize_retimed_mel_source', 'guidance')),
                    retimed_pause_frame_weight=float(hparams.get('rhythm_retimed_pause_frame_weight', 0.20)),
                    retimed_stretch_weight_min=float(hparams.get('rhythm_retimed_stretch_weight_min', 0.35)),
                    teacher_kwargs=_rhythm_teacher_kwargs_from_hparams(),
                    teacher_target_source=_resolve_rhythm_teacher_target_source(),
                    teacher_bundle_override=teacher_bundle_override,
                )
            )
        # print(f'f0_length: {f0.shape}, mel_length: {mel.shape},wav_length: {wav.shape}, content_length: {content.shape}, item_name: {item_name}')
        # except:
        #     # parselmouth
            # time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
            # f0_min = 80
            # f0_max = 800
            # if hparams['hop_size'] == 128:
            #     pad_size = 4
            # elif hparams['hop_size'] == 256:
            #     pad_size = 2
            # else:
            #     assert False
            # import parselmouth
            # f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
            #     time_step=time_step / 1000, voicing_threshold=0.6,
            #     pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            # lpad = pad_size * 2
            # rpad = len(mel) - len(f0) - lpad
            # f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
            # delta_l = len(mel) - len(f0)
            # assert np.abs(delta_l) <= 8
            # if delta_l > 0:
            #     f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
            # f0 = f0[:len(mel)]
            # item["f0"] = f0
        
        # cls.process_align(item["ph_durs"], mel, item)
            
        return item
    
    @staticmethod
    def process_align(ph_durs, mel, item, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        item['mel2ph'] = mel2ph



class EmformerBinarizer(VCBinarizer):
    # difference between EmformerBinarizer and ConanBinarizer: No f0 information needed
    # ph_encoder = build_token_encoder(os.path.join(hparams["processed_data_dir"], "phone_set.json"))
    @classmethod
    def process_item(cls, item, binarization_args, prefix=None, processed_data_dir=None):
        item_name = item['item_name']
        wav_fn = item['wav_fn']
        wav, mel = cls.process_audio(wav_fn, item, binarization_args)
        mel=item['mel']
        wav=item['wav']
        content = _parse_hubert_tokens(item.get('hubert'), item_name=item_name)
        
        # item["ph_token"] = cls.ph_encoder.encode(' '.join(item["ph"]))
        item["spk_id"] = cls._resolve_spk_id(item["item_name"], processed_data_dir=processed_data_dir)
        # item['txt']=" ".join(item['txt'])
        
        min_length = min(len(content), len(mel))
        if min_length <= 0:
            raise BinarizationError(f"Empty aligned sample after trimming: {item_name}")
        item['mel'] = mel = mel[:min_length]
        item['wav'] = wav = wav[:min_length * hparams['hop_size']]
        item['hubert'] = content = np.asarray(content[:min_length], dtype=np.int32)
        item['len'] = min_length
        if binarization_args.get('with_rhythm_cache', hparams.get('rhythm_enable_v2', False)):
            need_teacher_bundle = bool(hparams.get('rhythm_binarize_teacher_targets', False)) or str(hparams.get('rhythm_binarize_retimed_mel_source', 'guidance') or 'guidance').strip().lower() == 'teacher'
            teacher_bundle_override = _resolve_teacher_bundle_override(item, prefix=prefix) if need_teacher_bundle else None
            item.update(
                build_item_rhythm_bundle(
                    content_tokens=item['hubert'],
                    mel=item['mel'],
                    silent_token=hparams.get('silent_token', 57),
                    separator_aware=bool(hparams.get('rhythm_separator_aware', True)),
                    tail_open_units=int(hparams.get('rhythm_tail_open_units', 1)),
                    trace_bins=int(hparams.get('rhythm_trace_bins', 24)),
                    trace_horizon=float(hparams.get('rhythm_trace_horizon', 0.35)),
                    trace_smooth_kernel=int(hparams.get('rhythm_trace_smooth_kernel', 5)),
                    slow_topk=int(hparams.get('rhythm_slow_topk', 6)),
                    selector_cell_size=int(hparams.get('rhythm_selector_cell_size', 3)),
                    source_phrase_threshold=float(hparams.get('rhythm_source_phrase_threshold', 0.55)),
                    include_self_targets=bool(hparams.get('rhythm_binarize_self_targets', True)),
                    include_teacher_targets=bool(hparams.get('rhythm_binarize_teacher_targets', False)),
                    include_retimed_mel_target=bool(hparams.get('rhythm_binarize_retimed_mel_targets', False)),
                    retimed_mel_target_source=str(hparams.get('rhythm_binarize_retimed_mel_source', 'guidance')),
                    retimed_pause_frame_weight=float(hparams.get('rhythm_retimed_pause_frame_weight', 0.20)),
                    retimed_stretch_weight_min=float(hparams.get('rhythm_retimed_stretch_weight_min', 0.35)),
                    teacher_kwargs=_rhythm_teacher_kwargs_from_hparams(),
                    teacher_target_source=_resolve_rhythm_teacher_target_source(),
                    teacher_bundle_override=teacher_bundle_override,
                )
            )
        # print(f'f0_length: {f0.shape}, mel_length: {mel.shape},wav_length: {wav.shape}, content_length: {content.shape}, item_name: {item_name}')
        # except:
        #     # parselmouth
            # time_step = hparams['hop_size'] / hparams['audio_sample_rate'] * 1000
            # f0_min = 80
            # f0_max = 800
            # if hparams['hop_size'] == 128:
            #     pad_size = 4
            # elif hparams['hop_size'] == 256:
            #     pad_size = 2
            # else:
            #     assert False
            # import parselmouth
            # f0 = parselmouth.Sound(wav, hparams['audio_sample_rate']).to_pitch_ac(
            #     time_step=time_step / 1000, voicing_threshold=0.6,
            #     pitch_floor=f0_min, pitch_ceiling=f0_max).selected_array['frequency']
            # lpad = pad_size * 2
            # rpad = len(mel) - len(f0) - lpad
            # f0 = np.pad(f0, [[lpad, rpad]], mode='constant')
            # delta_l = len(mel) - len(f0)
            # assert np.abs(delta_l) <= 8
            # if delta_l > 0:
            #     f0 = np.concatenate([f0, [f0[-1]] * delta_l], 0)
            # f0 = f0[:len(mel)]
            # item["f0"] = f0
        
        # cls.process_align(item["ph_durs"], mel, item)
            
        return item
    
    @staticmethod
    def process_align(ph_durs, mel, item, hop_size=hparams['hop_size'], audio_sample_rate=hparams['audio_sample_rate']):
        mel2ph = np.zeros([mel.shape[0]], int)
        startTime = 0

        for i_ph in range(len(ph_durs)):
            start_frame = int(startTime * audio_sample_rate / hop_size + 0.5)
            end_frame = int((startTime + ph_durs[i_ph]) * audio_sample_rate / hop_size + 0.5)
            mel2ph[start_frame:end_frame] = i_ph + 1
            startTime = startTime + ph_durs[i_ph]

        item['mel2ph'] = mel2ph
