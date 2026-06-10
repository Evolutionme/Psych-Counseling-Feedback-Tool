"""Run speaker diarization and export speaker_0/speaker_1 sample audio.

This script is meant for manual role assignment:
listen to outputs/speaker_samples/speaker_0_sample.wav and speaker_1_sample.wav,
then decide which one is the client and which one is the therapist.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import torch

from .utils import dump_json, ensure_dir


def patch_hf_hub_download_compat():
    import huggingface_hub

    original = huggingface_hub.hf_hub_download
    if getattr(original, "_codex_compat", False):
        return

    def wrapped(*args, **kwargs):
        if "use_auth_token" in kwargs and "token" not in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        else:
            kwargs.pop("use_auth_token", None)
        return original(*args, **kwargs)

    wrapped._codex_compat = True
    huggingface_hub.hf_hub_download = wrapped
    try:
        import huggingface_hub.file_download as file_download

        file_download.hf_hub_download = wrapped
    except Exception:
        pass


def patch_torch_load_compat():
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    try:
        import torch.serialization as serialization
        from torch.torch_version import TorchVersion

        serialization.add_safe_globals([TorchVersion])
    except Exception:
        pass


def patch_numpy_compat():
    try:
        import numpy as np
    except Exception:
        return

    aliases = {
        "NaN": np.nan,
        "Inf": np.inf,
        "Infinity": np.inf,
        "infty": np.inf,
        "NINF": -np.inf,
        "PINF": np.inf,
        "float_": np.float64,
        "complex_": np.complex128,
        "string_": np.bytes_,
        "unicode_": np.str_,
    }
    for name, value in aliases.items():
        if not hasattr(np, name):
            setattr(np, name, value)


def normalize_speaker_labels(segments):
    first_seen = []
    for seg in sorted(segments, key=lambda x: (x["start"], x["end"])):
        speaker = seg["source_speaker"]
        if speaker not in first_seen:
            first_seen.append(speaker)

    mapping = {speaker: "speaker_{}".format(idx) for idx, speaker in enumerate(first_seen)}
    normalized = []
    for seg in segments:
        item = dict(seg)
        item["speaker"] = mapping[item["source_speaker"]]
        normalized.append(item)
    return normalized, mapping


def diarize_with_pyannote(args):
    patch_numpy_compat()
    patch_torch_load_compat()
    patch_hf_hub_download_compat()
    from pyannote.audio import Pipeline
    from huggingface_hub import login

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "pyannote.audio diarization needs a Hugging Face token. "
            "Set HF_TOKEN or pass --hf-token."
        )

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_TOKEN"] = token
    try:
        login(token=token)
    except Exception:
        pass

    try:
        print("[stage] Loading pyannote pipeline: {}".format(args.pyannote_model), flush=True)
        pipeline = Pipeline.from_pretrained(args.pyannote_model)
    except ImportError as exc:
        message = str(exc)
        if "flair" in message.lower():
            raise SystemExit(
                "pyannote loaded the diarization pipeline, but your environment is missing flair.\n"
                "Install it with:\n"
                "  pip install flair\n"
                "Then rerun the same command.\n"
                "Original error: {}".format(message)
            )
        if "numba" in message.lower():
            raise SystemExit(
                "pyannote loaded the diarization pipeline, but your environment is missing numba.\n"
                "For Python 3.11, install:\n"
                "  pip install numba==0.57.0\n"
                "For Python 3.7, install:\n"
                "  pip install numba==0.56.4\n"
                "Then rerun the same command.\n"
                "Original error: {}".format(message)
            )
        if "k2" in message.lower():
            raise SystemExit(
                "pyannote loaded the diarization pipeline, but your environment is missing k2.\n"
                "On Windows, install the precompiled k2 wheel that matches your PyTorch version.\n"
                "See: https://k2-fsa.github.io/k2/installation/from_wheels.html\n"
                "If you only need a CPU workflow, use the Windows CPU wheel.\n"
                "If you are staying on CUDA, install the matching CUDA wheel or build from source.\n"
                "Original error: {}".format(message)
            )
        raise
    if pipeline is None:
        raise SystemExit(
            "Could not load the pyannote diarization pipeline.\n"
            "Please open these Hugging Face pages in your browser and accept the access conditions:\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  - https://huggingface.co/pyannote/segmentation-3.0\n"
            "Then run the command again with the same HF token."
        )

    if args.device:
        print("[stage] Moving pyannote pipeline to device: {}".format(args.device), flush=True)
        pipeline.to(torch.device(args.device))

    kwargs = {}
    if args.num_speakers:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers:
        kwargs["max_speakers"] = args.max_speakers

    print("[stage] Running pyannote diarization on audio...", flush=True)
    output = pipeline(str(Path(args.audio).resolve()), **kwargs)
    print("[stage] Pyannote diarization inference completed.", flush=True)
    diarization = getattr(output, "speaker_diarization", output)
    return segments_from_pyannote_annotation(diarization)


def segments_from_pyannote_annotation(annotation):
    segments = []
    if hasattr(annotation, "itertracks"):
        iterator = annotation.itertracks(yield_label=True)
        for turn, _, speaker in iterator:
            segments.append(
                {
                    "source_speaker": str(speaker),
                    "start": float(turn.start),
                    "end": float(turn.end),
                }
            )
        return segments

    for item in annotation:
        if len(item) == 2:
            turn, speaker = item
        else:
            turn, _, speaker = item
        segments.append(
            {
                "source_speaker": str(speaker),
                "start": float(turn.start),
                "end": float(turn.end),
            }
        )
    return segments


def diarize_with_funasr(args):
    from .predict_from_audio import normalize_asr_segments, run_funasr

    print("[stage] Loading FunASR models and running speaker diarization...", flush=True)
    asr_result = run_funasr(args)
    print("[stage] FunASR model.generate returned.", flush=True)
    turns = normalize_asr_segments(asr_result)
    raw_segments = []
    for turn in turns:
        start = int(turn.get("start_ms", 0)) / 1000.0
        end = int(turn.get("end_ms", 0)) / 1000.0
        speaker = str(turn.get("speaker", "unknown"))
        if end <= start:
            continue
        raw_segments.append({"source_speaker": speaker, "start": start, "end": end})
    return raw_segments


def diarize_with_whisperx(args):
    patch_numpy_compat()
    patch_torch_load_compat()
    patch_hf_hub_download_compat()
    import whisperx
    from whisperx.diarize import DiarizationPipeline
    from huggingface_hub import login

    token = args.hf_token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
    if not token:
        raise SystemExit(
            "WhisperX diarization uses pyannote under the hood and needs a Hugging Face token. "
            "Set HF_TOKEN or pass --hf-token."
        )

    os.environ["HF_TOKEN"] = token
    os.environ["HUGGINGFACE_TOKEN"] = token
    try:
        login(token=token)
    except Exception:
        pass

    try:
        pipeline = DiarizationPipeline(device=args.device)
    except TypeError:
        pipeline = DiarizationPipeline(use_auth_token=token, device=args.device)
    if pipeline is None:
        raise SystemExit(
            "Could not load the WhisperX diarization pipeline.\n"
            "WhisperX also depends on gated pyannote models, so you still need to accept:\n"
            "  - https://huggingface.co/pyannote/speaker-diarization-3.1\n"
            "  - https://huggingface.co/pyannote/segmentation-3.0"
        )

    audio = whisperx.load_audio(str(Path(args.audio).resolve()))
    kwargs = {}
    if args.num_speakers:
        kwargs["num_speakers"] = args.num_speakers
    if args.min_speakers:
        kwargs["min_speakers"] = args.min_speakers
    if args.max_speakers:
        kwargs["max_speakers"] = args.max_speakers
    diarize_segments = pipeline(audio, **kwargs)

    records = diarize_segments.to_dict("records")
    segments = []
    for row in records:
        segment = row.get("segment")
        start = row.get("start", getattr(segment, "start", None))
        end = row.get("end", getattr(segment, "end", None))
        speaker = row.get("speaker", row.get("label", row.get("source_speaker", "")))
        if start is None or end is None or speaker == "":
            continue
        segments.append(
            {
                "source_speaker": str(speaker),
                "start": float(start),
                "end": float(end),
            }
        )
    return segments


def load_audio(path):
    try:
        import torchaudio

        waveform, sample_rate = torchaudio.load(str(path))
        waveform = waveform.mean(dim=0).cpu().numpy()
        return waveform, int(sample_rate)
    except Exception as exc:
        try:
            import soundfile as sf

            waveform, sample_rate = sf.read(str(path), dtype="float32")
        except Exception as sf_exc:
            raise RuntimeError(
                "Failed to load audio with torchaudio and soundfile. "
                "For mp3/m4a, install ffmpeg or convert to wav first.\n"
                "torchaudio error: {}\nsoundfile error: {}".format(exc, sf_exc)
            )
        if getattr(waveform, "ndim", 1) > 1:
            waveform = waveform.mean(axis=1)
        return waveform, int(sample_rate)


def silence(sample_rate, gap_seconds):
    import numpy as np

    return np.zeros(int(sample_rate * gap_seconds), dtype="float32")


def export_speaker_audio(audio_path, output_dir, segments, sample_seconds, min_segment_seconds, gap_seconds):
    import numpy as np
    import soundfile as sf

    print("[stage] Loading audio and exporting speaker preview samples...", flush=True)
    output_dir = ensure_dir(output_dir)
    waveform, sample_rate = load_audio(audio_path)
    by_speaker = defaultdict(list)
    for seg in sorted(segments, key=lambda x: (x["start"], x["end"])):
        duration = seg["end"] - seg["start"]
        if duration >= min_segment_seconds:
            by_speaker[seg["speaker"]].append(seg)

    sample_files = {}
    for speaker, speaker_segments in by_speaker.items():
        chunks = []
        total = 0.0
        for seg in speaker_segments:
            if total >= sample_seconds:
                break
            start_sample = max(0, int(seg["start"] * sample_rate))
            end_sample = min(len(waveform), int(seg["end"] * sample_rate))
            if end_sample <= start_sample:
                continue
            remaining = sample_seconds - total
            max_end = start_sample + int(remaining * sample_rate)
            chunk = waveform[start_sample : min(end_sample, max_end)]
            chunks.append(chunk.astype("float32"))
            total += len(chunk) / sample_rate
            chunks.append(silence(sample_rate, gap_seconds))

        if not chunks:
            continue
        sample = np.concatenate(chunks)
        output_path = Path(output_dir) / "{}_sample.wav".format(speaker)
        sf.write(str(output_path), sample, sample_rate)
        sample_files[speaker] = str(output_path.resolve())
    print("[stage] Exported {} speaker preview sample(s).".format(len(sample_files)), flush=True)
    return sample_files


def write_rttm(path, audio_id, segments):
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        for seg in segments:
            start = seg["start"]
            duration = max(0.0, seg["end"] - seg["start"])
            f.write(
                "SPEAKER {audio_id} 1 {start:.3f} {duration:.3f} <NA> <NA> {speaker} <NA> <NA>\n".format(
                    audio_id=audio_id,
                    start=start,
                    duration=duration,
                    speaker=seg["speaker"],
                )
            )


def main():
    parser = argparse.ArgumentParser(description="Export speaker samples for manual role assignment.")
    parser.add_argument("--audio", required=True, help="Input audio file")
    parser.add_argument("--backend", choices=["funasr", "pyannote", "whisperx"], default="pyannote")
    parser.add_argument("--output-dir", default="outputs/speaker_samples")
    parser.add_argument("--manifest", default="", help="Output JSON manifest path")
    parser.add_argument("--rttm", default="", help="Optional RTTM output path")
    parser.add_argument("--asr-model", default="iic/speech_seaco_paraformer_large_asr_nat-zh-cn-16k-common-vocab8404-pytorch")
    parser.add_argument("--vad-model", default="iic/speech_fsmn_vad_zh-cn-16k-common-pytorch")
    parser.add_argument("--punc-model", default="iic/punc_ct-transformer_cn-en-common-vocab471067-large")
    parser.add_argument("--spk-model", default="iic/speech_campplus_sv_zh-cn_16k-common")
    parser.add_argument("--asr-device", default="")
    parser.add_argument("--hub", default="ms")
    parser.add_argument("--max-single-segment-time", type=int, default=60000)
    parser.add_argument("--batch-size-s", type=int, default=300)
    parser.add_argument("--hotword", default="")
    parser.add_argument("--hf-token", default="", help="Hugging Face token, or set HF_TOKEN")
    parser.add_argument(
        "--pyannote-model",
        default="pyannote/speaker-diarization-3.1",
        help="pyannote pipeline id",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-speakers", type=int, default=2)
    parser.add_argument("--min-speakers", type=int, default=0)
    parser.add_argument("--max-speakers", type=int, default=0)
    parser.add_argument("--sample-seconds", type=float, default=15.0)
    parser.add_argument("--min-segment-seconds", type=float, default=0.8)
    parser.add_argument("--gap-seconds", type=float, default=0.25)
    args = parser.parse_args()

    audio_path = Path(args.audio)
    audio_id = audio_path.stem
    output_dir = Path(args.output_dir) / audio_id
    manifest_path = args.manifest or str(output_dir / "speaker_samples_manifest.json")
    rttm_path = args.rttm or str(output_dir / "{}.rttm".format(audio_id))

    if args.backend == "funasr":
        if not args.asr_device:
            args.asr_device = args.device
        raw_segments = diarize_with_funasr(args)
    elif args.backend == "whisperx":
        raw_segments = diarize_with_whisperx(args)
    else:
        raw_segments = diarize_with_pyannote(args)
    print("[stage] Raw diarization segment count: {}".format(len(raw_segments)), flush=True)

    if not raw_segments:
        raise SystemExit("No speaker segments were detected.")

    segments, source_mapping = normalize_speaker_labels(raw_segments)
    print("[stage] Normalized speaker count: {}".format(len(source_mapping)), flush=True)
    sample_files = export_speaker_audio(
        audio_path,
        output_dir,
        segments,
        args.sample_seconds,
        args.min_segment_seconds,
        args.gap_seconds,
    )
    print("[stage] Writing RTTM: {}".format(rttm_path), flush=True)
    write_rttm(rttm_path, audio_id, segments)

    speakers = sorted(set(seg["speaker"] for seg in segments))
    manifest = {
        "audio_id": audio_id,
        "audio_file": str(audio_path.resolve()),
        "backend": args.backend,
        "source_speaker_mapping": source_mapping,
        "sample_files": sample_files,
        "rttm": str(Path(rttm_path).resolve()),
        "segments": segments,
        "role_assignment_template": {
            "client_speaker": speakers[0] if speakers else "",
            "therapist_speaker": speakers[1] if len(speakers) > 1 else "",
        },
        "next_step_example": (
            "python -m psych_cte.predict_from_audio --audio {audio} "
            "--segment-checkpoint checkpoints\\segment_cte.pt "
            "--client-speaker {client} --therapist-speaker {therapist}"
        ).format(
            audio=str(audio_path),
            client=speakers[0] if speakers else "speaker_0",
            therapist=speakers[1] if len(speakers) > 1 else "speaker_1",
        ),
    }
    print("[stage] Writing speaker manifest: {}".format(manifest_path), flush=True)
    dump_json(manifest_path, manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
