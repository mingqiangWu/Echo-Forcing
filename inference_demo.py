import os
import queue
import re
import shutil
import threading
import time
import gc
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import torch
from einops import rearrange
from omegaconf import OmegaConf
from torchvision.io import write_video

from pipeline import CausalDiffusionInferencePipeline, CausalInferencePipeline
from utils.interactive import add_subtitles, attach_interactive_config, parse_total_duration
from utils.memory import DynamicSwapInstaller, get_cuda_free_memory_gb, gpu
from utils.misc import set_seed


REPO_ROOT = Path(__file__).resolve().parent
MODE_MARKERS = {
    "smooth": "",
    "hardcut": "#",
    "recall": "@",
}


@dataclass
class PromptSegment:
    prompt: str
    duration_seconds: int
    mode: str = "smooth"
    subtitle: str = ""


def sanitize_filename(text: str, max_length: int = 64) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", text)
    sanitized = re.sub(r"[\s_]+", "_", sanitized).strip("_.")
    return (sanitized or "segment")[:max_length]


def segment_to_prompt(segment: PromptSegment) -> str:
    marker = MODE_MARKERS.get(segment.mode, "")
    prompt = segment.prompt.strip()
    return f"{prompt}[{int(segment.duration_seconds)}s{marker}]"


def build_interactive_prompt(segments: List[PromptSegment], use_subtitles: bool = False) -> str:
    prompt_part = " | ".join(segment_to_prompt(segment) for segment in segments)
    if not use_subtitles:
        return prompt_part

    subtitles = [segment.subtitle.strip() for segment in segments]
    if not any(subtitles):
        return prompt_part
    return f"{prompt_part}; {' | '.join(subtitles)}"


def calculate_latent_frames_from_duration(
    total_duration_seconds: float,
    fps: float,
    temporal_compression: int,
    num_frame_per_block: int,
    independent_first_frame: bool,
    has_initial_latent: bool,
) -> int:
    import math

    total_output_frames = int(total_duration_seconds * fps)
    if has_initial_latent:
        base_latent_frames = (total_output_frames - 1) // temporal_compression
        latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
        return max(latent_frames, num_frame_per_block)

    if independent_first_frame:
        base_latent_frames = (total_output_frames - 1) // temporal_compression
        if base_latent_frames == 0:
            return 1
        return 1 + math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block

    base_latent_frames = total_output_frames // temporal_compression
    latent_frames = math.ceil(base_latent_frames / num_frame_per_block) * num_frame_per_block
    return max(latent_frames, num_frame_per_block)


class EchoForcingDemoEngine:
    def __init__(
        self,
        config_path: str = "configs/self_forcing_dmd.yaml",
        checkpoint_path: str = "checkpoints/self_forcing_dmd.pt",
        output_root: str = ".gradio_cache",
        seed: int = 0,
        use_ema: bool = True,
    ):
        self.config_path = REPO_ROOT / config_path
        self.checkpoint_path = REPO_ROOT / checkpoint_path
        self.output_root = REPO_ROOT / output_root
        self.seed = int(seed)
        self.use_ema = bool(use_ema)
        self.pipeline = None
        self.config = None
        self.device = None
        self.low_memory = False
        self._lock = threading.Lock()
        self._vae_decode_lock = threading.Lock()
        self.live_segment_decode = os.environ.get("ECHO_LIVE_SEGMENT_DECODE", "0") == "1"
        self.output_root.mkdir(parents=True, exist_ok=True)

    def load(self):
        if self.pipeline is not None:
            return self.pipeline, self.config, self.device

        if not torch.cuda.is_available():
            raise RuntimeError("Echo-Forcing demo requires CUDA, but torch.cuda.is_available() is false.")

        torch.set_grad_enabled(False)
        self.device = torch.device("cuda")
        set_seed(self.seed)

        config = OmegaConf.load(self.config_path)
        default_config = OmegaConf.load(REPO_ROOT / "configs/default_config.yaml")
        config = OmegaConf.merge(default_config, config)
        config = attach_interactive_config(config)
        self.config = config

        if hasattr(config, "denoising_step_list"):
            pipeline = CausalInferencePipeline(config, device=self.device)
        else:
            pipeline = CausalDiffusionInferencePipeline(config, device=self.device)

        state_dict = torch.load(self.checkpoint_path, map_location="cpu")
        key = "generator_ema" if self.use_ema else "generator"
        generator_state_dict = state_dict[key]
        renamed_state_dict = {
            name.replace("_fsdp_wrapped_module.", ""): value
            for name, value in generator_state_dict.items()
        }
        pipeline.generator.load_state_dict(renamed_state_dict)

        pipeline = pipeline.to(dtype=torch.bfloat16)
        inference_cfg = getattr(config, "inference", OmegaConf.create({}))
        threshold = float(getattr(inference_cfg, "low_memory_threshold_gb", 40))
        self.low_memory = get_cuda_free_memory_gb(gpu) < threshold
        if self.low_memory:
            DynamicSwapInstaller.install_model(pipeline.text_encoder, device=gpu)
        else:
            pipeline.text_encoder.to(device=gpu)
        pipeline.generator.to(device=gpu)
        pipeline.vae.to(device=gpu)
        pipeline.eval()

        self.pipeline = pipeline
        return self.pipeline, self.config, self.device

    def new_run_dir(self) -> Path:
        run_dir = self.output_root / f"run_{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000}"
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    def _write_video_tensor(self, video: torch.Tensor, output_path: Path, fps: int) -> str:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frames = video[0].clamp(0, 255).to(torch.uint8)
        write_video(str(output_path), frames, fps=fps)
        return str(output_path)

    def _decode_latents_to_file(
        self,
        latents: torch.Tensor,
        output_path: Path,
        fps: int,
        subtitle: str = "",
        duration_seconds: Optional[float] = None,
    ) -> str:
        pipeline, _, device = self.load()
        with self._vae_decode_lock:
            latents = latents.to(device=device, dtype=torch.bfloat16)
            decoded = pipeline.vae.decode_to_pixel(latents, use_cache=False)
            video = 255.0 * rearrange(decoded, "b t c h w -> b t h w c").cpu()
            pipeline.vae.model.clear_cache()

        if subtitle:
            video = add_subtitles(
                video,
                [subtitle],
                fps=float(fps),
                time_durations=[duration_seconds] if duration_seconds else None,
            )
        return self._write_video_tensor(video, output_path, fps)

    def generate(
        self,
        segments: List[PromptSegment],
        use_subtitles: bool = False,
        run_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> Dict:
        if not segments:
            raise ValueError("Please submit at least one prompt segment before generation.")

        with self._lock:
            if progress_callback is not None:
                progress_callback({
                    "type": "phase",
                    "percent": 5,
                    "label": "Loading model",
                    "message": "🔄 Loading Echo-Forcing model and preparing CUDA memory...",
                })
            pipeline, config, device = self.load()
            if progress_callback is not None:
                progress_callback({
                    "type": "phase",
                    "percent": 12,
                    "label": "Model ready",
                    "message": "✅ Model loaded. Preparing prompt, noise, and generation plan...",
                })
            run_dir = run_dir or self.new_run_dir()
            prompt_text = build_interactive_prompt(segments, use_subtitles=use_subtitles)
            total_duration = parse_total_duration(prompt_text, config.interactive)
            if total_duration is None:
                total_duration = sum(segment.duration_seconds for segment in segments)

            inference_cfg = getattr(config, "inference", OmegaConf.create({}))
            fps = int(round(float(getattr(inference_cfg, "generation_fps", config.interactive.scene_prompt.fps))))
            temporal_compression = int(getattr(inference_cfg, "temporal_compression", 4))
            num_frame_per_block = int(getattr(pipeline, "num_frame_per_block", getattr(config, "num_frame_per_block", 1)))
            independent_first_frame = bool(
                getattr(pipeline, "independent_first_frame", getattr(config, "independent_first_frame", False))
            )
            num_latent_frames = calculate_latent_frames_from_duration(
                total_duration,
                fps,
                temporal_compression,
                num_frame_per_block,
                independent_first_frame,
                has_initial_latent=False,
            )

            set_seed(self.seed)
            sampled_noise = torch.randn(
                [1, num_latent_frames, 16, 60, 104],
                device=device,
                dtype=torch.bfloat16,
            )
            if progress_callback is not None:
                progress_callback({
                    "type": "phase",
                    "percent": 15,
                    "label": "Generating",
                    "message": "🚀 Diffusion started. The first segment is now being generated...",
                })

            segment_paths: List[str] = []
            scene_events: List[Dict] = []
            decode_queue: "queue.Queue[Optional[Dict]]" = queue.Queue()
            decode_errors: List[BaseException] = []

            def decode_worker():
                while True:
                    event = decode_queue.get()
                    try:
                        if event is None:
                            return
                        decode_event(event)
                    except BaseException as exc:
                        decode_errors.append(exc)
                    finally:
                        decode_queue.task_done()

            decoder_thread = None
            if self.live_segment_decode:
                decoder_thread = threading.Thread(target=decode_worker, daemon=True)
                decoder_thread.start()

            def decode_event(event: Dict):
                scene_index = int(event["scene_index"])
                segment = segments[scene_index]
                stem = f"{scene_index + 1:02d}_{segment.mode}_{sanitize_filename(segment.prompt)}"
                path = run_dir / "segments" / f"{stem}.mp4"
                segment_path = self._decode_latents_to_file(
                    event["latents"],
                    path,
                    fps=fps,
                    subtitle=segment.subtitle.strip() if use_subtitles else "",
                    duration_seconds=segment.duration_seconds,
                )
                segment_paths.append(segment_path)
                if progress_callback is not None:
                    progress_callback({
                        "type": "segment",
                        "scene_index": scene_index,
                        "segment_path": segment_path,
                        "segment_paths": list(segment_paths),
                        "prompt": segment.prompt,
                    })

            def on_scene_done(event: Dict):
                scene_index = int(event["scene_index"])
                event = dict(event)
                scene_events.append({
                    "scene_index": scene_index,
                    "start_frame": int(event["start_frame"]),
                    "end_frame": int(event["end_frame"]),
                })
                if progress_callback is not None:
                    progress_callback({
                        "type": "progress",
                        "scene_index": scene_index,
                        "prompt": segments[scene_index].prompt,
                    })
                if self.live_segment_decode:
                    event["latents"] = event["latents"].detach().cpu()
                    decode_queue.put(event)

            started_at = time.time()
            inference_kwargs = {
                "noise": sampled_noise,
                "text_prompts": [prompt_text],
                "return_latents": True,
                "initial_latent": None,
                "low_memory": self.low_memory,
            }
            if isinstance(pipeline, CausalInferencePipeline):
                inference_kwargs["progress_callback"] = on_scene_done
            video, latents = pipeline.inference(**inference_kwargs)

            if self.live_segment_decode:
                decode_queue.put(None)
                decode_queue.join()
                if decoder_thread is not None:
                    decoder_thread.join()
            if decode_errors:
                raise RuntimeError(f"Segment decode failed: {decode_errors[0]}") from decode_errors[0]

            with self._vae_decode_lock:
                full_video = 255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
            if use_subtitles:
                subtitles = [segment.subtitle.strip() for segment in segments]
                durations = [float(segment.duration_seconds) for segment in segments]
                if any(subtitles):
                    full_video = add_subtitles(full_video, subtitles, fps=float(fps), time_durations=durations)

            latent_frame_count = int(latents.shape[1])
            final_path = self._write_video_tensor(full_video, run_dir / "final_echo_forcing.mp4", fps)
            if not self.live_segment_decode:
                segment_paths = self._write_segments_from_full_video(
                    full_video,
                    segments,
                    run_dir=run_dir,
                    fps=fps,
                    progress_callback=progress_callback,
                )
            del sampled_noise, video, latents
            self.clear_runtime_cache(unload_model=True)

            return {
                "prompt": prompt_text,
                "run_dir": str(run_dir),
                "segment_paths": segment_paths,
                "final_path": final_path,
                "latent_frames": latent_frame_count,
                "duration_seconds": float(total_duration),
                "elapsed_seconds": time.time() - started_at,
            }

    def _write_segments_from_full_video(
        self,
        full_video: torch.Tensor,
        segments: List[PromptSegment],
        run_dir: Path,
        fps: int,
        progress_callback: Optional[Callable[[Dict], None]] = None,
    ) -> List[str]:
        total_frames = int(full_video.shape[1])
        total_duration = max(1.0, float(sum(segment.duration_seconds for segment in segments)))
        frame_cursor = 0
        segment_paths: List[str] = []

        for scene_index, segment in enumerate(segments):
            if scene_index == len(segments) - 1:
                frame_end = total_frames
            else:
                frame_end = int(round(total_frames * sum(s.duration_seconds for s in segments[: scene_index + 1]) / total_duration))
                frame_end = max(frame_cursor + 1, min(total_frames, frame_end))

            segment_video = full_video[:, frame_cursor:frame_end]
            stem = f"{scene_index + 1:02d}_{segment.mode}_{sanitize_filename(segment.prompt)}"
            path = run_dir / "segments" / f"{stem}.mp4"
            segment_path = self._write_video_tensor(segment_video, path, fps=fps)
            segment_paths.append(segment_path)
            frame_cursor = frame_end

            if progress_callback is not None:
                progress_callback({
                    "type": "segment",
                    "scene_index": scene_index,
                    "segment_path": segment_path,
                    "segment_paths": list(segment_paths),
                    "prompt": segment.prompt,
                })

        return segment_paths

    def clear_run_cache(self, run_dir: str, unload_model: bool = False):
        if not run_dir:
            return
        path = Path(run_dir).resolve()
        root = self.output_root.resolve()
        if path == root or root not in path.parents:
            raise ValueError(f"Refusing to clear cache outside {root}: {path}")
        if path.exists():
            shutil.rmtree(path)
        self.clear_runtime_cache(unload_model=unload_model)

    def clear_runtime_cache(self, unload_model: bool = False):
        pipeline = self.pipeline
        if pipeline is not None:
            try:
                pipeline.vae.model.clear_cache()
            except Exception:
                pass
            if hasattr(pipeline, "kv_cache1"):
                pipeline.kv_cache1 = None
            if hasattr(pipeline, "crossattn_cache"):
                pipeline.crossattn_cache = None
            if hasattr(pipeline, "scene_pool"):
                pipeline.scene_pool = []
            if unload_model:
                try:
                    pipeline.text_encoder.to("cpu")
                    pipeline.generator.to("cpu")
                    pipeline.vae.to("cpu")
                except Exception:
                    pass
                self.pipeline = None
                self.config = None
                self.device = None
                self.low_memory = False
                del pipeline
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass


_ENGINE: Optional[EchoForcingDemoEngine] = None
_ENGINE_LOCK = threading.Lock()


def get_engine() -> EchoForcingDemoEngine:
    global _ENGINE
    with _ENGINE_LOCK:
        if _ENGINE is None:
            _ENGINE = EchoForcingDemoEngine()
        return _ENGINE
