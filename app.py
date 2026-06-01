import os
import queue
import re
import threading
import time
import traceback
from pathlib import Path
from typing import Dict, List


def configure_cuda_device():
    os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")
    requested = os.environ.get("ECHO_CUDA_DEVICE")
    if requested:
        os.environ["CUDA_VISIBLE_DEVICES"] = requested


configure_cuda_device()

import gradio as gr

from inference_demo import PromptSegment, get_engine


REPO_ROOT = Path(__file__).resolve().parent
GRADIO_TMP = REPO_ROOT / ".gradio_cache"
DEFAULT_PROMPT_PATH = REPO_ROOT / "prompts" / "demo_hard.txt"

GRADIO_TMP.mkdir(parents=True, exist_ok=True)
os.environ["GRADIO_TEMP_DIR"] = str(GRADIO_TMP)


MODE_LABELS = {
    "smooth": "🌊 Smooth Transition",
    "hardcut": "⚡ Hard Cut",
    "recall": "🧠 Scene Recall",
}
MARKER_TO_MODE = {"#": "hardcut", "@": "recall", "": "smooth", None: "smooth"}
SEGMENT_PATTERN = re.compile(r"(?P<prompt>.*?)\[(?P<seconds>\d+\.?\d*)\s*s(?P<marker>[#@])?\]\s*$", re.S)


def _normalize_segments(segments) -> List[Dict]:
    return list(segments or [])


def _to_segment_objects(segments: List[Dict]) -> List[PromptSegment]:
    return [
        PromptSegment(
            prompt=item["prompt"],
            duration_seconds=int(item["duration_seconds"]),
            mode=item.get("mode", "smooth"),
            subtitle=item.get("subtitle", ""),
        )
        for item in segments
    ]


def load_default_segments() -> List[Dict]:
    if not DEFAULT_PROMPT_PATH.exists():
        return []

    text = DEFAULT_PROMPT_PATH.read_text(encoding="utf-8", errors="replace").strip()
    if not text:
        return []

    first_line = next((line.strip() for line in text.splitlines() if line.strip()), "")
    prompt_part, _, subtitle_part = first_line.partition(";")
    subtitles = [item.strip() for item in subtitle_part.split("|")] if subtitle_part else []

    segments = []
    for index, raw_part in enumerate(part.strip() for part in prompt_part.split("|") if part.strip()):
        match = SEGMENT_PATTERN.match(raw_part)
        if match:
            prompt = match.group("prompt").strip()
            duration = int(float(match.group("seconds")))
            mode = MARKER_TO_MODE.get(match.group("marker"), "smooth")
        else:
            prompt = raw_part
            duration = 10
            mode = "smooth"

        segments.append(
            {
                "prompt": prompt,
                "duration_seconds": max(5, min(120, duration)),
                "mode": mode,
                "subtitle": subtitles[index] if index < len(subtitles) else "",
            }
        )
    return segments


def format_preview(segments: List[Dict]) -> str:
    segments = _normalize_segments(segments)
    if not segments:
        return "✨ No prompt segments yet. Add one segment at a time below."

    lines = []
    for idx, item in enumerate(segments, start=1):
        mode = MODE_LABELS.get(item.get("mode", "smooth"), item.get("mode", "smooth"))
        duration = int(item["duration_seconds"])
        prompt = item["prompt"].strip()
        lines.append(f"🎬 Segment {idx:02d}  ·  {mode}  ·  ⏱️ {duration}s")
        lines.append(prompt)
        subtitle = item.get("subtitle", "").strip()
        if subtitle:
            lines.append(f"💬 Subtitle: {subtitle}")
        lines.append("")
    return "\n".join(lines).strip()


def progress_panel(percent: int = 0, elapsed_seconds: float = 0.0, label: str = "Ready") -> str:
    percent = max(0, min(100, int(percent)))
    minutes = int(elapsed_seconds // 60)
    seconds = int(elapsed_seconds % 60)
    return f"""
    <div class="echo-progress-card">
      <div class="echo-progress-top">
        <span>{label}</span>
        <strong>{percent}%</strong>
      </div>
      <div class="echo-progress-track">
        <div class="echo-progress-fill" style="width: {percent}%"></div>
      </div>
      <div class="echo-progress-meta">⏱️ Runtime: {minutes:02d}:{seconds:02d}</div>
    </div>
    """




def final_video_placeholder() -> str:
    return """
    <div class="echo-placeholder echo-final-placeholder">
      <div class="echo-placeholder-title">Final Video</div>
      <div class="echo-placeholder-body">The final stitched video will appear here after generation.</div>
    </div>
    """


def gallery_placeholder() -> str:
    return """
    <div class="echo-placeholder echo-gallery-placeholder">
      <div class="echo-placeholder-title">Gallery</div>
      <div class="echo-placeholder-body">Segment previews will appear here after the final video is rendered.</div>
    </div>
    """

def add_segment(prompt, mode, duration, subtitle, segments, locked):
    segments = _normalize_segments(segments)
    if locked:
        return format_preview(segments), prompt, segments, "🔒 Generation is running. The prompt list is frozen."

    prompt = (prompt or "").strip()
    if not prompt:
        return format_preview(segments), "", segments, "⚠️ Please enter one prompt segment first."

    if "|" in prompt or len(re.findall(r"\[\d+\.?\d*\s*s[#@]?\]", prompt)) > 0:
        return (
            format_preview(segments),
            prompt,
            segments,
            "⚠️ Submit only one plain prompt segment at a time. Choose mode and duration with the controls.",
        )

    duration = max(5, min(120, int(duration)))
    segments.append(
        {
            "prompt": prompt,
            "duration_seconds": duration,
            "mode": mode or "smooth",
            "subtitle": (subtitle or "").strip(),
        }
    )
    return format_preview(segments), "", segments, f"✅ Segment {len(segments)} submitted."


def reset_to_default(locked):
    if locked:
        return gr.update(), gr.update(), gr.update(), "🔒 Generation is running. The default prompt cannot be reloaded."
    segments = load_default_segments()
    return format_preview(segments), segments, "", f"✨ Reloaded default prompt: {DEFAULT_PROMPT_PATH.name}"


def clear_segments(run_dir, segments):
    engine = get_engine()
    if run_dir:
        try:
            engine.clear_run_cache(run_dir, unload_model=True)
        except Exception:
            pass
    engine.clear_runtime_cache(unload_model=True)

    segments = []
    return (
        format_preview(segments),
        progress_panel(0, 0, "Ready"),
        gr.update(value=gallery_placeholder(), visible=True),
        gr.update(value=[], visible=False),
        gr.update(value=final_video_placeholder(), visible=True),
        gr.update(visible=False),
        segments,
        False,
        "",
        "Current task cache cleared. Ready for a new generation.",
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )

def _gallery_value(paths: List[str]):
    return [(path, f"🎞️ Segment {idx}") for idx, path in enumerate(paths, start=1)]


def generate_video(segments, use_subtitles):
    segments = _normalize_segments(segments)
    if not segments:
        yield (
            "Please submit at least one prompt segment first.",
            progress_panel(0, 0, "Waiting"),
            gr.update(value=gallery_placeholder(), visible=True),
            gr.update(value=[], visible=False),
            gr.update(value=final_video_placeholder(), visible=True),
            gr.update(visible=False),
            False,
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
        return

    segment_objects = _to_segment_objects(segments)
    event_queue: "queue.Queue[Dict]" = queue.Queue()
    result_holder: Dict = {}

    def progress_callback(event):
        event_queue.put(event)

    def worker():
        engine = get_engine()
        try:
            result_holder["result"] = engine.generate(
                segment_objects,
                use_subtitles=bool(use_subtitles),
                progress_callback=progress_callback,
            )
        except Exception as exc:
            result_holder["error"] = f"{exc}\n{traceback.format_exc()}"
        finally:
            engine.clear_runtime_cache(unload_model=True)
            event_queue.put({"type": "done"})

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    total_segments = len(segments)
    segment_paths: List[str] = []
    started_at = time.time()

    yield (
        f"Generation started with {total_segments} segment(s). Prompts are frozen while running.",
        progress_panel(3, 0, "Starting"),
        gr.update(value=gallery_placeholder(), visible=True),
        gr.update(value=[], visible=False),
        gr.update(value=final_video_placeholder(), visible=True),
        gr.update(visible=False),
        True,
        "",
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
        gr.update(interactive=False),
    )

    while True:
        event = event_queue.get()
        if event.get("type") == "phase":
            percent = int(event.get("percent", 5))
            label = str(event.get("label", "Working"))
            message = str(event.get("message", label))
            yield (
                message,
                progress_panel(percent, time.time() - started_at, label),
                gr.update(value=gallery_placeholder(), visible=True),
                gr.update(value=_gallery_value(segment_paths), visible=bool(segment_paths)),
                gr.update(value=final_video_placeholder(), visible=True),
                gr.update(visible=False),
                True,
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        elif event.get("type") == "segment":
            segment_paths = event.get("segment_paths", segment_paths)
            done = int(event.get("scene_index", len(segment_paths) - 1)) + 1
            percent = int(done * 100 / total_segments)
            yield (
                f"Finished segment {done}/{total_segments}. Continuing with the remaining segments.",
                progress_panel(percent, time.time() - started_at, "Writing previews"),
                gr.update(visible=False),
                gr.update(value=_gallery_value(segment_paths), visible=True),
                gr.update(value=final_video_placeholder(), visible=True),
                gr.update(visible=False),
                True,
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        elif event.get("type") == "progress":
            done = int(event.get("scene_index", -1)) + 1
            percent = max(12, int(done * 100 / total_segments))
            yield (
                f"Generated segment {done}/{total_segments}. Final-quality previews will appear after full-video decoding.",
                progress_panel(percent, time.time() - started_at, "Generating"),
                gr.update(value=gallery_placeholder(), visible=True),
                gr.update(value=_gallery_value(segment_paths), visible=bool(segment_paths)),
                gr.update(value=final_video_placeholder(), visible=True),
                gr.update(visible=False),
                True,
                "",
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
                gr.update(interactive=False),
            )
        elif event.get("type") == "done":
            break

    if "error" in result_holder:
        yield (
            f"Generation failed:\n{result_holder['error']}",
            progress_panel(0, time.time() - started_at, "Failed"),
            gr.update(value=gallery_placeholder(), visible=not bool(segment_paths)),
            gr.update(value=_gallery_value(segment_paths), visible=bool(segment_paths)),
            gr.update(value=final_video_placeholder(), visible=True),
            gr.update(visible=False),
            False,
            "",
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
            gr.update(interactive=True),
        )
        return

    result = result_holder["result"]
    segment_paths = result["segment_paths"] or segment_paths
    elapsed = result["elapsed_seconds"]
    duration = result["duration_seconds"]
    yield (
        f"Done: {total_segments} segment(s), target duration about {duration:.1f}s, elapsed {elapsed:.1f}s. Cache cleaned.",
        progress_panel(100, elapsed, "Complete"),
        gr.update(visible=False),
        gr.update(value=_gallery_value(segment_paths), visible=True),
        gr.update(visible=False),
        gr.update(value=result["final_path"], visible=True),
        False,
        result["run_dir"],
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
        gr.update(interactive=True),
    )


DEFAULT_SEGMENTS = load_default_segments()

CSS = """
:root {
  --echo-pink: #ff4fd8;
  --echo-blue: #2dd4ff;
  --echo-yellow: #ffe66d;
  --echo-green: #5dffb3;
}
body, .gradio-container {
  font-size: 18px !important;
  background:
    radial-gradient(circle at 12% 6%, rgba(255, 79, 216, 0.28), transparent 28%),
    radial-gradient(circle at 88% 10%, rgba(45, 212, 255, 0.28), transparent 30%),
    linear-gradient(135deg, #120724 0%, #151a3d 48%, #062b32 100%) !important;
  color: #f8fbff !important;
}
.echo-title {
  text-align: center;
  padding: 18px 16px 8px;
}
.echo-title h1 {
  margin: 0;
  font-size: 52px;
  line-height: 1.05;
  color: #ffffff;
  text-shadow: 0 0 18px rgba(45, 212, 255, 0.65), 0 0 28px rgba(255, 79, 216, 0.45);
}
.echo-title .subtitle {
  margin-top: 8px;
  font-size: 22px;
  color: #dff7ff;
}
.echo-card {
  border: 1px solid rgba(255, 255, 255, 0.16) !important;
  background: rgba(255, 255, 255, 0.08) !important;
  box-shadow: 0 16px 40px rgba(0, 0, 0, 0.28) !important;
}
.echo-preview textarea {
  font-size: 17px !important;
  line-height: 1.55 !important;
  color: #f9fdff !important;
  background: rgba(8, 10, 28, 0.74) !important;
}
textarea, input, .wrap, label, button, .prose, .markdown, .form, .block, .output-html {
  font-size: 20px !important;
  font-weight: 750 !important;
}
button {
  font-weight: 800 !important;
}
.primary button, button.primary {
  background: linear-gradient(90deg, var(--echo-pink), #8b5cf6, var(--echo-blue)) !important;
  color: white !important;
}
.echo-progress-card {
  padding: 20px;
  border-radius: 18px;
  border: 1px solid rgba(255, 255, 255, 0.18);
  background: linear-gradient(135deg, rgba(255, 79, 216, 0.20), rgba(45, 212, 255, 0.18));
  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.24);
}
.echo-progress-top {
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 20px;
  font-size: 26px;
  font-weight: 900;
  color: #ffffff;
}
.echo-progress-top strong {
  font-size: 42px;
  color: var(--echo-yellow);
  text-shadow: 0 0 18px rgba(255, 230, 109, 0.55);
}
.echo-progress-track {
  height: 28px;
  margin-top: 16px;
  border-radius: 999px;
  overflow: hidden;
  background: rgba(255, 255, 255, 0.18);
}
.echo-progress-fill {
  height: 100%;
  border-radius: 999px;
  background: linear-gradient(90deg, var(--echo-pink), var(--echo-yellow), var(--echo-green), var(--echo-blue));
  transition: width 0.35s ease;
}
.echo-progress-meta {
  margin-top: 12px;
  font-size: 22px;
  font-weight: 800;
  color: #e9fbff;
}

/* Layout refinements */
.echo-progress-card {
  background: linear-gradient(135deg, rgba(255,255,255,0.92), rgba(218,246,255,0.86)) !important;
  color: #132033 !important;
  border-color: rgba(255,255,255,0.7) !important;
}
.echo-progress-top { color: #132033 !important; }
.echo-progress-top strong { color: #7c2d12 !important; text-shadow: none !important; }
.echo-progress-meta { color: #24364f !important; }
.echo-progress-track { background: rgba(26, 43, 68, 0.14) !important; }
.echo-final-video video {
  width: 100% !important;
  max-height: 720px !important;
  min-height: 620px !important;
  object-fit: contain !important;
  background: #050816 !important;
}
.echo-final-video { min-height: 660px !important; }
.gallery, .gallery-container { min-height: 430px !important; }
.gallery {
  padding: 14px !important;
}
.gallery img, .gallery video {
  object-fit: contain !important;
  width: calc(100% - 12px) !important;
  height: calc(100% - 12px) !important;
  margin: 6px !important;
  border-radius: 8px !important;
}


.echo-placeholder {
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  min-height: 660px;
  border-radius: 10px;
  border: 1px solid rgba(255, 255, 255, 0.26);
  background: rgba(255, 255, 255, 0.94);
  color: #1f2937;
  text-align: center;
}
.echo-placeholder-title {
  font-size: 34px;
  font-weight: 950;
  margin-bottom: 12px;
}
.echo-placeholder-body {
  font-size: 22px;
  font-weight: 800;
  color: #526174;
  max-width: 620px;
}
.echo-final-placeholder {
  min-height: 660px;
  background: linear-gradient(135deg, rgba(255,255,255,0.96), rgba(224,246,255,0.92));
}
.echo-gallery-placeholder {
  min-height: 430px;
}
label, .label-wrap span {
  font-weight: 900 !important;
  font-size: 22px !important;
}
.block label span,
.block .label-wrap span,
.form label span {
  font-size: 22px !important;
  font-weight: 900 !important;
  color: #21304a !important;
}

"""


with gr.Blocks(title="Echo-Forcing Interactive Long Video Generation") as demo:
    segments_state = gr.State(DEFAULT_SEGMENTS)
    locked_state = gr.State(False)
    run_dir_state = gr.State("")

    gr.HTML(
        """
        <div class="echo-title">
          <h1>🎬 Echo-Forcing</h1>
          <div class="subtitle">Interactive Long Video Console · Segmented Prompts · Live Preview · Final Stitching</div>
        </div>
        """
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=5, min_width=420, elem_classes=["echo-card"]):
            prompt_preview = gr.Textbox(
                label="🧩 Prompt Segment Preview (read-only)",
                value=format_preview(DEFAULT_SEGMENTS),
                lines=10,
                interactive=False,
                elem_classes=["echo-preview"],
            )
            prompt_input = gr.Textbox(
                label="✍️ New Prompt Segment (submit one segment at a time)",
                placeholder="Example: Static shot, cinematic realism. A dramatic scene unfolds...",
                lines=5,
            )
            with gr.Row():
                mode_select = gr.Radio(
                    choices=[
                        ("🌊 Smooth Transition", "smooth"),
                        ("⚡ Hard Cut", "hardcut"),
                        ("🧠 Scene Recall", "recall"),
                    ],
                    value="hardcut",
                    label="🎛️ Segment Mode",
                )
                duration_slider = gr.Slider(5, 120, value=10, step=1, label="⏱️ Duration (seconds)")
            use_subtitles = gr.Checkbox(value=False, label="💬 Enable Subtitles")
            subtitle_input = gr.Textbox(label="💬 Subtitle for This Segment (optional)", lines=2)
            with gr.Row():
                submit_btn = gr.Button("➕ Submit Segment", variant="secondary")
                default_btn = gr.Button("✨ Reload Default Prompt", variant="secondary")
            with gr.Row():
                generate_btn = gr.Button("🚀 Start Generation", variant="primary")
                clear_btn = gr.Button("🧹 End / Clear Cache", variant="stop")

        with gr.Column(scale=7, min_width=560, elem_classes=["echo-card"]):
            progress_bar = gr.HTML(progress_panel(0, 0, "Ready"))
            status_box = gr.Textbox(label="📣 Status", value="✨ Default prompt loaded. You can generate directly or append more segments.", lines=8, interactive=False)
            final_placeholder = gr.HTML(final_video_placeholder())
            final_video = gr.Video(label="Final Video", autoplay=False, visible=False, elem_classes=["echo-final-video"])

    gallery_placeholder_component = gr.HTML(gallery_placeholder())
    segment_gallery = gr.Gallery(label="Gallery", columns=5, height=430, object_fit="contain", allow_preview=True, visible=False)

    submit_btn.click(
        add_segment,
        inputs=[prompt_input, mode_select, duration_slider, subtitle_input, segments_state, locked_state],
        outputs=[prompt_preview, prompt_input, segments_state, status_box],
    )

    default_btn.click(
        reset_to_default,
        inputs=[locked_state],
        outputs=[prompt_preview, segments_state, prompt_input, status_box],
    )

    generate_btn.click(
        generate_video,
        inputs=[segments_state, use_subtitles],
        outputs=[
            status_box,
            progress_bar,
            gallery_placeholder_component,
            segment_gallery,
            final_placeholder,
            final_video,
            locked_state,
            run_dir_state,
            prompt_input,
            mode_select,
            duration_slider,
            subtitle_input,
            submit_btn,
            generate_btn,
            clear_btn,
        ],
    )

    clear_btn.click(
        clear_segments,
        inputs=[run_dir_state, segments_state],
        outputs=[
            prompt_preview,
            progress_bar,
            gallery_placeholder_component,
            segment_gallery,
            final_placeholder,
            final_video,
            segments_state,
            locked_state,
            run_dir_state,
            status_box,
            prompt_input,
            mode_select,
            duration_slider,
            subtitle_input,
            submit_btn,
            generate_btn,
            clear_btn,
        ],
    )


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2)
    demo.launch(server_name="0.0.0.0", server_port=1324, css=CSS)
