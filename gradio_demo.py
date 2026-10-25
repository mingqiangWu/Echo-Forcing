"""Streaming UI. Launch with .venv/bin/python gradio_demo.py."""
import os
os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
if os.environ.get("ECHO_CUDA_DEVICE"):
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["ECHO_CUDA_DEVICE"]

import json
import threading
import uuid
from pathlib import Path

import gradio as gr
import torch

from streaming.engine import StreamingEngine
from streaming.control import StreamControl

ROOT = Path(__file__).resolve().parent
CSS = (ROOT / "assets" / "streaming.css").read_text(encoding="utf-8")
engine = StreamingEngine(decoder_device=os.environ.get("ECHO_VAE_DEVICE"))
sessions = {}
sessions_lock = threading.Lock()


def preset(mode):
    lines = [line.strip() for line in (ROOT / "prompts" / f"prompts_{mode}.txt").read_text().splitlines() if line.strip()]
    return lines[1 if mode == "hard" else 0].split(";")[0]


def warmup():
    with engine._lock, torch.inference_mode():
        engine.load()
    return "Model loaded. Ready to generate; later runs will reuse it."


def change_prompt(run_id, prompt, mode, duration):
    with sessions_lock:
        control = sessions.get(run_id)
    if control is None:
        return "Start generating before requesting a prompt switch."
    try:
        control.switch(prompt, duration, mode)
    except ValueError as exc:
        return str(exc)
    return "Switch queued for the next generation block. Buffered video will keep playing."


def stop(run_id):
    with sessions_lock:
        control = sessions.get(run_id)
    if control:
        control.stop()
        return "Stop requested. Finishing the current block and saving the generated video."
    return "No generation is running."


def generate(prompt, seed, two_steps, max_seconds):
    run_id = uuid.uuid4().hex
    control = StreamControl(interactive=True, max_seconds=int(max_seconds))
    with sessions_lock:
        sessions[run_id] = control
    skip = gr.skip
    switch_history = []
    # The player stays central; all other live feedback is concise text.
    yield skip(), "Starting generation…", run_id, "No switch requested", gr.update(interactive=False)
    try:
        for event in engine.stream(prompt, control, two_steps=bool(two_steps), seed=int(seed)):
            kind = event["type"]
            video, switch_log = skip(), skip()
            button = skip()
            if kind == "status":
                status = event["message"]
            elif kind == "preview":
                status = (f"Generated {event['frames']/16:.2f}s · {event['fps']:.2f} fps average · "
                          f"{event['elapsed']:.1f}s elapsed")
            elif kind == "chunk":
                video = event["path"]
                status = f"Streaming {event['video_seconds']:.2f}s of video"
            elif kind == "switch":
                status = f"Prompt switched · {event['mode']} · response {event['response_seconds']:.2f}s"
                switch_history.append(f"{event['mode']} applied · response {event['response_seconds']:.2f}s")
                switch_log = "\n".join(switch_history[-5:])
            elif kind == "done":
                result = event["result"]
                video = None  # Close the media stream rather than replaying its last chunk.
                status = (f"{'Stopped and saved' if result.get('cancelled') else 'Completed'} · "
                          f"{result.get('video_seconds',0):.2f}s video · "
                          f"steady rate {result.get('steady_fps') or 0:.2f} fps")
                button = gr.update(interactive=True)
            else:
                video = None
                status = event["message"]
                button = gr.update(interactive=True)
            yield video, status, run_id, switch_log, button
    finally:
        control.stop()
        with sessions_lock:
            sessions.pop(run_id, None)


with gr.Blocks(title="Echo-Forcing · Live") as demo:
    gr.HTML("""
    <div class="ef-hero">
      <div class="ef-brand">
        <svg class="ef-logo" viewBox="0 0 64 64" role="img" aria-label="Echo-Forcing logo">
          <defs><linearGradient id="ef-gradient" x1="0" y1="0" x2="1" y2="1">
            <stop offset="0%" stop-color="#4de7f7"/><stop offset="48%" stop-color="#7d71ff"/>
            <stop offset="100%" stop-color="#ff6caf"/>
          </linearGradient></defs>
          <rect x="2" y="2" width="60" height="60" rx="19" fill="#171d39"/>
          <path d="M17 20h29M17 32h21M17 44h29" fill="none" stroke="url(#ef-gradient)"
                stroke-width="6" stroke-linecap="round"/>
          <circle cx="47" cy="32" r="4" fill="#ff6caf"/>
        </svg>
        <span>Echo-Forcing</span>
      </div>
      <p>Imagine a scene. Watch it unfold. Change the story as it streams.</p>
    </div>
    """)
    run_state = gr.State("")
    with gr.Row(equal_height=False, elem_id="ef-workspace"):
        with gr.Column(scale=8, min_width=0, elem_id="ef-stage"):
            gr.HTML('<div class="ef-stage-label"><span class="ef-live-dot"></span> LIVE PREVIEW <small>16 FPS OUTPUT</small></div>')
            player = gr.Video(streaming=True, autoplay=True, interactive=False,
                              show_label=False, elem_id="ef-player")
            status = gr.Markdown("Ready · Live video will play here", elem_id="ef-status")
        with gr.Column(scale=5, min_width=0, elem_id="ef-controls"):
            gr.HTML('<div class="ef-panel-title">Director Console <span>CONTROL DECK</span></div>')
            plan = gr.Textbox(value=preset("smooth"), lines=3, max_lines=4,
                              label="Initial prompt", elem_id="ef-plan",
                              placeholder="Describe the scene. Use [10s], [10s#], or [10s@] for transitions.")
            new_prompt = gr.Textbox(lines=3, max_lines=4, label="New prompt · switch while running",
                                    elem_id="ef-new-prompt", placeholder="Describe the next scene…")
            with gr.Row(elem_id="ef-switch-options"):
                new_mode = gr.Dropdown([("Smooth", "smooth"), ("Hard", "hardcut"),
                                        ("Recall", "recall")], value="smooth", label="Transition")
                new_duration = gr.Number(value=10, precision=0, minimum=1, maximum=120,
                                         label="Next scene (seconds)")
            switch = gr.Button("Switch prompt ↗", variant="primary", elem_id="ef-switch")
            with gr.Row(elem_id="ef-actions"):
                warm = gr.Button("Load model")
                start = gr.Button("Start generating", variant="primary")
                halt = gr.Button("Stop & save", variant="stop", elem_id="ef-stop")
            switch_log = gr.Textbox(label="Switch log", value="No switch requested", lines=2,
                                    max_lines=4, interactive=False, elem_id="ef-switch-log")
            with gr.Accordion("Generation settings", open=False, elem_id="ef-settings"):
                examples = gr.Dropdown([("Smooth example", "smooth"), ("Hard example", "hard"),
                                        ("Recall example", "recall")], value="smooth", label="Load example")
                with gr.Row():
                    seed = gr.Number(value=0, precision=0, label="Random seed")
                    maximum = gr.Slider(10, 300, value=180, step=10, label="Maximum duration (seconds)")
                two_steps = gr.Checkbox(value=False, label="Experimental two-step sampling (may reduce quality)")
    examples.change(preset, examples, plan, queue=False, api_name="preset")
    warm.click(warmup, outputs=status, concurrency_id="gpu", concurrency_limit=1, api_name="warmup")
    start.click(generate, [plan, seed, two_steps, maximum],
                [player, status, run_state, switch_log, start],
                concurrency_id="gpu", concurrency_limit=1, api_name="generate")
    switch.click(change_prompt, [run_state, new_prompt, new_mode, new_duration],
                 switch_log, queue=False, api_name="switch_prompt")
    halt.click(stop, run_state, switch_log, queue=False, api_name="stop")


def main():
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.environ.get("ECHO_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("ECHO_PORT", "1325")),
        allowed_paths=[str(ROOT / ".streaming_cache")], show_error=True,
        css=CSS, theme=gr.themes.Base())


if __name__ == "__main__":
    main()
