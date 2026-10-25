# Streaming inference

Echo-Forcing supports incremental generation and VAE decoding through a command-line entry point and a Gradio interface. Install the base requirements and then run `pip install -r requirements-streaming.txt`. Download the checkpoints using the unchanged [Quick Start](README.md#-quick-start).

## Gradio

```bash
bash run_streaming.sh
```

Open <http://127.0.0.1:1325>. The default launcher assigns physical GPU 0 to generation and chooses another GPU with at least 16 GiB free for decoding. You can select devices explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1 bash run_streaming.sh
```

For single-GPU execution:

```bash
CUDA_VISIBLE_DEVICES=0 bash run_streaming.sh
```

`ECHO_PYTHON`, `ECHO_HOST`, `ECHO_PORT`, and `ECHO_VAE_DEVICE` override the interpreter, listen address, port, and decoder device. The default listen address is local-only. For remote access, forward port 1325 through SSH or set a listen address appropriate for your network.

The page keeps the live video and Director Console visible together. Enter an initial plan containing duration markers such as `[10s]`, `[10s#]`, and `[10s@]`, then select **Start generating**. During generation, enter a new prompt, choose Smooth, Hard, or Recall, and select **Switch prompt**. The request takes effect at the next generation-block boundary. Previously generated frames continue to play. **Stop & save** completes the current block and preserves the generated portion.

Generation uses four denoising steps by default. The two-step option in Generation settings is experimental and can change quality. **Load model** moves the initial model load ahead of the first generation request.

## Command line

```bash
CUDA_VISIBLE_DEVICES=0 python inference_stream.py \
  --prompt-file prompts/prompts_smooth.txt \
  --index 0 \
  --output output/streaming/smooth
```

For a dedicated decoder GPU, expose two GPUs and pass `--decoder-device cuda:1`. Use `--two-steps` only when you want to test the faster experimental sampling path.

## Output and timing

The Gradio run saves `chunk_*.mp4`, `final.mp4`, and `metrics.json` to `.streaming_cache/<run-id>/`. The command-line entry point uses its `--output` directory. MP4 chunks are emitted every 24 frames at 16 fps, with a final shorter chunk when necessary. CPU encoding can overlap GPU work; a second GPU allows VAE decoding to overlap generation of the next block.

Smooth transitions retain VAE decoding context. Hard cuts and recall start a new decoding context and apply frame-count alignment at the boundary. A live switch cannot revoke video already pushed to the player, so the switch acknowledgement can precede the visible change.
