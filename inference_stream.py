"""CLI streaming inference; chunks are visible on disk before final.mp4 exists."""
import argparse
import json
import os
import signal
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt")
    parser.add_argument("--prompt-file")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", default="output/streaming/cli")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--two-steps", action="store_true")
    parser.add_argument("--decoder-device")
    parser.add_argument("--max-seconds", type=int, default=180)
    args = parser.parse_args()
    from streaming.control import StreamControl
    from streaming.engine import StreamingEngine
    prompt = args.prompt
    if args.prompt_file:
        prompt = [line for line in Path(args.prompt_file).read_text().splitlines() if line.strip()][args.index]
    if not prompt:
        parser.error("Provide --prompt or --prompt-file")
    control = StreamControl(interactive=False, max_seconds=args.max_seconds)
    engine = StreamingEngine(decoder_device=args.decoder_device)
    previous_handler = signal.getsignal(signal.SIGINT)
    def request_stop(signum, frame):
        control.stop()
        print("Stop requested; finishing the current block and saving video.", flush=True)
    signal.signal(signal.SIGINT, request_stop)
    try:
        for event in engine.stream(prompt, control, seed=args.seed, two_steps=args.two_steps,
                                   directory=args.output):
            if event["type"] != "preview":
                print(json.dumps(event, ensure_ascii=False), flush=True)
            if event["type"] == "error":
                raise SystemExit(1)
    finally:
        signal.signal(signal.SIGINT, previous_handler)


if __name__ == "__main__":
    main()
