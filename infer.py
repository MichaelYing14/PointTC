"""Reconstruct one prepared 2048-point XYZ cloud without changing its scale."""

import argparse
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True, help="Already normalized 2048 x 3 .npy or .ply")
    parser.add_argument("--output", required=True, help="Reconstructed .npy or .ply file")
    parser.add_argument("--snr", type=float, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    import numpy as np
    import torch
    from pointtc.common import load_checkpoint, load_config, resolve_device, seed_everything
    from pointtc.metrics import read_points, write_ply
    from pointtc.models import METHOD_NAMES, build_model

    if not math.isfinite(args.snr):
        raise ValueError("--snr must be finite")
    output = Path(args.output)
    if output.suffix.lower() not in {".npy", ".ply"}:
        raise ValueError("--output must end with .npy or .ply")
    if output.resolve() == Path(args.input).resolve():
        raise ValueError("--output must differ from --input")
    config = load_config(args.config)
    points = read_points(args.input, config["num_points"])
    seed_everything(args.seed)
    device = resolve_device(args.device)
    model = build_model(config["method"], config["channel"], num_symbols=config["num_symbols"]).to(device)
    load_checkpoint(model, args.checkpoint, config["method"], config["channel"])
    model.eval()
    with torch.inference_mode():
        outputs = model(torch.from_numpy(points).unsqueeze(0).to(device), snr=args.snr)
    reconstruction = outputs[0].squeeze(0).detach().cpu().numpy()
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npy":
        with output.open("wb") as stream:
            np.save(stream, reconstruction, allow_pickle=False)
    else:
        write_ply(output, reconstruction)
    metadata = {"method": METHOD_NAMES[config["method"]], "channel": config["channel"],
                "num_symbols": config["num_symbols"],
                "snr_db": args.snr, "seed": args.seed, "device": str(device),
                "cd": float(outputs[1].item()),
                "symbols": float(outputs[2].item()) if len(outputs) == 3 else config["num_symbols"],
                "normalization": "none; input used as supplied"}
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{metadata['method']}: saved {output}; CD={metadata['cd']:.7g}, symbols={metadata['symbols']:.4g}")


if __name__ == "__main__":
    main()
