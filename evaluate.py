"""Evaluate a released PointTC configuration on prepared ShapeNet samples."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", required=True, help="Directory containing test*.h5 shards")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--snrs", nargs="+", type=float, default=[0, 2.5, 5, 7.5, 10, 12.5, 15])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--pc-error", default=str(Path(__file__).resolve().parent / "utils" / "pc_error"),
                        help="MPEG metric executable (default: bundled utils/pc_error)")
    parser.add_argument("--save-examples", type=int, default=0, help="Export the first N original/reconstructed PLY pairs per SNR")
    return parser.parse_args()


def main():
    args = parse_args()
    import numpy as np
    import torch
    from pointtc.common import load_checkpoint, load_config, resolve_device, seed_everything
    from pointtc.data import ShapeNetDataset
    from pointtc.metrics import METRIC_FIELDS, aggregate_metrics, point_cloud_metrics, write_ply
    from pointtc.models import METHOD_NAMES, build_model

    if args.max_samples is not None and args.max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    if args.save_examples < 0:
        raise ValueError("--save-examples must be nonnegative")
    if not all(np.isfinite(args.snrs)) or len(set(args.snrs)) != len(args.snrs):
        raise ValueError("--snrs must contain distinct finite SNRs")
    binary = shutil.which(args.pc_error)
    if binary is None:
        if Path(args.pc_error).is_file():
            raise PermissionError(f"pc_error is not executable; run: chmod +x '{args.pc_error}'")
        raise FileNotFoundError(f"pc_error executable not found: {args.pc_error}")
    binary = str(Path(binary).resolve())
    try:
        import open3d  # noqa: F401 — check before starting a long evaluation
    except ImportError as exc:
        raise RuntimeError("Install the evaluation dependencies: python -m pip install -r requirements.txt") from exc

    config = load_config(args.config)
    method, channel = config["method"], config["channel"]
    method_name = METHOD_NAMES[method]
    device = resolve_device(args.device)
    seed_everything(args.seed)
    dataset = ShapeNetDataset(args.data_root, "test", config["num_points"])
    count = min(len(dataset), args.max_samples) if args.max_samples is not None else len(dataset)
    model = build_model(method, channel, num_symbols=config["num_symbols"]).to(device)
    load_checkpoint(model, args.checkpoint, method, channel)
    model.eval()
    checkpoint_hash = hashlib.sha256()
    with Path(args.checkpoint).open("rb") as checkpoint_stream:
        for chunk in iter(lambda: checkpoint_stream.read(1024 * 1024), b""):
            checkpoint_hash.update(chunk)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata = {
        "config": config, "checkpoint": str(Path(args.checkpoint).resolve()),
        "checkpoint_sha256": checkpoint_hash.hexdigest(),
        "data_root": str(Path(args.data_root).resolve()), "device": str(device),
        "seed": args.seed, "snrs_db": args.snrs, "samples_per_snr": count,
        "batch_size": 1, "torch_version": torch.__version__, "pc_error": binary,
        "normal_radius": 0.1, "normal_max_nn": 30,
        "metric_resolution": 1.0,
        "aggregation": "arithmetic sample means; PSNR averaged in dB; mseF symmetric fields",
        "stochastic_evaluation": "Gumbel sampling and channel noise remain active; seed reset per SNR",
        "status": "running",
    }
    metadata_file = output / "metadata.json"
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    prefix = ["method", "channel", "num_symbols", "snr_db"]
    sample_fields = prefix + ["sample_index", "label"] + list(METRIC_FIELDS)
    summary_fields = prefix + ["samples"] + list(METRIC_FIELDS)
    with (output / "per_sample.csv").open("w", newline="", encoding="utf-8") as sample_stream, \
            (output / "summary.csv").open("w", newline="", encoding="utf-8") as summary_stream:
        samples_writer = csv.DictWriter(sample_stream, fieldnames=sample_fields)
        summary_writer = csv.DictWriter(summary_stream, fieldnames=summary_fields)
        samples_writer.writeheader()
        summary_writer.writeheader()
        with torch.inference_mode():
            for snr in args.snrs:
                seed_everything(args.seed)
                rows = []
                for index in range(count):
                    points, label = dataset[index]
                    outputs = model(points.unsqueeze(0).to(device), snr=snr)
                    reconstruction = outputs[0].squeeze(0).detach().cpu().numpy()
                    row = {"method": method_name, "channel": channel,
                           "num_symbols": config["num_symbols"], "snr_db": snr,
                           "sample_index": index, "label": label,
                           "cd": float(outputs[1].item()),
                           "symbols": float(outputs[2].item()) if len(outputs) == 3 else float(config["num_symbols"])}
                    if not np.isfinite(row["cd"]) or not np.isfinite(row["symbols"]):
                        raise ValueError(f"Non-finite model metrics at sample {index}, SNR {snr}")
                    original = points.numpy()
                    row.update(point_cloud_metrics(original, reconstruction, binary))
                    samples_writer.writerow(row)
                    rows.append(row)
                    if index < args.save_examples:
                        example = output / "examples" / f"snr_{snr:g}"
                        write_ply(example / f"{index:06d}_original.ply", original)
                        write_ply(example / f"{index:06d}_reconstruction.ply", reconstruction)
                summary = {"method": method_name, "channel": channel,
                           "num_symbols": config["num_symbols"], "snr_db": snr, **aggregate_metrics(rows)}
                summary_writer.writerow(summary)
                sample_stream.flush()
                summary_stream.flush()
                print(f"{method_name} ({channel.upper()}, max symbols={config['num_symbols']}), "
                      f"SNR {snr:g} dB: samples={count}, CD={summary['cd']:.7g}, "
                      f"D1 PSNR={summary['d1_psnr']:.4f} dB, D2 PSNR={summary['d2_psnr']:.4f} dB, "
                      f"symbols={summary['symbols']:.4g}")
    metadata["status"] = "complete"
    metadata_file.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Saved results to {output}")


if __name__ == "__main__":
    main()
