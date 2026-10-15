"""Point-cloud I/O and the bundled MPEG distortion metric adapter."""

from pathlib import Path
import re
import subprocess
import tempfile

import numpy as np


METRIC_FIELDS = ("cd", "symbols", "d1_mse", "d1_psnr", "d2_mse", "d2_psnr")
_FINAL_METRIC = re.compile(
    r"^\s*mseF(?P<psnr>,PSNR)?\s*\(p2(?P<kind>point|plane)\)\s*:\s*"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?inf(?:inity)?)\s*$",
    re.IGNORECASE,
)


def validate_xyz(points, num_points=None):
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or not len(points):
        raise ValueError(f"Expected nonempty XYZ array (N, 3), got {points.shape}")
    if num_points is not None and len(points) != num_points:
        raise ValueError(f"Expected exactly {num_points} points, got {len(points)}")
    if not np.isfinite(points).all():
        raise ValueError("Point coordinates must be finite")
    return points


def write_ply(path, points, normals=None):
    """Write an ASCII float32 PLY accepted by MPEG pc_error."""
    points = validate_xyz(points)
    names = ["x", "y", "z"]
    values = points
    if normals is not None:
        normals = validate_xyz(normals, len(points))
        names += ["nx", "ny", "nz"]
        values = np.concatenate((points, normals), axis=1)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as stream:
        stream.write(f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n")
        for name in names:
            stream.write(f"property float {name}\n")
        stream.write("end_header\n")
        np.savetxt(stream, values, fmt="%.9g")


def read_points(path, num_points=2048):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        points = np.load(path, allow_pickle=False)
    elif path.suffix.lower() == ".ply":
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError("Reading PLY files requires open3d; install requirements.txt") from exc
        points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
    else:
        raise ValueError("Input must be an .npy or .ply point cloud")
    return validate_xyz(points, num_points)


def parse_pc_error(output, require_d2=True):
    """Read symmetric mseF fields, never directional mse1/mse2 values."""
    values = {}
    for line in output.splitlines():
        match = _FINAL_METRIC.match(line)
        if match:
            metric = "d1" if match["kind"].lower() == "point" else "d2"
            name = f"{metric}_{'psnr' if match['psnr'] else 'mse'}"
            values[name] = float(match["value"])
    required = {"d1_mse", "d1_psnr"}
    if require_d2:
        required |= {"d2_mse", "d2_psnr"}
    missing = required - values.keys()
    if missing:
        raise ValueError(f"pc_error output lacks symmetric fields: {', '.join(sorted(missing))}")
    return values


def run_pc_error(binary, reference, reconstruction, normal_path=None, resolution=1.0):
    if resolution <= 0:
        raise ValueError("resolution must be positive")
    command = [str(binary), "-a", str(reference), "-b", str(reconstruction),
               f"--resolution={resolution:g}"]
    if normal_path is not None:
        command.extend(["-n", str(normal_path)])
    result = subprocess.run(command, capture_output=True, text=True, check=True, timeout=120)
    return parse_pc_error(result.stdout, require_d2=normal_path is not None)


def point_cloud_metrics(reference, reconstruction, binary):
    """Use paper normals (radius .1, max_nn 30) and peak resolution 1."""
    reference = validate_xyz(reference)
    reconstruction = validate_xyz(reconstruction)
    try:
        import open3d as o3d
    except ImportError as exc:
        raise RuntimeError("D1/D2 evaluation requires open3d; install requirements.txt") from exc
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(reference)
    cloud.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
    with tempfile.TemporaryDirectory(prefix="pointtc-metrics-") as directory:
        original = Path(directory) / "reference.ply"
        decoded = Path(directory) / "reconstruction.ply"
        write_ply(original, reference, np.asarray(cloud.normals))
        write_ply(decoded, reconstruction)
        return run_pc_error(binary, original, decoded, normal_path=original, resolution=1.0)


def aggregate_metrics(rows):
    """Arithmetic sample means, including PSNR means in dB.

    Do not average category means or convert a mean MSE into a mean PSNR.
    Missing optional metrics remain blank in the exported summary.
    """
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot aggregate an empty evaluation")
    result = {"samples": len(rows)}
    for name in METRIC_FIELDS:
        values = [float(row[name]) for row in rows if row.get(name) not in (None, "")]
        if values and len(values) != len(rows):
            raise ValueError(f"Metric {name} is missing from some samples")
        if any(np.isnan(value) for value in values):
            raise ValueError(f"Metric {name} contains NaN")
        result[name] = float(np.mean(values)) if values else ""
    return result
