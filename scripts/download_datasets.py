"""Download and prepare the EEG datasets described in Dataset/README.md."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import sys
from urllib.request import Request, urlopen
from zipfile import BadZipFile, ZipFile, ZipInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "Dataset"
CHUNK_SIZE = 1024 * 1024
USER_AGENT = "hybrid-photonic-mi-bci-dataset-downloader/0.1"


@dataclass(frozen=True)
class DatasetDownload:
    key: str
    title: str
    url: str
    archive_relative_path: Path


BCICIV_1_ASC = DatasetDownload(
    key="bciciv-1-asc",
    title="BCI Competition IV Dataset 1 (100 Hz ASCII)",
    url="https://bbci.de/competition/download/competition_iv/BCICIV_1_asc.zip",
    archive_relative_path=Path("BCICIV_1_asc") / "BCICIV_1_asc.zip",
)
BNCI2014_004 = DatasetDownload(
    key="bnci2014-004",
    title="BCI Competition IV Dataset 2b / BNCI2014_004 (GDF)",
    url="https://bbci.de/competition/download/competition_iv/BCICIV_2b_gdf.zip",
    archive_relative_path=Path("BNCI2014_004") / "BCICIV_2b_gdf.zip",
)
DOWNLOADS = {item.key: item for item in (BCICIV_1_ASC, BNCI2014_004)}

ASCII_NAME = re.compile(r"^BCICIV_calib_ds1[a-g]_(?:cnt|mrk|nfo)\.txt$", re.IGNORECASE)
GDF_NAME = re.compile(r"^B\d{4}[TE]\.gdf$", re.IGNORECASE)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=("all", *DOWNLOADS),
        default="all",
        help="Dataset to download. The default downloads both datasets.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Dataset root directory.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Download again and overwrite extracted files.",
    )
    args = parser.parse_args()

    selected = tuple(DOWNLOADS.values()) if args.dataset == "all" else (DOWNLOADS[args.dataset],)
    root = args.root.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    for dataset in selected:
        prepare_dataset(dataset, root=root, force=args.force)
    print("[datasets] all requested datasets are ready")


def prepare_dataset(dataset: DatasetDownload, *, root: Path, force: bool) -> None:
    archive_path = root / dataset.archive_relative_path
    print(f"\n[datasets] {dataset.title}")
    print(f"[datasets] source: {dataset.url}")
    download_file(dataset.url, archive_path, force=force)
    if dataset.key == BCICIV_1_ASC.key:
        extract_bciciv_1_ascii(archive_path, root / "BCICIV_1_asc", force=force)
        validate_bciciv_1_ascii(root / "BCICIV_1_asc")
    elif dataset.key == BNCI2014_004.key:
        extract_bnci2014_004(archive_path, root / "BNCI2014_004" / "gdf", force=force)
        validate_bnci2014_004(root / "BNCI2014_004" / "gdf")


def download_file(url: str, destination: Path, *, force: bool) -> None:
    if destination.exists() and destination.stat().st_size > 0 and not force:
        if _is_valid_zip(destination):
            print(f"[datasets] archive exists, reusing: {destination}")
            return
        print(f"[datasets] existing archive is incomplete, downloading again: {destination}")

    destination.parent.mkdir(parents=True, exist_ok=True)
    request = Request(url, headers={"User-Agent": USER_AGENT})
    print(f"[datasets] downloading to: {destination}")
    try:
        with urlopen(request, timeout=60) as response, destination.open("wb") as output:
            total = int(response.headers.get("Content-Length", "0"))
            downloaded = 0
            while True:
                chunk = response.read(CHUNK_SIZE)
                if not chunk:
                    break
                output.write(chunk)
                downloaded += len(chunk)
                _print_progress(downloaded, total)
        print()
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def extract_bciciv_1_ascii(archive_path: Path, destination: Path, *, force: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    members = _matching_members(archive_path, ASCII_NAME)
    _extract_flat(archive_path, members, destination, force=force)


def extract_bnci2014_004(archive_path: Path, destination: Path, *, force: bool) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    members = _matching_members(archive_path, GDF_NAME)
    _extract_flat(archive_path, members, destination, force=force)


def validate_bciciv_1_ascii(data_dir: Path) -> None:
    expected = {
        f"BCICIV_calib_ds1{subject}_{suffix}.txt"
        for subject in "abcdefg"
        for suffix in ("cnt", "mrk", "nfo")
    }
    _validate_expected(data_dir, expected, label="BCICIV_1_asc")


def validate_bnci2014_004(gdf_dir: Path) -> None:
    expected = {
        f"B{subject:02d}{session:02d}T.gdf"
        for subject in range(1, 10)
        for session in range(1, 4)
    }
    _validate_expected(gdf_dir, expected, label="BNCI2014_004 labeled sessions")


def _matching_members(archive_path: Path, pattern: re.Pattern[str]) -> tuple[ZipInfo, ...]:
    try:
        with ZipFile(archive_path) as archive:
            members = tuple(
                member
                for member in archive.infolist()
                if not member.is_dir() and pattern.fullmatch(Path(member.filename).name)
            )
    except BadZipFile as exc:
        raise RuntimeError(f"invalid ZIP archive: {archive_path}") from exc
    if not members:
        raise RuntimeError(f"no expected files found in {archive_path}")
    names = [Path(member.filename).name for member in members]
    if len(names) != len(set(names)):
        raise RuntimeError(f"duplicate file names found in {archive_path}")
    return members


def _extract_flat(
    archive_path: Path,
    members: tuple[ZipInfo, ...],
    destination: Path,
    *,
    force: bool,
) -> None:
    with ZipFile(archive_path) as archive:
        for member in members:
            output_path = destination / Path(member.filename).name
            if (
                output_path.exists()
                and output_path.stat().st_size == member.file_size
                and not force
            ):
                continue
            with archive.open(member) as source, output_path.open("wb") as output:
                shutil.copyfileobj(source, output, length=CHUNK_SIZE)
    print(f"[datasets] extracted {len(members)} files into: {destination}")


def _validate_expected(directory: Path, expected: set[str], *, label: str) -> None:
    missing = sorted(name for name in expected if not (directory / name).is_file())
    if missing:
        preview = ", ".join(missing[:5])
        raise RuntimeError(f"{label} is incomplete; missing {len(missing)} files: {preview}")
    print(f"[datasets] validation passed: {label} ({len(expected)} required files)")


def _is_valid_zip(path: Path) -> bool:
    try:
        with ZipFile(path) as archive:
            return archive.testzip() is None
    except (BadZipFile, OSError):
        return False


def _print_progress(downloaded: int, total: int) -> None:
    downloaded_mib = downloaded / (1024 * 1024)
    if total > 0:
        total_mib = total / (1024 * 1024)
        percent = downloaded / total * 100.0
        message = f"\r[datasets] {downloaded_mib:8.1f}/{total_mib:.1f} MiB ({percent:5.1f}%)"
    else:
        message = f"\r[datasets] {downloaded_mib:8.1f} MiB"
    print(message, end="", flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[datasets] cancelled", file=sys.stderr)
        raise SystemExit(130)
