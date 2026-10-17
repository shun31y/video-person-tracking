import io
import json
import tarfile
import tempfile
import unittest
from pathlib import Path

from track_people import iter_archive_clips, main, parse_args, validate_archives


CLIP_ID = "c7c73903476d4ea382ea07e827125e88"
SOURCE = f"part_001/c7/c7/{CLIP_ID}.mp4"


def make_archive(path: Path, member_name: str = SOURCE) -> None:
    with tarfile.open(path, "w:gz") as tar:
        for name, content in (("README.txt", b"ignore"), (member_name, b"video bytes")):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))


class ArchiveTests(unittest.TestCase):
    def test_stream_and_resume(self):
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            archive = root / "part_001.tar.gz"
            output = root / "masks"
            make_archive(archive)
            clips = list(iter_archive_clips(archive, 1, output))
            self.assertEqual(len(clips), 1)
            self.assertEqual((clips[0].clip_id, clips[0].source, clips[0].video),
                             (CLIP_ID, SOURCE, b"video bytes"))

            completed = output / "part_001" / CLIP_ID
            completed.mkdir(parents=True)
            (completed / "metadata.json").write_text(json.dumps({"status": "tracked"}))
            stats = {"skipped": 0}
            self.assertEqual(list(iter_archive_clips(archive, 1, output, stats)), [])
            self.assertEqual(stats["skipped"], 1)

    def test_rejects_unexpected_video_path(self):
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            archive = root / "part_001.tar.gz"
            make_archive(archive, f"part_002/c7/c7/{CLIP_ID}.mp4")
            with self.assertRaisesRegex(ValueError, "Unexpected MP4 path"):
                list(iter_archive_clips(archive, 1, root / "output"))

    def test_missing_selected_archive(self):
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            (root / "part_001.tar.gz").touch()
            with self.assertRaisesRegex(FileNotFoundError, "part_002.tar.gz"):
                validate_archives(root, range(1, 3))


class CliTests(unittest.TestCase):
    def test_dry_run_does_not_require_gpu(self):
        with tempfile.TemporaryDirectory() as root_name:
            root = Path(root_name)
            (root / "part_001.tar.gz").touch()
            result = main(["--dataset", str(root), "--start-part", "1", "--end-part", "1",
                           "--workers", "2", "--dry-run"])
            self.assertEqual(result, 0)

    def test_invalid_part_range(self):
        with self.assertRaises(SystemExit):
            parse_args(["--dataset", "/tmp", "--start-part", "8", "--end-part", "7",
                        "--workers", "1"])


if __name__ == "__main__":
    unittest.main()
