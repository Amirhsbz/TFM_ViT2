import os
import tempfile
from pathlib import Path

import cv2
import numpy as np


class H5TrajectoryWriter:
    DEPTH_DISPLAY_MIN_MM = 200
    DEPTH_DISPLAY_MAX_MM = 1500

    def __init__(
        self,
        file_path: Path,
        video_fps: float,
        compression: str = "gzip",
        compression_level: int = 4,
        metadata: dict | None = None,
    ):
        try:
            import h5py
        except ImportError as exc:
            raise RuntimeError(
                "Saving to H5 requires the 'h5py' package in the active Python environment."
            ) from exc

        self._h5py = h5py
        self.file_path = file_path
        self.file = h5py.File(file_path, "w")
        self.frames_group = self.file.create_group("frames")
        self.videos_group = self.file.create_group("videos")
        self.datasets = {}
        self.frame_count = 0
        self.compression = compression
        self.compression_level = compression_level
        self.video_fps = float(video_fps)
        self.timestamps = self.file.create_dataset(
            "timestamps",
            shape=(0,),
            maxshape=(None,),
            chunks=(256,),
            dtype=h5py.string_dtype(encoding="utf-8"),
        )
        self.video_tempfiles: dict[str, tempfile.NamedTemporaryFile] = {}
        self.video_writers: dict[str, cv2.VideoWriter] = {}
        self.video_source_kinds: dict[str, str] = {}
        self.file.attrs["video_fps"] = self.video_fps
        if metadata:
            for key, value in metadata.items():
                self.file.attrs[key] = value

    @staticmethod
    def _pack_depth_frame(array: np.ndarray) -> tuple[np.ndarray, dict[str, str]]:
        if array.dtype == np.uint16:
            depth_mm = array.astype(np.float32)
        elif array.dtype == np.uint8:
            depth_mm = array.astype(np.float32)
        else:
            raise ValueError(f"Unsupported depth dtype for video export: {array.dtype}")

        valid = depth_mm > 0
        depth_clipped = np.clip(
            depth_mm,
            H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM,
            H5TrajectoryWriter.DEPTH_DISPLAY_MAX_MM,
        )
        depth_scaled = (
            (depth_clipped - H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM)
            * 255.0
            / (
                H5TrajectoryWriter.DEPTH_DISPLAY_MAX_MM
                - H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM
            )
        )
        depth_uint8 = np.zeros(array.shape, dtype=np.uint8)
        depth_uint8[valid] = depth_scaled[valid].astype(np.uint8)
        depth_rgb = np.repeat(depth_uint8[..., None], 3, axis=-1)
        return (
            depth_rgb,
            {
                "source_kind": "depth",
                "source_dtype": str(array.dtype),
                "encoding": "depth_uint8_gray",
                "depth_min_mm": H5TrajectoryWriter.DEPTH_DISPLAY_MIN_MM,
                "depth_max_mm": H5TrajectoryWriter.DEPTH_DISPLAY_MAX_MM,
            },
        )

    @staticmethod
    def _iter_video_streams(key: str, array: np.ndarray):
        if key.endswith("_rgb") and array.dtype == np.uint8:
            if array.ndim == 3:
                yield key, array, {"source_kind": "rgb", "source_dtype": "uint8"}
                return
            if array.ndim == 4 and array.shape[-1] == 3:
                for idx, frame in enumerate(array):
                    yield (
                        f"{key}_{idx}",
                        frame,
                        {"source_kind": "rgb", "source_dtype": "uint8", "source_index": idx},
                    )
            return

        if key.endswith("_depth"):
            if array.ndim == 2:
                packed, attrs = H5TrajectoryWriter._pack_depth_frame(array)
                yield key, packed, attrs
                return
            if array.ndim == 3:
                for idx, frame in enumerate(array):
                    packed, attrs = H5TrajectoryWriter._pack_depth_frame(frame)
                    attrs = dict(attrs)
                    attrs["source_index"] = idx
                    yield f"{key}_{idx}", packed, attrs

    def _append_video_frame(
        self,
        key: str,
        array: np.ndarray,
        extra_attrs: dict | None = None,
    ) -> None:
        writer = self.video_writers.get(key)
        if writer is None:
            # MJPG (Motion JPEG) is always available in OpenCV on all platforms.
            temp_file = tempfile.NamedTemporaryFile(suffix=".avi", delete=False)
            temp_file.close()
            self.video_tempfiles[key] = temp_file
            height, width = array.shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*"MJPG")
            writer = cv2.VideoWriter(temp_file.name, fourcc, self.video_fps, (width, height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to open MJPG writer for key '{key}'")
            self.video_writers[key] = writer
            video_ds = self.videos_group.create_dataset(
                key,
                shape=(0,),
                maxshape=(None,),
                dtype=np.uint8,
            )
            video_ds.attrs["fps"] = self.video_fps
            video_ds.attrs["height"] = height
            video_ds.attrs["width"] = width
            video_ds.attrs["channels"] = array.shape[2]
            video_ds.attrs["codec"] = "MJPG"
            if extra_attrs is not None:
                for attr_key, attr_value in extra_attrs.items():
                    video_ds.attrs[attr_key] = attr_value
            self.video_source_kinds[key] = (
                str(extra_attrs.get("source_kind"))
                if extra_attrs is not None and "source_kind" in extra_attrs
                else "rgb"
            )

        if self.video_source_kinds.get(key) == "depth":
            writer.write(array)
        else:
            writer.write(cv2.cvtColor(array, cv2.COLOR_RGB2BGR))

    def _create_dataset(self, key: str, array: np.ndarray):
        if array.dtype.kind in {"U", "S", "O"} and array.ndim == 0:
            kwargs = {
                "shape": (0,),
                "maxshape": (None,),
                "dtype": self._h5py.string_dtype(encoding="utf-8"),
                "chunks": (256,),
            }
            self.datasets[key] = self.frames_group.create_dataset(key, **kwargs)
            return

        dataset_shape = (0,) + array.shape
        maxshape = (None,) + array.shape
        chunks = (1,) + array.shape if array.ndim > 0 else (256,)
        kwargs = {
            "shape": dataset_shape,
            "maxshape": maxshape,
            "dtype": array.dtype,
            "chunks": chunks,
        }
        if array.dtype.kind not in {"U", "S", "O"}:
            kwargs["compression"] = self.compression
            kwargs["compression_opts"] = self.compression_level
        self.datasets[key] = self.frames_group.create_dataset(key, **kwargs)

    def append(self, timestamp, record: dict[str, np.ndarray]) -> None:
        for key, value in record.items():
            if value is None:
                continue
            array = np.asarray(value)
            video_streams = list(self._iter_video_streams(key, array))
            if video_streams:
                for video_key, video_array, video_attrs in video_streams:
                    self._append_video_frame(video_key, video_array, video_attrs)
                continue

            if key not in self.datasets:
                self._create_dataset(key, array)
            dataset = self.datasets[key]
            if dataset.dtype.metadata is not None and dataset.dtype.metadata.get("vlen") is str:
                if array.ndim != 0:
                    raise ValueError(
                        f"Inconsistent shape/dtype for key '{key}': "
                        f"expected scalar string; got shape {array.shape}, dtype {array.dtype}"
                    )
                dataset.resize(self.frame_count + 1, axis=0)
                item = value.decode("utf-8") if isinstance(value, (bytes, np.bytes_)) else str(value)
                dataset[self.frame_count] = item
                continue
            if dataset.shape[1:] != array.shape or dataset.dtype != array.dtype:
                raise ValueError(
                    f"Inconsistent shape/dtype for key '{key}': "
                    f"expected shape {dataset.shape[1:]}, dtype {dataset.dtype}; "
                    f"got shape {array.shape}, dtype {array.dtype}"
                )
            dataset.resize(self.frame_count + 1, axis=0)
            dataset[self.frame_count] = array

        self.timestamps.resize(self.frame_count + 1, axis=0)
        self.timestamps[self.frame_count] = timestamp.isoformat()
        self.frame_count += 1
        self.file.attrs["frame_count"] = self.frame_count

    def close(self):
        if getattr(self, "file", None) is None:
            return

        for key, writer in self.video_writers.items():
            writer.release()
            temp_file = self.video_tempfiles[key]
            with open(temp_file.name, "rb") as f:
                video_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            dataset = self.videos_group[key]
            dataset.resize((video_bytes.shape[0],))
            dataset[...] = video_bytes
            os.unlink(temp_file.name)
        self.file.flush()
        self.file.close()
        self.file = None
