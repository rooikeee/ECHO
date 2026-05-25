import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch

DATA_DIR = Path("data")


@dataclass(order=True, frozen=True)
class DataInfo:
    req_id: int
    layer_id: int
    step: int

    def to_tuple(self) -> tuple[int, int, int]:
        return (self.req_id, self.layer_id, self.step)

    def __str__(self):
        return f"req_id: {self.req_id}, layer_id: {self.layer_id}, step: {self.step}"


@dataclass
class DataFile:
    info: DataInfo
    path: Path


@dataclass
class RequestData:
    req_id: int
    data_files: List[DataFile]


@functools.lru_cache(maxsize=128)
def _load_tensor_from_file_cached(file_path: Path) -> torch.Tensor:
    """Loads a tensor from a file and caches the result."""
    return torch.load(file_path, map_location="cpu")


class DataLoader:
    def __init__(self, data_dir: Path | str = DATA_DIR):
        self.data_dir = Path(data_dir)
        self._data_files: list[Path] = sorted(list(self.data_dir.rglob("*.pt")), key=self._parse_filename)
        print(f"Found {len(self._data_files)} data files.")

        self._metadata: list[DataInfo] = [self._parse_filename(p) for p in self._data_files]
        self._file_map: dict[tuple[int, int, int], Path] = {
            meta.to_tuple(): path for meta, path in zip(self._metadata, self._data_files)
        }
        self._group_by_request: dict[int, RequestData] = self._group_by_request()

    @staticmethod
    def _parse_filename(file_path: Path) -> DataInfo:
        parts = file_path.parts
        req_id = int(parts[-2])
        filename = file_path.stem
        layer_part, step_part = filename.split("_")
        layer_id = int(layer_part.replace("layer", ""))
        step = int(step_part.replace("step", ""))
        return DataInfo(req_id=req_id, step=step, layer_id=layer_id)

    def _load_tensor_from_file(self, file_path: Path) -> torch.Tensor:
        return _load_tensor_from_file_cached(file_path)

    def get_metadata(self) -> List[DataInfo]:
        """Returns a list of all data metadata."""
        return self._metadata

    def load_from_info(self, info: DataInfo) -> torch.Tensor:
        """Loads a tensor given a DataInfo object."""
        return self.load_from_tuple(info.to_tuple())

    def load_from_tuple(self, key: tuple[int, int, int]) -> torch.Tensor:
        """Loads a tensor given a (req_id, layer_id, step) tuple."""
        file_path = self._file_map.get(key)
        if file_path is None:
            raise KeyError(f"Data for {key} not found.")
        return self._load_tensor_from_file(file_path)

    def __iter__(self) -> Iterable[tuple[DataInfo, torch.Tensor]]:
        for info in self._metadata:
            yield info, self.load_from_info(info)

    def __len__(self) -> int:
        return len(self._data_files)

    def _group_by_request(self) -> dict[int, RequestData]:
        request_map: dict[int, RequestData] = {}
        for info in self._metadata:
            if info.req_id not in request_map:
                request_map[info.req_id] = RequestData(req_id=info.req_id, data_files=[])
            file_path = self._file_map[info.to_tuple()]
            request_map[info.req_id].data_files.append(DataFile(info=info, path=file_path))

        for request_data in request_map.values():
            request_data.data_files.sort(key=lambda df: (df.info.layer_id, df.info.step))

        return request_map

    def get_requests(self) -> List[RequestData]:
        """Returns a list of RequestData objects grouped by req_id."""
        return list(self._group_by_request.values())


if __name__ == "__main__":
    dataloader = DataLoader()

    print(f"\nTotal data items: {len(dataloader)}")

    print("\nFirst 5 metadata items:")
    for metadata in dataloader.get_metadata()[:5]:
        print(metadata)

    if dataloader.get_metadata():
        sample_info = dataloader.get_metadata()[0]
        print(f"\nLoading data for: {sample_info}")
        key = sample_info.to_tuple()
        tensor = dataloader.load_from_tuple(key)
        print(f"Loaded tensor shape: {tensor.shape}")

        print("\nLoading the same data again (should be cached):")
        tensor_cached = dataloader.load_from_tuple(key)
        print(f"Loaded tensor shape: {tensor_cached.shape}")

    print("\nIterating through the first 2 items in the dataloader:")
    for i, (info, tensor) in enumerate(dataloader):
        if i >= 2:
            break
        print(f"Loaded data: {info}, tensor shape: {tensor.shape}")
