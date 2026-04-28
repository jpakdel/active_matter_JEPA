# === VENDORED: origin ===
# Upstream repo: https://github.com/helenqu/physical-representation-learning
# Upstream path: physics_jepa/data.py
# Upstream commit: bb77f7b5b506ba793ca7e746e1d0e3c12f70c0db
# Status: trimmed to WellDatasetForJEPA only. Dead siblings removed
# (EmbeddingsDataset, DISCOLatentDataset, WellDatasetForMPP, hydra-style
# loaders) — none of them were referenced by the live D-JEPA pipeline.
# =========================
import torch
from torch.utils.data import Dataset
import h5py
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import weakref
from collections import OrderedDict


class WellDatasetForJEPA(Dataset):
    """
    Auto-discovers HDF5 shards and yields (context, target) windows from full trajectories.

    Assumptions:
      - Each HDF5 file contains one or more objects (top-level groups), each with a
        dataset called `data_key` containing the full trajectory.
      - Trajectory array shape is (H, W, C, T) (time is the last axis).
      - Optional `phys_key` dataset exists per object (any shape).

    Returns:
      dict with 'context': (C,T,H,W), 'target': (C,T,H,W), and 'physical_params' (tensor or empty tensor).
    """

    def __init__(
        self,
        data_dir: str | Path,
        num_frames: int,
        split: str,
        resolution: Optional[Tuple[int, int]] = None,   # (H_out, W_out)
        stride: int = None, # temporal overlap of training examples, default is num_frames
        subset_config_path: Optional[str | Path] = None, # path to config file containing subset_indices
        noise_std: float = 0.0, # standard deviation of Gaussian noise to add
        # HDF5 handle/cache tuning:
        max_open_files: int = 6,
        rdcc_nbytes: int = 512 * 1024**2,
        rdcc_nslots: int = 1_000_003,
        rdcc_w0: float = 0.75,
    ):
        if split == "val":
            split = "valid"

        self.data_dir = Path(data_dir) / "data" / split
        self.dataset_name = Path(data_dir).stem
        self.split = split
        self.num_frames = int(num_frames)
        assert self.num_frames > 0
        self.stride = stride
        self.resolution = resolution
        self.noise_std = float(noise_std)
        if self.noise_std > 0:
            print(f"Adding Gaussian noise with std {self.noise_std}", flush=True)

        # Load subset indices if provided
        self.subset_indices = None
        if subset_config_path is not None:
            subset_config_path = Path(subset_config_path)
            if subset_config_path.exists():
                import json
                with open(subset_config_path, 'r') as f:
                    config = json.load(f)
                    self.subset_indices = config.get('subset_indices', None)
                if self.subset_indices is not None:
                    print(f"Loaded {len(self.subset_indices)} subset indices from {subset_config_path}", flush=True)
            else:
                print(f"Warning: subset_config_path {subset_config_path} does not exist, using full dataset", flush=True)

        # Per-worker LRU of open files
        self._open: OrderedDict[int, tuple[h5py.File, dict]] | None = None
        self._max_open_files = int(max_open_files)
        self._rdcc = (int(rdcc_nbytes), int(rdcc_nslots), float(rdcc_w0))

        # Build flat index of (file_id, obj_id, t0) with stride=1 and non-overlapping (ctx,tgt)
        self.index, self.physical_params_idx = self._build_index()
        print(f"Found {len(self.index)} examples", flush=True)
        print(f"Physical params: {self.physical_params_idx}", flush=True)
        self._build_global_field_schema(Path(self.data_dir) / self.index[0][0])

        if len(self.index) == 0:
            raise ValueError("No valid windows found. "
                             "Check num_frames and that trajectories have at least 2*num_frames frames.")

    # ---- Discovery & indexing ----

    def _build_index(self) -> Tuple[List[tuple[int, int, int]], Dict[str, List[np.ndarray]]]:
        """
        Valid start t0 satisfy: t0 + 2*num_frames <= T.
        We step by 1 to allow maximal coverage; ctx=[t0, t0+F), tgt=[t0+F, t0+2F).
        """

        idx: List[tuple[int, int, int]] = []
        physical_params_idx: Dict[str, List[np.ndarray]] = {}

        F = self.num_frames
        paths = sorted(list(self.data_dir.rglob("*.h5")) + list(self.data_dir.rglob("*.hdf5")))

        for path in paths:
            with h5py.File(path, 'r') as f:
                example_scalar_field = f['t0_fields'][list(f['t0_fields'].keys())[0]]
                T = int(example_scalar_field.shape[1]) # expected shape: num_objs t h w
                max_t0 = T - 2 * F
                if max_t0 < 0:
                    continue
                stride = self.stride if self.stride is not None else F
                for obj_id in range(example_scalar_field.shape[0]):
                    for t0 in range(0, max_t0 + 1, stride):  # stride=1
                        idx.append((path.name, obj_id, t0))
                physical_params_idx[path.name] = [f['scalars'][key][()] for key in f['scalars'].keys() if key != "L"] # ignore L for active matter

        return idx, physical_params_idx

    def _build_global_field_schema(self, sample_path):
        field_paths, d_sizes, comp_shapes = [], [], []
        order = ["t0_fields", "t1_fields", "t2_fields"]
        with h5py.File(sample_path, "r") as f:
            for group in order:
                if group in f:
                    for name, ds in f[group].items():
                        if isinstance(ds, h5py.Dataset):
                            if not isinstance(ds, h5py.Dataset):
                                continue
                            comp = tuple(ds.shape[4:])      # () for scalars, (2,) for vectors, (2,2) for tensors...
                            d_sizes.append(int(np.prod(comp) or 1))
                            comp_shapes.append(comp)
                            field_paths.append(f"{group}/{name}")
            # basic sanity checks
            if not field_paths:
                raise RuntimeError(f"No fields found in {sample_path}")
            # take shape/dtype from the first field
            _, _, H, W = f[field_paths[0]].shape # t0_fields has shape (N, T, H, W)
            if self.dataset_name == "shear_flow":
                H = W = 256 # cut shear flow x axis in half to make square
            if self.dataset_name == "rayleigh_benard":
                H = W = 128 # take middle 128x128 square
            dtype = f[field_paths[0]].dtype

        d_sizes = np.asarray(d_sizes, dtype=np.int64)
        chan_offsets = np.concatenate(([0], np.cumsum(d_sizes)))
        self._field_paths = tuple(field_paths)
        self._d_sizes = d_sizes
        self._comp_shapes = comp_shapes
        self._chan_offsets = chan_offsets
        self._C_total = int(chan_offsets[-1])
        self._spatial_shape = (H, W)
        self._dtype = dtype

    # ---- Reading data ----

    def _get_ds_handle(self, f, state, path):
        ds_cache = state.setdefault("ds_cache", {})
        if path in ds_cache:
            return ds_cache[path]
        ds = f[path]  # fast path lookup; avoid tree walks
        try:
            ds.id.set_chunk_cache(self._rdcc[1], self._rdcc[0], self._rdcc[2])
        except Exception:
            pass
        ds_cache[path] = ds
        return ds

    # ---- Dataset API ----

    def __len__(self) -> int:
        if self.subset_indices is not None:
            return len(self.subset_indices)
        return len(self.index)

    def __getitem__(self, i):
        # Use subset index if available, otherwise use direct index
        if self.subset_indices is not None:
            actual_index = self.subset_indices[i]
        else:
            actual_index = i
        file_id, local_obj_idx, t0 = self.index[actual_index]
        F = self.num_frames

        f, state = self._open_file(file_id)  # per-worker LRU open
        H, W = self._spatial_shape
        C = self._C_total

        # Preallocate final outputs once per sample
        ctx = np.empty((F, H, W, C), dtype=self._dtype, order="C")
        tgt = np.empty((F, H, W, C), dtype=self._dtype, order="C")

       # selections: time-contiguous 2F slice
        if self.dataset_name == "shear_flow":
            h_slice = slice(None)
            w_slice = slice(0, W)
        elif self.dataset_name == "rayleigh_benard":
            h_slice = slice(192, 192+W)
            w_slice = slice(None)
        else:
            h_slice = slice(None)
            w_slice = slice(None)

        sel_2f_prefix = (local_obj_idx, slice(t0, t0 + 2*F), h_slice, w_slice)

        # per-worker cache of temporary buffers keyed by component shape
        tmp_cache = state.setdefault("twobuf_cache", {})  # e.g., {(): arr(2F,H,W), (2,): arr(2F,H,W,2), (2,2): arr(2F,H,W,2,2)}

        c0 = 0
        for path, dsize, comp_shape in zip(self._field_paths, self._d_sizes, self._comp_shapes):
            c1 = c0 + dsize
            ds = self._get_ds_handle(f, state, path)

             # ensure a reusable temp buffer of shape (2F, H, W, *comp_shape)
            need_shape = (2*F, H, W) + comp_shape
            buf = tmp_cache.get(comp_shape)
            if buf is None or buf.shape != need_shape or buf.dtype != self._dtype:
                buf = np.empty(need_shape, dtype=self._dtype, order="C")
                tmp_cache[comp_shape] = buf

            # build full source sel including component dims
            sel = sel_2f_prefix + (slice(None),) * len(comp_shape)
            ds.read_direct(buf, source_sel=sel)  # one I/O per field

            # flatten component axes to channels view and split into ctx/tgt
            view = buf.reshape(2*F, H, W, dsize)   # no copy; C-order
            c1 = c0 + dsize
            ctx[..., c0:c1] = view[:F]
            tgt[..., c0:c1] = view[F:]
            c0 = c1

        # -> torch (C, T, H, W)
        ctx_t = torch.from_numpy(ctx).permute(3, 0, 1, 2).contiguous()
        tgt_t = torch.from_numpy(tgt).permute(3, 0, 1, 2).contiguous()

        # Optional resize
        if self.resolution is not None and tuple(ctx_t.shape[-2:]) != tuple(self.resolution):
            ctx_t = torch.nn.functional.interpolate(ctx_t, size=self.resolution, mode='bilinear', align_corners=False)
            tgt_t = torch.nn.functional.interpolate(tgt_t, size=self.resolution, mode='bilinear', align_corners=False)

        # Add Gaussian noise if specified
        if self.noise_std > 0:
            noise_ctx = torch.randn_like(ctx_t) * self.noise_std
            noise_tgt = torch.randn_like(tgt_t) * self.noise_std
            ctx_t = ctx_t + noise_ctx
            tgt_t = tgt_t + noise_tgt

        return {"context": ctx_t, "target": tgt_t, "physical_params": torch.tensor(self.physical_params_idx[file_id])}

    # ---- Worker-local file LRU ----

    def _ensure_worker_state(self):
        if self._open is None:
            self._open = OrderedDict()  # file_id -> (h5file, {"_dummy": True})
            weakref.finalize(self, self._close_all)

    def _close_all(self):
        if self._open:
            for (f, _) in self._open.values():
                try:
                    f.close()
                except Exception:
                    pass
            self._open.clear()

    def _open_file(self, file_id: int) -> tuple[h5py.File, dict]:
        self._ensure_worker_state()
        if file_id in self._open:
            f, st = self._open.pop(file_id)
            self._open[file_id] = (f, st)  # move to MRU
            return f, st

        # Evict LRU if needed
        while len(self._open) >= self._max_open_files:
            _, (old_f, _) = self._open.popitem(last=False)
            try:
                old_f.close()
            except Exception:
                pass

        path = self.data_dir / file_id
        f = h5py.File(
            path, mode='r', libver='latest', swmr=True,
            rdcc_nbytes=self._rdcc[0], rdcc_nslots=self._rdcc[1], rdcc_w0=self._rdcc[2]
        )
        st = {}
        self._open[file_id] = (f, st)
        return f, st

    def __getstate__(self):
        # Drop open handles when DataLoader forks
        st = self.__dict__.copy()
        st["_open"] = None
        return st
