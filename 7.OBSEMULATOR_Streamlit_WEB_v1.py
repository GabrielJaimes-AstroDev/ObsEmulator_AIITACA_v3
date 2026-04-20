import os
import re
import sys
import io
import json
import time
import glob
import gc
import hashlib
import importlib.util
import tempfile
import subprocess
import collections
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from urllib.parse import urlparse, parse_qs

import numpy as np
import h5py
import joblib
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
import plotly.graph_objects as go

try:
	from PIL import Image, UnidentifiedImageError
except Exception:
	Image = None
	UnidentifiedImageError = Exception

try:
	from astropy.io import fits
except Exception:
	fits = None


# ======================================================
# DEFAULT CONFIG (single-file, no 4.SYNGEN dependency)
# ======================================================
DEFAULT_MERGED_H5 = ""
DEFAULT_NOISE_NN_H5 = ""
DEFAULT_FILTER_FILE = ""
DEFAULT_GDRIVE_MODELS_LINK = "https://drive.google.com/drive/folders/1uSZRFgBIqytuJqv0-IeYPm5D57lCIZVz?usp=drive_link"

DEFAULT_TARGET_FREQS = [
	84.299,
	110.855,
]

DEFAULT_ALLOW_NEAREST = True
DEFAULT_NOISE_SCALE = 1.0
DEFAULT_PROGRESS_EVERY = 40
DEFAULT_OUTPUT_DIR = os.path.join(tempfile.gettempdir(), "predobs_outputs")
DEFAULT_OUT_PREFIX = "PREDOBS6_FAST"

DEFAULT_PARAM_MAPS_DIR = ""
DEFAULT_PARAM_MAP_FILES = {
	"tex": "CH3OCHO_TRIANGLE_SYNTH_TEX.fits",
	"logn": "CH3OCHO_TRIANGLE_SYNTH_LOGN.fits",
	"velo": "CH3OCHO_TRIANGLE_SYNTH_VELO.fits",
	"fwhm": "CH3OCHO_TRIANGLE_SYNTH_FWHM.fits",
}

DEFAULT_PRED_MODE = "ensemble_mean"
DEFAULT_SELECTED_MODEL_NAME = "GradientBoosting"


def _extract_gdrive_folder_id(url_or_id: str) -> Optional[str]:
	v = str(url_or_id or "").strip()
	if not v:
		return None
	if "/" not in v and "?" not in v and len(v) >= 10:
		return v
	try:
		u = urlparse(v)
		if "drive.google.com" not in (u.netloc or ""):
			return None
		m = re.search(r"/folders/([a-zA-Z0-9_-]+)", u.path or "")
		if m:
			return str(m.group(1))
		q = parse_qs(u.query or "")
		if "id" in q and len(q["id"]) > 0:
			return str(q["id"][0])
	except Exception:
		return None
	return None


def _h5_has_groups_or_datasets(h5_path: str, keys: List[str]) -> bool:
	try:
		with h5py.File(str(h5_path), "r") as hf:
			for k in keys:
				if k not in hf:
					return False
			return True
	except Exception:
		return False


def _is_probable_filter_file(path: str) -> bool:
	if not os.path.isfile(path):
		return False
	ext = os.path.splitext(str(path))[1].lower()
	if ext not in (".txt", ".dat", ".csv"):
		return False
	try:
		d = np.asarray(np.loadtxt(path), dtype=float)
		if d.ndim == 1:
			return d.size >= 8
		return d.shape[1] >= 1 and d.shape[0] >= 4
	except Exception:
		return False


def _detect_model_data_paths(root_dir: str) -> dict:
	result = {
		"signal_models_source": "",
		"noise_models_root": "",
		"filter_file": "",
		"warnings": [],
	}
	if not root_dir or (not os.path.isdir(root_dir)):
		result["warnings"].append("Downloaded folder is missing.")
		return result

	all_files: List[str] = []
	for b, _, files in os.walk(root_dir):
		for n in files:
			all_files.append(os.path.join(b, n))

	h5_files = [p for p in all_files if str(p).lower().endswith(".h5")]
	txt_files = [p for p in all_files if _is_probable_filter_file(p)]

	signal_candidates: List[str] = []
	noise_candidates: List[str] = []
	for p in h5_files:
		if _h5_has_groups_or_datasets(p, ["models"]):
			signal_candidates.append(p)
		if _h5_has_groups_or_datasets(p, ["noise_models"]) or _h5_has_groups_or_datasets(p, ["config_json", "state_dict", "scaler"]):
			noise_candidates.append(p)

	if len(signal_candidates) == 1:
		result["signal_models_source"] = signal_candidates[0]
	elif len(signal_candidates) > 1:
		signal_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
		result["signal_models_source"] = signal_candidates[0]
		result["warnings"].append("Multiple signal model candidates found in Drive folder; newest one was selected.")
	else:
		for p in sorted(set(os.path.dirname(x) for x in h5_files)):
			if len(glob.glob(os.path.join(p, "CH_*_f*GHz", "*", "model", "final_model.joblib"))) > 0:
				result["signal_models_source"] = p
				break

	if len(noise_candidates) == 1:
		result["noise_models_root"] = noise_candidates[0]
	elif len(noise_candidates) > 1:
		noise_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
		result["noise_models_root"] = noise_candidates[0]
		result["warnings"].append("Multiple noise model candidates found in Drive folder; newest one was selected.")
	else:
		noise_dirs = [d for d in sorted(set(os.path.dirname(x) for x in all_files)) if os.path.isfile(os.path.join(d, "final_noise_model.h5"))]
		if noise_dirs:
			result["noise_models_root"] = noise_dirs[0]

	if len(txt_files) == 1:
		result["filter_file"] = txt_files[0]
	elif len(txt_files) > 1:
		txt_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
		result["filter_file"] = txt_files[0]
		result["warnings"].append("Multiple filter-file candidates found in Drive folder; newest one was selected.")

	if not result["signal_models_source"]:
		result["warnings"].append("Signal models source could not be auto-detected in Drive folder.")
	if not result["noise_models_root"]:
		result["warnings"].append("Noise models source could not be auto-detected in Drive folder.")
	if not result["filter_file"]:
		result["warnings"].append("Filter file could not be auto-detected in Drive folder.")
	return result


def _download_gdrive_folder_temp(folder_url_or_id: str) -> Tuple[Optional[str], Optional[str]]:
	folder_id = _extract_gdrive_folder_id(folder_url_or_id)
	if not folder_id:
		return None, "Invalid Google Drive folder link or ID."

	try:
		gdown = __import__("gdown")
	except Exception:
		return None, "Package 'gdown' is required to download from Google Drive. Install it in the Streamlit environment."

	tmp_root = os.path.join(tempfile.gettempdir(), "predobs_gdrive_cache")
	os.makedirs(tmp_root, exist_ok=True)
	dst = os.path.join(tmp_root, f"folder_{folder_id}")
	os.makedirs(dst, exist_ok=True)

	try:
		url = f"https://drive.google.com/drive/folders/{folder_id}"
		try:
			gdown.download_folder(url=url, output=dst, quiet=True, use_cookies=False, remaining_ok=True)
		except TypeError:
			gdown.download_folder(url=url, output=dst, quiet=True, use_cookies=False)
		files_count = 0
		for _, _, files in os.walk(dst):
			files_count += int(len(files))
		if files_count <= 0:
			return None, "Download completed but no files were found in cache folder."
		return dst, None
	except Exception as e:
		return None, f"Google Drive download failed: {e}"


def _save_uploaded_file_to_temp(upload_obj, prefix: str) -> Optional[str]:
	if upload_obj is None:
		return None
	try:
		raw = bytes(upload_obj.getbuffer())
		if not raw:
			return None
		safe_name = os.path.basename(str(getattr(upload_obj, "name", "upload.bin")))
		ext = os.path.splitext(safe_name)[1]
		h = hashlib.md5(raw).hexdigest()[:16]
		root = os.path.join(tempfile.gettempdir(), "predobs_manual_uploads")
		os.makedirs(root, exist_ok=True)
		dst = os.path.join(root, f"{prefix}_{h}{ext if ext else ''}")
		if not os.path.isfile(dst):
			with open(dst, "wb") as f:
				f.write(raw)
		return dst
	except Exception:
		return None


def parse_freq_list(text: str) -> List[float]:
	vals: List[float] = []
	for tok in re.split(r"[\s,;]+", str(text).strip()):
		if not tok:
			continue
		try:
			vals.append(float(tok))
		except Exception:
			pass
	return vals


def parse_channel_freq_from_dirname(channel_dir_name: str) -> Optional[float]:
	m = re.search(r"_f([0-9]+(?:\.[0-9]+)?)GHz", channel_dir_name)
	if not m:
		return None
	try:
		return float(m.group(1))
	except Exception:
		return None


def normalize_model_name(name: str) -> str:
	s = str(name).strip().lower().replace("_", "").replace("-", "").replace(" ", "")
	aliases = {
		"rf": "randomforest",
		"randomforestregressor": "randomforest",
		"gb": "gradientboosting",
		"gradientboostingregressor": "gradientboosting",
		"xgb": "xgboost",
		"lgbm": "lightgbm",
	}
	return aliases.get(s, s)


def inverse_target_transform(y_t, transform_name="none", scale=1.0):
	yy = np.asarray(y_t, dtype=np.float64)
	mode = str(transform_name).strip().lower()
	s = np.asarray(scale, dtype=np.float64)
	s = np.maximum(1e-12, s)
	if mode == "asinh":
		return s * np.sinh(yy)
	if mode == "tanh":
		z = np.clip(yy, -0.999999, 0.999999)
		return s * np.arctanh(z)
	if mode == "arctanh":
		return s * np.tanh(yy)
	return yy


def _apply_velocity_shift_to_frequency(freq_ghz: np.ndarray, velocity_kms: float) -> np.ndarray:
	c_kms = 299792.458
	return np.asarray(freq_ghz, dtype=np.float64) * (1.0 - float(velocity_kms) / c_kms)


def _apply_velocity_shift_by_spw_center(freq_ghz: np.ndarray, velocity_kms: float) -> np.ndarray:
	f = np.asarray(freq_ghz, dtype=np.float64)
	if f.size == 0:
		return f
	spw_center_ghz = float(0.5 * (np.nanmin(f) + np.nanmax(f)))
	c_kms = 299792.458
	delta_f = -float(spw_center_ghz) * (float(velocity_kms) / c_kms)
	return f + delta_f


def load_filter_data(path: str) -> Tuple[Optional[np.ndarray], np.ndarray]:
	d = np.asarray(np.loadtxt(path), dtype=float)
	if d.ndim == 1:
		return None, d
	ff = np.asarray(d[:, 0], dtype=float)
	mm = np.asarray(d[:, 1], dtype=float)
	if np.nanmedian(np.abs(ff)) > 1e6:
		ff = ff / 1e9
	return ff, mm


def remap_filter_mask_to_axis(filter_freq_ghz: Optional[np.ndarray], filter_mask: np.ndarray, target_freq_ghz: np.ndarray) -> np.ndarray:
	fm = np.asarray(filter_mask, dtype=float)
	tf = np.asarray(target_freq_ghz, dtype=float)
	if fm.shape[0] == tf.shape[0]:
		return (fm > 0.5).astype(int)
	if filter_freq_ghz is None:
		raise ValueError("Cannot remap filter without filter frequency axis")
	ff = np.asarray(filter_freq_ghz, dtype=float)
	valid = np.isfinite(ff) & np.isfinite(fm)
	ff = ff[valid]
	fm = fm[valid]
	if ff.size < 2:
		return np.zeros((len(tf),), dtype=int)
	order = np.argsort(ff)
	ff = ff[order]
	fm = fm[order]
	ff_u, idx_u = np.unique(ff, return_index=True)
	fm_u = fm[idx_u]
	mapped = np.interp(tf, ff_u, fm_u, left=0.0, right=0.0)
	return (mapped > 0.5).astype(int)


def get_regions_from_mask(mask: np.ndarray, freq_axis_ghz: Optional[np.ndarray] = None, split_on_gap: bool = True, gap_factor: float = 20.0) -> List[Tuple[int, int]]:
	m = np.asarray(mask, dtype=bool)
	if not np.any(m):
		return []
	idx = np.where(m)[0]
	starts = [idx[0]]
	ends = []
	for i in range(1, len(idx)):
		if idx[i] != idx[i - 1] + 1:
			ends.append(idx[i - 1])
			starts.append(idx[i])
	ends.append(idx[-1])
	regions = [(int(a), int(b)) for a, b in zip(starts, ends)]
	if (not split_on_gap) or (freq_axis_ghz is None) or (len(regions) == 0):
		return regions
	f = np.asarray(freq_axis_ghz, dtype=np.float64)
	out: List[Tuple[int, int]] = []
	gf = float(max(1.0, gap_factor))
	for a, b in regions:
		if b <= a:
			out.append((a, b))
			continue
		seg = np.asarray(f[a:b + 1], dtype=np.float64)
		d = np.abs(np.diff(seg))
		dpos = d[np.isfinite(d) & (d > 0.0)]
		if dpos.size == 0:
			out.append((a, b))
			continue
		med = float(np.median(dpos))
		thr = gf * med
		jump_local = np.where(d > thr)[0]
		if jump_local.size == 0:
			out.append((a, b))
			continue
		s = int(a)
		for j in jump_local.tolist():
			e = int(a + j)
			if e >= s:
				out.append((s, e))
			s = int(e + 1)
		if s <= int(b):
			out.append((s, int(b)))
	return out


def pick_roi_by_target_frequency(regions: List[Tuple[int, int]], freq_axis_ghz: np.ndarray, target_frequency_ghz: float, allow_nearest: bool = True):
	f = np.asarray(freq_axis_ghz, dtype=float)
	t = float(target_frequency_ghz)
	for ridx, (a, b) in enumerate(regions, start=1):
		lo = min(float(f[a]), float(f[b]))
		hi = max(float(f[a]), float(f[b]))
		if lo <= t <= hi:
			return (a, b)
	if allow_nearest and regions:
		best = None
		for ridx, (a, b) in enumerate(regions, start=1):
			c = 0.5 * (float(f[a]) + float(f[b]))
			d = abs(c - t)
			if best is None or d < best[0]:
				best = (d, ridx, a, b)
		if best is not None:
			_, _, a, b = best
			return (a, b)
	return None


def list_h5_models(h5_path: str):
	out = {}
	with h5py.File(h5_path, "r") as hf:
		grp = hf.get("models")
		if grp is None:
			return []
		def visitor(name, obj):
			if not isinstance(obj, h5py.Dataset):
				return
			ds_path = str(obj.name).strip("/")
			if not ds_path.endswith("/joblib"):
				return
			parts = ds_path.split("/")
			if len(parts) < 4 or parts[0] != "models":
				return
			ch_name = str(parts[1])
			model_name = str(parts[2])
			fch = parse_channel_freq_from_dirname(ch_name)
			if fch is None:
				return
			out.setdefault(ch_name, {"fch": float(fch), "models": []})
			out[ch_name]["models"].append((model_name, ds_path))
		grp.visititems(visitor)
	rows = []
	for ch_name, val in out.items():
		rows.append((ch_name, float(val["fch"]), sorted(val["models"], key=lambda t: t[0].lower())))
	return sorted(rows, key=lambda t: t[1])


def list_folder_models(models_root: str):
	rows = []
	ch_dirs = sorted(glob.glob(os.path.join(models_root, "CH_*_f*GHz")))
	for chd in ch_dirs:
		fch = parse_channel_freq_from_dirname(os.path.basename(chd))
		if fch is None:
			continue
		mps = sorted(glob.glob(os.path.join(chd, "*", "model", "final_model.joblib")))
		if not mps:
			continue
		items = []
		for mp in mps:
			model_name = os.path.basename(os.path.dirname(os.path.dirname(mp)))
			items.append((model_name, mp))
		rows.append((os.path.basename(chd), float(fch), items))
	return sorted(rows, key=lambda t: t[1])


def load_joblib_package_from_h5(h5_path: str, dataset_path: str):
	with h5py.File(h5_path, "r") as hf:
		blob = np.asarray(hf[dataset_path], dtype=np.uint8).tobytes()
	return joblib.load(io.BytesIO(blob))


class NoiseNN(nn.Module):
	def __init__(self, input_size, hidden_sizes, output_size, dropout_rate=0.2):
		super().__init__()
		layers = []
		prev = int(input_size)
		for h in hidden_sizes:
			layers.append(nn.Linear(prev, int(h)))
			layers.append(nn.BatchNorm1d(int(h)))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(float(dropout_rate)))
			prev = int(h)
		layers.append(nn.Linear(prev, int(output_size)))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)


def _infer_noisenn_feature_spec(cfg_noise: dict):
	feature_spec = cfg_noise.get("feature_spec", {}) if isinstance(cfg_noise, dict) else {}
	n_spw_total = int(feature_spec.get("n_spw_total", len(cfg_noise.get("cube_files", [])) if isinstance(cfg_noise, dict) else 0))
	if n_spw_total <= 0:
		n_spw_total = 1
	input_size = int(cfg_noise.get("input_size", 0))
	output_size = int(cfg_noise.get("output_size", 0))
	extra_from_sizes = int(input_size - output_size) if (input_size > 0 and output_size > 0 and input_size >= output_size) else None
	if "use_spw_onehot" in feature_spec or "use_synth_stats_features" in feature_spec:
		use_onehot = bool(feature_spec.get("use_spw_onehot", False))
		use_stats = bool(feature_spec.get("use_synth_stats_features", False))
		spw_feat_dim = n_spw_total if use_onehot else 1
		extra_feat_dim = int(spw_feat_dim + 4 + (4 if use_stats else 0))
		return use_onehot, use_stats, n_spw_total, extra_feat_dim
	if extra_from_sizes is not None:
		for use_onehot in (True, False):
			spw_feat_dim = n_spw_total if use_onehot else 1
			for use_stats in (True, False):
				extra = int(spw_feat_dim + 4 + (4 if use_stats else 0))
				if extra == extra_from_sizes:
					return use_onehot, use_stats, n_spw_total, extra
	return False, False, n_spw_total, 5


def _load_noisenn_from_hf(hf):
	cfg_raw = hf["config_json"][()]
	cfg_json = cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray)) else cfg_raw
	cfg_noise = json.loads(cfg_json)
	output_size = int(cfg_noise.get("output_size", 0))
	if output_size <= 0:
		output_size = int(np.array(hf["scaler"]["mean_"]).shape[0])
	model = NoiseNN(
		input_size=int(cfg_noise["input_size"]),
		hidden_sizes=list(cfg_noise["hidden_sizes"]),
		output_size=output_size,
		dropout_rate=float(cfg_noise.get("dropout_rate", 0.2)),
	).cpu()
	sd = collections.OrderedDict()
	for k in hf["state_dict"].keys():
		sd[k] = torch.from_numpy(np.array(hf["state_dict"][k]))
	model.load_state_dict(sd)
	model.eval()
	scaler_y = StandardScaler()
	scaler_y.mean_ = np.array(hf["scaler"]["mean_"])
	scaler_y.scale_ = np.array(hf["scaler"]["scale_"])
	scaler_y.var_ = np.array(hf["scaler"].get("var_", scaler_y.scale_ ** 2))
	nfi = np.array(hf["scaler"]["n_features_in_"])
	scaler_y.n_features_in_ = int(nfi[0]) if np.ndim(nfi) > 0 else int(nfi)
	return model, scaler_y, cfg_noise


def load_noisenn_h5(h5_path: str):
	with h5py.File(h5_path, "r") as hf:
		return _load_noisenn_from_hf(hf)


def load_noisenn_h5_bytes(h5_bytes: bytes):
	with h5py.File(io.BytesIO(h5_bytes), "r") as hf:
		return _load_noisenn_from_hf(hf)


def _is_valid_noise_source(noise_source: str) -> bool:
	if not noise_source:
		return False
	s = str(noise_source)
	return os.path.isdir(s) or (os.path.isfile(s) and s.lower().endswith(".h5"))


def _list_noise_model_entries(noise_source: str) -> List[dict]:
	if not _is_valid_noise_source(noise_source):
		return []
	s = str(noise_source)
	if os.path.isdir(s):
		noise_files = sorted(glob.glob(os.path.join(s, "**", "final_noise_model.h5"), recursive=True))
		noise_files = [p for p in noise_files if os.path.isfile(p)]
		return [{"kind": "file", "path": p, "display": p} for p in noise_files]

	entries: List[dict] = []
	try:
		with h5py.File(s, "r") as hf:
			grp = hf.get("noise_models")
			if isinstance(grp, h5py.Group):
				cfg_grp = hf.get("noise_cfg_json")
				for name in sorted(grp.keys()):
					obj = grp.get(name)
					if not isinstance(obj, h5py.Dataset):
						continue
					entry = {
						"kind": "bundle",
						"bundle_path": s,
						"dataset_path": str(obj.name),
						"display": str(obj.attrs.get("rel_path", name)),
					}
					if isinstance(cfg_grp, h5py.Group) and (name in cfg_grp):
						entry["cfg_dataset_path"] = str(cfg_grp[name].name)
					entries.append(entry)
				return entries

			if ("config_json" in hf) and ("state_dict" in hf) and ("scaler" in hf):
				entries.append({"kind": "file", "path": s, "display": os.path.basename(s)})
	except Exception:
		return []
	return entries


def _read_noise_cfg_from_entry(entry: dict) -> Optional[dict]:
	try:
		if str(entry.get("kind", "")) == "file":
			with h5py.File(str(entry["path"]), "r") as hf:
				cfg_raw = hf["config_json"][()]
				cfg_json = cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray)) else cfg_raw
				return json.loads(cfg_json)

		bundle_path = str(entry["bundle_path"])
		cfg_dataset_path = entry.get("cfg_dataset_path", None)
		if cfg_dataset_path:
			with h5py.File(bundle_path, "r") as hf:
				cfg_raw = hf[str(cfg_dataset_path)][()]
				cfg_json = cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray, np.bytes_)) else str(cfg_raw)
				return json.loads(cfg_json)

		with h5py.File(bundle_path, "r") as hf:
			blob = np.asarray(hf[str(entry["dataset_path"])], dtype=np.uint8).tobytes()
		with h5py.File(io.BytesIO(blob), "r") as hf_inner:
			cfg_raw = hf_inner["config_json"][()]
			cfg_json = cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray)) else cfg_raw
			return json.loads(cfg_json)
	except Exception:
		return None


def _load_noisenn_from_entry(entry: dict):
	if str(entry.get("kind", "")) == "file":
		return load_noisenn_h5(str(entry["path"]))
	with h5py.File(str(entry["bundle_path"]), "r") as hf:
		blob = np.asarray(hf[str(entry["dataset_path"])], dtype=np.uint8).tobytes()
	return load_noisenn_h5_bytes(blob)


def get_noise_segments_for_axis(cfg_noise: dict, freq_axis_ghz: np.ndarray) -> List[Tuple[np.ndarray, int]]:
	out = []
	f = np.asarray(freq_axis_ghz, dtype=np.float64)
	roi_sel = cfg_noise.get("roi_selection_summary", {}) if isinstance(cfg_noise, dict) else {}
	roi_detail = roi_sel.get("roi_detail", {}) if isinstance(roi_sel, dict) else {}
	spw_det = roi_detail.get("spw", {}) if isinstance(roi_detail, dict) else {}
	if not isinstance(spw_det, dict):
		return []
	for spw_key, info in spw_det.items():
		if not isinstance(info, dict):
			continue
		fmin = info.get("f_min_ghz", None)
		fmax = info.get("f_max_ghz", None)
		if fmin is None or fmax is None:
			continue
		lo = float(min(float(fmin), float(fmax)))
		hi = float(max(float(fmin), float(fmax)))
		idx = np.where((f >= lo) & (f <= hi))[0]
		if idx.size == 0:
			continue
		try:
			spw_idx = int(spw_key)
		except Exception:
			spw_idx = 1
		out.append((idx.astype(int), int(spw_idx)))
	return out


def load_map_2d(path, blank_value=-1000.0):
	arr = np.asarray(fits.getdata(path), dtype=np.float32)
	if arr.ndim == 4:
		arr = arr[0, 0, :, :]
	elif arr.ndim == 3:
		arr = arr[0, :, :]
	elif arr.ndim != 2:
		raise ValueError(f"Unsupported map shape: {arr.shape}")
	arr[np.isclose(arr, float(blank_value))] = np.nan
	arr[~np.isfinite(arr)] = np.nan
	return arr


def _resample_rows_by_index(y2d: np.ndarray, target_len: int) -> np.ndarray:
	y2d = np.asarray(y2d, dtype=np.float32)
	n, cur_len = int(y2d.shape[0]), int(y2d.shape[1])
	target_len = int(target_len)
	if cur_len == target_len:
		return y2d
	x_old = np.linspace(0.0, 1.0, cur_len, dtype=np.float32)
	x_new = np.linspace(0.0, 1.0, target_len, dtype=np.float32)
	out = np.empty((n, target_len), dtype=np.float32)
	for i in range(n):
		out[i] = np.interp(x_new, x_old, y2d[i]).astype(np.float32)
	return out


def predict_with_joblib_package_batch(package, x_features_2d: np.ndarray) -> np.ndarray:
	model = package["model"]
	scaler_x = package["scaler_x"]
	transform_name = str(package.get("target_transform", "none"))
	transform_scale = float(package.get("target_transform_scale", 1.0))
	x = np.asarray(x_features_2d, dtype=np.float32)
	x_n = scaler_x.transform(x)
	y_t = np.asarray(model.predict(x_n), dtype=np.float64).reshape(-1, 1)
	y = np.asarray(inverse_target_transform(y_t, transform_name=transform_name, scale=transform_scale), dtype=np.float32).reshape(-1)
	return y


def predict_noise_segment_batch(model, scaler_y, cfg_noise: dict, y_synth_segment_batch: np.ndarray, x_features_batch: np.ndarray, spw_idx: int, noise_scale: float = 1.0, batch_size: int = 2048) -> np.ndarray:
	input_size = int(cfg_noise["input_size"])
	output_size = int(cfg_noise.get("output_size", 0))
	use_onehot, use_stats, n_spw_total, extra_feat_dim = _infer_noisenn_feature_spec(cfg_noise)
	expected_spec_len = int(input_size - extra_feat_dim)
	if expected_spec_len <= 0 and output_size > 0:
		expected_spec_len = int(output_size)
	if expected_spec_len <= 0:
		raise RuntimeError("Invalid expected spectral length from NoiseNN config")
	ys = np.asarray(y_synth_segment_batch, dtype=np.float32)
	n, seg_len = int(ys.shape[0]), int(ys.shape[1])
	ys_model = _resample_rows_by_index(ys, expected_spec_len)
	mn = np.nanmin(ys_model, axis=1, keepdims=True)
	mx = np.nanmax(ys_model, axis=1, keepdims=True)
	den = np.maximum(mx - mn, 1e-8)
	ys_norm = (ys_model - mn) / den
	flat_mask = (mx <= mn).reshape(-1)
	if np.any(flat_mask):
		ys_norm[flat_mask, :] = 0.0
	if use_onehot:
		spw_feats = np.zeros((n, int(n_spw_total)), dtype=np.float32)
		if 1 <= int(spw_idx) <= int(n_spw_total):
			spw_feats[:, int(spw_idx) - 1] = 1.0
	else:
		spw_norm = float(int(spw_idx) - 1) / max(1.0, float(int(n_spw_total) - 1))
		spw_feats = np.full((n, 1), spw_norm, dtype=np.float32)
	xfb = np.asarray(x_features_batch, dtype=np.float32)
	tex_col = xfb[:, 1:2]
	logn_col = xfb[:, 0:1]
	velo_col = xfb[:, 2:3]
	fwhm_col = xfb[:, 3:4]
	phys_feats = np.concatenate([tex_col, logn_col, velo_col, fwhm_col], axis=1)
	if use_stats:
		synth_stats = np.stack([np.nanmin(ys_model, axis=1), np.nanmax(ys_model, axis=1), np.nanmean(ys_model, axis=1), np.nanstd(ys_model, axis=1)], axis=1).astype(np.float32)
		extra_feats = np.concatenate([spw_feats, phys_feats, synth_stats], axis=1)
	else:
		extra_feats = np.concatenate([spw_feats, phys_feats], axis=1)
	x_mat = np.concatenate([ys_norm, extra_feats], axis=1).astype(np.float32)
	if x_mat.shape[1] != input_size:
		raise RuntimeError(f"NoiseNN input mismatch: got {x_mat.shape[1]} expected {input_size}")
	pred_scaled = np.empty((n, int(output_size if output_size > 0 else scaler_y.mean_.shape[0])), dtype=np.float32)
	bsz = int(max(1, batch_size))
	with torch.no_grad():
		for i0 in range(0, n, bsz):
			i1 = min(i0 + bsz, n)
			xb = torch.from_numpy(x_mat[i0:i1]).to(torch.float32)
			pred_scaled[i0:i1] = model(xb).cpu().numpy().astype(np.float32)
	noise_model = scaler_y.inverse_transform(pred_scaled).astype(np.float32)
	noise_out = _resample_rows_by_index(noise_model, seg_len)
	noise_out *= float(max(0.0, noise_scale))
	return noise_out


def _set_spectral_header(out_hdr, freq_ghz: np.ndarray):
	f = np.asarray(freq_ghz, dtype=np.float64).reshape(-1)
	out_hdr["CTYPE3"] = "FREQ"
	out_hdr["CUNIT3"] = "Hz"
	out_hdr["CRPIX3"] = 1.0
	out_hdr["CRVAL3"] = float(f[0] * 1e9)
	out_hdr["CDELT3"] = float(np.median(np.diff(f)) * 1e9) if f.size > 1 else 1.0
	out_hdr["SPECSYS"] = "LSRK"


def write_cube_fits(out_fits_path: str, cube_data, roi_freq: np.ndarray, ref_spatial_hdr: Optional[object], history_text: str):
	arr = np.asarray(cube_data, dtype=np.float32)
	hdr = fits.Header()
	hdr["WCSAXES"] = 3
	hdr["BUNIT"] = "a.u."
	if ref_spatial_hdr is not None:
		for k in ["CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CROTA1", "CROTA2", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE"]:
			if k in ref_spatial_hdr:
				hdr[k] = ref_spatial_hdr[k]
		for k in ref_spatial_hdr.keys():
			ks = str(k)
			if ks.startswith("CD1_") or ks.startswith("CD2_") or ks.startswith("PC1_") or ks.startswith("PC2_"):
				hdr[ks] = ref_spatial_hdr[ks]
	else:
		hdr["CTYPE1"] = "X"
		hdr["CTYPE2"] = "Y"
	_set_spectral_header(hdr, roi_freq)
	hdr["HISTORY"] = str(history_text)
	fits.writeto(out_fits_path, arr, header=hdr, overwrite=True)


def save_progress_png(cube_partial: np.ndarray, target_freq: float, done_steps: int, total_steps: int, out_png: str, processed_mask: Optional[np.ndarray] = None):
	integ_map = np.nansum(cube_partial, axis=0)
	fig = plt.figure(figsize=(6, 5))
	plt.imshow(integ_map, origin="lower", cmap="viridis")
	plt.xlabel("x")
	plt.ylabel("y")
	if processed_mask is not None:
		pm = np.asarray(processed_mask, dtype=bool)
		if pm.ndim == 2 and np.any(pm):
			yy, xx = np.where(pm)
			h, w = int(pm.shape[0]), int(pm.shape[1])
			span_y = int(np.max(yy) - np.min(yy) + 1)
			span_x = int(np.max(xx) - np.min(xx) + 1)
			pad = int(max(2, round(0.08 * max(span_y, span_x))))
			y0 = max(0, int(np.min(yy)) - pad)
			y1 = min(h - 1, int(np.max(yy)) + pad)
			x0 = max(0, int(np.min(xx)) - pad)
			x1 = min(w - 1, int(np.max(xx)) + pad)
			plt.xlim(float(x0) - 0.5, float(x1) + 0.5)
			plt.ylim(float(y0) - 0.5, float(y1) + 0.5)
	plt.colorbar(fraction=0.046, pad=0.04)
	plt.tight_layout()
	fig.savefig(out_png, dpi=170)
	plt.close(fig)
	info_path = os.path.splitext(out_png)[0] + ".json"
	info = {
		"title": f"Progress integrated map | target {float(target_freq):.6f} GHz | pixels processed: {int(done_steps)}/{int(total_steps)}",
		"target_freq_ghz": float(target_freq),
		"done_steps": int(done_steps),
		"total_steps": int(total_steps),
	}
	try:
		with open(info_path, "w", encoding="utf-8") as f:
			json.dump(info, f, ensure_ascii=False, indent=2)
	except Exception:
		pass


def _spiral_pixel_order_valid(valid_mask: np.ndarray) -> List[Tuple[int, int]]:
	vm = np.asarray(valid_mask, dtype=bool)
	if vm.ndim != 2:
		return []
	h, w = int(vm.shape[0]), int(vm.shape[1])
	cy, cx = int(h // 2), int(w // 2)
	need = int(h * w)
	seen = set()
	coords_all: List[Tuple[int, int]] = []

	x, y = int(cx), int(cy)
	if 0 <= x < w and 0 <= y < h:
		coords_all.append((y, x))
		seen.add((y, x))

	dirs = [(1, 0), (0, 1), (-1, 0), (0, -1)]
	step_len = 1
	max_steps = int(2 * max(h, w) + 5)
	while len(coords_all) < need and step_len <= max_steps:
		for d_i, (dx, dy) in enumerate(dirs):
			for _ in range(step_len):
				x += int(dx)
				y += int(dy)
				if 0 <= x < w and 0 <= y < h and (y, x) not in seen:
					coords_all.append((y, x))
					seen.add((y, x))
			if d_i % 2 == 1:
				step_len += 1
			if len(coords_all) >= need:
				break

	coords_valid = [(yy, xx) for (yy, xx) in coords_all if vm[yy, xx]]
	return coords_valid


def build_signal_index_for_roi(signal_source: str, filter_file: str, target_frequency_ghz: float, pred_mode: str, selected_model_name: str, allow_nearest: bool):
	is_h5 = os.path.isfile(signal_source) and str(signal_source).lower().endswith(".h5")
	entries = list_h5_models(signal_source) if is_h5 else list_folder_models(signal_source)
	if not entries:
		raise RuntimeError("No signal channels found")
	f_axis = np.asarray([float(f) for _, f, _ in entries], dtype=np.float64)
	ff, fm = load_filter_data(filter_file)
	mask = remap_filter_mask_to_axis(ff, fm, f_axis)
	regions = get_regions_from_mask(mask, freq_axis_ghz=f_axis, split_on_gap=True, gap_factor=20.0)
	if not regions:
		raise RuntimeError("No ROI regions found")
	roi_region = pick_roi_by_target_frequency(regions, f_axis, target_frequency_ghz, allow_nearest=allow_nearest)
	if roi_region is None:
		raise RuntimeError("Could not select ROI")
	a, b = roi_region
	lo = float(min(f_axis[a], f_axis[b]))
	hi = float(max(f_axis[a], f_axis[b]))
	selected_model_norm = normalize_model_name(selected_model_name)
	pm = str(pred_mode).strip().lower()
	roi_entries = []
	for ch_name, fch, model_items in entries:
		if not (lo <= float(fch) <= hi):
			continue
		if pm == "single_model":
			model_items = [(mn, ref) for (mn, ref) in model_items if normalize_model_name(mn) == selected_model_norm]
			if not model_items:
				continue
		roi_entries.append((ch_name, float(fch), model_items))
	if not roi_entries:
		raise RuntimeError("No channels inside ROI")
	roi_entries = sorted(roi_entries, key=lambda t: t[1])
	roi_freq = np.asarray([r[1] for r in roi_entries], dtype=np.float64)
	return is_h5, roi_entries, roi_freq


def _intervals_overlap(lo1: float, hi1: float, lo2: float, hi2: float) -> bool:
	a1, b1 = float(min(lo1, hi1)), float(max(lo1, hi1))
	a2, b2 = float(min(lo2, hi2)), float(max(lo2, hi2))
	return max(a1, a2) <= min(b1, b2)


def _pick_default_roi_index(rois: List[dict], guide_freq_ghz: Optional[float]) -> int:
	if not rois:
		return 0
	if guide_freq_ghz is None:
		return 0
	t = float(guide_freq_ghz)
	for i, r in enumerate(rois):
		if float(r["lo"]) <= t <= float(r["hi"]):
			return int(i)
	best_i = 0
	best_d = None
	for i, r in enumerate(rois):
		c = 0.5 * (float(r["lo"]) + float(r["hi"]))
		d = abs(c - t)
		if best_d is None or d < best_d:
			best_d = d
			best_i = i
	return int(best_i)


def _collect_signal_rois_for_ui(signal_source: str, filter_file: str) -> List[dict]:
	if (not signal_source) or (not filter_file):
		return []
	if ((not os.path.isfile(signal_source)) and (not os.path.isdir(signal_source))) or (not os.path.isfile(filter_file)):
		return []
	is_h5 = os.path.isfile(signal_source) and str(signal_source).lower().endswith(".h5")
	entries = list_h5_models(signal_source) if is_h5 else list_folder_models(signal_source)
	if not entries:
		return []
	f_axis = np.asarray([float(f) for _, f, _ in entries], dtype=np.float64)
	ff, fm = load_filter_data(filter_file)
	mask = remap_filter_mask_to_axis(ff, fm, f_axis)
	regions = get_regions_from_mask(mask, freq_axis_ghz=f_axis, split_on_gap=True, gap_factor=20.0)
	out = []
	for i, (a, b) in enumerate(regions, start=1):
		lo = float(min(f_axis[a], f_axis[b]))
		hi = float(max(f_axis[a], f_axis[b]))
		out.append({"index": int(i), "lo": lo, "hi": hi, "a": int(a), "b": int(b), "overlap": False})
	return out


def _collect_noise_rois_for_ui(noise_models_root: str) -> List[dict]:
	if not _is_valid_noise_source(noise_models_root):
		return []
	noise_entries = _list_noise_model_entries(noise_models_root)
	if not noise_entries:
		return []
	interval_map: Dict[Tuple[float, float], dict] = {}
	for entry in noise_entries:
		try:
			cfg_noise = _read_noise_cfg_from_entry(entry)
			if not isinstance(cfg_noise, dict):
				continue
			roi_sel = cfg_noise.get("roi_selection_summary", {}) if isinstance(cfg_noise, dict) else {}
			roi_detail = roi_sel.get("roi_detail", {}) if isinstance(roi_sel, dict) else {}
			spw_det = roi_detail.get("spw", {}) if isinstance(roi_detail, dict) else {}
			if not isinstance(spw_det, dict):
				continue
			for spw_key, info in spw_det.items():
				if not isinstance(info, dict):
					continue
				fmin = info.get("f_min_ghz", None)
				fmax = info.get("f_max_ghz", None)
				if fmin is None or fmax is None:
					continue
				lo = float(min(float(fmin), float(fmax)))
				hi = float(max(float(fmin), float(fmax)))
				key = (round(lo, 6), round(hi, 6))
				if key not in interval_map:
					interval_map[key] = {
						"lo": lo,
						"hi": hi,
						"spw": set(),
						"n_models": 0,
						"overlap": False,
					}
				interval_map[key]["spw"].add(str(spw_key))
				interval_map[key]["n_models"] = int(interval_map[key]["n_models"]) + 1
		except Exception:
			continue
	out = []
	for i, (_, item) in enumerate(sorted(interval_map.items(), key=lambda kv: (kv[0][0], kv[0][1])), start=1):
		out.append({
			"index": int(i),
			"lo": float(item["lo"]),
			"hi": float(item["hi"]),
			"spw": sorted(list(item["spw"])),
			"n_models": int(item["n_models"]),
			"overlap": False,
		})
	return out


def _mark_roi_overlaps(signal_rois: List[dict], noise_rois: List[dict]) -> Tuple[List[dict], List[dict]]:
	for s in signal_rois:
		s["overlap"] = False
	for n in noise_rois:
		n["overlap"] = False
	for s in signal_rois:
		for n in noise_rois:
			if _intervals_overlap(float(s["lo"]), float(s["hi"]), float(n["lo"]), float(n["hi"])):
				s["overlap"] = True
				n["overlap"] = True
	return signal_rois, noise_rois


def _get_overlapping_noise_roi_indices(signal_roi: dict, noise_rois: List[dict]) -> List[int]:
	out: List[int] = []
	if not isinstance(signal_roi, dict):
		return out
	for n in noise_rois:
		if not isinstance(n, dict):
			continue
		if _intervals_overlap(float(signal_roi["lo"]), float(signal_roi["hi"]), float(n["lo"]), float(n["hi"])):
			out.append(int(n["index"]))
	return out


def _get_overlapping_signal_roi_indices(noise_roi: dict, signal_rois: List[dict]) -> List[int]:
	out: List[int] = []
	if not isinstance(noise_roi, dict):
		return out
	for s in signal_rois:
		if not isinstance(s, dict):
			continue
		if _intervals_overlap(float(noise_roi["lo"]), float(noise_roi["hi"]), float(s["lo"]), float(s["hi"])):
			out.append(int(s["index"]))
	return out


def _roi_center_ghz(roi: dict) -> Optional[float]:
	if not isinstance(roi, dict):
		return None
	try:
		return 0.5 * (float(roi["lo"]) + float(roi["hi"]))
	except Exception:
		return None


def _augment_target_freqs_with_selected_rois(base_freqs: List[float], signal_rois: List[dict], noise_rois: List[dict], selected_signal_pos: Optional[int], selected_noise_pos: Optional[int]) -> List[float]:
	freqs = [float(v) for v in (base_freqs or [])]
	if selected_signal_pos is not None and signal_rois and 0 <= int(selected_signal_pos) < len(signal_rois):
		cs = _roi_center_ghz(signal_rois[int(selected_signal_pos)])
		if cs is not None:
			freqs.append(float(cs))
	if selected_noise_pos is not None and noise_rois and 0 <= int(selected_noise_pos) < len(noise_rois):
		cn = _roi_center_ghz(noise_rois[int(selected_noise_pos)])
		if cn is not None:
			freqs.append(float(cn))
	out: List[float] = []
	for v in sorted(freqs):
		if not out or abs(float(v) - float(out[-1])) > 1e-9:
			out.append(float(v))
	return out


def _append_selected_rois_to_freq_list(base_freqs: List[float], signal_rois: List[dict], noise_rois: List[dict], selected_signal_pos: Optional[int], selected_noise_pos: Optional[int]) -> List[float]:
	freqs = [float(v) for v in (base_freqs or [])]
	if selected_signal_pos is not None and signal_rois and 0 <= int(selected_signal_pos) < len(signal_rois):
		cs = _roi_center_ghz(signal_rois[int(selected_signal_pos)])
		if cs is not None:
			freqs.append(float(cs))
	if selected_noise_pos is not None and noise_rois and 0 <= int(selected_noise_pos) < len(noise_rois):
		cn = _roi_center_ghz(noise_rois[int(selected_noise_pos)])
		if cn is not None:
			freqs.append(float(cn))
	out: List[float] = []
	for v in sorted(freqs):
		if not out or abs(float(v) - float(out[-1])) > 1e-9:
			out.append(float(v))
	return out


def _freqs_to_text(freqs: List[float]) -> str:
	return ", ".join([f"{float(v):.6f}" for v in (freqs or [])])


def _normalize_target_freqs_for_run(freqs: List[float]) -> List[float]:
	out: List[float] = []
	for v in (freqs or []):
		try:
			fv = float(v)
		except Exception:
			continue
		if not np.isfinite(fv):
			continue
		if not any(abs(float(fv) - float(prev)) <= 1e-9 for prev in out):
			out.append(float(fv))
	return out


def _selected_roi_combo_freqs(signal_rois: List[dict], noise_rois: List[dict], selected_signal_pos: Optional[int], selected_noise_pos: Optional[int]) -> List[float]:
	return _append_selected_rois_to_freq_list(
		base_freqs=[],
		signal_rois=signal_rois,
		noise_rois=noise_rois,
		selected_signal_pos=selected_signal_pos,
		selected_noise_pos=selected_noise_pos,
	)


def _plot_roi_overview(signal_rois: List[dict], noise_rois: List[dict], guide_freqs_ghz: Optional[List[float]] = None, selected_combo_freqs_ghz: Optional[List[float]] = None, selected_signal_index: Optional[int] = None, selected_noise_index: Optional[int] = None, chart_key: Optional[str] = None):
	fig = go.Figure()
	for r in signal_rois:
		is_sel = (selected_signal_index is not None) and (int(r["index"]) == int(selected_signal_index))
		color = "#2ca02c" if bool(r.get("overlap", False)) else "#1f77b4"
		width = 12 if is_sel else 8
		fig.add_trace(go.Scatter(
			x=[float(r["lo"]), float(r["hi"])],
			y=[1.0, 1.0],
			mode="lines+markers",
			line=dict(color=color, width=width),
			marker=dict(size=6, color=color),
			name="Synthetic ROI",
			showlegend=False,
			hovertemplate=f"Synthetic ROI {int(r['index'])}<br>{float(r['lo']):.6f} - {float(r['hi']):.6f} GHz<extra></extra>",
		))
	for r in noise_rois:
		is_sel = (selected_noise_index is not None) and (int(r["index"]) == int(selected_noise_index))
		color = "#2ca02c" if bool(r.get("overlap", False)) else "#d62728"
		width = 12 if is_sel else 8
		fig.add_trace(go.Scatter(
			x=[float(r["lo"]), float(r["hi"])],
			y=[0.0, 0.0],
			mode="lines+markers",
			line=dict(color=color, width=width),
			marker=dict(size=6, color=color),
			name="Noise ROI",
			showlegend=False,
			hovertemplate=f"Noise ROI {int(r['index'])}<br>{float(r['lo']):.6f} - {float(r['hi']):.6f} GHz<extra></extra>",
		))
	if guide_freqs_ghz:
		for gf in guide_freqs_ghz:
			fig.add_vline(x=float(gf), line=dict(color="#9467bd", dash="dash"))
	if selected_combo_freqs_ghz:
		for cf in selected_combo_freqs_ghz:
			fig.add_vline(x=float(cf), line=dict(color="#ff7f0e", dash="dash"))
	fig.update_layout(
		title="ROI overview (Signal vs Noise)",
		xaxis_title="Frequency (GHz)",
		yaxis=dict(
			tickmode="array",
			tickvals=[0.0, 1.0],
			ticktext=["Noise ROIs", "Synthetic ROIs"],
			range=[-0.5, 1.5],
		),
		template="plotly_white",
		height=310,
		margin=dict(l=40, r=20, t=45, b=40),
	)
	st.plotly_chart(fig, width="stretch", key=chart_key)


def run_cube_worker(cfg_path: str) -> int:
	if fits is None:
		print("FITS backend not available")
		return 2
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)
	out_dir = str(cfg["out_dir"])
	os.makedirs(out_dir, exist_ok=True)
	param_maps_dir = str(cfg["param_maps_dir"])
	param_map_files = dict(cfg["param_map_files"])
	signal_models_source = str(cfg["signal_models_source"])
	noise_models_root = str(cfg["noise_models_root"])
	filter_file = str(cfg["filter_file"])
	target_freqs = [float(v) for v in cfg["target_freqs"]]
	progress_every = int(max(1, int(cfg.get("progress_every", 40))))
	allow_nearest = bool(cfg.get("allow_nearest", True))
	noise_scale = float(cfg.get("noise_scale", 1.0))
	pred_mode = str(cfg.get("pred_mode", "ensemble_mean"))
	selected_model_name = str(cfg.get("selected_model_name", "GradientBoosting"))
	out_prefix = str(cfg.get("out_prefix", "PREDOBS6_FAST"))

	tex_path = os.path.join(param_maps_dir, param_map_files["tex"])
	ref_hdr = fits.getheader(tex_path) if os.path.isfile(tex_path) else None

	param_maps = {}
	for k in ["tex", "logn", "velo", "fwhm"]:
		p = os.path.join(param_maps_dir, param_map_files[k])
		if not os.path.isfile(p):
			raise FileNotFoundError(f"Missing map: {p}")
		param_maps[k] = load_map_2d(p, blank_value=-1000.0)

	ny, nx = param_maps["tex"].shape
	valid_mask = np.isfinite(param_maps["tex"]) & np.isfinite(param_maps["logn"]) & np.isfinite(param_maps["velo"]) & np.isfinite(param_maps["fwhm"])
	yy_valid, xx_valid = np.where(valid_mask)
	n_valid = int(len(yy_valid))
	if n_valid == 0:
		raise RuntimeError("No valid pixels in parameter maps")
	x_valid = np.stack([
		param_maps["logn"][yy_valid, xx_valid],
		param_maps["tex"][yy_valid, xx_valid],
		param_maps["velo"][yy_valid, xx_valid],
		param_maps["fwhm"][yy_valid, xx_valid],
	], axis=1).astype(np.float32)
	valid_id_map = np.full((ny, nx), -1, dtype=np.int64)
	valid_id_map[yy_valid, xx_valid] = np.arange(n_valid, dtype=np.int64)

	noise_entries = _list_noise_model_entries(noise_models_root)
	if not noise_entries:
		raise FileNotFoundError(f"No noise models found in source: {noise_models_root}")
	noise_models = []
	for entry in noise_entries:
		try:
			m, sy, c = _load_noisenn_from_entry(entry)
			m.eval()
			noise_models.append((m, sy, c))
		except Exception:
			pass
	if not noise_models:
		raise RuntimeError("No valid NoiseNN models loaded")

	signal_pkg_cache: Dict[str, object] = {}
	n_ok = 0
	n_fail = 0
	for target_freq in target_freqs:
		try:
			is_h5_signal, roi_entries, roi_freq = build_signal_index_for_roi(
				signal_source=signal_models_source,
				filter_file=filter_file,
				target_frequency_ghz=float(target_freq),
				pred_mode=pred_mode,
				selected_model_name=selected_model_name,
				allow_nearest=allow_nearest,
			)

			target_tag = f"{float(target_freq):.6f}"
			y_syn_channels: List[np.ndarray] = []
			kept_freqs: List[float] = []
			for _, fch, model_refs in roi_entries:
				pred_acc = np.zeros((n_valid,), dtype=np.float64)
				pred_cnt = 0
				for model_name, ref in model_refs:
					cache_key = f"{model_name}|{ref}"
					try:
						if cache_key not in signal_pkg_cache:
							if is_h5_signal:
								signal_pkg_cache[cache_key] = load_joblib_package_from_h5(signal_models_source, ref)
							else:
								signal_pkg_cache[cache_key] = joblib.load(ref)
						pkg = signal_pkg_cache[cache_key]
						pred = predict_with_joblib_package_batch(pkg, x_valid)
						pred_acc += pred.astype(np.float64)
						pred_cnt += 1
					except Exception as em:
						print(f"[WARN] target {target_tag} channel {float(fch):.6f} model {str(model_name)} failed: {em}")
						continue
				if pred_cnt > 0:
					y_syn_channels.append((pred_acc / float(pred_cnt)).astype(np.float32))
					kept_freqs.append(float(fch))
				else:
					print(f"[WARN] target {target_tag} channel {float(fch):.6f} skipped: no valid signal predictions")

			if not kept_freqs:
				raise RuntimeError("No valid signal channels after model prediction")

			roi_freq = np.asarray(kept_freqs, dtype=np.float64)
			nchan = int(roi_freq.size)
			tag = f"{float(target_freq):.6f}".replace(".", "p")
			final_fits = os.path.join(out_dir, f"{out_prefix}_target{tag}.fits")
			synth_fits = os.path.join(out_dir, f"{out_prefix}_target{tag}_SYNTHONLY.fits")
			progress_fits = os.path.join(out_dir, f"{out_prefix}_target{tag}_INPROGRESS.fits")
			progress_png = os.path.join(out_dir, f"{out_prefix}_target{tag}_INPROGRESS_MAP.png")

			cube_final = np.full((nchan, ny, nx), np.nan, dtype=np.float32)
			cube_syn = np.full((nchan, ny, nx), np.nan, dtype=np.float32)

			y_syn_valid = np.stack(y_syn_channels, axis=1).astype(np.float32)

			noise_sum = np.zeros((n_valid, nchan), dtype=np.float64)
			noise_cnt = np.zeros((n_valid, nchan), dtype=np.float64)
			for noise_model, noise_scaler, noise_cfg in noise_models:
				segs = get_noise_segments_for_axis(noise_cfg, roi_freq)
				for idx, spw_idx in segs:
					idx = np.asarray(idx, dtype=np.int64)
					ys_seg = y_syn_valid[:, idx]
					try:
						noise_seg = predict_noise_segment_batch(
							model=noise_model,
							scaler_y=noise_scaler,
							cfg_noise=noise_cfg,
							y_synth_segment_batch=ys_seg,
							x_features_batch=x_valid,
							spw_idx=int(spw_idx),
							noise_scale=noise_scale,
							batch_size=2048,
						)
						noise_sum[:, idx] += noise_seg.astype(np.float64)
						noise_cnt[:, idx] += 1.0
					except Exception:
						continue
			y_noise_valid = np.zeros((n_valid, nchan), dtype=np.float32)
			m = noise_cnt > 0
			if np.any(m):
				y_noise_valid[m] = (noise_sum[m] / noise_cnt[m]).astype(np.float32)
				covered_channels = int(np.sum(np.any(m, axis=0)))
				if covered_channels < int(nchan):
					print(
						f"[WARN] target {target_tag} partial noise coverage: {covered_channels}/{int(nchan)} channels. "
						"Missing channels will be generated as synthetic-only.")
			else:
				print(f"[WARN] target {target_tag} has no overlapping noise ROI segments. Generating synthetic-only cube for this target.")
			y_final_valid = (y_syn_valid + y_noise_valid).astype(np.float32)

			pixel_order = _spiral_pixel_order_valid(valid_mask)
			total_pixels = int(len(pixel_order))
			processed_mask = np.zeros((ny, nx), dtype=bool)
			for p_done, (y, x) in enumerate(pixel_order, start=1):
				vid = int(valid_id_map[y, x])
				cube_syn[:, y, x] = y_syn_valid[vid]
				cube_final[:, y, x] = y_final_valid[vid]
				processed_mask[y, x] = True
				if (p_done % progress_every) == 0 or p_done == total_pixels:
					write_cube_fits(progress_fits, cube_final, roi_freq, ref_hdr, f"Checkpoint: {p_done}/{total_pixels}")
					save_progress_png(cube_final, target_freq, p_done, total_pixels, progress_png, processed_mask=processed_mask)

			write_cube_fits(final_fits, cube_final, roi_freq, ref_hdr, "Final synthetic cube")
			write_cube_fits(synth_fits, cube_syn, roi_freq, ref_hdr, "Final synthetic-only cube")
			n_ok += 1
		except Exception as e:
			n_fail += 1
			print(f"[WARN] target {float(target_freq):.6f} failed: {e}")
			continue
	if n_ok <= 0:
		raise RuntimeError(f"No cubes were generated successfully. Failed targets: {n_fail}/{len(target_freqs)}")
	return 0


def _write_map_fits_2d(out_fits_path: str, map_2d: np.ndarray, ref_hdr: Optional[object], history_text: str):
	if fits is None:
		return
	arr = np.asarray(map_2d, dtype=np.float32)
	hdr = fits.Header()
	hdr["WCSAXES"] = 2
	hdr["BUNIT"] = "a.u."
	if ref_hdr is not None:
		for k in ["CTYPE1", "CTYPE2", "CUNIT1", "CUNIT2", "CRPIX1", "CRPIX2", "CRVAL1", "CRVAL2", "CDELT1", "CDELT2", "CROTA1", "CROTA2", "RADESYS", "EQUINOX", "LONPOLE", "LATPOLE"]:
			if k in ref_hdr:
				hdr[k] = ref_hdr[k]
		for k in ref_hdr.keys():
			ks = str(k)
			if ks.startswith("CD1_") or ks.startswith("CD2_") or ks.startswith("PC1_") or ks.startswith("PC2_"):
				hdr[ks] = ref_hdr[ks]
	hdr["HISTORY"] = str(history_text)
	fits.writeto(out_fits_path, arr, header=hdr, overwrite=True)


def _save_cubefit_progress_png(
	logn_map: np.ndarray,
	tex_map: np.ndarray,
	velo_map: np.ndarray,
	fwhm_map: np.ndarray,
	done_steps: int,
	total_steps: int,
	out_png: str,
):
	fig, axes = plt.subplots(2, 2, figsize=(8.6, 7.4))
	items = [
		("logN", np.asarray(logn_map, dtype=np.float32), "viridis"),
		("Tex", np.asarray(tex_map, dtype=np.float32), "magma"),
		("Velocity", np.asarray(velo_map, dtype=np.float32), "coolwarm"),
		("FWHM", np.asarray(fwhm_map, dtype=np.float32), "plasma"),
	]
	for ax, (ttl, arr, cmap) in zip(axes.ravel(), items):
		fin = np.isfinite(arr)
		if np.any(fin):
			v = arr[fin]
			vmin = float(np.nanpercentile(v, 1.0))
			vmax = float(np.nanpercentile(v, 99.0))
			if vmax <= vmin:
				vmax = vmin + 1e-6
			im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		else:
			im = ax.imshow(np.zeros_like(arr, dtype=np.float32), origin="lower", cmap=cmap)
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		ax.set_title(ttl)
		ax.set_xlabel("x")
		ax.set_ylabel("y")
	fig.suptitle(f"Cube Fitting progress | pixels processed: {int(done_steps)}/{int(total_steps)}", y=0.98)
	plt.tight_layout()
	fig.savefig(out_png, dpi=170)
	plt.close(fig)
	info_path = os.path.splitext(out_png)[0] + ".json"
	info = {
		"title": f"Cube fitting parameter maps | pixels processed: {int(done_steps)}/{int(total_steps)}",
		"done_steps": int(done_steps),
		"total_steps": int(total_steps),
	}
	try:
		with open(info_path, "w", encoding="utf-8") as f:
			json.dump(info, f, ensure_ascii=False, indent=2)
	except Exception:
		pass


def run_cube_fit_worker(cfg_path: str) -> int:
	if fits is None:
		print("FITS backend not available")
		return 2
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	out_dir = str(cfg["out_dir"])
	os.makedirs(out_dir, exist_ok=True)
	obs_cube_path = str(cfg["obs_cube_path"])
	signal_models_source = str(cfg["signal_models_source"])
	noise_models_root = str(cfg["noise_models_root"])
	filter_file = str(cfg["filter_file"])
	target_freqs = [float(v) for v in cfg.get("target_freqs", [])]
	case_mode = str(cfg.get("case_mode", "synthetic_only"))
	fit_criterion = str(cfg.get("fit_criterion", "mae"))
	global_weight_mode = str(cfg.get("global_weight_mode", "uniform"))
	global_search_mode = str(cfg.get("global_search_mode", "per_roi"))
	candidate_mode = str(cfg.get("candidate_mode", "random"))
	n_candidates = int(cfg.get("n_candidates", 600))
	ranges = dict(cfg.get("ranges", {}))
	noise_scale = float(cfg.get("noise_scale", 1.0))
	allow_nearest = bool(cfg.get("allow_nearest", True))
	seed = int(cfg.get("seed", 42))
	progress_every = int(max(1, int(cfg.get("progress_every", 40))))
	spatial_stride = int(max(1, int(cfg.get("spatial_stride", 1))))
	obs_shift_enabled = bool(cfg.get("obs_shift_enabled", True))
	obs_shift_mode = str(cfg.get("obs_shift_mode", "per_frequency"))
	obs_shift_kms = float(cfg.get("obs_shift_kms", 0.0))
	out_prefix = str(cfg.get("out_prefix", "CUBEFIT"))

	if not os.path.isfile(obs_cube_path):
		raise FileNotFoundError(f"Observational cube not found: {obs_cube_path}")

	with fits.open(obs_cube_path, memmap=True) as hdul:
		arr = np.asarray(hdul[0].data, dtype=np.float32)
		ref_hdr = hdul[0].header.copy()
	if arr.ndim == 4:
		arr = arr[0]
	if arr.ndim != 3:
		raise RuntimeError(f"Unexpected observational cube shape: {arr.shape}")
	nchan, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
	obs_freq = _build_freq_axis_from_header(ref_hdr, nchan)
	if obs_shift_enabled:
		if str(obs_shift_mode).strip().lower() == "spw_center":
			obs_freq = _apply_velocity_shift_by_spw_center(obs_freq, float(obs_shift_kms))
		else:
			obs_freq = _apply_velocity_shift_to_frequency(obs_freq, float(obs_shift_kms))

	X_shared = _sample_fit_candidates(
		n_samples=int(n_candidates),
		ranges=ranges,
		seed=int(seed),
		mode=str(candidate_mode),
	)
	noise_models_shared = None
	if str(case_mode).strip().lower() == "synthetic_plus_noise":
		entries = _list_noise_model_entries(noise_models_root)
		nm = []
		for e in entries:
			try:
				m, sy, c = _load_noisenn_from_entry(e)
				m.eval()
				nm.append((m, sy, c))
			except Exception:
				continue
		if not nm:
			raise RuntimeError("Case 2 requires valid noise models. None could be loaded.")
		noise_models_shared = nm

	pkg_cache_shared: Dict[str, object] = {}

	map_logn = np.full((ny, nx), np.nan, dtype=np.float32)
	map_tex = np.full((ny, nx), np.nan, dtype=np.float32)
	map_velo = np.full((ny, nx), np.nan, dtype=np.float32)
	map_fwhm = np.full((ny, nx), np.nan, dtype=np.float32)
	map_obj = np.full((ny, nx), np.nan, dtype=np.float32)
	map_mae = np.full((ny, nx), np.nan, dtype=np.float32)

	valid_mask = np.any(np.isfinite(arr), axis=0)
	pixel_order = _spiral_pixel_order_valid(valid_mask)
	if int(spatial_stride) > 1:
		pixel_order = [(yy, xx) for (yy, xx) in pixel_order if (int(yy) % int(spatial_stride) == 0) and (int(xx) % int(spatial_stride) == 0)]
	total_pixels = int(len(pixel_order))
	if total_pixels <= 0:
		raise RuntimeError("No valid observational pixels in cube")

	progress_png = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_MAP.png")
	fit_count = 0
	for p_done, (y, x) in enumerate(pixel_order, start=1):
		y_obs = np.asarray(arr[:, y, x], dtype=np.float64)
		if int(np.count_nonzero(np.isfinite(y_obs))) < 3:
			continue
		res = _run_roi_fitting(
			signal_models_source=signal_models_source,
			noise_models_root=noise_models_root,
			filter_file=filter_file,
			target_freqs=target_freqs,
			obs_freq=np.asarray(obs_freq, dtype=np.float64),
			obs_intensity=np.asarray(y_obs, dtype=np.float64),
			case_mode=case_mode,
			fit_criterion=fit_criterion,
			global_weight_mode=global_weight_mode,
			global_search_mode=global_search_mode,
			candidate_mode=candidate_mode,
			n_candidates=int(n_candidates),
			ranges=ranges,
			noise_scale=float(noise_scale),
			allow_nearest=bool(allow_nearest),
			seed=int(seed),
			x_candidates_override=np.asarray(X_shared, dtype=np.float32),
			noise_models_loaded_override=noise_models_shared,
			pkg_cache_override=pkg_cache_shared,
		)
		if isinstance(res, dict) and bool(res.get("ok", False)):
			bp = res.get("best_global_params", {}) if isinstance(res.get("best_global_params", {}), dict) else {}
			map_logn[y, x] = float(bp.get("logN", np.nan))
			map_tex[y, x] = float(bp.get("Tex", np.nan))
			map_velo[y, x] = float(bp.get("Velocity", np.nan))
			map_fwhm[y, x] = float(bp.get("FWHM", np.nan))
			map_obj[y, x] = float(res.get("best_global_mean_objective", np.nan))
			map_mae[y, x] = float(res.get("best_global_mean_MAE", np.nan))
			fit_count += 1

		if (p_done % progress_every) == 0 or p_done == total_pixels:
			_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_INPROGRESS_LOGN.fits"), map_logn, ref_hdr, f"Checkpoint: {p_done}/{total_pixels}")
			_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_INPROGRESS_TEX.fits"), map_tex, ref_hdr, f"Checkpoint: {p_done}/{total_pixels}")
			_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_INPROGRESS_VELOCITY.fits"), map_velo, ref_hdr, f"Checkpoint: {p_done}/{total_pixels}")
			_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_INPROGRESS_FWHM.fits"), map_fwhm, ref_hdr, f"Checkpoint: {p_done}/{total_pixels}")
			_save_cubefit_progress_png(map_logn, map_tex, map_velo, map_fwhm, p_done, total_pixels, progress_png)

	if fit_count <= 0:
		raise RuntimeError("Cube fitting produced no valid fitted pixels")

	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_LOGN.fits"), map_logn, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_TEX.fits"), map_tex, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_VELOCITY.fits"), map_velo, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_FWHM.fits"), map_fwhm, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_OBJECTIVE.fits"), map_obj, ref_hdr, "Final cube fitting objective map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_MAE.fits"), map_mae, ref_hdr, "Final cube fitting MAE map")
	print(f"[INFO] Cube fitting completed | valid fitted pixels: {fit_count}/{total_pixels}")
	return 0


def run_sim_worker(cfg_path: str) -> int:
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	result_path = str(cfg["result_path"])
	os.makedirs(os.path.dirname(result_path), exist_ok=True)

	syngen_path = _resolve_local_file("4.SYNGEN_Streamlit_v1.py")
	syn_mod = _load_module_from_path(syngen_path, "syngen_v4_for_6_simworker")

	payload, n_with_noise, filter_roi_bounds = syn_mod.generate_exact_style16_payload(
		signal_models_root=str(cfg["signal_models_root"]),
		noise_models_root=str(cfg["noise_models_root"]),
		x_features=[
			float(cfg["logn"]),
			float(cfg["tex"]),
			float(cfg["velo"]),
			float(cfg["fwhm"]),
		],
		pred_mode="ensemble_mean",
		selected_model_name="GradientBoosting",
		noise_scale=float(cfg["obs_noise_scale"]),
		filter_file=str(cfg["filter_file"]),
		target_freqs=[float(v) for v in cfg["target_freqs"]],
		allow_nearest=bool(cfg["allow_nearest"]),
	)

	fig = syn_mod.build_exact_style16_interactive_figure(
		payload=payload,
		tex=float(cfg["tex"]),
		logn=float(cfg["logn"]),
		fwhm=float(cfg["fwhm"]),
		velo=float(cfg["velo"]),
		filter_roi_bounds=filter_roi_bounds,
		show_obs_cube=False,
		show_filter_rois=bool(cfg.get("show_filter_rois", True)),
		show_noise_model_rois=bool(cfg.get("show_noise_rois", True)),
		show_noise_applied_rois=bool(cfg.get("show_applied_rois", True)),
		plot_height=int(cfg.get("plot_height", 1200)),
	)

	result_obj = {
		"n_with_noise": int(n_with_noise),
		"fig_json": fig.to_json(),
	}
	with open(result_path, "w", encoding="utf-8") as f:
		json.dump(result_obj, f, ensure_ascii=False)
	return 0


def _project_dir() -> Path:
	return Path(__file__).resolve().parent


def _find_latest_progress_png(out_dir: str) -> Optional[str]:
	if not out_dir or (not os.path.isdir(out_dir)):
		return None
	pngs = [os.path.join(out_dir, n) for n in os.listdir(out_dir) if n.lower().endswith("_inprogress_map.png")]
	if not pngs:
		return None
	pngs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return pngs[0]


def _read_progress_png_stable_bytes(path: str, retries: int = 4, delay_s: float = 0.12) -> Optional[bytes]:
	if (not path) or (not os.path.isfile(path)):
		return None
	for _ in range(int(max(1, retries))):
		try:
			size1 = os.path.getsize(path)
			if size1 <= 0:
				time.sleep(float(delay_s))
				continue
			with open(path, "rb") as f:
				data = f.read()
			size2 = len(data)
			if size2 <= 0 or size1 != size2:
				time.sleep(float(delay_s))
				continue
			if Image is not None:
				try:
					im = Image.open(io.BytesIO(data))
					im.verify()
				except Exception:
					time.sleep(float(delay_s))
					continue
			return data
		except Exception:
			time.sleep(float(delay_s))
	return None


def _read_progress_info_caption(progress_png_path: Optional[str]) -> Optional[str]:
	if not progress_png_path:
		return None
	info_path = os.path.splitext(progress_png_path)[0] + ".json"
	if not os.path.isfile(info_path):
		return None
	try:
		with open(info_path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		msg = str(obj.get("title", "")).strip()
		return msg if msg else None
	except Exception:
		return None


def _read_progress_info(progress_png_path: Optional[str]) -> Optional[dict]:
	if not progress_png_path:
		return None
	info_path = os.path.splitext(progress_png_path)[0] + ".json"
	if not os.path.isfile(info_path):
		return None
	try:
		with open(info_path, "r", encoding="utf-8") as f:
			obj = json.load(f)
		return obj if isinstance(obj, dict) else None
	except Exception:
		return None


def _read_log_tail(log_path: str, n_lines: int = 60) -> str:
	if (not log_path) or (not os.path.isfile(log_path)):
		return ""
	try:
		with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
			lines = f.readlines()
		if not lines:
			return ""
		n = int(max(1, n_lines))
		return "".join(lines[-n:]).strip()
	except Exception:
		return ""


def _find_latest_final_main_cube(out_dir: str) -> Optional[str]:
	if not out_dir or (not os.path.isdir(out_dir)):
		return None
	files = [os.path.join(out_dir, n) for n in os.listdir(out_dir) if n.lower().endswith(".fits")]
	main = [p for p in files if ("_inprogress" not in p.lower()) and ("_synthonly" not in p.lower())]
	if not main:
		return None
	main.sort(key=lambda p: os.path.getmtime(p), reverse=True)
	return main[0]


def _find_all_final_main_cubes(out_dir: str) -> List[str]:
	if not out_dir or (not os.path.isdir(out_dir)):
		return []
	files = [os.path.join(out_dir, n) for n in os.listdir(out_dir) if n.lower().endswith(".fits")]
	main = [p for p in files if ("_inprogress" not in p.lower()) and ("_synthonly" not in p.lower())]
	main = [p for p in main if os.path.isfile(p)]
	main.sort(key=lambda p: os.path.getmtime(p))
	return main


def _extract_target_freq_from_cube_filename(cube_path: str) -> Optional[float]:
	name = os.path.basename(str(cube_path))
	m = re.search(r"_target([0-9]+(?:p[0-9]+)?)", name, flags=re.IGNORECASE)
	if not m:
		return None
	tok = str(m.group(1)).replace("p", ".")
	try:
		return float(tok)
	except Exception:
		return None


def _filter_cubes_by_target_freqs(cube_paths: List[str], target_freqs: List[float], tol: float = 1e-6) -> List[str]:
	if not cube_paths:
		return []
	if not target_freqs:
		return list(cube_paths)
	tg = [float(v) for v in target_freqs if np.isfinite(float(v))]
	if not tg:
		return list(cube_paths)
	out: List[str] = []
	for p in cube_paths:
		ft = _extract_target_freq_from_cube_filename(p)
		if ft is None:
			continue
		if any(abs(float(ft) - float(v)) <= float(tol) for v in tg):
			out.append(p)
	return out


def _find_missing_target_freqs(requested_freqs: List[float], cube_paths: List[str], tol: float = 1e-6) -> List[float]:
	requested = [float(v) for v in (requested_freqs or []) if np.isfinite(float(v))]
	if not requested:
		return []
	available = []
	for p in (cube_paths or []):
		ft = _extract_target_freq_from_cube_filename(p)
		if ft is not None and np.isfinite(float(ft)):
			available.append(float(ft))
	missing: List[float] = []
	for r in requested:
		if not any(abs(float(r) - float(a)) <= float(tol) for a in available):
			missing.append(float(r))
	return missing


def _read_warn_lines(log_path: str, max_lines: int = 100) -> List[str]:
	if (not log_path) or (not os.path.isfile(log_path)):
		return []
	try:
		with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
			lines = [ln.rstrip("\n") for ln in f.readlines()]
		warns = [ln for ln in lines if "[WARN]" in str(ln)]
		if len(warns) > int(max_lines):
			warns = warns[-int(max_lines):]
		return warns
	except Exception:
		return []


def _read_target_failure_reasons(log_path: str) -> Dict[float, List[str]]:
	out: Dict[float, List[str]] = {}
	if (not log_path) or (not os.path.isfile(log_path)):
		return out
	pat = re.compile(r"\[WARN\]\s+target\s+([0-9]+(?:\.[0-9]+)?)\s+failed:\s*(.*)", flags=re.IGNORECASE)
	try:
		with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
			for ln in f:
				m = pat.search(str(ln).strip())
				if not m:
					continue
				try:
					ft = float(m.group(1))
				except Exception:
					continue
				reason = str(m.group(2)).strip()
				out.setdefault(float(ft), []).append(reason)
	except Exception:
		return out
	return out


def _get_cube_ny_nx(cube_fits_path: str):
	if fits is None or (not cube_fits_path) or (not os.path.isfile(cube_fits_path)):
		return None
	try:
		with fits.open(cube_fits_path, memmap=True) as hdul:
			arr = hdul[0].data
			if arr is None or arr.ndim != 3:
				return None
			return int(arr.shape[1]), int(arr.shape[2])
	except Exception:
		return None


def _build_freq_axis_from_header(hdr, nchan: int) -> np.ndarray:
	try:
		crval = float(hdr.get("CRVAL3", 0.0))
		cdelt = float(hdr.get("CDELT3", 1.0))
		crpix = float(hdr.get("CRPIX3", 1.0))
		idx = np.arange(int(nchan), dtype=np.float64)
		freq_hz = crval + (idx + 1.0 - crpix) * cdelt
		return (freq_hz / 1e9).astype(np.float64)
	except Exception:
		return np.arange(int(nchan), dtype=np.float64)


def _extract_pixel_spectra(final_fits_path: str, ypix: Optional[int] = None, xpix: Optional[int] = None):
	if fits is None or (not final_fits_path) or (not os.path.isfile(final_fits_path)):
		return None, None, None, None, "Missing final FITS"
	synth_path = final_fits_path[:-5] + "_SYNTHONLY.fits"
	try:
		with fits.open(final_fits_path, memmap=True) as hdul:
			arr = hdul[0].data
			hdr = hdul[0].header
			if arr is None or arr.ndim != 3:
				return None, None, None, None, "Unexpected final cube shape"
			nchan, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])
			yc = int(ny // 2) if ypix is None else int(max(0, min(ny - 1, int(ypix))))
			xc = int(nx // 2) if xpix is None else int(max(0, min(nx - 1, int(xpix))))
			y_final = np.asarray(arr[:, yc, xc], dtype=np.float32)
			freq = _build_freq_axis_from_header(hdr, nchan)
		y_syn = None
		if os.path.isfile(synth_path):
			with fits.open(synth_path, memmap=True) as hdul_s:
				arr_s = hdul_s[0].data
				if arr_s is not None and arr_s.ndim == 3 and arr_s.shape == arr.shape:
					y_syn = np.asarray(arr_s[:, yc, xc], dtype=np.float32)
		y_noise = (y_final - y_syn).astype(np.float32) if y_syn is not None else None
		return freq, y_syn, y_noise, y_final, None
	except Exception as e:
		return None, None, None, None, str(e)


def _build_concatenated_spectra_from_cubes(cube_paths: List[str], ypix: int = 0, xpix: int = 0):
	segments = []
	errors = []
	for p in (cube_paths or []):
		freq, y_syn, y_noise, y_final, err = _extract_pixel_spectra(str(p), ypix=int(ypix), xpix=int(xpix))
		if err is not None:
			errors.append(f"{os.path.basename(str(p))}: {err}")
			continue
		if freq is None or y_final is None:
			errors.append(f"{os.path.basename(str(p))}: empty spectrum")
			continue
		f = np.asarray(freq, dtype=np.float64).reshape(-1)
		ys = np.asarray(y_syn, dtype=np.float64).reshape(-1) if y_syn is not None else np.full_like(f, np.nan, dtype=np.float64)
		yn = np.asarray(y_noise, dtype=np.float64).reshape(-1) if y_noise is not None else np.full_like(f, np.nan, dtype=np.float64)
		yf = np.asarray(y_final, dtype=np.float64).reshape(-1)
		if f.size == 0 or yf.size != f.size:
			errors.append(f"{os.path.basename(str(p))}: invalid spectrum shape")
			continue
		segments.append((float(np.nanmin(f)), f, ys, yn, yf, str(p)))

	if not segments:
		return None, None, None, None, [], errors

	segments = sorted(segments, key=lambda t: t[0])
	f_parts = []
	ys_parts = []
	yn_parts = []
	yf_parts = []
	used_paths = []
	for i, (_, f, ys, yn, yf, p) in enumerate(segments):
		if i > 0:
			f_parts.append(np.array([np.nan], dtype=np.float64))
			ys_parts.append(np.array([np.nan], dtype=np.float64))
			yn_parts.append(np.array([np.nan], dtype=np.float64))
			yf_parts.append(np.array([np.nan], dtype=np.float64))
		f_parts.append(f)
		ys_parts.append(ys)
		yn_parts.append(yn)
		yf_parts.append(yf)
		used_paths.append(p)

	f_concat = np.concatenate(f_parts).astype(np.float64)
	ys_concat = np.concatenate(ys_parts).astype(np.float64)
	yn_concat = np.concatenate(yn_parts).astype(np.float64)
	yf_concat = np.concatenate(yf_parts).astype(np.float64)

	if not np.any(np.isfinite(ys_concat)):
		ys_concat = None
	if not np.any(np.isfinite(yn_concat)):
		yn_concat = None

	return f_concat, ys_concat, yn_concat, yf_concat, used_paths, errors


def _plot_spectrum(freq, y_syn, y_noise, y_final, chart_key: Optional[str] = None):
	if freq is None or y_final is None:
		return
	fig = go.Figure()
	if y_syn is not None:
		fig.add_trace(go.Scatter(x=freq, y=y_syn, mode="lines", name="Synthetic", line=dict(dash="dash")))
	if y_noise is not None:
		fig.add_trace(go.Scatter(x=freq, y=y_noise, mode="lines", name="Predicted noise", line=dict(dash="dot")))
	fig.add_trace(go.Scatter(x=freq, y=y_final, mode="lines", name="Synthetic + noise"))
	fig.update_layout(xaxis_title="Frequency (GHz)", yaxis_title="Intensity", template="plotly_white", height=380, margin=dict(l=40, r=20, t=40, b=40))
	st.plotly_chart(fig, width="stretch", key=chart_key)


def _spectrum_to_csv_bytes(freq, y_syn, y_noise, y_final) -> Optional[bytes]:
	if freq is None or y_final is None:
		return None
	f = np.asarray(freq, dtype=np.float64).reshape(-1)
	yf = np.asarray(y_final, dtype=np.float64).reshape(-1)
	if f.size != yf.size or f.size == 0:
		return None
	ys = np.asarray(y_syn, dtype=np.float64).reshape(-1) if y_syn is not None else None
	yn = np.asarray(y_noise, dtype=np.float64).reshape(-1) if y_noise is not None else None

	out = io.StringIO()
	head = ["freq_ghz", "synthetic", "predicted_noise", "synthetic_plus_noise"]
	out.write(",".join(head) + "\n")
	for i in range(int(f.size)):
		vs = "" if ys is None or ys.size != f.size else f"{float(ys[i]):.10g}"
		vn = "" if yn is None or yn.size != f.size else f"{float(yn[i]):.10g}"
		out.write(f"{float(f[i]):.10g},{vs},{vn},{float(yf[i]):.10g}\n")
	return out.getvalue().encode("utf-8")


def _spectrum_to_txt_bytes(freq, y_syn, y_noise, y_final) -> Optional[bytes]:
	if freq is None or y_final is None:
		return None
	f = np.asarray(freq, dtype=np.float64).reshape(-1)
	yf = np.asarray(y_final, dtype=np.float64).reshape(-1)
	if f.size != yf.size or f.size == 0:
		return None
	ys = np.asarray(y_syn, dtype=np.float64).reshape(-1) if y_syn is not None else None
	yn = np.asarray(y_noise, dtype=np.float64).reshape(-1) if y_noise is not None else None

	out = io.StringIO()
	head = ["freq_ghz", "synthetic", "predicted_noise", "synthetic_plus_noise"]
	out.write("\t".join(head) + "\n")
	for i in range(int(f.size)):
		vs = "" if ys is None or ys.size != f.size else f"{float(ys[i]):.10g}"
		vn = "" if yn is None or yn.size != f.size else f"{float(yn[i]):.10g}"
		out.write(f"{float(f[i]):.10g}\t{vs}\t{vn}\t{float(yf[i]):.10g}\n")
	return out.getvalue().encode("utf-8")


def _synthetic_spectrum_to_txt_bytes(freq, y_syn) -> Optional[bytes]:
	if freq is None or y_syn is None:
		return None
	f = np.asarray(freq, dtype=np.float64).reshape(-1)
	ys = np.asarray(y_syn, dtype=np.float64).reshape(-1)
	if f.size != ys.size or f.size == 0:
		return None
	out = io.StringIO()
	out.write("freq_ghz\tsynthetic\n")
	for i in range(int(f.size)):
		out.write(f"{float(f[i]):.10g}\t{float(ys[i]):.10g}\n")
	return out.getvalue().encode("utf-8")


def _read_uploaded_spectrum_any(upload_obj):
	if upload_obj is None:
		return None, None, "No file uploaded"
	try:
		raw = upload_obj.getvalue()
		if not raw:
			return None, None, "Empty file"
		text = raw.decode("utf-8", errors="ignore")
	except Exception as e:
		return None, None, str(e)

	freqs: List[float] = []
	vals: List[float] = []
	for line in str(text).splitlines():
		s = str(line).strip()
		if (not s) or s.startswith("#") or s.startswith("//") or s.startswith("!"):
			continue
		parts = re.split(r"[\s,;\t]+", s)
		if len(parts) < 2:
			continue
		try:
			ff = float(parts[0])
			yy = float(parts[1])
		except Exception:
			continue
		freqs.append(float(ff))
		vals.append(float(yy))

	if not freqs:
		return None, None, "No valid frequency-intensity rows found"

	f = np.asarray(freqs, dtype=np.float64)
	y = np.asarray(vals, dtype=np.float64)
	if np.nanmedian(np.abs(f)) > 1e6:
		f = f / 1e9
	ord_idx = np.argsort(f)
	return f[ord_idx], y[ord_idx], None


def _uploaded_file_signature(upload_obj) -> str:
	if upload_obj is None:
		return ""
	try:
		raw = upload_obj.getvalue()
		name = str(getattr(upload_obj, "name", ""))
		if raw is None:
			return f"{name}|0"
		h = hashlib.md5(bytes(raw)).hexdigest()
		return f"{name}|{len(raw)}|{h}"
	except Exception:
		return str(getattr(upload_obj, "name", ""))


def _build_noise_cube_bytes_from_pair(final_fits_path: str, synth_fits_path: str):
	if fits is None:
		return None, "FITS backend not available"
	if (not final_fits_path) or (not os.path.isfile(final_fits_path)):
		return None, "Final cube file not found"
	if (not synth_fits_path) or (not os.path.isfile(synth_fits_path)):
		return None, "Synthetic cube file not found"
	try:
		with fits.open(final_fits_path, memmap=True) as hfin:
			arr_fin = np.asarray(hfin[0].data, dtype=np.float32)
			hdr_fin = hfin[0].header.copy()
		with fits.open(synth_fits_path, memmap=True) as hsyn:
			arr_syn = np.asarray(hsyn[0].data, dtype=np.float32)
		if arr_fin.shape != arr_syn.shape:
			return None, f"Shape mismatch: final={arr_fin.shape}, synth={arr_syn.shape}"
		arr_noise = (arr_fin - arr_syn).astype(np.float32)
		hdr_fin["HISTORY"] = "Derived noise-only cube: FINAL - SYNTHONLY"
		bio = io.BytesIO()
		fits.writeto(bio, arr_noise, header=hdr_fin, overwrite=True)
		return bio.getvalue(), None
	except Exception as e:
		return None, str(e)


def _generate_synthetic_spectra_for_targets(
	signal_models_source: str,
	filter_file: str,
	target_freqs: List[float],
	x_features: List[float],
	pred_mode: str,
	selected_model_name: str,
	allow_nearest: bool,
):
	results: Dict[str, dict] = {}
	warnings_out: List[str] = []
	if (not signal_models_source) or ((not os.path.isfile(signal_models_source)) and (not os.path.isdir(signal_models_source))):
		return results, ["Signal models source invalid."]
	if not os.path.isfile(filter_file):
		return results, [f"Filter file not found: {filter_file}"]

	x_arr = np.asarray(x_features, dtype=np.float32).reshape(1, -1)
	is_h5_signal = os.path.isfile(signal_models_source) and str(signal_models_source).lower().endswith(".h5")
	pkg_cache: Dict[str, object] = {}

	for tf in [float(v) for v in (target_freqs or [])]:
		tag = f"{float(tf):.6f}"
		try:
			is_h5, roi_entries, _ = build_signal_index_for_roi(
				signal_source=signal_models_source,
				filter_file=filter_file,
				target_frequency_ghz=float(tf),
				pred_mode=pred_mode,
				selected_model_name=selected_model_name,
				allow_nearest=allow_nearest,
			)
			use_h5 = bool(is_h5 and is_h5_signal)
			freqs: List[float] = []
			yvals: List[float] = []
			for _, fch, model_refs in roi_entries:
				pred_acc = 0.0
				pred_cnt = 0
				for model_name, ref in model_refs:
					cache_key = f"{model_name}|{ref}"
					try:
						if cache_key not in pkg_cache:
							if use_h5:
								pkg_cache[cache_key] = load_joblib_package_from_h5(signal_models_source, ref)
							else:
								pkg_cache[cache_key] = joblib.load(ref)
						pkg = pkg_cache[cache_key]
						pred = predict_with_joblib_package_batch(pkg, x_arr)
						pred_acc += float(pred[0])
						pred_cnt += 1
					except Exception:
						continue
				if pred_cnt > 0:
					freqs.append(float(fch))
					yvals.append(float(pred_acc / float(pred_cnt)))

			if not freqs:
				warnings_out.append(f"target {tag} failed: no valid synthetic predictions in selected ROI")
				continue

			ord_idx = np.argsort(np.asarray(freqs, dtype=np.float64))
			f_arr = np.asarray(freqs, dtype=np.float64)[ord_idx]
			y_arr = np.asarray(yvals, dtype=np.float64)[ord_idx]
			results[tag] = {
				"target_freq_ghz": float(tf),
				"freq": f_arr,
				"synthetic": y_arr,
			}
		except Exception as e:
			warnings_out.append(f"target {tag} failed: {e}")
			continue

	return results, warnings_out


def _sample_fit_candidates(n_samples: int, ranges: dict, seed: int, mode: str = "random") -> np.ndarray:
	n = int(max(1, n_samples))
	mode_norm = str(mode or "").strip().lower()

	if mode_norm == "ordered_grid":
		# Build a structured 4D grid and select evenly spaced points to keep exact n.
		side = int(np.ceil(float(n) ** 0.25))
		side = int(max(2, min(side, 64)))
		grid_logn = np.linspace(float(ranges["logn_min"]), float(ranges["logn_max"]), num=side, dtype=np.float64)
		grid_tex = np.linspace(float(ranges["tex_min"]), float(ranges["tex_max"]), num=side, dtype=np.float64)
		grid_velo = np.linspace(float(ranges["velo_min"]), float(ranges["velo_max"]), num=side, dtype=np.float64)
		grid_fwhm = np.linspace(float(ranges["fwhm_min"]), float(ranges["fwhm_max"]), num=side, dtype=np.float64)
		g0, g1, g2, g3 = np.meshgrid(grid_logn, grid_tex, grid_velo, grid_fwhm, indexing="ij")
		full = np.stack([g0.reshape(-1), g1.reshape(-1), g2.reshape(-1), g3.reshape(-1)], axis=1)
		total = int(full.shape[0])
		if total <= n:
			X = full.astype(np.float32)
		else:
			idx = np.linspace(0, total - 1, num=n, dtype=np.int64)
			idx = np.unique(idx)
			if idx.size < n:
				missing = int(n - idx.size)
				pool = np.setdiff1d(np.arange(total, dtype=np.int64), idx, assume_unique=False)
				idx = np.sort(np.concatenate([idx, pool[:missing]]))
			X = full[idx].astype(np.float32)
	else:
		rng = np.random.default_rng(int(seed))
		logn = rng.uniform(float(ranges["logn_min"]), float(ranges["logn_max"]), size=n)
		tex = rng.uniform(float(ranges["tex_min"]), float(ranges["tex_max"]), size=n)
		velo = rng.uniform(float(ranges["velo_min"]), float(ranges["velo_max"]), size=n)
		fwhm = rng.uniform(float(ranges["fwhm_min"]), float(ranges["fwhm_max"]), size=n)
		X = np.stack([logn, tex, velo, fwhm], axis=1).astype(np.float32)

	# Ensure center candidate is present
	X[0, 0] = 0.5 * (float(ranges["logn_min"]) + float(ranges["logn_max"]))
	X[0, 1] = 0.5 * (float(ranges["tex_min"]) + float(ranges["tex_max"]))
	X[0, 2] = 0.5 * (float(ranges["velo_min"]) + float(ranges["velo_max"]))
	X[0, 3] = 0.5 * (float(ranges["fwhm_min"]) + float(ranges["fwhm_max"]))
	return X


def _predict_synthetic_batch_single_target(
	signal_models_source: str,
	filter_file: str,
	target_freq_ghz: float,
	x_candidates: np.ndarray,
	pred_mode: str,
	selected_model_name: str,
	allow_nearest: bool,
	pkg_cache: Dict[str, object],
):
	is_h5_signal = os.path.isfile(signal_models_source) and str(signal_models_source).lower().endswith(".h5")
	is_h5, roi_entries, _ = build_signal_index_for_roi(
		signal_source=signal_models_source,
		filter_file=filter_file,
		target_frequency_ghz=float(target_freq_ghz),
		pred_mode=pred_mode,
		selected_model_name=selected_model_name,
		allow_nearest=allow_nearest,
	)
	use_h5 = bool(is_h5 and is_h5_signal)
	n_cand = int(x_candidates.shape[0])
	cols = []
	kept_freqs: List[float] = []

	for _, fch, model_refs in roi_entries:
		pred_acc = np.zeros((n_cand,), dtype=np.float64)
		pred_cnt = 0
		for model_name, ref in model_refs:
			cache_key = f"{model_name}|{ref}"
			try:
				if cache_key not in pkg_cache:
					if use_h5:
						pkg_cache[cache_key] = load_joblib_package_from_h5(signal_models_source, ref)
					else:
						pkg_cache[cache_key] = joblib.load(ref)
				pkg = pkg_cache[cache_key]
				pred = predict_with_joblib_package_batch(pkg, x_candidates)
				pred_acc += pred.astype(np.float64)
				pred_cnt += 1
			except Exception:
				continue
		if pred_cnt > 0:
			cols.append((pred_acc / float(pred_cnt)).astype(np.float32))
			kept_freqs.append(float(fch))

	if not cols:
		return None, None, "No valid synthetic predictions for selected ROI"

	roi_freq = np.asarray(kept_freqs, dtype=np.float64)
	Y_syn = np.stack(cols, axis=1).astype(np.float32)
	ord_idx = np.argsort(roi_freq)
	return roi_freq[ord_idx], Y_syn[:, ord_idx], None


def _add_noise_batch_for_target(
	noise_models_loaded: List[tuple],
	roi_freq: np.ndarray,
	y_syn_batch: np.ndarray,
	x_candidates: np.ndarray,
	noise_scale: float,
):
	n_cand, n_chan = int(y_syn_batch.shape[0]), int(y_syn_batch.shape[1])
	noise_sum = np.zeros((n_cand, n_chan), dtype=np.float32)
	noise_cnt = np.zeros((n_cand, n_chan), dtype=np.float32)
	for noise_model, noise_scaler, noise_cfg in (noise_models_loaded or []):
		segs = get_noise_segments_for_axis(noise_cfg, roi_freq)
		for idx, spw_idx in segs:
			idx = np.asarray(idx, dtype=np.int64)
			ys_seg = y_syn_batch[:, idx]
			try:
				noise_seg = predict_noise_segment_batch(
					model=noise_model,
					scaler_y=noise_scaler,
					cfg_noise=noise_cfg,
					y_synth_segment_batch=ys_seg,
					x_features_batch=x_candidates,
					spw_idx=int(spw_idx),
					noise_scale=float(noise_scale),
					batch_size=2048,
				)
				noise_sum[:, idx] += noise_seg.astype(np.float32)
				noise_cnt[:, idx] += 1.0
			except Exception:
				continue

	y_noise = np.zeros((n_cand, n_chan), dtype=np.float32)
	m = noise_cnt > 0
	if np.any(m):
		y_noise[m] = (noise_sum[m] / noise_cnt[m]).astype(np.float32)
	return y_noise, m


def _downsample_for_plot_arrays(freq: np.ndarray, arrays: List[Optional[np.ndarray]], max_points: int = 2400):
	f = np.asarray(freq, dtype=np.float64).reshape(-1)
	if f.size == 0:
		return f, [None if a is None else np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
	max_p = int(max(16, max_points))
	if f.size <= max_p:
		return f, [None if a is None else np.asarray(a, dtype=np.float64).reshape(-1) for a in arrays]
	idx = np.linspace(0, int(f.size) - 1, num=max_p, dtype=np.int64)
	idx = np.unique(idx)
	f_ds = f[idx]
	arr_ds = []
	for a in arrays:
		if a is None:
			arr_ds.append(None)
			continue
		av = np.asarray(a, dtype=np.float64).reshape(-1)
		if av.size != f.size:
			arr_ds.append(av)
		else:
			arr_ds.append(av[idx])
	return f_ds, arr_ds


def _vectorized_fit_metrics(y_true: np.ndarray, y_pred_batch: np.ndarray):
	y = np.asarray(y_true, dtype=np.float64).reshape(1, -1)
	p = np.asarray(y_pred_batch, dtype=np.float64)
	err = p - y
	mae = np.mean(np.abs(err), axis=1)
	rmse = np.sqrt(np.mean(err ** 2, axis=1))
	den = float(np.sum((y.reshape(-1) - float(np.mean(y))) ** 2))
	if den > 0:
		r2 = 1.0 - (np.sum(err ** 2, axis=1) / den)
	else:
		r2 = np.full((p.shape[0],), np.nan, dtype=np.float64)
	eps = float(max(1e-12, np.quantile(np.abs(y.reshape(-1)), 0.1)))
	chi_like = np.mean((err ** 2) / (np.abs(y) + eps), axis=1)
	return mae.astype(np.float64), rmse.astype(np.float64), r2.astype(np.float64), chi_like.astype(np.float64)


def _criterion_aware_roi_quality_weight(
	criterion: str,
	best_mae: float,
	best_rmse: float,
	best_chi_like: float,
	best_r2: float,
) -> float:
	crit = str(criterion).strip().lower()
	if crit == "rmse":
		err = float(abs(best_rmse))
		return float(np.clip(1.0 / max(err, 1e-12), 1e-6, 1e6))
	if crit == "chi_like":
		err = float(abs(best_chi_like))
		return float(np.clip(1.0 / max(err, 1e-12), 1e-6, 1e6))
	if crit == "r2":
		r2v = float(best_r2) if np.isfinite(best_r2) else -1.0
		# R2 higher is better: map [-1, +1] to [0, 1] and clip.
		qual = 0.5 * (r2v + 1.0)
		return float(np.clip(qual, 1e-6, 1.0))
	# Default MAE-like behavior (lower is better).
	err = float(abs(best_mae))
	return float(np.clip(1.0 / max(err, 1e-12), 1e-6, 1e6))


def _format_freqs_short(freqs: List[float], max_show: int = 4) -> str:
	v = [float(x) for x in (freqs or []) if np.isfinite(float(x))]
	if not v:
		return ""
	v = sorted(v)
	ms = int(max(1, max_show))
	if len(v) <= ms:
		return ", ".join([f"{float(x):.6f}" for x in v])
	head = ", ".join([f"{float(x):.6f}" for x in v[:ms]])
	return f"{head}, ... (+{int(len(v) - ms)} more)"


def _group_target_freqs_by_signal_roi(
	signal_models_source: str,
	filter_file: str,
	target_freqs: List[float],
	allow_nearest: bool,
) -> List[dict]:
	uniq = _normalize_target_freqs_for_run([float(v) for v in (target_freqs or [])])
	out: List[dict] = []
	key_to_idx: Dict[tuple, int] = {}

	for tf in uniq:
		grp_key = ("freq", round(float(tf), 9))
		roi_lo = None
		roi_hi = None
		n_ch = None
		try:
			_, _, roi_freq = build_signal_index_for_roi(
				signal_source=signal_models_source,
				filter_file=filter_file,
				target_frequency_ghz=float(tf),
				pred_mode=DEFAULT_PRED_MODE,
				selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
				allow_nearest=bool(allow_nearest),
			)
			rf = np.asarray(roi_freq, dtype=np.float64).reshape(-1)
			if rf.size > 0 and np.any(np.isfinite(rf)):
				roi_lo = float(np.nanmin(rf))
				roi_hi = float(np.nanmax(rf))
				n_ch = int(rf.size)
				grp_key = ("roi", int(n_ch), round(float(roi_lo), 9), round(float(roi_hi), 9))
		except Exception:
			pass

		if grp_key not in key_to_idx:
			key_to_idx[grp_key] = int(len(out))
			out.append({
				"representative_target_freq_ghz": float(tf),
				"guide_freqs_ghz": [float(tf)],
				"roi_f_min_ghz": (None if roi_lo is None else float(roi_lo)),
				"roi_f_max_ghz": (None if roi_hi is None else float(roi_hi)),
				"n_roi_channels": (None if n_ch is None else int(n_ch)),
			})
		else:
			out[int(key_to_idx[grp_key])]["guide_freqs_ghz"].append(float(tf))

	for g in out:
		g["guide_freqs_ghz"] = sorted([float(x) for x in g.get("guide_freqs_ghz", [])])

	return out


def _run_roi_fitting(
	signal_models_source: str,
	noise_models_root: str,
	filter_file: str,
	target_freqs: List[float],
	obs_freq: np.ndarray,
	obs_intensity: np.ndarray,
	case_mode: str,
	fit_criterion: str,
	global_weight_mode: str,
	global_search_mode: str,
	candidate_mode: str,
	n_candidates: int,
	ranges: dict,
	noise_scale: float,
	allow_nearest: bool,
	seed: int,
	x_candidates_override: Optional[np.ndarray] = None,
	noise_models_loaded_override: Optional[List[tuple]] = None,
	pkg_cache_override: Optional[Dict[str, object]] = None,
):
	crit = str(fit_criterion).strip().lower()
	if crit not in {"mae", "rmse", "chi_like", "r2"}:
		crit = "mae"
	weight_mode = str(global_weight_mode).strip().lower()
	if weight_mode not in {"uniform", "overlap_points", "inverse_best_error"}:
		weight_mode = "uniform"
	search_mode = str(global_search_mode).strip().lower()
	if search_mode not in {"per_roi", "concatenated"}:
		search_mode = "per_roi"

	if isinstance(x_candidates_override, np.ndarray) and x_candidates_override.ndim == 2 and x_candidates_override.shape[1] == 4:
		X = np.asarray(x_candidates_override, dtype=np.float32)
	else:
		X = _sample_fit_candidates(
			n_samples=int(n_candidates),
			ranges=ranges,
			seed=int(seed),
			mode=str(candidate_mode),
		)
	n = int(X.shape[0])
	pkg_cache: Dict[str, object] = (pkg_cache_override if isinstance(pkg_cache_override, dict) else {})
	warnings_out: List[str] = []
	target_groups = _group_target_freqs_by_signal_roi(
		signal_models_source=str(signal_models_source),
		filter_file=str(filter_file),
		target_freqs=[float(v) for v in (target_freqs or [])],
		allow_nearest=bool(allow_nearest),
	)
	target_freqs_eval = [float(g.get("representative_target_freq_ghz", np.nan)) for g in target_groups]
	target_meta_by_rep: Dict[float, dict] = {}
	for g in target_groups:
		target_meta_by_rep[float(g.get("representative_target_freq_ghz", np.nan))] = {
			"guide_freqs_ghz": [float(v) for v in g.get("guide_freqs_ghz", [])],
			"guide_freqs_label": _format_freqs_short([float(v) for v in g.get("guide_freqs_ghz", [])]),
			"n_guide_freqs_in_roi": int(len(g.get("guide_freqs_ghz", []))),
			"roi_f_min_ghz": g.get("roi_f_min_ghz", None),
			"roi_f_max_ghz": g.get("roi_f_max_ghz", None),
		}
	if not target_freqs_eval:
		return {
			"ok": False,
			"warnings": ["Guide frequencies did not produce any valid ROI group."],
			"message": "No ROI could be built from Guide frequencies.",
		}

	noise_models_loaded = list(noise_models_loaded_override) if isinstance(noise_models_loaded_override, list) else []
	if str(case_mode).strip().lower() == "synthetic_plus_noise":
		if not noise_models_loaded:
			entries = _list_noise_model_entries(noise_models_root)
			for e in entries:
				try:
					m, sy, c = _load_noisenn_from_entry(e)
					m.eval()
					noise_models_loaded.append((m, sy, c))
				except Exception:
					continue
		if not noise_models_loaded:
			return {
				"ok": False,
				"warnings": ["No valid noise models loaded for Case 2 (synthetic + noise)."],
				"message": "Case 2 requires valid noise models. No fitting was performed.",
			}

	objective_rows: List[np.ndarray] = []
	mae_rows: List[np.ndarray] = []
	roi_weights: List[float] = []
	per_roi_rows: List[dict] = []
	best_plot_payload: List[dict] = []
	fit_batch_size = int(max(128, min(1024, n)))

	concat_count = 0
	concat_sum_y = 0.0
	concat_sum_y2 = 0.0
	concat_abs_err = np.zeros((n,), dtype=np.float64)
	concat_sq_err = np.zeros((n,), dtype=np.float64)
	concat_chi_term = np.zeros((n,), dtype=np.float64)

	for tf in target_freqs_eval:
		tag = f"{float(tf):.6f}"
		try:
			mae_all = np.full((n,), np.inf, dtype=np.float64)
			rmse_all = np.full((n,), np.inf, dtype=np.float64)
			r2_all = np.full((n,), np.nan, dtype=np.float64)
			chi_all = np.full((n,), np.inf, dtype=np.float64)
			obj_all = np.full((n,), np.inf, dtype=np.float64)

			roi_freq_eval_ref = None
			y_obs_roi_ref = None
			valid_ref = None
			best_i = None
			best_obj = np.inf
			best_syn = None
			best_noise = None
			best_pred = None
			skip_target = False

			for i0 in range(0, n, fit_batch_size):
				i1 = int(min(i0 + fit_batch_size, n))
				x_chunk = np.asarray(X[i0:i1], dtype=np.float32)

				roi_freq, y_syn_batch, err = _predict_synthetic_batch_single_target(
					signal_models_source=signal_models_source,
					filter_file=filter_file,
					target_freq_ghz=float(tf),
					x_candidates=x_chunk,
					pred_mode=DEFAULT_PRED_MODE,
					selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
					allow_nearest=allow_nearest,
					pkg_cache=pkg_cache,
				)
				if err is not None or roi_freq is None or y_syn_batch is None:
					warnings_out.append(f"target {tag} skipped: {err if err else 'empty ROI prediction'}")
					skip_target = True
					break

				y_eval = y_syn_batch
				y_syn_eval = y_syn_batch
				y_noise_eval_batch = None
				roi_freq_eval = np.asarray(roi_freq, dtype=np.float64)

				if (str(case_mode).strip().lower() == "synthetic_plus_noise") and noise_models_loaded:
					y_noise_batch, noise_mask_batch = _add_noise_batch_for_target(
						noise_models_loaded=noise_models_loaded,
						roi_freq=roi_freq,
						y_syn_batch=y_syn_batch,
						x_candidates=x_chunk,
						noise_scale=float(noise_scale),
					)
					noise_channel_mask = np.any(np.asarray(noise_mask_batch, dtype=bool), axis=0)
					if not np.any(noise_channel_mask):
						warnings_out.append(f"target {tag} skipped: no overlapping noise ROI for this synthetic ROI")
						skip_target = True
						break
					roi_freq_eval = np.asarray(roi_freq, dtype=np.float64)[noise_channel_mask]
					y_syn_eval = np.asarray(y_syn_batch, dtype=np.float32)[:, noise_channel_mask]
					y_noise_eval_batch = np.asarray(y_noise_batch, dtype=np.float32)[:, noise_channel_mask]
					y_eval = (y_syn_eval + y_noise_eval_batch).astype(np.float32)

				if roi_freq_eval_ref is None:
					roi_freq_eval_ref = np.asarray(roi_freq_eval, dtype=np.float64)
					y_obs_roi_ref = np.interp(
						roi_freq_eval_ref,
						np.asarray(obs_freq, dtype=np.float64),
						np.asarray(obs_intensity, dtype=np.float64),
						left=np.nan,
						right=np.nan,
					)
					valid_ref = np.isfinite(y_obs_roi_ref)
					if int(np.count_nonzero(valid_ref)) < 3:
						warnings_out.append(f"target {tag} skipped: insufficient overlap with uploaded observational spectrum")
						skip_target = True
						break
				else:
					if int(np.asarray(roi_freq_eval, dtype=np.float64).size) != int(np.asarray(roi_freq_eval_ref, dtype=np.float64).size):
						warnings_out.append(f"target {tag} skipped: inconsistent ROI channel count across candidate chunks")
						skip_target = True
						break

				y_true = np.asarray(y_obs_roi_ref[valid_ref], dtype=np.float64)
				y_pred_batch = np.asarray(y_eval[:, valid_ref], dtype=np.float64)
				mae, rmse, r2, chi_like = _vectorized_fit_metrics(y_true, y_pred_batch)
				if crit == "rmse":
					obj = np.asarray(rmse, dtype=np.float64)
				elif crit == "chi_like":
					obj = np.asarray(chi_like, dtype=np.float64)
				elif crit == "r2":
					obj = -np.asarray(r2, dtype=np.float64)
				else:
					obj = np.asarray(mae, dtype=np.float64)
				obj[~np.isfinite(obj)] = np.inf

				mae_all[i0:i1] = np.asarray(mae, dtype=np.float64)
				rmse_all[i0:i1] = np.asarray(rmse, dtype=np.float64)
				r2_all[i0:i1] = np.asarray(r2, dtype=np.float64)
				chi_all[i0:i1] = np.asarray(chi_like, dtype=np.float64)
				obj_all[i0:i1] = np.asarray(obj, dtype=np.float64)

				loc = int(np.argmin(obj))
				loc_obj = float(obj[loc])
				if np.isfinite(loc_obj) and loc_obj < float(best_obj):
					best_obj = float(loc_obj)
					best_i = int(i0 + loc)
					best_syn = np.asarray(y_syn_eval[loc], dtype=np.float64)
					best_noise = (None if y_noise_eval_batch is None else np.asarray(y_noise_eval_batch[loc], dtype=np.float64))
					best_pred = np.asarray(y_eval[loc], dtype=np.float64)

			if skip_target or (best_i is None) or (roi_freq_eval_ref is None) or (y_obs_roi_ref is None) or (valid_ref is None):
				continue

			obj = np.asarray(obj_all, dtype=np.float64)
			mae = np.asarray(mae_all, dtype=np.float64)
			rmse = np.asarray(rmse_all, dtype=np.float64)
			r2 = np.asarray(r2_all, dtype=np.float64)
			chi_like = np.asarray(chi_all, dtype=np.float64)
			roi_freq_eval = np.asarray(roi_freq_eval_ref, dtype=np.float64)
			y_obs_roi = np.asarray(y_obs_roi_ref, dtype=np.float64)
			valid = np.asarray(valid_ref, dtype=bool)

			y_true = np.asarray(y_obs_roi[valid], dtype=np.float64)
			# Accumulate concatenated global metrics across ROIs for all candidates.
			# Summary-by-candidate accumulation for concatenated mode (independent of ROI weighting).
			# Use current ROI candidate predictions available through the objective vectors:
			# mae = mean(|err|), rmse = sqrt(mean(err^2)), chi_like = mean(err^2/(|y|+eps)).
			n_pts_roi = int(y_true.size)
			if n_pts_roi > 0:
				concat_count += int(n_pts_roi)
				concat_sum_y += float(np.sum(y_true))
				concat_sum_y2 += float(np.sum(y_true ** 2))
				concat_abs_err += np.asarray(mae, dtype=np.float64) * float(n_pts_roi)
				concat_sq_err += (np.asarray(rmse, dtype=np.float64) ** 2) * float(n_pts_roi)
				concat_chi_term += np.asarray(chi_like, dtype=np.float64) * float(n_pts_roi)

			roi_n_overlap = int(np.count_nonzero(valid))
			if weight_mode == "overlap_points":
				roi_weight = float(max(roi_n_overlap, 1))
			elif weight_mode == "inverse_best_error":
				roi_weight = _criterion_aware_roi_quality_weight(
					criterion=str(crit),
					best_mae=float(mae[best_i]),
					best_rmse=float(rmse[best_i]),
					best_chi_like=float(chi_like[best_i]),
					best_r2=float(r2[best_i]) if np.isfinite(float(r2[best_i])) else np.nan,
				)
			else:
				roi_weight = 1.0

			objective_rows.append(np.asarray(obj, dtype=np.float64))
			mae_rows.append(np.asarray(mae, dtype=np.float64))
			roi_weights.append(float(roi_weight))

			per_roi_rows.append({
				"target_freq_ghz": float(tf),
				"guide_freqs_ghz": list(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_ghz", [float(tf)])),
				"guide_freqs_label": str(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_label", f"{float(tf):.6f}")),
				"n_guide_freqs_in_roi": int(target_meta_by_rep.get(float(tf), {}).get("n_guide_freqs_in_roi", 1)),
				"roi_f_min_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", None) is not None else np.nan,
				"roi_f_max_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", None) is not None else np.nan,
				"n_channels": int(roi_freq_eval.size),
				"n_overlap_points": int(roi_n_overlap),
				"best_MAE": float(mae[best_i]),
				"best_RMSE": float(rmse[best_i]),
				"best_R2": float(r2[best_i]) if np.isfinite(float(r2[best_i])) else np.nan,
				"best_CHI_like": float(chi_like[best_i]),
				"best_logN": float(X[best_i, 0]),
				"best_Tex": float(X[best_i, 1]),
				"best_Velocity": float(X[best_i, 2]),
				"best_FWHM": float(X[best_i, 3]),
				"criterion_used": str(crit),
				"best_objective": float(obj[best_i]),
				"roi_weight_used": float(roi_weight),
				"global_weight_mode": str(weight_mode),
				"roi_weight_rule": ("criterion_aware" if str(weight_mode) == "inverse_best_error" else "fixed"),
			})

			ds_freq, ds_arrays = _downsample_for_plot_arrays(
				np.asarray(roi_freq_eval, dtype=np.float64),
				[
					np.asarray(y_obs_roi, dtype=np.float64),
					(None if best_syn is None else np.asarray(best_syn, dtype=np.float64)),
					(None if best_noise is None else np.asarray(best_noise, dtype=np.float64)),
					(None if best_pred is None else np.asarray(best_pred, dtype=np.float64)),
				],
				max_points=2400,
			)

			best_plot_payload.append({
				"target_freq_ghz": float(tf),
				"guide_freqs_ghz": list(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_ghz", [float(tf)])),
				"guide_freqs_label": str(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_label", f"{float(tf):.6f}")),
				"n_guide_freqs_in_roi": int(target_meta_by_rep.get(float(tf), {}).get("n_guide_freqs_in_roi", 1)),
				"roi_f_min_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", None) is not None else np.nan,
				"roi_f_max_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", None) is not None else np.nan,
				"freq": ds_freq,
				"obs_interp": ds_arrays[0],
				"best_synthetic": ds_arrays[1],
				"best_noise": ds_arrays[2],
				"best_pred": ds_arrays[3],
				"best_idx": int(best_i),
			})
		except Exception as e:
			warnings_out.append(f"target {tag} skipped: {e}")
			continue

	if not objective_rows:
		return {
			"ok": False,
			"warnings": warnings_out,
			"message": "No ROI could be fitted against uploaded spectrum.",
		}

	if search_mode == "concatenated":
		if int(concat_count) <= 0:
			return {
				"ok": False,
				"warnings": warnings_out,
				"message": "No valid concatenated ROI points available for global fitting.",
			}
		global_mae = (np.asarray(concat_abs_err, dtype=np.float64) / float(concat_count)).astype(np.float64)
		global_rmse = np.sqrt(np.asarray(concat_sq_err, dtype=np.float64) / float(concat_count)).astype(np.float64)
		global_chi = (np.asarray(concat_chi_term, dtype=np.float64) / float(concat_count)).astype(np.float64)
		mean_y = float(concat_sum_y) / float(concat_count)
		sst = float(concat_sum_y2 - 2.0 * mean_y * concat_sum_y + float(concat_count) * (mean_y ** 2))
		if sst > 0.0:
			global_r2 = 1.0 - (np.asarray(concat_sq_err, dtype=np.float64) / float(sst))
		else:
			global_r2 = np.full((n,), np.nan, dtype=np.float64)

		if crit == "rmse":
			global_obj = np.asarray(global_rmse, dtype=np.float64)
		elif crit == "chi_like":
			global_obj = np.asarray(global_chi, dtype=np.float64)
		elif crit == "r2":
			global_obj = -np.asarray(global_r2, dtype=np.float64)
		else:
			global_obj = np.asarray(global_mae, dtype=np.float64)
		global_obj[~np.isfinite(global_obj)] = np.inf
		best_global_idx = int(np.argmin(global_obj))
		weighting_used = "not_used_in_concatenated_mode"
	else:
		obj_mat = np.vstack(objective_rows).astype(np.float64)
		mae_mat = np.vstack(mae_rows).astype(np.float64)
		w = np.asarray(roi_weights, dtype=np.float64)
		w[~np.isfinite(w)] = 0.0
		w = np.clip(w, 0.0, np.inf)
		if float(np.sum(w)) <= 0.0:
			w = np.ones_like(w, dtype=np.float64)

		global_obj = np.average(obj_mat, axis=0, weights=w).astype(np.float64)
		best_global_idx = int(np.nanargmin(global_obj))
		global_mae = np.average(mae_mat, axis=0, weights=w).astype(np.float64)
		weighting_used = str(weight_mode)

	# Build global overlay using a single best parameter vector across all fitted ROIs
	x_best = np.asarray(X[best_global_idx:best_global_idx + 1], dtype=np.float32)
	global_overlay = []
	global_per_roi_rows: List[dict] = []
	global_plot_payload: List[dict] = []
	for tf in target_freqs_eval:
		tag = f"{float(tf):.6f}"
		try:
			roi_freq_g, y_syn_g, err_g = _predict_synthetic_batch_single_target(
				signal_models_source=signal_models_source,
				filter_file=filter_file,
				target_freq_ghz=float(tf),
				x_candidates=x_best,
				pred_mode=DEFAULT_PRED_MODE,
				selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
				allow_nearest=allow_nearest,
				pkg_cache=pkg_cache,
			)
			if err_g is not None or roi_freq_g is None or y_syn_g is None:
				continue

			y_pred_g = np.asarray(y_syn_g[0], dtype=np.float64)
			y_noise_g = None
			if (str(case_mode).strip().lower() == "synthetic_plus_noise") and noise_models_loaded:
				y_noise_b, noise_mask_b = _add_noise_batch_for_target(
					noise_models_loaded=noise_models_loaded,
					roi_freq=roi_freq_g,
					y_syn_batch=np.asarray(y_syn_g, dtype=np.float32),
					x_candidates=x_best,
					noise_scale=float(noise_scale),
				)
				noise_channel_mask_g = np.any(np.asarray(noise_mask_b, dtype=bool), axis=0)
				if not np.any(noise_channel_mask_g):
					continue
				roi_freq_g = np.asarray(roi_freq_g, dtype=np.float64)[noise_channel_mask_g]
				y_syn_g = np.asarray(y_syn_g, dtype=np.float32)[:, noise_channel_mask_g]
				y_noise_g = np.asarray(y_noise_b[0], dtype=np.float64)[noise_channel_mask_g]
				y_pred_g = np.asarray(y_syn_g[0], dtype=np.float64) + y_noise_g
 
			if (str(case_mode).strip().lower() != "synthetic_plus_noise"):
				roi_freq_g = np.asarray(roi_freq_g, dtype=np.float64)
				y_syn_g = np.asarray(y_syn_g, dtype=np.float32)

			y_obs_g = np.interp(
				roi_freq_g,
				np.asarray(obs_freq, dtype=np.float64),
				np.asarray(obs_intensity, dtype=np.float64),
				left=np.nan,
				right=np.nan,
			)
			ds_fg, ds_g = _downsample_for_plot_arrays(
				np.asarray(roi_freq_g, dtype=np.float64),
				[
					np.asarray(y_obs_g, dtype=np.float64),
					np.asarray(y_syn_g[0], dtype=np.float64),
					(None if y_noise_g is None else np.asarray(y_noise_g, dtype=np.float64)),
					np.asarray(y_pred_g, dtype=np.float64),
				],
				max_points=2400,
			)

			# Per-ROI metrics using the single global-best parameter vector.
			vg = np.isfinite(np.asarray(y_obs_g, dtype=np.float64)) & np.isfinite(np.asarray(y_pred_g, dtype=np.float64))
			if int(np.count_nonzero(vg)) >= 3:
				yt = np.asarray(y_obs_g, dtype=np.float64)[vg]
				yp = np.asarray(y_pred_g, dtype=np.float64)[vg]
				err = np.asarray(yp - yt, dtype=np.float64)
				mae_g = float(np.mean(np.abs(err)))
				rmse_g = float(np.sqrt(np.mean(err ** 2)))
				den_g = float(np.sum((yt - float(np.mean(yt))) ** 2))
				r2_g = (float(1.0 - (np.sum(err ** 2) / den_g)) if den_g > 0 else np.nan)
				eps_g = float(max(1e-12, np.quantile(np.abs(yt), 0.1)))
				chi_g = float(np.mean((err ** 2) / (np.abs(yt) + eps_g)))
				if crit == "rmse":
					obj_g = rmse_g
				elif crit == "chi_like":
					obj_g = chi_g
				elif crit == "r2":
					obj_g = -r2_g if np.isfinite(r2_g) else np.inf
				else:
					obj_g = mae_g

				global_per_roi_rows.append({
					"target_freq_ghz": float(tf),
					"guide_freqs_ghz": list(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_ghz", [float(tf)])),
					"guide_freqs_label": str(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_label", f"{float(tf):.6f}")),
					"n_guide_freqs_in_roi": int(target_meta_by_rep.get(float(tf), {}).get("n_guide_freqs_in_roi", 1)),
					"roi_f_min_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", None) is not None else np.nan,
					"roi_f_max_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", None) is not None else np.nan,
					"n_channels": int(np.asarray(roi_freq_g, dtype=np.float64).size),
					"n_overlap_points": int(np.count_nonzero(vg)),
					"best_MAE": float(mae_g),
					"best_RMSE": float(rmse_g),
					"best_R2": float(r2_g) if np.isfinite(float(r2_g)) else np.nan,
					"best_CHI_like": float(chi_g),
					"best_logN": float(X[best_global_idx, 0]),
					"best_Tex": float(X[best_global_idx, 1]),
					"best_Velocity": float(X[best_global_idx, 2]),
					"best_FWHM": float(X[best_global_idx, 3]),
					"criterion_used": str(crit),
					"best_objective": float(obj_g),
					"roi_weight_used": (np.nan if str(search_mode) == "concatenated" else 1.0),
					"global_weight_mode": str(weighting_used),
					"roi_weight_rule": "global_single_param_eval",
				})

				global_plot_payload.append({
					"target_freq_ghz": float(tf),
					"guide_freqs_ghz": list(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_ghz", [float(tf)])),
					"guide_freqs_label": str(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_label", f"{float(tf):.6f}")),
					"n_guide_freqs_in_roi": int(target_meta_by_rep.get(float(tf), {}).get("n_guide_freqs_in_roi", 1)),
					"roi_f_min_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", None) is not None else np.nan,
					"roi_f_max_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", None) is not None else np.nan,
					"freq": ds_fg,
					"obs_interp": ds_g[0],
					"best_synthetic": ds_g[1],
					"best_noise": ds_g[2],
					"best_pred": ds_g[3],
					"best_idx": int(best_global_idx),
				})

			global_overlay.append({
				"target_freq_ghz": float(tf),
				"guide_freqs_ghz": list(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_ghz", [float(tf)])),
				"guide_freqs_label": str(target_meta_by_rep.get(float(tf), {}).get("guide_freqs_label", f"{float(tf):.6f}")),
				"n_guide_freqs_in_roi": int(target_meta_by_rep.get(float(tf), {}).get("n_guide_freqs_in_roi", 1)),
				"roi_f_min_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_min_ghz", None) is not None else np.nan,
				"roi_f_max_ghz": float(target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", np.nan)) if target_meta_by_rep.get(float(tf), {}).get("roi_f_max_ghz", None) is not None else np.nan,
				"freq": ds_fg,
				"obs_interp": ds_g[0],
				"best_global_synthetic": ds_g[1],
				"best_global_noise": ds_g[2],
				"best_global_pred": ds_g[3],
			})
		except Exception:
			warnings_out.append(f"target {tag} skipped in global overlay build")
			continue

	per_roi_out = per_roi_rows
	plot_payload_out = best_plot_payload
	if str(search_mode) == "concatenated":
		if global_per_roi_rows:
			per_roi_out = global_per_roi_rows
		if global_plot_payload:
			plot_payload_out = global_plot_payload

	return {
		"ok": True,
		"case_mode": str(case_mode),
		"global_search_mode": str(search_mode),
		"n_guide_freqs_input": int(len([float(v) for v in (target_freqs or [])])),
		"n_unique_rois_requested": int(len(target_groups)),
		"n_candidates": int(n),
		"best_global_index": int(best_global_idx),
		"best_global_params": {
			"logN": float(X[best_global_idx, 0]),
			"Tex": float(X[best_global_idx, 1]),
			"Velocity": float(X[best_global_idx, 2]),
			"FWHM": float(X[best_global_idx, 3]),
		},
		"fit_criterion": str(crit),
		"global_weight_mode": str(weighting_used),
		"candidate_mode": str(candidate_mode),
		"best_global_mean_objective": float(global_obj[best_global_idx]),
		"best_global_mean_MAE": float(global_mae[best_global_idx]),
		"n_rois_fitted": int(len(per_roi_out)),
		"per_roi": per_roi_out,
		"plot_payload": plot_payload_out,
		"global_overlay": global_overlay,
		"warnings": warnings_out,
	}


def _ensure_state():
	if "cube_proc" not in st.session_state:
		st.session_state.cube_proc = None
	if "cube_log_path" not in st.session_state:
		st.session_state.cube_log_path = ""
	if "cube_cfg_path" not in st.session_state:
		st.session_state.cube_cfg_path = ""
	if "cube_log_handle" not in st.session_state:
		st.session_state.cube_log_handle = None
	if "cube_last_final_fits" not in st.session_state:
		st.session_state.cube_last_final_fits = ""
	if "cube_last_final_mtime" not in st.session_state:
		st.session_state.cube_last_final_mtime = 0.0
	if "cube_last_spectrum_data" not in st.session_state:
		st.session_state.cube_last_spectrum_data = None
	if "cube2_last_final_fits" not in st.session_state:
		st.session_state.cube2_last_final_fits = ""
	if "cube2_last_final_mtime" not in st.session_state:
		st.session_state.cube2_last_final_mtime = 0.0
	if "cube2_last_spectrum_data" not in st.session_state:
		st.session_state.cube2_last_spectrum_data = None
	if "sim_proc" not in st.session_state:
		st.session_state.sim_proc = None
	if "sim_log_path" not in st.session_state:
		st.session_state.sim_log_path = ""
	if "sim_cfg_path" not in st.session_state:
		st.session_state.sim_cfg_path = ""
	if "sim_log_handle" not in st.session_state:
		st.session_state.sim_log_handle = None
	if "sim_result_path" not in st.session_state:
		st.session_state.sim_result_path = ""
	if "sim_last_result" not in st.session_state:
		st.session_state.sim_last_result = None
	if "cubefit_proc" not in st.session_state:
		st.session_state.cubefit_proc = None
	if "cubefit_log_path" not in st.session_state:
		st.session_state.cubefit_log_path = ""
	if "cubefit_cfg_path" not in st.session_state:
		st.session_state.cubefit_cfg_path = ""
	if "cubefit_log_handle" not in st.session_state:
		st.session_state.cubefit_log_handle = None
	if "drive_cache_dir" not in st.session_state:
		st.session_state.drive_cache_dir = ""
	if "drive_auto_paths" not in st.session_state:
		st.session_state.drive_auto_paths = {}
	if "drive_last_error" not in st.session_state:
		st.session_state.drive_last_error = ""
	if "p6_guide_freqs_main_input" not in st.session_state:
		st.session_state.p6_guide_freqs_main_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
	if "p6_guide_freqs_cube2_input" not in st.session_state:
		st.session_state.p6_guide_freqs_cube2_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
	if "p6_guide_freqs_main_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_main_pending = ""
	if "p6_guide_freqs_cube2_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_cube2_pending = ""
	if "p6_guide_main_refresh" not in st.session_state:
		st.session_state.p6_guide_main_refresh = False
	if "p6_guide_cube2_refresh" not in st.session_state:
		st.session_state.p6_guide_cube2_refresh = False
	if "p6_cube2_last_run_target_freqs" not in st.session_state:
		st.session_state.p6_cube2_last_run_target_freqs = []
	if "p6_cube_last_run_target_freqs" not in st.session_state:
		st.session_state.p6_cube_last_run_target_freqs = []
	if "p6_cube_download_cache" not in st.session_state:
		st.session_state.p6_cube_download_cache = []
	if "p6_cube_download_selected" not in st.session_state:
		st.session_state.p6_cube_download_selected = ""
	if "p6_guide_freqs_main_last_nonempty" not in st.session_state:
		st.session_state.p6_guide_freqs_main_last_nonempty = ""
	if "p6_guide_freqs_cube2_last_nonempty" not in st.session_state:
		st.session_state.p6_guide_freqs_cube2_last_nonempty = ""
	if "p6_guide_freqs_cube3_input" not in st.session_state:
		st.session_state.p6_guide_freqs_cube3_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
	if "p6_guide_freqs_cube3_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_cube3_pending = ""
	if "p6_guide_cube3_refresh" not in st.session_state:
		st.session_state.p6_guide_cube3_refresh = False
	if "p6_guide_freqs_cube3_last_nonempty" not in st.session_state:
		st.session_state.p6_guide_freqs_cube3_last_nonempty = ""
	if "p6_synth_only_results" not in st.session_state:
		st.session_state.p6_synth_only_results = {}
	if "p6_synth_only_warnings" not in st.session_state:
		st.session_state.p6_synth_only_warnings = []
	if "p6_guide_freqs_fit_input" not in st.session_state:
		st.session_state.p6_guide_freqs_fit_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
	if "p6_guide_freqs_fit_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_fit_pending = ""
	if "p6_guide_fit_refresh" not in st.session_state:
		st.session_state.p6_guide_fit_refresh = False
	if "p6_guide_freqs_fit_last_nonempty" not in st.session_state:
		st.session_state.p6_guide_freqs_fit_last_nonempty = ""
	if "p6_fit_last_result" not in st.session_state:
		st.session_state.p6_fit_last_result = None
	if "p6_fit_upload_signature" not in st.session_state:
		st.session_state.p6_fit_upload_signature = ""
	if "p6_fit_sources_signature" not in st.session_state:
		st.session_state.p6_fit_sources_signature = ""
	if "p6_fit_candidate_mode_prev" not in st.session_state:
		st.session_state.p6_fit_candidate_mode_prev = ""


def _clear_fitting_outputs():
	# Clear previous fitting payload and dynamic output keys.
	for k in list(st.session_state.keys()):
		ks = str(k)
		if ks == "p6_fit_last_result":
			st.session_state.pop(k, None)
		elif ks.startswith("p6_fit_plot_"):
			st.session_state.pop(k, None)
		elif ks.startswith("p6_fit_tmp_"):
			st.session_state.pop(k, None)
	st.session_state.pop("p6_fit_global_overlay_plot", None)
	st.session_state.pop("p6_roi_overview_fit", None)
	try:
		gc.collect()
	except Exception:
		pass
	try:
		if bool(getattr(torch, "cuda", None)) and bool(torch.cuda.is_available()):
			torch.cuda.empty_cache()
			torch.cuda.ipc_collect()
	except Exception:
		pass


def _is_running() -> bool:
	proc = st.session_state.get("cube_proc", None)
	return proc is not None and proc.poll() is None


def _is_sim_running() -> bool:
	proc = st.session_state.get("sim_proc", None)
	return proc is not None and proc.poll() is None


def _is_cubefit_running() -> bool:
	proc = st.session_state.get("cubefit_proc", None)
	return proc is not None and proc.poll() is None


def _stop_process():
	proc = st.session_state.get("cube_proc", None)
	if proc is not None and proc.poll() is None:
		try:
			proc.terminate()
			proc.wait(timeout=6)
		except Exception:
			try:
				proc.kill()
			except Exception:
				pass
	st.session_state.cube_proc = None
	fh = st.session_state.get("cube_log_handle", None)
	if fh is not None:
		try:
			fh.close()
		except Exception:
			pass
	st.session_state.cube_log_handle = None
	cfgp = st.session_state.get("cube_cfg_path", "")
	if cfgp and os.path.isfile(cfgp):
		try:
			os.remove(cfgp)
		except Exception:
			pass
	st.session_state.cube_cfg_path = ""


def _stop_sim_process():
	proc = st.session_state.get("sim_proc", None)
	if proc is not None and proc.poll() is None:
		try:
			proc.terminate()
			proc.wait(timeout=6)
		except Exception:
			try:
				proc.kill()
			except Exception:
				pass
	st.session_state.sim_proc = None
	fh = st.session_state.get("sim_log_handle", None)
	if fh is not None:
		try:
			fh.close()
		except Exception:
			pass
	st.session_state.sim_log_handle = None
	cfgp = st.session_state.get("sim_cfg_path", "")
	if cfgp and os.path.isfile(cfgp):
		try:
			os.remove(cfgp)
		except Exception:
			pass
	st.session_state.sim_cfg_path = ""


def _stop_cubefit_process():
	proc = st.session_state.get("cubefit_proc", None)
	if proc is not None and proc.poll() is None:
		try:
			proc.terminate()
			proc.wait(timeout=6)
		except Exception:
			try:
				proc.kill()
			except Exception:
				pass
	st.session_state.cubefit_proc = None
	fh = st.session_state.get("cubefit_log_handle", None)
	if fh is not None:
		try:
			fh.close()
		except Exception:
			pass
	st.session_state.cubefit_log_handle = None
	cfgp = st.session_state.get("cubefit_cfg_path", "")
	if cfgp and os.path.isfile(cfgp):
		try:
			os.remove(cfgp)
		except Exception:
			pass
	st.session_state.cubefit_cfg_path = ""


def _load_uploaded_map_preview(upload_obj):
	if upload_obj is None or fits is None:
		return None, None
	try:
		upload_obj.seek(0)
		raw = upload_obj.read()
		if not raw:
			return None, "Empty file"
		from io import BytesIO
		arr = np.asarray(fits.getdata(BytesIO(raw)), dtype=np.float32)
		if arr.ndim == 4:
			arr = arr[0, 0, :, :]
		elif arr.ndim == 3:
			arr = arr[0, :, :]
		elif arr.ndim != 2:
			return None, f"Unsupported shape: {arr.shape}"
		arr[~np.isfinite(arr)] = np.nan
		return arr, None
	except Exception as e:
		return None, str(e)


def _show_fits_preview(title: str, arr: np.ndarray):
	if arr is None:
		return
	v = np.asarray(arr, dtype=np.float32)
	fin = np.isfinite(v)
	if not np.any(fin):
		return
	vf = v[fin]
	vmin = float(np.percentile(vf, 1.0))
	vmax = float(np.percentile(vf, 99.0))
	if vmax <= vmin:
		vmax = vmin + 1e-6
	fig, ax = plt.subplots(figsize=(4.2, 4.0))
	im = ax.imshow(v, origin="lower", cmap="viridis", vmin=vmin, vmax=vmax)
	ax.set_title(f"{title} | shape={v.shape[0]}x{v.shape[1]}")
	plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
	plt.tight_layout()
	st.pyplot(fig, width="stretch")
	plt.close(fig)


def _create_implicit_param_maps(
	out_root_dir: str,
	logn_val: float,
	tex_val: float,
	velo_val: float,
	fwhm_val: float,
	ny: int,
	nx: int,
	reference_fits_path: str = "",
):
	if fits is None:
		raise RuntimeError("FITS backend not available")
	header = None
	shape_ny = int(max(1, ny))
	shape_nx = int(max(1, nx))
	ref_path = str(reference_fits_path).strip()
	if ref_path and os.path.isfile(ref_path):
		try:
			arr_ref = np.asarray(fits.getdata(ref_path), dtype=np.float32)
			if arr_ref.ndim == 4:
				arr_ref = arr_ref[0, 0, :, :]
			elif arr_ref.ndim == 3:
				arr_ref = arr_ref[0, :, :]
			elif arr_ref.ndim != 2:
				raise RuntimeError(f"Unsupported reference shape: {arr_ref.shape}")
			shape_ny, shape_nx = int(arr_ref.shape[0]), int(arr_ref.shape[1])
			header = fits.getheader(ref_path)
		except Exception:
			header = None

	maps_dir = os.path.join(str(out_root_dir), f"implicit_maps_{time.strftime('%Y%m%d_%H%M%S')}")
	os.makedirs(maps_dir, exist_ok=True)

	arr_logn = np.full((shape_ny, shape_nx), float(logn_val), dtype=np.float32)
	arr_tex = np.full((shape_ny, shape_nx), float(tex_val), dtype=np.float32)
	arr_velo = np.full((shape_ny, shape_nx), float(velo_val), dtype=np.float32)
	arr_fwhm = np.full((shape_ny, shape_nx), float(fwhm_val), dtype=np.float32)

	map_files = {
		"tex": "IMPLICIT_TEX.fits",
		"logn": "IMPLICIT_LOGN.fits",
		"velo": "IMPLICIT_VELO.fits",
		"fwhm": "IMPLICIT_FWHM.fits",
	}

	fits.writeto(os.path.join(maps_dir, map_files["tex"]), arr_tex, header=header, overwrite=True)
	fits.writeto(os.path.join(maps_dir, map_files["logn"]), arr_logn, header=header, overwrite=True)
	fits.writeto(os.path.join(maps_dir, map_files["velo"]), arr_velo, header=header, overwrite=True)
	fits.writeto(os.path.join(maps_dir, map_files["fwhm"]), arr_fwhm, header=header, overwrite=True)
	return maps_dir, map_files


def _resolve_local_file(filename: str) -> Path:
	p = _project_dir() / str(filename)
	if not p.is_file():
		raise FileNotFoundError(f"Missing required file: {p}")
	return p


def _cleanup_generated_outputs_on_startup_once():
	if bool(st.session_state.get("p6_cleanup_done", False)):
		return
	roots = [
		str(DEFAULT_OUTPUT_DIR),
		os.path.join(str(DEFAULT_OUTPUT_DIR), "cube2"),
		os.path.join(str(DEFAULT_OUTPUT_DIR), "cube_fit"),
	]
	for root in roots:
		if (not root) or (not os.path.isdir(root)):
			continue
		for name in os.listdir(root):
			p = os.path.join(root, name)
			ln = str(name).lower()
			try:
				if os.path.isfile(p):
					is_cube_fits = str(name).startswith(f"{DEFAULT_OUT_PREFIX}_target") and ln.endswith(".fits")
					is_progress_png = ln.endswith("_inprogress_map.png")
					is_progress_json = ln.endswith("_inprogress_map.json")
					is_run_log = (ln.startswith("cube_run_") or ln.startswith("cube2_run_") or ln.startswith("cubefit_run_")) and ln.endswith(".log")
					if is_cube_fits or is_progress_png or is_progress_json or is_run_log:
						os.remove(p)
				elif os.path.isdir(p):
					if str(name).startswith("uploaded_maps_") or str(name).startswith("implicit_maps_"):
						shutil.rmtree(p, ignore_errors=True)
			except Exception:
				pass
	st.session_state.p6_cleanup_done = True


def _cleanup_generated_outputs_for_dir(out_dir: str, include_cube2_logs: bool = False):
	if (not out_dir) or (not os.path.isdir(out_dir)):
		return
	for name in os.listdir(out_dir):
		p = os.path.join(out_dir, name)
		ln = str(name).lower()
		try:
			if os.path.isfile(p):
				is_cube_fits = str(name).startswith(f"{DEFAULT_OUT_PREFIX}_target") and ln.endswith(".fits")
				is_progress_png = ln.endswith("_inprogress_map.png")
				is_progress_json = ln.endswith("_inprogress_map.json")
				is_run_log = ln.startswith("cube_run_") and ln.endswith(".log")
				if include_cube2_logs:
					is_run_log = is_run_log or (ln.startswith("cube2_run_") and ln.endswith(".log"))
				if is_cube_fits or is_progress_png or is_progress_json or is_run_log:
					os.remove(p)
			elif os.path.isdir(p):
				if str(name).startswith("uploaded_maps_") or str(name).startswith("implicit_maps_"):
					shutil.rmtree(p, ignore_errors=True)
		except Exception:
			pass


def _cleanup_cubefit_outputs_for_dir(out_dir: str):
	if (not out_dir) or (not os.path.isdir(out_dir)):
		return
	for name in os.listdir(out_dir):
		p = os.path.join(out_dir, name)
		ln = str(name).lower()
		try:
			if os.path.isfile(p):
				is_cubefit_fits = ln.startswith("cubefit_") and ln.endswith(".fits")
				is_progress_png = ln.endswith("_inprogress_map.png")
				is_progress_json = ln.endswith("_inprogress_map.json")
				is_run_log = ln.startswith("cubefit_run_") and ln.endswith(".log")
				if is_cubefit_fits or is_progress_png or is_progress_json or is_run_log:
					os.remove(p)
		except Exception:
			pass


def _load_module_from_path(path: Path, module_name: str):
	if not path.is_file():
		raise FileNotFoundError(f"Cannot load module from missing file: {path}")
	with open(path, "r", encoding="utf-8", errors="ignore") as f:
		full_code = f.read()

	cut = len(full_code)
	patterns = [
		r"^\s*st\.set_page_config\s*\(",
		r"^\s*#\s*=+\s*$\n^\s*#\s*UI\s*$",
		r"^\s*with\s+st\.sidebar\s*:",
	]
	for pat in patterns:
		m = re.search(pat, full_code, flags=re.MULTILINE)
		if m:
			cut = min(cut, int(m.start()))

	core_code = full_code[:cut]
	if not core_code.strip():
		raise RuntimeError(f"Could not isolate non-UI code from: {path}")

	module = importlib.util.module_from_spec(importlib.util.spec_from_loader(module_name, loader=None))
	exec(compile(core_code, str(path), "exec"), module.__dict__)
	return module


@st.cache_resource(show_spinner=False)
def _load_syn_module_cached(path_str: str):
	path = Path(path_str)
	return _load_module_from_path(path, "syngen_v4_for_6_cached")


@st.cache_data(show_spinner=False, max_entries=128)
def _generate_obs_payload_cached(
	signal_models_root: str,
	noise_models_root: str,
	filter_file: str,
	target_freqs_tuple: Tuple[float, ...],
	allow_nearest: bool,
	logn: float,
	tex: float,
	velo: float,
	fwhm: float,
	obs_noise_scale: float,
):
	syngen_path = _resolve_local_file("4.SYNGEN_Streamlit_v1.py")
	syn_mod = _load_syn_module_cached(str(syngen_path))
	payload, n_with_noise, filter_roi_bounds = syn_mod.generate_exact_style16_payload(
		signal_models_root=signal_models_root,
		noise_models_root=noise_models_root,
		x_features=[float(logn), float(tex), float(velo), float(fwhm)],
		pred_mode="ensemble_mean",
		selected_model_name="GradientBoosting",
		noise_scale=float(obs_noise_scale),
		filter_file=filter_file,
		target_freqs=[float(v) for v in target_freqs_tuple],
		allow_nearest=bool(allow_nearest),
	)
	return payload, int(n_with_noise), filter_roi_bounds


def run_streamlit_app():
	st.set_page_config(page_title="OBSEMULATOR", page_icon="🧪", layout="wide")
	_ensure_state()
	_cleanup_generated_outputs_on_startup_once()
	st.title("OBSEMULATOR")

	intro_img = _project_dir() / "NGC6523_BVO_2.jpg"
	if intro_img.is_file():
		st.image(str(intro_img), width="stretch")

	st.markdown(
		"""
A remarkable upsurge in the complexity of molecules identified in the interstellar medium (ISM) is currently occurring, with over 80 new species discovered in the last three years. A number of them have been emphasized by prebiotic experiments as vital molecular building blocks of life. Since our Solar System was formed from a molecular cloud in the ISM, it prompts the query as to whether the rich interstellar chemical reservoir could have played a role in the emergence of life. The improved sensitivities of state-of-the-art astronomical facilities, such as the Atacama Large Millimeter/submillimeter Array (ALMA) and the James Webb Space Telescope (JWST), are revolutionizing the discovery of new molecules in space. However, we are still just scraping the tip of the iceberg. We are far from knowing the complete catalogue of molecules that astrochemistry can offer, as well as the complexity they can reach.

**Artificial Intelligence Integral Tool for AstroChemical Analysis (AI-ITACA)** proposes to combine complementary machine learning (ML) techniques to address all the challenges that astrochemistry is currently facing. AI-ITACA will significantly contribute to the development of new AI-based cutting-edge analysis software that will allow us to make a crucial leap in the characterization of the level of chemical complexity in the ISM, and in our understanding of the contribution that interstellar chemistry might have in the origin of life.
"""
	)
	signal_models_root = str(DEFAULT_MERGED_H5)
	noise_models_root = str(DEFAULT_NOISE_NN_H5)
	filter_file = str(DEFAULT_FILTER_FILE)

	with st.sidebar:
		st.header("Model Upload")
		st.markdown("**Model Upload**")
		up_signal_h5 = st.file_uploader("Upload signal models (.h5)", type=["h5", "hdf5"], key="p6_up_signal_h5")
		up_noise_h5 = st.file_uploader("Upload noise models (.h5 bundle or single model)", type=["h5", "hdf5"], key="p6_up_noise_h5")
		up_filter = st.file_uploader("Upload spectral filter (.txt/.dat/.csv)", type=["txt", "dat", "csv"], key="p6_up_filter")

		uploaded_signal_path = _save_uploaded_file_to_temp(up_signal_h5, "signal")
		uploaded_noise_path = _save_uploaded_file_to_temp(up_noise_h5, "noise")
		uploaded_filter_path = _save_uploaded_file_to_temp(up_filter, "filter")
		if uploaded_signal_path:
			signal_models_root = str(uploaded_signal_path)
		if uploaded_noise_path:
			noise_models_root = str(uploaded_noise_path)
		if uploaded_filter_path:
			filter_file = str(uploaded_filter_path)

		st.markdown("---")
		st.markdown("**Optional: use temporary Google Drive download**")
		use_drive_temp = st.checkbox("Use Google Drive temporary models", value=False, key="p6_use_drive_temp")
		drive_link = st.text_input("Google Drive folder link", value=DEFAULT_GDRIVE_MODELS_LINK, key="p6_drive_link")
		download_drive_now = st.button("Download / refresh from Drive", key="p6_drive_download_btn")

		if use_drive_temp and (download_drive_now or (not str(st.session_state.get("drive_cache_dir", "")).strip())):
			with st.spinner("Downloading models from Google Drive to temporary storage..."):
				drive_dir, drive_err = _download_gdrive_folder_temp(drive_link)
			if drive_err is not None:
				st.session_state.drive_last_error = str(drive_err)
			else:
				st.session_state.drive_cache_dir = str(drive_dir)
				st.session_state.drive_last_error = ""
				auto_paths = _detect_model_data_paths(str(drive_dir))
				st.session_state.drive_auto_paths = dict(auto_paths)

		if use_drive_temp:
			drive_err_text = str(st.session_state.get("drive_last_error", "")).strip()
			if drive_err_text:
				st.error(drive_err_text)
			auto_paths = st.session_state.get("drive_auto_paths", {}) if isinstance(st.session_state.get("drive_auto_paths", {}), dict) else {}
			if auto_paths:
				for w in auto_paths.get("warnings", []):
					st.warning(str(w))
				if auto_paths.get("signal_models_source", ""):
					signal_models_root = str(auto_paths["signal_models_source"])
				if auto_paths.get("noise_models_root", ""):
					noise_models_root = str(auto_paths["noise_models_root"])
				if auto_paths.get("filter_file", ""):
					filter_file = str(auto_paths["filter_file"])
				st.caption("Active source mode: Google Drive temporary download")
			else:
				st.caption("Active source mode: manual paths (Drive not ready yet)")
		else:
			st.caption("Active source mode: manual paths")

		st.caption(f"Signal source in use: {signal_models_root}")
		st.caption(f"Noise source in use: {noise_models_root}")
		st.caption(f"Filter file in use: {filter_file}")

		# If model/filter sources change, clear fitting outputs to avoid stale state.
		sources_sig = "|".join([
			str(signal_models_root or ""),
			str(noise_models_root or ""),
			str(filter_file or ""),
		])
		prev_sources_sig = str(st.session_state.get("p6_fit_sources_signature", ""))
		if prev_sources_sig and (prev_sources_sig != sources_sig):
			_clear_fitting_outputs()
		st.session_state.p6_fit_sources_signature = str(sources_sig)

		target_text = st.text_area("Default target frequencies (GHz)", value=", ".join([str(v) for v in DEFAULT_TARGET_FREQS]), height=120)
		target_freqs = parse_freq_list(target_text)
		if not target_freqs:
			target_freqs = [float(v) for v in DEFAULT_TARGET_FREQS]
		allow_nearest = False
		st.caption("ROI selection mode for cube generation: exact overlap only (nearest disabled).")
		noise_scale = st.number_input("Noise scale", min_value=0.0, value=float(DEFAULT_NOISE_SCALE), step=0.1, format="%.3f")

	tab_cube, tab_cube2, tab_cube3, tab_fit, tab_cube_fit = st.tabs(["Cube Generator", "Simulate Single Spectrum", "Simulate Single Synthetic Spectrum", "Fitting", "Cube Fitting"])

	try:
		syngen_path = _resolve_local_file("4.SYNGEN_Streamlit_v1.py")
		syn = _load_syn_module_cached(str(syngen_path))
		syn_load_error = None
	except Exception as e:
		syn = None
		syn_load_error = str(e)

	with tab_cube:
		st.subheader("Cube Generator | CH3OCHO")
		st.markdown("**ROI explorer (signal and noise models)**")
		if bool(st.session_state.get("p6_guide_main_refresh", False)):
			st.session_state.p6_guide_freqs_main_input = str(st.session_state.get("p6_guide_freqs_main_pending", "")).strip()
			st.session_state.p6_guide_main_refresh = False
			st.session_state.p6_guide_freqs_main_pending = ""
		if not str(st.session_state.get("p6_guide_freqs_main_input", "")).strip():
			last_main = str(st.session_state.get("p6_guide_freqs_main_last_nonempty", "")).strip()
			if last_main:
				st.session_state.p6_guide_freqs_main_input = last_main
			else:
				st.session_state.p6_guide_freqs_main_input = _freqs_to_text([float(v) for v in target_freqs])
		guide_freqs_text = st.text_input(
			"Guide frequencies (GHz; main list used for Start cube generation)",
			key="p6_guide_freqs_main_input",
		)
		if str(guide_freqs_text).strip():
			st.session_state.p6_guide_freqs_main_last_nonempty = str(guide_freqs_text).strip()
		guide_freqs = parse_freq_list(guide_freqs_text)
		guide_freq = float(guide_freqs[0]) if len(guide_freqs) > 0 else None

		signal_rois = _collect_signal_rois_for_ui(signal_models_root, filter_file)
		noise_rois = _collect_noise_rois_for_ui(noise_models_root)
		signal_rois, noise_rois = _mark_roi_overlaps(signal_rois, noise_rois)
		target_freqs_cube = [float(v) for v in target_freqs]

		if (not signal_rois) and (not noise_rois):
			st.info("Could not load ROIs yet. Check model/filter paths.")
		else:
			if "p6_roi_guide_prev" not in st.session_state:
				st.session_state.p6_roi_guide_prev = None
			guide_key = tuple([round(float(v), 9) for v in guide_freqs])
			guide_changed = st.session_state.p6_roi_guide_prev != guide_key
			if guide_changed:
				if signal_rois:
					st.session_state.p6_signal_roi_select = int(_pick_default_roi_index(signal_rois, guide_freq))
				if noise_rois:
					st.session_state.p6_noise_roi_select = int(_pick_default_roi_index(noise_rois, guide_freq))
				st.session_state.p6_roi_guide_prev = guide_key

			if signal_rois:
				if int(st.session_state.get("p6_signal_roi_select", 0)) >= len(signal_rois):
					st.session_state.p6_signal_roi_select = 0
			if noise_rois:
				if int(st.session_state.get("p6_noise_roi_select", 0)) >= len(noise_rois):
					st.session_state.p6_noise_roi_select = 0

			c_roi_1, c_roi_2 = st.columns(2)
			with c_roi_1:
				if signal_rois:
					sig_opts = list(range(len(signal_rois)))
					st.selectbox(
						"Synthetic-model ROIs",
						options=sig_opts,
						format_func=lambda i: (
							f"ROI S{signal_rois[i]['index']} | {signal_rois[i]['lo']:.6f}–{signal_rois[i]['hi']:.6f} GHz"
							+ (
								f" | MATCHED BETWEEN MODELS: N{',N'.join([str(v) for v in _get_overlapping_noise_roi_indices(signal_rois[i], noise_rois)])}"
								if _get_overlapping_noise_roi_indices(signal_rois[i], noise_rois)
								else " | no overlap"
							)
						),
						key="p6_signal_roi_select",
					)
					sel_s = signal_rois[int(st.session_state.p6_signal_roi_select)]
					match_n = _get_overlapping_noise_roi_indices(sel_s, noise_rois)
					match_txt = ",".join([str(v) for v in match_n]) if match_n else "none"
					st.caption(f"Selected: ROI S{int(sel_s['index'])} | range {float(sel_s['lo']):.6f}–{float(sel_s['hi']):.6f} GHz | matching Noise ROI(s): {match_txt}")
				else:
					st.caption("No signal ROIs available")

			with c_roi_2:
				if noise_rois:
					noi_opts = list(range(len(noise_rois)))
					st.selectbox(
						"Noise-model ROIs",
						options=noi_opts,
						format_func=lambda i: (
							f"ROI N{noise_rois[i]['index']} | {noise_rois[i]['lo']:.6f}–{noise_rois[i]['hi']:.6f} GHz"
							+ (
								f" | MATCHED BETWEEN MODELS: S{',S'.join([str(v) for v in _get_overlapping_signal_roi_indices(noise_rois[i], signal_rois)])}"
								if _get_overlapping_signal_roi_indices(noise_rois[i], signal_rois)
								else " | no overlap"
							)
						),
						key="p6_noise_roi_select",
					)
					sel_n = noise_rois[int(st.session_state.p6_noise_roi_select)]
					spw_txt = ",".join(sel_n.get("spw", [])) if sel_n.get("spw", []) else "-"
					match_s = _get_overlapping_signal_roi_indices(sel_n, signal_rois)
					match_s_txt = ",".join([f"S{v}" for v in match_s]) if match_s else "none"
					st.caption(f"Selected: ROI N{int(sel_n['index'])} | range {float(sel_n['lo']):.6f}–{float(sel_n['hi']):.6f} GHz | SPW: {spw_txt} | matching Signal ROI(s): {match_s_txt}")
				else:
					st.caption("No noise ROIs available")

			sel_sig_idx = None if not signal_rois else int(signal_rois[int(st.session_state.get("p6_signal_roi_select", 0))]["index"])
			sel_noi_idx = None if not noise_rois else int(noise_rois[int(st.session_state.get("p6_noise_roi_select", 0))]["index"])
			combo_freqs = _selected_roi_combo_freqs(
				signal_rois=signal_rois,
				noise_rois=noise_rois,
				selected_signal_pos=int(st.session_state.get("p6_signal_roi_select", 0)) if signal_rois else None,
				selected_noise_pos=int(st.session_state.get("p6_noise_roi_select", 0)) if noise_rois else None,
			)
			_plot_roi_overview(signal_rois, noise_rois, guide_freqs_ghz=guide_freqs, selected_combo_freqs_ghz=combo_freqs, selected_signal_index=sel_sig_idx, selected_noise_index=sel_noi_idx, chart_key="p6_roi_overview_cube")

		if st.button("Add selected ROI combination to Guide frequencies", key="p6_add_rois_to_guide"):
			updated_freqs = _append_selected_rois_to_freq_list(
				base_freqs=guide_freqs,
				signal_rois=signal_rois,
				noise_rois=noise_rois,
				selected_signal_pos=int(st.session_state.get("p6_signal_roi_select", 0)) if signal_rois else None,
				selected_noise_pos=int(st.session_state.get("p6_noise_roi_select", 0)) if noise_rois else None,
			)
			st.session_state.p6_guide_freqs_main_pending = _freqs_to_text(updated_freqs)
			st.session_state.p6_guide_main_refresh = True
			st.rerun()

		guide_freqs_run = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_main_input", ""))))
		target_freqs_cube = [float(v) for v in guide_freqs_run]
		if target_freqs_cube:
			st.caption("Target frequencies used for Cube Generator: " + _freqs_to_text(target_freqs_cube))
		else:
			st.caption("Target frequencies used for Cube Generator: (empty)")

		default_out = DEFAULT_OUTPUT_DIR
		cube_out_dir = st.text_input("Output directory", value=default_out)
		st.markdown("**Upload parameter maps (FITS)**")
		mu1, mu2 = st.columns(2)
		with mu1:
			up_tex = st.file_uploader("Tex map (.fits)", type=["fits"], key="p6_map_tex")
			arr, _ = _load_uploaded_map_preview(up_tex)
			if arr is not None:
				_show_fits_preview("Tex preview", arr)
			up_logn = st.file_uploader("LogN map (.fits)", type=["fits"], key="p6_map_logn")
			arr, _ = _load_uploaded_map_preview(up_logn)
			if arr is not None:
				_show_fits_preview("LogN preview", arr)
		with mu2:
			up_velo = st.file_uploader("Velo map (.fits)", type=["fits"], key="p6_map_velo")
			arr, _ = _load_uploaded_map_preview(up_velo)
			if arr is not None:
				_show_fits_preview("Velo preview", arr)
			up_fwhm = st.file_uploader("FWHM map (.fits)", type=["fits"], key="p6_map_fwhm")
			arr, _ = _load_uploaded_map_preview(up_fwhm)
			if arr is not None:
				_show_fits_preview("FWHM preview", arr)

		progress_every = st.number_input("Progress every N pixels", min_value=1, value=int(DEFAULT_PROGRESS_EVERY), step=1)
		col_a, col_b = st.columns(2)
		with col_a:
			start_cube = st.button("Start cube generation", type="primary", disabled=_is_running())
		with col_b:
			stop_cube = st.button("Stop process", disabled=not _is_running())
		live_every_sec = 5
		st.caption("Live refresh (seconds): 5")

		if start_cube:
			target_freqs_cube_run = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_main_input", ""))))
			if not target_freqs_cube_run:
				st.error("Guide frequencies está vacío. Agrega al menos una frecuencia o usa 'Add selected ROI combination to Guide frequencies'.")
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			elif not _is_valid_noise_source(noise_models_root):
				st.error("Noise models root invalid.")
			else:
				try:
					os.makedirs(cube_out_dir, exist_ok=True)
					_cleanup_generated_outputs_for_dir(str(cube_out_dir), include_cube2_logs=False)
					st.session_state.p6_cube_download_cache = []
					st.session_state.p6_cube_download_selected = ""
					uploaded = {"tex": up_tex, "logn": up_logn, "velo": up_velo, "fwhm": up_fwhm}
					n_uploaded = sum(v is not None for v in uploaded.values())
					if n_uploaded not in (0, 4):
						raise RuntimeError("Upload all 4 maps or none.")
					if n_uploaded == 4:
						maps_dir = os.path.join(cube_out_dir, f"uploaded_maps_{time.strftime('%Y%m%d_%H%M%S')}")
						os.makedirs(maps_dir, exist_ok=True)
						map_files = {}
						for k, uf in uploaded.items():
							name = os.path.basename(str(uf.name))
							dst = os.path.join(maps_dir, name)
							with open(dst, "wb") as f_out:
								f_out.write(uf.getbuffer())
							map_files[k] = name
					else:
						maps_dir = str(DEFAULT_PARAM_MAPS_DIR)
						map_files = dict(DEFAULT_PARAM_MAP_FILES)
						if (not maps_dir) or (not os.path.isdir(maps_dir)):
							raise RuntimeError("No default parameter-map directory available. Upload the 4 FITS maps (Tex, LogN, Velo, FWHM).")
						missing_maps = [
							k for k, fn in map_files.items()
							if not os.path.isfile(os.path.join(maps_dir, fn))
						]
						if missing_maps:
							raise RuntimeError(f"Missing default parameter maps: {', '.join(missing_maps)}. Upload the 4 FITS maps.")

					cfg = {
						"out_dir": str(cube_out_dir),
						"param_maps_dir": str(maps_dir),
						"param_map_files": map_files,
						"signal_models_source": str(signal_models_root),
						"noise_models_root": str(noise_models_root),
						"filter_file": str(filter_file),
						"target_freqs": [float(v) for v in target_freqs_cube_run],
						"progress_every": int(progress_every),
						"allow_nearest": bool(allow_nearest),
						"noise_scale": float(noise_scale),
						"pred_mode": DEFAULT_PRED_MODE,
						"selected_model_name": DEFAULT_SELECTED_MODEL_NAME,
						"out_prefix": DEFAULT_OUT_PREFIX,
					}
					fd, cfg_path = tempfile.mkstemp(prefix="predobs6_cfg_", suffix=".json", dir=tempfile.gettempdir())
					os.close(fd)
					with open(cfg_path, "w", encoding="utf-8") as f:
						json.dump(cfg, f, ensure_ascii=False, indent=2)
					log_path = os.path.join(cube_out_dir, f"cube_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
					log_fh = open(log_path, "a", encoding="utf-8", buffering=1)
					proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--cube-worker", cfg_path], cwd=str(_project_dir()), stdout=log_fh, stderr=subprocess.STDOUT, text=True)
					st.session_state.cube_proc = proc
					st.session_state.cube_log_path = log_path
					st.session_state.cube_cfg_path = cfg_path
					st.session_state.cube_log_handle = log_fh
					st.session_state.p6_cube_last_run_target_freqs = [float(v) for v in target_freqs_cube_run]
					st.success("Cube generation started.")
				except Exception as e:
					st.error(f"Could not start process: {e}")

		if stop_cube:
			_stop_process()
			st.warning("Cube generation stopped by user.")

		running = _is_running()
		if running:
			st.info("Status: running")
		else:
			proc = st.session_state.get("cube_proc", None)
			if proc is not None:
				code = proc.poll()
				if code == 0:
					st.success("Status: finished successfully")
					warn_lines = _read_warn_lines(str(st.session_state.get("cube_log_path", "")), max_lines=120)
					if warn_lines:
						st.warning("Se detectaron frecuencias objetivo con fallo. Revisa el detalle del log.")
						with st.expander("Show worker warnings"):
							st.text("\n".join(warn_lines))
				elif code is not None:
					st.error(f"Status: finished with code {code}")
					log_tail = _read_log_tail(str(st.session_state.get("cube_log_path", "")), n_lines=80)
					if log_tail:
						with st.expander("Show last worker log lines"):
							st.text(log_tail)
				_stop_process()
			else:
				st.caption("Status: idle")

		progress_png = _find_latest_progress_png(cube_out_dir)
		if progress_png:
			st.markdown("**Cube progress**")
			progress_caption = _read_progress_info_caption(progress_png)
			progress_info = _read_progress_info(progress_png)
			if isinstance(progress_info, dict):
				done_steps = int(progress_info.get("done_steps", 0))
				total_steps = int(max(1, progress_info.get("total_steps", 1)))
				pct = 100.0 * float(done_steps) / float(total_steps)
				st.success(f"**Pixels processed:** {done_steps}/{total_steps} ({pct:.1f}%)")
			if progress_caption:
				st.caption(progress_caption)
			cp1, cp2, cp3 = st.columns([1, 2, 1])
			with cp2:
				img_bytes = _read_progress_png_stable_bytes(progress_png)
				if img_bytes is not None:
					try:
						st.image(img_bytes, caption=os.path.basename(progress_png), width=430)
					except Exception:
						st.caption("Progress image is being updated, retrying on next refresh...")
				else:
					st.caption("Progress image is being written, retrying on next refresh...")

		final_cubes_all_main = _find_all_final_main_cubes(cube_out_dir)
		guide_targets_for_cube = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_main_input", ""))))
		if not guide_targets_for_cube:
			guide_targets_for_cube = [float(v) for v in st.session_state.get("p6_cube_last_run_target_freqs", []) if np.isfinite(float(v))]
		final_cubes_main = _filter_cubes_by_target_freqs(final_cubes_all_main, guide_targets_for_cube)
		if guide_targets_for_cube:
			st.caption("ROIs simuladas para Guide frequencies: " + _freqs_to_text(guide_targets_for_cube))
		missing_targets_main = _find_missing_target_freqs(guide_targets_for_cube, final_cubes_all_main)
		if missing_targets_main:
			fail_reasons_main = _read_target_failure_reasons(str(st.session_state.get("cube_log_path", "")))
			if fail_reasons_main:
				msg_lines_main: List[str] = []
				for mf in missing_targets_main:
					reasons = fail_reasons_main.get(float(mf), [])
					if reasons:
						msg_lines_main.append(f"{float(mf):.6f} GHz -> {reasons[-1]}")
				if msg_lines_main:
					with st.expander("Why did these target frequencies fail?"):
						st.text("\n".join(msg_lines_main))

		latest_final = final_cubes_main[-1] if final_cubes_main else _find_latest_final_main_cube(cube_out_dir)
		shape_ref = _get_cube_ny_nx(latest_final) if latest_final else None
		if shape_ref is not None:
			ny_ref, nx_ref = int(shape_ref[0]), int(shape_ref[1])
			cube_changed = st.session_state.get("p6_spec_last_cube", "") != str(latest_final)
			shape_changed = (
				int(st.session_state.get("p6_spec_last_ny", -1)) != int(ny_ref)
				or int(st.session_state.get("p6_spec_last_nx", -1)) != int(nx_ref)
			)
			if ("p6_spec_pixel_y" not in st.session_state) or ("p6_spec_pixel_x" not in st.session_state) or cube_changed or shape_changed:
				st.session_state.p6_spec_pixel_y = int(ny_ref // 2)
				st.session_state.p6_spec_pixel_x = int(nx_ref // 2)
			st.session_state.p6_spec_last_cube = str(latest_final)
			st.session_state.p6_spec_last_ny = int(ny_ref)
			st.session_state.p6_spec_last_nx = int(nx_ref)
			st.session_state.p6_spec_pixel_y = int(max(0, min(ny_ref - 1, int(st.session_state.p6_spec_pixel_y))))
			st.session_state.p6_spec_pixel_x = int(max(0, min(nx_ref - 1, int(st.session_state.p6_spec_pixel_x))))
			sp1, sp2 = st.columns(2)
			with sp1:
				pix_y = st.number_input("Spectrum pixel Y", min_value=0, max_value=max(0, ny_ref - 1), value=int(st.session_state.p6_spec_pixel_y), step=1, key="p6_spec_pixel_y")
			with sp2:
				pix_x = st.number_input("Spectrum pixel X", min_value=0, max_value=max(0, nx_ref - 1), value=int(st.session_state.p6_spec_pixel_x), step=1, key="p6_spec_pixel_x")
		else:
			sp1, sp2 = st.columns(2)
			with sp1:
				pix_y = st.number_input("Spectrum pixel Y", min_value=0, value=0, step=1, key="p6_spec_pixel_y")
			with sp2:
				pix_x = st.number_input("Spectrum pixel X", min_value=0, value=0, step=1, key="p6_spec_pixel_x")

		plot_cubes_main = list(final_cubes_all_main)
		if (not running) and plot_cubes_main:
			st.markdown(f"**Final spectra grid (all generated cubes) | pixel (y={int(pix_y)}, x={int(pix_x)})**")
			n_cols_main = 2 if len(plot_cubes_main) <= 4 else 3
			cols_main = st.columns(n_cols_main)
			for i_pc, pc_path in enumerate(plot_cubes_main):
				freq_pc, y_syn_pc, y_noise_pc, y_final_pc, err_pc = _extract_pixel_spectra(pc_path, ypix=int(pix_y), xpix=int(pix_x))
				with cols_main[i_pc % n_cols_main]:
					st.caption(os.path.basename(pc_path))
					if err_pc is None:
						plot_key_pc = f"p6_spec_plot_cube_{os.path.basename(pc_path)}_{int(os.path.getmtime(pc_path))}_{int(pix_y)}_{int(pix_x)}"
						_plot_spectrum(freq_pc, y_syn_pc, y_noise_pc, y_final_pc, chart_key=plot_key_pc)
					else:
						st.error(f"Could not read spectrum: {err_pc}")
		elif running:
			st.caption("Spectrum grid will be shown after cube generation finishes.")

		st.markdown("**Download generated cube**")
		cubes_for_download_now = _find_all_final_main_cubes(cube_out_dir)
		cached_cubes = [
			str(p) for p in st.session_state.get("p6_cube_download_cache", [])
			if isinstance(p, str) and os.path.isfile(str(p))
		]
		merged_download = []
		seen_download = set()
		for p in list(cubes_for_download_now) + list(cached_cubes):
			sp = str(p)
			if (not sp) or (sp in seen_download) or (not os.path.isfile(sp)):
				continue
			seen_download.add(sp)
			merged_download.append(sp)
		st.session_state.p6_cube_download_cache = list(merged_download)

		if merged_download:
			prev_sel = str(st.session_state.get("p6_cube_download_selected", "")).strip()
			if prev_sel in merged_download:
				default_idx = int(merged_download.index(prev_sel))
			else:
				default_idx = int(max(0, len(merged_download) - 1))
			sel_cube_dl = st.selectbox(
				"Select cube",
				options=merged_download,
				index=default_idx,
				format_func=lambda p: os.path.basename(str(p)),
				key="p6_cube_download_select",
			)
			st.session_state.p6_cube_download_selected = str(sel_cube_dl)
			try:
				with open(str(sel_cube_dl), "rb") as f_in:
					cube_bytes = f_in.read()
				st.download_button(
					"Download selected observational cube (.fits)",
					data=cube_bytes,
					file_name=os.path.basename(str(sel_cube_dl)),
					mime="application/fits",
					key="p6_cube_download_button_obs",
				)

				synth_cube_path = str(sel_cube_dl)[:-5] + "_SYNTHONLY.fits"
				if os.path.isfile(synth_cube_path):
					with open(synth_cube_path, "rb") as f_syn:
						synth_bytes = f_syn.read()
					st.download_button(
						"Download synthetic cube (.fits)",
						data=synth_bytes,
						file_name=os.path.basename(str(synth_cube_path)),
						mime="application/fits",
						key="p6_cube_download_button_syn",
					)
				else:
					st.caption("Synthetic cube file not found for selected observational cube.")

				noise_bytes, noise_err = _build_noise_cube_bytes_from_pair(str(sel_cube_dl), str(synth_cube_path))
				if (noise_bytes is not None) and (noise_err is None):
					base_name = os.path.splitext(os.path.basename(str(sel_cube_dl)))[0]
					st.download_button(
						"Download noise cube (.fits) [observational - synthetic]",
						data=noise_bytes,
						file_name=f"{base_name}_NOISEONLY.fits",
						mime="application/fits",
						key="p6_cube_download_button_noise",
					)
				elif noise_err:
					st.caption(f"Noise cube not available: {noise_err}")
			except Exception as e:
				st.error(f"Could not prepare cube download: {e}")
		else:
			st.caption("No generated cubes available yet.")

		if running:
			wait_s = float(max(1, int(live_every_sec)))
			st.caption(f"Live view active (auto-refresh every {int(wait_s)}s)")
			time.sleep(wait_s)
			st.rerun()

	with tab_cube2:
		st.subheader("Simulate Single Spectrum | CH3OCHO")
		st.caption("Same workflow as Cube Generator, using implicit scalar values for LogN, Tex, FWHM, and Velocity.")

		st.markdown("**ROI explorer (signal and noise models)**")
		if bool(st.session_state.get("p6_guide_cube2_refresh", False)):
			st.session_state.p6_guide_freqs_cube2_input = str(st.session_state.get("p6_guide_freqs_cube2_pending", "")).strip()
			st.session_state.p6_guide_cube2_refresh = False
			st.session_state.p6_guide_freqs_cube2_pending = ""
		if not str(st.session_state.get("p6_guide_freqs_cube2_input", "")).strip():
			last_cube2 = str(st.session_state.get("p6_guide_freqs_cube2_last_nonempty", "")).strip()
			if last_cube2:
				st.session_state.p6_guide_freqs_cube2_input = last_cube2
			else:
				last_run_cube2 = [float(v) for v in st.session_state.get("p6_cube2_last_run_target_freqs", []) if np.isfinite(float(v))]
				if last_run_cube2:
					st.session_state.p6_guide_freqs_cube2_input = _freqs_to_text(last_run_cube2)
				else:
					st.session_state.p6_guide_freqs_cube2_input = _freqs_to_text([float(v) for v in target_freqs])
		guide_freqs_text2 = st.text_input(
			"Guide frequencies (GHz; main list used for Generate Observation)",
			key="p6_guide_freqs_cube2_input",
		)
		if str(guide_freqs_text2).strip():
			st.session_state.p6_guide_freqs_cube2_last_nonempty = str(guide_freqs_text2).strip()
		guide_freqs2 = parse_freq_list(guide_freqs_text2)
		guide_freq2 = float(guide_freqs2[0]) if len(guide_freqs2) > 0 else None

		signal_rois2 = _collect_signal_rois_for_ui(signal_models_root, filter_file)
		noise_rois2 = _collect_noise_rois_for_ui(noise_models_root)
		signal_rois2, noise_rois2 = _mark_roi_overlaps(signal_rois2, noise_rois2)
		target_freqs_cube2 = [float(v) for v in target_freqs]

		if (not signal_rois2) and (not noise_rois2):
			st.info("Could not load ROIs yet. Check model/filter paths.")
		else:
			if "p6_roi2_guide_prev" not in st.session_state:
				st.session_state.p6_roi2_guide_prev = None
			guide_key2 = tuple([round(float(v), 9) for v in guide_freqs2])
			guide_changed2 = st.session_state.p6_roi2_guide_prev != guide_key2
			if guide_changed2:
				if signal_rois2:
					st.session_state.p6_signal_roi_select2 = int(_pick_default_roi_index(signal_rois2, guide_freq2))
				if noise_rois2:
					st.session_state.p6_noise_roi_select2 = int(_pick_default_roi_index(noise_rois2, guide_freq2))
				st.session_state.p6_roi2_guide_prev = guide_key2

			if signal_rois2:
				if int(st.session_state.get("p6_signal_roi_select2", 0)) >= len(signal_rois2):
					st.session_state.p6_signal_roi_select2 = 0
			if noise_rois2:
				if int(st.session_state.get("p6_noise_roi_select2", 0)) >= len(noise_rois2):
					st.session_state.p6_noise_roi_select2 = 0

			c2_roi_1, c2_roi_2 = st.columns(2)
			with c2_roi_1:
				if signal_rois2:
					sig_opts2 = list(range(len(signal_rois2)))
					st.selectbox(
						"Synthetic-model ROIs",
						options=sig_opts2,
						format_func=lambda i: (
							f"ROI S{signal_rois2[i]['index']} | {signal_rois2[i]['lo']:.6f}–{signal_rois2[i]['hi']:.6f} GHz"
							+ (
								f" | MATCHED BETWEEN MODELS: N{',N'.join([str(v) for v in _get_overlapping_noise_roi_indices(signal_rois2[i], noise_rois2)])}"
								if _get_overlapping_noise_roi_indices(signal_rois2[i], noise_rois2)
								else " | no overlap"
							)
						),
						key="p6_signal_roi_select2",
					)
					sel_s2 = signal_rois2[int(st.session_state.p6_signal_roi_select2)]
					match_n2 = _get_overlapping_noise_roi_indices(sel_s2, noise_rois2)
					match_txt2 = ",".join([str(v) for v in match_n2]) if match_n2 else "none"
					st.caption(f"Selected: ROI S{int(sel_s2['index'])} | range {float(sel_s2['lo']):.6f}–{float(sel_s2['hi']):.6f} GHz | matching Noise ROI(s): {match_txt2}")
				else:
					st.caption("No signal ROIs available")

			with c2_roi_2:
				if noise_rois2:
					noi_opts2 = list(range(len(noise_rois2)))
					st.selectbox(
						"Noise-model ROIs",
						options=noi_opts2,
						format_func=lambda i: (
							f"ROI N{noise_rois2[i]['index']} | {noise_rois2[i]['lo']:.6f}–{noise_rois2[i]['hi']:.6f} GHz"
							+ (
								f" | MATCHED BETWEEN MODELS: S{',S'.join([str(v) for v in _get_overlapping_signal_roi_indices(noise_rois2[i], signal_rois2)])}"
								if _get_overlapping_signal_roi_indices(noise_rois2[i], signal_rois2)
								else " | no overlap"
							)
						),
						key="p6_noise_roi_select2",
					)
					sel_n2 = noise_rois2[int(st.session_state.p6_noise_roi_select2)]
					spw_txt2 = ",".join(sel_n2.get("spw", [])) if sel_n2.get("spw", []) else "-"
					match_s2 = _get_overlapping_signal_roi_indices(sel_n2, signal_rois2)
					match_s_txt2 = ",".join([f"S{v}" for v in match_s2]) if match_s2 else "none"
					st.caption(f"Selected: ROI N{int(sel_n2['index'])} | range {float(sel_n2['lo']):.6f}–{float(sel_n2['hi']):.6f} GHz | SPW: {spw_txt2} | matching Signal ROI(s): {match_s_txt2}")
				else:
					st.caption("No noise ROIs available")

			sel_sig_idx2 = None if not signal_rois2 else int(signal_rois2[int(st.session_state.get("p6_signal_roi_select2", 0))]["index"])
			sel_noi_idx2 = None if not noise_rois2 else int(noise_rois2[int(st.session_state.get("p6_noise_roi_select2", 0))]["index"])
			combo_freqs2 = _selected_roi_combo_freqs(
				signal_rois=signal_rois2,
				noise_rois=noise_rois2,
				selected_signal_pos=int(st.session_state.get("p6_signal_roi_select2", 0)) if signal_rois2 else None,
				selected_noise_pos=int(st.session_state.get("p6_noise_roi_select2", 0)) if noise_rois2 else None,
			)
			_plot_roi_overview(signal_rois2, noise_rois2, guide_freqs_ghz=guide_freqs2, selected_combo_freqs_ghz=combo_freqs2, selected_signal_index=sel_sig_idx2, selected_noise_index=sel_noi_idx2, chart_key="p6_roi_overview_cube2")

		if st.button("Add selected ROI combination to Guide frequencies", key="p6_add_rois_to_guide_cube2"):
			updated_freqs2 = _append_selected_rois_to_freq_list(
				base_freqs=guide_freqs2,
				signal_rois=signal_rois2,
				noise_rois=noise_rois2,
				selected_signal_pos=int(st.session_state.get("p6_signal_roi_select2", 0)) if signal_rois2 else None,
				selected_noise_pos=int(st.session_state.get("p6_noise_roi_select2", 0)) if noise_rois2 else None,
			)
			st.session_state.p6_guide_freqs_cube2_pending = _freqs_to_text(updated_freqs2)
			st.session_state.p6_guide_cube2_refresh = True
			st.rerun()

		guide_freqs_run2 = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_cube2_input", ""))))
		target_freqs_cube2 = [float(v) for v in guide_freqs_run2]
		if target_freqs_cube2:
			st.caption("Target frequencies used for Simulate Single Spectrum: " + _freqs_to_text(target_freqs_cube2))
		else:
			st.caption("Target frequencies used for Simulate Single Spectrum: (empty)")
		st.caption("Las ROIs seleccionadas en desplegables solo se usarán si se agregan a Guide frequencies.")

		cube2_out_dir = st.text_input("Output directory", value=os.path.join(DEFAULT_OUTPUT_DIR, "cube2"), key="p6_cube2_outdir")
		p21, p22, p23, p24 = st.columns(4)
		with p21:
			logn_cube2 = st.number_input("LogN", value=18.248, format="%.4f", key="p6_cube2_logn")
		with p22:
			tex_cube2 = st.number_input("Tex", value=209.06, format="%.4f", key="p6_cube2_tex")
		with p23:
			fwhm_cube2 = st.number_input("FWHM", value=6.198, format="%.4f", key="p6_cube2_fwhm")
		with p24:
			velo_cube2 = st.number_input("Velocity", value=97.549, format="%.4f", key="p6_cube2_velo")

		col2_a, col2_b = st.columns(2)
		with col2_a:
			start_cube2 = st.button("Generate Observation", type="primary", key="p6_start_cube2", disabled=_is_running())
		with col2_b:
			stop_cube2 = st.button("Stop process", key="p6_stop_cube2", disabled=not _is_running())

		if start_cube2:
			target_freqs_cube2_run = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_cube2_input", ""))))
			guide_text2_now = str(st.session_state.get("p6_guide_freqs_cube2_input", "")).strip()
			if guide_text2_now:
				st.session_state.p6_guide_freqs_cube2_last_nonempty = guide_text2_now
			if not target_freqs_cube2_run:
				st.error("Guide frequencies está vacío. Agrega al menos una frecuencia o usa 'Add selected ROI combination to Guide frequencies'.")
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			elif not _is_valid_noise_source(noise_models_root):
				st.error("Noise models root invalid.")
			else:
				try:
					os.makedirs(cube2_out_dir, exist_ok=True)
					_cleanup_generated_outputs_for_dir(str(cube2_out_dir), include_cube2_logs=True)
					maps_dir2, map_files2 = _create_implicit_param_maps(
						out_root_dir=str(cube2_out_dir),
						logn_val=float(logn_cube2),
						tex_val=float(tex_cube2),
						velo_val=float(velo_cube2),
						fwhm_val=float(fwhm_cube2),
						ny=1,
						nx=1,
						reference_fits_path="",
					)

					cfg2 = {
						"out_dir": str(cube2_out_dir),
						"param_maps_dir": str(maps_dir2),
						"param_map_files": map_files2,
						"signal_models_source": str(signal_models_root),
						"noise_models_root": str(noise_models_root),
						"filter_file": str(filter_file),
						"target_freqs": [float(v) for v in target_freqs_cube2_run],
						"progress_every": int(DEFAULT_PROGRESS_EVERY),
						"allow_nearest": bool(allow_nearest),
						"noise_scale": float(noise_scale),
						"pred_mode": DEFAULT_PRED_MODE,
						"selected_model_name": DEFAULT_SELECTED_MODEL_NAME,
						"out_prefix": DEFAULT_OUT_PREFIX,
					}
					fd2, cfg_path2 = tempfile.mkstemp(prefix="predobs6_cfg2_", suffix=".json", dir=tempfile.gettempdir())
					os.close(fd2)
					with open(cfg_path2, "w", encoding="utf-8") as f:
						json.dump(cfg2, f, ensure_ascii=False, indent=2)
					log_path2 = os.path.join(cube2_out_dir, f"cube2_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
					log_fh2 = open(log_path2, "a", encoding="utf-8", buffering=1)
					proc2 = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "--cube-worker", cfg_path2], cwd=str(_project_dir()), stdout=log_fh2, stderr=subprocess.STDOUT, text=True)
					st.session_state.cube_proc = proc2
					st.session_state.cube_log_path = log_path2
					st.session_state.cube_cfg_path = cfg_path2
					st.session_state.cube_log_handle = log_fh2
					st.session_state.p6_cube2_last_run_target_freqs = [float(v) for v in target_freqs_cube2_run]
					st.session_state.p6_guide_freqs_cube2_input = _freqs_to_text([float(v) for v in target_freqs_cube2_run])
					st.session_state.p6_guide_freqs_cube2_last_nonempty = str(st.session_state.p6_guide_freqs_cube2_input)
					st.success("Cube generation started.")
				except Exception as e:
					st.error(f"Could not start process: {e}")

		if stop_cube2:
			_stop_process()
			st.warning("Cube generation stopped by user.")

		running2 = _is_running()
		if running2:
			st.info("Status: running")
		else:
			proc2 = st.session_state.get("cube_proc", None)
			if proc2 is not None:
				code2 = proc2.poll()
				if code2 == 0:
					st.success("Status: finished successfully")
					warn_lines2 = _read_warn_lines(str(st.session_state.get("cube_log_path", "")), max_lines=120)
					if warn_lines2:
						st.warning("Se detectaron frecuencias objetivo con fallo. Revisa el detalle del log.")
						with st.expander("Show worker warnings"):
							st.text("\n".join(warn_lines2))
				elif code2 is not None:
					st.error(f"Status: finished with code {code2}")
					log_tail2 = _read_log_tail(str(st.session_state.get("cube_log_path", "")), n_lines=80)
					if log_tail2:
						with st.expander("Show last worker log lines"):
							st.text(log_tail2)
				_stop_process()
			else:
				st.caption("Status: idle")

		final_cubes2_all = _find_all_final_main_cubes(cube2_out_dir)
		guide_targets_for_sim = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_cube2_input", ""))))
		if not guide_targets_for_sim:
			guide_targets_for_sim = [float(v) for v in st.session_state.get("p6_cube2_last_run_target_freqs", []) if np.isfinite(float(v))]
		final_cubes2 = _filter_cubes_by_target_freqs(final_cubes2_all, guide_targets_for_sim)
		if guide_targets_for_sim:
			st.caption("ROIs simuladas para Guide frequencies: " + _freqs_to_text(guide_targets_for_sim))
		missing_targets2 = _find_missing_target_freqs(guide_targets_for_sim, final_cubes2_all)
		if missing_targets2:
			fail_reasons2 = _read_target_failure_reasons(str(st.session_state.get("cube_log_path", "")))
			if fail_reasons2:
				msg_lines2: List[str] = []
				for mf in missing_targets2:
					reasons = fail_reasons2.get(float(mf), [])
					if reasons:
						msg_lines2.append(f"{float(mf):.6f} GHz -> {reasons[-1]}")
				if msg_lines2:
					with st.expander("Why did these target frequencies fail?"):
						st.text("\n".join(msg_lines2))
		plot_cubes2 = list(final_cubes2_all)
		freq_concat2 = None
		y_syn_concat2 = None
		y_noise_concat2 = None
		y_final_concat2 = None
		concat_used_paths2 = []
		concat_errors2 = []
		if (not running2) and plot_cubes2:
			freq_concat2, y_syn_concat2, y_noise_concat2, y_final_concat2, concat_used_paths2, concat_errors2 = _build_concatenated_spectra_from_cubes(
				plot_cubes2,
				ypix=0,
				xpix=0,
			)
			if (freq_concat2 is not None) and (y_final_concat2 is not None):
				st.markdown("**Concatenated spectrum overview (all generated targets)**")
				_plot_spectrum(freq_concat2, y_syn_concat2, y_noise_concat2, y_final_concat2, chart_key="p6_spec_plot_cube2_concat")
			elif concat_errors2:
				st.caption("Could not build concatenated spectrum from generated cubes.")
				with st.expander("Show concatenation details"):
					st.text("\n".join([str(v) for v in concat_errors2]))

		if (not running2) and plot_cubes2:
			st.markdown("**Final spectra by target frequency (1x1 cube)**")
			n_cols = 2 if len(plot_cubes2) <= 4 else 3
			cols_spec = st.columns(n_cols)
			for i_fc, fc_path in enumerate(plot_cubes2):
				freq2, y_syn2, y_noise2, y_final2, err2 = _extract_pixel_spectra(fc_path, ypix=0, xpix=0)
				with cols_spec[i_fc % n_cols]:
					st.caption(os.path.basename(fc_path))
					if err2 is None:
						plot_key = f"p6_spec_plot_cube2_{os.path.basename(fc_path)}_{int(os.path.getmtime(fc_path))}"
						_plot_spectrum(freq2, y_syn2, y_noise2, y_final2, chart_key=plot_key)
					else:
						st.error(f"Could not read spectrum: {err2}")
		elif (not running2):
			st.caption("No final spectra available yet.")

		with st.expander("Download spectrum (Simulate Single Spectrum)"):
			if (freq_concat2 is not None) and (y_final_concat2 is not None):
				txt_concat = _spectrum_to_txt_bytes(freq_concat2, y_syn_concat2, y_noise_concat2, y_final_concat2)
				if txt_concat is not None:
					st.download_button(
						"Download concatenated spectrum (.txt)",
						data=txt_concat,
						file_name="simulated_concatenated_spectrum.txt",
						mime="text/plain",
						key="p6_spec_download_concat_button",
					)
			if plot_cubes2:
				sel_spec_cube = st.selectbox(
					"Select cube to export spectrum",
					options=plot_cubes2,
					format_func=lambda p: os.path.basename(str(p)),
					key="p6_spec_download_select",
				)
				freq_dl, y_syn_dl, y_noise_dl, y_final_dl, err_dl = _extract_pixel_spectra(sel_spec_cube, ypix=0, xpix=0)
				if err_dl is not None:
					st.error(f"Could not read spectrum: {err_dl}")
				else:
					txt_bytes = _spectrum_to_txt_bytes(freq_dl, y_syn_dl, y_noise_dl, y_final_dl)
					if txt_bytes is None:
						st.error("Could not serialize spectrum to TXT.")
					else:
						base_name = os.path.splitext(os.path.basename(str(sel_spec_cube)))[0]
						st.download_button(
							"Download spectrum (.txt)",
							data=txt_bytes,
							file_name=f"{base_name}_spectrum.txt",
							mime="text/plain",
							key="p6_spec_download_button",
						)
			else:
				st.caption("No spectra available yet.")

		if running2:
			st.caption("Auto-updating every 5 seconds...")
			time.sleep(5)
			st.rerun()

	with tab_cube3:
		st.subheader("Simulate Single Synthetic Spectrum | CH3OCHO")
		st.caption("Same workflow as Simulate Single Spectrum, but using only synthetic-spectrum models (no noise models).")

		st.markdown("**ROI explorer (signal and noise models)**")
		if bool(st.session_state.get("p6_guide_cube3_refresh", False)):
			st.session_state.p6_guide_freqs_cube3_input = str(st.session_state.get("p6_guide_freqs_cube3_pending", "")).strip()
			st.session_state.p6_guide_cube3_refresh = False
			st.session_state.p6_guide_freqs_cube3_pending = ""
		if not str(st.session_state.get("p6_guide_freqs_cube3_input", "")).strip():
			last_cube3 = str(st.session_state.get("p6_guide_freqs_cube3_last_nonempty", "")).strip()
			if last_cube3:
				st.session_state.p6_guide_freqs_cube3_input = last_cube3
			else:
				st.session_state.p6_guide_freqs_cube3_input = _freqs_to_text([float(v) for v in target_freqs])

		guide_freqs_text3 = st.text_input(
			"Guide frequencies (GHz; main list used for Generate Synthetic Spectrum)",
			key="p6_guide_freqs_cube3_input",
		)
		if str(guide_freqs_text3).strip():
			st.session_state.p6_guide_freqs_cube3_last_nonempty = str(guide_freqs_text3).strip()
		guide_freqs3 = parse_freq_list(guide_freqs_text3)
		guide_freq3 = float(guide_freqs3[0]) if len(guide_freqs3) > 0 else None

		signal_rois3 = _collect_signal_rois_for_ui(signal_models_root, filter_file)
		noise_rois3 = _collect_noise_rois_for_ui(noise_models_root)
		signal_rois3, noise_rois3 = _mark_roi_overlaps(signal_rois3, noise_rois3)

		if (not signal_rois3) and (not noise_rois3):
			st.info("Could not load ROIs yet. Check model/filter paths.")
		else:
			if "p6_roi3_guide_prev" not in st.session_state:
				st.session_state.p6_roi3_guide_prev = None
			guide_key3 = tuple([round(float(v), 9) for v in guide_freqs3])
			guide_changed3 = st.session_state.p6_roi3_guide_prev != guide_key3
			if guide_changed3:
				if signal_rois3:
					st.session_state.p6_signal_roi_select3 = int(_pick_default_roi_index(signal_rois3, guide_freq3))
				st.session_state.p6_roi3_guide_prev = guide_key3

			if signal_rois3 and int(st.session_state.get("p6_signal_roi_select3", 0)) >= len(signal_rois3):
				st.session_state.p6_signal_roi_select3 = 0

			c3_roi_1, c3_roi_2 = st.columns([3, 2])
			with c3_roi_1:
				if signal_rois3:
					sig_opts3 = list(range(len(signal_rois3)))
					st.selectbox(
						"Synthetic-model ROIs",
						options=sig_opts3,
						format_func=lambda i: (
							f"ROI S{signal_rois3[i]['index']} | {signal_rois3[i]['lo']:.6f}–{signal_rois3[i]['hi']:.6f} GHz"
							+ (
								f" | MATCHED BETWEEN MODELS: N{',N'.join([str(v) for v in _get_overlapping_noise_roi_indices(signal_rois3[i], noise_rois3)])}"
								if _get_overlapping_noise_roi_indices(signal_rois3[i], noise_rois3)
								else " | no overlap"
							)
						),
						key="p6_signal_roi_select3",
					)
				else:
					st.caption("No signal ROIs available")

			with c3_roi_2:
				st.caption("Noise ROI selector hidden in this tab (synthetic-only mode).")

			sel_sig_idx3 = None if not signal_rois3 else int(signal_rois3[int(st.session_state.get("p6_signal_roi_select3", 0))]["index"])
			combo_freqs3 = _selected_roi_combo_freqs(
				signal_rois=signal_rois3,
				noise_rois=noise_rois3,
				selected_signal_pos=int(st.session_state.get("p6_signal_roi_select3", 0)) if signal_rois3 else None,
				selected_noise_pos=None,
			)
			_plot_roi_overview(signal_rois3, noise_rois3, guide_freqs_ghz=guide_freqs3, selected_combo_freqs_ghz=combo_freqs3, selected_signal_index=sel_sig_idx3, selected_noise_index=None, chart_key="p6_roi_overview_cube3")

		if st.button("Add selected ROI combination to Guide frequencies", key="p6_add_rois_to_guide_cube3"):
			updated_freqs3 = _append_selected_rois_to_freq_list(
				base_freqs=guide_freqs3,
				signal_rois=signal_rois3,
				noise_rois=noise_rois3,
				selected_signal_pos=int(st.session_state.get("p6_signal_roi_select3", 0)) if signal_rois3 else None,
				selected_noise_pos=None,
			)
			st.session_state.p6_guide_freqs_cube3_pending = _freqs_to_text(updated_freqs3)
			st.session_state.p6_guide_cube3_refresh = True
			st.rerun()

		guide_freqs_run3 = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_cube3_input", ""))))
		if guide_freqs_run3:
			st.caption("Target frequencies used for Simulate Single Synthetic Spectrum: " + _freqs_to_text(guide_freqs_run3))
		else:
			st.caption("Target frequencies used for Simulate Single Synthetic Spectrum: (empty)")

		p31, p32, p33, p34 = st.columns(4)
		with p31:
			logn_cube3 = st.number_input("LogN", value=18.248, format="%.4f", key="p6_cube3_logn")
		with p32:
			tex_cube3 = st.number_input("Tex", value=209.06, format="%.4f", key="p6_cube3_tex")
		with p33:
			fwhm_cube3 = st.number_input("FWHM", value=6.198, format="%.4f", key="p6_cube3_fwhm")
		with p34:
			velo_cube3 = st.number_input("Velocity", value=97.549, format="%.4f", key="p6_cube3_velo")

		generate_synth_only = st.button("Generate Synthetic Spectrum", type="primary", key="p6_start_cube3")
		if generate_synth_only:
			if not guide_freqs_run3:
				st.error("Guide frequencies está vacío. Agrega al menos una frecuencia o usa 'Add selected ROI combination to Guide frequencies'.")
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			else:
				with st.spinner("Generating synthetic-only spectra..."):
					res3, warn3 = _generate_synthetic_spectra_for_targets(
						signal_models_source=str(signal_models_root),
						filter_file=str(filter_file),
						target_freqs=[float(v) for v in guide_freqs_run3],
						x_features=[float(logn_cube3), float(tex_cube3), float(velo_cube3), float(fwhm_cube3)],
						pred_mode=DEFAULT_PRED_MODE,
						selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
						allow_nearest=bool(allow_nearest),
					)
				st.session_state.p6_synth_only_results = dict(res3)
				st.session_state.p6_synth_only_warnings = list(warn3)
				if res3:
					st.success("Synthetic-only spectra generated.")
				else:
					st.warning("No synthetic spectra could be generated for the selected targets.")

		uploaded_overlay = st.file_uploader(
			"Upload synthetic spectrum (.txt/.dat/.csv) to overlay",
			type=None,
			key="p6_syn_only_upload_txt",
		)
		up_freq3, up_vals3, up_err3 = _read_uploaded_spectrum_any(uploaded_overlay) if uploaded_overlay is not None else (None, None, None)
		if uploaded_overlay is not None and up_err3 is not None:
			st.warning(f"Uploaded spectrum could not be parsed: {up_err3}")

		results3 = st.session_state.get("p6_synth_only_results", {})
		warns3 = st.session_state.get("p6_synth_only_warnings", [])
		if isinstance(warns3, list) and warns3:
			with st.expander("Show synthetic-only warnings"):
				st.text("\n".join([str(w) for w in warns3]))

		if isinstance(results3, dict) and results3:
			keys3 = sorted(list(results3.keys()), key=lambda s: float(s))

			# Global overview plot (always shown when synthetic results exist)
			all_fmins: List[float] = []
			all_fmaxs: List[float] = []
			all_ymins: List[float] = []
			all_ymaxs: List[float] = []
			fig3_all = go.Figure()
			for k3 in keys3:
				it3 = results3.get(k3, {})
				fg = np.asarray(it3.get("freq", []), dtype=np.float64)
				yg = np.asarray(it3.get("synthetic", []), dtype=np.float64)
				if fg.size == 0 or yg.size == 0:
					continue
				all_fmins.append(float(np.nanmin(fg)))
				all_fmaxs.append(float(np.nanmax(fg)))
				all_ymins.append(float(np.nanmin(yg)))
				all_ymaxs.append(float(np.nanmax(yg)))
				fig3_all.add_trace(
					go.Scatter(
						x=fg,
						y=yg,
						mode="lines",
						name=f"Synthetic target {float(k3):.6f} GHz",
					)
				)

			if (up_freq3 is not None) and (up_vals3 is not None):
				all_fmins.append(float(np.nanmin(up_freq3)))
				all_fmaxs.append(float(np.nanmax(up_freq3)))
				all_ymins.append(float(np.nanmin(up_vals3)))
				all_ymaxs.append(float(np.nanmax(up_vals3)))
				fig3_all.add_trace(
					go.Scatter(
						x=up_freq3,
						y=up_vals3,
						mode="lines",
						name="Uploaded synthetic",
						line=dict(dash="dot", color="#444444"),
					)
				)

			if all_fmins and all_fmaxs:
				gx_min = float(np.nanmin(np.asarray(all_fmins, dtype=np.float64)))
				gx_max = float(np.nanmax(np.asarray(all_fmaxs, dtype=np.float64)))
				gx_span = float(max(1e-6, gx_max - gx_min))
				gx_pad = float(max(5e-5, 0.06 * gx_span))
				fig3_all.update_xaxes(range=[gx_min - gx_pad, gx_max + gx_pad])

			if all_ymins and all_ymaxs:
				gy_min = float(np.nanmin(np.asarray(all_ymins, dtype=np.float64)))
				gy_max = float(np.nanmax(np.asarray(all_ymaxs, dtype=np.float64)))
				gy_span = float(max(1e-8, gy_max - gy_min))
				gy_pad = float(max(1e-6, 0.10 * gy_span))
				fig3_all.update_yaxes(range=[gy_min - gy_pad, gy_max + gy_pad])

			fig3_all.update_layout(
				title="Global synthetic spectrum overview",
				xaxis_title="Frequency (GHz)",
				yaxis_title="Intensity",
				template="plotly_white",
				height=430,
				margin=dict(l=40, r=20, t=50, b=40),
			)
			st.plotly_chart(fig3_all, width="stretch", key="p6_synthonly_global_plot")

			st.markdown("**Synthetic-only spectra by target frequency**")
			n_cols3 = 2 if len(keys3) <= 4 else 3
			cols3 = st.columns(n_cols3)
			for i_k, k3 in enumerate(keys3):
				item3 = results3.get(k3, {})
				f3 = np.asarray(item3.get("freq", []), dtype=np.float64)
				y3 = np.asarray(item3.get("synthetic", []), dtype=np.float64)
				with cols3[i_k % n_cols3]:
					st.caption(f"target {float(item3.get('target_freq_ghz', float(k3))):.6f} GHz")
					if f3.size == 0 or y3.size == 0:
						st.caption("No data")
						continue
					fmin3 = float(np.nanmin(f3))
					fmax3 = float(np.nanmax(f3))
					span3 = float(max(1e-6, fmax3 - fmin3))
					pad3 = float(max(5e-5, 0.08 * span3))
					fig3 = go.Figure()
					fig3.add_trace(go.Scatter(x=f3, y=y3, mode="lines", name="Synthetic"))
					if (up_freq3 is not None) and (up_vals3 is not None):
						fig3.add_trace(go.Scatter(x=up_freq3, y=up_vals3, mode="lines", name="Uploaded synthetic", line=dict(dash="dot")))

					y_min_local = float(np.nanmin(y3))
					y_max_local = float(np.nanmax(y3))
					if (up_freq3 is not None) and (up_vals3 is not None):
						m_up = (np.asarray(up_freq3, dtype=np.float64) >= (fmin3 - pad3)) & (np.asarray(up_freq3, dtype=np.float64) <= (fmax3 + pad3))
						if np.any(m_up):
							y_up = np.asarray(up_vals3, dtype=np.float64)[m_up]
							y_min_local = float(min(y_min_local, float(np.nanmin(y_up))))
							y_max_local = float(max(y_max_local, float(np.nanmax(y_up))))
					y_span_local = float(max(1e-8, y_max_local - y_min_local))
					y_pad_local = float(max(1e-6, 0.12 * y_span_local))

					fig3.update_layout(
						xaxis=dict(range=[fmin3 - pad3, fmax3 + pad3]),
						yaxis=dict(range=[y_min_local - y_pad_local, y_max_local + y_pad_local]),
						xaxis_title="Frequency (GHz)",
						yaxis_title="Intensity",
						template="plotly_white",
						height=380,
						margin=dict(l=40, r=20, t=40, b=40),
					)
					st.plotly_chart(fig3, width="stretch", key=f"p6_synthonly_plot_{k3}")

			with st.expander("Download spectrum (Simulate Single Synthetic Spectrum)"):
				sel_key3 = st.selectbox(
					"Select target frequency to export spectrum",
					options=keys3,
					format_func=lambda k: f"target {float(k):.6f} GHz",
					key="p6_synthonly_download_select",
				)
				sel_item3 = results3.get(str(sel_key3), {})
				f_dl3 = np.asarray(sel_item3.get("freq", []), dtype=np.float64)
				y_dl3 = np.asarray(sel_item3.get("synthetic", []), dtype=np.float64)
				txt_bytes3 = _synthetic_spectrum_to_txt_bytes(f_dl3, y_dl3)
				if txt_bytes3 is None:
					st.error("Could not serialize synthetic spectrum to TXT.")
				else:
					st.download_button(
						"Download spectrum (.txt)",
						data=txt_bytes3,
						file_name=f"synthetic_target_{str(sel_key3).replace('.', 'p')}_spectrum.txt",
						mime="text/plain",
						key="p6_synthonly_download_button",
					)
		else:
			st.caption("No synthetic spectra available yet.")

	with tab_fit:
		st.subheader("Fitting")
		st.caption("Upload an observational spectrum and fit synthetic models per ROI using Guide frequencies.")

		fit_case = st.radio(
			"Fitting mode",
			options=["Case 1: Synthetic only", "Case 2: Synthetic + noise"],
			horizontal=True,
			key="p6_fit_case_mode",
		)
		fit_case_mode = "synthetic_only" if "Case 1" in str(fit_case) else "synthetic_plus_noise"

		if bool(st.session_state.get("p6_guide_fit_refresh", False)):
			st.session_state.p6_guide_freqs_fit_input = str(st.session_state.get("p6_guide_freqs_fit_pending", "")).strip()
			st.session_state.p6_guide_fit_refresh = False
			st.session_state.p6_guide_freqs_fit_pending = ""
		if not str(st.session_state.get("p6_guide_freqs_fit_input", "")).strip():
			last_fit = str(st.session_state.get("p6_guide_freqs_fit_last_nonempty", "")).strip()
			if last_fit:
				st.session_state.p6_guide_freqs_fit_input = last_fit
			else:
				st.session_state.p6_guide_freqs_fit_input = _freqs_to_text([float(v) for v in target_freqs])

		guide_freqs_fit_text = st.text_input(
			"Guide frequencies (GHz; defines ROIs to fit)",
			key="p6_guide_freqs_fit_input",
		)
		if str(guide_freqs_fit_text).strip():
			st.session_state.p6_guide_freqs_fit_last_nonempty = str(guide_freqs_fit_text).strip()
		guide_freqs_fit = _normalize_target_freqs_for_run(parse_freq_list(str(guide_freqs_fit_text)))
		if guide_freqs_fit:
			st.caption("Target frequencies used for fitting: " + _freqs_to_text(guide_freqs_fit))

		# ROI quick explorer for fitting context
		signal_rois_fit = _collect_signal_rois_for_ui(signal_models_root, filter_file)
		noise_rois_fit = _collect_noise_rois_for_ui(noise_models_root)
		signal_rois_fit, noise_rois_fit = _mark_roi_overlaps(signal_rois_fit, noise_rois_fit)
		if signal_rois_fit or noise_rois_fit:
			_plot_roi_overview(
				signal_rois_fit,
				noise_rois_fit,
				guide_freqs_ghz=guide_freqs_fit,
				selected_combo_freqs_ghz=None,
				selected_signal_index=None,
				selected_noise_index=None,
				chart_key="p6_roi_overview_fit",
			)

		up_obs_fit = st.file_uploader(
			"Upload observational spectrum (.txt/.dat/.csv)",
			type=None,
			key="p6_fit_upload_obs",
		)
		current_upload_sig = _uploaded_file_signature(up_obs_fit)
		prev_upload_sig = str(st.session_state.get("p6_fit_upload_signature", ""))
		if prev_upload_sig and (prev_upload_sig != current_upload_sig):
			_clear_fitting_outputs()
			st.info("New observational file detected: previous fitting outputs were cleared.")
		st.session_state.p6_fit_upload_signature = str(current_upload_sig)

		obs_freq_fit, obs_vals_fit, obs_err_fit = _read_uploaded_spectrum_any(up_obs_fit) if up_obs_fit is not None else (None, None, None)
		if up_obs_fit is not None and obs_err_fit is not None:
			st.error(f"Could not parse uploaded observational spectrum: {obs_err_fit}")

		# Optional observational frequency shift (same spirit as 1.MODELS_Ana workflows)
		obs_shift_enabled = st.checkbox("Apply observational frequency shift", value=True, key="p6_fit_shift_enabled")
		obs_shift_mode = st.selectbox(
			"Shift mode",
			options=["per_frequency", "spw_center"],
			index=0,
			key="p6_fit_shift_mode",
		)
		obs_shift_kms = st.number_input(
			"Observational shift (km/s)",
			value=-98.0,
			step=0.1,
			format="%.4f",
			key="p6_fit_shift_kms",
		)

		obs_freq_fit_used = None if obs_freq_fit is None else np.asarray(obs_freq_fit, dtype=np.float64).copy()
		obs_vals_fit_used = None if obs_vals_fit is None else np.asarray(obs_vals_fit, dtype=np.float64).copy()
		if (obs_freq_fit_used is not None) and bool(obs_shift_enabled):
			if str(obs_shift_mode).strip().lower() == "spw_center":
				obs_freq_fit_used = _apply_velocity_shift_by_spw_center(obs_freq_fit_used, float(obs_shift_kms))
			else:
				obs_freq_fit_used = _apply_velocity_shift_to_frequency(obs_freq_fit_used, float(obs_shift_kms))
			st.caption(f"Observational spectrum shifted by {float(obs_shift_kms):+.4f} km/s using mode: {str(obs_shift_mode)}")

		with st.expander("Fitting search ranges and speed settings", expanded=False):
			fit_global_mode_ui = st.selectbox(
				"Global fit strategy",
				options=["Per-ROI aggregate", "Concatenated ROIs (single objective)"],
				index=0,
				key="p6_fit_global_mode",
			)
			fit_global_mode_map = {
				"Per-ROI aggregate": "per_roi",
				"Concatenated ROIs (single objective)": "concatenated",
			}
			fit_criterion_ui = st.selectbox(
				"Fitting criterion",
				options=["MAE", "RMSE", "CHI_like", "R2"],
				index=2,
				key="p6_fit_criterion",
			)
			fit_candidate_mode_ui = st.selectbox(
				"Candidate generation",
				options=["Smart ordered grid", "Random"],
				index=1,
				key="p6_fit_candidate_mode",
			)
			fit_candidate_mode_map = {
				"Smart ordered grid": "ordered_grid",
				"Random": "random",
			}
			curr_fit_mode = str(fit_candidate_mode_ui)
			prev_fit_mode = str(st.session_state.get("p6_fit_candidate_mode_prev", ""))
			if prev_fit_mode and (prev_fit_mode != curr_fit_mode):
				_clear_fitting_outputs()
				st.info("Candidate generation mode changed: previous fitting outputs were cleared.")
			st.session_state.p6_fit_candidate_mode_prev = str(curr_fit_mode)
			fit_weight_mode_ui = st.selectbox(
				"Global aggregation weighting",
				options=[
					"Uniform (all ROIs equal)",
					"By overlap points per ROI",
					"By ROI fit quality (criterion-aware)",
				],
				index=2,
				key="p6_fit_weight_mode",
			)
			fit_weight_mode_map = {
				"Uniform (all ROIs equal)": "uniform",
				"By overlap points per ROI": "overlap_points",
				"By ROI fit quality (inverse best error)": "inverse_best_error",
				"By ROI fit quality (criterion-aware)": "inverse_best_error",
			}
			cfr1, cfr2, cfr3, cfr4 = st.columns(4)
			with cfr1:
				fit_logn_min = st.number_input("logN min", value=14.0, key="p6_fit_logn_min")
				fit_logn_max = st.number_input("logN max", value=19.5, key="p6_fit_logn_max")
			with cfr2:
				fit_tex_min = st.number_input("Tex min", value=100.0, key="p6_fit_tex_min")
				fit_tex_max = st.number_input("Tex max", value=380.0, key="p6_fit_tex_max")
			with cfr3:
				fit_velo_min = st.number_input("Velocity min", value=90.0, key="p6_fit_velo_min")
				fit_velo_max = st.number_input("Velocity max", value=105.0, key="p6_fit_velo_max")
			with cfr4:
				fit_fwhm_min = st.number_input("FWHM min", value=5.0, key="p6_fit_fwhm_min")
				fit_fwhm_max = st.number_input("FWHM max", value=8.0, key="p6_fit_fwhm_max")

			cfsp1, cfsp2 = st.columns(2)
			with cfsp1:
				n_candidates_fit = st.number_input("Number of candidates", min_value=50, max_value=4000, value=600, step=50, key="p6_fit_n_candidates")
			with cfsp2:
				seed_fit = st.number_input("Random seed", min_value=0, value=42, step=1, key="p6_fit_seed")

		run_fit = st.button("Run fitting", type="primary", key="p6_run_fitting_btn")
		if run_fit:
			# Always clear previous fitting output/state before starting a new run
			_clear_fitting_outputs()
			if up_obs_fit is None or obs_freq_fit_used is None or obs_vals_fit_used is None:
				st.error("Upload a valid observational spectrum first.")
			elif not guide_freqs_fit:
				st.error("Guide frequencies is empty. Add at least one frequency.")
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			else:
				ranges_fit = {
					"logn_min": float(min(fit_logn_min, fit_logn_max)),
					"logn_max": float(max(fit_logn_min, fit_logn_max)),
					"tex_min": float(min(fit_tex_min, fit_tex_max)),
					"tex_max": float(max(fit_tex_min, fit_tex_max)),
					"velo_min": float(min(fit_velo_min, fit_velo_max)),
					"velo_max": float(max(fit_velo_min, fit_velo_max)),
					"fwhm_min": float(min(fit_fwhm_min, fit_fwhm_max)),
					"fwhm_max": float(max(fit_fwhm_min, fit_fwhm_max)),
				}
				with st.spinner("Running efficient batch fitting per ROI..."):
					fit_result = _run_roi_fitting(
						signal_models_source=str(signal_models_root),
						noise_models_root=str(noise_models_root),
						filter_file=str(filter_file),
						target_freqs=[float(v) for v in guide_freqs_fit],
						obs_freq=np.asarray(obs_freq_fit_used, dtype=np.float64),
						obs_intensity=np.asarray(obs_vals_fit_used, dtype=np.float64),
						case_mode=str(fit_case_mode),
						fit_criterion=str(fit_criterion_ui).strip().lower(),
						global_weight_mode=str(fit_weight_mode_map.get(str(fit_weight_mode_ui), "uniform")),
						global_search_mode=str(fit_global_mode_map.get(str(fit_global_mode_ui), "per_roi")),
						candidate_mode=str(fit_candidate_mode_map.get(str(fit_candidate_mode_ui), "random")),
						n_candidates=int(n_candidates_fit),
						ranges=ranges_fit,
						noise_scale=float(noise_scale),
						allow_nearest=bool(allow_nearest),
						seed=int(seed_fit),
					)
				st.session_state.p6_fit_last_result = fit_result

		fit_result = st.session_state.get("p6_fit_last_result", None)
		if isinstance(fit_result, dict):
			if not bool(fit_result.get("ok", False)):
				st.warning(str(fit_result.get("message", "No fitting result available.")))
			else:
				bp = fit_result.get("best_global_params", {}) if isinstance(fit_result.get("best_global_params", {}), dict) else {}
				fit_crit_used = str(fit_result.get("fit_criterion", "mae"))
				fit_obj = float(fit_result.get("best_global_mean_objective", np.nan))
				fit_obj_show = (-fit_obj if fit_crit_used == "r2" else fit_obj)
				st.success(
					"Best global fit | "
					f"logN={float(bp.get('logN', np.nan)):.4f}, "
					f"Tex={float(bp.get('Tex', np.nan)):.4f}, "
					f"Velocity={float(bp.get('Velocity', np.nan)):.4f}, "
					f"FWHM={float(bp.get('FWHM', np.nan)):.4f}, "
					f"mean {fit_crit_used.upper()}={float(fit_obj_show):.6g} | "
					f"mean MAE={float(fit_result.get('best_global_mean_MAE', np.nan)):.6g}"
				)
				st.caption(
					f"Mode: {fit_result.get('case_mode', '')} | "
					f"Global strategy: {fit_result.get('global_search_mode', 'per_roi')} | "
					f"Candidates: {int(fit_result.get('n_candidates', 0))} | "
					f"Guide freqs input: {int(fit_result.get('n_guide_freqs_input', 0))} | "
					f"Unique ROIs from guide freqs: {int(fit_result.get('n_unique_rois_requested', 0))} | "
					f"ROIs fitted: {int(fit_result.get('n_rois_fitted', 0))} | "
					f"Sampling: {fit_result.get('candidate_mode', 'ordered_grid')} | "
					f"Weighting: {fit_result.get('global_weight_mode', 'uniform')}"
				)

				global_overlay = fit_result.get("global_overlay", [])
				if isinstance(global_overlay, list) and global_overlay:
					segments = []
					for gg in global_overlay:
						fg = np.asarray(gg.get("freq", []), dtype=np.float64)
						yg_obs = np.asarray(gg.get("obs_interp", []), dtype=np.float64)
						yg_pred = np.asarray(gg.get("best_global_pred", []), dtype=np.float64)
						if fg.size == 0 or yg_pred.size != fg.size:
							continue
						segments.append((float(np.nanmin(fg)), fg, yg_obs, yg_pred, gg))

					if segments:
						segments = sorted(segments, key=lambda t: t[0])
						f_cat = []
						o_cat = []
						p_cat = []
						for i_s, (_, fg, og, pg, _) in enumerate(segments):
							if i_s > 0:
								f_cat.append(np.array([np.nan], dtype=np.float64))
								o_cat.append(np.array([np.nan], dtype=np.float64))
								p_cat.append(np.array([np.nan], dtype=np.float64))
							f_cat.append(fg)
							o_cat.append(og)
							p_cat.append(pg)

						fig_global = go.Figure()
						# Full uploaded observational spectrum (if available)
						if (obs_freq_fit_used is not None) and (obs_vals_fit_used is not None):
							fig_global.add_trace(go.Scatter(
								x=np.asarray(obs_freq_fit_used, dtype=np.float64),
								y=np.asarray(obs_vals_fit_used, dtype=np.float64),
								mode="lines",
								name="Observed (uploaded)",
								line=dict(width=1.4, color="green"),
							))
						fig_global.add_trace(go.Scatter(
							x=np.concatenate(f_cat),
							y=np.concatenate(o_cat),
							mode="lines",
							name="Observed (ROI interp)",
							line=dict(dash="dot", width=1.2, color="#2ca02c"),
						))
						fig_global.add_trace(go.Scatter(
							x=np.concatenate(f_cat),
							y=np.concatenate(p_cat),
							mode="lines",
							name="Best fit (global)",
							line=dict(width=2.0),
						))
						fig_global.update_layout(
							title="Observed spectrum vs best global fit",
							xaxis_title="Frequency (GHz)",
							yaxis_title="Intensity",
							template="plotly_white",
							height=430,
							margin=dict(l=40, r=20, t=45, b=40),
						)
						st.plotly_chart(fig_global, width="stretch", key="p6_fit_global_overlay_plot")

				rows_fit = fit_result.get("per_roi", [])
				if isinstance(rows_fit, list) and rows_fit:
					st.markdown("**Per-ROI fitting statistics**")
					st.dataframe(rows_fit, use_container_width=True)

				plots_fit = fit_result.get("plot_payload", [])
				if isinstance(plots_fit, list) and plots_fit:
					st.markdown("**Best-match spectrum per ROI**")
					n_cols_fit = 2 if len(plots_fit) <= 4 else 3
					cols_fit = st.columns(n_cols_fit)
					for i_pf, pf in enumerate(plots_fit):
						fpf = np.asarray(pf.get("freq", []), dtype=np.float64)
						yobs = np.asarray(pf.get("obs_interp", []), dtype=np.float64)
						ys = np.asarray(pf.get("best_synthetic", []), dtype=np.float64)
						yn = pf.get("best_noise", None)
						yp = np.asarray(pf.get("best_pred", []), dtype=np.float64)
						with cols_fit[i_pf % n_cols_fit]:
							gf_list = [float(v) for v in pf.get("guide_freqs_ghz", [])]
							gf_label = str(pf.get("guide_freqs_label", "")).strip()
							if (not gf_label) and gf_list:
								gf_label = _format_freqs_short(gf_list)
							n_gf = int(pf.get("n_guide_freqs_in_roi", max(1, len(gf_list))))
							roi_lo = pf.get("roi_f_min_ghz", np.nan)
							roi_hi = pf.get("roi_f_max_ghz", np.nan)
							if np.isfinite(float(roi_lo)) and np.isfinite(float(roi_hi)):
								st.caption(
									f"ROI [{float(roi_lo):.6f}, {float(roi_hi):.6f}] GHz | "
									f"Guide freqs in ROI ({int(n_gf)}): {gf_label if gf_label else f'{float(pf.get('target_freq_ghz', np.nan)):.6f}'} GHz"
								)
							else:
								st.caption(
									f"Guide freqs in ROI ({int(n_gf)}): {gf_label if gf_label else f'{float(pf.get('target_freq_ghz', np.nan)):.6f}'} GHz"
								)
							fig_fit = go.Figure()
							fig_fit.add_trace(go.Scatter(x=fpf, y=yobs, mode="lines", name="Observed (interp)", line=dict(color="green")))
							fig_fit.add_trace(go.Scatter(x=fpf, y=ys, mode="lines", name="Best synthetic", line=dict(dash="dash")))
							if yn is not None:
								fig_fit.add_trace(go.Scatter(x=fpf, y=np.asarray(yn, dtype=np.float64), mode="lines", name="Best noise", line=dict(dash="dot")))
							fig_fit.add_trace(go.Scatter(x=fpf, y=yp, mode="lines", name="Best predicted"))
							fig_fit.update_layout(
								xaxis_title="Frequency (GHz)",
								yaxis_title="Intensity",
								template="plotly_white",
								height=360,
								margin=dict(l=40, r=20, t=35, b=35),
							)
							st.plotly_chart(fig_fit, width="stretch", key=f"p6_fit_plot_{i_pf}")

			warns_fit = fit_result.get("warnings", [])
			rows_fit_for_map = fit_result.get("per_roi", [])
			if isinstance(rows_fit_for_map, list) and rows_fit_for_map:
				with st.expander("Guide frequencies mapped to fitted ROIs"):
					for rr in rows_fit_for_map:
						gfl = str(rr.get("guide_freqs_label", "")).strip()
						if not gfl:
							gfl = f"{float(rr.get('target_freq_ghz', np.nan)):.6f}"
						lo = rr.get("roi_f_min_ghz", np.nan)
						hi = rr.get("roi_f_max_ghz", np.nan)
						if np.isfinite(float(lo)) and np.isfinite(float(hi)):
							st.write(
								f"ROI [{float(lo):.6f}, {float(hi):.6f}] GHz <- Guide freqs: {gfl} GHz"
							)
						else:
							st.write(f"Guide freqs: {gfl} GHz")
			if isinstance(warns_fit, list) and warns_fit:
				with st.expander("Show fitting warnings"):
					st.text("\n".join([str(w) for w in warns_fit]))

	with tab_cube_fit:
		st.subheader("Cube Fitting")
		st.caption("Same fitting parameterization as 'Fitting', but applied pixel-by-pixel to an uploaded observational cube to produce LogN/Tex/Velocity/FWHM maps.")

		guide_freqs_cfit_text = st.text_input(
			"Guide frequencies (GHz; defines ROIs to fit in every pixel)",
			value=_freqs_to_text([float(v) for v in target_freqs]),
			key="p6_guide_freqs_cfit_input",
		)
		guide_freqs_cfit = _normalize_target_freqs_for_run(parse_freq_list(str(guide_freqs_cfit_text)))
		if guide_freqs_cfit:
			st.caption("Target frequencies used for cube fitting: " + _freqs_to_text(guide_freqs_cfit))

		up_obs_cube_fit = st.file_uploader(
			"Upload observational cube (.fits)",
			type=["fits"],
			key="p6_cubefit_upload_cube",
		)
		obs_cube_fit_path = _save_uploaded_file_to_temp(up_obs_cube_fit, "cubefit_obs_cube") if up_obs_cube_fit is not None else None

		cubefit_out_dir = st.text_input("Output directory", value=os.path.join(DEFAULT_OUTPUT_DIR, "cube_fit"), key="p6_cubefit_out_dir")
		cubefit_progress_every = st.number_input("Progress every N pixels", min_value=1, value=40, step=1, key="p6_cubefit_progress_every")
		cubefit_spatial_stride = st.number_input("Spatial stride (1=all pixels, 2=every 2 pixels)", min_value=1, value=1, step=1, key="p6_cubefit_spatial_stride")

		cubefit_case = st.radio(
			"Fitting mode",
			options=["Case 1: Synthetic only", "Case 2: Synthetic + noise"],
			horizontal=True,
			key="p6_cubefit_case_mode",
		)
		cubefit_case_mode = "synthetic_only" if "Case 1" in str(cubefit_case) else "synthetic_plus_noise"

		cubefit_shift_enabled = st.checkbox("Apply observational frequency shift", value=True, key="p6_cubefit_shift_enabled")
		cubefit_shift_mode = st.selectbox(
			"Shift mode",
			options=["per_frequency", "spw_center"],
			index=0,
			key="p6_cubefit_shift_mode",
		)
		cubefit_shift_kms = st.number_input(
			"Observational shift (km/s)",
			value=-98.0,
			step=0.1,
			format="%.4f",
			key="p6_cubefit_shift_kms",
		)

		with st.expander("Fitting search ranges and speed settings", expanded=False):
			cubefit_global_mode_ui = st.selectbox(
				"Global fit strategy",
				options=["Per-ROI aggregate", "Concatenated ROIs (single objective)"],
				index=0,
				key="p6_cubefit_global_mode",
			)
			cubefit_global_mode_map = {
				"Per-ROI aggregate": "per_roi",
				"Concatenated ROIs (single objective)": "concatenated",
			}
			cubefit_criterion_ui = st.selectbox(
				"Fitting criterion",
				options=["MAE", "RMSE", "CHI_like", "R2"],
				index=2,
				key="p6_cubefit_criterion",
			)
			cubefit_candidate_mode_ui = st.selectbox(
				"Candidate generation",
				options=["Smart ordered grid", "Random"],
				index=1,
				key="p6_cubefit_candidate_mode",
			)
			cubefit_candidate_mode_map = {
				"Smart ordered grid": "ordered_grid",
				"Random": "random",
			}
			cubefit_weight_mode_ui = st.selectbox(
				"Global aggregation weighting",
				options=[
					"Uniform (all ROIs equal)",
					"By overlap points per ROI",
					"By ROI fit quality (criterion-aware)",
				],
				index=2,
				key="p6_cubefit_weight_mode",
			)
			cubefit_weight_mode_map = {
				"Uniform (all ROIs equal)": "uniform",
				"By overlap points per ROI": "overlap_points",
				"By ROI fit quality (criterion-aware)": "inverse_best_error",
			}
			cc1, cc2, cc3, cc4 = st.columns(4)
			with cc1:
				cubefit_logn_min = st.number_input("logN min", value=14.0, key="p6_cubefit_logn_min")
				cubefit_logn_max = st.number_input("logN max", value=19.5, key="p6_cubefit_logn_max")
			with cc2:
				cubefit_tex_min = st.number_input("Tex min", value=100.0, key="p6_cubefit_tex_min")
				cubefit_tex_max = st.number_input("Tex max", value=380.0, key="p6_cubefit_tex_max")
			with cc3:
				cubefit_velo_min = st.number_input("Velocity min", value=90.0, key="p6_cubefit_velo_min")
				cubefit_velo_max = st.number_input("Velocity max", value=105.0, key="p6_cubefit_velo_max")
			with cc4:
				cubefit_fwhm_min = st.number_input("FWHM min", value=5.0, key="p6_cubefit_fwhm_min")
				cubefit_fwhm_max = st.number_input("FWHM max", value=8.0, key="p6_cubefit_fwhm_max")

			ccs1, ccs2 = st.columns(2)
			with ccs1:
				cubefit_n_candidates = st.number_input("Number of candidates", min_value=50, max_value=4000, value=600, step=50, key="p6_cubefit_n_candidates")
			with ccs2:
				cubefit_seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="p6_cubefit_seed")

		cbf1, cbf2 = st.columns(2)
		with cbf1:
			run_cubefit = st.button("Run cube fitting", type="primary", key="p6_run_cubefit_btn", disabled=_is_cubefit_running())
		with cbf2:
			stop_cubefit = st.button("Stop cube fitting", key="p6_stop_cubefit_btn", disabled=not _is_cubefit_running())

		if run_cubefit:
			if obs_cube_fit_path is None or (not os.path.isfile(str(obs_cube_fit_path))):
				st.error("Upload a valid observational cube first.")
			elif not guide_freqs_cfit:
				st.error("Guide frequencies is empty. Add at least one frequency.")
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			elif (str(cubefit_case_mode).strip().lower() == "synthetic_plus_noise") and (not _is_valid_noise_source(noise_models_root)):
				st.error("Noise models source invalid for Case 2.")
			else:
				try:
					os.makedirs(cubefit_out_dir, exist_ok=True)
					_cleanup_cubefit_outputs_for_dir(str(cubefit_out_dir))
					ranges_cubefit = {
						"logn_min": float(min(cubefit_logn_min, cubefit_logn_max)),
						"logn_max": float(max(cubefit_logn_min, cubefit_logn_max)),
						"tex_min": float(min(cubefit_tex_min, cubefit_tex_max)),
						"tex_max": float(max(cubefit_tex_min, cubefit_tex_max)),
						"velo_min": float(min(cubefit_velo_min, cubefit_velo_max)),
						"velo_max": float(max(cubefit_velo_min, cubefit_velo_max)),
						"fwhm_min": float(min(cubefit_fwhm_min, cubefit_fwhm_max)),
						"fwhm_max": float(max(cubefit_fwhm_min, cubefit_fwhm_max)),
					}
					cfg_cfit = {
						"out_dir": str(cubefit_out_dir),
						"obs_cube_path": str(obs_cube_fit_path),
						"signal_models_source": str(signal_models_root),
						"noise_models_root": str(noise_models_root),
						"filter_file": str(filter_file),
						"target_freqs": [float(v) for v in guide_freqs_cfit],
						"case_mode": str(cubefit_case_mode),
						"fit_criterion": str(cubefit_criterion_ui).strip().lower(),
						"global_weight_mode": str(cubefit_weight_mode_map.get(str(cubefit_weight_mode_ui), "uniform")),
						"global_search_mode": str(cubefit_global_mode_map.get(str(cubefit_global_mode_ui), "per_roi")),
						"candidate_mode": str(cubefit_candidate_mode_map.get(str(cubefit_candidate_mode_ui), "random")),
						"n_candidates": int(cubefit_n_candidates),
						"ranges": ranges_cubefit,
						"noise_scale": float(noise_scale),
						"allow_nearest": bool(allow_nearest),
						"seed": int(cubefit_seed),
						"progress_every": int(cubefit_progress_every),
						"spatial_stride": int(cubefit_spatial_stride),
						"obs_shift_enabled": bool(cubefit_shift_enabled),
						"obs_shift_mode": str(cubefit_shift_mode),
						"obs_shift_kms": float(cubefit_shift_kms),
						"out_prefix": "CUBEFIT",
					}
					fdc, cfg_cfit_path = tempfile.mkstemp(prefix="predobs6_cubefit_cfg_", suffix=".json", dir=tempfile.gettempdir())
					os.close(fdc)
					with open(cfg_cfit_path, "w", encoding="utf-8") as f:
						json.dump(cfg_cfit, f, ensure_ascii=False, indent=2)
					log_cfit_path = os.path.join(cubefit_out_dir, f"cubefit_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
					log_cfit_fh = open(log_cfit_path, "a", encoding="utf-8", buffering=1)
					proc_cfit = subprocess.Popen(
						[sys.executable, str(Path(__file__).resolve()), "--cube-fit-worker", cfg_cfit_path],
						cwd=str(_project_dir()),
						stdout=log_cfit_fh,
						stderr=subprocess.STDOUT,
						text=True,
					)
					st.session_state.cubefit_proc = proc_cfit
					st.session_state.cubefit_log_path = log_cfit_path
					st.session_state.cubefit_cfg_path = cfg_cfit_path
					st.session_state.cubefit_log_handle = log_cfit_fh
					st.success("Cube fitting started.")
				except Exception as e:
					st.error(f"Could not start cube fitting: {e}")

		if stop_cubefit:
			_stop_cubefit_process()
			st.warning("Cube fitting stopped by user.")

		if _is_cubefit_running():
			st.info("Cube fitting status: running")
		else:
			proc_cf = st.session_state.get("cubefit_proc", None)
			if proc_cf is not None:
				code_cf = proc_cf.poll()
				if code_cf == 0:
					st.success("Cube fitting status: finished successfully")
				elif code_cf is not None:
					st.error(f"Cube fitting status: finished with code {code_cf}")
					log_tail_cf = _read_log_tail(str(st.session_state.get("cubefit_log_path", "")), n_lines=120)
					if log_tail_cf:
						with st.expander("Show last cube fitting log lines"):
							st.text(log_tail_cf)
				_stop_cubefit_process()
			else:
				st.caption("Cube fitting status: idle")

		progress_png_cf = _find_latest_progress_png(str(cubefit_out_dir))
		if progress_png_cf:
			st.markdown("**Cube fitting progress**")
			progress_info_cf = _read_progress_info(progress_png_cf)
			if isinstance(progress_info_cf, dict):
				done_steps = int(progress_info_cf.get("done_steps", 0))
				total_steps = int(max(1, progress_info_cf.get("total_steps", 1)))
				pct = 100.0 * float(done_steps) / float(total_steps)
				st.success(f"**Pixels processed:** {done_steps}/{total_steps} ({pct:.1f}%)")
			img_bytes_cf = _read_progress_png_stable_bytes(progress_png_cf)
			if img_bytes_cf is not None:
				st.image(img_bytes_cf, caption=os.path.basename(progress_png_cf), width=520)

		progress_map_files = {
			"logN": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_LOGN.fits"),
			"Tex": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_TEX.fits"),
			"Velocity": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_VELOCITY.fits"),
			"FWHM": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_FWHM.fits"),
		}
		progress_maps_available = {k: v for k, v in progress_map_files.items() if os.path.isfile(v)}
		if progress_maps_available:
			st.markdown("**Live in-progress parameter maps (one plot per parameter)**")
			pm1, pm2 = st.columns(2)
			pm_cols = [pm1, pm2]
			for i_pm, (mk, mp) in enumerate(progress_maps_available.items()):
				with pm_cols[i_pm % 2]:
					try:
						arr_pm = np.asarray(fits.getdata(mp), dtype=np.float32)
						if arr_pm.ndim == 3:
							arr_pm = arr_pm[0]
						_show_fits_preview(f"{mk} (in progress)", arr_pm)
					except Exception:
						st.caption(f"Could not render in-progress map for {mk}")

		if not _is_cubefit_running():
			map_files = {
				"logN": os.path.join(str(cubefit_out_dir), "CUBEFIT_LOGN.fits"),
				"Tex": os.path.join(str(cubefit_out_dir), "CUBEFIT_TEX.fits"),
				"Velocity": os.path.join(str(cubefit_out_dir), "CUBEFIT_VELOCITY.fits"),
				"FWHM": os.path.join(str(cubefit_out_dir), "CUBEFIT_FWHM.fits"),
			}
			available_maps = {k: v for k, v in map_files.items() if os.path.isfile(v)}
			if available_maps:
				st.markdown("**Cube fitting parameter maps (final)**")
				mc1, mc2 = st.columns(2)
				cols_map = [mc1, mc2]
				for i_m, (mk, mp) in enumerate(available_maps.items()):
					with cols_map[i_m % 2]:
						try:
							arr_m = np.asarray(fits.getdata(mp), dtype=np.float32)
							if arr_m.ndim == 3:
								arr_m = arr_m[0]
							_show_fits_preview(mk, arr_m)
						except Exception:
							st.caption(f"Could not render preview for {mk}")
						try:
							with open(mp, "rb") as f_mp:
								st.download_button(
									f"Download {mk} map (.fits)",
									data=f_mp.read(),
									file_name=os.path.basename(mp),
									mime="application/fits",
									key=f"p6_cubefit_download_{mk}",
								)
						except Exception:
							pass

		if _is_cubefit_running():
			st.caption("Auto-updating every 5 seconds...")
			time.sleep(5)
			st.rerun()


def _worker_entry_if_needed() -> bool:
	if "--cube-fit-worker" in sys.argv:
		idx = sys.argv.index("--cube-fit-worker")
		if idx + 1 >= len(sys.argv):
			print("Missing cube-fit config path")
			sys.exit(2)
		cfg_path = sys.argv[idx + 1]
		code = run_cube_fit_worker(cfg_path)
		sys.exit(int(code))
	if "--cube-worker" not in sys.argv:
		if "--sim-worker" not in sys.argv:
			return False
		idx = sys.argv.index("--sim-worker")
		if idx + 1 >= len(sys.argv):
			print("Missing sim config path")
			sys.exit(2)
		cfg_path = sys.argv[idx + 1]
		code = run_sim_worker(cfg_path)
		sys.exit(int(code))
	idx = sys.argv.index("--cube-worker")
	if idx + 1 >= len(sys.argv):
		print("Missing config path")
		sys.exit(2)
	cfg_path = sys.argv[idx + 1]
	code = run_cube_worker(cfg_path)
	sys.exit(int(code))


if __name__ == "__main__":
	if not _worker_entry_if_needed():
		run_streamlit_app()
