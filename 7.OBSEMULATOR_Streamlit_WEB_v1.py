import os
import re
import sys
import io
import contextlib
import json
import csv
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

# ======================================================
# Runtime resource policy (CPU-only, efficient local use)
# ======================================================
_CPU_COUNT = int(os.cpu_count() or 4)
_DEFAULT_CPU_THREADS = int(max(1, _CPU_COUNT - 1))

# Respect user-provided env vars; otherwise set sensible defaults.
os.environ.setdefault("OBSEMULATOR_CPU_THREADS", str(_DEFAULT_CPU_THREADS))
os.environ.setdefault("OMP_NUM_THREADS", os.environ.get("OBSEMULATOR_CPU_THREADS", str(_DEFAULT_CPU_THREADS)))
os.environ.setdefault("MKL_NUM_THREADS", os.environ.get("OBSEMULATOR_CPU_THREADS", str(_DEFAULT_CPU_THREADS)))
os.environ.setdefault("OPENBLAS_NUM_THREADS", os.environ.get("OBSEMULATOR_CPU_THREADS", str(_DEFAULT_CPU_THREADS)))
os.environ.setdefault("NUMEXPR_NUM_THREADS", os.environ.get("OBSEMULATOR_CPU_THREADS", str(_DEFAULT_CPU_THREADS)))

# Force CPU-only execution (no GPU).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

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
	from scipy.optimize import least_squares
except Exception:
	least_squares = None

try:
	import xgboost as xgb
except Exception:
	xgb = None

try:
	from PIL import Image, UnidentifiedImageError
except Exception:
	Image = None
	UnidentifiedImageError = Exception

try:
	from astropy.io import fits
except Exception:
	fits = None

try:
	from astropy.wcs import WCS
except Exception:
	WCS = None


# ======================================================
# DEFAULT CONFIG (single-file, no 4.SYNGEN dependency)
# ======================================================
DEFAULT_MERGED_H5 = r"D:\4.DATASETS\MODELS_CH3OCHO_ENSEMBLES_PERCHANNEL_ROI_dynamicrange_v2_PER_ROI"
DEFAULT_NOISE_NN_H5 = ""
DEFAULT_FILTER_FILE = ""
DEFAULT_GDRIVE_MODELS_LINK = "https://drive.google.com/drive/folders/1eV0TVZbhOe9ouW76cNgerZ89R4KNTBTw?usp=drive_link"

# Local preset (Windows)

# DEFAULT_LOCAL_SIGNAL_H5 = r"D:\4.DATASETS\CH3OCHO_MODELS_COMPRESSED_v2_PER_ROI.h5"
# DEFAULT_LOCAL_NOISE_H5 = r"D:\4.DATASETS\NOISE_MODELS_CH3OCHO_NN_GUAPOS_v12plotstyle_roi_bundle.h5"
# DEFAULT_LOCAL_FILTER_FILE = r"D:\4.DATASETS\filter_reference_CH3OCHO_100spectra.txt"
# DEFAULT_LOCAL_ROI_RANK_MODEL_DIR = r"D:\4.DATASETS\RANKING_MODELS\ROI_RANKING_MODELS_CH3OCHO_v3\roi_rank_model_bundle.h5"
# DEFAULT_LOCAL_INVERSE_PARAM_MODELS_DIR = r"D:\4.DATASETS\INVERSE_MODELS_FROM_SYNTH_ROI_CH3OCHO_v3"


# DEFAULT_LOCAL_SIGNAL_H5 = r"D:\4.DATASETS\MODELS_C2H5OH_ENSEMBLES_PERCHANNEL_ROI_dynamicrange_v2_PER_ROI.h5"
# DEFAULT_LOCAL_NOISE_H5 = r"D:\4.DATASETS\RANKING_MODELS\NOISE_MODELS_C2H5OH_NN_GUAPOS_v12plotstyle_roi.h5"
# DEFAULT_LOCAL_FILTER_FILE = r"D:\4.DATASETS\filter_reference_C2H5OH_100spectra.txt"
# DEFAULT_LOCAL_ROI_RANK_MODEL_DIR = r"D:\4.DATASETS\RANKING_MODELS\ROI_RANKING_MODELS_C2H5OH_v3\RANKING.h5"

DEFAULT_LOCAL_SIGNAL_H5 = r"D:\4.DATASETS\MODELS_C2H5OH_ENSEMBLES_PERCHANNEL_ROI_dynamicrange_v3_PER_ROI_TEST_v1.h5"
DEFAULT_LOCAL_NOISE_H5 = r"D:\4.DATASETS\NOISE_MODELS_C2H5OH_NN_GUAPOS_v12plotstyle_roi_TEST1.h5"
DEFAULT_LOCAL_FILTER_FILE = r"D:\4.DATASETS\filter_reference_C2H5OH_100spectra.txt"
DEFAULT_LOCAL_ROI_RANK_MODEL_DIR = r"D:\4.DATASETS\RANKING_MODELS\ROI_RANKING_MODELS_CH3OCHO_v3\roi_rank_model_bundle.h5"
DEFAULT_LOCAL_INVERSE_PARAM_MODELS_DIR = r"D:\4.DATASETS\INVERSE_MODELS_FROM_SYNTH_ROI_CH3OCHO_v3"
DEFAULT_LOCAL_INVERSE_CUBE_MODELS_DIR = r"D:\4.DATASETS\MODELS_C2H5OH_NN_FROM_SYNTHDB_CUSTOMROI_TARGETFREQ_v1"


DEFAULT_TARGET_FREQS = [
	84.299,
	110.855,
]

DEFAULT_CUBEFIT_GUIDE_FREQS = [
	108.4385616,
	106.9312289,
]

DEFAULT_CUBEFIT_OUTDIR = os.path.join(tempfile.gettempdir(), "predobs_outputs", "cube_C2H5OH_v3")
DEFAULT_INVERSE_CUBEPRED_OUTDIR = os.path.join(tempfile.gettempdir(), "predobs_outputs", "inverse_cube_C2H5OH_v1")
DEFAULT_OBS_CUBE_PATH = r"D:\4.DATASETS\3.W51\MAD_CUB_MOD_member.uid___A001_X879_X36f.W51_sci.spw29.cube.I.pbcor_kelvins.fits"

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
DEFAULT_SELECTED_MODEL_NAME = "XGBoost"

GUIDE_FREQS_EMPTY_ERROR = (
	"Guide frequencies is empty. Add at least one frequency or use "
	"'Add selected ROI combination to Guide frequencies'."
)


def _get_int_env(name: str, fallback: int) -> int:
	try:
		v = int(str(os.environ.get(name, str(fallback))).strip())
		return int(max(1, v))
	except Exception:
		return int(max(1, fallback))


def _configure_runtime_resources_cpu_only() -> dict:
	threads = _get_int_env("OBSEMULATOR_CPU_THREADS", _DEFAULT_CPU_THREADS)
	interop = int(max(1, min(8, threads // 2)))

	# Re-apply explicitly to current process as well.
	os.environ["OMP_NUM_THREADS"] = str(threads)
	os.environ["MKL_NUM_THREADS"] = str(threads)
	os.environ["OPENBLAS_NUM_THREADS"] = str(threads)
	os.environ["NUMEXPR_NUM_THREADS"] = str(threads)
	os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

	try:
		torch.set_num_threads(int(threads))
	except Exception:
		pass
	try:
		if hasattr(torch, "set_num_interop_threads"):
			torch.set_num_interop_threads(int(interop))
	except Exception:
		pass

	return {
		"cpu_count": int(_CPU_COUNT),
		"cpu_threads": int(threads),
		"interop_threads": int(interop),
		"cuda_visible_devices": str(os.environ.get("CUDA_VISIBLE_DEVICES", "")),
	}


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
		"roi_rank_model_dir": "",
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
	pt_files = [p for p in all_files if str(p).lower().endswith(".pt")]
	npz_files = [p for p in all_files if str(p).lower().endswith(".npz")]
	json_files = [p for p in all_files if str(p).lower().endswith(".json")]

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
			if len(glob.glob(os.path.join(p, "ROI_*", "*", "model", "final_model.joblib"))) > 0:
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

	# ROI ranking model dir auto-detection (expects 1.5 artifacts in same directory).
	rank_dirs = set()
	rank_h5_candidates: List[str] = []
	for p in h5_files:
		if _h5_has_groups_or_datasets(p, ["state_dict_blob", "scalers", "meta_json"]):
			rank_h5_candidates.append(str(p))
	for p in pt_files:
		if os.path.basename(str(p)).lower() == "roi_rank_nn.pt":
			rank_dirs.add(os.path.dirname(p))
	for p in npz_files:
		if os.path.basename(str(p)).lower() == "roi_rank_scalers.npz":
			rank_dirs.add(os.path.dirname(p))
	for p in json_files:
		if os.path.basename(str(p)).lower() == "roi_rank_training_meta.json":
			rank_dirs.add(os.path.dirname(p))

	rank_candidates: List[str] = []
	for d in sorted(rank_dirs):
		if (
			os.path.isfile(os.path.join(d, "roi_rank_nn.pt"))
			and os.path.isfile(os.path.join(d, "roi_rank_scalers.npz"))
			and os.path.isfile(os.path.join(d, "roi_rank_training_meta.json"))
		):
			rank_candidates.append(str(d))

	if len(rank_candidates) == 1:
		result["roi_rank_model_dir"] = rank_candidates[0]
	elif len(rank_candidates) > 1:
		rank_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
		result["roi_rank_model_dir"] = rank_candidates[0]
		result["warnings"].append("Multiple ROI ranking model candidates found in Drive folder; newest one was selected.")
	elif len(rank_h5_candidates) == 1:
		result["roi_rank_model_dir"] = rank_h5_candidates[0]
	elif len(rank_h5_candidates) > 1:
		rank_h5_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
		result["roi_rank_model_dir"] = rank_h5_candidates[0]
		result["warnings"].append("Multiple ROI ranking H5 bundle candidates found in Drive folder; newest one was selected.")

	if not result["signal_models_source"]:
		result["warnings"].append("Signal models source could not be auto-detected in Drive folder.")
	if not result["noise_models_root"]:
		result["warnings"].append("Noise models source could not be auto-detected in Drive folder.")
	if not result["filter_file"]:
		result["warnings"].append("Filter file could not be auto-detected in Drive folder.")
	if not result["roi_rank_model_dir"]:
		result["warnings"].append("ROI ranking model directory could not be auto-detected in Drive folder.")
	return result


def _prepare_uploaded_roi_rank_model_dir(
	weights_path: Optional[str],
	scalers_path: Optional[str],
	meta_path: Optional[str],
) -> Tuple[Optional[str], Optional[str]]:
	"""
	Build a temporary ROI ranking model directory from uploaded 1.5 artifacts.
	Returns: (prepared_dir, warning_message)
	"""
	provided = int(bool(weights_path)) + int(bool(scalers_path)) + int(bool(meta_path))
	if provided == 0:
		return None, None
	if provided != 3:
		return None, "ROI ranking upload is incomplete. Please upload .pt + .npz + .json artifacts."

	rank_dir_tmp = os.path.join(tempfile.gettempdir(), "predobs_roi_rank_upload")
	os.makedirs(rank_dir_tmp, exist_ok=True)
	try:
		shutil.copyfile(str(weights_path), os.path.join(rank_dir_tmp, "roi_rank_nn.pt"))
		shutil.copyfile(str(scalers_path), os.path.join(rank_dir_tmp, "roi_rank_scalers.npz"))
		shutil.copyfile(str(meta_path), os.path.join(rank_dir_tmp, "roi_rank_training_meta.json"))
		return str(rank_dir_tmp), None
	except Exception as e:
		return None, f"Could not prepare uploaded ROI ranking model artifacts: {e}"


def _prepare_roi_rank_model_dir_from_h5_bundle(h5_bundle_path: str) -> Tuple[Optional[str], Optional[str]]:
	"""Materialize a temporary ROI ranking model directory from single-file H5 bundle."""
	bp = str(h5_bundle_path or "").strip()
	if (not bp) or (not os.path.isfile(bp)):
		return None, "ROI ranking H5 bundle was not found."

	rank_dir_tmp = os.path.join(tempfile.gettempdir(), "predobs_roi_rank_h5_bundle")
	os.makedirs(rank_dir_tmp, exist_ok=True)
	weights_out = os.path.join(rank_dir_tmp, "roi_rank_nn.pt")
	scalers_out = os.path.join(rank_dir_tmp, "roi_rank_scalers.npz")
	meta_out = os.path.join(rank_dir_tmp, "roi_rank_training_meta.json")

	try:
		with h5py.File(bp, "r") as hf:
			if ("state_dict_blob" not in hf) or ("scalers" not in hf) or ("meta_json" not in hf):
				return None, "Invalid ROI ranking H5 bundle: missing one of state_dict_blob/scalers/meta_json."

			blob = np.asarray(hf["state_dict_blob"], dtype=np.uint8).tobytes()
			with open(weights_out, "wb") as fw:
				fw.write(blob)

			sc = hf["scalers"]
			x_mean = np.asarray(sc.get("x_mean", []), dtype=np.float64)
			x_scale = np.asarray(sc.get("x_scale", []), dtype=np.float64)
			y_mean = np.asarray(sc.get("y_mean", []), dtype=np.float64)
			y_scale = np.asarray(sc.get("y_scale", []), dtype=np.float64)
			if (x_mean.size == 0) or (x_scale.size == 0) or (y_mean.size == 0) or (y_scale.size == 0):
				return None, "Invalid ROI ranking H5 bundle: scalers group is incomplete."
			np.savez(
				scalers_out,
				x_mean=x_mean,
				x_scale=x_scale,
				y_mean=y_mean,
				y_scale=y_scale,
			)

			meta_raw = hf["meta_json"][()]
			if isinstance(meta_raw, bytes):
				meta_text = meta_raw.decode("utf-8", errors="ignore")
			else:
				meta_text = str(meta_raw)
			meta_obj = json.loads(meta_text)
			with open(meta_out, "w", encoding="utf-8") as fm:
				json.dump(meta_obj, fm, indent=2, ensure_ascii=False)

			# Optional ranking payloads for downstream target-frequency extraction.
			if "ranking_json" in hf:
				try:
					rj_raw = hf["ranking_json"][()]
					if isinstance(rj_raw, bytes):
						rj_text = rj_raw.decode("utf-8", errors="ignore")
					else:
						rj_text = str(rj_raw)
					rows = json.loads(rj_text)
					if isinstance(rows, list):
						with open(os.path.join(rank_dir_tmp, "roi_ranking_global_test.json"), "w", encoding="utf-8") as frj:
							json.dump(rows, frj, indent=2, ensure_ascii=False)
				except Exception:
					pass

		return str(rank_dir_tmp), None
	except Exception as e:
		return None, f"Could not read ROI ranking H5 bundle: {e}"


def _resolve_roi_rank_model_dir(model_source: str) -> Tuple[str, Optional[str]]:
	"""Resolve ROI ranking source path (directory or single .h5 bundle) into artifact directory."""
	src = str(model_source or "").strip()
	if (not src) or (not os.path.isfile(src) and not os.path.isdir(src)):
		return src, None
	if os.path.isdir(src):
		return src, None
	ext = os.path.splitext(src)[1].lower()
	if ext in (".h5", ".hdf5"):
		return _prepare_roi_rank_model_dir_from_h5_bundle(src)
	return src, None


def _apply_drive_auto_paths(
	signal_models_root: str,
	noise_models_root: str,
	filter_file: str,
	roi_rank_model_dir: str,
	auto_paths: dict,
) -> Tuple[str, str, str, str]:
	"""Apply detected Drive paths over current sources when available."""
	out_signal = str(signal_models_root)
	out_noise = str(noise_models_root)
	out_filter = str(filter_file)
	out_rank = str(roi_rank_model_dir)
	if not isinstance(auto_paths, dict):
		return out_signal, out_noise, out_filter, out_rank

	if auto_paths.get("signal_models_source", ""):
		out_signal = str(auto_paths["signal_models_source"])
	if auto_paths.get("noise_models_root", ""):
		out_noise = str(auto_paths["noise_models_root"])
	if auto_paths.get("filter_file", ""):
		out_filter = str(auto_paths["filter_file"])
	if auto_paths.get("roi_rank_model_dir", ""):
		out_rank = str(auto_paths["roi_rank_model_dir"])
	return out_signal, out_noise, out_filter, out_rank


def _validate_local_preset_sources(
	signal_models_root: str,
	noise_models_root: str,
	filter_file: str,
	roi_rank_model_dir: str,
) -> List[str]:
	"""Return warning strings for missing local preset paths."""
	warnings: List[str] = []
	if (not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root)):
		warnings.append(f"Local signal source not found: {signal_models_root}")
	if not os.path.isfile(noise_models_root):
		warnings.append(f"Local noise file not found: {noise_models_root}")
	if not os.path.isfile(filter_file):
		warnings.append(f"Local filter file not found: {filter_file}")
	if (not os.path.isdir(roi_rank_model_dir)) and (not os.path.isfile(roi_rank_model_dir)):
		warnings.append(f"Local ROI ranking model directory not found: {roi_rank_model_dir}")
	return warnings


def _roi_rank_artifact_paths(model_dir: str) -> Tuple[str, str, str]:
	md = str(model_dir or "").strip()
	return (
		os.path.join(md, "roi_rank_nn.pt"),
		os.path.join(md, "roi_rank_scalers.npz"),
		os.path.join(md, "roi_rank_training_meta.json"),
	)


def _validate_roi_rank_artifacts(model_dir: str) -> Optional[str]:
	weights_path, scalers_path, meta_path = _roi_rank_artifact_paths(model_dir)
	if not os.path.isfile(meta_path):
		return f"Model meta not found: {meta_path}"
	if not os.path.isfile(scalers_path):
		return f"Model scalers not found: {scalers_path}"
	if not os.path.isfile(weights_path):
		return f"Model weights not found: {weights_path}"
	return None


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
		safe_name = os.path.basename(str(getattr(upload_obj, "name", "upload.bin")))
		ext = os.path.splitext(safe_name)[1]
		if hasattr(upload_obj, "seek"):
			try:
				upload_obj.seek(0)
			except Exception:
				pass
		hash_md5 = hashlib.md5()
		total = 0
		root = os.path.join(tempfile.gettempdir(), "predobs_manual_uploads")
		os.makedirs(root, exist_ok=True)
		tmp_dst = os.path.join(root, f"{prefix}_{int(time.time() * 1000)}_{os.getpid()}{ext if ext else ''}.tmp")
		_read_once_without_size = False
		with open(tmp_dst, "wb") as f:
			while True:
				try:
					chunk = upload_obj.read(8 * 1024 * 1024)
				except TypeError:
					if _read_once_without_size:
						chunk = b""
					else:
						chunk = upload_obj.read()
						_read_once_without_size = True
				if not chunk:
					break
				if isinstance(chunk, str):
					chunk = chunk.encode("utf-8", errors="ignore")
				f.write(chunk)
				hash_md5.update(chunk)
				total += int(len(chunk))
		if int(total) <= 0 and hasattr(upload_obj, "getbuffer"):
			try:
				raw = bytes(upload_obj.getbuffer())
				if raw:
					with open(tmp_dst, "wb") as f:
						f.write(raw)
					hash_md5 = hashlib.md5(raw)
					total = int(len(raw))
			except Exception:
				pass
		if int(total) <= 0:
			try:
				os.remove(tmp_dst)
			except Exception:
				pass
			return None
		h = hash_md5.hexdigest()[:16]
		dst = os.path.join(root, f"{prefix}_{h}{ext if ext else ''}")
		if not os.path.isfile(dst):
			os.replace(tmp_dst, dst)
		else:
			try:
				os.remove(tmp_dst)
			except Exception:
				pass
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


def parse_roi_freq_bounds_from_dirname(roi_dir_name: str) -> Tuple[Optional[float], Optional[float]]:
	m = re.search(r"_f([0-9]+(?:\.[0-9]+)?)-([0-9]+(?:\.[0-9]+)?)GHz$", str(roi_dir_name).strip())
	if not m:
		return None, None
	try:
		lo = float(m.group(1))
		hi = float(m.group(2))
		return float(min(lo, hi)), float(max(lo, hi))
	except Exception:
		return None, None


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
	if mode == "asinh_log10":
		z = np.sinh(yy)
		return np.sign(z) * s * (np.power(10.0, np.abs(z)) - 1.0)
	if mode == "tanh":
		z = np.clip(yy, -0.999999, 0.999999)
		return s * np.arctanh(z)
	if mode == "arctanh":
		return s * np.tanh(yy)
	return yy


def apply_target_transform(y, transform_name="none", scale=1.0):
	yy = np.asarray(y, dtype=np.float64)
	mode = str(transform_name).strip().lower()
	s = np.asarray(scale, dtype=np.float64)
	s = np.maximum(1e-12, s)
	if mode == "asinh":
		return np.arcsinh(yy / s)
	if mode == "asinh_log10":
		z = np.sign(yy) * np.log10(1.0 + (np.abs(yy) / s))
		return np.arcsinh(z)
	if mode == "tanh":
		return np.tanh(yy / s)
	if mode == "arctanh":
		ratio = np.clip(yy / s, -0.999999, 0.999999)
		return np.arctanh(ratio)
	return yy


class SerializedXGBoostBoosterRegressor:
	"""Compatibility class to unpickle compact models generated by script 1.4."""

	def __init__(self, booster_bytes):
		self.booster_bytes = bytes(booster_bytes)
		self._booster = None

	def _get_booster(self):
		if self._booster is None:
			if xgb is None:
				raise RuntimeError("xgboost is required to use compact XGBoost models")
			booster = xgb.Booster()
			booster.load_model(bytearray(self.booster_bytes))
			self._booster = booster
		return self._booster

	def predict(self, X):
		if xgb is None:
			raise RuntimeError("xgboost is required for prediction")
		xx = np.asarray(X, dtype=np.float32)
		dmat = xgb.DMatrix(xx)
		pred = self._get_booster().predict(dmat)
		return np.asarray(pred, dtype=np.float64).reshape(-1)


class PhysicalDomainCalibratedRegressor:
	"""Compatibility class to unpickle calibrated wrappers generated by script 1.4."""

	def __init__(self, base_estimator, a=1.0, b=0.0, transform_name="none", transform_scale=1.0):
		self.base_estimator = base_estimator
		self.a = float(a)
		self.b = float(b)
		self.transform_name = str(transform_name)
		self.transform_scale = float(transform_scale)

	def predict(self, X):
		pred_t = np.asarray(self.base_estimator.predict(X), dtype=np.float64).reshape(-1, 1)
		pred_raw = inverse_target_transform(pred_t, self.transform_name, self.transform_scale).reshape(-1)
		pred_raw_cal = (self.a * pred_raw) + self.b
		pred_t_cal = apply_target_transform(pred_raw_cal.reshape(-1, 1), self.transform_name, self.transform_scale).reshape(-1)
		return pred_t_cal


# Ensure joblib can resolve classes pickled from training script executed as __main__.
try:
	_main_mod = sys.modules.get("__main__", None)
	if _main_mod is not None:
		setattr(_main_mod, "SerializedXGBoostBoosterRegressor", SerializedXGBoostBoosterRegressor)
		setattr(_main_mod, "PhysicalDomainCalibratedRegressor", PhysicalDomainCalibratedRegressor)
except Exception:
	pass


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


def _apply_velocity_shift_by_spw_centers_segmented(freq_ghz: np.ndarray, velocity_kms: float, gap_factor: float = 20.0) -> np.ndarray:
	"""Apply SPW-center shift per contiguous spectral segment.
	Useful for concatenated multi-SPW spectra where a single global center is incorrect.
	"""
	f = np.asarray(freq_ghz, dtype=np.float64).reshape(-1)
	if f.size == 0:
		return f
	out = np.asarray(f, dtype=np.float64).copy()

	if f.size == 1:
		return _apply_velocity_shift_by_spw_center(out, float(velocity_kms))

	d = np.abs(np.diff(f))
	d_valid = d[np.isfinite(d) & (d > 0.0)]
	if d_valid.size == 0:
		return _apply_velocity_shift_by_spw_center(out, float(velocity_kms))

	base_step = float(np.nanmedian(d_valid))
	th_gap = float(max(base_step * float(max(2.0, gap_factor)), base_step * 5.0))
	split_after = np.where(d > th_gap)[0].astype(int)

	starts = [0]
	ends = []
	for k in split_after:
		ends.append(int(k))
		starts.append(int(k) + 1)
	ends.append(int(f.size - 1))

	c_kms = 299792.458
	for a, b in zip(starts, ends):
		idx = np.arange(int(a), int(b) + 1, dtype=int)
		if idx.size <= 0:
			continue
		seg = out[idx]
		if not np.any(np.isfinite(seg)):
			continue
		spw_center_ghz = float(0.5 * (np.nanmin(seg) + np.nanmax(seg)))
		delta_f = -float(spw_center_ghz) * (float(velocity_kms) / c_kms)
		out[idx] = seg + delta_f

	return out


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


def list_h5_roi_models_v14(h5_path: str) -> List[dict]:
	out_map: Dict[str, dict] = {}
	with h5py.File(h5_path, "r") as hf:
		grp = hf.get("models")
		if grp is None:
			return []

		def visitor(_name, obj):
			if not isinstance(obj, h5py.Dataset):
				return
			ds_path = str(obj.name).strip("/")
			if not ds_path.endswith("/joblib"):
				return
			parts = ds_path.split("/")
			if len(parts) < 4 or parts[0] != "models":
				return

			roi_name = str(parts[1])
			model_name = str(parts[2])
			lo, hi = parse_roi_freq_bounds_from_dirname(roi_name)
			if lo is None or hi is None:
				return

			if roi_name not in out_map:
				out_map[roi_name] = {
					"roi_name": str(roi_name),
					"roi_lo_ghz": float(lo),
					"roi_hi_ghz": float(hi),
					"roi_center_ghz": float(0.5 * (float(lo) + float(hi))),
					"model_refs": [],
				}

			out_map[roi_name]["model_refs"].append((model_name, ds_path))

		grp.visititems(visitor)

	out = []
	for roi_name, rec in out_map.items():
		_ = roi_name
		rec["model_refs"] = sorted(list(rec.get("model_refs", [])), key=lambda t: t[0].lower())
		out.append(rec)
	return sorted(out, key=lambda d: float(d["roi_center_ghz"]))


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

	roi_dirs = sorted(glob.glob(os.path.join(models_root, "ROI_*")))
	for rd in roi_dirs:
		roi_name = os.path.basename(rd)
		lo, hi = parse_roi_freq_bounds_from_dirname(roi_name)
		if lo is None or hi is None:
			continue
		mps = sorted(glob.glob(os.path.join(rd, "*", "model", "final_model.joblib")))
		if not mps:
			continue
		items = []
		for mp in mps:
			model_name = os.path.basename(os.path.dirname(os.path.dirname(mp)))
			items.append((model_name, mp))
		rows.append((roi_name, 0.5 * (float(lo) + float(hi)), items))
	return sorted(rows, key=lambda t: t[1])


def list_folder_roi_models_v14(models_root: str) -> List[dict]:
	out: List[dict] = []
	roi_dirs = sorted(glob.glob(os.path.join(str(models_root), "ROI_*")))
	for rd in roi_dirs:
		roi_name = os.path.basename(rd)
		lo, hi = parse_roi_freq_bounds_from_dirname(roi_name)
		if lo is None or hi is None:
			continue
		mps = sorted(glob.glob(os.path.join(rd, "*", "model", "final_model.joblib")))
		if not mps:
			continue
		models = []
		for mp in mps:
			model_name = os.path.basename(os.path.dirname(os.path.dirname(mp)))
			models.append((model_name, mp))
		out.append({
			"roi_name": str(roi_name),
			"roi_lo_ghz": float(lo),
			"roi_hi_ghz": float(hi),
			"roi_center_ghz": float(0.5 * (float(lo) + float(hi))),
			"model_refs": sorted(models, key=lambda t: t[0].lower()),
		})
	return sorted(out, key=lambda d: float(d["roi_center_ghz"]))


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


def _package_is_channel_aware_roi_model(package: dict) -> bool:
	if not isinstance(package, dict):
		return False
	fc = [str(v).strip().lower() for v in list(package.get("feature_columns", []))]
	has_ch_freq = "channel_freq_ghz" in fc
	has_ch_rel = "channel_relpos" in fc
	roi_idx = package.get("roi_channel_indices", None)
	has_roi_idx = isinstance(roi_idx, (list, tuple, np.ndarray)) and len(roi_idx) > 0
	return bool(has_ch_freq and has_ch_rel and has_roi_idx)


def _estimate_roi_frequency_axis(package: dict, roi_lo_ghz: Optional[float] = None, roi_hi_ghz: Optional[float] = None) -> Optional[np.ndarray]:
	if not isinstance(package, dict):
		return None
	roi_idx = package.get("roi_channel_indices", None)
	if not isinstance(roi_idx, (list, tuple, np.ndarray)):
		return None
	n_ch = int(len(roi_idx))
	if n_ch <= 0:
		return None
	if roi_lo_ghz is None or roi_hi_ghz is None:
		return None
	lo = float(min(float(roi_lo_ghz), float(roi_hi_ghz)))
	hi = float(max(float(roi_lo_ghz), float(roi_hi_ghz)))
	if n_ch == 1:
		return np.asarray([0.5 * (lo + hi)], dtype=np.float64)
	return np.linspace(lo, hi, num=n_ch, dtype=np.float64)


def predict_with_joblib_roi_package_batch(package: dict, x_features_2d: np.ndarray, roi_freq_ghz: np.ndarray) -> np.ndarray:
	if not _package_is_channel_aware_roi_model(package):
		y = predict_with_joblib_package_batch(package, x_features_2d)
		return np.asarray(y, dtype=np.float32).reshape(-1, 1)

	model = package["model"]
	scaler_x = package["scaler_x"]
	transform_name = str(package.get("target_transform", "none"))
	transform_scale = float(package.get("target_transform_scale", 1.0))
	feature_columns = [str(v).strip().lower() for v in list(package.get("feature_columns", []))]

	x_phys_raw = np.asarray(x_features_2d, dtype=np.float32)
	freq = np.asarray(roi_freq_ghz, dtype=np.float32).reshape(-1)
	n_spec = int(x_phys_raw.shape[0])
	n_ch = int(freq.size)
	if n_spec <= 0 or n_ch <= 0:
		return np.zeros((n_spec, 0), dtype=np.float32)

	# 1.4 training scales only physical variables (4 cols), then appends channel features.
	n_scaler_in = int(getattr(scaler_x, "n_features_in_", x_phys_raw.shape[1]))
	if n_scaler_in == int(x_phys_raw.shape[1]):
		x_phys = np.asarray(scaler_x.transform(x_phys_raw), dtype=np.float32)
		apply_scaler_after_assembly = False
	else:
		x_phys = np.asarray(x_phys_raw, dtype=np.float32)
		apply_scaler_after_assembly = True

	if n_ch == 1:
		rel = np.zeros((1,), dtype=np.float32)
	else:
		rel = np.linspace(0.0, 1.0, num=n_ch, dtype=np.float32)

	base = {
		"logn": np.repeat(x_phys[:, 0], n_ch),
		"tex": np.repeat(x_phys[:, 1], n_ch),
		"velo": np.repeat(x_phys[:, 2], n_ch),
		"fwhm": np.repeat(x_phys[:, 3], n_ch),
		"channel_freq_ghz": np.tile(freq, n_spec),
		"channel_relpos": np.tile(rel, n_spec),
	}

	cols = []
	for c in feature_columns:
		if c in base:
			cols.append(base[c])
		else:
			cols.append(np.zeros((n_spec * n_ch,), dtype=np.float32))
	x_full = np.column_stack(cols).astype(np.float32)
	if apply_scaler_after_assembly:
		x_n = scaler_x.transform(x_full)
	else:
		x_n = x_full
	y_t = np.asarray(model.predict(x_n), dtype=np.float64).reshape(-1, 1)
	y_raw = np.asarray(inverse_target_transform(y_t, transform_name=transform_name, scale=transform_scale), dtype=np.float32).reshape(-1)
	return y_raw.reshape(n_spec, n_ch)


def predict_signal_roi_batch(
	signal_models_source: str,
	is_h5_signal: bool,
	roi_entries,
	x_features_2d: np.ndarray,
	pkg_cache: Dict[str, object],
) -> Tuple[np.ndarray, np.ndarray, Optional[str]]:
	if not roi_entries:
		return None, None, "No ROI entries"

	# New hierarchy (1.4): one ROI-level model predicts all channels.
	if isinstance(roi_entries[0], dict) and (str(roi_entries[0].get("entry_type", "")).lower() == "roi_model_v14"):
		entry = dict(roi_entries[0])
		model_refs = list(entry.get("model_refs", []))
		if not model_refs:
			return None, None, "No model references in selected ROI"

		pred_acc = None
		pred_cnt = 0
		roi_freq_ref = None
		for model_name, ref in model_refs:
			cache_key = f"{model_name}|{ref}"
			try:
				if cache_key not in pkg_cache:
					if is_h5_signal:
						pkg_cache[cache_key] = load_joblib_package_from_h5(signal_models_source, ref)
					else:
						pkg_cache[cache_key] = joblib.load(ref)
				pkg = pkg_cache[cache_key]
				roi_freq = _estimate_roi_frequency_axis(pkg, entry.get("roi_lo_ghz", None), entry.get("roi_hi_ghz", None))
				if roi_freq is None or roi_freq.size <= 0:
					continue
				pred2d = predict_with_joblib_roi_package_batch(pkg, x_features_2d, roi_freq)
				if pred_acc is None:
					pred_acc = np.asarray(pred2d, dtype=np.float64)
					roi_freq_ref = np.asarray(roi_freq, dtype=np.float64)
				else:
					if pred_acc.shape != pred2d.shape:
						continue
					pred_acc += np.asarray(pred2d, dtype=np.float64)
				pred_cnt += 1
			except Exception:
				continue

		if pred_cnt <= 0 or pred_acc is None or roi_freq_ref is None:
			return None, None, "No valid predictions for selected ROI"

		y_mean = (pred_acc / float(pred_cnt)).astype(np.float32)
		return np.asarray(roi_freq_ref, dtype=np.float64), y_mean, None

	# Legacy channel-by-channel hierarchy.
	n_spec = int(np.asarray(x_features_2d).shape[0])
	freqs: List[float] = []
	cols: List[np.ndarray] = []
	for _, fch, model_refs in roi_entries:
		pred_acc = np.zeros((n_spec,), dtype=np.float64)
		pred_cnt = 0
		for model_name, ref in model_refs:
			cache_key = f"{model_name}|{ref}"
			try:
				if cache_key not in pkg_cache:
					if is_h5_signal:
						pkg_cache[cache_key] = load_joblib_package_from_h5(signal_models_source, ref)
					else:
						pkg_cache[cache_key] = joblib.load(ref)
				pkg = pkg_cache[cache_key]
				pred = predict_with_joblib_package_batch(pkg, x_features_2d)
				pred_acc += np.asarray(pred, dtype=np.float64)
				pred_cnt += 1
			except Exception:
				continue
		if pred_cnt > 0:
			freqs.append(float(fch))
			cols.append((pred_acc / float(pred_cnt)).astype(np.float32))

	if not cols:
		return None, None, "No valid synthetic predictions in selected ROI"

	roi_freq = np.asarray(freqs, dtype=np.float64)
	Y = np.stack(cols, axis=1).astype(np.float32)
	ord_idx = np.argsort(roi_freq)
	return roi_freq[ord_idx], Y[:, ord_idx], None


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


def _spiral_pixel_order_valid(valid_mask: np.ndarray, center_y: Optional[int] = None, center_x: Optional[int] = None) -> List[Tuple[int, int]]:
	vm = np.asarray(valid_mask, dtype=bool)
	if vm.ndim != 2:
		return []
	h, w = int(vm.shape[0]), int(vm.shape[1])
	if center_y is None or center_x is None:
		cy, cx = int(h // 2), int(w // 2)
	else:
		cy = int(max(0, min(h - 1, int(center_y))))
		cx = int(max(0, min(w - 1, int(center_x))))
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
	if is_h5:
		roi_rows_h5 = list_h5_roi_models_v14(signal_source)
		if roi_rows_h5:
			t = float(target_frequency_ghz)
			selected = None
			for r in roi_rows_h5:
				if float(r["roi_lo_ghz"]) <= t <= float(r["roi_hi_ghz"]):
					selected = dict(r)
					break
			if selected is None and allow_nearest:
				selected = min(roi_rows_h5, key=lambda rr: abs(float(rr["roi_center_ghz"]) - t))
			if selected is None:
				selected = min(roi_rows_h5, key=lambda rr: abs(float(rr["roi_center_ghz"]) - t))
			if selected is None:
				raise RuntimeError("Could not select ROI")

			selected_model_norm = normalize_model_name(selected_model_name)
			pm = str(pred_mode).strip().lower()
			model_refs = list(selected.get("model_refs", []))
			if pm == "single_model":
				model_refs = [(mn, ref) for (mn, ref) in model_refs if normalize_model_name(mn) == selected_model_norm]
				if not model_refs:
					raise RuntimeError("No channels inside ROI")

			selected["model_refs"] = model_refs
			selected["entry_type"] = "roi_model_v14"

			n_hint = 1
			if model_refs:
				try:
					pkg0 = load_joblib_package_from_h5(signal_source, model_refs[0][1])
					if isinstance(pkg0, dict) and isinstance(pkg0.get("roi_channel_indices", None), (list, tuple, np.ndarray)):
						n_hint = int(max(1, len(pkg0.get("roi_channel_indices", []))))
				except Exception:
					pass
			if n_hint <= 1:
				roi_freq = np.asarray([float(selected["roi_center_ghz"])], dtype=np.float64)
			else:
				roi_freq = np.linspace(float(selected["roi_lo_ghz"]), float(selected["roi_hi_ghz"]), num=int(n_hint), dtype=np.float64)
			return is_h5, [selected], roi_freq

	if (not is_h5) and os.path.isdir(signal_source):
		roi_rows = list_folder_roi_models_v14(signal_source)
		if roi_rows:
			t = float(target_frequency_ghz)
			selected = None
			for r in roi_rows:
				if float(r["roi_lo_ghz"]) <= t <= float(r["roi_hi_ghz"]):
					selected = dict(r)
					break
			if selected is None and allow_nearest:
				selected = min(roi_rows, key=lambda rr: abs(float(rr["roi_center_ghz"]) - t))
			# Robust fallback for ROI-folder hierarchy: choose nearest ROI if no overlap.
			if selected is None:
				selected = min(roi_rows, key=lambda rr: abs(float(rr["roi_center_ghz"]) - t))
			if selected is None:
				raise RuntimeError("Could not select ROI")

			selected_model_norm = normalize_model_name(selected_model_name)
			pm = str(pred_mode).strip().lower()
			model_refs = list(selected.get("model_refs", []))
			if pm == "single_model":
				model_refs = [(mn, ref) for (mn, ref) in model_refs if normalize_model_name(mn) == selected_model_norm]
				if not model_refs:
					raise RuntimeError("No channels inside ROI")

			selected["model_refs"] = model_refs
			selected["entry_type"] = "roi_model_v14"

			n_hint = 1
			if model_refs:
				try:
					pkg0 = joblib.load(model_refs[0][1])
					if isinstance(pkg0, dict) and isinstance(pkg0.get("roi_channel_indices", None), (list, tuple, np.ndarray)):
						n_hint = int(max(1, len(pkg0["roi_channel_indices"])))
				except Exception:
					pass
			if n_hint <= 1:
				roi_freq = np.asarray([float(selected["roi_center_ghz"])], dtype=np.float64)
			else:
				roi_freq = np.linspace(float(selected["roi_lo_ghz"]), float(selected["roi_hi_ghz"]), num=int(n_hint), dtype=np.float64)
			return is_h5, [selected], roi_freq

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


def _resolve_roi_selected_pos(value, rois: List[dict], default_pos: int = 0) -> int:
	"""
	Convert a stored session value into a valid list position.
	Accepts:
	- integer position (0..len-1)
	- ROI index (field roi['index'])
	- UI labels like "ROI S1 | ..." or "ROI N2 | ..."
	"""
	if not rois:
		return int(default_pos)

	def _clamp_pos(p):
		try:
			pp = int(p)
		except Exception:
			pp = int(default_pos)
		if pp < 0:
			return 0
		if pp >= len(rois):
			return int(len(rois) - 1)
		return int(pp)

	# 1) Already a valid list position
	try:
		iv = int(value)
		if 0 <= int(iv) < len(rois):
			return int(iv)
	except Exception:
		iv = None

	# 2) Interpretable as ROI index
	if iv is not None:
		for i, r in enumerate(rois):
			try:
				if int(r.get("index", -999999)) == int(iv):
					return int(i)
			except Exception:
				continue

	# 3) Parse labels like "ROI S1 | ..."
	try:
		txt = str(value)
	except Exception:
		txt = ""

	m = re.search(r"ROI\s*[A-Za-z]?\s*(\d+)", txt)
	if m is None:
		m = re.search(r"(\d+)", txt)
	if m is not None:
		try:
			roi_idx = int(m.group(1))
			for i, r in enumerate(rois):
				try:
					if int(r.get("index", -999999)) == roi_idx:
						return int(i)
				except Exception:
					continue
		except Exception:
			pass

	return _clamp_pos(default_pos)


def _collect_signal_rois_for_ui(signal_source: str, filter_file: str) -> List[dict]:
	if not signal_source:
		return []
	if (not os.path.isfile(signal_source)) and (not os.path.isdir(signal_source)):
		return []
	if os.path.isfile(signal_source) and str(signal_source).lower().endswith(".h5"):
		roi_rows_h5 = list_h5_roi_models_v14(signal_source)
		if roi_rows_h5:
			out = []
			for i, r in enumerate(roi_rows_h5, start=1):
				out.append({
					"index": int(i),
					"lo": float(r["roi_lo_ghz"]),
					"hi": float(r["roi_hi_ghz"]),
					"a": 0,
					"b": 0,
					"overlap": False,
				})
			return out
	if os.path.isdir(signal_source):
		roi_rows = list_folder_roi_models_v14(signal_source)
		if roi_rows:
			out = []
			for i, r in enumerate(roi_rows, start=1):
				out.append({
					"index": int(i),
					"lo": float(r["roi_lo_ghz"]),
					"hi": float(r["roi_hi_ghz"]),
					"a": 0,
					"b": 0,
					"overlap": False,
				})
			return out
	if not filter_file or (not os.path.isfile(filter_file)):
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


def _propagate_selected_freqs_to_all_guides(selected_freqs: List[float]):
	add = [float(v) for v in (selected_freqs or []) if np.isfinite(float(v))]
	if not add:
		return

	def _merge_to_input_key(input_key: str, last_nonempty_key: Optional[str] = None, pending_key: Optional[str] = None, refresh_key: Optional[str] = None):
		base_txt = str(st.session_state.get(pending_key or input_key, st.session_state.get(input_key, "")))
		base_vals = parse_freq_list(base_txt)
		merged = _normalize_target_freqs_for_run([float(v) for v in (base_vals + add)])
		merged_txt = _freqs_to_text(merged)
		# IMPORTANT: do not write directly to widget keys after instantiation.
		# Use pending + refresh pattern to avoid StreamlitAPIException.
		if last_nonempty_key:
			st.session_state[last_nonempty_key] = str(merged_txt)
		if pending_key:
			st.session_state[pending_key] = str(merged_txt)
		else:
			st.session_state[input_key] = str(merged_txt)
		if refresh_key:
			st.session_state[refresh_key] = True

	_merge_to_input_key("p6_guide_freqs_main_input", "p6_guide_freqs_main_last_nonempty", "p6_guide_freqs_main_pending", "p6_guide_main_refresh")
	_merge_to_input_key("p6_guide_freqs_cube2_input", "p6_guide_freqs_cube2_last_nonempty", "p6_guide_freqs_cube2_pending", "p6_guide_cube2_refresh")
	_merge_to_input_key("p6_guide_freqs_cube3_input", "p6_guide_freqs_cube3_last_nonempty", "p6_guide_freqs_cube3_pending", "p6_guide_cube3_refresh")
	_merge_to_input_key("p6_guide_freqs_fit_input", "p6_guide_freqs_fit_last_nonempty", "p6_guide_freqs_fit_pending", "p6_guide_fit_refresh")

	# Cube fitting tab (now using pending/refresh too).
	_merge_to_input_key("p6_guide_freqs_cfit_input", pending_key="p6_guide_freqs_cfit_pending", refresh_key="p6_guide_cfit_refresh")
	# Inverse cube prediction tab.
	_merge_to_input_key("p6_guide_freqs_icp_input", pending_key="p6_guide_freqs_icp_pending", refresh_key="p6_guide_icp_refresh")


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
			roi_freq, y_syn_valid, pred_err = predict_signal_roi_batch(
				signal_models_source=signal_models_source,
				is_h5_signal=bool(is_h5_signal),
				roi_entries=roi_entries,
				x_features_2d=x_valid,
				pkg_cache=signal_pkg_cache,
			)
			if pred_err is not None or roi_freq is None or y_syn_valid is None:
				raise RuntimeError(pred_err if pred_err is not None else "No valid signal channels after model prediction")

			nchan = int(roi_freq.size)
			tag = f"{float(target_freq):.6f}".replace(".", "p")
			final_fits = os.path.join(out_dir, f"{out_prefix}_target{tag}.fits")
			synth_fits = os.path.join(out_dir, f"{out_prefix}_target{tag}_SYNTHONLY.fits")
			progress_fits = os.path.join(out_dir, f"{out_prefix}_target{tag}_INPROGRESS.fits")
			progress_png = os.path.join(out_dir, f"{out_prefix}_target{tag}_INPROGRESS_MAP.png")

			cube_final = np.full((nchan, ny, nx), np.nan, dtype=np.float32)
			cube_syn = np.full((nchan, ny, nx), np.nan, dtype=np.float32)

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


def _write_done_mask_fits_2d(out_fits_path: str, done_mask: np.ndarray, ref_hdr: Optional[object], history_text: str):
	if fits is None:
		return
	arr = np.asarray(done_mask, dtype=np.uint8)
	hdr = fits.Header()
	hdr["WCSAXES"] = 2
	hdr["BUNIT"] = "mask"
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


def _header_to_celestial_wcs(ref_hdr: Optional[object]):
	if (WCS is None) or (ref_hdr is None):
		return None
	try:
		w = WCS(ref_hdr)
		wc = w.celestial
		if wc is None:
			return None
		return wc
	except Exception:
		return None


def _compute_zoom_limits_from_mask(mask_2d: Optional[np.ndarray], pad_frac: float = 0.08):
	if mask_2d is None:
		return None
	pm = np.asarray(mask_2d, dtype=bool)
	if pm.ndim != 2 or (not np.any(pm)):
		return None
	yy, xx = np.where(pm)
	h, w = int(pm.shape[0]), int(pm.shape[1])
	span_y = int(np.max(yy) - np.min(yy) + 1)
	span_x = int(np.max(xx) - np.min(xx) + 1)
	pad = int(max(2, round(float(pad_frac) * max(span_y, span_x))))
	y0 = max(0, int(np.min(yy)) - pad)
	y1 = min(h - 1, int(np.max(yy)) + pad)
	x0 = max(0, int(np.min(xx)) - pad)
	x1 = min(w - 1, int(np.max(xx)) + pad)
	return (int(x0), int(x1), int(y0), int(y1))


def _save_cubefit_progress_png(
	logn_map: np.ndarray,
	tex_map: np.ndarray,
	velo_map: np.ndarray,
	fwhm_map: np.ndarray,
	done_steps: int,
	total_steps: int,
	out_png: str,
	ref_hdr: Optional[object] = None,
	processed_mask: Optional[np.ndarray] = None,
):
	wcs2d = _header_to_celestial_wcs(ref_hdr)
	if wcs2d is not None:
		fig = plt.figure(figsize=(10.8, 8.6))
		axes = [
			fig.add_subplot(2, 2, 1, projection=wcs2d),
			fig.add_subplot(2, 2, 2, projection=wcs2d),
			fig.add_subplot(2, 2, 3, projection=wcs2d),
			fig.add_subplot(2, 2, 4, projection=wcs2d),
		]
	else:
		fig, ax_grid = plt.subplots(2, 2, figsize=(10.8, 8.6))
		axes = list(ax_grid.ravel())
	items = [
		("logN", np.asarray(logn_map, dtype=np.float32), "viridis"),
		("Tex", np.asarray(tex_map, dtype=np.float32), "magma"),
		("Velocity", np.asarray(velo_map, dtype=np.float32), "coolwarm"),
		("FWHM", np.asarray(fwhm_map, dtype=np.float32), "plasma"),
	]
	zoom_lim = _compute_zoom_limits_from_mask(processed_mask, pad_frac=0.08)
	for ax, (ttl, arr, cmap) in zip(axes, items):
		fin = np.isfinite(arr)
		if np.any(fin):
			vmin, vmax = _compute_display_limits(arr)
			im = ax.imshow(arr, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		else:
			im = ax.imshow(np.zeros_like(arr, dtype=np.float32), origin="lower", cmap=cmap)
			plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
		ax.set_title(ttl)
		if wcs2d is not None:
			ax.set_xlabel("RA")
			ax.set_ylabel("Dec")
		else:
			ax.set_xlabel("x")
			ax.set_ylabel("y")
		if zoom_lim is not None:
			x0, x1, y0, y1 = zoom_lim
			ax.set_xlim(float(x0) - 0.5, float(x1) + 0.5)
			ax.set_ylim(float(y0) - 0.5, float(y1) + 0.5)
	fig.suptitle(f"Cube Fitting progress | pixels processed: {int(done_steps)}/{int(total_steps)}", y=0.98)
	plt.tight_layout()
	fig.savefig(out_png, dpi=180)
	plt.close(fig)
	info_path = os.path.splitext(out_png)[0] + ".json"
	info = {
		"title": f"Cube fitting parameter maps | pixels processed: {int(done_steps)}/{int(total_steps)}",
		"done_steps": int(done_steps),
		"total_steps": int(total_steps),
		"has_wcs": bool(wcs2d is not None),
	}
	if zoom_lim is not None:
		info["zoom_bbox"] = {
			"x0": int(zoom_lim[0]),
			"x1": int(zoom_lim[1]),
			"y0": int(zoom_lim[2]),
			"y1": int(zoom_lim[3]),
		}
	try:
		with open(info_path, "w", encoding="utf-8") as f:
			json.dump(info, f, ensure_ascii=False, indent=2)
	except Exception:
		pass


def _compute_display_limits(arr: np.ndarray, q_low: float = 1.0, q_high: float = 99.0):
	v = np.asarray(arr, dtype=np.float32)
	fin = np.isfinite(v)
	if not np.any(fin):
		return 0.0, 1.0
	vf = v[fin]
	vmin = float(np.nanpercentile(vf, float(q_low)))
	vmax = float(np.nanpercentile(vf, float(q_high)))
	if vmax <= vmin:
		vmax = vmin + 1e-6
	return vmin, vmax


def _append_cubefit_progress_log(log_txt_path: str, msg: str):
	if not log_txt_path:
		return
	try:
		ts = time.strftime("%Y-%m-%d %H:%M:%S")
		with open(log_txt_path, "a", encoding="utf-8") as f:
			f.write(f"[{ts}] {str(msg)}\n")
	except Exception:
		pass


def _load_resume_map2d(path: str, expected_shape: Tuple[int, int]) -> Optional[np.ndarray]:
	if (not path) or (not os.path.isfile(path)) or (fits is None):
		return None
	try:
		arr = np.asarray(fits.getdata(path), dtype=np.float32)
		if arr.ndim == 3:
			arr = arr[0]
		if arr.ndim != 2:
			return None
		if (int(arr.shape[0]), int(arr.shape[1])) != (int(expected_shape[0]), int(expected_shape[1])):
			return None
		arr = np.asarray(arr, dtype=np.float32)
		arr[~np.isfinite(arr)] = np.nan
		return arr
	except Exception:
		return None


def _build_region_mask_from_cfg(ny: int, nx: int, cfg: dict):
	mode = str(cfg.get("region_mode", "full")).strip().lower()
	if mode != "bbox":
		return np.ones((int(ny), int(nx)), dtype=bool), {
			"mode": "full",
			"x_min": 0,
			"x_max": int(nx) - 1,
			"y_min": 0,
			"y_max": int(ny) - 1,
			"center_x": int(nx // 2),
			"center_y": int(ny // 2),
		}

	x0 = int(cfg.get("region_x_min", 0))
	x1 = int(cfg.get("region_x_max", int(nx) - 1))
	y0 = int(cfg.get("region_y_min", 0))
	y1 = int(cfg.get("region_y_max", int(ny) - 1))

	x0 = int(max(0, min(int(nx) - 1, x0)))
	x1 = int(max(0, min(int(nx) - 1, x1)))
	y0 = int(max(0, min(int(ny) - 1, y0)))
	y1 = int(max(0, min(int(ny) - 1, y1)))

	if x1 < x0:
		x0, x1 = x1, x0
	if y1 < y0:
		y0, y1 = y1, y0

	mask = np.zeros((int(ny), int(nx)), dtype=bool)
	mask[int(y0):int(y1)+1, int(x0):int(x1)+1] = True
	meta = {
		"mode": "bbox",
		"x_min": int(x0),
		"x_max": int(x1),
		"y_min": int(y0),
		"y_max": int(y1),
		"center_x": int((x0 + x1) // 2),
		"center_y": int((y0 + y1) // 2),
	}
	return mask, meta


def run_cube_fit_worker(cfg_path: str) -> int:
	if fits is None:
		print("FITS backend not available")
		return 2
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	out_dir = str(cfg["out_dir"])
	os.makedirs(out_dir, exist_ok=True)
	obs_cube_paths_cfg = cfg.get("obs_cube_paths", None)
	if isinstance(obs_cube_paths_cfg, (list, tuple)):
		obs_cube_paths = [str(p).strip() for p in obs_cube_paths_cfg if str(p).strip()]
	else:
		obs_cube_paths = []
	if not obs_cube_paths:
		obs_cube_paths = [str(cfg.get("obs_cube_path", "")).strip()]
	obs_cube_paths = [p for p in obs_cube_paths if p]
	signal_models_source = str(cfg["signal_models_source"])
	noise_models_root = str(cfg["noise_models_root"])
	filter_file = str(cfg["filter_file"])
	target_freqs = [float(v) for v in cfg.get("target_freqs", [])]
	case_mode = str(cfg.get("case_mode", "synthetic_only"))
	fit_criterion = str(cfg.get("fit_criterion", "rmse"))
	global_weight_mode = str(cfg.get("global_weight_mode", "inverse_best_error"))
	global_search_mode = str(cfg.get("global_search_mode", "concatenated"))
	candidate_mode = str(cfg.get("candidate_mode", "random"))
	n_candidates = int(cfg.get("n_candidates", 300))
	ranges = dict(cfg.get("ranges", {}))
	noise_scale = float(cfg.get("noise_scale", 1.0))
	allow_nearest = bool(cfg.get("allow_nearest", True))
	seed = int(cfg.get("seed", 42))
	independent_pixel_candidates = bool(cfg.get("independent_pixel_candidates", False))
	local_optimizer_method = str(cfg.get("local_optimizer_method", "none"))
	local_optimizer_max_nfev = int(max(8, int(cfg.get("local_optimizer_max_nfev", 24))))
	progress_every = int(max(1, int(cfg.get("progress_every", 40))))
	spatial_stride = int(max(1, int(cfg.get("spatial_stride", 1))))
	obs_shift_enabled = bool(cfg.get("obs_shift_enabled", True))
	obs_shift_mode = str(cfg.get("obs_shift_mode", "per_frequency"))
	obs_shift_kms = float(cfg.get("obs_shift_kms", 0.0))
	out_prefix = str(cfg.get("out_prefix", "CUBEFIT"))
	resume_enabled = bool(cfg.get("resume_enabled", True))

	log_txt_path = os.path.join(out_dir, "Log.txt")
	state_json_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_STATE.json")
	done_mask_fits_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_DONE_MASK.fits")
	inprog_logn_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_LOGN.fits")
	inprog_tex_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_TEX.fits")
	inprog_velo_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_VELOCITY.fits")
	inprog_fwhm_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_FWHM.fits")
	inprog_obj_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_OBJECTIVE.fits")
	inprog_mae_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_MAE.fits")
	run_t0 = float(time.time())
	try:
		elapsed_accum_seconds = float(cfg.get("elapsed_accum_seconds", 0.0))
	except Exception:
		elapsed_accum_seconds = 0.0
	if (elapsed_accum_seconds <= 0.0) and bool(resume_enabled) and os.path.isfile(state_json_path):
		try:
			with open(state_json_path, "r", encoding="utf-8") as f_prev_state:
				_prev_state = json.load(f_prev_state)
			if isinstance(_prev_state, dict):
				elapsed_accum_seconds = float(_prev_state.get("elapsed_total_seconds", 0.0))
		except Exception:
			pass
	if not np.isfinite(elapsed_accum_seconds) or elapsed_accum_seconds < 0.0:
		elapsed_accum_seconds = 0.0

	if not obs_cube_paths:
		raise FileNotFoundError("No observational cube paths were provided.")

	cube_arrays: List[np.ndarray] = []
	cube_freqs: List[np.ndarray] = []
	ref_hdr = None
	ny = nx = None
	total_channels = 0
	integ_map_accum = None
	lastpixel_npz_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_LASTPIXEL_SPECTRA.npz")
	integ_map_fits_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_INTEG_MAP.fits")

	for i_c, cp in enumerate(obs_cube_paths, start=1):
		if not os.path.isfile(cp):
			raise FileNotFoundError(f"Observational cube not found: {cp}")
		with fits.open(cp, memmap=True) as hdul:
			arr_i = np.asarray(hdul[0].data, dtype=np.float32)
			hdr_i = hdul[0].header.copy()
		if arr_i.ndim == 4:
			arr_i = arr_i[0]
		if arr_i.ndim != 3:
			raise RuntimeError(f"Unexpected observational cube shape in '{cp}': {arr_i.shape}")

		nchan_i, ny_i, nx_i = int(arr_i.shape[0]), int(arr_i.shape[1]), int(arr_i.shape[2])
		if ny is None or nx is None:
			ny, nx = int(ny_i), int(nx_i)
			ref_hdr = hdr_i.copy()
		elif int(ny_i) != int(ny) or int(nx_i) != int(nx):
			raise RuntimeError(
				"All cubes must share the same spatial shape (ny, nx). "
				f"Got first=({ny},{nx}) vs '{cp}'=({ny_i},{nx_i})."
			)

		freq_i = _build_freq_axis_from_header(hdr_i, nchan_i)
		if obs_shift_enabled:
			if str(obs_shift_mode).strip().lower() == "spw_center":
				freq_i = _apply_velocity_shift_by_spw_center(freq_i, float(obs_shift_kms))
			else:
				freq_i = _apply_velocity_shift_to_frequency(freq_i, float(obs_shift_kms))

		cube_arrays.append(np.asarray(arr_i, dtype=np.float32))
		cube_freqs.append(np.asarray(freq_i, dtype=np.float64))
		map_i = np.asarray(np.nansum(np.where(np.isfinite(arr_i), arr_i, 0.0), axis=0), dtype=np.float64)
		if integ_map_accum is None:
			integ_map_accum = np.asarray(map_i, dtype=np.float64)
		elif np.asarray(integ_map_accum).shape == np.asarray(map_i).shape:
			integ_map_accum = np.asarray(integ_map_accum + map_i, dtype=np.float64)
		total_channels += int(nchan_i)
		_append_cubefit_progress_log(
			log_txt_path,
			f"Input cube {i_c}/{len(obs_cube_paths)} | path='{cp}' | shape=({nchan_i},{ny_i},{nx_i})",
		)

	if integ_map_accum is not None:
		try:
			_write_map_fits_2d(integ_map_fits_path, np.asarray(integ_map_accum, dtype=np.float32), ref_hdr, "Cube-fitting integrated intensity preview map")
		except Exception:
			pass

	_append_cubefit_progress_log(
		log_txt_path,
		f"Start cube fitting | n_cubes={len(obs_cube_paths)} | total_channels={int(total_channels)} | spatial_shape=({int(ny)},{int(nx)}) | resume_enabled={bool(resume_enabled)}",
	)
	region_mask, region_meta = _build_region_mask_from_cfg(ny, nx, cfg)
	_append_cubefit_progress_log(
		log_txt_path,
		f"Region mode={region_meta.get('mode')} | x=[{region_meta.get('x_min')},{region_meta.get('x_max')}] | y=[{region_meta.get('y_min')},{region_meta.get('y_max')}]",
	)
	obs_freq_concat = np.concatenate([np.asarray(ff, dtype=np.float64) for ff in cube_freqs], axis=0)
	sort_idx_freq = np.argsort(np.asarray(obs_freq_concat, dtype=np.float64))
	obs_freq_sorted = np.asarray(obs_freq_concat, dtype=np.float64)[sort_idx_freq]

	X_shared = None
	if not bool(independent_pixel_candidates):
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
	processed_mask = np.zeros((ny, nx), dtype=bool)

	valid_mask = np.zeros((int(ny), int(nx)), dtype=bool)
	for arr_i in cube_arrays:
		valid_mask |= np.any(np.isfinite(arr_i), axis=0)
	valid_mask = np.asarray(valid_mask, dtype=bool) & np.asarray(region_mask, dtype=bool)
	pixel_order = _spiral_pixel_order_valid(
		valid_mask,
		center_y=int(region_meta.get("center_y", ny // 2)),
		center_x=int(region_meta.get("center_x", nx // 2)),
	)
	if int(spatial_stride) > 1:
		pixel_order = [(yy, xx) for (yy, xx) in pixel_order if (int(yy) % int(spatial_stride) == 0) and (int(xx) % int(spatial_stride) == 0)]
	total_pixels = int(len(pixel_order))
	if total_pixels <= 0:
		raise RuntimeError("No valid observational pixels in cube")

	if bool(resume_enabled):
		dm = _load_resume_map2d(done_mask_fits_path, (ny, nx))
		if dm is not None:
			processed_mask = np.asarray(dm > 0.5, dtype=bool)
			m_logn = _load_resume_map2d(inprog_logn_path, (ny, nx))
			m_tex = _load_resume_map2d(inprog_tex_path, (ny, nx))
			m_vel = _load_resume_map2d(inprog_velo_path, (ny, nx))
			m_fwhm = _load_resume_map2d(inprog_fwhm_path, (ny, nx))
			m_obj = _load_resume_map2d(inprog_obj_path, (ny, nx))
			m_mae = _load_resume_map2d(inprog_mae_path, (ny, nx))
			if m_logn is not None:
				map_logn = m_logn
			if m_tex is not None:
				map_tex = m_tex
			if m_vel is not None:
				map_velo = m_vel
			if m_fwhm is not None:
				map_fwhm = m_fwhm
			if m_obj is not None:
				map_obj = m_obj
			if m_mae is not None:
				map_mae = m_mae
			done_prev = int(np.sum([1 for (yy, xx) in pixel_order if bool(processed_mask[yy, xx])]))
			_append_cubefit_progress_log(log_txt_path, f"Resume detected | previously processed pixels: {done_prev}/{total_pixels}")

	progress_png = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_MAP.png")
	fit_count = int(np.count_nonzero(np.isfinite(map_logn) & processed_mask))
	done_steps = int(np.sum([1 for (yy, xx) in pixel_order if bool(processed_mask[yy, xx])]))
	for p_done, (y, x) in enumerate(pixel_order, start=1):
		if bool(processed_mask[y, x]):
			continue
		processed_mask[y, x] = True
		done_steps += 1

		y_obs = np.concatenate([
			np.asarray(arr_i[:, y, x], dtype=np.float64)
			for arr_i in cube_arrays
		], axis=0)
		y_obs = np.asarray(y_obs, dtype=np.float64)[sort_idx_freq]
		if int(np.count_nonzero(np.isfinite(y_obs))) < 3:
			if (done_steps % progress_every) == 0 or done_steps == total_pixels:
				elapsed_total_seconds = float(elapsed_accum_seconds + max(0.0, float(time.time()) - run_t0))
				_write_map_fits_2d(inprog_logn_path, map_logn, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_write_map_fits_2d(inprog_tex_path, map_tex, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_write_map_fits_2d(inprog_velo_path, map_velo, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_write_map_fits_2d(inprog_fwhm_path, map_fwhm, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_write_map_fits_2d(inprog_obj_path, map_obj, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_write_map_fits_2d(inprog_mae_path, map_mae, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_write_done_mask_fits_2d(done_mask_fits_path, processed_mask, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
				_save_cubefit_progress_png(map_logn, map_tex, map_velo, map_fwhm, done_steps, total_pixels, progress_png, ref_hdr=ref_hdr, processed_mask=processed_mask)
				try:
					with open(state_json_path, "w", encoding="utf-8") as f_state:
						json.dump({
							"done_steps": int(done_steps),
							"total_steps": int(total_pixels),
							"fit_count": int(fit_count),
							"elapsed_total_seconds": float(elapsed_total_seconds),
							"elapsed_hms": _format_elapsed_hms(float(elapsed_total_seconds)),
							"last_pixel": [int(y), int(x)],
							"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
						}, f_state, ensure_ascii=False, indent=2)
				except Exception:
					pass
				_append_cubefit_progress_log(log_txt_path, f"Checkpoint {done_steps}/{total_pixels} | fit_count={fit_count}")
			continue

		if bool(independent_pixel_candidates):
			seed_px = int((int(seed) * 1000003 + int(y) * 10007 + int(x) * 10009) % 2147483647)
			X_fit = _sample_fit_candidates(
				n_samples=int(n_candidates),
				ranges=ranges,
				seed=int(seed_px),
				mode=str(candidate_mode),
			)
		else:
			seed_px = int(seed)
			X_fit = np.asarray(X_shared, dtype=np.float32)

		res = _run_roi_fitting(
			signal_models_source=signal_models_source,
			noise_models_root=noise_models_root,
			filter_file=filter_file,
			target_freqs=target_freqs,
			obs_freq=np.asarray(obs_freq_sorted, dtype=np.float64),
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
			seed=int(seed_px),
			x_candidates_override=np.asarray(X_fit, dtype=np.float32),
			noise_models_loaded_override=noise_models_shared,
			pkg_cache_override=pkg_cache_shared,
			local_optimizer_method=str(local_optimizer_method),
			local_optimizer_max_nfev=int(local_optimizer_max_nfev),
			refine_after_first_fit=False,
		)
		last_fit_ok = bool(isinstance(res, dict) and bool(res.get("ok", False)))
		if last_fit_ok:
			bp = res.get("best_global_params", {}) if isinstance(res.get("best_global_params", {}), dict) else {}
			map_logn[y, x] = float(bp.get("logN", np.nan))
			map_tex[y, x] = float(bp.get("Tex", np.nan))
			map_velo[y, x] = float(bp.get("Velocity", np.nan))
			map_fwhm[y, x] = float(bp.get("FWHM", np.nan))
			map_obj[y, x] = float(res.get("best_global_mean_objective", np.nan))
			map_mae[y, x] = float(res.get("best_global_mean_MAE", np.nan))
			fit_count += 1

		if (done_steps % progress_every) == 0 or done_steps == total_pixels:
			elapsed_total_seconds = float(elapsed_accum_seconds + max(0.0, float(time.time()) - run_t0))
			try:
				if last_fit_ok:
					overlay = res.get("global_overlay", []) if isinstance(res.get("global_overlay", []), list) else []
					freq_parts = []
					obs_parts = []
					syn_parts = []
					noise_parts = []
					pred_parts = []
					has_noise = False
					for seg in overlay:
						if not isinstance(seg, dict):
							continue
						ff = np.asarray(seg.get("freq", []), dtype=np.float64).reshape(-1)
						oo = np.asarray(seg.get("obs_interp", []), dtype=np.float64).reshape(-1)
						sy = np.asarray(seg.get("best_global_synthetic", []), dtype=np.float64).reshape(-1)
						pp = np.asarray(seg.get("best_global_pred", []), dtype=np.float64).reshape(-1)
						nn_raw = seg.get("best_global_noise", None)
						nn = np.asarray(nn_raw, dtype=np.float64).reshape(-1) if nn_raw is not None else np.asarray([], dtype=np.float64)
						if ff.size < 2:
							continue
						if oo.size != ff.size or sy.size != ff.size or pp.size != ff.size:
							continue
						if nn.size not in (0, ff.size):
							continue
						freq_parts.append(np.asarray(ff, dtype=np.float64))
						obs_parts.append(np.asarray(np.where(np.isfinite(oo), oo, 0.0), dtype=np.float64))
						syn_parts.append(np.asarray(np.where(np.isfinite(sy), sy, 0.0), dtype=np.float64))
						pred_parts.append(np.asarray(np.where(np.isfinite(pp), pp, 0.0), dtype=np.float64))
						if nn.size == ff.size:
							has_noise = True
							noise_parts.append(np.asarray(np.where(np.isfinite(nn), nn, 0.0), dtype=np.float64))
						else:
							noise_parts.append(np.zeros((ff.size,), dtype=np.float64))

					if freq_parts:
						freq_cat, obs_cat = concat_segments_with_zero_gaps(freq_parts, obs_parts, insert_zero_gaps=True)
						_, syn_cat = concat_segments_with_zero_gaps(freq_parts, syn_parts, insert_zero_gaps=True)
						_, pred_cat = concat_segments_with_zero_gaps(freq_parts, pred_parts, insert_zero_gaps=True)
						if has_noise:
							_, noise_cat = concat_segments_with_zero_gaps(freq_parts, noise_parts, insert_zero_gaps=True)
						else:
							noise_cat = np.asarray([], dtype=np.float64)
					else:
						freq_cat = np.asarray(obs_freq_sorted, dtype=np.float64)
						obs_cat = np.asarray(np.where(np.isfinite(y_obs), y_obs, 0.0), dtype=np.float64)
						syn_cat = np.asarray([], dtype=np.float64)
						noise_cat = np.asarray([], dtype=np.float64)
						pred_cat = np.asarray([], dtype=np.float64)
				else:
					freq_cat = np.asarray(obs_freq_sorted, dtype=np.float64)
					obs_cat = np.asarray(np.where(np.isfinite(y_obs), y_obs, 0.0), dtype=np.float64)
					syn_cat = np.asarray([], dtype=np.float64)
					noise_cat = np.asarray([], dtype=np.float64)
					pred_cat = np.asarray([], dtype=np.float64)

				np.savez_compressed(
					lastpixel_npz_path,
					x=np.asarray([int(x)], dtype=np.int32),
					y=np.asarray([int(y)], dtype=np.int32),
					done_steps=np.asarray([int(done_steps)], dtype=np.int32),
					total_steps=np.asarray([int(total_pixels)], dtype=np.int32),
					fit_ok=np.asarray([1 if bool(last_fit_ok) else 0], dtype=np.int32),
					freq=np.asarray(freq_cat, dtype=np.float32),
					obs=np.asarray(obs_cat, dtype=np.float32),
					syn=np.asarray(syn_cat, dtype=np.float32),
					noise=np.asarray(noise_cat, dtype=np.float32),
					pred=np.asarray(pred_cat, dtype=np.float32),
				)
			except Exception:
				pass

			_write_map_fits_2d(inprog_logn_path, map_logn, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_write_map_fits_2d(inprog_tex_path, map_tex, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_write_map_fits_2d(inprog_velo_path, map_velo, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_write_map_fits_2d(inprog_fwhm_path, map_fwhm, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_write_map_fits_2d(inprog_obj_path, map_obj, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_write_map_fits_2d(inprog_mae_path, map_mae, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_write_done_mask_fits_2d(done_mask_fits_path, processed_mask, ref_hdr, f"Checkpoint: {done_steps}/{total_pixels}")
			_save_cubefit_progress_png(map_logn, map_tex, map_velo, map_fwhm, done_steps, total_pixels, progress_png, ref_hdr=ref_hdr, processed_mask=processed_mask)
			try:
				with open(state_json_path, "w", encoding="utf-8") as f_state:
					json.dump({
						"done_steps": int(done_steps),
						"total_steps": int(total_pixels),
						"fit_count": int(fit_count),
						"elapsed_total_seconds": float(elapsed_total_seconds),
						"elapsed_hms": _format_elapsed_hms(float(elapsed_total_seconds)),
						"last_pixel": [int(y), int(x)],
						"region": dict(region_meta),
						"region": dict(region_meta),
						"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
					}, f_state, ensure_ascii=False, indent=2)
			except Exception:
				pass
			_append_cubefit_progress_log(log_txt_path, f"Checkpoint {done_steps}/{total_pixels} | fit_count={fit_count}")

	if fit_count <= 0:
		raise RuntimeError("Cube fitting produced no valid fitted pixels")

	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_LOGN.fits"), map_logn, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_TEX.fits"), map_tex, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_VELOCITY.fits"), map_velo, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_FWHM.fits"), map_fwhm, ref_hdr, "Final cube fitting map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_OBJECTIVE.fits"), map_obj, ref_hdr, "Final cube fitting objective map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_MAE.fits"), map_mae, ref_hdr, "Final cube fitting MAE map")
	_append_cubefit_progress_log(log_txt_path, f"Completed cube fitting | valid fitted pixels: {fit_count}/{total_pixels}")
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


def _format_elapsed_hms(seconds: float) -> str:
	try:
		t = int(max(0, round(float(seconds))))
	except Exception:
		t = 0
	h = t // 3600
	m = (t % 3600) // 60
	s = t % 60
	return f"{h:02d}:{m:02d}:{s:02d}"
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


def _show_worker_warnings(log_path: str, max_lines: int = 120):
	"""Render worker warnings in a consistent Streamlit block."""
	warns = _read_warn_lines(str(log_path or ""), max_lines=int(max_lines))
	if warns:
		st.warning("Failed target frequencies were detected. Check worker warnings for details.")
		with st.expander("Show worker warnings"):
			st.text("\n".join(warns))


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
		with fits.open(cube_fits_path, memmap=True, lazy_load_hdus=True) as hdul:
			hdr = hdul[0].header
			naxis = int(hdr.get("NAXIS", 0))
			nx = int(hdr.get("NAXIS1", 0))
			ny = int(hdr.get("NAXIS2", 0))
			if naxis >= 3 and nx > 0 and ny > 0:
				return int(ny), int(nx)
			arr = hdul[0].data
			if arr is not None:
				shape = tuple(arr.shape)
				if len(shape) == 3:
					return int(shape[1]), int(shape[2])
				if len(shape) == 4:
					return int(shape[2]), int(shape[3])
			return None
	except Exception:
		return None


def _build_freq_axis_from_header(hdr, nchan: int) -> np.ndarray:
	try:
		crval = float(hdr.get("CRVAL3", 0.0))
		cdelt = float(hdr.get("CDELT3", 1.0))
		crpix = float(hdr.get("CRPIX3", 1.0))
		cunit = str(hdr.get("CUNIT3", "Hz")).strip().lower()
		idx = np.arange(int(nchan), dtype=np.float64)
		f = crval + (idx + 1.0 - crpix) * cdelt
		if "ghz" in cunit:
			freq_ghz = f
		elif "mhz" in cunit:
			freq_ghz = f / 1e3
		elif "khz" in cunit:
			freq_ghz = f / 1e6
		elif "hz" in cunit:
			freq_ghz = f / 1e9
		else:
			# Conservative fallback: most ALMA cubes store spectral axis in Hz.
			freq_ghz = f / 1e9
		return np.asarray(freq_ghz, dtype=np.float64)
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


class _InverseParamNNLite(nn.Module):
	def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 4, dropout_rate: float = 0.2):
		super().__init__()
		layers = []
		prev = int(input_size)
		for h in (hidden_sizes or []):
			hh = int(max(8, int(h)))
			layers.append(nn.Linear(prev, hh))
			layers.append(nn.BatchNorm1d(hh))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(float(dropout_rate)))
			prev = hh
		layers.append(nn.Linear(prev, int(output_size)))
		self.model = nn.Sequential(*layers)

	def forward(self, x):
		return self.model(x)


def _load_standard_scaler_from_h5_group(grp) -> StandardScaler:
	sc = StandardScaler()
	sc.mean_ = np.asarray(grp["mean_"], dtype=np.float64)
	sc.scale_ = np.asarray(grp["scale_"], dtype=np.float64)
	sc.var_ = np.asarray(grp.get("var_", sc.scale_ ** 2), dtype=np.float64)
	nfi = np.asarray(grp.get("n_features_in_", np.array([sc.mean_.shape[0]], dtype=np.int64)))
	sc.n_features_in_ = int(nfi[0]) if np.ndim(nfi) > 0 else int(nfi)
	return sc


def _normalize_inverse_state_dict_keys(sd: "collections.OrderedDict") -> "collections.OrderedDict":
	"""Normalize checkpoint key prefixes so loader accepts both net.* and model.* formats."""
	if not sd:
		return sd
	keys = list(sd.keys())
	has_model = any(str(k).startswith("model.") for k in keys)
	has_net = any(str(k).startswith("net.") for k in keys)
	if has_model and (not has_net):
		return sd
	if has_net and (not has_model):
		sd2 = collections.OrderedDict()
		for k, v in sd.items():
			kk = str(k)
			if kk.startswith("net."):
				kk = "model." + kk[len("net."):]
			sd2[kk] = v
		return sd2
	return sd


def _load_inverse_param_model_h5(model_h5_path: str):
	with h5py.File(model_h5_path, "r") as hf:
		cfg_raw = hf["config_json"][()]
		cfg_json = cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray)) else str(cfg_raw)
		cfg = json.loads(cfg_json)
		roi_freq_axis = None
		if "roi_freq_axis_ghz" in hf:
			try:
				roi_freq_axis = np.asarray(hf["roi_freq_axis_ghz"], dtype=np.float64).reshape(-1)
			except Exception:
				roi_freq_axis = None

		input_size = int(cfg.get("input_size", 0))
		hidden_sizes = [int(v) for v in list(cfg.get("hidden_sizes", [256, 128, 64]))]
		output_size = int(cfg.get("output_size", 4))
		dropout_rate = float(cfg.get("dropout_rate", 0.2))

		model = _InverseParamNNLite(
			input_size=input_size,
			hidden_sizes=hidden_sizes,
			output_size=output_size,
			dropout_rate=dropout_rate,
		).cpu()

		sd = collections.OrderedDict()
		for k in hf["state_dict"].keys():
			sd[k] = torch.from_numpy(np.asarray(hf["state_dict"][k]))
		sd = _normalize_inverse_state_dict_keys(sd)
		model.load_state_dict(sd)
		model.eval()

		scaler_x = _load_standard_scaler_from_h5_group(hf["scaler_x"])
		scaler_y = _load_standard_scaler_from_h5_group(hf["scaler_y"])

	return model, scaler_x, scaler_y, cfg, roi_freq_axis


def _load_inverse_param_models_cached(models_root: str):
	root = str(models_root or "").strip()
	if (not root) or (not os.path.isdir(root)):
		return [], [f"Inverse models directory not found: {root}"]

	# Flexible discovery: accept any .h5 inside a "Model" folder.
	# Some runs may not use the exact filename "final_inverse_model.h5".
	paths_primary = sorted(glob.glob(os.path.join(root, "**", "Model", "*.h5"), recursive=True))
	paths_fallback = sorted(glob.glob(os.path.join(root, "**", "final_inverse_model*.h5"), recursive=True))
	paths = sorted({str(p) for p in (paths_primary + paths_fallback) if os.path.isfile(p)})
	if not paths:
		return [], [f"No inverse model files found under: {root}"]

	entries: List[dict] = []
	warnings: List[str] = []
	for p in paths:
		try:
			m, sx, sy, cfg, roi_freq_axis = _load_inverse_param_model_h5(p)
			lo = float(cfg.get("roi_f_min_ghz", np.nan))
			hi = float(cfg.get("roi_f_max_ghz", np.nan))

			# Fallback: infer ROI bounds from roi_name / folder name pattern *_fLO-HIGHz
			if (not np.isfinite(lo)) or (not np.isfinite(hi)):
				roi_name_cfg = str(cfg.get("roi_name", "")).strip()
				lo2, hi2 = parse_roi_freq_bounds_from_dirname(roi_name_cfg)
				if (lo2 is None) or (hi2 is None):
					folder_roi_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
					lo2, hi2 = parse_roi_freq_bounds_from_dirname(folder_roi_name)
				if (lo2 is not None) and (hi2 is not None):
					lo, hi = float(lo2), float(hi2)

			if not np.isfinite(lo) or not np.isfinite(hi):
				warnings.append(f"Skipping model without finite ROI bounds: {p}")
				continue
			if hi < lo:
				lo, hi = hi, lo
			entries.append(
				{
					"path": str(p),
					"roi_name": str(cfg.get("roi_name", os.path.basename(os.path.dirname(os.path.dirname(p))))),
					"f_min_ghz": float(lo),
					"f_max_ghz": float(hi),
					"input_size": int(cfg.get("input_size", 0)),
					"collapse_any": bool(cfg.get("collapse_any", False)),
					"improvement_vs_baseline_pct": float(cfg.get("improvement_vs_baseline_pct", 0.0)),
					"model": m,
					"scaler_x": sx,
					"scaler_y": sy,
					"cfg": cfg,
					"roi_freq_axis_ghz": None if roi_freq_axis is None else np.asarray(roi_freq_axis, dtype=np.float64),
				}
			)
		except Exception as e:
			warnings.append(f"Could not load inverse model {p}: {e}")

	entries = sorted(entries, key=lambda d: (float(d.get("f_min_ghz", np.inf)), float(d.get("f_max_ghz", np.inf))))
	if not entries:
		warnings.append("No valid inverse models could be loaded.")
	return entries, warnings


def _prepare_inverse_input_segment(
	f_obs: np.ndarray,
	y_obs: np.ndarray,
	roi_lo_ghz: float,
	roi_hi_ghz: float,
	n_in: int,
	roi_freq_axis_ghz: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], int]:
	"""Build model input segment for an ROI using frequency-aware interpolation when possible.

	Returns (segment, n_overlap_channels). segment is float64 shape (n_in,) or None.
	"""
	f = np.asarray(f_obs, dtype=np.float64).reshape(-1)
	y = np.asarray(y_obs, dtype=np.float64).reshape(-1)
	if f.size != y.size or f.size <= 1 or n_in <= 0:
		return None, 0

	valid = np.isfinite(f) & np.isfinite(y)
	f = f[valid]
	y = y[valid]
	if f.size <= 1:
		return None, 0

	order = np.argsort(f)
	f = f[order]
	y = y[order]
	fu, idx_u = np.unique(f, return_index=True)
	yu = y[idx_u]
	if fu.size <= 1:
		return None, 0

	lo = float(min(float(roi_lo_ghz), float(roi_hi_ghz)))
	hi = float(max(float(roi_lo_ghz), float(roi_hi_ghz)))
	idx_overlap = np.where((fu >= lo) & (fu <= hi))[0]
	n_overlap = int(idx_overlap.size)

	rf = None if roi_freq_axis_ghz is None else np.asarray(roi_freq_axis_ghz, dtype=np.float64).reshape(-1)
	if (rf is not None) and (rf.size >= 2) and np.all(np.isfinite(rf)):
		rf_ord = np.sort(rf)
		seg = np.interp(rf_ord, fu, yu, left=np.nan, right=np.nan)
		if np.any(np.isfinite(seg)):
			if not np.all(np.isfinite(seg)):
				first = int(np.where(np.isfinite(seg))[0][0])
				last = int(np.where(np.isfinite(seg))[0][-1])
				seg[:first] = seg[first]
				seg[last + 1:] = seg[last]
				nan_mid = ~np.isfinite(seg)
				if np.any(nan_mid):
					seg[nan_mid] = float(np.nanmedian(seg[np.isfinite(seg)]))
			if seg.size != int(n_in):
				seg = _resample_1d_by_index_float64(seg, int(n_in))
			return np.asarray(seg, dtype=np.float64).reshape(-1), n_overlap

	# Fallback legacy behavior when model has no stored ROI axis
	idx = np.where((f >= lo) & (f <= hi))[0]
	if idx.size <= 1:
		return None, n_overlap
	seg = np.asarray(y[idx], dtype=np.float64)
	if seg.size != int(n_in):
		seg = _resample_1d_by_index_float64(seg, int(n_in))
	return np.asarray(seg, dtype=np.float64).reshape(-1), n_overlap


def _get_inverse_bounds_from_cfg(cfg: dict) -> Tuple[np.ndarray, np.ndarray]:
	"""Return low/high bounds for [logn, tex, fwhm, velo]."""
	# Safe fallback bounds to avoid extreme out-of-distribution values.
	lo = np.asarray([8.0, 5.0, 0.1, -300.0], dtype=np.float64)
	hi = np.asarray([22.0, 800.0, 30.0, 300.0], dtype=np.float64)
	if not isinstance(cfg, dict):
		return lo, hi

	rng = cfg.get("target_train_ranges", None)
	if isinstance(rng, dict):
		order = ["logn", "tex", "fwhm", "velo"]
		for i, nm in enumerate(order):
			it = rng.get(str(nm), None)
			if not isinstance(it, dict):
				continue
			try:
				mn = float(it.get("min", np.nan))
				mx = float(it.get("max", np.nan))
			except Exception:
				continue
			if np.isfinite(mn) and np.isfinite(mx) and mx > mn:
				pad = 0.10 * float(mx - mn)
				lo[i] = float(mn - pad)
				hi[i] = float(mx + pad)

	# Ensure valid interval after overrides.
	for i in range(4):
		if not (np.isfinite(lo[i]) and np.isfinite(hi[i])) or (hi[i] <= lo[i]):
			lo[i], hi[i] = float(lo[i]), float(max(lo[i] + 1.0, hi[i]))
	return lo, hi


def _predict_inverse_params_from_models(freq_ghz: np.ndarray, intensity: np.ndarray, inverse_models: List[dict], min_overlap_channels: int = 8):
	f = np.asarray(freq_ghz, dtype=np.float64).reshape(-1)
	y = np.asarray(intensity, dtype=np.float64).reshape(-1)
	if f.size != y.size or f.size == 0:
		return [], None

	rows: List[dict] = []
	for ent in (inverse_models or []):
		lo = float(ent.get("f_min_ghz", np.nan))
		hi = float(ent.get("f_max_ghz", np.nan))
		if (not np.isfinite(lo)) or (not np.isfinite(hi)):
			continue
		n_in = int(ent.get("input_size", 0))
		if n_in <= 0:
			continue
		seg, n_overlap = _prepare_inverse_input_segment(
			f_obs=f,
			y_obs=y,
			roi_lo_ghz=float(lo),
			roi_hi_ghz=float(hi),
			n_in=int(n_in),
			roi_freq_axis_ghz=ent.get("roi_freq_axis_ghz", None),
		)
		if seg is None or int(n_overlap) < int(max(4, min_overlap_channels)):
			continue
		if not np.any(np.isfinite(seg)):
			continue
		seg = np.where(np.isfinite(seg), seg, np.nanmedian(seg[np.isfinite(seg)]) if np.any(np.isfinite(seg)) else 0.0)
		if seg.size != int(n_in):
			seg = _resample_1d_by_index_float64(seg, int(n_in))

		sx = ent.get("scaler_x", None)
		sy = ent.get("scaler_y", None)
		mdl = ent.get("model", None)
		if sx is None or sy is None or mdl is None:
			continue

		try:
			x = np.asarray(seg, dtype=np.float32).reshape(1, -1)
			xn = sx.transform(x).astype(np.float32)
			with torch.no_grad():
				pred_s = mdl(torch.from_numpy(xn)).cpu().numpy().astype(np.float64)
			pred = sy.inverse_transform(pred_s).reshape(-1)
			if pred.size < 4:
				continue
			cfg = ent.get("cfg", {}) if isinstance(ent.get("cfg", {}), dict) else {}
			b_lo, b_hi = _get_inverse_bounds_from_cfg(cfg)
			pred_clipped = np.clip(np.asarray(pred[:4], dtype=np.float64), b_lo, b_hi)
			n_clipped = int(np.count_nonzero(np.abs(pred_clipped - np.asarray(pred[:4], dtype=np.float64)) > 1e-12))
			imp = float(ent.get("improvement_vs_baseline_pct", 0.0))
			w = float(max(1e-6, imp + 1.0))
			if bool(ent.get("collapse_any", False)):
				w *= 0.25
			if n_clipped > 0:
				w *= float(0.7 ** n_clipped)
			rows.append(
				{
					"roi_name": str(ent.get("roi_name", "")),
					"roi_f_min_ghz": float(lo),
					"roi_f_max_ghz": float(hi),
					"n_overlap_channels": int(n_overlap),
					"collapse_any": bool(ent.get("collapse_any", False)),
					"improvement_vs_baseline_pct": float(imp),
					"n_clipped_dims": int(n_clipped),
					"weight": float(w),
					"pred_logn": float(pred_clipped[0]),
					"pred_tex": float(pred_clipped[1]),
					"pred_fwhm": float(pred_clipped[2]),
					"pred_velo": float(pred_clipped[3]),
				}
			)
		except Exception:
			continue

	if not rows:
		return rows, None

	arr = np.asarray([[r["pred_logn"], r["pred_tex"], r["pred_fwhm"], r["pred_velo"]] for r in rows], dtype=np.float64)
	w = np.asarray([max(1e-9, float(r.get("weight", 1.0))) for r in rows], dtype=np.float64)

	# Robust row-level outlier filtering before final aggregation.
	outlier_removed = 0
	if arr.shape[0] >= 5:
		med = np.median(arr, axis=0)
		mad = np.median(np.abs(arr - med.reshape(1, -1)), axis=0)
		scale = np.maximum(1e-6, 1.4826 * mad)
		z = np.abs(arr - med.reshape(1, -1)) / scale.reshape(1, -1)
		keep = ~np.any(z > 6.0, axis=1)
		if int(np.count_nonzero(keep)) >= int(max(2, round(0.4 * arr.shape[0]))):
			outlier_removed = int(arr.shape[0] - np.count_nonzero(keep))
			arr = arr[keep]
			w = w[keep]

	w = w / max(1e-12, float(np.sum(w)))

	weighted = np.sum(arr * w.reshape(-1, 1), axis=0)
	median = np.median(arr, axis=0)
	best_idx = int(np.argmax(np.asarray([float(r.get("weight", 0.0)) for r in rows], dtype=np.float64)))

	summary = {
		"n_rois_used": int(arr.shape[0]),
		"n_rois_total_before_filter": int(len(rows)),
		"n_outlier_rows_removed": int(outlier_removed),
		"n_clipped_rows": int(np.count_nonzero(np.asarray([int(r.get("n_clipped_dims", 0)) for r in rows], dtype=np.int64) > 0)),
		"weighted_logn": float(weighted[0]),
		"weighted_tex": float(weighted[1]),
		"weighted_fwhm": float(weighted[2]),
		"weighted_velo": float(weighted[3]),
		"median_logn": float(median[0]),
		"median_tex": float(median[1]),
		"median_fwhm": float(median[2]),
		"median_velo": float(median[3]),
		"best_roi_name": str(rows[best_idx].get("roi_name", "")),
	}
	return rows, summary


def _decode_freq_token_customroi(tok: str) -> Optional[float]:
	t = str(tok or "").strip().lower()
	if not t:
		return None
	neg = False
	if t.startswith("m"):
		neg = True
		t = t[1:]
	t = t.replace("p", ".")
	try:
		v = float(t)
		return float(-v if neg else v)
	except Exception:
		return None


def _parse_customroi_freq_bounds_from_name(name: str) -> Tuple[Optional[float], Optional[float]]:
	s = str(name or "").strip()
	m = re.search(r"_f([mp0-9]+)to([mp0-9]+)ghz", s, flags=re.IGNORECASE)
	if m is None:
		return None, None
	lo = _decode_freq_token_customroi(str(m.group(1)))
	hi = _decode_freq_token_customroi(str(m.group(2)))
	if (lo is None) or (hi is None):
		return None, None
	return float(min(lo, hi)), float(max(lo, hi))


def _load_synthdb_roi_model_h5(model_h5_path: str):
	if h5py is None:
		raise RuntimeError("h5py is required to load inverse cube models")
	with h5py.File(model_h5_path, "r") as hf:
		cfg_raw = hf["config_json"][()]
		cfg_json = cfg_raw.decode("utf-8") if isinstance(cfg_raw, (bytes, bytearray)) else str(cfg_raw)
		cfg = json.loads(cfg_json)

		input_size = int(cfg.get("input_size", 0))
		hidden_sizes = [int(v) for v in list(cfg.get("hidden_sizes", [512, 256, 128]))]
		target_columns = [str(v).strip().lower() for v in list(cfg.get("target_columns", ["logn", "tex", "velo", "fwhm"]))]
		output_size = int(max(1, len(target_columns)))
		dropout_rate = float(cfg.get("dropout_rate", 0.2))

		model = _InverseParamNNLite(
			input_size=input_size,
			hidden_sizes=hidden_sizes,
			output_size=output_size,
			dropout_rate=dropout_rate,
		).cpu()

		sd = collections.OrderedDict()
		for k in hf["state_dict"].keys():
			sd[k] = torch.from_numpy(np.asarray(hf["state_dict"][k]))
		sd = _normalize_inverse_state_dict_keys(sd)
		model.load_state_dict(sd)
		model.eval()

		scaler_y = _load_standard_scaler_from_h5_group(hf["scaler"])

	return model, scaler_y, cfg, target_columns


def _load_synthdb_roi_models_cached(models_root: str):
	root = str(models_root or "").strip()
	if (not root) or (not os.path.isdir(root)):
		return [], [f"Synthetic ROI models directory not found: {root}"]

	paths = sorted(glob.glob(os.path.join(root, "**", "models", "final_model.h5"), recursive=True))
	paths = [str(p) for p in paths if os.path.isfile(p)]
	if not paths:
		return [], [f"No final_model.h5 files found under: {root}"]

	entries: List[dict] = []
	warnings: List[str] = []
	for p in paths:
		try:
			model, scaler_y, cfg, target_columns = _load_synthdb_roi_model_h5(p)
			roi_dir_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
			lo, hi = _parse_customroi_freq_bounds_from_name(roi_dir_name)
			if (lo is None) or (hi is None):
				lo2, hi2 = parse_roi_freq_bounds_from_dirname(roi_dir_name)
				if (lo2 is not None) and (hi2 is not None):
					lo, hi = float(lo2), float(hi2)
			if (lo is None) or (hi is None):
				warnings.append(f"Skipping model without ROI frequency bounds in folder name: {p}")
				continue

			entries.append(
				{
					"path": str(p),
					"roi_name": str(roi_dir_name),
					"f_min_ghz": float(min(lo, hi)),
					"f_max_ghz": float(max(lo, hi)),
					"input_size": int(cfg.get("input_size", 0)),
					"target_columns": [str(v).strip().lower() for v in target_columns],
					"model": model,
					"scaler_y": scaler_y,
				}
			)
		except Exception as e:
			warnings.append(f"Could not load synthetic ROI model {p}: {e}")

	entries = sorted(entries, key=lambda d: (float(d.get("f_min_ghz", np.inf)), float(d.get("f_max_ghz", np.inf))))
	if not entries:
		warnings.append("No valid synthetic ROI models could be loaded.")
	return entries, warnings


def _select_synthdb_models_by_guides(models: List[dict], guide_freqs_ghz: List[float], allow_nearest: bool = True) -> List[dict]:
	mods = [m for m in (models or []) if isinstance(m, dict)]
	freqs = [float(v) for v in (guide_freqs_ghz or []) if np.isfinite(float(v))]
	if not mods or not freqs:
		return []

	selected: List[dict] = []
	selected_paths = set()
	for g in freqs:
		matches = []
		for m in mods:
			lo = float(m.get("f_min_ghz", np.nan))
			hi = float(m.get("f_max_ghz", np.nan))
			if np.isfinite(lo) and np.isfinite(hi) and (min(lo, hi) <= float(g) <= max(lo, hi)):
				matches.append(m)
		if matches:
			for m in matches:
				p = str(m.get("path", ""))
				if p not in selected_paths:
					selected.append(m)
					selected_paths.add(p)
			continue

		if bool(allow_nearest):
			best = None
			for m in mods:
				lo = float(m.get("f_min_ghz", np.nan))
				hi = float(m.get("f_max_ghz", np.nan))
				if not (np.isfinite(lo) and np.isfinite(hi)):
					continue
				c = 0.5 * (float(lo) + float(hi))
				d = abs(float(c) - float(g))
				if (best is None) or (d < best[0]):
					best = (d, m)
			if best is not None:
				m = best[1]
				p = str(m.get("path", ""))
				if p not in selected_paths:
					selected.append(m)
					selected_paths.add(p)

	return selected


def _predict_pixel_params_from_synthdb_models(
	obs_freq_ghz: np.ndarray,
	obs_intensity: np.ndarray,
	models: List[dict],
	min_overlap_channels: int = 2,
):
	f = np.asarray(obs_freq_ghz, dtype=np.float64).reshape(-1)
	y = np.asarray(obs_intensity, dtype=np.float64).reshape(-1)
	if f.size != y.size or f.size <= 1:
		return None

	rows = []
	for ent in (models or []):
		lo = float(ent.get("f_min_ghz", np.nan))
		hi = float(ent.get("f_max_ghz", np.nan))
		n_in = int(ent.get("input_size", 0))
		if (not np.isfinite(lo)) or (not np.isfinite(hi)) or n_in <= 1:
			continue
		seg, n_overlap = _prepare_inverse_input_segment(
			f_obs=f,
			y_obs=y,
			roi_lo_ghz=float(lo),
			roi_hi_ghz=float(hi),
			n_in=int(n_in),
			roi_freq_axis_ghz=None,
		)
		if (seg is None) or (int(n_overlap) < int(max(1, min_overlap_channels))):
			continue

		x = np.asarray(seg, dtype=np.float32).reshape(1, -1)
		mn = np.nanmin(x, axis=1, keepdims=True)
		mx = np.nanmax(x, axis=1, keepdims=True)
		den = np.maximum(mx - mn, 1e-8)
		xn = np.where(np.isfinite(x), (x - mn) / den, 0.0).astype(np.float32)

		mdl = ent.get("model", None)
		sy = ent.get("scaler_y", None)
		tcols = [str(v).strip().lower() for v in list(ent.get("target_columns", []))]
		if mdl is None or sy is None or (not tcols):
			continue

		try:
			with torch.no_grad():
				pred_s = mdl(torch.from_numpy(xn)).cpu().numpy().astype(np.float64)
			pred = sy.inverse_transform(pred_s).reshape(-1)
			idx = {k: i for i, k in enumerate(tcols)}
			if not all(k in idx for k in ["logn", "tex", "velo", "fwhm"]):
				continue
			rows.append(
				{
					"n_overlap": int(n_overlap),
					"pred_logn": float(pred[int(idx["logn"])]),
					"pred_tex": float(pred[int(idx["tex"])]),
					"pred_velo": float(pred[int(idx["velo"])]),
					"pred_fwhm": float(pred[int(idx["fwhm"])]),
				}
			)
		except Exception:
			continue

	if not rows:
		return None

	arr = np.asarray([[r["pred_logn"], r["pred_tex"], r["pred_velo"], r["pred_fwhm"]] for r in rows], dtype=np.float64)
	w = np.asarray([max(1.0, float(r.get("n_overlap", 1))) for r in rows], dtype=np.float64)
	w = w / max(1e-12, float(np.sum(w)))
	mu = np.sum(arr * w.reshape(-1, 1), axis=0)
	sig = np.sqrt(np.sum(((arr - mu.reshape(1, -1)) ** 2) * w.reshape(-1, 1), axis=0))

	return {
		"logn": float(mu[0]),
		"tex": float(mu[1]),
		"velo": float(mu[2]),
		"fwhm": float(mu[3]),
		"objective": float(np.nanmean(sig)),
		"mae_proxy": float(np.nanmean(np.abs(arr - mu.reshape(1, -1)))),
		"n_models_used": int(arr.shape[0]),
	}


def run_inverse_cube_pred_worker(cfg_path: str) -> int:
	if fits is None:
		raise RuntimeError("astropy is required for inverse cube prediction")
	if h5py is None:
		raise RuntimeError("h5py is required for inverse cube prediction")
	with open(cfg_path, "r", encoding="utf-8") as f:
		cfg = json.load(f)

	out_dir = str(cfg["out_dir"])
	os.makedirs(out_dir, exist_ok=True)
	obs_cube_path = str(cfg["obs_cube_path"])
	inverse_models_root = str(cfg["inverse_models_root"])
	use_all_models = bool(cfg.get("use_all_models", True))
	target_freqs = [float(v) for v in cfg.get("target_freqs", [])]
	allow_nearest = bool(cfg.get("allow_nearest", True))
	progress_every = int(max(1, int(cfg.get("progress_every", 40))))
	spatial_stride = int(max(1, int(cfg.get("spatial_stride", 1))))
	obs_shift_enabled = bool(cfg.get("obs_shift_enabled", True))
	obs_shift_mode = str(cfg.get("obs_shift_mode", "per_frequency"))
	obs_shift_kms = float(cfg.get("obs_shift_kms", 0.0))
	resume_enabled = bool(cfg.get("resume_enabled", True))
	min_overlap_channels = int(max(1, int(cfg.get("min_overlap_channels", 2))))
	out_prefix = str(cfg.get("out_prefix", "INVCUBEPRED"))

	log_txt_path = os.path.join(out_dir, "Log.txt")
	state_json_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_STATE.json")
	done_mask_fits_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_DONE_MASK.fits")
	inprog_logn_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_LOGN.fits")
	inprog_tex_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_TEX.fits")
	inprog_velo_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_VELOCITY.fits")
	inprog_fwhm_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_FWHM.fits")
	inprog_obj_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_OBJECTIVE.fits")
	inprog_mae_path = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_MAE.fits")

	if not os.path.isfile(obs_cube_path):
		raise RuntimeError(f"Observational cube not found: {obs_cube_path}")

	with fits.open(obs_cube_path, memmap=True) as hdul:
		arr = np.asarray(hdul[0].data, dtype=np.float32)
		ref_hdr = hdul[0].header.copy()
	if arr.ndim == 4:
		arr = arr[0]
	if arr.ndim != 3:
		raise RuntimeError(f"Unsupported cube shape: {arr.shape}")
	nchan, ny, nx = int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2])

	_append_cubefit_progress_log(log_txt_path, f"Start inverse cube prediction | cube='{obs_cube_path}' | shape=({nchan},{ny},{nx}) | resume_enabled={bool(resume_enabled)}")
	region_mask, region_meta = _build_region_mask_from_cfg(ny, nx, cfg)
	_append_cubefit_progress_log(
		log_txt_path,
		f"Region mode={region_meta.get('mode')} | x=[{region_meta.get('x_min')},{region_meta.get('x_max')}] | y=[{region_meta.get('y_min')},{region_meta.get('y_max')}]",
	)

	obs_freq = _build_freq_axis_from_header(ref_hdr, nchan)
	if obs_shift_enabled:
		if str(obs_shift_mode).strip().lower() == "spw_center":
			obs_freq = _apply_velocity_shift_by_spw_center(obs_freq, float(obs_shift_kms))
		else:
			obs_freq = _apply_velocity_shift_to_frequency(obs_freq, float(obs_shift_kms))

	models_all, load_warns = _load_synthdb_roi_models_cached(str(inverse_models_root))
	for w in (load_warns or []):
		_append_cubefit_progress_log(log_txt_path, f"[WARN] {w}")
	if not models_all:
		raise RuntimeError("No valid synthetic ROI models available")

	if bool(use_all_models):
		models_sel = list(models_all)
		_append_cubefit_progress_log(log_txt_path, f"Using all ROI models (script4 mode): {len(models_sel)}/{len(models_all)}")
	else:
		models_sel = _select_synthdb_models_by_guides(models_all, target_freqs, allow_nearest=bool(allow_nearest))
		if not models_sel:
			raise RuntimeError("No ROI models selected by Guide frequencies")
		_append_cubefit_progress_log(log_txt_path, f"Selected ROI models by guides: {len(models_sel)}/{len(models_all)}")

	map_logn = np.full((ny, nx), np.nan, dtype=np.float32)
	map_tex = np.full((ny, nx), np.nan, dtype=np.float32)
	map_velo = np.full((ny, nx), np.nan, dtype=np.float32)
	map_fwhm = np.full((ny, nx), np.nan, dtype=np.float32)
	map_obj = np.full((ny, nx), np.nan, dtype=np.float32)
	map_mae = np.full((ny, nx), np.nan, dtype=np.float32)
	processed_mask = np.zeros((ny, nx), dtype=bool)

	valid_mask = np.any(np.isfinite(arr), axis=0)
	valid_mask = np.asarray(valid_mask, dtype=bool) & np.asarray(region_mask, dtype=bool)
	pixel_order = _spiral_pixel_order_valid(
		valid_mask,
		center_y=int(region_meta.get("center_y", ny // 2)),
		center_x=int(region_meta.get("center_x", nx // 2)),
	)
	if int(spatial_stride) > 1:
		pixel_order = pixel_order[:: int(spatial_stride)]
	total_pixels = int(len(pixel_order))
	if total_pixels <= 0:
		raise RuntimeError("No valid pixels selected for inverse cube prediction")

	if bool(resume_enabled):
		rm = _load_resume_map2d(inprog_logn_path, (ny, nx))
		if rm is not None:
			map_logn = rm
		rm = _load_resume_map2d(inprog_tex_path, (ny, nx))
		if rm is not None:
			map_tex = rm
		rm = _load_resume_map2d(inprog_velo_path, (ny, nx))
		if rm is not None:
			map_velo = rm
		rm = _load_resume_map2d(inprog_fwhm_path, (ny, nx))
		if rm is not None:
			map_fwhm = rm
		rm = _load_resume_map2d(inprog_obj_path, (ny, nx))
		if rm is not None:
			map_obj = rm
		rm = _load_resume_map2d(inprog_mae_path, (ny, nx))
		if rm is not None:
			map_mae = rm
		if os.path.isfile(done_mask_fits_path):
			try:
				dm = np.asarray(fits.getdata(done_mask_fits_path), dtype=np.float32)
				if dm.ndim == 3:
					dm = dm[0]
				if dm.shape == (ny, nx):
					processed_mask = np.asarray(dm > 0.5, dtype=bool)
			except Exception:
				pass

	progress_png = os.path.join(out_dir, f"{out_prefix}_INPROGRESS_MAP.png")
	fit_count = int(np.count_nonzero(np.isfinite(map_logn) & processed_mask))
	done_steps = int(np.sum([1 for (yy, xx) in pixel_order if bool(processed_mask[yy, xx])]))
	for p_done, (y, x) in enumerate(pixel_order, start=1):
		if bool(processed_mask[y, x]):
			continue

		y_obs = np.asarray(arr[:, y, x], dtype=np.float64)
		pred = _predict_pixel_params_from_synthdb_models(
			obs_freq_ghz=np.asarray(obs_freq, dtype=np.float64),
			obs_intensity=np.asarray(y_obs, dtype=np.float64),
			models=models_sel,
			min_overlap_channels=int(min_overlap_channels),
		)
		if isinstance(pred, dict):
			map_logn[y, x] = float(pred.get("logn", np.nan))
			map_tex[y, x] = float(pred.get("tex", np.nan))
			map_velo[y, x] = float(pred.get("velo", np.nan))
			map_fwhm[y, x] = float(pred.get("fwhm", np.nan))
			map_obj[y, x] = float(pred.get("objective", np.nan))
			map_mae[y, x] = float(pred.get("mae_proxy", np.nan))
			fit_count += 1
		processed_mask[y, x] = True
		done_steps += 1

		if (done_steps % int(progress_every) == 0) or (p_done == total_pixels):
			_write_map_fits_2d(inprog_logn_path, map_logn, ref_hdr, "In-progress inverse cube map")
			_write_map_fits_2d(inprog_tex_path, map_tex, ref_hdr, "In-progress inverse cube map")
			_write_map_fits_2d(inprog_velo_path, map_velo, ref_hdr, "In-progress inverse cube map")
			_write_map_fits_2d(inprog_fwhm_path, map_fwhm, ref_hdr, "In-progress inverse cube map")
			_write_map_fits_2d(inprog_obj_path, map_obj, ref_hdr, "In-progress inverse cube objective map")
			_write_map_fits_2d(inprog_mae_path, map_mae, ref_hdr, "In-progress inverse cube MAE proxy map")
			_write_done_mask_fits_2d(done_mask_fits_path, processed_mask, ref_hdr, "In-progress done mask")
			_save_cubefit_progress_png(
				logn_map=map_logn,
				tex_map=map_tex,
				velo_map=map_velo,
				fwhm_map=map_fwhm,
				done_steps=int(done_steps),
				total_steps=int(total_pixels),
				out_png=progress_png,
				ref_hdr=ref_hdr,
				processed_mask=processed_mask,
			)
			try:
				with open(state_json_path, "w", encoding="utf-8") as f:
					json.dump({"done_steps": int(done_steps), "total_steps": int(total_pixels), "fit_count": int(fit_count)}, f, ensure_ascii=False, indent=2)
			except Exception:
				pass
			_append_cubefit_progress_log(log_txt_path, f"Progress {done_steps}/{total_pixels} | fitted={fit_count}")

	if fit_count <= 0:
		raise RuntimeError("Inverse cube prediction produced no valid fitted pixels")

	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_LOGN.fits"), map_logn, ref_hdr, "Final inverse cube map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_TEX.fits"), map_tex, ref_hdr, "Final inverse cube map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_VELOCITY.fits"), map_velo, ref_hdr, "Final inverse cube map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_FWHM.fits"), map_fwhm, ref_hdr, "Final inverse cube map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_OBJECTIVE.fits"), map_obj, ref_hdr, "Final inverse cube objective map")
	_write_map_fits_2d(os.path.join(out_dir, f"{out_prefix}_MAE.fits"), map_mae, ref_hdr, "Final inverse cube MAE proxy map")
	_append_cubefit_progress_log(log_txt_path, f"Completed inverse cube prediction | valid fitted pixels: {fit_count}/{total_pixels}")
	print(f"[INFO] Inverse cube prediction completed | valid fitted pixels: {fit_count}/{total_pixels}")
	return 0


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
			roi_freq, y_syn_2d, pred_err = predict_signal_roi_batch(
				signal_models_source=signal_models_source,
				is_h5_signal=bool(use_h5),
				roi_entries=roi_entries,
				x_features_2d=x_arr,
				pkg_cache=pkg_cache,
			)
			if pred_err is not None or roi_freq is None or y_syn_2d is None or y_syn_2d.size <= 0:
				warnings_out.append(f"target {tag} failed: no valid synthetic predictions in selected ROI")
				continue

			f_arr = np.asarray(roi_freq, dtype=np.float64).reshape(-1)
			y_arr = np.asarray(y_syn_2d[0], dtype=np.float64).reshape(-1)
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


def _axis_from_min_max_step(vmin: float, vmax: float, step: float) -> np.ndarray:
	lo = float(min(vmin, vmax))
	hi = float(max(vmin, vmax))
	dv = float(abs(step))
	if not np.isfinite(dv) or dv <= 0.0:
		dv = 1.0
	if not np.isfinite(lo) or not np.isfinite(hi):
		return np.asarray([], dtype=np.float64)
	if hi <= lo:
		return np.asarray([lo], dtype=np.float64)
	n = int(np.floor((hi - lo) / dv + 1e-12)) + 1
	n = int(max(1, min(n, 2_000_000)))
	ax = lo + np.arange(n, dtype=np.float64) * dv
	if ax.size == 0:
		ax = np.asarray([lo], dtype=np.float64)
	eps = max(1e-12, dv * 1e-6)
	if abs(float(ax[-1]) - hi) <= eps:
		ax[-1] = hi
	elif float(ax[-1]) < hi:
		ax = np.append(ax, hi)
	if float(ax[0]) != lo:
		ax = np.insert(ax, 0, lo)
	return np.asarray(ax, dtype=np.float64)


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
	roi_freq, Y_syn, pred_err = predict_signal_roi_batch(
		signal_models_source=signal_models_source,
		is_h5_signal=bool(use_h5),
		roi_entries=roi_entries,
		x_features_2d=np.asarray(x_candidates, dtype=np.float32),
		pkg_cache=pkg_cache,
	)
	if pred_err is not None or roi_freq is None or Y_syn is None:
		return None, None, "No valid synthetic predictions for selected ROI"
	return np.asarray(roi_freq, dtype=np.float64), np.asarray(Y_syn, dtype=np.float32), None


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


def _build_concatenated_residual_vector(
	signal_models_source: str,
	noise_models_loaded: List[tuple],
	filter_file: str,
	target_freqs_eval: List[float],
	obs_freq: np.ndarray,
	obs_intensity: np.ndarray,
	case_mode: str,
	noise_scale: float,
	allow_nearest: bool,
	pkg_cache: Dict[str, object],
	x_params_4: np.ndarray,
) -> np.ndarray:
	xc = np.asarray(x_params_4, dtype=np.float32).reshape(1, 4)
	res_parts: List[np.ndarray] = []
	obs_f = np.asarray(obs_freq, dtype=np.float64)
	obs_y = np.asarray(obs_intensity, dtype=np.float64)

	for tf in [float(v) for v in (target_freqs_eval or [])]:
		roi_freq, y_syn_batch, err = _predict_synthetic_batch_single_target(
			signal_models_source=signal_models_source,
			filter_file=filter_file,
			target_freq_ghz=float(tf),
			x_candidates=xc,
			pred_mode=DEFAULT_PRED_MODE,
			selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
			allow_nearest=allow_nearest,
			pkg_cache=pkg_cache,
		)
		if err is not None or roi_freq is None or y_syn_batch is None:
			continue

		y_eval = np.asarray(y_syn_batch[0], dtype=np.float64)
		f_eval = np.asarray(roi_freq, dtype=np.float64)

		if (str(case_mode).strip().lower() == "synthetic_plus_noise") and noise_models_loaded:
			y_noise_batch, noise_mask_batch = _add_noise_batch_for_target(
				noise_models_loaded=noise_models_loaded,
				roi_freq=roi_freq,
				y_syn_batch=np.asarray(y_syn_batch, dtype=np.float32),
				x_candidates=xc,
				noise_scale=float(noise_scale),
			)
			mask_ch = np.any(np.asarray(noise_mask_batch, dtype=bool), axis=0)
			if not np.any(mask_ch):
				continue
			f_eval = np.asarray(roi_freq, dtype=np.float64)[mask_ch]
			y_eval = np.asarray(y_syn_batch[0], dtype=np.float64)[mask_ch] + np.asarray(y_noise_batch[0], dtype=np.float64)[mask_ch]

		y_obs_eval = np.interp(f_eval, obs_f, obs_y, left=np.nan, right=np.nan)
		valid = np.isfinite(y_obs_eval) & np.isfinite(y_eval)
		if int(np.count_nonzero(valid)) < 3:
			continue
		res_parts.append(np.asarray(y_eval[valid] - y_obs_eval[valid], dtype=np.float64).reshape(-1))

	if not res_parts:
		return np.zeros((0,), dtype=np.float64)
	return np.concatenate(res_parts, axis=0).astype(np.float64)


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
	raw = [float(v) for v in (target_freqs or []) if np.isfinite(float(v))]
	# Last occurrence wins: keep unique frequencies preserving the order of their
	# latest appearance in the input list.
	seen = set()
	uniq_rev: List[float] = []
	for tf in reversed(raw):
		k = round(float(tf), 12)
		if k in seen:
			continue
		seen.add(k)
		uniq_rev.append(float(tf))
	uniq = list(reversed(uniq_rev))
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
			ix = int(key_to_idx[grp_key])
			out[ix]["guide_freqs_ghz"].append(float(tf))
			# Use the most recently added guide frequency as representative.
			out[ix]["representative_target_freq_ghz"] = float(tf)
			if roi_lo is not None and roi_hi is not None:
				out[ix]["roi_f_min_ghz"] = float(roi_lo)
				out[ix]["roi_f_max_ghz"] = float(roi_hi)
			if n_ch is not None:
				out[ix]["n_roi_channels"] = int(n_ch)

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
	refine_after_first_fit: bool = True,
	refine_span_fraction: float = 0.20,
	refine_n_candidates: Optional[int] = None,
	local_optimizer_method: str = "none",
	local_optimizer_max_nfev: int = 24,
	_internal_refine_stage: int = 0,
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
	local_opt_method = str(local_optimizer_method).strip().lower()
	if local_opt_method not in {"none", "trf"}:
		local_opt_method = "none"
	if local_opt_method == "trf" and least_squares is None:
		local_opt_method = "none"
	# Legacy compatibility placeholder (old constraint removed).
	enforce_below_obs = False

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
	skipped_zero_observed_targets: List[float] = []
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
					y_obs_valid = np.asarray(y_obs_roi_ref[valid_ref], dtype=np.float64)
					if y_obs_valid.size > 0 and np.all(np.isclose(y_obs_valid, 0.0, rtol=0.0, atol=1e-12)):
						warnings_out.append(
							f"target {tag} skipped: observational ROI is a continuous zero line (all points are 0), excluded from fitting"
						)
						skipped_zero_observed_targets.append(float(tf))
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

	local_opt_applied = False
	local_opt_used_result = False
	local_opt_status = "not_requested"
	if (
		str(local_opt_method) == "trf"
		and int(_internal_refine_stage) == 0
		and least_squares is not None
		and int(n) >= 1
	):
		try:
			anchor = np.asarray(X[best_global_idx], dtype=np.float64).reshape(-1)
			if anchor.size == 4:
				local_opt_applied = True
				local_opt_status = "started"
				lb = np.asarray([
					float(min(float(ranges["logn_min"]), float(ranges["logn_max"]))),
					float(min(float(ranges["tex_min"]), float(ranges["tex_max"]))),
					float(min(float(ranges["velo_min"]), float(ranges["velo_max"]))),
					float(min(float(ranges["fwhm_min"]), float(ranges["fwhm_max"]))),
				], dtype=np.float64)
				hb = np.asarray([
					float(max(float(ranges["logn_min"]), float(ranges["logn_max"]))),
					float(max(float(ranges["tex_min"]), float(ranges["tex_max"]))),
					float(max(float(ranges["velo_min"]), float(ranges["velo_max"]))),
					float(max(float(ranges["fwhm_min"]), float(ranges["fwhm_max"]))),
				], dtype=np.float64)

				def _resid_fn(theta):
					r = _build_concatenated_residual_vector(
						signal_models_source=signal_models_source,
						noise_models_loaded=noise_models_loaded,
						filter_file=filter_file,
						target_freqs_eval=target_freqs_eval,
						obs_freq=np.asarray(obs_freq, dtype=np.float64),
						obs_intensity=np.asarray(obs_intensity, dtype=np.float64),
						case_mode=case_mode,
						noise_scale=float(noise_scale),
						allow_nearest=bool(allow_nearest),
						pkg_cache=pkg_cache,
						x_params_4=np.asarray(theta, dtype=np.float64),
					)
					if int(r.size) <= 0:
						return np.asarray([1e6], dtype=np.float64)
					return np.asarray(r, dtype=np.float64)

				ls = least_squares(
					_resid_fn,
					x0=np.asarray(anchor, dtype=np.float64),
					bounds=(lb, hb),
					method="trf",
					max_nfev=int(max(8, int(local_optimizer_max_nfev))),
					ftol=1e-6,
					xtol=1e-6,
					gtol=1e-6,
				)

				if bool(getattr(ls, "success", False)) and np.all(np.isfinite(np.asarray(ls.x, dtype=np.float64))):
					x_ref = np.asarray(ls.x, dtype=np.float32).reshape(1, 4)
					X_ref = np.vstack([np.asarray(anchor, dtype=np.float32).reshape(1, 4), x_ref]).astype(np.float32)
					res_ref = _run_roi_fitting(
						signal_models_source=signal_models_source,
						noise_models_root=noise_models_root,
						filter_file=filter_file,
						target_freqs=target_freqs,
						obs_freq=obs_freq,
						obs_intensity=obs_intensity,
						case_mode=case_mode,
						fit_criterion=fit_criterion,
						global_weight_mode=global_weight_mode,
						global_search_mode=global_search_mode,
						candidate_mode=candidate_mode,
						n_candidates=2,
						ranges=ranges,
						noise_scale=noise_scale,
						allow_nearest=allow_nearest,
						seed=seed,
						x_candidates_override=X_ref,
						noise_models_loaded_override=noise_models_loaded,
						pkg_cache_override=pkg_cache,
						refine_after_first_fit=False,
						refine_span_fraction=refine_span_fraction,
						refine_n_candidates=refine_n_candidates,
						local_optimizer_method="none",
						local_optimizer_max_nfev=int(local_optimizer_max_nfev),
						_internal_refine_stage=1,
					)
					if isinstance(res_ref, dict) and bool(res_ref.get("ok", False)):
						obj_ref = float(res_ref.get("best_global_mean_objective", np.inf))
						obj_cur = float(global_obj[best_global_idx])
						if np.isfinite(obj_ref) and (obj_ref <= obj_cur):
							local_opt_used_result = True
							res_ref["local_optimizer_method"] = "trf"
							res_ref["local_optimizer_applied"] = True
							res_ref["local_optimizer_used_result"] = True
							res_ref["local_optimizer_max_nfev"] = int(max(8, int(local_optimizer_max_nfev)))
							return res_ref
						warnings_out.append(
							f"TRF local optimizer tried but initial solution kept (objective did not improve: initial={obj_cur:.6g}, trf={obj_ref:.6g})"
						)
						local_opt_status = "no_improve"
					else:
						warnings_out.append("TRF local optimizer did not produce a valid fitted result")
						local_opt_status = "invalid_result"
				else:
					local_opt_status = "solver_failed"
					warnings_out.append("TRF local optimizer did not converge")
		except Exception as e:
			local_opt_status = "error"
			warnings_out.append(f"TRF local optimizer skipped due to error: {e}")

	refinement_applied = False
	refinement_used_result = False
	refinement_span_used = float("nan")
	refinement_n_used = 0
	do_refine = bool(refine_after_first_fit) and int(_internal_refine_stage) == 0 and (not isinstance(x_candidates_override, np.ndarray))
	if do_refine and n >= 2:
		try:
			anchor = np.asarray(X[best_global_idx], dtype=np.float64).reshape(-1)
			if anchor.size == 4:
				span_frac = float(np.clip(float(refine_span_fraction), 0.02, 0.95))
				refinement_span_used = float(span_frac)
				keys = ["logn", "tex", "velo", "fwhm"]
				local_ranges = {}
				for i_k, k in enumerate(keys):
					g_lo = float(min(float(ranges[f"{k}_min"]), float(ranges[f"{k}_max"])))
					g_hi = float(max(float(ranges[f"{k}_min"]), float(ranges[f"{k}_max"])))
					g_span = float(max(1e-9, g_hi - g_lo))
					half = 0.5 * g_span * span_frac
					lo = max(g_lo, float(anchor[i_k]) - half)
					hi = min(g_hi, float(anchor[i_k]) + half)
					if hi <= lo:
						lo = max(g_lo, float(anchor[i_k]) - 0.01 * g_span)
						hi = min(g_hi, float(anchor[i_k]) + 0.01 * g_span)
					local_ranges[f"{k}_min"] = float(lo)
					local_ranges[f"{k}_max"] = float(max(lo + 1e-9, hi))

				if refine_n_candidates is None:
					n_ref = int(max(80, min(1200, max(100, int(n * 0.66)))))
				else:
					n_ref = int(max(20, int(refine_n_candidates)))
				refinement_n_used = int(n_ref)

				X_ref = _sample_fit_candidates(
					n_samples=int(n_ref),
					ranges=local_ranges,
					seed=int(seed) + 7919,
					mode=str(candidate_mode),
				)
				X_ref[0, :] = anchor.astype(np.float32)

				refined = _run_roi_fitting(
					signal_models_source=signal_models_source,
					noise_models_root=noise_models_root,
					filter_file=filter_file,
					target_freqs=target_freqs,
					obs_freq=obs_freq,
					obs_intensity=obs_intensity,
					case_mode=case_mode,
					fit_criterion=fit_criterion,
					global_weight_mode=global_weight_mode,
					global_search_mode=global_search_mode,
					candidate_mode=candidate_mode,
					n_candidates=int(n_ref),
					ranges=local_ranges,
					noise_scale=noise_scale,
					allow_nearest=allow_nearest,
					seed=int(seed) + 7919,
					x_candidates_override=X_ref,
					noise_models_loaded_override=noise_models_loaded,
					pkg_cache_override=pkg_cache,
					refine_after_first_fit=False,
					refine_span_fraction=refine_span_fraction,
					refine_n_candidates=refine_n_candidates,
					local_optimizer_method="none",
					local_optimizer_max_nfev=int(local_optimizer_max_nfev),
					_internal_refine_stage=1,
				)
				refinement_applied = True
				if isinstance(refined, dict) and bool(refined.get("ok", False)):
					obj_ref = float(refined.get("best_global_mean_objective", np.inf))
					obj_cur = float(global_obj[best_global_idx])
					if np.isfinite(obj_ref) and (obj_ref <= obj_cur):
						refinement_used_result = True
						merged_warn = list(warnings_out)
						for wmsg in list(refined.get("warnings", []) if isinstance(refined.get("warnings", []), list) else []):
							merged_warn.append(wmsg)
						refined["warnings"] = merged_warn
						refined["refinement_applied"] = True
						refined["refinement_used_result"] = True
						refined["refinement_span_fraction"] = float(span_frac)
						refined["refinement_n_candidates"] = int(n_ref)
						refined["n_candidates_initial"] = int(n)
						refined["best_global_params_initial"] = {
							"logN": float(X[best_global_idx, 0]),
							"Tex": float(X[best_global_idx, 1]),
							"Velocity": float(X[best_global_idx, 2]),
							"FWHM": float(X[best_global_idx, 3]),
						}
						refined["best_global_mean_objective_initial"] = float(global_obj[best_global_idx])
						return refined
					warnings_out.append(
						f"refinement tried but kept initial solution (objective did not improve: initial={obj_cur:.6g}, refined={obj_ref:.6g})"
					)
				else:
					warnings_out.append("refinement stage did not return a valid fitted result; initial solution kept")
		except Exception as e:
			warnings_out.append(f"refinement stage skipped due to error: {e}")

	# Build global overlay using a single best parameter vector across all fitted ROIs
	x_best = np.asarray(X[best_global_idx:best_global_idx + 1], dtype=np.float32)
	global_overlay = []
	global_per_roi_rows: List[dict] = []
	global_plot_payload: List[dict] = []
	for tf in target_freqs_eval:
		if any(np.isclose(float(tf), float(v), rtol=0.0, atol=1e-9) for v in skipped_zero_observed_targets):
			continue
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
		"n_rois_skipped_zero_observed": int(len(skipped_zero_observed_targets)),
		"skipped_zero_observed_targets_ghz": sorted([float(v) for v in skipped_zero_observed_targets]),
		"n_candidates": int(n),
		"best_global_index": int(best_global_idx),
		"best_global_params": {
			"logN": float(X[best_global_idx, 0]),
			"Tex": float(X[best_global_idx, 1]),
			"Velocity": float(X[best_global_idx, 2]),
			"FWHM": float(X[best_global_idx, 3]),
		},
		"fit_criterion": str(crit),
		"local_optimizer_method": str(local_opt_method),
		"local_optimizer_applied": bool(local_opt_applied),
		"local_optimizer_used_result": bool(local_opt_used_result),
		"local_optimizer_status": str(local_opt_status),
		"local_optimizer_max_nfev": int(max(8, int(local_optimizer_max_nfev))),
		"global_weight_mode": str(weighting_used),
		"candidate_mode": str(candidate_mode),
		"best_global_mean_objective": float(global_obj[best_global_idx]),
		"best_global_mean_MAE": float(global_mae[best_global_idx]),
		"n_rois_fitted": int(len(per_roi_out)),
		"per_roi": per_roi_out,
		"plot_payload": plot_payload_out,
		"global_overlay": global_overlay,
		"refinement_applied": bool(refinement_applied),
		"refinement_used_result": bool(refinement_used_result),
		"refinement_span_fraction": (float(refinement_span_used) if np.isfinite(float(refinement_span_used)) else np.nan),
		"refinement_n_candidates": int(refinement_n_used),
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
	if "cubefit_start_ts" not in st.session_state:
		st.session_state.cubefit_start_ts = None
	if "invcubepred_proc" not in st.session_state:
		st.session_state.invcubepred_proc = None
	if "invcubepred_log_path" not in st.session_state:
		st.session_state.invcubepred_log_path = ""
	if "invcubepred_cfg_path" not in st.session_state:
		st.session_state.invcubepred_cfg_path = ""
	if "invcubepred_log_handle" not in st.session_state:
		st.session_state.invcubepred_log_handle = None
	if "invcubepred_start_ts" not in st.session_state:
		st.session_state.invcubepred_start_ts = None
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
	if "p6_synth_only_group_map" not in st.session_state:
		st.session_state.p6_synth_only_group_map = {}
	if "p6_guide_freqs_fit_input" not in st.session_state:
		st.session_state.p6_guide_freqs_fit_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
	if "p6_guide_freqs_fit_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_fit_pending = ""
	if "p6_guide_fit_refresh" not in st.session_state:
		st.session_state.p6_guide_fit_refresh = False
	if "p6_guide_freqs_fit_last_nonempty" not in st.session_state:
		st.session_state.p6_guide_freqs_fit_last_nonempty = ""
	if "p6_guide_freqs_cfit_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_cfit_pending = ""
	if "p6_guide_cfit_refresh" not in st.session_state:
		st.session_state.p6_guide_cfit_refresh = False
	if "p6_guide_freqs_icp_input" not in st.session_state:
		st.session_state.p6_guide_freqs_icp_input = _freqs_to_text([float(v) for v in DEFAULT_CUBEFIT_GUIDE_FREQS])
	if "p6_guide_freqs_icp_pending" not in st.session_state:
		st.session_state.p6_guide_freqs_icp_pending = ""
	if "p6_guide_icp_refresh" not in st.session_state:
		st.session_state.p6_guide_icp_refresh = False
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
		# CPU-only runtime: no GPU cache maintenance.
		pass
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


def _is_invcubepred_running() -> bool:
	proc = st.session_state.get("invcubepred_proc", None)
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
	st.session_state.cubefit_start_ts = None


def _stop_invcubepred_process():
	proc = st.session_state.get("invcubepred_proc", None)
	if proc is not None and proc.poll() is None:
		try:
			proc.terminate()
			proc.wait(timeout=6)
		except Exception:
			try:
				proc.kill()
			except Exception:
				pass
	st.session_state.invcubepred_proc = None
	fh = st.session_state.get("invcubepred_log_handle", None)
	if fh is not None:
		try:
			fh.close()
		except Exception:
			pass
	st.session_state.invcubepred_log_handle = None
	cfgp = st.session_state.get("invcubepred_cfg_path", "")
	if cfgp and os.path.isfile(cfgp):
		try:
			os.remove(cfgp)
		except Exception:
			pass
	st.session_state.invcubepred_cfg_path = ""
	st.session_state.invcubepred_start_ts = None


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


def _show_fits_preview(
	title: str,
	arr: np.ndarray,
	cmap: str = "viridis",
	ref_hdr: Optional[object] = None,
	zoom_mask: Optional[np.ndarray] = None,
):
	if arr is None:
		return
	v = np.asarray(arr, dtype=np.float32)
	fin = np.isfinite(v)
	if not np.any(fin):
		return
	vmin, vmax = _compute_display_limits(v)
	wcs2d = _header_to_celestial_wcs(ref_hdr)
	if wcs2d is not None:
		fig = plt.figure(figsize=(4.6, 4.2))
		ax = fig.add_subplot(111, projection=wcs2d)
	else:
		fig, ax = plt.subplots(figsize=(4.2, 4.0))
	im = ax.imshow(v, origin="lower", cmap=str(cmap), vmin=vmin, vmax=vmax)
	ax.set_title(f"{title} | shape={v.shape[0]}x{v.shape[1]}")
	if wcs2d is not None:
		ax.set_xlabel("RA")
		ax.set_ylabel("Dec")
	else:
		ax.set_xlabel("x")
		ax.set_ylabel("y")
	zoom_lim = _compute_zoom_limits_from_mask(zoom_mask, pad_frac=0.08)
	if zoom_lim is not None:
		x0, x1, y0, y1 = zoom_lim
		ax.set_xlim(float(x0) - 0.5, float(x1) + 0.5)
		ax.set_ylim(float(y0) - 0.5, float(y1) + 0.5)
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
				is_resume_state = ln == "cubefit_inprogress_state.json"
				is_resume_logtxt = ln == "log.txt"
				if is_cubefit_fits or is_progress_png or is_progress_json or is_run_log or is_resume_state or is_resume_logtxt:
					os.remove(p)
		except Exception:
			pass


def _cleanup_invcubepred_outputs_for_dir(out_dir: str):
	if (not out_dir) or (not os.path.isdir(out_dir)):
		return
	for name in os.listdir(out_dir):
		p = os.path.join(out_dir, name)
		ln = str(name).lower()
		try:
			if os.path.isfile(p):
				is_inv_fits = ln.startswith("invcubepred_") and ln.endswith(".fits")
				is_progress_png = ln.endswith("_inprogress_map.png")
				is_progress_json = ln.endswith("_inprogress_map.json")
				is_run_log = ln.startswith("invcubepred_run_") and ln.endswith(".log")
				is_resume_state = ln == "invcubepred_inprogress_state.json"
				is_resume_logtxt = ln == "log.txt"
				if is_inv_fits or is_progress_png or is_progress_json or is_run_log or is_resume_state or is_resume_logtxt:
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


@st.cache_resource(show_spinner=False)
def _load_eval16_module_cached(path_str: str):
	path = Path(path_str)
	return _load_module_from_path(path, "eval16_for_p6_cached")


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


class _RoiRankNNLite(nn.Module):
	def __init__(self, input_size: int, hidden_sizes: List[int], output_size: int = 2, dropout: float = 0.15):
		super().__init__()
		layers = []
		prev = int(input_size)
		for h in hidden_sizes:
			hh = int(h)
			layers.append(nn.Linear(prev, hh))
			layers.append(nn.BatchNorm1d(hh))
			layers.append(nn.ReLU())
			layers.append(nn.Dropout(float(dropout)))
			prev = hh
		layers.append(nn.Linear(prev, int(output_size)))
		self.net = nn.Sequential(*layers)

	def forward(self, x):
		return self.net(x)


def _safe_trapezoid_np(y):
	yy = np.asarray(y, dtype=np.float64)
	if yy.size < 2:
		return 0.0
	trapz_fn = getattr(np, "trapezoid", None)
	if callable(trapz_fn):
		return float(trapz_fn(yy))
	trapz_fn = getattr(np, "trapz", None)
	if callable(trapz_fn):
		return float(trapz_fn(yy))
	return float(np.sum((yy[:-1] + yy[1:]) * 0.5))


def _resample_1d_by_index_float64(y: np.ndarray, target_len: int) -> np.ndarray:
	yy = np.asarray(y, dtype=np.float64).reshape(-1)
	t = int(max(2, int(target_len)))
	if yy.size == t:
		return yy
	if yy.size < 2:
		return np.full((t,), float(yy[0]) if yy.size == 1 else np.nan, dtype=np.float64)
	x_old = np.linspace(0.0, 1.0, int(yy.size), dtype=np.float64)
	x_new = np.linspace(0.0, 1.0, t, dtype=np.float64)
	return np.interp(x_new, x_old, yy).astype(np.float64)


def _estimate_edge_step_ghz(freqs: np.ndarray) -> float:
	f = np.asarray(freqs, dtype=np.float64).reshape(-1)
	if f.size < 2:
		return 1e-6
	d = np.abs(np.diff(f))
	d = d[np.isfinite(d) & (d > 0.0)]
	if d.size == 0:
		return 1e-6
	return float(np.nanmedian(d))


def concat_segments_with_zero_gaps(
	freqs_segments: List[np.ndarray],
	spec_segments: List[np.ndarray],
	insert_zero_gaps: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
	if len(freqs_segments) != len(spec_segments):
		raise ValueError("freqs_segments and spec_segments must have the same length")
	if len(freqs_segments) == 0:
		return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

	f_out_parts: List[np.ndarray] = []
	y_out_parts: List[np.ndarray] = []
	nseg = int(len(freqs_segments))

	for i in range(nseg):
		fi = np.asarray(freqs_segments[i], dtype=np.float64).reshape(-1)
		yi = np.asarray(spec_segments[i], dtype=np.float64).reshape(-1)
		if fi.size == 0 or yi.size == 0:
			continue
		if fi.size != yi.size:
			raise ValueError("Each segment must have same frequency/intensity length")

		f_out_parts.append(fi)
		y_out_parts.append(yi)

		if (not insert_zero_gaps) or (i >= nseg - 1):
			continue

		fj = np.asarray(freqs_segments[i + 1], dtype=np.float64).reshape(-1)
		if fj.size == 0:
			continue

		f_end = float(fi[-1])
		f_next = float(fj[0])
		if (not np.isfinite(f_end)) or (not np.isfinite(f_next)):
			continue

		gap = float(abs(f_next - f_end))
		step_i = _estimate_edge_step_ghz(fi)
		step_j = _estimate_edge_step_ghz(fj)
		base_step = float(max(1e-9, min(step_i, step_j)))
		direction = 1.0 if (f_next >= f_end) else -1.0

		edge_eps = float(max(1e-9, 0.45 * base_step))
		if gap > 0.0:
			edge_eps = float(min(edge_eps, 0.45 * gap))

		z1 = float(f_end + direction * edge_eps)
		z2 = float(f_next - direction * edge_eps)

		if direction > 0.0 and z1 >= z2:
			mid = float(0.5 * (f_end + f_next))
			z1 = float(mid - edge_eps)
			z2 = float(mid + edge_eps)
		elif direction < 0.0 and z1 <= z2:
			mid = float(0.5 * (f_end + f_next))
			z1 = float(mid + edge_eps)
			z2 = float(mid - edge_eps)

		f_out_parts.append(np.asarray([z1, z2], dtype=np.float64))
		y_out_parts.append(np.asarray([0.0, 0.0], dtype=np.float64))

	if not f_out_parts:
		return np.array([], dtype=np.float64), np.array([], dtype=np.float64)

	return np.concatenate(f_out_parts, axis=0), np.concatenate(y_out_parts, axis=0)


def _build_obs_features_for_rank(obs_seg, spw_idx, n_spw):
	o = np.asarray(obs_seg, dtype=np.float64)
	if o.size < 2:
		return None
	obs_mean = float(np.mean(o))
	obs_std = float(np.std(o))
	obs_peak = float(np.max(o))
	obs_p95 = float(np.quantile(o, 0.95))
	obs_p05 = float(np.quantile(o, 0.05))
	obs_median = float(np.median(o))
	abs_mean = float(np.mean(np.abs(o)))
	snr_proxy = float(abs(obs_peak) / max(1e-12, obs_std))
	area_abs = float(_safe_trapezoid_np(np.abs(o)))
	area_signed = float(_safe_trapezoid_np(o))
	g = np.diff(o)
	grad_std = float(np.std(g)) if g.size > 0 else 0.0
	grad_abs_mean = float(np.mean(np.abs(g))) if g.size > 0 else 0.0
	centered = o - obs_median
	if centered.size > 1:
		zc = np.count_nonzero((centered[:-1] * centered[1:]) < 0.0)
		zcr = float(zc) / float(centered.size - 1)
	else:
		zcr = 0.0
	spw_norm = float(spw_idx - 1) / max(1.0, float(n_spw - 1))
	return np.array([
		obs_mean, obs_std, obs_peak, obs_p95,
		obs_p05, obs_median, abs_mean, snr_proxy,
		area_abs, area_signed, grad_std, grad_abs_mean,
		zcr, spw_norm,
	], dtype=np.float32)


def _is_invalid_obs_roi_line_rank(obs_seg: np.ndarray) -> bool:
	o = np.asarray(obs_seg, dtype=np.float64).reshape(-1)
	if o.size < 4:
		return True
	if not np.all(np.isfinite(o)):
		return True
	if np.all(np.isclose(o, 0.0, rtol=0.0, atol=1e-12)):
		return True
	if float(np.nanstd(o)) <= 1e-8:
		return True
	med = float(np.nanmedian(o))
	abs_o = np.abs(o - med)
	amp = float(np.nanmax(abs_o))
	if amp <= 1e-12:
		return True
	zero_tol = max(1e-12, 2e-3 * amp)
	near_base = abs_o <= zero_tol
	if float(np.mean(near_base)) >= 0.85:
		return True
	# longest continuous near-baseline run
	best = 0
	run = 0
	for v in near_base:
		if bool(v):
			run += 1
			if run > best:
				best = run
		else:
			run = 0
	if float(best) / float(o.size) >= 0.50:
		return True
	return False


def _build_auto_roi_defs(freq_ghz: np.ndarray) -> List[dict]:
	f = np.asarray(freq_ghz, dtype=np.float64).reshape(-1)
	n = int(f.size)
	if n < 8:
		return []
	n_bins = int(max(8, min(30, n // 120)))
	edges = np.linspace(0, n, num=n_bins + 1, dtype=int)
	out: List[dict] = []
	for i in range(n_bins):
		a = int(edges[i])
		b = int(edges[i + 1]) - 1
		if b <= a:
			continue
		idx = np.arange(a, b + 1, dtype=int)
		if idx.size < 4:
			continue
		fmin = float(np.min(f[idx]))
		fmax = float(np.max(f[idx]))
		fc = 0.5 * (fmin + fmax)
		out.append({
			"roi_name": f"roi_{len(out)+1:02d}_{fc:.6f}GHz",
			"target_freq_ghz": float(fc),
			"f_min_ghz": float(fmin),
			"f_max_ghz": float(fmax),
			"idx": idx,
		})
	return out


def _load_model_target_freqs_for_ranking(model_dir: str, meta_path: str) -> List[float]:
	targets: List[float] = []
	meta = {}
	try:
		with open(str(meta_path), "r", encoding="utf-8") as f:
			meta = json.load(f)
	except Exception:
		meta = {}

	# 1) Best case: explicit list in meta.
	raw = meta.get("target_region_freqs_ghz", None)
	if isinstance(raw, list):
		for v in raw:
			try:
				vv = float(v)
				if np.isfinite(vv):
					targets.append(vv)
			except Exception:
				pass

	# 2) Fallback: ranking files referenced in meta.
	candidates = []
	for k in ["ranking_json", "ranking_csv"]:
		p = str(meta.get(k, "")).strip()
		if p:
			candidates.append(p)

	# 3) Fallback in model directory.
	md = str(model_dir or "").strip()
	if md:
		candidates.extend([
			os.path.join(md, "roi_ranking_global_test.json"),
			os.path.join(md, "roi_ranking_global_test.csv"),
			os.path.join(md, "roi_ranking.json"),
			os.path.join(md, "roi_ranking.csv"),
		])

	seen = set()
	for p in candidates:
		pp = str(p).strip()
		if (not pp) or (pp in seen) or (not os.path.isfile(pp)):
			continue
		seen.add(pp)
		try:
			if pp.lower().endswith(".json"):
				with open(pp, "r", encoding="utf-8") as f:
					rows = json.load(f)
				if isinstance(rows, list):
					for r in rows:
						if isinstance(r, dict) and ("target_freq_ghz" in r):
							vv = float(r.get("target_freq_ghz", np.nan))
							if np.isfinite(vv):
								targets.append(vv)
			elif pp.lower().endswith(".csv"):
				with open(pp, "r", encoding="utf-8", errors="ignore") as f:
					rdr = csv.DictReader(f)
					for r in rdr:
						if ("target_freq_ghz" in r):
							vv = float(r.get("target_freq_ghz", "nan"))
							if np.isfinite(vv):
								targets.append(vv)
		except Exception:
			continue

	if not targets:
		return []
	arr = np.asarray([float(v) for v in targets], dtype=np.float64)
	arr = arr[np.isfinite(arr)]
	if arr.size == 0:
		return []
	arr = np.unique(np.round(arr, 9))
	return [float(v) for v in arr]


def _build_roi_defs_from_model_targets(freq_ghz: np.ndarray, target_freqs_ghz: List[float]) -> List[dict]:
	f = np.asarray(freq_ghz, dtype=np.float64).reshape(-1)
	if f.size < 8:
		return []
	tg = np.asarray([float(v) for v in (target_freqs_ghz or []) if np.isfinite(float(v))], dtype=np.float64)
	if tg.size == 0:
		return []
	tg = np.unique(np.sort(tg))

	# Keep only targets within observed band (+small margin).
	fmin = float(np.nanmin(f))
	fmax = float(np.nanmax(f))
	margin = 5.0 * float(max(1e-9, np.nanmedian(np.abs(np.diff(f)))))
	tg = tg[(tg >= (fmin - margin)) & (tg <= (fmax + margin))]
	if tg.size == 0:
		return []

	chan_step = float(max(1e-9, np.nanmedian(np.abs(np.diff(f)))))
	provisional: List[dict] = []
	for i, tf in enumerate(tg):
		# Width tied to nearest model target spacing, capped to avoid spanning large empty bands.
		if tg.size == 1:
			nn = float(0.03)
		elif i == 0:
			nn = float(tg[i + 1] - tf)
		elif i == (tg.size - 1):
			nn = float(tf - tg[i - 1])
		else:
			nn = float(min(tf - tg[i - 1], tg[i + 1] - tf))
		nn = float(max(nn, 30.0 * chan_step))
		half_w = float(min(0.008, max(12.0 * chan_step, 0.45 * nn)))

		lo = float(tf - half_w)
		hi = float(tf + half_w)
		provisional.append({"lo": float(lo), "hi": float(hi), "tf": float(tf)})

	if not provisional:
		return []

	# Merge overlapping/very-close provisional windows so nearby target lines map
	# to one ROI region (avoids repeated ROIs in ranking table).
	provisional = sorted(provisional, key=lambda d: (float(d["lo"]), float(d["hi"])))
	merge_gap = float(max(2.0 * chan_step, 1e-8))
	merged: List[dict] = []
	for p in provisional:
		if not merged:
			merged.append({
				"lo": float(p["lo"]),
				"hi": float(p["hi"]),
				"target_freqs_ghz": [float(p["tf"])],
			})
			continue
		last = merged[-1]
		if float(p["lo"]) <= (float(last["hi"]) + merge_gap):
			last["hi"] = float(max(float(last["hi"]), float(p["hi"])))
			last["target_freqs_ghz"].append(float(p["tf"]))
		else:
			merged.append({
				"lo": float(p["lo"]),
				"hi": float(p["hi"]),
				"target_freqs_ghz": [float(p["tf"])],
			})

	out: List[dict] = []
	for m in merged:
		idx = np.where((f >= float(m["lo"])) & (f <= float(m["hi"])))[0]
		if idx.size < 4:
			continue
		tfs = sorted({float(v) for v in m.get("target_freqs_ghz", []) if np.isfinite(float(v))})
		rep_tf = float(tfs[0]) if tfs else float(0.5 * (float(np.min(f[idx])) + float(np.max(f[idx]))))
		out.append({
			"roi_name": f"roi_{len(out)+1:02d}_{rep_tf:.6f}GHz",
			"target_freq_ghz": float(rep_tf),
			"target_freqs_ghz": [float(v) for v in tfs],
			"n_target_freqs_in_roi": int(len(tfs)),
			"f_min_ghz": float(np.min(f[idx])),
			"f_max_ghz": float(np.max(f[idx])),
			"idx": idx.astype(int),
		})
	return out


def _render_model_sources_sidebar(runtime_info: dict) -> dict:
	"""Render model/source controls in sidebar and return active configuration."""
	signal_models_root = str(DEFAULT_MERGED_H5)
	noise_models_root = str(DEFAULT_NOISE_NN_H5)
	filter_file = str(DEFAULT_FILTER_FILE)
	roi_rank_model_dir = str(DEFAULT_LOCAL_ROI_RANK_MODEL_DIR)
	source_mode_label = "manual paths"

	with st.sidebar:
		st.header("Model Upload")
		st.caption(
			f"Runtime: CPU-only | cores={runtime_info['cpu_count']} | "
			f"threads={runtime_info['cpu_threads']}"
		)
		st.markdown("**Model Upload**")
		up_signal_h5 = st.file_uploader("Upload signal models (.h5)", type=["h5", "hdf5"], key="p6_up_signal_h5")
		up_noise_h5 = st.file_uploader("Upload noise models (.h5 bundle or single model)", type=["h5", "hdf5"], key="p6_up_noise_h5")
		up_filter = st.file_uploader("Upload spectral filter (.txt/.dat/.csv)", type=["txt", "dat", "csv"], key="p6_up_filter")
		st.markdown("**ROI Ranking model upload (.h5 bundle or 1.5 artifacts)**")
		up_rank_bundle_h5 = st.file_uploader("Upload ROI ranking model bundle (.h5)", type=["h5", "hdf5"], key="p6_up_rank_bundle_h5")

		uploaded_signal_path = _save_uploaded_file_to_temp(up_signal_h5, "signal")
		uploaded_noise_path = _save_uploaded_file_to_temp(up_noise_h5, "noise")
		uploaded_filter_path = _save_uploaded_file_to_temp(up_filter, "filter")
		uploaded_rank_bundle_h5 = _save_uploaded_file_to_temp(up_rank_bundle_h5, "roi_rank_bundle_h5")
		if uploaded_signal_path:
			signal_models_root = str(uploaded_signal_path)
		if uploaded_noise_path:
			noise_models_root = str(uploaded_noise_path)
		if uploaded_filter_path:
			filter_file = str(uploaded_filter_path)
		rank_dir_uploaded = None
		rank_upload_warning = None
		if uploaded_rank_bundle_h5:
			rank_dir_uploaded, rank_upload_warning = _prepare_roi_rank_model_dir_from_h5_bundle(uploaded_rank_bundle_h5)
		if rank_dir_uploaded:
			roi_rank_model_dir = str(rank_dir_uploaded)
		if rank_upload_warning:
			st.warning(str(rank_upload_warning))

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
				signal_models_root, noise_models_root, filter_file, roi_rank_model_dir = _apply_drive_auto_paths(
					signal_models_root=signal_models_root,
					noise_models_root=noise_models_root,
					filter_file=filter_file,
					roi_rank_model_dir=roi_rank_model_dir,
					auto_paths=auto_paths,
				)
				source_mode_label = "Google Drive temporary download"
				st.caption("Active source mode: Google Drive temporary download")
			else:
				source_mode_label = "manual paths (Drive not ready yet)"
				st.caption("Active source mode: manual paths (Drive not ready yet)")
		else:
			source_mode_label = "manual paths"
			st.caption("Active source mode: manual paths")

		st.markdown("---")
		st.markdown("**Optional: use local preset paths (D:\\4.DATASETS)**")
		use_local_preset = st.checkbox("Use local preset models", value=False, key="p6_use_local_preset")
		if use_local_preset:
			signal_models_root = str(DEFAULT_LOCAL_SIGNAL_H5)
			noise_models_root = str(DEFAULT_LOCAL_NOISE_H5)
			filter_file = str(DEFAULT_LOCAL_FILTER_FILE)
			roi_rank_model_dir = str(DEFAULT_LOCAL_ROI_RANK_MODEL_DIR)
			source_mode_label = "local preset paths"
			st.caption("Active source mode: local preset paths")
			st.caption(f"Includes ROI ranking model dir: {DEFAULT_LOCAL_ROI_RANK_MODEL_DIR}")
			for w in _validate_local_preset_sources(signal_models_root, noise_models_root, filter_file, roi_rank_model_dir):
				st.warning(str(w))

		st.caption(f"Signal source in use: {signal_models_root}")
		st.caption(f"Noise source in use: {noise_models_root}")
		st.caption(f"Filter file in use: {filter_file}")
		st.caption(f"ROI ranking model source in use: {roi_rank_model_dir}")

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

	return {
		"signal_models_root": str(signal_models_root),
		"noise_models_root": str(noise_models_root),
		"filter_file": str(filter_file),
		"roi_rank_model_dir": str(roi_rank_model_dir),
		"source_mode_label": str(source_mode_label),
		"target_freqs": [float(v) for v in target_freqs],
		"allow_nearest": bool(allow_nearest),
		"noise_scale": float(noise_scale),
	}


def run_streamlit_app():
	st.set_page_config(page_title="OBSEMULATOR", page_icon="📡", layout="wide")
	runtime_info = _configure_runtime_resources_cpu_only()
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
	sidebar_cfg = _render_model_sources_sidebar(runtime_info)
	signal_models_root = str(sidebar_cfg["signal_models_root"])
	noise_models_root = str(sidebar_cfg["noise_models_root"])
	filter_file = str(sidebar_cfg["filter_file"])
	roi_rank_model_dir = str(sidebar_cfg["roi_rank_model_dir"])
	source_mode_label = str(sidebar_cfg.get("source_mode_label", "manual paths"))
	target_freqs = [float(v) for v in sidebar_cfg["target_freqs"]]
	allow_nearest = bool(sidebar_cfg["allow_nearest"])
	noise_scale = float(sidebar_cfg["noise_scale"])

	tab_cube, tab_cube2, tab_cube3, tab_synth_batch, tab_eval16, tab_fit, tab_inv_params, tab_cube_fit, tab_pred_from_cube = st.tabs(["Cube Generator", "Simulate Single Spectrum", "Simulate Single Synthetic Spectrum", "Generate Synthetic Spectra", "ROI Ranking Eval", "Fitting", "Inverse Param Prediction", "Cube Fitting", "Inverse Cube Prediction"])

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
			st.session_state.p6_guide_freqs_main_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
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

			sel_sig_pos = _resolve_roi_selected_pos(st.session_state.get("p6_signal_roi_select", 0), signal_rois, default_pos=0)
			sel_noi_pos = _resolve_roi_selected_pos(st.session_state.get("p6_noise_roi_select", 0), noise_rois, default_pos=0)
			if signal_rois:
				st.session_state.p6_signal_roi_select = int(sel_sig_pos)
			if noise_rois:
				st.session_state.p6_noise_roi_select = int(sel_noi_pos)

			if signal_rois:
				if int(sel_sig_pos) >= len(signal_rois):
					st.session_state.p6_signal_roi_select = 0
					sel_sig_pos = 0
			if noise_rois:
				if int(sel_noi_pos) >= len(noise_rois):
					st.session_state.p6_noise_roi_select = 0
					sel_noi_pos = 0

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
					sel_s = signal_rois[int(sel_sig_pos)]
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
					sel_n = noise_rois[int(sel_noi_pos)]
					spw_txt = ",".join(sel_n.get("spw", [])) if sel_n.get("spw", []) else "-"
					match_s = _get_overlapping_signal_roi_indices(sel_n, signal_rois)
					match_s_txt = ",".join([f"S{v}" for v in match_s]) if match_s else "none"
					st.caption(f"Selected: ROI N{int(sel_n['index'])} | range {float(sel_n['lo']):.6f}–{float(sel_n['hi']):.6f} GHz | SPW: {spw_txt} | matching Signal ROI(s): {match_s_txt}")
				else:
					st.caption("No noise ROIs available")

			sel_sig_idx = None if not signal_rois else int(signal_rois[int(sel_sig_pos)]["index"])
			sel_noi_idx = None if not noise_rois else int(noise_rois[int(sel_noi_pos)]["index"])
			combo_freqs = _selected_roi_combo_freqs(
				signal_rois=signal_rois,
				noise_rois=noise_rois,
				selected_signal_pos=int(sel_sig_pos) if signal_rois else None,
				selected_noise_pos=int(sel_noi_pos) if noise_rois else None,
			)
			_plot_roi_overview(signal_rois, noise_rois, guide_freqs_ghz=guide_freqs, selected_combo_freqs_ghz=combo_freqs, selected_signal_index=sel_sig_idx, selected_noise_index=sel_noi_idx, chart_key="p6_roi_overview_cube")

		if st.button("Add selected ROI combination to Guide frequencies", key="p6_add_rois_to_guide"):
			updated_freqs = _append_selected_rois_to_freq_list(
				base_freqs=guide_freqs,
				signal_rois=signal_rois,
				noise_rois=noise_rois,
				selected_signal_pos=int(_resolve_roi_selected_pos(st.session_state.get("p6_signal_roi_select", 0), signal_rois, default_pos=0)) if signal_rois else None,
				selected_noise_pos=int(_resolve_roi_selected_pos(st.session_state.get("p6_noise_roi_select", 0), noise_rois, default_pos=0)) if noise_rois else None,
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
				st.error(GUIDE_FREQS_EMPTY_ERROR)
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
					_show_worker_warnings(str(st.session_state.get("cube_log_path", "")), max_lines=120)
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
			st.session_state.p6_guide_freqs_cube2_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])
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

			sel_sig_pos2 = _resolve_roi_selected_pos(st.session_state.get("p6_signal_roi_select2", 0), signal_rois2, default_pos=0)
			sel_noi_pos2 = _resolve_roi_selected_pos(st.session_state.get("p6_noise_roi_select2", 0), noise_rois2, default_pos=0)
			if signal_rois2:
				st.session_state.p6_signal_roi_select2 = int(sel_sig_pos2)
			if noise_rois2:
				st.session_state.p6_noise_roi_select2 = int(sel_noi_pos2)

			if signal_rois2:
				if int(sel_sig_pos2) >= len(signal_rois2):
					st.session_state.p6_signal_roi_select2 = 0
					sel_sig_pos2 = 0
			if noise_rois2:
				if int(sel_noi_pos2) >= len(noise_rois2):
					st.session_state.p6_noise_roi_select2 = 0
					sel_noi_pos2 = 0

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
					sel_s2 = signal_rois2[int(sel_sig_pos2)]
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
					sel_n2 = noise_rois2[int(sel_noi_pos2)]
					spw_txt2 = ",".join(sel_n2.get("spw", [])) if sel_n2.get("spw", []) else "-"
					match_s2 = _get_overlapping_signal_roi_indices(sel_n2, signal_rois2)
					match_s_txt2 = ",".join([f"S{v}" for v in match_s2]) if match_s2 else "none"
					st.caption(f"Selected: ROI N{int(sel_n2['index'])} | range {float(sel_n2['lo']):.6f}–{float(sel_n2['hi']):.6f} GHz | SPW: {spw_txt2} | matching Signal ROI(s): {match_s_txt2}")
				else:
					st.caption("No noise ROIs available")

			sel_sig_idx2 = None if not signal_rois2 else int(signal_rois2[int(sel_sig_pos2)]["index"])
			sel_noi_idx2 = None if not noise_rois2 else int(noise_rois2[int(sel_noi_pos2)]["index"])
			combo_freqs2 = _selected_roi_combo_freqs(
				signal_rois=signal_rois2,
				noise_rois=noise_rois2,
				selected_signal_pos=int(sel_sig_pos2) if signal_rois2 else None,
				selected_noise_pos=int(sel_noi_pos2) if noise_rois2 else None,
			)
			_plot_roi_overview(signal_rois2, noise_rois2, guide_freqs_ghz=guide_freqs2, selected_combo_freqs_ghz=combo_freqs2, selected_signal_index=sel_sig_idx2, selected_noise_index=sel_noi_idx2, chart_key="p6_roi_overview_cube2")

		if st.button("Add selected ROI combination to Guide frequencies", key="p6_add_rois_to_guide_cube2"):
			updated_freqs2 = _append_selected_rois_to_freq_list(
				base_freqs=guide_freqs2,
				signal_rois=signal_rois2,
				noise_rois=noise_rois2,
				selected_signal_pos=int(_resolve_roi_selected_pos(st.session_state.get("p6_signal_roi_select2", 0), signal_rois2, default_pos=0)) if signal_rois2 else None,
				selected_noise_pos=int(_resolve_roi_selected_pos(st.session_state.get("p6_noise_roi_select2", 0), noise_rois2, default_pos=0)) if noise_rois2 else None,
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
		st.caption("ROIs selected in dropdowns are used only after being added to Guide frequencies.")

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
				st.error(GUIDE_FREQS_EMPTY_ERROR)
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
					guide_txt2_new = _freqs_to_text([float(v) for v in target_freqs_cube2_run])
					st.session_state.p6_guide_freqs_cube2_pending = str(guide_txt2_new)
					st.session_state.p6_guide_cube2_refresh = True
					st.session_state.p6_guide_freqs_cube2_last_nonempty = str(guide_txt2_new)
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
					_show_worker_warnings(str(st.session_state.get("cube_log_path", "")), max_lines=120)
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
			st.session_state.p6_guide_freqs_cube3_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])

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

			sel_sig_pos3 = _resolve_roi_selected_pos(st.session_state.get("p6_signal_roi_select3", 0), signal_rois3, default_pos=0)
			if signal_rois3:
				st.session_state.p6_signal_roi_select3 = int(sel_sig_pos3)

			if signal_rois3 and int(sel_sig_pos3) >= len(signal_rois3):
				st.session_state.p6_signal_roi_select3 = 0
				sel_sig_pos3 = 0

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

			sel_sig_idx3 = None if not signal_rois3 else int(signal_rois3[int(sel_sig_pos3)]["index"])
			combo_freqs3 = _selected_roi_combo_freqs(
				signal_rois=signal_rois3,
				noise_rois=noise_rois3,
				selected_signal_pos=int(sel_sig_pos3) if signal_rois3 else None,
				selected_noise_pos=None,
			)
			_plot_roi_overview(signal_rois3, noise_rois3, guide_freqs_ghz=guide_freqs3, selected_combo_freqs_ghz=combo_freqs3, selected_signal_index=sel_sig_idx3, selected_noise_index=None, chart_key="p6_roi_overview_cube3")

		if st.button("Add selected ROI combination to Guide frequencies", key="p6_add_rois_to_guide_cube3"):
			updated_freqs3 = _append_selected_rois_to_freq_list(
				base_freqs=guide_freqs3,
				signal_rois=signal_rois3,
				noise_rois=noise_rois3,
				selected_signal_pos=int(_resolve_roi_selected_pos(st.session_state.get("p6_signal_roi_select3", 0), signal_rois3, default_pos=0)) if signal_rois3 else None,
				selected_noise_pos=None,
			)
			st.session_state.p6_guide_freqs_cube3_pending = _freqs_to_text(updated_freqs3)
			st.session_state.p6_guide_cube3_refresh = True
			st.rerun()

		guide_freqs_run3 = _normalize_target_freqs_for_run(parse_freq_list(str(st.session_state.get("p6_guide_freqs_cube3_input", ""))))
		guide_freqs_run3_effective = [float(v) for v in guide_freqs_run3]
		guide_group_map3: Dict[str, List[float]] = {}
		if guide_freqs_run3:
			try:
				groups3 = _group_target_freqs_by_signal_roi(
					signal_models_source=str(signal_models_root),
					filter_file=str(filter_file),
					target_freqs=[float(v) for v in guide_freqs_run3],
					allow_nearest=bool(allow_nearest),
				)
				reps3: List[float] = []
				for g3 in groups3:
					rep3 = g3.get("representative_target_freq_ghz", np.nan)
					if not np.isfinite(float(rep3)):
						continue
					rep3f = float(rep3)
					reps3.append(rep3f)
					gf3 = [float(v) for v in g3.get("guide_freqs_ghz", []) if np.isfinite(float(v))]
					guide_group_map3[f"{rep3f:.9f}"] = (sorted(gf3) if gf3 else [rep3f])
				reps3 = _normalize_target_freqs_for_run(reps3)
				if reps3:
					guide_freqs_run3_effective = [float(v) for v in reps3]
			except Exception:
				guide_freqs_run3_effective = [float(v) for v in guide_freqs_run3]

		if guide_freqs_run3_effective:
			st.caption("Target frequencies used for Simulate Single Synthetic Spectrum: " + _freqs_to_text(guide_freqs_run3_effective))
			if len(guide_freqs_run3_effective) < len(guide_freqs_run3):
				st.info(
					f"Plot cleanup by ROI applied: {int(len(guide_freqs_run3) - len(guide_freqs_run3_effective))} guide frequency(ies) collapsed because they fall inside ROIs already represented."
				)
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
			if not guide_freqs_run3_effective:
				st.error(GUIDE_FREQS_EMPTY_ERROR)
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			else:
				with st.spinner("Generating synthetic-only spectra..."):
					res3, warn3 = _generate_synthetic_spectra_for_targets(
						signal_models_source=str(signal_models_root),
						filter_file=str(filter_file),
						target_freqs=[float(v) for v in guide_freqs_run3_effective],
						x_features=[float(logn_cube3), float(tex_cube3), float(velo_cube3), float(fwhm_cube3)],
						pred_mode=DEFAULT_PRED_MODE,
						selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
						allow_nearest=bool(allow_nearest),
					)
				cleanup_warn3: List[str] = []
				if len(guide_freqs_run3_effective) < len(guide_freqs_run3):
					cleanup_warn3.append(
						f"Guide-frequency ROI cleanup applied: input={len(guide_freqs_run3)} -> effective={len(guide_freqs_run3_effective)}"
					)
				st.session_state.p6_synth_only_results = dict(res3)
				st.session_state.p6_synth_only_warnings = list(cleanup_warn3) + list(warn3)
				st.session_state.p6_synth_only_group_map = dict(guide_group_map3)
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

		syn_shift_enabled = st.checkbox("Apply uploaded spectrum frequency shift", value=True, key="p6_syn_only_shift_enabled")
		syn_shift_mode = st.selectbox(
			"Shift mode",
			options=["per_frequency", "spw_center"],
			index=0,
			key="p6_syn_only_shift_mode",
		)
		syn_shift_kms = st.number_input(
			"Uploaded spectrum shift (km/s)",
			value=-98.0,
			step=0.1,
			format="%.4f",
			key="p6_syn_only_shift_kms",
		)
		up_freq3_used = None if up_freq3 is None else np.asarray(up_freq3, dtype=np.float64).copy()
		if (up_freq3_used is not None) and bool(syn_shift_enabled):
			if str(syn_shift_mode).strip().lower() == "spw_center":
				up_freq3_used = _apply_velocity_shift_by_spw_center(up_freq3_used, float(syn_shift_kms))
			else:
				up_freq3_used = _apply_velocity_shift_to_frequency(up_freq3_used, float(syn_shift_kms))
			st.caption(f"Uploaded overlay shifted by {float(syn_shift_kms):+.4f} km/s using mode: {str(syn_shift_mode)}")

		results3 = st.session_state.get("p6_synth_only_results", {})
		warns3 = st.session_state.get("p6_synth_only_warnings", [])
		group_map3_state = st.session_state.get("p6_synth_only_group_map", {})
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

			if (up_freq3_used is not None) and (up_vals3 is not None):
				all_fmins.append(float(np.nanmin(up_freq3_used)))
				all_fmaxs.append(float(np.nanmax(up_freq3_used)))
				all_ymins.append(float(np.nanmin(up_vals3)))
				all_ymaxs.append(float(np.nanmax(up_vals3)))
				fig3_all.add_trace(
					go.Scatter(
						x=up_freq3_used,
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
				guide_label3 = ""
				try:
					gvals3 = group_map3_state.get(f"{float(k3):.9f}", []) if isinstance(group_map3_state, dict) else []
					gvals3 = [float(v) for v in gvals3 if np.isfinite(float(v))]
					if gvals3:
						guide_label3 = _format_freqs_short(gvals3, max_show=6)
				except Exception:
					guide_label3 = ""
				with cols3[i_k % n_cols3]:
					if guide_label3:
						st.caption(f"Guide freqs in ROI: {guide_label3} GHz")
					else:
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
					if (up_freq3_used is not None) and (up_vals3 is not None):
						fig3.add_trace(go.Scatter(x=up_freq3_used, y=up_vals3, mode="lines", name="Uploaded synthetic", line=dict(dash="dot")))

					y_min_local = float(np.nanmin(y3))
					y_max_local = float(np.nanmax(y3))
					if (up_freq3_used is not None) and (up_vals3 is not None):
						m_up = (np.asarray(up_freq3_used, dtype=np.float64) >= (fmin3 - pad3)) & (np.asarray(up_freq3_used, dtype=np.float64) <= (fmax3 + pad3))
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
					format_func=lambda k: (
						f"Guide freqs: {_format_freqs_short([float(v) for v in group_map3_state.get(f'{float(k):.9f}', [])], max_show=6)} GHz"
						if isinstance(group_map3_state, dict) and group_map3_state.get(f"{float(k):.9f}", [])
						else f"target {float(k):.6f} GHz"
					),
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

	with tab_synth_batch:
		st.subheader("Generate Synthetic Spectra | CH3OCHO")
		st.caption("Generate spectra from selected ROIs/guide frequencies and parameter ranges, then save as TXT.")

		batch_mode_ui = st.radio(
			"Fitting mode",
			options=["Case 1: Synthetic only", "Case 2: Synthetic + noise"],
			index=0,
			horizontal=True,
			key="p6_synthbatch_mode_output",
		)
		batch_mode = "synthetic_plus_noise" if str(batch_mode_ui).strip().lower().endswith("noise") else "synthetic_only"
		batch_noise_scale = st.number_input(
			"Noise scale (for Synthetic + noise)",
			min_value=0.0,
			value=float(noise_scale),
			step=0.1,
			format="%.3f",
			key="p6_synthbatch_noise_scale",
			disabled=(batch_mode != "synthetic_plus_noise"),
		)

		if "p6_synthbatch_guide_input" not in st.session_state:
			st.session_state.p6_synthbatch_guide_input = _freqs_to_text([float(v) for v in target_freqs])

		signal_rois_sb = _collect_signal_rois_for_ui(signal_models_root, filter_file)
		guide_text_sb = st.text_input("Guide frequencies (GHz)", key="p6_synthbatch_guide_input")
		guide_freqs_sb = parse_freq_list(guide_text_sb)

		use_roi_centers_sb = st.checkbox("Add selected signal ROI centers", value=True, key="p6_synthbatch_use_roi_centers")
		roi_center_freqs_sb: List[float] = []
		if use_roi_centers_sb and signal_rois_sb:
			roi_opts_sb = list(range(len(signal_rois_sb)))
			default_sel_sb = roi_opts_sb
			sel_roi_pos_sb = st.multiselect(
				"Signal ROIs to include",
				options=roi_opts_sb,
				default=default_sel_sb,
				format_func=lambda i: f"ROI S{signal_rois_sb[i]['index']} | {signal_rois_sb[i]['lo']:.6f}–{signal_rois_sb[i]['hi']:.6f} GHz",
				key="p6_synthbatch_roi_multiselect",
			)
			for ii in sel_roi_pos_sb:
				cc = _roi_center_ghz(signal_rois_sb[int(ii)])
				if cc is not None and np.isfinite(float(cc)):
					roi_center_freqs_sb.append(float(cc))
		elif use_roi_centers_sb and (not signal_rois_sb):
			st.info("No signal ROIs detected yet. Check signal model/filter paths.")

		target_freqs_sb = _normalize_target_freqs_for_run([float(v) for v in (guide_freqs_sb + roi_center_freqs_sb)])
		if target_freqs_sb:
			st.caption("Target frequencies to generate: " + _freqs_to_text(target_freqs_sb))
		else:
			st.caption("Target frequencies to generate: (empty)")

		sb1, sb2, sb3, sb4 = st.columns(4)
		with sb1:
			logn_min_sb = st.number_input("LogN min", value=14.0, key="p6_synthbatch_logn_min")
			logn_max_sb = st.number_input("LogN max", value=19.5, key="p6_synthbatch_logn_max")
			logn_step_sb = st.number_input("LogN step", min_value=1e-6, value=0.1, step=0.01, format="%.6f", key="p6_synthbatch_logn_step")
		with sb2:
			tex_min_sb = st.number_input("Tex min", value=100.0, key="p6_synthbatch_tex_min")
			tex_max_sb = st.number_input("Tex max", value=380.0, key="p6_synthbatch_tex_max")
			tex_step_sb = st.number_input("Tex step", min_value=1e-6, value=20.0, step=1.0, format="%.6f", key="p6_synthbatch_tex_step")
		with sb3:
			velo_min_sb = st.number_input("Velocity min", value=90.0, key="p6_synthbatch_velo_min")
			velo_max_sb = st.number_input("Velocity max", value=105.0, key="p6_synthbatch_velo_max")
			velo_step_sb = st.number_input("Velocity step", min_value=1e-6, value=1.0, step=0.1, format="%.6f", key="p6_synthbatch_velo_step")
		with sb4:
			fwhm_min_sb = st.number_input("FWHM min", value=5.0, key="p6_synthbatch_fwhm_min")
			fwhm_max_sb = st.number_input("FWHM max", value=8.0, key="p6_synthbatch_fwhm_max")
			fwhm_step_sb = st.number_input("FWHM step", min_value=1e-6, value=2.0, step=0.1, format="%.6f", key="p6_synthbatch_fwhm_step")

		sb5, sb6, sb7 = st.columns(3)
		with sb5:
			candidate_mode_sb_ui = st.selectbox("Sampling mode", options=["Random", "Ordered grid"], index=0, key="p6_synthbatch_mode")
		is_ordered_grid_sb = str(candidate_mode_sb_ui).strip().lower().startswith("ordered")
		with sb6:
			n_spectra_sb = st.number_input("Number of spectra", min_value=1, max_value=1000000, value=200, step=10, key="p6_synthbatch_n", disabled=bool(is_ordered_grid_sb))
		with sb7:
			random_seed_sb = st.number_input("Random seed", min_value=0, value=42, step=1, key="p6_synthbatch_seed")

		logn_axis_sb = _axis_from_min_max_step(float(logn_min_sb), float(logn_max_sb), float(logn_step_sb))
		tex_axis_sb = _axis_from_min_max_step(float(tex_min_sb), float(tex_max_sb), float(tex_step_sb))
		velo_axis_sb = _axis_from_min_max_step(float(velo_min_sb), float(velo_max_sb), float(velo_step_sb))
		fwhm_axis_sb = _axis_from_min_max_step(float(fwhm_min_sb), float(fwhm_max_sb), float(fwhm_step_sb))
		total_grid_sb = int(logn_axis_sb.size * tex_axis_sb.size * velo_axis_sb.size * fwhm_axis_sb.size)
		st.caption(
			f"Ordered grid total spectra (from steps): {total_grid_sb} = "
			f"{logn_axis_sb.size}×{tex_axis_sb.size}×{velo_axis_sb.size}×{fwhm_axis_sb.size}"
		)

		out_dir_sb = st.text_input(
			"Output directory",
			value=r"D:\4.DATASETS\OBSEMU_Synthetic_Spectra",
			key="p6_synthbatch_outdir",
		)

		if st.button("Generate and save synthetic spectra", type="primary", key="p6_synthbatch_run"):
			if not target_freqs_sb:
				st.error("Define at least one target frequency (Guide frequencies and/or selected ROIs).")
			elif not os.path.isfile(filter_file):
				st.error(f"Filter file not found: {filter_file}")
			elif (not signal_models_root) or ((not os.path.isfile(signal_models_root)) and (not os.path.isdir(signal_models_root))):
				st.error("Signal models source invalid.")
			elif (batch_mode == "synthetic_plus_noise") and (not _is_valid_noise_source(noise_models_root)):
				st.error("Noise models source invalid for Synthetic + noise mode.")
			elif is_ordered_grid_sb and total_grid_sb <= 0:
				st.error("Ordered grid from current step settings is empty.")
			else:
				try:
					os.makedirs(str(out_dir_sb), exist_ok=True)

					noise_models_loaded_sb: List[tuple] = []
					if batch_mode == "synthetic_plus_noise":
						entries_sb = _list_noise_model_entries(noise_models_root)
						for ent_sb in entries_sb:
							try:
								m_sb, sy_sb, c_sb = _load_noisenn_from_entry(ent_sb)
								noise_models_loaded_sb.append((m_sb, sy_sb, c_sb))
							except Exception as e:
								warn_agg_local = str(e)
								# Delay surfacing as aggregated warning.
								pass

					ranges_sb = {
						"logn_min": float(min(logn_min_sb, logn_max_sb)),
						"logn_max": float(max(logn_min_sb, logn_max_sb)),
						"tex_min": float(min(tex_min_sb, tex_max_sb)),
						"tex_max": float(max(tex_min_sb, tex_max_sb)),
						"velo_min": float(min(velo_min_sb, velo_max_sb)),
						"velo_max": float(max(velo_min_sb, velo_max_sb)),
						"fwhm_min": float(min(fwhm_min_sb, fwhm_max_sb)),
						"fwhm_max": float(max(fwhm_min_sb, fwhm_max_sb)),
					}
					mode_sb = "ordered_grid" if str(candidate_mode_sb_ui).strip().lower().startswith("ordered") else "random"
					if mode_sb == "ordered_grid":
						g0_sb, g1_sb, g2_sb, g3_sb = np.meshgrid(
							logn_axis_sb,
							tex_axis_sb,
							velo_axis_sb,
							fwhm_axis_sb,
							indexing="ij",
						)
						X_sb = np.stack(
							[g0_sb.reshape(-1), g1_sb.reshape(-1), g2_sb.reshape(-1), g3_sb.reshape(-1)],
							axis=1,
						).astype(np.float32)
					else:
						X_sb = _sample_fit_candidates(
							n_samples=int(n_spectra_sb),
							ranges=ranges_sb,
							seed=int(random_seed_sb),
							mode=mode_sb,
						)

					prog = st.progress(0.0, text="Generating synthetic spectra...")
					st.caption("Live preview: one generated spectrum is updated every 5 simulations.")
					preview_slot_sb = st.empty()
					n_ok_sb = 0
					n_fail_sb = 0
					warn_agg_sb: List[str] = []
					for i_sb in range(int(X_sb.shape[0])):
						x_logN = float(X_sb[i_sb, 0])
						x_tex = float(X_sb[i_sb, 1])
						x_velo = float(X_sb[i_sb, 2])
						x_fwhm = float(X_sb[i_sb, 3])
						x_feat_sb = np.asarray([[x_logN, x_tex, x_velo, x_fwhm]], dtype=np.float32)
						res_sb, warns_sb = _generate_synthetic_spectra_for_targets(
							signal_models_source=str(signal_models_root),
							filter_file=str(filter_file),
							target_freqs=[float(v) for v in target_freqs_sb],
							x_features=[x_logN, x_tex, x_velo, x_fwhm],
							pred_mode=DEFAULT_PRED_MODE,
							selected_model_name=DEFAULT_SELECTED_MODEL_NAME,
							allow_nearest=bool(allow_nearest),
						)

						segs_sb: List[Tuple[float, np.ndarray, np.ndarray]] = []
						for _, item_sb in (res_sb or {}).items():
							ff_sb = np.asarray(item_sb.get("freq", []), dtype=np.float64).reshape(-1)
							yy_sb = np.asarray(item_sb.get("synthetic", []), dtype=np.float64).reshape(-1)
							if ff_sb.size < 2 or yy_sb.size != ff_sb.size:
								continue
							m_sb = np.isfinite(ff_sb) & np.isfinite(yy_sb)
							if int(np.count_nonzero(m_sb)) < 2:
								continue
							ff_sb = ff_sb[m_sb]
							yy_sb = yy_sb[m_sb]
							if batch_mode == "synthetic_plus_noise":
								try:
									y_noise_sb = _add_noise_batch_for_target(
										noise_models_loaded=noise_models_loaded_sb,
										roi_freq=np.asarray(ff_sb, dtype=np.float64),
										y_syn_batch=np.asarray(yy_sb, dtype=np.float32).reshape(1, -1),
										x_candidates=x_feat_sb,
										noise_scale=float(batch_noise_scale),
									).reshape(-1)
									yy_sb = np.asarray(yy_sb, dtype=np.float64) + np.asarray(y_noise_sb, dtype=np.float64)
								except Exception as e:
									warn_agg_sb.append(f"Noise add failed (sample #{int(i_sb+1)}): {e}")
							segs_sb.append((float(np.nanmin(ff_sb)), ff_sb, yy_sb))

						if not segs_sb:
							n_fail_sb += 1
							warn_agg_sb.extend([str(w) for w in (warns_sb or [])])
						else:
							segs_sb = sorted(segs_sb, key=lambda t: t[0])
							freq_segments_sb = [t[1] for t in segs_sb]
							spec_segments_sb = [t[2] for t in segs_sb]
							# One TXT per simulation: concatenate all ROI/target segments into a
							# single spectrum, inserting zero bridges between windows.
							f_cat_sb, y_cat_sb = concat_segments_with_zero_gaps(
								freqs_segments=freq_segments_sb,
								spec_segments=spec_segments_sb,
								insert_zero_gaps=True,
							)
							if f_cat_sb.size < 2 or y_cat_sb.size != f_cat_sb.size:
								n_fail_sb += 1
								warn_agg_sb.append(
									f"Simulation #{int(i_sb+1)} could not be concatenated into a valid single spectrum."
								)
								prog.progress(float(i_sb + 1) / float(max(1, X_sb.shape[0])), text=f"Generating synthetic spectra... ({i_sb+1}/{int(X_sb.shape[0])})")
								continue
							out_name_sb = (
								f"synthetic_{i_sb+1:04d}_"
								f"logN_{x_logN:.4f}_tex_{x_tex:.4f}_"
								f"fwhm_{x_fwhm:.4f}_velo_{x_velo:.4f}.txt"
							)
							out_path_sb = os.path.join(str(out_dir_sb), out_name_sb)
							np.savetxt(out_path_sb, np.column_stack([f_cat_sb, y_cat_sb]), fmt="%.10f")
							n_ok_sb += 1

							if ((i_sb + 1) % 5 == 0) or ((i_sb + 1) == int(X_sb.shape[0])):
								fig_prev_sb = go.Figure()
								fig_prev_sb.add_trace(
									go.Scatter(
										x=np.asarray(f_cat_sb, dtype=np.float64),
										y=np.asarray(y_cat_sb, dtype=np.float64),
										mode="lines",
										name=f"Sim #{int(i_sb+1)}",
										line=dict(width=1.3, color="#1f77b4"),
									)
								)
								fig_prev_sb.update_layout(
									title=f"Live spectrum preview (simulation #{int(i_sb+1)})",
									xaxis_title="Frequency (GHz)",
									yaxis_title="Intensity",
									template="plotly_white",
									height=320,
									margin=dict(l=40, r=20, t=45, b=40),
								)
								preview_slot_sb.plotly_chart(fig_prev_sb, use_container_width=True)

						prog.progress(float(i_sb + 1) / float(max(1, X_sb.shape[0])), text=f"Generating synthetic spectra... ({i_sb+1}/{int(X_sb.shape[0])})")

					prog.empty()
					if n_ok_sb > 0:
						st.success(f"Synthetic spectra generation completed. Saved: {n_ok_sb} | Failed: {n_fail_sb} | Output: {out_dir_sb}")
					else:
						st.error("No synthetic spectra could be generated with current settings.")
					if warn_agg_sb:
						with st.expander("Show generation warnings"):
							st.text("\n".join(warn_agg_sb[:200]))
				except Exception as e:
					st.error(f"Synthetic batch generation failed: {e}")

	with tab_eval16:
		st.subheader("ROI Ranking Eval")
		st.caption("Independent ranking: uses only model artifacts and the uploaded observed spectrum.")
		st.caption(f"Model source mode in use: {source_mode_label}")

		eval16_model_dir = str(roi_rank_model_dir)
		with st.expander("Optional: upload ROI ranking model (.h5 bundle or 1.5 artifacts)", expanded=False):
			st.caption("If provided, this upload overrides the current ROI ranking model directory for this evaluation.")
			e16_up_rank_bundle_h5 = st.file_uploader("Upload ROI ranking model bundle (.h5)", type=["h5", "hdf5"], key="p6_eval16_up_rank_bundle_h5")

			e16_uploaded_rank_bundle_h5 = _save_uploaded_file_to_temp(e16_up_rank_bundle_h5, "eval16_roi_rank_bundle_h5")
			e16_rank_dir_uploaded = None
			e16_rank_upload_warning = None
			if e16_uploaded_rank_bundle_h5:
				e16_rank_dir_uploaded, e16_rank_upload_warning = _prepare_roi_rank_model_dir_from_h5_bundle(e16_uploaded_rank_bundle_h5)
			if e16_rank_dir_uploaded:
				eval16_model_dir = str(e16_rank_dir_uploaded)
				st.success("ROI ranking model uploaded and ready for this evaluation.")
			if e16_rank_upload_warning:
				st.warning(str(e16_rank_upload_warning))

		eval16_model_dir_resolved, eval16_model_dir_warn = _resolve_roi_rank_model_dir(str(eval16_model_dir))
		if eval16_model_dir_warn:
			st.warning(str(eval16_model_dir_warn))
		st.caption(f"ROI ranking model source in use: {eval16_model_dir}")
		st.caption(f"ROI ranking artifacts directory in use: {eval16_model_dir_resolved}")

		eval16_model_weights, eval16_model_scalers, eval16_model_meta = _roi_rank_artifact_paths(str(eval16_model_dir_resolved))

		upload_obs_eval16 = st.file_uploader(
			"Upload observed spectrum (.txt/.dat/.csv)",
			type=None,
			key="p6_eval16_upload_obs",
		)
		eval16_out_dir = os.path.join(str(eval16_model_dir_resolved), "eval_new_spectrum")

		c16g, c16h = st.columns(2)
		with c16g:
			eval16_shift_kms = st.number_input(
				"Obs shift (km/s)",
				value=-98.0,
				step=0.1,
				format="%.4f",
				key="p6_eval16_shift_kms",
			)
		with c16h:
			eval16_shift_mode = st.selectbox(
				"Shift mode",
				options=["per_frequency", "spw_center"],
				index=0,
				key="p6_eval16_shift_mode",
			)

		eval16_allow_nearest = st.checkbox(
			"Allow nearest ROI if no overlap",
			value=False,
			key="p6_eval16_allow_nearest",
			disabled=True,
		)

		run_eval16 = st.button("Run ROI Ranking Eval", type="primary", key="p6_eval16_run")
		if run_eval16:
			try:
				if upload_obs_eval16 is None:
					raise RuntimeError("Upload observed spectrum is required.")
				tmp_obs = _save_uploaded_file_to_temp(upload_obs_eval16, "eval16_obs")
				if not tmp_obs:
					raise RuntimeError("Could not save uploaded observed spectrum to temporary file.")
				f_obs_raw, y_obs_raw, parse_err = _read_uploaded_spectrum_any(upload_obs_eval16)
				if parse_err is not None or f_obs_raw is None or y_obs_raw is None:
					raise RuntimeError(f"Could not parse uploaded observed spectrum: {parse_err}")
				f_obs = np.asarray(f_obs_raw, dtype=np.float64)
				y_obs = np.asarray(y_obs_raw, dtype=np.float64)
				if str(eval16_shift_mode).strip().lower() == "spw_center":
					f_obs = _apply_velocity_shift_by_spw_centers_segmented(f_obs, float(eval16_shift_kms), gap_factor=20.0)
				else:
					f_obs = _apply_velocity_shift_to_frequency(f_obs, float(eval16_shift_kms))

				artifacts_error = _validate_roi_rank_artifacts(str(eval16_model_dir_resolved))
				if artifacts_error:
					raise FileNotFoundError(str(artifacts_error))

				with open(eval16_model_meta, "r", encoding="utf-8") as fm16:
					meta16 = json.load(fm16)
				input_dim16 = int(meta16.get("input_dim", 14))
				hidden16 = [int(v) for v in meta16.get("hidden_sizes", [128, 64])]
				dropout16 = float(meta16.get("dropout", 0.15))
				eval16_resample_len = int(meta16.get("feature_resample_len", meta16.get("roi_feature_resample_len", 128)))
				eval16_resample_len = int(max(8, min(4096, eval16_resample_len)))

				z16 = np.load(eval16_model_scalers)
				x_mean16 = np.asarray(z16["x_mean"], dtype=np.float64)
				x_scale16 = np.asarray(z16["x_scale"], dtype=np.float64)
				y_mean16 = np.asarray(z16["y_mean"], dtype=np.float64)
				y_scale16 = np.asarray(z16["y_scale"], dtype=np.float64)

				m16 = _RoiRankNNLite(input_size=input_dim16, hidden_sizes=hidden16, output_size=2, dropout=dropout16).cpu()
				m16.load_state_dict(torch.load(eval16_model_weights, map_location="cpu"))
				m16.eval()

				target_freqs_model = _load_model_target_freqs_for_ranking(str(eval16_model_dir_resolved), str(eval16_model_meta))
				roi_defs16 = _build_roi_defs_from_model_targets(f_obs, target_freqs_model)
				if not roi_defs16:
					# Fallback only if model does not expose target frequencies.
					roi_defs16 = _build_auto_roi_defs(f_obs)
				if not roi_defs16:
					raise RuntimeError("Could not build ROI windows from uploaded spectrum.")

				rows16 = []
				n_spw16 = int(max(1, len(roi_defs16)))
				for j16, rd16 in enumerate(roi_defs16, start=1):
					idx16 = np.asarray(rd16["idx"], dtype=int)
					oo16 = np.asarray(y_obs[idx16], dtype=np.float64)
					valid16 = np.isfinite(oo16)
					if int(np.count_nonzero(valid16)) < 4:
						continue
					oo16 = oo16[valid16]
					# Match training-time behavior when spectral channel counts vary.
					# The ranking model is more stable if each ROI is represented with
					# a consistent channel length before feature extraction.
					if int(oo16.size) != int(eval16_resample_len):
						oo16 = _resample_1d_by_index_float64(oo16, int(eval16_resample_len))
					if _is_invalid_obs_roi_line_rank(oo16):
						rows16.append({
							"roi_name": str(rd16["roi_name"]),
							"target_freq_ghz": float(rd16["target_freq_ghz"]),
							"f_min_ghz": float(rd16["f_min_ghz"]),
							"f_max_ghz": float(rd16["f_max_ghz"]),
							"pred_rmse": float("nan"),
							"pred_intensity": float("nan"),
							"n_points": int(oo16.size),
							"is_invalid_obs_line": True,
							"ranking_note": "excluded_from_model_order_flat_or_zero_obs_roi",
						})
						continue
					feat16 = _build_obs_features_for_rank(oo16, spw_idx=j16, n_spw=n_spw16)
					if feat16 is None:
						continue
					feat_s16 = (feat16.astype(np.float64) - x_mean16) / np.maximum(1e-12, x_scale16)
					feat_s16 = feat_s16.astype(np.float32).reshape(1, -1)
					with torch.no_grad():
						yp_s16 = m16(torch.from_numpy(feat_s16)).cpu().numpy().reshape(-1)
					yp16 = yp_s16.astype(np.float64) * y_scale16 + y_mean16
					rows16.append({
						"roi_name": str(rd16["roi_name"]),
						"target_freq_ghz": float(rd16["target_freq_ghz"]),
						"f_min_ghz": float(rd16["f_min_ghz"]),
						"f_max_ghz": float(rd16["f_max_ghz"]),
						"pred_rmse": float(yp16[0]),
						"pred_intensity": float(yp16[1]),
						"n_points": int(oo16.size),
						"is_invalid_obs_line": False,
						"ranking_note": "ok",
					})

				if not rows16:
					raise RuntimeError("No evaluable ROIs from uploaded spectrum.")

				rows16 = sorted(
					rows16,
					key=lambda d: (
						1 if bool(d.get("is_invalid_obs_line", False)) else 0,
						(-float(d["pred_rmse"]) if np.isfinite(float(d.get("pred_rmse", np.nan))) else np.inf),
						(float(d["pred_intensity"]) if np.isfinite(float(d.get("pred_intensity", np.nan))) else np.inf),
					),
				)
				for i16, r16 in enumerate(rows16, start=1):
					r16["rank"] = int(i16)

				topk16 = int(len(rows16))
				auto_sel_n = int(min(6, len(rows16)))
				auto_sel_map = {}
				for rr in rows16:
					row_id = f"{str(rr.get('roi_name', ''))}|{float(rr.get('target_freq_ghz', np.nan)):.9f}"
					auto_sel_map[row_id] = (int(rr.get("rank", 10**9)) <= auto_sel_n)
				st.session_state.p6_eval16_select_map = auto_sel_map
				st.session_state.p6_eval16_rows = rows16
				st.session_state.p6_eval16_topk = int(topk16)
				st.session_state.p6_eval16_f_obs = np.asarray(f_obs, dtype=np.float64)
				st.session_state.p6_eval16_y_obs = np.asarray(y_obs, dtype=np.float64)
				st.session_state.p6_eval16_last_log = f"ROI ranking completed | ROIs={len(rows16)} | TopK={topk16}"
				st.session_state.p6_eval16_last_out_dir = str(eval16_out_dir)
				st.success("ROI ranking evaluation completed.")
			except Exception as e:
				st.session_state.p6_eval16_last_log = ""
				st.session_state.p6_eval16_rows = []
				st.error(f"1.6 evaluation failed: {e}")

		last_log16 = str(st.session_state.get("p6_eval16_last_log", ""))
		if last_log16:
			with st.expander("Show execution log"):
				st.text(last_log16)

		rows16 = st.session_state.get("p6_eval16_rows", [])
		f_obs16 = st.session_state.get("p6_eval16_f_obs", None)
		y_obs16 = st.session_state.get("p6_eval16_y_obs", None)
		topk16 = int(st.session_state.get("p6_eval16_topk", 0))
		if isinstance(rows16, list) and rows16:
			st.markdown("**ROI ranking output**")

			if "p6_eval16_select_map" not in st.session_state:
				st.session_state.p6_eval16_select_map = {}
			sel_map = st.session_state.get("p6_eval16_select_map", {}) if isinstance(st.session_state.get("p6_eval16_select_map", {}), dict) else {}

			row_ids16 = [
				f"{str(rr.get('roi_name', ''))}|{float(rr.get('target_freq_ghz', np.nan)):.9f}"
				for rr in rows16
			]
			csel1, csel2, _ = st.columns([1.2, 1.2, 3.6])
			with csel1:
				if st.button("Select all rows", key="p6_eval16_select_all_rows"):
					st.session_state.p6_eval16_select_map = {rid: True for rid in row_ids16}
					st.rerun()
			with csel2:
				if st.button("Clear selection", key="p6_eval16_clear_selection"):
					st.session_state.p6_eval16_select_map = {rid: False for rid in row_ids16}
					st.rerun()

			table_rows16 = []
			for rr16 in rows16:
				row_id = f"{str(rr16.get('roi_name', ''))}|{float(rr16.get('target_freq_ghz', np.nan)):.9f}"
				table_rows16.append({
					"Select": bool(sel_map.get(row_id, False)),
					"rank": int(rr16.get("rank", 0)),
					"roi_name": str(rr16.get("roi_name", "")),
					"target_freq_ghz": float(rr16.get("target_freq_ghz", np.nan)),
					"f_min_ghz": float(rr16.get("f_min_ghz", np.nan)),
					"f_max_ghz": float(rr16.get("f_max_ghz", np.nan)),
					"pred_rmse": float(rr16.get("pred_rmse", np.nan)),
					"pred_intensity": float(rr16.get("pred_intensity", np.nan)),
					"n_points": int(rr16.get("n_points", 0)),
					"ranking_note": str(rr16.get("ranking_note", "")),
				})

			edited16 = st.data_editor(
				table_rows16,
				use_container_width=True,
				num_rows="fixed",
				hide_index=True,
				disabled=["rank", "roi_name", "target_freq_ghz", "f_min_ghz", "f_max_ghz", "pred_rmse", "pred_intensity", "n_points", "ranking_note"],
				key="p6_eval16_table_editor",
			)

			selected_freqs16: List[float] = []
			new_map16 = {}
			if isinstance(edited16, list):
				edited_rows16 = edited16
			elif hasattr(edited16, "to_dict"):
				try:
					edited_rows16 = edited16.to_dict("records")
				except Exception:
					edited_rows16 = []
			else:
				edited_rows16 = []
			for er in edited_rows16:
				row_id = f"{str(er.get('roi_name', ''))}|{float(er.get('target_freq_ghz', np.nan)):.9f}"
				is_sel = bool(er.get("Select", False))
				new_map16[row_id] = is_sel
				if is_sel:
					try:
						fv = float(er.get("target_freq_ghz", np.nan))
						if np.isfinite(fv):
							selected_freqs16.append(float(fv))
					except Exception:
						pass
			st.session_state.p6_eval16_select_map = new_map16

			selected_freqs16 = _normalize_target_freqs_for_run(selected_freqs16)
			st.markdown("**Guide frequencies from selected ROIs**")
			if selected_freqs16:
				st.caption(_freqs_to_text(selected_freqs16))
			else:
				st.caption("(none selected)")

			st.markdown(
				"""
<style>
#p6_eval16_add_btn_anchor + div[data-testid="stButton"] > button {
  background-color: #d62728 !important;
  color: white !important;
  border: 1px solid #d62728 !important;
}
#p6_eval16_add_btn_anchor + div[data-testid="stButton"] > button:hover {
  background-color: #b61f21 !important;
  border-color: #b61f21 !important;
}
</style>
<div id="p6_eval16_add_btn_anchor"></div>
""",
				unsafe_allow_html=True,
			)

			if st.button("Add selected frequencies to Guide frequencies (all tabs)", key="p6_eval16_add_selected_to_guides"):
				if not selected_freqs16:
					st.warning("Select at least one ROI first.")
				else:
					_propagate_selected_freqs_to_all_guides(selected_freqs16)
					st.success("Selected frequencies added to guide frequencies in all tabs.")
					st.rerun()

			json_bytes16 = json.dumps(rows16, ensure_ascii=False, indent=2).encode("utf-8")
			csv_io16 = io.StringIO()
			fieldnames16 = [
				"rank", "roi_name", "target_freq_ghz", "f_min_ghz", "f_max_ghz",
				"pred_rmse", "pred_intensity", "n_points", "is_invalid_obs_line", "ranking_note",
			]
			w16 = csv.DictWriter(csv_io16, fieldnames=fieldnames16)
			w16.writeheader()
			for rr16 in rows16:
				w16.writerow({k: rr16.get(k, None) for k in fieldnames16})
			csv_bytes16 = csv_io16.getvalue().encode("utf-8")

			d_cols16 = st.columns(2)
			with d_cols16[0]:
				st.download_button(
					"Download ranking JSON",
					data=json_bytes16,
					file_name="new_spectrum_roi_ranking.json",
					mime="application/json",
					key="p6_eval16_dl_json",
				)
			with d_cols16[1]:
				st.download_button(
					"Download ranking CSV",
					data=csv_bytes16,
					file_name="new_spectrum_roi_ranking.csv",
					mime="text/csv",
					key="p6_eval16_dl_csv",
				)

			if isinstance(f_obs16, np.ndarray) and isinstance(y_obs16, np.ndarray):
				topk16 = int(len(rows16)) if int(topk16) <= 0 else int(max(1, min(int(topk16), len(rows16))))

				# 1) RMSE vs Intensity first (smaller vertical height)
				labels16 = [f"#{int(r['rank'])}\n{float(r['target_freq_ghz']):.4f}" for r in rows16[:topk16]]
				v_rmse16 = np.array([float(r["pred_rmse"]) for r in rows16[:topk16]], dtype=np.float64)
				v_int16 = np.array([float(r["pred_intensity"]) for r in rows16[:topk16]], dtype=np.float64)
				x16 = np.arange(len(labels16))
				fig_bar16 = go.Figure()
				fig_bar16.add_trace(go.Bar(
					x=x16,
					y=v_rmse16,
					name="Pred RMSE",
					marker_color="#d62728",
					opacity=0.85,
					width=0.28,
					yaxis="y1",
					offsetgroup="rmse",
				))
				fig_bar16.add_trace(go.Bar(
					x=x16,
					y=v_int16,
					name="Pred Intensity",
					marker_color="#1f77b4",
					opacity=0.65,
					width=0.28,
					yaxis="y2",
					offsetgroup="intensity",
				))
				fig_bar16.update_layout(
					title="Top-ranked ROIs: predicted RMSE vs intensity",
					xaxis=dict(
						title="ROI",
						tickmode="array",
						tickvals=list(x16),
						ticktext=labels16,
					),
					yaxis=dict(title="Pred RMSE"),
					yaxis2=dict(title="Pred Intensity", overlaying="y", side="right"),
					barmode="group",
					template="plotly_white",
					height=300,
					margin=dict(l=40, r=40, t=45, b=40),
				)
				st.plotly_chart(fig_bar16, width="stretch", key="p6_eval16_bar_plot")

				# 2) Interactive global overlay
				shapes16 = []
				for rr16 in rows16:
					shapes16.append({
						"type": "rect",
						"xref": "x",
						"yref": "paper",
						"x0": float(rr16["f_min_ghz"]),
						"x1": float(rr16["f_max_ghz"]),
						"y0": 0.0,
						"y1": 1.0,
						"fillcolor": "rgba(120,120,120,0.12)",
						"line": {"width": 0},
						"layer": "below",
					})
				for rr16 in rows16[:topk16]:
					shapes16.append({
						"type": "rect",
						"xref": "x",
						"yref": "paper",
						"x0": float(rr16["f_min_ghz"]),
						"x1": float(rr16["f_max_ghz"]),
						"y0": 0.0,
						"y1": 1.0,
						"fillcolor": "rgba(44,160,44,0.22)",
						"line": {"width": 0},
						"layer": "below",
					})
				fig_ov16 = go.Figure()
				fig_ov16.add_trace(go.Scatter(
					x=np.asarray(f_obs16, dtype=np.float64),
					y=np.asarray(y_obs16, dtype=np.float64),
					mode="lines",
					name="Observed",
					line=dict(color="orange", width=1.4),
				))
				fig_ov16.update_layout(
					title="Observed spectrum with best-ranked ROIs",
					xaxis_title="Frequency (GHz)",
					yaxis_title="Intensity",
					shapes=shapes16,
					template="plotly_white",
					height=420,
					margin=dict(l=40, r=20, t=45, b=40),
				)
				st.plotly_chart(fig_ov16, width="stretch", key="p6_eval16_overlay_plot")

				# 3) Interactive per-ROI zoom panels (similar to fitting tab)
				st.markdown("**Zoom by ROI of interest (Top-K)**")
				n_cols16 = 2 if int(topk16) <= 4 else 3
				cols16 = st.columns(n_cols16)
				for ip16, rr16 in enumerate(rows16[:topk16]):
					lo16 = float(rr16["f_min_ghz"])
					hi16 = float(rr16["f_max_ghz"])
					span16 = float(max(1e-9, hi16 - lo16))
					pad16 = 0.18 * span16
					m16 = (f_obs16 >= (lo16 - pad16)) & (f_obs16 <= (hi16 + pad16))
					fx16 = np.asarray(f_obs16[m16], dtype=np.float64)
					oy16 = np.asarray(y_obs16[m16], dtype=np.float64)
					with cols16[ip16 % n_cols16]:
						fig_z16 = go.Figure()
						fig_z16.add_trace(go.Scatter(x=fx16, y=oy16, mode="lines", name="Observed", line=dict(color="orange", width=1.2)))
						fig_z16.add_shape(
							type="rect",
							xref="x",
							yref="paper",
							x0=lo16,
							x1=hi16,
							y0=0.0,
							y1=1.0,
							fillcolor="rgba(44,160,44,0.20)",
							line=dict(width=0),
							layer="below",
						)
						fig_z16.update_layout(
							title=f"#{int(rr16['rank'])} | {float(rr16['target_freq_ghz']):.6f} GHz | pred_rmse={float(rr16['pred_rmse']):.4g}",
							xaxis=dict(title="Frequency (GHz)", range=[lo16 - pad16, hi16 + pad16]),
							yaxis=dict(title="Intensity"),
							template="plotly_white",
							height=300,
							margin=dict(l=40, r=20, t=42, b=36),
						)
						st.plotly_chart(fig_z16, width="stretch", key=f"p6_eval16_zoom_{ip16}")

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
			st.session_state.p6_guide_freqs_fit_input = _freqs_to_text([float(v) for v in DEFAULT_TARGET_FREQS])

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

		obs_freq_fit_raw = None if obs_freq_fit is None else np.asarray(obs_freq_fit, dtype=np.float64).copy()
		obs_freq_fit_used = None if obs_freq_fit_raw is None else np.asarray(obs_freq_fit_raw, dtype=np.float64).copy()
		obs_vals_fit_used = None if obs_vals_fit is None else np.asarray(obs_vals_fit, dtype=np.float64).copy()
		if (obs_freq_fit_used is not None) and bool(obs_shift_enabled):
			if str(obs_shift_mode).strip().lower() == "spw_center":
				obs_freq_fit_used = _apply_velocity_shift_by_spw_center(obs_freq_fit_used, float(obs_shift_kms))
			else:
				obs_freq_fit_used = _apply_velocity_shift_to_frequency(obs_freq_fit_used, float(obs_shift_kms))
			st.caption(f"Observational spectrum shifted by {float(obs_shift_kms):+.4f} km/s using mode: {str(obs_shift_mode)}")

		fit_shift_optimize = st.checkbox(
			"Optimize observational shift in first round",
			value=False,
			key="p6_fit_shift_optimize",
		)
		fso1, fso2 = st.columns(2)
		with fso1:
			fit_shift_scan_half_window = st.number_input(
				"Shift scan half-window (km/s)",
				min_value=0.0,
				value=2.0,
				step=0.1,
				format="%.3f",
				key="p6_fit_shift_scan_half_window",
				disabled=not bool(fit_shift_optimize),
			)
		with fso2:
			fit_shift_scan_points = st.number_input(
				"Shift scan points",
				min_value=3,
				max_value=21,
				value=5,
				step=2,
				key="p6_fit_shift_scan_points",
				disabled=not bool(fit_shift_optimize),
			)

		with st.expander("Fitting search ranges and speed settings", expanded=False):
			fit_global_mode_ui = st.selectbox(
				"Global fit strategy",
				options=["Per-ROI aggregate", "Concatenated ROIs (single objective)"],
				index=1,
				key="p6_fit_global_mode",
			)
			fit_global_mode_map = {
				"Per-ROI aggregate": "per_roi",
				"Concatenated ROIs (single objective)": "concatenated",
			}
			fit_criterion_ui = st.selectbox(
				"Fitting criterion",
				options=["MAE", "RMSE", "CHI_like", "R2"],
				index=1,
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
			fit_local_opt_method_ui = st.selectbox(
				"Local optimizer after candidate search",
				options=["None", "TRF (Trust Region Reflective)"],
				index=0,
				key="p6_fit_local_opt_method",
			)
			fit_local_opt_method_map = {
				"None": "none",
				"TRF (Trust Region Reflective)": "trf",
			}
			fit_local_opt_max_nfev = st.number_input(
				"Local optimizer max evaluations",
				min_value=8,
				max_value=200,
				value=24,
				step=4,
				key="p6_fit_local_opt_max_nfev",
				disabled=(str(fit_local_opt_method_ui) == "None"),
			)
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
				n_candidates_fit = st.number_input("Number of candidates", min_value=50, max_value=4000, value=300, step=50, key="p6_fit_n_candidates")
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
					fit_candidate_mode_internal = str(fit_candidate_mode_map.get(str(fit_candidate_mode_ui), "random"))
					X_shared_fit = _sample_fit_candidates(
						n_samples=int(n_candidates_fit),
						ranges=ranges_fit,
						seed=int(seed_fit),
						mode=str(fit_candidate_mode_internal),
					)

					noise_models_shared_fit = None
					if str(fit_case_mode).strip().lower() == "synthetic_plus_noise":
						entries_fit = _list_noise_model_entries(str(noise_models_root))
						nm_fit = []
						for efit in entries_fit:
							try:
								mfit, syfit, cfit = _load_noisenn_from_entry(efit)
								mfit.eval()
								nm_fit.append((mfit, syfit, cfit))
							except Exception:
								continue
						noise_models_shared_fit = nm_fit

					pkg_cache_shared_fit: Dict[str, object] = {}

					shift_values: List[float] = []
					if bool(fit_shift_optimize) and bool(obs_shift_enabled) and (obs_freq_fit_raw is not None):
						n_pts = int(max(3, min(21, int(fit_shift_scan_points))))
						if (n_pts % 2) == 0:
							n_pts += 1
						half = float(max(0.0, float(fit_shift_scan_half_window)))
						shift_values = [
							float(v) for v in np.linspace(float(obs_shift_kms) - half, float(obs_shift_kms) + half, num=n_pts)
						]
					else:
						shift_values = [float(obs_shift_kms)]

					fit_result = None
					best_result = None
					best_shift = None
					best_obj = np.inf
					n_ok = 0
					for sh in shift_values:
						obs_freq_trial = np.asarray(obs_freq_fit_used, dtype=np.float64)
						if (obs_freq_fit_raw is not None) and bool(obs_shift_enabled):
							obs_freq_trial = np.asarray(obs_freq_fit_raw, dtype=np.float64).copy()
							if str(obs_shift_mode).strip().lower() == "spw_center":
								obs_freq_trial = _apply_velocity_shift_by_spw_center(obs_freq_trial, float(sh))
							else:
								obs_freq_trial = _apply_velocity_shift_to_frequency(obs_freq_trial, float(sh))

						res_trial = _run_roi_fitting(
							signal_models_source=str(signal_models_root),
							noise_models_root=str(noise_models_root),
							filter_file=str(filter_file),
							target_freqs=[float(v) for v in guide_freqs_fit],
							obs_freq=np.asarray(obs_freq_trial, dtype=np.float64),
							obs_intensity=np.asarray(obs_vals_fit_used, dtype=np.float64),
							case_mode=str(fit_case_mode),
							fit_criterion=str(fit_criterion_ui).strip().lower(),
							global_weight_mode=str(fit_weight_mode_map.get(str(fit_weight_mode_ui), "inverse_best_error")),
							global_search_mode=str(fit_global_mode_map.get(str(fit_global_mode_ui), "concatenated")),
							candidate_mode=str(fit_candidate_mode_internal),
							n_candidates=int(n_candidates_fit),
							ranges=ranges_fit,
							noise_scale=float(noise_scale),
							allow_nearest=bool(allow_nearest),
							seed=int(seed_fit),
							x_candidates_override=np.asarray(X_shared_fit, dtype=np.float32),
							noise_models_loaded_override=noise_models_shared_fit,
							pkg_cache_override=pkg_cache_shared_fit,
							local_optimizer_method=str(fit_local_opt_method_map.get(str(fit_local_opt_method_ui), "none")),
							local_optimizer_max_nfev=int(fit_local_opt_max_nfev),
							refine_after_first_fit=False,
						)
						if isinstance(res_trial, dict) and bool(res_trial.get("ok", False)):
							n_ok += 1
							obj = float(res_trial.get("best_global_mean_objective", np.inf))
							if np.isfinite(obj) and (obj < best_obj):
								best_obj = float(obj)
								best_shift = float(sh)
								best_result = res_trial
						if fit_result is None:
							fit_result = res_trial

					if best_result is not None:
						fit_result = best_result
					if not isinstance(fit_result, dict):
						fit_result = {"ok": False, "message": "No fitting result available.", "warnings": []}

					fit_result["obs_shift_scan_enabled"] = bool(fit_shift_optimize and obs_shift_enabled)
					fit_result["obs_shift_scan_values_kms"] = [float(v) for v in shift_values]
					fit_result["obs_shift_scan_n_ok"] = int(n_ok)
					fit_result["obs_shift_kms_used"] = float(best_shift if best_shift is not None else obs_shift_kms)
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
					f"Local optimizer: {fit_result.get('local_optimizer_method', 'none')} | "
					f"Candidates: {int(fit_result.get('n_candidates', 0))} | "
					f"Guide freqs input: {int(fit_result.get('n_guide_freqs_input', 0))} | "
					f"Unique ROIs from guide freqs: {int(fit_result.get('n_unique_rois_requested', 0))} | "
					f"ROIs fitted: {int(fit_result.get('n_rois_fitted', 0))} | "
					f"Sampling: {fit_result.get('candidate_mode', 'ordered_grid')} | "
					f"Weighting: {fit_result.get('global_weight_mode', 'uniform')}"
				)
				if str(fit_result.get("local_optimizer_method", "none")).lower() != "none":
					st.caption(
						f"Local optimizer status: {fit_result.get('local_optimizer_status', 'unknown')} | "
						f"used result: {bool(fit_result.get('local_optimizer_used_result', False))} | "
						f"max evals: {int(fit_result.get('local_optimizer_max_nfev', 0))}"
					)
				if "obs_shift_kms_used" in fit_result:
					st.caption(
						f"Observational shift used for best fit: {float(fit_result.get('obs_shift_kms_used', obs_shift_kms)):+.4f} km/s"
					)
					if bool(fit_result.get("obs_shift_scan_enabled", False)):
						st.caption(
							f"Shift scan points: {len(list(fit_result.get('obs_shift_scan_values_kms', [])))} | successful fits: {int(fit_result.get('obs_shift_scan_n_ok', 0))}"
						)
				if bool(fit_result.get("refinement_applied", False)):
					if bool(fit_result.get("refinement_used_result", False)):
						st.info(
							f"Two-stage fitting applied: refined around initial best with ±{100.0 * float(fit_result.get('refinement_span_fraction', np.nan)):.1f}% local window | "
							f"candidates initial/refined = {int(fit_result.get('n_candidates_initial', 0))}/{int(fit_result.get('refinement_n_candidates', 0))}"
						)
					else:
						st.caption("Two-stage fitting attempted, but initial global solution was retained.")
				n_zero_skip = int(fit_result.get("n_rois_skipped_zero_observed", 0))
				if n_zero_skip > 0:
					st.info(f"ROIs excluded from fitting because observed segment is all zeros: {n_zero_skip}")

				global_overlay = fit_result.get("global_overlay", [])
				if isinstance(global_overlay, list) and global_overlay:
					segments = []
					for gg in global_overlay:
						fg = np.asarray(gg.get("freq", []), dtype=np.float64)
						yg_obs = np.asarray(gg.get("obs_interp", []), dtype=np.float64)
						yg_syn = np.asarray(gg.get("best_global_synthetic", []), dtype=np.float64)
						yg_noise = gg.get("best_global_noise", None)
						yg_pred = np.asarray(gg.get("best_global_pred", []), dtype=np.float64)
						if fg.size == 0 or yg_pred.size != fg.size:
							continue
						if yg_syn.size != fg.size:
							yg_syn = np.full_like(fg, np.nan, dtype=np.float64)
						if yg_noise is None:
							yg_noise_arr = np.full_like(fg, np.nan, dtype=np.float64)
						else:
							yg_noise_arr = np.asarray(yg_noise, dtype=np.float64)
							if yg_noise_arr.size != fg.size:
								yg_noise_arr = np.full_like(fg, np.nan, dtype=np.float64)
						segments.append((float(np.nanmin(fg)), fg, yg_obs, yg_syn, yg_noise_arr, yg_pred, gg))

					if segments:
						segments = sorted(segments, key=lambda t: t[0])
						f_cat = []
						s_cat = []
						n_cat = []
						p_cat = []
						roi_shapes = []
						for i_s, (_, fg, og, sg, ng, pg, _) in enumerate(segments):
							roi_shapes.append({
								"type": "rect",
								"xref": "x",
								"yref": "paper",
								"x0": float(np.nanmin(fg)),
								"x1": float(np.nanmax(fg)),
								"y0": 0.0,
								"y1": 1.0,
								"fillcolor": "rgba(140,140,140,0.18)",
								"line": {"width": 0},
								"layer": "below",
							})
							if i_s > 0:
								f_cat.append(np.array([np.nan], dtype=np.float64))
								s_cat.append(np.array([np.nan], dtype=np.float64))
								n_cat.append(np.array([np.nan], dtype=np.float64))
								p_cat.append(np.array([np.nan], dtype=np.float64))
							f_cat.append(fg)
							s_cat.append(sg)
							n_cat.append(ng)
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
							y=np.concatenate(s_cat),
							mode="lines",
							name="Best synthetic (global)",
							line=dict(dash="dash", width=1.6, color="#1f77b4"),
						))
						fig_global.add_trace(go.Scatter(
							x=np.concatenate(f_cat),
							y=np.concatenate(n_cat),
							mode="lines",
							name="Best noise (global)",
							line=dict(dash="dot", width=1.4, color="#9467bd"),
						))
						fig_global.add_trace(go.Scatter(
							x=np.concatenate(f_cat),
							y=np.concatenate(p_cat),
							mode="lines",
							name="Best fit (global)",
							line=dict(width=2.2, color="red"),
						))
						fig_global.update_layout(
							title="Observed spectrum vs best global fit",
							xaxis_title="Frequency (GHz)",
							yaxis_title="Intensity",
							shapes=roi_shapes,
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

	with tab_inv_params:
		st.subheader("Inverse Parameter Prediction")
		st.caption("Upload a spectrum and estimate LogN/Tex/FWHM/Velocity using inverse ROI models generated by script 13.")

		inv_models_dir = st.text_input(
			"Inverse ROI models directory",
			value=str(DEFAULT_LOCAL_INVERSE_PARAM_MODELS_DIR),
			key="p6_inv_models_dir",
		)

		inv_models, inv_warnings = _load_inverse_param_models_cached(str(inv_models_dir))
		if inv_warnings:
			with st.expander("Inverse model loading warnings", expanded=False):
				st.text("\n".join([str(w) for w in inv_warnings]))
		if inv_models:
			fmins = [float(m.get("f_min_ghz", np.nan)) for m in inv_models]
			fmaxs = [float(m.get("f_max_ghz", np.nan)) for m in inv_models]
			st.caption(
				f"Loaded inverse models: {len(inv_models)} | "
				f"coverage: {float(np.nanmin(fmins)):.6f} - {float(np.nanmax(fmaxs)):.6f} GHz"
			)
		else:
			st.info("No inverse models loaded. Check the directory path.")
			if inv_warnings:
				st.warning(str(inv_warnings[0]))

		up_obs_inv = st.file_uploader(
			"Upload spectrum for inverse prediction (.txt/.dat/.csv)",
			type=None,
			key="p6_inv_upload_obs",
		)
		obs_freq_inv, obs_vals_inv, obs_err_inv = _read_uploaded_spectrum_any(up_obs_inv) if up_obs_inv is not None else (None, None, None)
		if up_obs_inv is not None and obs_err_inv is not None:
			st.error(f"Could not parse uploaded spectrum: {obs_err_inv}")

		inv_shift_enabled = st.checkbox("Apply uploaded spectrum frequency shift", value=True, key="p6_inv_shift_enabled")
		inv_shift_mode = st.selectbox(
			"Shift mode",
			options=["per_frequency", "spw_center"],
			index=0,
			key="p6_inv_shift_mode",
		)
		inv_shift_kms = st.number_input(
			"Uploaded spectrum shift (km/s)",
			value=-98.0,
			step=0.1,
			format="%.4f",
			key="p6_inv_shift_kms",
		)
		obs_freq_inv_used = None if obs_freq_inv is None else np.asarray(obs_freq_inv, dtype=np.float64).copy()
		if (obs_freq_inv_used is not None) and bool(inv_shift_enabled):
			if str(inv_shift_mode).strip().lower() == "spw_center":
				obs_freq_inv_used = _apply_velocity_shift_by_spw_center(obs_freq_inv_used, float(inv_shift_kms))
			else:
				obs_freq_inv_used = _apply_velocity_shift_to_frequency(obs_freq_inv_used, float(inv_shift_kms))
			st.caption(f"Uploaded spectrum shifted by {float(inv_shift_kms):+.4f} km/s using mode: {str(inv_shift_mode)}")

		run_inv = st.button("Run inverse prediction", type="primary", key="p6_inv_run_btn")
		if run_inv:
			if up_obs_inv is None or obs_freq_inv_used is None or obs_vals_inv is None:
				st.error("Upload a valid spectrum first.")
			elif not inv_models:
				st.error("No inverse models available to run prediction.")
			else:
				rows_inv, summary_inv = _predict_inverse_params_from_models(
					freq_ghz=np.asarray(obs_freq_inv_used, dtype=np.float64),
					intensity=np.asarray(obs_vals_inv, dtype=np.float64),
					inverse_models=inv_models,
					min_overlap_channels=8,
				)
				st.session_state.p6_inv_last_result = {
					"rows": rows_inv,
					"summary": summary_inv,
					"models_dir": str(inv_models_dir),
					"obs_freq_used": np.asarray(obs_freq_inv_used, dtype=np.float64),
					"obs_vals_used": np.asarray(obs_vals_inv, dtype=np.float64),
				}

		inv_res = st.session_state.get("p6_inv_last_result", None)
		if isinstance(inv_res, dict):
			rows_inv = inv_res.get("rows", []) if isinstance(inv_res.get("rows", []), list) else []
			summary_inv = inv_res.get("summary", None)

			if (summary_inv is None) or (not rows_inv):
				st.warning("No valid ROI predictions were produced for this spectrum.")
			else:
				cmi1, cmi2, cmi3, cmi4 = st.columns(4)
				cmi1.metric("Weighted LogN", f"{float(summary_inv.get('weighted_logn', np.nan)):.5g}")
				cmi2.metric("Weighted Tex", f"{float(summary_inv.get('weighted_tex', np.nan)):.5g}")
				cmi3.metric("Weighted FWHM", f"{float(summary_inv.get('weighted_fwhm', np.nan)):.5g}")
				cmi4.metric("Weighted Velocity", f"{float(summary_inv.get('weighted_velo', np.nan)):.5g}")

				st.caption(
					f"ROIs used: {int(summary_inv.get('n_rois_used', 0))} | "
					f"outliers removed: {int(summary_inv.get('n_outlier_rows_removed', 0))} | "
					f"rows clipped: {int(summary_inv.get('n_clipped_rows', 0))} | "
					f"Best ROI by weight: {str(summary_inv.get('best_roi_name', ''))} | "
					f"Median estimate = "
					f"[{float(summary_inv.get('median_logn', np.nan)):.5g}, "
					f"{float(summary_inv.get('median_tex', np.nan)):.5g}, "
					f"{float(summary_inv.get('median_fwhm', np.nan)):.5g}, "
					f"{float(summary_inv.get('median_velo', np.nan)):.5g}]"
				)

				rows_inv_sorted = sorted(rows_inv, key=lambda r: float(r.get("weight", 0.0)), reverse=True)
				st.markdown("**Per-ROI inverse predictions**")
				st.dataframe(rows_inv_sorted, use_container_width=True)

				obs_freq_plot = inv_res.get("obs_freq_used", None)
				obs_vals_plot = inv_res.get("obs_vals_used", None)
				if (obs_freq_plot is not None) and (obs_vals_plot is not None):
					fig_inv = go.Figure()
					fig_inv.add_trace(
						go.Scatter(
							x=np.asarray(obs_freq_plot, dtype=np.float64),
							y=np.asarray(obs_vals_plot, dtype=np.float64),
							mode="lines",
							name="Uploaded spectrum",
							line=dict(color="green", width=1.5),
						)
					)
					for r in rows_inv_sorted:
						fig_inv.add_vrect(
							x0=float(r.get("roi_f_min_ghz", np.nan)),
							x1=float(r.get("roi_f_max_ghz", np.nan)),
							fillcolor="rgba(120,120,220,0.10)",
							line_width=0,
							layer="below",
						)
					fig_inv.update_layout(
						title="Uploaded spectrum with ROIs used by inverse predictors",
						xaxis_title="Frequency (GHz)",
						yaxis_title="Intensity",
						template="plotly_white",
						height=360,
						margin=dict(l=40, r=20, t=45, b=40),
					)
					st.plotly_chart(fig_inv, width="stretch", key="p6_inv_overlay_plot")

	with tab_pred_from_cube:
		st.subheader("Inverse Cube Prediction")
		st.caption("Equivalent to script 4.PREDICT_ParamMaps_FromCube... inside Streamlit (no external script dependency).")

		models_root_predcube = st.text_input(
			"ROI models directory",
			value=str(DEFAULT_LOCAL_INVERSE_CUBE_MODELS_DIR),
			key="p6_predcube_models_root",
		)
		model_selection_mode = st.selectbox(
			"Model selection",
			options=["all_roi_models", "guide_frequencies"],
			index=0,
			key="p6_predcube_model_selection_mode",
		)
		guide_freqs_predcube: List[float] = []
		if str(model_selection_mode) == "guide_frequencies":
			if not str(st.session_state.get("p6_predcube_guide_freqs_input", "")).strip():
				st.session_state.p6_predcube_guide_freqs_input = _freqs_to_text([float(v) for v in DEFAULT_CUBEFIT_GUIDE_FREQS])
			guide_freqs_text = st.text_input(
				"Guide frequencies (GHz)",
				key="p6_predcube_guide_freqs_input",
			)
			guide_freqs_predcube = _normalize_target_freqs_for_run(parse_freq_list(str(guide_freqs_text)))
			if guide_freqs_predcube:
				st.caption("Guide frequencies: " + _freqs_to_text(guide_freqs_predcube))

		up_obs_cube_predcube = st.file_uploader(
			"Upload observational cube (.fits)",
			type=["fits"],
			key="p6_predcube_upload_cube",
		)
		obs_cube_predcube_path_manual = st.text_input(
			"Or set observational cube path (.fits)",
			value=str(st.session_state.get("p6_predcube_obs_cube_path", "")),
			key="p6_predcube_obs_cube_path_input",
		)
		obs_cube_predcube_path_upload = _save_uploaded_file_to_temp(up_obs_cube_predcube, "predcube_obs_cube") if up_obs_cube_predcube is not None else None
		if (up_obs_cube_predcube is not None) and (not obs_cube_predcube_path_upload):
			st.warning("The uploaded file could not be cached locally. For large cubes, use the manual path field below.")
		if obs_cube_predcube_path_upload and os.path.isfile(str(obs_cube_predcube_path_upload)):
			st.session_state.p6_predcube_obs_cube_path = str(obs_cube_predcube_path_upload)
		obs_cube_predcube_path_self = str(st.session_state.get("p6_predcube_obs_cube_path", "")).strip()
		obs_cube_predcube_path_manual = str(obs_cube_predcube_path_manual or "").strip().strip('"').strip("'")
		if obs_cube_predcube_path_manual:
			st.caption(f"Manual path exists: {'yes' if os.path.isfile(obs_cube_predcube_path_manual) else 'no'}")
		obs_cube_predcube_path = ""
		obs_cube_predcube_shape = None
		for _cand in [obs_cube_predcube_path_upload, obs_cube_predcube_path_manual, obs_cube_predcube_path_self]:
			_c = str(_cand or "").strip()
			if (not _c) or (not os.path.isfile(_c)):
				continue
			_sh = _get_cube_ny_nx(_c)
			if (_sh is not None) and (int(_sh[0]) > 0) and (int(_sh[1]) > 0):
				obs_cube_predcube_path = str(_c)
				obs_cube_predcube_shape = (int(_sh[0]), int(_sh[1]))
				break
		if (obs_cube_predcube_shape is not None) and (int(obs_cube_predcube_shape[0]) > 0) and (int(obs_cube_predcube_shape[1]) > 0):
			st.session_state.p6_predcube_obs_cube_path = str(obs_cube_predcube_path)
			st.caption(f"Cube shape: ny={int(obs_cube_predcube_shape[0])}, nx={int(obs_cube_predcube_shape[1])}")
		elif obs_cube_predcube_path_manual:
			st.caption("Manual cube path is set, but it could not be validated as a readable FITS cube.")
		if obs_cube_predcube_path_upload and (obs_cube_predcube_shape is None):
			st.warning("The uploaded cube could not be read as a valid FITS cube.")

		predcube_shift_enabled = st.checkbox("Apply frequency shift", value=True, key="p6_predcube_shift_enabled")
		predcube_shift_mode = st.selectbox("Shift mode", options=["per_frequency", "spw_center"], index=0, key="p6_predcube_shift_mode")
		predcube_shift_kms = st.number_input("Shift (km/s)", value=-55.0, step=0.1, format="%.4f", key="p6_predcube_shift_kms")

		predcube_out_dir = st.text_input("Output directory", value=str(DEFAULT_INVERSE_CUBEPRED_OUTDIR), key="p6_predcube_out_dir")
		predcube_out_prefix = st.text_input("Output prefix", value="PRED_FROMCUBE", key="p6_predcube_out_prefix")
		predcube_progress_every = st.number_input("Progress every N pixels", min_value=1, value=3000, step=1, key="p6_predcube_progress_every")
		predcube_spatial_stride = st.number_input("Spatial stride", min_value=1, value=1, step=1, key="p6_predcube_spatial_stride")
		predcube_min_overlap = st.number_input("Minimum overlap channels per ROI", min_value=1, value=2, step=1, key="p6_predcube_min_overlap")

		pc1, pc2 = st.columns(2)
		with pc1:
			run_predcube = st.button("Run map prediction", type="primary", key="p6_run_predcube_btn", disabled=_is_invcubepred_running())
		with pc2:
			stop_predcube = st.button("Stop map prediction", key="p6_stop_predcube_btn", disabled=not _is_invcubepred_running())

		if run_predcube:
			if (not obs_cube_predcube_path) or (not os.path.isfile(str(obs_cube_predcube_path))) or (obs_cube_predcube_shape is None):
				st.error("Upload a valid observational cube first.")
			elif (str(model_selection_mode) == "guide_frequencies") and (not guide_freqs_predcube):
				st.error("Guide frequencies is empty. Add at least one frequency.")
			elif (not models_root_predcube) or (not os.path.isdir(str(models_root_predcube))):
				st.error("ROI models directory is invalid.")
			else:
				try:
					os.makedirs(predcube_out_dir, exist_ok=True)
					_cleanup_invcubepred_outputs_for_dir(str(predcube_out_dir))
					cfg_icp = {
						"out_dir": str(predcube_out_dir),
						"obs_cube_path": str(obs_cube_predcube_path),
						"inverse_models_root": str(models_root_predcube),
						"use_all_models": bool(str(model_selection_mode) == "all_roi_models"),
						"target_freqs": [float(v) for v in guide_freqs_predcube],
						"allow_nearest": True,
						"progress_every": int(predcube_progress_every),
						"spatial_stride": int(predcube_spatial_stride),
						"obs_shift_enabled": bool(predcube_shift_enabled),
						"obs_shift_mode": str(predcube_shift_mode),
						"obs_shift_kms": float(predcube_shift_kms),
						"resume_enabled": False,
						"min_overlap_channels": int(predcube_min_overlap),
						"region_mode": "full",
						"out_prefix": str(predcube_out_prefix).strip() or "PRED_FROMCUBE",
					}
					fd_icp, cfg_icp_path = tempfile.mkstemp(prefix="predobs6_predcube_cfg_", suffix=".json", dir=tempfile.gettempdir())
					os.close(fd_icp)
					with open(cfg_icp_path, "w", encoding="utf-8") as f:
						json.dump(cfg_icp, f, ensure_ascii=False, indent=2)
					log_icp_path = os.path.join(predcube_out_dir, f"predcube_run_{time.strftime('%Y%m%d_%H%M%S')}.log")
					log_icp_fh = open(log_icp_path, "a", encoding="utf-8", buffering=1)
					proc_icp = subprocess.Popen(
						[sys.executable, str(Path(__file__).resolve()), "--inverse-cube-worker", cfg_icp_path],
						cwd=str(_project_dir()),
						stdout=log_icp_fh,
						stderr=subprocess.STDOUT,
						text=True,
					)
					st.session_state.invcubepred_proc = proc_icp
					st.session_state.invcubepred_log_path = log_icp_path
					st.session_state.invcubepred_cfg_path = cfg_icp_path
					st.session_state.invcubepred_log_handle = log_icp_fh
					st.session_state.invcubepred_start_ts = float(time.time())
					st.session_state.p6_predcube_last_out_dir = str(predcube_out_dir)
					st.session_state.p6_predcube_last_out_prefix = str(cfg_icp.get("out_prefix", "PRED_FROMCUBE"))
					st.success("Map prediction started.")
				except Exception as e:
					st.error(f"Could not start map prediction: {e}")

		if stop_predcube:
			_stop_invcubepred_process()
			st.warning("Map prediction stopped by user.")

		if _is_invcubepred_running():
			start_ts_icp = st.session_state.get("invcubepred_start_ts", None)
			if start_ts_icp is not None:
				elapsed_icp = _format_elapsed_hms(float(time.time()) - float(start_ts_icp))
				st.info(f"Map prediction status: running | elapsed: {elapsed_icp}")
			else:
				st.info("Map prediction status: running")
		else:
			proc_icp = st.session_state.get("invcubepred_proc", None)
			if proc_icp is not None:
				code_icp = proc_icp.poll()
				if code_icp == 0:
					st.success("Map prediction status: finished successfully")
				elif code_icp is not None:
					st.error(f"Map prediction status: finished with code {code_icp}")
					log_tail_icp = _read_log_tail(str(st.session_state.get("invcubepred_log_path", "")), n_lines=120)
					if log_tail_icp:
						with st.expander("Show last map-prediction log lines"):
							st.text(log_tail_icp)
				_stop_invcubepred_process()
			else:
				st.caption("Map prediction status: idle")

		out_dir_show = str(st.session_state.get("p6_predcube_last_out_dir", predcube_out_dir))
		out_prefix_show = str(st.session_state.get("p6_predcube_last_out_prefix", predcube_out_prefix)).strip() or "PRED_FROMCUBE"
		progress_png_icp = _find_latest_progress_png(str(out_dir_show))
		if progress_png_icp:
			with st.expander("Map prediction progress", expanded=False):
				progress_info_icp = _read_progress_info(progress_png_icp)
				if isinstance(progress_info_icp, dict):
					done_steps = int(progress_info_icp.get("done_steps", 0))
					total_steps = int(max(1, progress_info_icp.get("total_steps", 1)))
					pct = 100.0 * float(done_steps) / float(total_steps)
					st.success(f"**Pixels processed:** {done_steps}/{total_steps} ({pct:.1f}%)")
				img_bytes_icp = _read_progress_png_stable_bytes(progress_png_icp)
				if img_bytes_icp is not None:
					st.image(img_bytes_icp, caption=os.path.basename(progress_png_icp))

		map_files_icp = {
			"logN": os.path.join(str(out_dir_show), f"{out_prefix_show}_LOGN.fits"),
			"Tex": os.path.join(str(out_dir_show), f"{out_prefix_show}_TEX.fits"),
			"Velocity": os.path.join(str(out_dir_show), f"{out_prefix_show}_VELOCITY.fits"),
			"FWHM": os.path.join(str(out_dir_show), f"{out_prefix_show}_FWHM.fits"),
		}
		if not _is_invcubepred_running():
			available_maps_icp = {k: v for k, v in map_files_icp.items() if os.path.isfile(v)}
			if available_maps_icp:
				st.markdown("**Predicted parameter maps (final, RA/Dec)**")
				cmap_by_param_icp = {
					"logN": "viridis",
					"Tex": "magma",
					"Velocity": "coolwarm",
					"FWHM": "plasma",
				}
				imc1, imc2 = st.columns(2)
				cols_map_icp = [imc1, imc2]
				for i_m, (mk, mp) in enumerate(available_maps_icp.items()):
					with cols_map_icp[i_m % 2]:
						try:
							arr_m = np.asarray(fits.getdata(mp), dtype=np.float32)
							hdr_m = fits.getheader(mp)
							if arr_m.ndim == 3:
								arr_m = arr_m[0]
							_show_fits_preview(mk, arr_m, cmap=str(cmap_by_param_icp.get(str(mk), "viridis")), ref_hdr=hdr_m)
						except Exception:
							st.caption(f"Could not render preview for {mk}")
						try:
							with open(mp, "rb") as f_mp:
								st.download_button(
									f"Download {mk} map (.fits)",
									data=f_mp.read(),
									file_name=os.path.basename(mp),
									mime="application/fits",
									key=f"p6_predcube_download_{mk}",
								)
						except Exception:
							pass

		log_txt_icp = os.path.join(str(out_dir_show), "Log.txt")
		if os.path.isfile(log_txt_icp):
			with st.expander("Map prediction Log.txt", expanded=False):
				st.text(_read_log_tail(log_txt_icp, n_lines=300))

		if _is_invcubepred_running():
			st.caption("Auto-updating every 5 seconds...")
			time.sleep(5)
			st.rerun()

	with tab_cube_fit:
		st.subheader("Cube Fitting")
		st.caption("Same fitting parameterization as 'Fitting', but applied pixel-by-pixel to an uploaded observational cube to produce LogN/Tex/Velocity/FWHM maps.")

		if not str(st.session_state.get("p6_cubefit_obs_cube_paths_text", "")).strip():
			st.session_state.p6_cubefit_obs_cube_paths_text = str(DEFAULT_OBS_CUBE_PATH)

		if bool(st.session_state.get("p6_guide_cfit_refresh", False)):
			st.session_state.p6_guide_freqs_cfit_input = str(st.session_state.get("p6_guide_freqs_cfit_pending", "")).strip()
			st.session_state.p6_guide_cfit_refresh = False
			st.session_state.p6_guide_freqs_cfit_pending = ""
		if not str(st.session_state.get("p6_guide_freqs_cfit_input", "")).strip():
			st.session_state.p6_guide_freqs_cfit_input = _freqs_to_text([float(v) for v in DEFAULT_CUBEFIT_GUIDE_FREQS])

		guide_freqs_cfit_text = st.text_input(
			"Guide frequencies (GHz; defines ROIs to fit in every pixel)",
			key="p6_guide_freqs_cfit_input",
		)
		guide_freqs_cfit = _normalize_target_freqs_for_run(parse_freq_list(str(guide_freqs_cfit_text)))
		if guide_freqs_cfit:
			st.caption("Target frequencies used for cube fitting: " + _freqs_to_text(guide_freqs_cfit))

		up_obs_cube_fit = st.file_uploader(
			"Upload one or more observational cubes (.fits)",
			type=["fits"],
			accept_multiple_files=True,
			key="p6_cubefit_upload_cube",
		)
		obs_cube_fit_paths_manual_text = st.text_area(
			"Or set observational cube path(s) (.fits) | one per line",
			value=str(st.session_state.get("p6_cubefit_obs_cube_paths_text", str(DEFAULT_OBS_CUBE_PATH))),
			key="p6_cubefit_obs_cube_paths_input",
			height=90,
		)

		upload_paths: List[str] = []
		for i_up, up_item in enumerate((up_obs_cube_fit or []), start=1):
			p_up = _save_uploaded_file_to_temp(up_item, f"cubefit_obs_cube_{i_up}")
			if p_up and os.path.isfile(str(p_up)):
				upload_paths.append(str(p_up))
		if (up_obs_cube_fit is not None) and (len(up_obs_cube_fit) > 0) and (len(upload_paths) <= 0):
			st.warning("Uploaded files could not be cached locally. For large cubes, use the manual paths field below.")

		manual_paths: List[str] = []
		for _ln in str(obs_cube_fit_paths_manual_text or "").replace(";", "\n").splitlines():
			_p = str(_ln or "").strip().strip('"').strip("'")
			if _p:
				manual_paths.append(_p)

		if manual_paths:
			n_exist = int(sum(1 for p in manual_paths if os.path.isfile(p)))
			st.caption(f"Manual paths existing: {n_exist}/{len(manual_paths)}")

		obs_cube_fit_paths: List[str] = []
		obs_cube_shape = None
		shape_ref = None
		if len(upload_paths) > 0:
			candidate_paths = list(upload_paths)
			st.caption("Using uploaded cube(s). Manual path(s) are ignored for this run.")
		else:
			candidate_paths = list(manual_paths)

		for _cand in candidate_paths:
			_c = str(_cand or "").strip()
			if (not _c) or (not os.path.isfile(_c)):
				continue
			_sh = _get_cube_ny_nx(_c)
			if (_sh is None) or (int(_sh[0]) <= 0) or (int(_sh[1]) <= 0):
				continue
			if shape_ref is None:
				shape_ref = (int(_sh[0]), int(_sh[1]))
			elif (int(_sh[0]), int(_sh[1])) != shape_ref:
				st.warning(f"Cube skipped due to shape mismatch: {_c} | shape=({_sh[0]},{_sh[1]}) | expected={shape_ref}")
				continue
			if _c not in obs_cube_fit_paths:
				obs_cube_fit_paths.append(_c)

		if len(obs_cube_fit_paths) > 0:
			obs_cube_shape = shape_ref
			st.session_state.p6_last_cubefit_obs_cube_paths = [str(p) for p in obs_cube_fit_paths]
			st.session_state.p6_cubefit_obs_cube_paths_text = "\n".join([str(p) for p in obs_cube_fit_paths])
			st.caption(f"Valid cubes for fitting: {len(obs_cube_fit_paths)}")
			st.caption(f"Reference spatial shape: ny={int(obs_cube_shape[0])}, nx={int(obs_cube_shape[1])}")
		elif manual_paths:
			st.caption("Manual cube paths are set, but no valid readable FITS cubes were found.")

		cubefit_out_dir = st.text_input("Output directory", value=str(DEFAULT_CUBEFIT_OUTDIR), key="p6_cubefit_out_dir")
		cubefit_progress_every = st.number_input("Progress every N pixels", min_value=1, value=40, step=1, key="p6_cubefit_progress_every")
		cubefit_spatial_stride = st.number_input("Spatial stride (1=all pixels, 2=every 2 pixels)", min_value=1, value=1, step=1, key="p6_cubefit_spatial_stride")
		cubefit_use_region = st.checkbox(
			"Fit only a selected pixel region (bounding box)",
			value=True,
			key="p6_cubefit_use_region",
		)
		if obs_cube_shape is not None:
			ny_cf, nx_cf = int(obs_cube_shape[0]), int(obs_cube_shape[1])
		else:
			ny_cf, nx_cf = 1, 1
		def_y_min = int(max(0, min(ny_cf - 1, 139)))
		def_y_max = int(max(0, min(ny_cf - 1, 201)))
		def_x_min = int(max(0, min(nx_cf - 1, 161)))
		def_x_max = int(max(0, min(nx_cf - 1, 200)))
		region_sig = f"{int(ny_cf)}x{int(nx_cf)}|n={len(obs_cube_fit_paths)}"
		if str(st.session_state.get("p6_cubefit_region_shape_sig", "")) != str(region_sig):
			st.session_state.p6_cubefit_region_y_min = int(def_y_min)
			st.session_state.p6_cubefit_region_y_max = int(def_y_max)
			st.session_state.p6_cubefit_region_x_min = int(def_x_min)
			st.session_state.p6_cubefit_region_x_max = int(def_x_max)
			st.session_state.p6_cubefit_region_shape_sig = str(region_sig)
		rg_c1, rg_c2 = st.columns(2)
		with rg_c1:
			region_x_min = st.number_input("Region x_min", min_value=0, max_value=max(0, nx_cf - 1), value=int(def_x_min), step=1, key="p6_cubefit_region_x_min", disabled=not bool(cubefit_use_region))
			region_y_min = st.number_input("Region y_min", min_value=0, max_value=max(0, ny_cf - 1), value=int(def_y_min), step=1, key="p6_cubefit_region_y_min", disabled=not bool(cubefit_use_region))
		with rg_c2:
			region_x_max = st.number_input("Region x_max", min_value=0, max_value=max(0, nx_cf - 1), value=int(def_x_max), step=1, key="p6_cubefit_region_x_max", disabled=not bool(cubefit_use_region))
			region_y_max = st.number_input("Region y_max", min_value=0, max_value=max(0, ny_cf - 1), value=int(def_y_max), step=1, key="p6_cubefit_region_y_max", disabled=not bool(cubefit_use_region))
		if bool(cubefit_use_region):
			st.caption(f"Selected region (pixels): x=[{int(min(region_x_min, region_x_max))},{int(max(region_x_min, region_x_max))}], y=[{int(min(region_y_min, region_y_max))},{int(max(region_y_min, region_y_max))}]")
		cubefit_resume_enabled = st.checkbox(
			"Resume from previous in-progress checkpoint (if available)",
			value=True,
			key="p6_cubefit_resume_enabled",
		)

		cubefit_case = st.radio(
			"Fitting mode",
			options=["Case 1: Synthetic only", "Case 2: Synthetic + noise"],
			index=1,
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
			value=-55.0,
			step=0.1,
			format="%.4f",
			key="p6_cubefit_shift_kms",
		)

		with st.expander("Preview pixel spectrum and ROI windows", expanded=False):
			if (obs_cube_shape is None) or (len(obs_cube_fit_paths) <= 0):
				st.caption("Load at least one valid cube to enable the preview.")
			else:
				ny_prev, nx_prev = int(obs_cube_shape[0]), int(obs_cube_shape[1])
				px_default_x = int(max(0, min(nx_prev - 1, nx_prev // 2)))
				px_default_y = int(max(0, min(ny_prev - 1, ny_prev // 2)))
				pcx1, pcx2 = st.columns(2)
				with pcx1:
					preview_x = st.number_input(
						"Preview pixel x",
						min_value=0,
						max_value=max(0, nx_prev - 1),
						value=int(px_default_x),
						step=1,
						key="p6_cubefit_preview_x",
					)
				with pcx2:
					preview_y = st.number_input(
						"Preview pixel y",
						min_value=0,
						max_value=max(0, ny_prev - 1),
						value=int(px_default_y),
						step=1,
						key="p6_cubefit_preview_y",
					)

				freq_parts = []
				val_parts = []
				integ_map_accum = None
				for i_cp, cp in enumerate(obs_cube_fit_paths, start=1):
					try:
						with fits.open(str(cp), memmap=True) as hdul_cp:
							arr_cp = np.asarray(hdul_cp[0].data, dtype=np.float32)
							hdr_cp = hdul_cp[0].header.copy()
						if arr_cp.ndim == 4:
							arr_cp = arr_cp[0]
						if arr_cp.ndim != 3:
							continue
						nchan_cp = int(arr_cp.shape[0])
						f_cp = _build_freq_axis_from_header(hdr_cp, nchan_cp)
						if bool(cubefit_shift_enabled):
							if str(cubefit_shift_mode).strip().lower() == "spw_center":
								f_cp = _apply_velocity_shift_by_spw_center(f_cp, float(cubefit_shift_kms))
							else:
								f_cp = _apply_velocity_shift_to_frequency(f_cp, float(cubefit_shift_kms))

						map_cp = np.asarray(np.nansum(np.where(np.isfinite(arr_cp), arr_cp, 0.0), axis=0), dtype=np.float64)
						if integ_map_accum is None:
							integ_map_accum = np.asarray(map_cp, dtype=np.float64)
						else:
							if integ_map_accum.shape == map_cp.shape:
								integ_map_accum = np.asarray(integ_map_accum + map_cp, dtype=np.float64)

						y_cp = np.asarray(arr_cp[:, int(preview_y), int(preview_x)], dtype=np.float64)
						freq_parts.append(np.asarray(f_cp, dtype=np.float64))
						val_parts.append(np.asarray(y_cp, dtype=np.float64))
						st.caption(f"Cube {i_cp}: {os.path.basename(str(cp))} | nchan={int(nchan_cp)}")
					except Exception as e:
						st.warning(f"Could not read preview from cube: {cp} | {e}")

				# Keep spectrum detail similar to the previous full-width view,
				# while placing the integrated map at the left.
				col_map, col_spec = st.columns([0.85, 2.15], vertical_alignment="top")

				with col_map:
					if integ_map_accum is not None:
						v_map = np.asarray(integ_map_accum, dtype=np.float64)
						m_map = np.isfinite(v_map)
						if np.any(m_map):
							vmin_map = float(np.nanpercentile(v_map[m_map], 5.0))
							vmax_map = float(np.nanpercentile(v_map[m_map], 99.0))
							if (not np.isfinite(vmin_map)) or (not np.isfinite(vmax_map)) or (vmax_map <= vmin_map):
								vmin_map = float(np.nanmin(v_map[m_map]))
								vmax_map = float(np.nanmax(v_map[m_map]))
						else:
							vmin_map, vmax_map = 0.0, 1.0

						fig_cf_map = go.Figure()
						fig_cf_map.add_trace(
							go.Heatmap(
								z=v_map,
								colorscale="Viridis",
								zmin=vmin_map,
								zmax=vmax_map,
								colorbar=dict(title="Integrated intensity"),
								hovertemplate="x=%{x}<br>y=%{y}<br>Iint=%{z:.4g}<extra></extra>",
							)
						)
						fig_cf_map.add_trace(
							go.Scatter(
								x=[int(preview_x)],
								y=[int(preview_y)],
								mode="markers",
								name="Preview pixel",
								marker=dict(size=11, color="#ff3b30", symbol="x", line=dict(width=1, color="#ffffff")),
								hovertemplate="Preview pixel<br>x=%{x}<br>y=%{y}<extra></extra>",
							)
						)
						fig_cf_map.update_layout(
							title="Integrated intensity map (sum across channels and cubes)",
							xaxis_title="x pixel",
							yaxis_title="y pixel",
							template="plotly_white",
							height=360,
							margin=dict(l=40, r=20, t=45, b=40),
						)
						fig_cf_map.update_yaxes(scaleanchor="x", scaleratio=1)
						st.plotly_chart(fig_cf_map, width="stretch", key="p6_cubefit_preview_map")
					else:
						st.caption("Integrated map preview not available.")

				with col_spec:
					if (len(freq_parts) > 0) and (len(val_parts) > 0):
						f_all = np.concatenate([np.asarray(ff, dtype=np.float64) for ff in freq_parts], axis=0)
						y_all = np.concatenate([np.asarray(yy, dtype=np.float64) for yy in val_parts], axis=0)
						ord_idx = np.argsort(np.asarray(f_all, dtype=np.float64))
						f_all = np.asarray(f_all, dtype=np.float64)[ord_idx]
						y_all = np.asarray(y_all, dtype=np.float64)[ord_idx]

						fig_cf_prev = go.Figure()
						fig_cf_prev.add_trace(
							go.Scatter(
								x=f_all,
								y=y_all,
								mode="lines",
								name="Pixel spectrum (concatenated cubes)",
								line=dict(color="#1f77b4", width=1.3),
							)
						)

						if guide_freqs_cfit:
							try:
								groups_prev = _group_target_freqs_by_signal_roi(
									signal_models_source=str(signal_models_root),
									filter_file=str(filter_file),
									target_freqs=[float(v) for v in guide_freqs_cfit],
									allow_nearest=bool(allow_nearest),
								)
							except Exception:
								groups_prev = []

							for g in groups_prev:
								lo = g.get("roi_f_min_ghz", None)
								hi = g.get("roi_f_max_ghz", None)
								if (lo is None) or (hi is None):
									continue
								try:
									fig_cf_prev.add_vrect(
										x0=float(lo),
										x1=float(hi),
										fillcolor="rgba(120,120,220,0.12)",
										line_width=0,
										layer="below",
									)
								except Exception:
									pass

							for gf in [float(v) for v in guide_freqs_cfit]:
								fig_cf_prev.add_vline(x=float(gf), line=dict(color="#9467bd", dash="dash"))

						fig_cf_prev.update_layout(
							title=f"Cube-fitting preview | pixel (x={int(preview_x)}, y={int(preview_y)})",
							xaxis_title="Frequency (GHz)",
							yaxis_title="Intensity",
							template="plotly_white",
							height=360,
							margin=dict(l=40, r=20, t=45, b=40),
						)
						st.plotly_chart(fig_cf_prev, width="stretch", key="p6_cubefit_preview_plot")
					else:
						st.caption("No preview available from selected cubes.")

		with st.expander("Fitting search ranges and speed settings", expanded=False):
			cubefit_global_mode_ui = st.selectbox(
				"Global fit strategy",
				options=["Per-ROI aggregate", "Concatenated ROIs (single objective)"],
				index=1,
				key="p6_cubefit_global_mode",
			)
			cubefit_global_mode_map = {
				"Per-ROI aggregate": "per_roi",
				"Concatenated ROIs (single objective)": "concatenated",
			}
			cubefit_criterion_ui = st.selectbox(
				"Fitting criterion",
				options=["MAE", "RMSE", "CHI_like", "R2"],
				index=1,
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
				cubefit_n_candidates = st.number_input("Number of candidates", min_value=50, max_value=4000, value=300, step=50, key="p6_cubefit_n_candidates")
			with ccs2:
				cubefit_seed = st.number_input("Random seed", min_value=0, value=42, step=1, key="p6_cubefit_seed")
			cubefit_local_opt_method_ui = st.selectbox(
				"Local optimizer after candidate search",
				options=["None", "TRF (Trust Region Reflective)"],
				index=0,
				key="p6_cubefit_local_opt_method",
			)
			cubefit_local_opt_method_map = {
				"None": "none",
				"TRF (Trust Region Reflective)": "trf",
			}
			cubefit_local_opt_max_nfev = st.number_input(
				"Local optimizer max evaluations",
				min_value=8,
				max_value=120,
				value=16,
				step=4,
				key="p6_cubefit_local_opt_max_nfev",
				disabled=(str(cubefit_local_opt_method_ui) == "None"),
			)
			cubefit_independent_pixel_candidates = st.checkbox(
				"Independent candidates per pixel (not reused from previous pixel)",
				value=False,
				key="p6_cubefit_independent_pixel_candidates",
			)
			if bool(cubefit_independent_pixel_candidates):
				st.caption("Each pixel uses its own candidate set (deterministic by pixel coordinate). This is more independent but slower.")

		cbf1, cbf2 = st.columns(2)
		with cbf1:
			run_cubefit = st.button("Run cube fitting", type="primary", key="p6_run_cubefit_btn", disabled=_is_cubefit_running())
		with cbf2:
			stop_cubefit = st.button("Stop cube fitting", key="p6_stop_cubefit_btn", disabled=not _is_cubefit_running())

		if run_cubefit:
			if len(obs_cube_fit_paths) <= 0:
				st.error("Upload or set at least one valid observational cube first.")
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
					if not bool(cubefit_resume_enabled):
						_cleanup_cubefit_outputs_for_dir(str(cubefit_out_dir))
						st.caption("Starting fresh: previous cube-fitting checkpoints were cleared.")
					else:
						st.caption("Resume mode enabled: existing checkpoints (if any) will be reused.")
					elapsed_accum_seconds_cfg = 0.0
					if bool(cubefit_resume_enabled):
						state_prev_path = os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_STATE.json")
						if os.path.isfile(state_prev_path):
							try:
								with open(state_prev_path, "r", encoding="utf-8") as f_prev:
									state_prev_obj = json.load(f_prev)
								if isinstance(state_prev_obj, dict):
									elapsed_accum_seconds_cfg = float(state_prev_obj.get("elapsed_total_seconds", 0.0))
							except Exception:
								elapsed_accum_seconds_cfg = 0.0
					if (not np.isfinite(float(elapsed_accum_seconds_cfg))) or (float(elapsed_accum_seconds_cfg) < 0.0):
						elapsed_accum_seconds_cfg = 0.0
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
						"obs_cube_path": str(obs_cube_fit_paths[0]),
						"obs_cube_paths": [str(p) for p in obs_cube_fit_paths],
						"signal_models_source": str(signal_models_root),
						"noise_models_root": str(noise_models_root),
						"filter_file": str(filter_file),
						"target_freqs": [float(v) for v in guide_freqs_cfit],
						"case_mode": str(cubefit_case_mode),
						"fit_criterion": str(cubefit_criterion_ui).strip().lower(),
						"global_weight_mode": str(cubefit_weight_mode_map.get(str(cubefit_weight_mode_ui), "inverse_best_error")),
						"global_search_mode": str(cubefit_global_mode_map.get(str(cubefit_global_mode_ui), "concatenated")),
						"candidate_mode": str(cubefit_candidate_mode_map.get(str(cubefit_candidate_mode_ui), "random")),
						"n_candidates": int(cubefit_n_candidates),
						"local_optimizer_method": str(cubefit_local_opt_method_map.get(str(cubefit_local_opt_method_ui), "none")),
						"local_optimizer_max_nfev": int(cubefit_local_opt_max_nfev),
						"independent_pixel_candidates": bool(cubefit_independent_pixel_candidates),
						"ranges": ranges_cubefit,
						"noise_scale": float(noise_scale),
						"allow_nearest": bool(allow_nearest),
						"seed": int(cubefit_seed),
						"progress_every": int(cubefit_progress_every),
						"spatial_stride": int(cubefit_spatial_stride),
						"obs_shift_enabled": bool(cubefit_shift_enabled),
						"obs_shift_mode": str(cubefit_shift_mode),
						"obs_shift_kms": float(cubefit_shift_kms),
						"resume_enabled": bool(cubefit_resume_enabled),
						"elapsed_accum_seconds": float(elapsed_accum_seconds_cfg),
						"region_mode": ("bbox" if bool(cubefit_use_region) else "full"),
						"region_x_min": int(min(region_x_min, region_x_max)),
						"region_x_max": int(max(region_x_min, region_x_max)),
						"region_y_min": int(min(region_y_min, region_y_max)),
						"region_y_max": int(max(region_y_min, region_y_max)),
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
					st.session_state.cubefit_start_ts = float(time.time())
					st.success("Cube fitting started.")
				except Exception as e:
					st.error(f"Could not start cube fitting: {e}")

		if stop_cubefit:
			_stop_cubefit_process()
			st.warning("Cube fitting stopped by user.")

		if _is_cubefit_running():
			elapsed_cf = None
			state_cf_path = os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_STATE.json")
			if os.path.isfile(state_cf_path):
				try:
					with open(state_cf_path, "r", encoding="utf-8") as f_state_cf:
						state_cf_obj = json.load(f_state_cf)
					if isinstance(state_cf_obj, dict):
						et = float(state_cf_obj.get("elapsed_total_seconds", np.nan))
						if np.isfinite(et) and et >= 0.0:
							elapsed_cf = _format_elapsed_hms(float(et))
				except Exception:
					pass
			if elapsed_cf is None:
				start_ts_cf = st.session_state.get("cubefit_start_ts", None)
				if start_ts_cf is not None:
					elapsed_cf = _format_elapsed_hms(float(time.time()) - float(start_ts_cf))
			if elapsed_cf is not None:
				st.info(f"Cube fitting status: running | elapsed: {elapsed_cf}")
			else:
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
			with st.expander("Cube fitting progress", expanded=False):
				progress_info_cf = _read_progress_info(progress_png_cf)
				if isinstance(progress_info_cf, dict):
					done_steps = int(progress_info_cf.get("done_steps", 0))
					total_steps = int(max(1, progress_info_cf.get("total_steps", 1)))
					pct = 100.0 * float(done_steps) / float(total_steps)
					st.success(f"**Pixels processed:** {done_steps}/{total_steps} ({pct:.1f}%)")
				img_bytes_cf = _read_progress_png_stable_bytes(progress_png_cf)
				if img_bytes_cf is not None:
					st.image(img_bytes_cf, caption=os.path.basename(progress_png_cf))

		with st.expander("Live checkpoint spectra and pixel location", expanded=False):
			lastpixel_npz_cf = os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_LASTPIXEL_SPECTRA.npz")
			integ_map_cf = os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_INTEG_MAP.fits")
			if not os.path.isfile(lastpixel_npz_cf):
				st.caption("No checkpoint spectra available yet. They appear every 'Progress every N pixels'.")
			else:
				try:
					with np.load(lastpixel_npz_cf, allow_pickle=False) as dd:
						px = int(np.asarray(dd.get("x", [0])).reshape(-1)[0])
						py = int(np.asarray(dd.get("y", [0])).reshape(-1)[0])
						ds = int(np.asarray(dd.get("done_steps", [0])).reshape(-1)[0])
						ts = int(max(1, np.asarray(dd.get("total_steps", [1])).reshape(-1)[0]))
						fit_ok = bool(int(np.asarray(dd.get("fit_ok", [0])).reshape(-1)[0]) == 1)
						f_sp = np.asarray(dd.get("freq", np.asarray([], dtype=np.float32)), dtype=np.float64).reshape(-1)
						y_obs_sp = np.asarray(dd.get("obs", np.asarray([], dtype=np.float32)), dtype=np.float64).reshape(-1)
						y_syn_sp = np.asarray(dd.get("syn", np.asarray([], dtype=np.float32)), dtype=np.float64).reshape(-1)
						y_noise_sp = np.asarray(dd.get("noise", np.asarray([], dtype=np.float32)), dtype=np.float64).reshape(-1)
						y_pred_sp = np.asarray(dd.get("pred", np.asarray([], dtype=np.float32)), dtype=np.float64).reshape(-1)

					pct = 100.0 * float(ds) / float(ts)
					st.caption(f"Checkpoint pixel: (x={px}, y={py}) | processed: {ds}/{ts} ({pct:.1f}%)")

					col_live_map, col_live_spec = st.columns([0.85, 2.15], vertical_alignment="top")

					with col_live_map:
						if os.path.isfile(integ_map_cf):
							try:
								arr_map = np.asarray(fits.getdata(integ_map_cf), dtype=np.float32)
								if arr_map.ndim == 3:
									arr_map = arr_map[0]
								arr_map = np.asarray(arr_map, dtype=np.float64)
								m_live = np.isfinite(arr_map)
								if np.any(m_live):
									vmin_live = float(np.nanpercentile(arr_map[m_live], 5.0))
									vmax_live = float(np.nanpercentile(arr_map[m_live], 99.0))
									if (not np.isfinite(vmin_live)) or (not np.isfinite(vmax_live)) or (vmax_live <= vmin_live):
										vmin_live = float(np.nanmin(arr_map[m_live]))
										vmax_live = float(np.nanmax(arr_map[m_live]))
								else:
									vmin_live, vmax_live = 0.0, 1.0
								fig_live_map = go.Figure()
								fig_live_map.add_trace(
									go.Heatmap(
										z=arr_map,
										colorscale="Viridis",
										zmin=vmin_live,
										zmax=vmax_live,
										colorbar=dict(title="Integrated intensity"),
										hovertemplate="x=%{x}<br>y=%{y}<br>Iint=%{z:.4g}<extra></extra>",
									)
								)
								fig_live_map.add_trace(
									go.Scatter(
										x=[int(px)],
										y=[int(py)],
										mode="markers",
										name="Checkpoint pixel",
										marker=dict(size=11, color="#ff3b30", symbol="x", line=dict(width=1, color="#ffffff")),
									)
								)
								fig_live_map.update_layout(
									title="Checkpoint pixel location",
									xaxis_title="x pixel",
									yaxis_title="y pixel",
									template="plotly_white",
									height=340,
									margin=dict(l=40, r=20, t=45, b=40),
								)
								fig_live_map.update_yaxes(scaleanchor="x", scaleratio=1)
								st.plotly_chart(fig_live_map, width="stretch", key="p6_cubefit_live_map")
							except Exception:
								st.caption("Could not render checkpoint location map.")
						else:
							st.caption("Integrated map not available yet.")

					with col_live_spec:
						if f_sp.size >= 2 and y_obs_sp.size == f_sp.size:
							fig_live = go.Figure()
							fig_live.add_trace(go.Scatter(x=f_sp, y=y_obs_sp, mode="lines", name="Observed", line=dict(color="#2ca02c", width=1.4)))
							if y_syn_sp.size == f_sp.size:
								fig_live.add_trace(go.Scatter(x=f_sp, y=y_syn_sp, mode="lines", name="Synthetic", line=dict(color="#1f77b4", width=1.2)))
							if y_noise_sp.size == f_sp.size:
								fig_live.add_trace(go.Scatter(x=f_sp, y=y_noise_sp, mode="lines", name="Noise", line=dict(color="#ff7f0e", width=1.1, dash="dot")))
							if y_pred_sp.size == f_sp.size:
								fig_live.add_trace(go.Scatter(x=f_sp, y=y_pred_sp, mode="lines", name="Synthetic+Noise", line=dict(color="#d62728", width=1.2)))
							fig_live.update_layout(
								title=f"Checkpoint spectral fit | pixel (x={int(px)}, y={int(py)})",
								xaxis_title="Frequency (GHz)",
								yaxis_title="Intensity",
								template="plotly_white",
								height=340,
								margin=dict(l=40, r=20, t=45, b=40),
							)
							st.plotly_chart(fig_live, width="stretch", key="p6_cubefit_live_spectra")
							if not bool(fit_ok):
								st.caption("Last checkpoint pixel had no valid fit. Showing observed spectrum only.")
						else:
							st.caption("No checkpoint spectrum available yet.")
				except Exception as e:
					st.caption(f"Could not load checkpoint preview: {e}")

		progress_map_files = {
			"logN": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_LOGN.fits"),
			"Tex": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_TEX.fits"),
			"Velocity": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_VELOCITY.fits"),
			"FWHM": os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_FWHM.fits"),
		}
		done_mask_cf_path = os.path.join(str(cubefit_out_dir), "CUBEFIT_INPROGRESS_DONE_MASK.fits")
		done_mask_cf = None
		if os.path.isfile(done_mask_cf_path):
			try:
				done_mask_cf = np.asarray(fits.getdata(done_mask_cf_path), dtype=np.float32)
				if done_mask_cf.ndim == 3:
					done_mask_cf = done_mask_cf[0]
				done_mask_cf = np.asarray(done_mask_cf > 0.5, dtype=bool)
			except Exception:
				done_mask_cf = None
		progress_maps_available = {k: v for k, v in progress_map_files.items() if os.path.isfile(v)}
		if progress_maps_available:
			st.markdown("**Live in-progress parameter maps (RA/Dec, zoom to fitted region)**")
			cmap_by_param = {
				"logN": "viridis",
				"Tex": "magma",
				"Velocity": "coolwarm",
				"FWHM": "plasma",
			}
			pm1, pm2 = st.columns(2)
			pm_cols = [pm1, pm2]
			for i_pm, (mk, mp) in enumerate(progress_maps_available.items()):
				with pm_cols[i_pm % 2]:
					try:
						arr_pm = np.asarray(fits.getdata(mp), dtype=np.float32)
						hdr_pm = fits.getheader(mp)
						if arr_pm.ndim == 3:
							arr_pm = arr_pm[0]
						_show_fits_preview(
							f"{mk} (in progress)",
							arr_pm,
							cmap=str(cmap_by_param.get(str(mk), "viridis")),
							ref_hdr=hdr_pm,
							zoom_mask=done_mask_cf,
						)
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
				st.markdown("**Cube fitting parameter maps (final, RA/Dec)**")
				cmap_by_param = {
					"logN": "viridis",
					"Tex": "magma",
					"Velocity": "coolwarm",
					"FWHM": "plasma",
				}
				mc1, mc2 = st.columns(2)
				cols_map = [mc1, mc2]
				for i_m, (mk, mp) in enumerate(available_maps.items()):
					with cols_map[i_m % 2]:
						try:
							arr_m = np.asarray(fits.getdata(mp), dtype=np.float32)
							hdr_m = fits.getheader(mp)
							if arr_m.ndim == 3:
								arr_m = arr_m[0]
							_show_fits_preview(mk, arr_m, cmap=str(cmap_by_param.get(str(mk), "viridis")), ref_hdr=hdr_m)
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

		log_txt_cf = os.path.join(str(cubefit_out_dir), "Log.txt")
		if os.path.isfile(log_txt_cf):
			with st.expander("Cube fitting progress Log.txt", expanded=False):
				st.text(_read_log_tail(log_txt_cf, n_lines=300))

		if _is_cubefit_running():
			st.caption("Auto-updating every 5 seconds...")
			time.sleep(5)
			st.rerun()


def _worker_entry_if_needed() -> bool:
	if "--inverse-cube-worker" in sys.argv:
		idx = sys.argv.index("--inverse-cube-worker")
		if idx + 1 >= len(sys.argv):
			print("Missing inverse-cube config path")
			sys.exit(2)
		cfg_path = sys.argv[idx + 1]
		code = run_inverse_cube_pred_worker(cfg_path)
		sys.exit(int(code))
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
