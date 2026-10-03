"""Frozen full-native export consumer for the official Track1 packer boundary.

Hash/JSON lineage is verified before opening only trajectory.npz. The native
archive, decoded target, models and videos are never deserialized here. Actual
producer replay receipts bind geometry fidelity; this consumer independently
checks engineering integrity and schema, not reconstruction quality/eligibility.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np
import cari_clip_inputs as public
import cari_full_export as export
from world_reward.contracts import Reconstruction
from world_reward.submission import Track1Episode

EXPORT_FILES = {"report.json", *export.OUTPUTS}
TRAJECTORY_KEYS = {"pose", "scales", "shape", "expression", "object_rotation", "object_translation",
                   "object_scale", "object_vertices", "object_faces", "camera_K", "frame_index"}


@dataclass(frozen=True)
class LoadedTrack1Episode:
    episode: Track1Episode
    manifest: dict


def _json(path):
    def unique(pairs):
        result={}
        for key,value in pairs:
            if key in result:raise ValueError("Duplicate JSON producer/pin key forbidden")
            result[key]=value
        return result
    def nonfinite(_):
        raise ValueError("Nonfinite JSON forbidden")
    return json.loads(path.read_text(),object_pairs_hook=unique,parse_constant=nonfinite)


def _directory(value):
    path=Path(value).absolute()
    if (path.resolve()!=path or not path.is_dir() or ".." in path.parts
            or any(parent.is_symlink() for parent in (path,*path.parents))):
        raise ValueError("Canonical existing non-symlink source/code directory required")
    return path


def validate_export_pins(spec,pins):
    if type(spec) is not public.PublicClipSpec:
        raise ValueError("Explicit original full PublicClipSpec required")
    if (type(pins) is not dict or set(pins)!={"schema","clip_spec","export","export_files"}
            or pins["schema"]!="world-reward-cari-shared-export-pins-v1"
            or type(pins["clip_spec"]) is not dict or pins["clip_spec"]!=asdict(spec)
            or public.PublicClipSpec(**pins["clip_spec"])!=spec
            or type(pins["export_files"]) is not dict or set(pins["export_files"])!=EXPORT_FILES):
        raise ValueError("Complete explicit exact full-native export pins required")
    public._receipt(pins["export"],producer=True)
    for row in pins["export_files"].values():public._receipt(row)
    if pins["export_files"]["report.json"]!={key:pins["export"][key] for key in ("sha256","bytes")}:
        raise ValueError("Export report identity differs from the full inventory")


def _inventory(directory,pins):
    paths=tuple(directory.iterdir())
    if {path.name for path in paths}!=EXPORT_FILES or any(path.is_symlink() or not path.is_file() for path in paths):
        raise ValueError("Exactly five regular frozen export files required")
    # Complete five-file hash/readonly proof precedes JSON/NPZ interpretation.
    observed={path.name:export.lineage.identity(path) for path in paths}
    if observed!=pins:raise ValueError("Frozen export SHA/bytes differ from explicit pins")
    return observed


def _array(value,name,shape,dtype=np.float32):
    if (type(value) is not np.ndarray or value.dtype!=np.dtype(dtype) or value.shape!=shape
            or not np.isfinite(value).all()):
        raise ValueError(f"{name}: exact unmasked finite {np.dtype(dtype)} {shape} required")
    return value.copy(order="C")


def _trajectory(path,spec,report,input_sha256):
    count=spec.total_frames
    shapes={"pose":(count,136),"scales":(68,),"shape":(45,),"expression":(72,),
            "object_rotation":(count,3,3),"object_translation":(count,3),"object_scale":()}
    with np.load(path,allow_pickle=False) as archive:
        if len(archive.files)!=len(TRAJECTORY_KEYS) or set(archive.files)!=TRAJECTORY_KEYS:
            raise ValueError("Exact eleven trajectory keys without duplicate members required")
        values={key:_array(archive[key],key,shape) for key,shape in shapes.items()}
        vertices=archive["object_vertices"];faces=archive["object_faces"]
        if vertices.ndim!=2 or vertices.shape[1:]!=(3,) or not 3<=len(vertices)<=4096:
            raise ValueError("Original fixed mesh vertices must fit the official4096-row budget")
        if faces.ndim!=2 or faces.shape[1:]!=(3,) or not 1<=len(faces)<=4096:
            raise ValueError("Original fixed mesh faces must fit the official4096-row budget")
        values["object_vertices"]=_array(vertices,"object_vertices",vertices.shape)
        values["object_faces"]=_array(faces,"object_faces",faces.shape,np.int64)
        values["camera_K"]=_array(archive["camera_K"],"camera_K",(3,3),np.float64)
        values["frame_index"]=_array(archive["frame_index"],"frame_index",(count,),np.int64)
    if not np.array_equal(values["frame_index"],np.arange(count,dtype=np.int64)):
        raise ValueError("Every original frame required in exact order; no padding/drop/repeat")
    if not np.array_equal(values["camera_K"],public.inferred_camera(spec)):
        raise ValueError("Original RGB-size inferred source camera must remain unchanged")
    if float(values["object_scale"])!=1. or np.any(values["expression"]):
        raise ValueError("Original baked object scale1 and one zero expression72 required")
    arrays=tuple(values.values())
    if (any(not value.flags.owndata or not value.flags.c_contiguous for value in arrays)
            or any(np.shares_memory(value,other) for i,value in enumerate(arrays) for other in arrays[i+1:])):
        raise ValueError("Returned trajectory arrays must be independent owned copies")
    reconstruction=Reconstruction(**{key:values[key] for key in shapes if key!="expression"})
    episode=Track1Episode(reconstruction,values["object_vertices"],values["object_faces"],values["expression"],
        dict(input_track="track_1",ground_truth_used=False,hand_labeled_test=False,oracle_modes=[],
             input_sha256=input_sha256,input_dataset_revision=public.DATASET_REVISION),count)
    episode.validate()
    poses=np.tile(np.eye(4,dtype=np.float32),(count,1,1))
    poses[:,:3,:3]=values["object_rotation"];poses[:,:3,3]=values["object_translation"]
    actual=export.object_roundtrip(values["object_vertices"],values["object_faces"],poses,count)
    stored=report.get("object_roundtrip")
    if (type(stored) is not dict or set(stored)!=set(actual)
            or any(type(stored[key]) is not type(value) or stored[key]!=value for key,value in actual.items())):
        raise ValueError("Original object camera-frame schema/positive-depth receipt differs")
    return episode,values,actual


def load_shared_track1_episode(root,code,spec,export_pins):
    """Consume one externally pinned actual export, never old LM predictions.

    Only trajectory.npz is decompressed; all other exports and predecessor
    sources/assets are hash-only. Current immutable producer source and bundled
    actual refined/input pins are mandatory. Returned arrays never alias a
    source archive. No model-forward fidelity/held-out quality is re-established.
    """
    root,code=_directory(root),_directory(code)
    validate_export_pins(spec,export_pins);original_pins=copy.deepcopy(export_pins)
    directory=root/export.output_relative(spec.episode_index)
    observed=_inventory(directory,export_pins["export_files"])
    report=_json(directory/"report.json");export.validate_export_report(report,spec)
    if (any(report.get(key)!=export_pins["export"][key] for key in ("producer_revision","script_sha256"))
            or report["output_files"]!={name:row for name,row in observed.items() if name!="report.json"}):
        raise ValueError("Actual export producer/output manifest differs from explicit pins")
    helpers=export.source_helpers(code)
    if report.get("source_helpers")!=helpers or report.get("script_sha256")!=helpers["infra/cari_full_export.py"]["sha256"]:
        raise ValueError("Current immutable actual export generator/helper closure differs")
    refined_path=code/f"configs/cari_clip_{spec.episode_index:06d}_shared_refined_pins.json"
    input_path=code/f"configs/cari_clip_{spec.episode_index:06d}_input_pins.json"
    refined_id=export.lineage.identity(refined_path);input_id=export.lineage.identity(input_path)
    refined_pins=_json(refined_path);input_pins=_json(input_path);public.validate_pins(spec,input_pins)
    if report.get("refined_pins")!=refined_id or report.get("input_pins")!=input_id:
        raise ValueError("Export must use the exact bundled actual refined/public pins")
    chain=export.lineage.verify_refined_artifacts(root,code,spec,refined_pins)
    refined,prepared=chain["report"],chain["prepare"]["report"]
    sources=chain["prepare"]["source_files"]
    if (sources!=input_pins["source_files"] or report.get("source_files")!=sources
            or report.get("refined_report_sha256")!=chain["files"]["report.json"]["sha256"]
            or report.get("refined_bundle_sha256")!=chain["files"]["refined.pth"]["sha256"]
            or refined.get("input_sha256")!=prepared.get("input_sha256")
            or report.get("body_assets")!=prepared.get("body_assets")
            or report.get("inference_source_identity")!=prepared.get("inference_source_identity")
            or report.get("decoder_identity")!=prepared.get("decoder_identity")):
        raise ValueError("Actual export/refined/public video/model source lineage differs")
    bindings={Path(path):row for path,row in chain["bindings"].items()}
    bindings.update({root/name:row for name,row in sources.items()})
    bindings.update({code/name:row for name,row in helpers.items()})
    bindings.update({refined_path:refined_id,input_path:input_id})
    if any(public.identity(path)!=row for path,row in bindings.items()):
        raise ValueError("Complete original predecessor/source asset bindings changed")
    paths=public.relative_paths(spec)
    wild=_json(root/paths["wild_export"]);edex=_json(root/paths["export_seq"]/"edex")
    public.validate_wild(wild,edex,root,spec)
    source_mesh=sources[paths["mesh"]]
    if (report["aligned_object_mesh_sha256"]!=source_mesh["sha256"]
            or observed["object_aligned.glb"]!=source_mesh or refined.get("object_mesh_sha256")!=source_mesh["sha256"]):
        raise ValueError("Export must retain byte-exact original aligned GLB geometry")
    episode,values,roundtrip=_trajectory(directory/"trajectory.npz",spec,report,prepared["input_sha256"])
    if (_inventory(directory,observed)!=observed or export_pins!=original_pins
            or export.source_helpers(code)!=helpers
            or any(public.identity(path)!=row for path,row in bindings.items())):
        raise ValueError("Frozen export/source/helper/pin bytes changed during consumption")
    manifest=dict(stage="world_reward_shared_native_episode_consumer",episode_index=spec.episode_index,
        frames=spec.total_frames,clip_spec=asdict(spec),input_video_sha256=prepared["input_sha256"],
        input_dataset_revision=public.DATASET_REVISION,input_track="track_1",ground_truth_used=False,
        hand_labeled_test=False,oracle_modes=[],export_report_sha256=observed["report.json"]["sha256"],
        trajectory_sha256=observed["trajectory.npz"]["sha256"],export_files=observed,
        refined_report_sha256=report["refined_report_sha256"],refined_bundle_sha256=report["refined_bundle_sha256"],
        aligned_object_mesh_sha256=source_mesh["sha256"],original_frame_coverage_verified=True,
        integrity_and_schema_verified=True,object_schema_roundtrip=roundtrip,camera_K=values["camera_K"].tolist(),
        original_source_binding_count=len(bindings),native_direct_export_consumed=True,old_LM_conversion_used=False,
        numerical_geometry_independently_reverified=False,quality_verified=False,challenge_performance_verified=False,
        submission_eligibility_verified=False,submission_eligible=False,final_Parquet_produced=False)
    return LoadedTrack1Episode(episode,manifest)
