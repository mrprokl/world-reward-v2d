"""Public, pinned DA3 metric weights/minimal source acquisition on remote Linux.

No imports of upstream code, installs, inference, challenge data or credentials.
Verified source/license records are retained; disposable downloads are removed.
An Apache declaration is not a verified challenge-overlap or accuracy audit.
"""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import re
import shutil
import signal
import sys
import time
import urllib.parse
import urllib.request

MODEL_REV = "4010e39f3634a45bc60553321fb49fb760bd594e"
SOURCE_REV = "3d835ec1a5802d64a8b8b15f817a1ab54809bfe4"
MODEL_REPO = "depth-anything/DA3METRIC-LARGE"
SOURCE_REPO = "ByteDance-Seed/Depth-Anything-3"
WEIGHTS = "weights/research/da3_metric_v1"
SOURCE = "vendor/research/da3_metric_v1"
REPORT = "results/da3-metric-acquisition-v1.json"
MODEL_POINTER_SHA = "3c8151143044c67fbf5afaa7ae4e30c151e9bb15"
MODEL_RECORDS = (
    ("config.json", 847, "a336f3e76fe375aaae17a9aed9130c9f2aa061535d317ec57dcb2f1f02e1dd53", "3f50fc09637d19fdf0998946aa383bc377de9ade"),
    ("README.md", 4789, "4fdd25492cf6401d1459483fd829f6abb399bda8218e1a3ef5ad4a66e9aa5a2a", "0856df0c9e59168474ff856705849ca6738cb713"),
    ("model.safetensors", 1336734448, "bbea5b0b3ee389849cffa7ddae89de064a90abd2b055fc5aa99aac68db324776", None),
)
# Each primary Git tree member and raw SHA256 was independently checked at pin.
SOURCE_RECORDS = (
    ("LICENSE", 11355, "78446e29c48900cda82620a8df183cca61f0a595e05a49d0401a5fd604dd1870", "a2fa36578f5094f56f4b0ec57e70eb30aced4519"),
    ("README.md", 18759, "91679cf6df290b6af439c85f29a2440ce10353a1063ae4f599e9c734ec3096a8", "9ef877b6ddf6d86ed58566183e79ac74ec00284c"),
    ("pyproject.toml", 2039, "3e178d62b8ca674ee4685a5c2a89ab8363e204b9af24d7313bdb8e3c03926544", "6a11c7ae0e2978de8a726d3ac9e6668db0c5248a"),
    ("requirements.txt", 214, "0809fc8cf0bfe2a8c79b7e525b96bedef71e5866a654c0cc915c572d20c8eba8", "878db3f3cbb194289faa99b4c96c5e63488096c8"),
    ("src/depth_anything_3/cfg.py", 5037, "655a018ab35ad715454fdbca047354591287b6b86c10c91b8c0a920df0153014", "4607ff8c0983b67e6ccd2d80b78a163c4e487250"),
    ("src/depth_anything_3/configs/da3metric-large.yaml", 508, "b7051f0e1b000b7bc92d190fe87db5338364560aea2d290e0400eacf0d7dcb07", "124635cfd952c25c8857ee3da63ce0444c4377f2"),
    ("src/depth_anything_3/model/__init__.py", 755, "e6fd99e59aec06a5f8a19255d11e7ecd9fa6482bdacad0e3e1a8fe2991cd64b6", "57a2a45132eeae8d58a11a26036c54feef9cfe16"),
    ("src/depth_anything_3/model/da3.py", 17641, "c7d5f88962e05c021cf7dab505840c8892e5e9d6880fb2901d20f4120ca173b3", "f9e5bd683a8a697647da30e4d93a0ead03e9545d"),
    ("src/depth_anything_3/model/dinov2/dinov2.py", 1824, "5d8e603113731365b3f46a3df4d159ad58f8723ec011440bf1961a4cbe101ca2", "ee0d88bdd6d33edda6c0fb237ee0c77ffc3f6034"),
    ("src/depth_anything_3/model/dinov2/layers/__init__.py", 644, "321a97ffc00375d6d48569cbc851a38e7d2dc8bbdbfcffbee722c6c27783acda", "97dfba90c12b2ffc5f0f1f823b6384c9b4cd6fa2"),
    ("src/depth_anything_3/model/dinov2/layers/attention.py", 3264, "d9db2bd66c612df689ec19aa2b60138c4185e38c8cfaaef48fc17f39a5b17c81", "096b9d41ddc95b9c4652597b18f53aee31a573b6"),
    ("src/depth_anything_3/model/dinov2/layers/block.py", 5010, "88924cf5cd9fb18b17a8329e4c9910528205c302d8be40be29ad74e97ae7a3b8", "731519b68b16c8936b765dd1620fbb9c81087f96"),
    ("src/depth_anything_3/model/dinov2/layers/drop_path.py", 1146, "80b5cc50297f3049d313c5659c1244901407bdac80cf424c5d7ed64ae79cd141", "1c2cc94e969711f1eb9f62093b79a0139b9bfb1e"),
    ("src/depth_anything_3/model/dinov2/layers/layer_scale.py", 1000, "23aba2710d7d425b5abd50c006edef9c9224360d6499171b6c1ec18ef73c0b00", "898ee12d8b4b65d30d8c041588a8277a8f13d4f2"),
    ("src/depth_anything_3/model/dinov2/layers/mlp.py", 1271, "4f17c26120d04e4e4368b07a2396a977044b00d1dcbf1bbd1a9d3447be1aa6cf", "78ad0d8897ddb77e95d2e188a579f6e1d21e3fb5"),
    ("src/depth_anything_3/model/dinov2/layers/patch_embed.py", 2903, "72f14d7b3daf3f545ba93c085b0a90f3b63efcf1a14f8d694dc77763c0607528", "64bf6be8994fe52b2fdf1753fdb9f7a691e14980"),
    ("src/depth_anything_3/model/dinov2/layers/rope.py", 7793, "dcd09808cf03ceae7abfdcbe6805ee36990412b5308b09a86c4b51a040cff38b", "f75ba37c160cd806a32576e2a1704b3352dec7af"),
    ("src/depth_anything_3/model/dinov2/layers/swiglu_ffn.py", 1858, "f0e0bab6e9e835b81c53bd69616af43d254a91af4b119923160b7df1602b1562", "c8f58e5b265f74c82a4c40adc3ee33a965e503cf"),
    ("src/depth_anything_3/model/dinov2/vision_transformer.py", 17574, "c28650e5db0b578745f5a41336709d54d08f952bc08cf1bd2009e2b765111221", "f0f8e23ad21d8c5a60626955d38d740f98bb5c48"),
    ("src/depth_anything_3/model/dpt.py", 17438, "d4f70e867cb31fd26e807746b63fe21517e5385464acee0096a3021ec928576c", "337a8a964a7f96fae997c8b12e3ae13e99dbaa58"),
    ("src/depth_anything_3/model/reference_view_selector.py", 7858, "5e1238298a0053e76eab1f2cfd207ae8b0f30fb441c24c915320efed349301e3", "f406f5ff2e0dad5ce2d12f3a1d3919773fc21e66"),
    ("src/depth_anything_3/model/utils/head_utils.py", 8107, "efed71ecd3b742d5969d1a938bbf34c9923d202dfd5232f204641c4674f0bd6b", "c1209582bc6e9b1658a0358df03f5ea79fe61daf"),
    ("src/depth_anything_3/model/utils/transform.py", 6549, "98d5c499f40e6e3e63cdaf0e269b8eeeb998ed125bcd63da41e2dec9ad32b8b9", "8d732b093e5ad1578bc0ba5eb0e31b1b69b766ed"),
    ("src/depth_anything_3/specs.py", 1686, "d1835f7e15db46d50d6cb27ecedbd8a358ad06d12fc6588fd271fd0fdbcd2df0", "fe5b30255e9fd48d988ff00c896fb3dbadf197ea"),
    ("src/depth_anything_3/utils/alignment.py", 4907, "7a18fe1c050e42970c7e243adb073ca695e24d197f671d74d16655760a827a88", "ceb8983fbb85fd8cad51fdd82b794aa3a114df78"),
    ("src/depth_anything_3/utils/constants.py", 9486, "631d18da0159ef6f4a30be446b733b60cbc10f7b867ef61fabeddf89ab380612", "25c330ee00820f44df05aa624f0aa4763afe1164"),
    ("src/depth_anything_3/utils/geometry.py", 16998, "3d67391f2eeed57ee5d07387947dea7a5d99fc0ce0a63c6890485183c851c29b", "41a42190d637ef25db3599cc90f9a4f40be3b8d7"),
    ("src/depth_anything_3/utils/io/input_processor.py", 18924, "cdebb6021f5f1001d3aa56261c224a80b9d8431efedfb040537c58a5509410c1", "fa60194123c22c84a0cdeb577cbf10e2704278bb"),
    ("src/depth_anything_3/utils/io/output_processor.py", 5863, "4c877deb4550736907162ae397c11a4b726a49fcbb2ab2c901e536814033469e", "c317eb9d596c1687b5281891a035993868cc5f8c"),
    ("src/depth_anything_3/utils/logger.py", 2513, "03ea0032564443ae6df63aa88ab9e4e63b5933ed10463acee607d0bea5836910", "0eb4f60696a085001cf4866ccfe1654170702a2d"),
    ("src/depth_anything_3/utils/parallel_utils.py", 4288, "e691c31ab459e31e5e98231d67ef519199576b9425efd07273b7e137592514b6", "9ff108e95d205097f9f2012ab87ad9e265d58d8d"),
    ("src/depth_anything_3/utils/ray_utils.py", 20115, "d27adc973a03dec912d39f7b7bfa0328c3fd8eabe9f07a43b1bf04ed326f8563", "6244dc40089a7bfda7889ddc153d48c44f717da7"),
)


def digest(path):
    result = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""): result.update(block)
    return result.hexdigest()


def validate_https(url):
    parsed = urllib.parse.urlsplit(url); host = parsed.hostname or ""
    if (parsed.scheme != "https" or parsed.username or parsed.password or parsed.port not in (None, 443)
            or not (host in ("huggingface.co", "raw.githubusercontent.com", "api.github.com") or host.endswith((".hf.co", ".huggingface.co")))):
        raise ValueError("Require public allowlisted HTTPS source")


class HTTPSRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, url):
        validate_https(url)
        return super().redirect_request(request, fp, code, message, headers, url)


def open_public(url):
    validate_https(url)
    return urllib.request.build_opener(HTTPSRedirects()).open(
        urllib.request.Request(url, headers={"Accept-Encoding": "identity"}), timeout=30)


def fetch_json(url):
    with open_public(url) as response:
        validate_https(response.geturl()); data = response.read(300_001)
    if len(data) > 300_000: raise ValueError("Primary metadata exceeds fixed byte bound")
    return json.loads(data)


def primary_metadata():
    tree_url = f"https://api.github.com/repos/{SOURCE_REPO}/git/trees/{SOURCE_REV}?recursive=1"
    tree = fetch_json(tree_url)
    if tree.get("sha") != SOURCE_REV or tree.get("truncated") is not False: raise ValueError("Require complete pinned source Git tree")
    members = {r["path"]: r for r in tree["tree"]}
    for name, size, _, blob in SOURCE_RECORDS:
        member = members.get(name, {})
        if (member.get("sha"), member.get("size"), member.get("type"), member.get("mode")) != (blob, size, "blob", "100644"):
            raise ValueError("Pinned source path/Git blob/size mismatch")
    model_url = f"https://huggingface.co/api/models/{MODEL_REPO}/revision/{MODEL_REV}?blobs=true"
    model = fetch_json(model_url)
    if model.get("sha") != MODEL_REV or model.get("cardData", {}).get("license") != "apache-2.0": raise ValueError("Model revision/licence differs from publisher")
    files = {r["rfilename"]: r for r in model["siblings"]}
    for name, size, sha, blob in MODEL_RECORDS:
        item = files.get(name, {})
        if item.get("size") != size or item.get("blobId") != (blob or MODEL_POINTER_SHA): raise ValueError("Publisher model file identity differs")
        if blob is None and (item.get("lfs", {}).get("sha256"), item.get("lfs", {}).get("size")) != (sha, size): raise ValueError("Independent publisher model SHA/size differs")
    return {"source_git_tree_verified": True, "source_tree_url": tree_url, "publisher_model_manifest_verified": True, "publisher_manifest_url": model_url}


def download(url, destination, size, sha, git_blob):
    """Exclusive bounded streaming download; never report signed redirect URLs."""
    if not 0 < size < 2_000_000_000: raise ValueError("Invalid fixed download size")
    count = 0; content = hashlib.sha256(); git = hashlib.sha1(f"blob {size}\0".encode())
    with open_public(url) as response, destination.open("xb") as output:
        validate_https(response.geturl())
        if response.headers.get("Content-Encoding", "identity") != "identity": raise ValueError("Unexpected encoded asset")
        declared = response.headers.get("Content-Length")
        if declared is not None and int(declared) != size: raise ValueError("Pinned asset declared byte count mismatch")
        while block := response.read(min(1024*1024, size-count+1)):
            count += len(block)
            if count > size: raise ValueError("Pinned asset exceeded exact byte count")
            content.update(block); git.update(block); output.write(block)
    if count != size or content.hexdigest() != sha or (git_blob is not None and git.hexdigest() != git_blob):
        raise ValueError("Pinned asset SHA/byte/Git blob mismatch")
    destination.chmod(0o444)


def save_json(path, value):
    with path.open("x") as stream: json.dump(value, stream, indent=2, allow_nan=False); stream.write("\n")
    path.chmod(0o444)


def acquire(root, report, persist):
    weights, source = root/WEIGHTS, root/SOURCE; staging = weights/".downloads"; staging.mkdir(mode=0o700)
    try:
        report.update(primary_metadata()); persist()
        for base, revision, repository, records in ((SOURCE, SOURCE_REV, SOURCE_REPO, SOURCE_RECORDS), (WEIGHTS, MODEL_REV, MODEL_REPO, MODEL_RECORDS)):
            for name, size, sha, blob in records:
                relative = PurePosixPath(name)
                if relative.is_absolute() or any(p in ("", ".", "..") for p in name.split("/")) or "\\" in name: raise ValueError("Unsafe pinned asset path")
                url = (f"https://raw.githubusercontent.com/{repository}/{revision}/{name}" if base == SOURCE else
                       f"https://huggingface.co/{repository}/{'resolve' if blob is None else 'raw'}/{revision}/{name}")
                report.update(phase="download", active_file=base+"/"+name); persist()
                temporary = staging/"asset"; download(url, temporary, size, sha, blob)
                destination = root/base/name; destination.parent.mkdir(parents=True, exist_ok=True)
                os.link(temporary, destination); temporary.unlink()  # exclusive publication; no overwrite
                report["files"].append({"file": base+"/"+name, "url": url, "bytes": size, "sha256": sha, "git_blob_sha1": blob}); persist()
            if base == SOURCE:
                entries = [{**r, "file": str(Path(r["file"]).relative_to(SOURCE))} for r in report["files"]]
                manifest = {"schema": "world-reward-da3-metric-source-v1", "source_revision": SOURCE_REV, "python_files": 27,
                            "python_bytes": 192452, "source_bytes": 225327, "files": entries, "source_modified": False,
                            "root_namespace_init_fabricated": False, "high_level_api_acquired": False}
                save_json(source/"source_manifest.json", manifest); report["source_manifest_sha256"] = digest(source/"source_manifest.json")
        requirements = (source/"requirements.txt").read_text().splitlines(); imports = set()
        for name, *_ in SOURCE_RECORDS:
            if name.endswith(".py"):
                for node in ast.walk(ast.parse((source/name).read_text())):
                    if isinstance(node, ast.Import): imports.update(a.name.split(".")[0] for a in node.names)
                    elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0: imports.add(node.module.split(".")[0])
        report.update(upstream_requirements=[r for r in requirements if r and not r.startswith("#")],
                      runtime_external_import_names=sorted(imports-{"depth_anything_3"}-sys.stdlib_module_names),
                      requirements_installed=False, runtime_dependencies_resolved=False, phase="complete", status="pass")
        report.pop("active_file", None)
    finally:
        shutil.rmtree(staging); report["disposable_downloads_removed"] = True
        for directory in (weights, source):
            for folder in sorted((p for p in directory.rglob("*") if p.is_dir()), reverse=True): folder.chmod(0o555)
            directory.chmod(0o555)


def main(argv=None):
    argparse.ArgumentParser(description=__doc__, allow_abbrev=False).parse_args(argv)
    if platform.system() != "Linux": raise RuntimeError("Model acquisition stays remote Linux only")
    root = Path(os.environ["WR_ROOT"]); revision = os.environ.get("WR_CODE_REVISION", "")
    if (not root.is_dir() or root.resolve() != root.absolute() or not re.fullmatch(r"[0-9a-f]{40}", revision)
            or os.environ.get("WR_DA3_OUTPUTS_RESERVED") != "1" or os.geteuid() != 1000): raise ValueError("Require canonical remote root, source revision and UID1000 reservation")
    for path in (root/WEIGHTS, root/SOURCE, root/REPORT):
        if any(p.is_symlink() for p in (path, *path.parents)) or (path == root/REPORT and path.exists()): raise FileExistsError("DA3 targets must be fresh nonsymlink paths")
        if path != root/REPORT and (not path.is_dir() or any(path.iterdir())): raise ValueError("DA3 output directories must be reserved empty")
    report = {"stage": "pinned_da3_metric_model_and_minimal_source_acquisition", "status": "fail", "phase": "primary_metadata", "files": [],
              "code_revision": revision, "script_sha256": digest(Path(__file__)), "model_revision": MODEL_REV, "source_revision": SOURCE_REV,
              "model_repository": MODEL_REPO, "source_repository": SOURCE_REPO, "license": "Apache-2.0", "model_separate_license_file_present": False,
              "source_license_primary_url": f"https://raw.githubusercontent.com/{SOURCE_REPO}/{SOURCE_REV}/LICENSE",
              "model_license_primary_url": f"https://huggingface.co/{MODEL_REPO}/raw/{MODEL_REV}/README.md",
              "budget_seconds": 720, "device": "cpu", "gpu_used": False, "challenge_inputs_used": False, "inference_performed": False,
              "source_modified": False, "challenge_overlap_verified": False, "accuracy_verified": False, "network": "public_https_acquisition_only"}
    started = time.perf_counter(); path = root/REPORT
    with path.open("x") as stream:
        def persist():
            report["elapsed_seconds"] = time.perf_counter()-started; stream.seek(0); json.dump(report, stream, indent=2, allow_nan=False)
            stream.write("\n"); stream.truncate(); stream.flush(); os.fsync(stream.fileno())
        def expired(*_): raise TimeoutError("Pinned DA3 metric acquisition exceeded720s")
        alarm = signal.signal(signal.SIGALRM, expired); term = signal.signal(signal.SIGTERM, expired); signal.alarm(720)
        try: persist(); acquire(root, report, persist)
        except Exception as error: report.update(status="fail", error_type=type(error).__name__, error="Acquisition failed; no signed URLs or credentials logged"); raise RuntimeError("Pinned DA3 acquisition failed; inspect immutable receipt") from None
        finally:
            signal.alarm(0); signal.signal(signal.SIGALRM, alarm); signal.signal(signal.SIGTERM, term); persist(); path.chmod(0o444)


if __name__ == "__main__": main()
