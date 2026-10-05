"""Manufactured NumPy-backed native API controls, NOT Torch/MMCV qualification."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from types import SimpleNamespace
import ast

import numpy as np
import pytest

from world_reward.hoi_detr_observations import (
    HOIDetrObservations, NativeHOIOperations, infer_hoi_detr_frame,
)


class Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)
        self.device = "fake_native_device"

    @property
    def shape(self):
        return self.values.shape

    def __getitem__(self, item):
        if isinstance(item, Tensor):
            item = item.values
        return Tensor(self.values[item])

    def __mod__(self, number):
        return Tensor(self.values % number)

    def __floordiv__(self, number):
        return Tensor(self.values // number)

    def __mul__(self, value):
        return Tensor(self.values * (value.values if isinstance(value, Tensor) else value))

    def sigmoid(self):
        return Tensor(1 / (1 + np.exp(-self.values)))

    def view(self, *shape):
        return Tensor(self.values.reshape(shape))

    def topk(self, count):
        order = np.argsort(-self.values, kind="stable")[:count].astype(np.int64)
        return Tensor(self.values[order]), Tensor(order)

    def new_tensor(self, values):
        return Tensor(np.array(values, dtype=self.values.dtype))

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class TensorOps:
    long = np.int64

    def __init__(self, events):
        self.events = events
        self.backends = SimpleNamespace(cuda=SimpleNamespace(matmul=SimpleNamespace(allow_tf32=False)),
                                        cudnn=SimpleNamespace(allow_tf32=False))

    def is_autocast_enabled(self, device="cuda"):
        assert device in ("cuda", "cpu")
        return False

    def no_grad(self):
        owner = self
        class Context:
            def __enter__(self):
                owner.events.append("no_grad_enter")
            def __exit__(self, *args):
                owner.events.append("no_grad_exit")
        return Context()

    def as_tensor(self, values, dtype, device):
        assert dtype == np.int64 and device == "fake_native_device"
        return Tensor(np.array(values, dtype=dtype))

    def cat(self, values, dim):
        return Tensor(np.concatenate([v.values for v in values], axis=dim))


class Interaction:
    training = False

    def __init__(self, events):
        self.events = events
        self.inputs = []
        self.loss = None

    def forward(self, pair_embeddings, interaction_targets):
        assert interaction_targets is None
        self.events.append("original_interaction_forward")
        self.inputs.append(pair_embeddings.values.copy())
        # Values deliberately outside[0,1]: logits must never become softmax.
        v = pair_embeddings.values
        return Tensor(np.column_stack((v[:, 0] - v[:, 256], v[:, 0] + v[:, 256])).astype(np.float32)), self.loss

    def forward_logits(self, *_):
        raise AssertionError("Probability shortcut forbidden")


class QueryHead:
    training = False
    num_query = 1500
    num_classes = 3
    embed_dims = 256
    test_cfg = {"max_per_img": 1000}

    def __init__(self, events):
        self.events = events
        self.interaction_head = Interaction(events)
        self.logits = np.full((6, 1, 1500, 3), -8, np.float32)
        for q, cls, score in ((20, 2, .95), (10, 0, .9), (11, 1, .85), (12, 0, .8), (10, 1, .75)):
            self.logits[-1, 0, q, cls] = np.log(score/(1-score))
        self.coords = np.zeros((6, 1, 1500, 4), np.float32)
        self.coords[-1, 0] = [.5, .5, .25, .5]
        self.tokens = np.broadcast_to(np.arange(1500, dtype=np.float32)[None, None, :, None], (6, 1, 1500, 256)).copy()

    def __call__(self, features, metas, *, return_hs):
        assert features == "actual_features" and return_hs is True
        assert metas[0]["batch_input_shape"] == (4, 8)
        self.events.append("query_head_return_hs")
        return (Tensor(self.logits), Tensor(self.coords), None, None, None), Tensor(self.tokens)


class Model:
    training = False

    def __init__(self, events):
        self.events = events
        self.query_head = QueryHead(events)

    def extract_feat(self, image):
        assert image.shape == (1, 3, 4, 8)
        self.events.append("extract_feat")
        return "actual_features"


def fixture(*, keep=(3, 2, 0, 1, 4), scale=(2, 2), nested=True):
    events = []
    model = Model(events)
    rgb = np.arange(2*4*3, dtype=np.uint8).reshape(2, 4, 3)
    metadata = dict(img_shape=(4, 8, 3), ori_shape=rgb.shape, scale_factor=scale)

    def prepare(image):
        assert image.flags.writeable is False and np.array_equal(image, rgb)
        events.append("prepare_original_rgb")
        return dict(img=Tensor(np.zeros((1, 3, 4, 8), np.float32)), img_metas=metadata)

    def collate(rows, *, samples_per_gpu):
        assert len(rows) == 1 and samples_per_gpu == 1
        events.append("collate_one")
        row = rows[0]
        return dict(img=[row["img"]], img_metas=[[row["img_metas"]]] if nested else [row["img_metas"]])

    def scatter(data, devices):
        assert devices == ["fake_native_device"]
        events.append("scatter_one")
        return [data]

    def bbox(coordinates):
        events.append("original_bbox_conversion")
        a = coordinates.values
        return Tensor(np.concatenate((a[:, :2]-a[:, 2:]/2, a[:, :2]+a[:, 2:]/2), axis=-1))

    def nms(boxes, scores, labels, config):
        assert boxes.shape == (1000, 4) and scores.shape == labels.shape == (1000,)
        assert config == dict(type="soft_nms", iou_threshold=.5, min_score=.3)
        events.append("original_cpu_soft_nms")
        indices = np.array(keep, np.int64)
        # Retain a decayed score below.3 to prove original RAW retention remains.
        decayed = np.full(len(indices), .2, np.float32)
        return Tensor(np.column_stack((boxes.values[indices], decayed))), Tensor(indices)

    ops = NativeHOIOperations(prepare, collate, scatter, bbox, nms, TensorOps(events), "fake_native_device")
    return model, rgb, ops, events


def test_real_extract_query_and_original_pair_forward_in_native_order():
    model, rgb, ops, events = fixture()
    result = infer_hoi_detr_frame(model, rgb, 37, ops)
    assert events[:7] == ["prepare_original_rgb", "collate_one", "scatter_one", "no_grad_enter",
                          "extract_feat", "query_head_return_hs", "original_bbox_conversion"]
    assert events[7:] == ["original_cpu_soft_nms", "original_interaction_forward", "original_interaction_forward", "no_grad_exit"]
    assert result.original_frame_index == 37 and result.image_size == (2, 4)
    assert result.query_ids.tolist() == [12, 11, 20, 10, 10]
    assert result.class_ids.tolist() == [0, 1, 2, 0, 1]
    assert result.hand_object_pairs.tolist() == [[0, 1], [0, 4], [3, 1], [3, 4]]
    assert result.object_target_pairs.tolist() == [[1, 2], [4, 2]]
    assert model.query_head.interaction_head.inputs[0][:, (0, 256)].tolist() == [[12, 11], [12, 10], [10, 11], [10, 10]]
    assert result.hand_object_logits.tolist() == [[1, 23], [2, 22], [-1, 21], [0, 20]]
    assert result.object_target_logits.tolist() == [[-9, 31], [-10, 30]]
    np.testing.assert_array_equal(result.boxes_original_xyxy, np.tile(np.array([1.5, .5, 2.5, 1.5], np.float32), (5, 1)))
    assert np.all(result.raw_scores >= .3) and np.all(result.decayed_scores == np.float32(.2))
    assert result.query_tokens.shape == (1500, 256)  # Not only surviving tokens.


def test_readonly_no_aliases_and_original_input_unchanged():
    model, rgb, ops, _ = fixture(); original = rgb.copy()
    result = infer_hoi_detr_frame(model, rgb, 0, ops)
    for name, value in vars(result).items():
        if isinstance(value, np.ndarray):
            assert not value.flags.writeable and value.flags.owndata
            with pytest.raises(ValueError):
                value.reshape(-1)[0] = 42
    assert not np.shares_memory(result.query_tokens, model.query_head.tokens)
    model.query_head.tokens[:] = 99
    assert result.query_tokens[10, 0] == 10
    np.testing.assert_array_equal(rgb, original)
    with pytest.raises(FrozenInstanceError):
        result.original_frame_index = 99


@pytest.mark.parametrize("scale", [(2,), (2, 2), (2, 2, 2, 2)])
def test_only_exact_native_scale_factor_forms_no_added_transform(scale):
    model, rgb, ops, _ = fixture(scale=scale, nested=False)
    np.testing.assert_array_equal(infer_hoi_detr_frame(model, rgb, 0, ops).boxes_original_xyxy[0], [1.5, .5, 2.5, 1.5])


@pytest.mark.parametrize("keep", [(), (0,), (1, 3), (2, 4)])
def test_genuine_empty_or_one_role_has_empty_pairs_without_dummy_forward(keep):
    model, rgb, ops, events = fixture(keep=keep)
    result = infer_hoi_detr_frame(model, rgb, 0, ops)
    assert len(result.query_ids) == len(keep)
    assert result.hand_object_pairs.shape == result.object_target_pairs.shape == (0, 2)
    assert result.hand_object_logits.shape == result.object_target_logits.shape == (0, 2)
    assert "original_interaction_forward" not in events


def test_native_raw_threshold_and_all_native_keep_slots_retained():
    model, rgb, ops, _ = fixture(keep=(999, 0, 1))
    result = infer_hoi_detr_frame(model, rgb, 0, ops)
    assert result.native_nms_keep.tolist() == [999, 0, 1]
    assert result.retained_nms_positions.tolist() == [1, 2]
    assert result.query_ids.tolist() == [20, 10]
    assert len(result.native_nms_detections) == 3


@pytest.mark.parametrize("fault", ["train", "query_count", "class_count", "topk", "token_dim", "shape", "dtype", "nonfinite", "unselected_nan", "keep_repeat", "keep_oob", "scale", "supervised_loss"])
def test_malformed_native_outputs_or_policy_propagate_not_empty(fault):
    model, rgb, ops, _ = fixture()
    head = model.query_head
    if fault == "train": model.training = True
    elif fault == "query_count": head.num_query = 900
    elif fault == "class_count": head.num_classes = 2
    elif fault == "topk": head.test_cfg = {"max_per_img": 300}
    elif fault == "token_dim": head.embed_dims = 128
    elif fault == "shape": head.logits = head.logits[:1]
    elif fault == "dtype": head.tokens = head.tokens.astype(np.float64)
    elif fault == "nonfinite": head.tokens[-1, 0, 10, 0] = np.nan
    elif fault == "unselected_nan": head.logits[-1, 0, 1499, 0] = np.nan
    elif fault == "keep_repeat": ops = fixture(keep=(0, 0))[2]
    elif fault == "keep_oob": ops = replace(ops, batched_nms=lambda *_: (Tensor(np.zeros((1, 5), np.float32)), Tensor([1000])))
    elif fault == "scale": ops = fixture(scale=(0, 2))[2]
    else: head.interaction_head.loss = 0.0
    with pytest.raises(ValueError):
        infer_hoi_detr_frame(model, rgb, 0, ops)


@pytest.mark.parametrize("kind", ["autocast_cuda", "autocast_cpu", "matmul", "cudnn"])
def test_amp_tf32_caller_policy_fail_before_any_native_call(kind):
    model, rgb, ops, events = fixture()
    if kind.startswith("autocast"):
        ops.tensor_ops.is_autocast_enabled = lambda device="cuda": device == kind.split("_")[1]
    elif kind == "matmul": ops.tensor_ops.backends.cuda.matmul.allow_tf32 = True
    else: ops.tensor_ops.backends.cudnn.allow_tf32 = True
    with pytest.raises(ValueError, match="disable AMP and TF32"):
        infer_hoi_detr_frame(model, rgb, 0, ops)
    assert not events


@pytest.mark.parametrize("kind", ["original_shape", "processed_shape", "extra_reference_key", "mutate_both_images"])
def test_original_native_preprocessing_metadata_no_reference_or_image_mutation(kind):
    model, rgb, ops, _ = fixture()
    original = ops.prepare_rgb
    def bad(image):
        data = original(image)
        if kind == "original_shape": data["img_metas"]["ori_shape"] = (2, 4, 4)
        elif kind == "processed_shape": data["img_metas"]["img_shape"] = (9, 8, 3)
        elif kind == "extra_reference_key": data["gt_labels"] = [0]
        else:
            image.flags.writeable = True; image[0, 0, 0] = 255; rgb[0, 0, 0] = 255
        return data
    with pytest.raises(ValueError):
        infer_hoi_detr_frame(model, rgb, 0, replace(ops, prepare_rgb=bad))


def test_all_raw_queries_finite_before_any_policy_selection():
    model, rgb, ops, events = fixture()
    model.query_head.tokens[-1, 0, 1499, 255] = np.inf
    with pytest.raises(ValueError, match="all native tokens"):
        infer_hoi_detr_frame(model, rgb, 0, ops)
    assert "original_bbox_conversion" not in events and "original_cpu_soft_nms" not in events


def test_native_boxes_outside_grid_preserved_not_clipped():
    model, rgb, ops, _ = fixture()
    model.query_head.coords[-1, 0, :, :] = [1.5, -.5, .25, .5]
    result = infer_hoi_detr_frame(model, rgb, 0, ops)
    np.testing.assert_array_equal(result.boxes_original_xyxy[0], [5.5, -1.5, 6.5, -.5])


def test_postprocessing_cannot_change_preserved_query_tokens():
    model, rgb, ops, _ = fixture()
    original = model.query_head.interaction_head.forward
    def bad(*args, **kwargs):
        value = original(*args, **kwargs)
        model.query_head.tokens[-1, 0, 10, 0] += 1
        return value
    model.query_head.interaction_head.forward = bad
    with pytest.raises(ValueError, match="changed decoder evidence"):
        infer_hoi_detr_frame(model, rgb, 0, ops)


@pytest.mark.parametrize("where", ["pipeline", "extract", "query", "nms", "pair"])
def test_native_errors_propagate_no_retry_or_exception_to_empty(where):
    model, rgb, ops, events = fixture()
    def fail(*args, **kwargs):
        events.append("failed_once")
        raise RuntimeError("native failure")
    if where == "pipeline": ops = replace(ops, prepare_rgb=fail)
    elif where == "extract": model.extract_feat = fail
    elif where == "query": model.query_head.__class__ = type("BadQuery", (QueryHead,), {"__call__": fail})
    elif where == "nms": ops = replace(ops, batched_nms=fail)
    else: model.query_head.interaction_head.forward = fail
    with pytest.raises(RuntimeError, match="native failure"):
        infer_hoi_detr_frame(model, rgb, 0, ops)
    assert events.count("failed_once") == 1


@pytest.mark.parametrize("image,index", [(np.zeros((2, 4, 3), np.float32), 0), (np.zeros((2, 4), np.uint8), 0),
                                        (np.zeros((0, 4, 3), np.uint8), 0), (np.zeros((2, 4, 3), np.uint8), True)])
def test_original_rgb_and_frame_contract(image, index):
    model, _, ops, events = fixture()
    with pytest.raises(ValueError): infer_hoi_detr_frame(model, image, index, ops)
    assert not events


def test_dataclass_refuses_selected_or_permuted_pairs_and_invalid_classes():
    model, rgb, ops, _ = fixture()
    result = infer_hoi_detr_frame(model, rgb, 0, ops)
    with pytest.raises(ValueError, match="Every ordered"):
        replace(result, hand_object_pairs=result.hand_object_pairs[::-1])
    with pytest.raises(ValueError, match="Original proposal"):
        replace(result, class_ids=np.full(5, 3, np.int64))
    with pytest.raises(ValueError, match="Raw/native-decayed"):
        replace(result, decayed_scores=result.raw_scores)


def test_source_is_direct_native_image_only_not_dynamic_upstream_import_or_probability():
    source = Path(__file__).resolve().parents[1]/"src/world_reward/hoi_detr_observations.py"
    tree = ast.parse(source.read_text())
    imports = [a.name for n in ast.walk(tree) if isinstance(n, (ast.Import, ast.ImportFrom)) for a in n.names]
    assert not any(n.startswith(("torch", "mmdet", "mmcv", "cv2", "importlib")) for n in imports)
    calls = [ast.unparse(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)]
    assert "model.extract_feat" in calls and "head" in calls and "head.interaction_head.forward" in calls
    assert not any("forward_logits" in n or "softmax" in n for n in calls)
    assert not any(isinstance(n, ast.ExceptHandler) for n in ast.walk(tree))
    assert HOIDetrObservations.__dataclass_params__.frozen
