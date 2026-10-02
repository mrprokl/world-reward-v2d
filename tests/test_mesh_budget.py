"""Dependency stubs exercise retry policy without local simplification/rendering."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest

from world_reward.mesh_budget import _allocations, fit_topology_preserving_budget


class FakeMesh:
    """Tiny tetrahedral arrays with controllable component topology metadata."""

    def __init__(self, vertices=None, faces=None, process=False, *, count=5000, sign=1,
                 components=None, closed=True, euler=2):
        self.vertices = np.array([[0., 0, 0], [1., 0, 0], [0., 1., 0], [0., 0, 1.]]) if vertices is None else np.array(vertices)
        tetra_faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
        self.faces = np.resize(tetra_faces, (count, 3)) if faces is None else np.array(faces)
        self.volume = float(sign)
        self.is_watertight = closed
        self.is_winding_consistent = True
        self.euler_number = euler
        self.edges_unique_inverse = np.repeat(np.arange(6), 2) if closed else np.array([0, 0, 0])
        self._components = components

    def split(self, *, only_watertight, repair):
        assert only_watertight is False and repair is False
        return [self] if self._components is None else self._components

    def copy(self):
        result = FakeMesh(self.vertices.copy(), self.faces.copy(), sign=np.sign(self.volume),
                          components=self._components, closed=self.is_watertight, euler=self.euler_number)
        result.volume = self.volume
        return result


def nested_source():
    outer, cavity = FakeMesh(count=4000), FakeMesh(count=1000, sign=-1)
    source = FakeMesh(count=5000, components=[outer, cavity])
    source.volume = .1
    return source


@pytest.fixture
def dependencies(monkeypatch):
    calls = []
    produced = []

    def construct(vertices, faces, process=False):
        assert process is False
        if produced:
            result = produced.pop(0)
            return result
        return FakeMesh(vertices, faces)

    # isinstance requires a real type; its metaclass routes constructor results
    # while retaining FakeMesh instances as accepted mesh objects.
    class Meta(type):
        def __instancecheck__(cls, instance):
            return isinstance(instance, FakeMesh)

        def __call__(cls, *args, **kwargs):
            return construct(*args, **kwargs)

    class Trimesh(metaclass=Meta):
        pass

    def concatenate(parts):
        faces = np.concatenate([part.faces for part in parts])
        result = FakeMesh(count=len(faces), components=parts)
        result.volume = sum(part.volume for part in parts)
        return result

    def simplify(vertices, faces, *, target_count):
        calls.append((len(faces), target_count))
        return vertices, faces[:target_count]

    monkeypatch.setitem(sys.modules, "trimesh", SimpleNamespace(Trimesh=Trimesh, util=SimpleNamespace(concatenate=concatenate)))
    monkeypatch.setitem(sys.modules, "fast_simplification", SimpleNamespace(simplify=simplify))
    return calls, produced


def test_first_valid_global_target_wins_without_more_attempts(dependencies):
    calls, produced = dependencies
    source = FakeMesh()
    produced.append(FakeMesh(count=4096))
    result, report = fit_topology_preserving_budget(source)
    assert len(result.faces) == 4096 and calls == [(5000, 4096)]
    assert report["selected_mode"] == "global" and len(report["attempts"]) == 1
    assert report["target_faces"] == [4096, 4080, 4000, 3840]


def test_nonmanifold_artifact_is_regenerated_at_next_fixed_target(dependencies):
    calls, produced = dependencies
    produced.extend([FakeMesh(count=4096, closed=False), FakeMesh(count=4080)])
    _, report = fit_topology_preserving_budget(FakeMesh())
    assert calls == [(5000, 4096), (5000, 4080)]
    assert report["attempts"][0]["status"] == "fail" and report["selected_target"] == 4080


def test_componentwise_retains_outer_and_negative_cavity_without_inversion(dependencies):
    calls, produced = dependencies
    source = nested_source()
    # Four invalid globals; then exact proportional reductions preserving signs.
    produced.extend([FakeMesh(count=count) for count in [4096, 4080, 4000, 3840]])
    allocation = _allocations([4000, 1000], 4096)
    outer_reduced = FakeMesh(count=allocation[0], sign=1)
    cavity_reduced = FakeMesh(count=allocation[1], sign=-1)
    outer_reduced.volume, cavity_reduced.volume = .93, -.87
    produced.extend([outer_reduced, cavity_reduced])
    result, report = fit_topology_preserving_budget(source)
    assert report["selected_mode"] == "componentwise"
    assert [component["volume_sign"] for component in report["selected"]["components"]] == [1, -1]
    assert report["attempts"][-1]["component_target_faces"] == allocation
    assert result.volume > 0 and report["normals_inverted"] is False


def test_extra_closed_component_is_not_removed_to_force_validity(dependencies):
    _, produced = dependencies
    parts = [FakeMesh(count=2044), FakeMesh(count=2048, sign=-1), FakeMesh(count=4)]
    artifact = FakeMesh(count=4096, components=parts)
    artifact.volume = .1
    produced.extend([artifact, FakeMesh(count=4080)])
    _, report = fit_topology_preserving_budget(FakeMesh())
    assert report["attempts"][0]["status"] == "fail"
    assert "component count" in report["attempts"][0]["reason"]


def test_all_bad_targets_fail_instead_of_deleting_triangles(dependencies):
    calls, produced = dependencies
    produced.extend([FakeMesh(count=100, closed=False) for _ in range(8)])
    with pytest.raises(ValueError, match="No topology-preserving"):
        fit_topology_preserving_budget(FakeMesh())
    assert len(calls) == 8


def test_unchanged_valid_geometry_is_copied_without_simplifier(dependencies):
    calls, _ = dependencies
    source = FakeMesh(count=4)
    result, report = fit_topology_preserving_budget(source)
    assert result is not source and report["selected_mode"] == "unchanged" and calls == []
    np.testing.assert_array_equal(result.vertices, source.vertices)
    np.testing.assert_array_equal(result.faces, source.faces)


def test_vertex_budget_failure_is_not_ignored(dependencies):
    _, produced = dependencies
    candidate = FakeMesh(count=4096)
    candidate.vertices = np.concatenate((candidate.vertices, np.zeros((10, 3))))
    produced.extend([candidate, FakeMesh(count=4080)])
    _, report = fit_topology_preserving_budget(FakeMesh(), vertex_budget=4)
    assert "budgets" in report["attempts"][0]["reason"]


@pytest.mark.parametrize("counts,target", [([100, 100], 50), ([1000, 4], 20), ([4000, 1000], 4096), ([4, 4], 100)])
def test_proportional_allocations_are_even_bounded_and_keep_components(counts, target):
    allocation = _allocations(counts, target)
    assert len(allocation) == len(counts) and all(value >= 4 and value % 2 == 0 for value in allocation)
    assert sum(allocation) <= target and all(value <= original for value, original in zip(allocation, counts))


def test_insufficient_component_budget_fails():
    with pytest.raises(ValueError, match="four faces"):
        _allocations([100, 100, 100], 8)


@pytest.mark.parametrize("key,value", [("face_budget", True), ("face_budget", 3), ("face_budget", 4096.),
                                      ("vertex_budget", -1), ("vertex_budget", 0)])
def test_invalid_budgets_fail_before_optional_import(key, value):
    with pytest.raises(ValueError, match=key):
        fit_topology_preserving_budget(None, **{key: value})


@pytest.mark.parametrize("change", ["open", "negative", "nonfinite", "euler"])
def test_invalid_source_or_changed_topology_fails(dependencies, change):
    _, produced = dependencies
    source = FakeMesh()
    if change == "open":
        source.is_watertight = False
    elif change == "negative":
        source.volume = -1
    elif change == "nonfinite":
        source.vertices[0, 0] = np.nan
    else:
        produced.extend([FakeMesh(count=100, euler=0) for _ in range(8)])
    with pytest.raises(ValueError):
        fit_topology_preserving_budget(source)
