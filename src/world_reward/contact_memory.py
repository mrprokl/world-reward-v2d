"""Automatic evolving anatomical patches with bounded temporal associations.

Inspired by the soft correspondence/temporal-memory idea in Open-CHOIR
(arXiv:2605.20992v4), not a reproduction or a learned contact model. Native
activations stay frozen. Association proposals use referenced object vertices,
not exact surface distance or signed penetration. Normal parallelness is
orientation-agnostic: unknown outward orientation cannot establish opposition.
The existing pose fitter supplies exact continuous-surface factors afterwards.
"""
from dataclasses import dataclass
import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ContactMemoryConfig:
    distance_sigma_diameter: float
    transition_sigma_diameter: float
    normal_sigma_rad: float
    proposal_count: int
    patch_count: int
    development_reference: str

    def __post_init__(self):
        if (any(type(v) not in (int,float) or not np.isfinite(v) or v<=0 for v in
                (self.distance_sigma_diameter,self.transition_sigma_diameter,self.normal_sigma_rad))
                or type(self.proposal_count) is not int or not 1<=self.proposal_count<=8
                or type(self.patch_count) is not int or not 1<=self.patch_count<=8
                or type(self.development_reference) is not str or not self.development_reference.strip()):
            raise ValueError('Externally frozen positive association scales and <=8 proposals/patch points required')


def immutable(a):
    a=np.ascontiguousarray(a)
    return np.frombuffer(a.tobytes(),dtype=a.dtype).reshape(a.shape)


def vertex_normals(vertices,faces):
    """Area-weighted triangle normals. Zero is unknown, not a guessed normal."""
    v,f=map(np.asarray,(vertices,faces))
    if (v.dtype.kind!='f' or v.ndim!=2 or v.shape[1:]!=(3,) or not len(v)
            or not np.isfinite(v).all() or f.dtype.kind not in 'iu' or f.ndim!=2
            or f.shape[1:]!=(3,) or not len(f) or np.any(f<0) or np.any(f>=len(v))):
        raise ValueError('Exact finite vertices and their original triangle topology required')
    tri=v[f];normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
    accumulated=np.zeros_like(v,dtype=np.float64)
    for column in range(3):np.add.at(accumulated,f[:,column],normal)
    length=np.linalg.norm(accumulated,axis=1)
    if not np.isfinite(length).all():raise ValueError('Finite normal arithmetic required')
    valid=length>0
    normals=np.zeros_like(accumulated);normals[valid]=accumulated[valid]/length[valid,None]
    return normals,valid


def temporal_path(candidate_ids,candidate_points,unary,transition_sigma_m,previous_points=None):
    """Globally optimal bounded chain, deterministic ties; no motion freezing.

    Transition compares two anatomical candidates at the SAME current frame.
    Persistent ID costs zero, but its observed trajectory can move or slide.
    A switch is possible when geometric evidence outweighs anatomical separation.
    Each frame may have a different candidate set. Input rows remain untouched.
    """
    if (not candidate_ids or len(candidate_ids)!=len(candidate_points) or len(unary)!=len(candidate_ids)
            or type(transition_sigma_m) not in (int,float)
            or not np.isfinite(transition_sigma_m) or transition_sigma_m<=0):
        raise ValueError('A nonempty bounded chain and positive transition scale required')
    if previous_points is not None and len(previous_points)!=len(candidate_ids):
        raise ValueError('Previous-state current-frame coordinates must follow the full chain')
    back=[];cost=None;previous=None
    for frame,(ids,positions,scores) in enumerate(zip(candidate_ids,candidate_points,unary)):
        ids,positions,scores=map(np.asarray,(ids,positions,scores))
        if (ids.dtype.kind not in 'iu' or ids.ndim!=1 or not 1<=len(ids)<=16
                or len(np.unique(ids))!=len(ids) or np.any(ids<0)
                or positions.dtype.kind!='f' or positions.shape!=(len(ids),3)
                or not np.isfinite(positions).all() or scores.dtype.kind!='f'
                or scores.shape!=ids.shape or not np.isfinite(scores).all() or np.any(scores<0)):
            raise ValueError('Finite unique candidate states, positions and nonnegative costs required')
        if cost is None:
            cost=scores.copy();back.append(None)
        else:
            # Caller includes the previous candidate positions expressed at the
            # current frame; these are keyed coordinates, never previous poses.
            if previous_points is None:
                missing=[int(value) for value in previous if value not in ids]
                if missing:raise ValueError('Chain requires previous states at current coordinates')
                mapping={int(value):i for i,value in enumerate(ids)}
                prior_positions=positions[[mapping[int(value)] for value in previous]]
            else:
                prior_positions=np.asarray(previous_points[frame])
                if (prior_positions.dtype.kind!='f' or prior_positions.shape!=(len(previous),3)
                        or not np.isfinite(prior_positions).all()):
                    raise ValueError('Finite previous anatomical states expressed at current frame required')
            transition=np.sum(((prior_positions[:,None]-positions[None])/transition_sigma_m)**2,axis=-1)
            combined=cost[:,None]+transition
            ancestor=np.argmin(combined,axis=0)
            cost=scores+combined[ancestor,np.arange(len(ids))]
            if not np.isfinite(cost).all():raise ValueError('Finite association objective required')
            back.append(ancestor)
        previous=ids
    state=int(np.argmin(cost));path=np.empty(len(back),np.int64)
    for frame in range(len(back)-1,-1,-1):
        path[frame]=candidate_ids[frame][state]
        if frame:state=int(back[frame][state])
    return immutable(path)


def _adjacency(faces,count):
    adjacency=[set() for _ in range(count)]
    for a,b,c in faces:
        adjacency[a].update((int(b),int(c)));adjacency[b].update((int(a),int(c)));adjacency[c].update((int(a),int(b)))
    return adjacency


def _topological_patch(adjacency,source_ids,count,centre):
    selected=[];pending=[int(centre)];seen=set(pending)
    while pending and len(selected)<count:
        selected.extend(sorted(pending,key=lambda index:int(source_ids[index])))
        following=set()
        for index in pending:following.update(adjacency[index])
        pending=list(following-seen);seen.update(pending)
    if len(selected)<count:raise ValueError('Selected anatomical component cannot cover the bounded patch')
    return np.asarray(selected[:count],np.int64)


def evolving_contact_patch(hands,source_indices,activations,proposal_rotations,proposal_translations,
                           object_vertices,object_faces,hand_faces,config):
    """Seal one changing patch from automatic geometry before one full-T fit.

    Shapes are hands[T,2,J,3], source IDs[2,J], native activations[T,2]. Geometry
    is predicted full support, NOT optical visibility. Approximate proposals are
    current referenced-vertex distance/normal plus temporal anatomical memory.
    No synthetic samples, changed activation, shape, gauge or target labels.
    """
    inputs=(hands,source_indices,activations,proposal_rotations,proposal_translations,object_vertices,object_faces)
    if any(np.ma.isMaskedArray(value) for value in inputs):raise ValueError('Unmasked original geometry required')
    h,ids,active,r,t,v,f=map(np.asarray,inputs)
    if (type(config) is not ContactMemoryConfig or h.dtype.kind!='f' or h.ndim!=4
            or h.shape[1]!=2 or h.shape[-1]!=3 or h.shape[0]<1 or h.shape[2]<1 or not np.isfinite(h).all()
            or ids.dtype.kind not in 'iu' or ids.shape!=h.shape[1:3] or np.any(ids<0)
            or np.max(ids)>np.iinfo(np.int64).max
            or any(len(np.unique(row))!=len(row) for row in ids)
            or active.dtype!=np.bool_ or active.shape!=h.shape[:2]
            or r.dtype.kind!='f' or r.shape!=(len(h),3,3) or not np.isfinite(r).all()
            or t.dtype.kind!='f' or t.shape!=(len(h),3) or not np.isfinite(t).all()
            or len(hand_faces)!=2):raise ValueError('Exact full-T finite predicted anatomy, native activation and rigid proposals required')
    if not np.allclose(r@r.swapaxes(-1,-2),np.eye(3),atol=1e-5,rtol=0) or not np.allclose(np.linalg.det(r),1,atol=1e-5,rtol=0):
        raise ValueError('Proper original proposal rotations required')
    object_normals,object_normal_valid=vertex_normals(v,f)
    referenced=np.unique(f);tree=cKDTree(v[referenced]);diameter=2*np.sqrt(np.mean(np.sum((v-v.mean(0))**2,axis=1)))
    if not np.isfinite(diameter) or diameter<=0:raise ValueError('Nondegenerate unchanged object geometry required')
    p=min(config.patch_count,h.shape[2]);k=min(config.proposal_count,h.shape[2])
    selected=np.full((len(h),2,p),-1,np.int64);points=np.full((len(h),2,p,3),np.nan)
    supported=np.zeros((len(h),2,p),bool);winners=np.full((len(h),2),-1,np.int64)
    gaps=np.full((len(h),2),np.nan);normal_score=np.full((len(h),2),np.nan)
    normal_supported=np.zeros((len(h),2),bool);patch_cache={};candidate_states=0
    for side in range(2):
        faces=np.asarray(hand_faces[side]);vertex_normals(h[0,side],faces)  # Validate native local topology even when inactive.
        adjacency=_adjacency(faces,len(ids[side]))
        boundaries=np.diff(np.r_[False,active[:,side],False].astype(np.int8))
        for begin,end in zip(np.flatnonzero(boundaries==1),np.flatnonzero(boundaries==-1)):
            candidate_ids=[];candidate_points=[];unaries=[];previous_points=[];previous=None;previous_states=None
            distance_rows=[];angle_rows=[];normal_rows=[]
            for frame in range(begin,end):
                local=(h[frame,side]-t[frame])@r[frame]
                distance,nearest=tree.query(local,k=1,eps=0);nearest=referenced[nearest]
                hand_normals,hand_valid=vertex_normals(h[frame,side],faces)
                hand_normals=hand_normals@r[frame]
                valid=hand_valid & object_normal_valid[nearest]
                angle=np.zeros(len(local));angle[valid]=np.arccos(np.clip(np.abs(np.sum(hand_normals[valid]*object_normals[nearest[valid]],axis=1)),0,1))
                distance_rows.append(distance);angle_rows.append(angle);normal_rows.append(valid)
                cost=(distance/(diameter*config.distance_sigma_diameter))**2+(angle/config.normal_sigma_rad)**2
                proposed=np.lexsort((ids[side],cost))[:k]
                # Keep current K and previous K candidates, not all past states.
                current=np.unique(np.r_[proposed,previous if previous is not None else np.empty(0,np.int64)])
                current=current[np.argsort(ids[side,current],kind='stable')]
                candidate_ids.append(ids[side,current].astype(np.int64));candidate_points.append(h[frame,side,current])
                unaries.append(cost[current])
                previous_points.append(None if previous_states is None else h[frame,side,previous_states])
                previous=proposed;previous_states=current;candidate_states+=len(current)
            lookup={int(value):j for j,value in enumerate(ids[side])}
            path=temporal_path(candidate_ids,candidate_points,unaries,
                float(diameter*config.transition_sigma_diameter),previous_points)
            for offset,winner in enumerate(path):
                frame=begin+offset;index=lookup[int(winner)]
                cache_key=(side,index)
                if cache_key not in patch_cache:patch_cache[cache_key]=_topological_patch(adjacency,ids[side],p,index)
                pool=patch_cache[cache_key];selected[frame,side]=ids[side,pool]
                points[frame,side]=h[frame,side,pool];supported[frame,side]=True;winners[frame,side]=winner
                gaps[frame,side]=distance_rows[offset][index]
                normal_supported[frame,side]=normal_rows[offset][index]
                if normal_supported[frame,side]:normal_score[frame,side]=np.cos(angle_rows[offset][index])
    return dict(selected_ids=immutable(selected),hand_points_camera=immutable(points),geometry_supported=immutable(supported),
        winner_ids=immutable(winners),proposal_vertex_gap_m=immutable(gaps),normal_parallelness=immutable(normal_score),
        normal_supported=immutable(normal_supported),native_activations=immutable(active),
        candidate_states=int(candidate_states),orientation_verified=False,exact_association_surface_distance=False,
        physical_contact_verified=False,material_trajectory_frozen=False)
