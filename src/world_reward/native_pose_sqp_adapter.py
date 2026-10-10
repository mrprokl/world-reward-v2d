"""Exact native MHR tangent/Jacobian adapter; Torch is runtime-only.

The original complete native loss (fixed stage181, including PEN/silhouette)
and its image extension are supplied as the live optimizer instance. No loss
term is dropped/reweighted and its loop/scheduler are not run. Six VJPs per
batch16 obtain two same-ID witness Jacobians without a dense full-video matrix.
Actual source/asset identities and constructor activation equality are caller
gates; tiny NumPy tests do not establish native GPU correctness/performance.
"""
import time

import numpy as np

from .native_contact_continuation import _so3
from .native_pose_sqp import ROTATION_DIM, _BudgetExhausted
from .shared_identity import validate_native_parameters

LAYER_PIN = dict(bytes=20514,sha256='a753ab8e730b6730fca275384fab629859311983292a407390d88c66ffe68c23')
OPTIMIZER_PIN = dict(bytes=92824,sha256='84e0e818a3bc0935bb30b75fcd82fd7c5e3730ed812864594cd759697ddb406b')
OBJECTIVE_STAGE = 181


def body_gradient_to_tangent(body_controls, gradient):
    """Chain any [...,254] derivative through right-SO3/58 SO2 at zero.

    Operates on FP64 NumPy gradient copies; native autograd itself remains the
    pinned FP32 decoder. No guessed compact136 controls are differentiated.
    """
    body=np.asarray(body_controls);g=np.asarray(gradient)
    if (body.ndim!=2 or body.shape[1]!=260 or body.dtype!=np.float32 or g.shape[-2:]!=(len(body),254)
        or g.dtype!=np.float64 or not np.isfinite(body).all() or not np.isfinite(g).all()):
        raise ValueError('Complete original native body260 and FP64 derivative254 required')
    raw=body[:,:138].astype(float).reshape(len(body),23,6);rotations=_so3(raw)
    first_norm=np.linalg.norm(raw[...,:3],axis=-1)
    parallel=np.sum(rotations[..., :,0]*raw[...,3:],axis=-1)
    orthogonal=np.linalg.norm(raw[...,3:]-parallel[...,None]*rotations[..., :,0],axis=-1)
    left=g[...,:138].reshape(*g.shape[:-1],23,6)
    columns=np.stack((left[...,:3],left[...,3:]),axis=-2)
    result=np.empty((*g.shape[:-1],ROTATION_DIM),np.float64)
    basis=np.eye(3);derivatives=[]
    for axis in basis:
        # d(R exp([omega]x) e_j)/domega at0 = R(axis x e_j).
        first=rotations@np.cross(axis,basis[0]);second=rotations@np.cross(axis,basis[1])
        derivative=np.stack((first_norm[...,None]*first,
            parallel[...,None]*first+orthogonal[...,None]*second),axis=-2)
        derivatives.append(np.sum(columns*derivative,axis=(-2,-1)))
    result[...,:69]=np.stack(derivatives,axis=-1).reshape(*g.shape[:-1],69)
    sc=body[:,138:254].astype(float).reshape(len(body),58,2)
    norm=np.linalg.norm(sc,axis=-1)
    if (norm<=1e-12).any():raise ValueError('Degenerate original native SO2 controls')
    derivative=np.stack((sc[...,1],-sc[...,0]),axis=-1)
    result[...,69:]=np.sum(g[...,138:254].reshape(*g.shape[:-1],58,2)*derivative,axis=-1)
    return result


def prime_native_autograd(torch,layer,parameters):
    """Load lazy native head/caches as normal tensors BEFORE inference decode.

    Pinned MHRLayer.from_mhr_assets is lazy: first mhr_forward calls _ensure_head
    and loads persistent model buffers. An inference_mode first decode poisons
    those constants for later backward. Initialize with no_grad, NOT inference;
    no tensor-value cloning/reweighting or source/model mutation is performed.
    """
    started=time.monotonic()
    with torch.inference_mode(False),torch.no_grad(),torch.jit.optimized_execution(False):
        p={k:torch.tensor(v[:16].copy(),device='cuda',dtype=torch.float32) for k,v in parameters.items()}
        decoded=layer.mhr_forward(p)
        values=(decoded.vertices,decoded.joints,decoded.keypoints)
        if any(torch.is_inference(v) or not torch.isfinite(v).all() for v in values):
            raise ValueError('Native autograd priming must generate finite normal tensors')
        head=layer.backend.head
        if head is None:raise ValueError('Native lazy head initialization absent')
        constants=list(head.named_parameters())+list(head.named_buffers())
        if any(torch.is_inference(v) for _,v in constants):
            raise ValueError('Persistent native head inference tensors cannot enter autograd; fresh normal-mode initialization required')
    return dict(schema='world_reward.native_autograd_priming.v1',seconds=time.monotonic()-started,
        initialization_batch=min(16,len(parameters['mhr_trans'])),inference_mode_disabled=True,
        no_grad_initialization=True,persistent_native_tensors_checked=len(constants),
        persistent_inference_tensors=0,model_values_cloned_or_changed=False,
        lazy_head_initialized_before_first_inference_decode=True)


class NativeSQPCallbacks:
    """Live exact native loss + bounded-memory anatomical VJP callbacks."""
    def __init__(self,torch,layer,instance,decode_geometry,evidence,*,deadline,source_binding,chunk=16,on_completion=None):
        if chunk!=16 or source_binding.get('layer')!=LAYER_PIN or source_binding.get('optimizer')!=OPTIMIZER_PIN:
            raise ValueError('Pinned original native layer/optimizer and batch16 required')
        self.torch=torch;self.layer=layer;self.instance=instance;self.decode=decode_geometry
        self.evidence=evidence;self.deadline=deadline;self.calls=[];self.chunk=chunk
        self.on_completion=on_completion
        self.fixed_parameters={k:v.detach().cpu().numpy().copy() for k,v in instance.params_fixed.items()}
        self.fixed_parameters={k:np.frombuffer(v.tobytes(),dtype=v.dtype).reshape(v.shape) for k,v in self.fixed_parameters.items()}
        if (not np.array_equal(instance.contact_mask.detach().cpu().numpy().astype(bool),evidence.activations)
            or not np.array_equal(instance.frame_indices,evidence.frame_index)):
            raise ValueError('Original native constructor activation/timeline differs from saved frozen evidence')

    def check(self):
        if time.monotonic()>=self.deadline:raise _BudgetExhausted('Full-real native SQP/Jacobian wall budget exhausted')

    def completed(self,row):
        self.calls.append(row)
        if self.on_completion is not None:self.on_completion(row)

    def validate_fixed(self,p):
        # Keep the raw forward priors/cache FIXED. Only native body254 and
        # independent humanT/objectT are variables, exactly as the image ABI.
        for k in p.keys()-{'mhr_trans','mhr_body_pose_cont'}:
            if p[k].tobytes()!=self.fixed_parameters[k].tobytes():
                raise ValueError('Candidate changed original raw native fixed block: '+k)
        if p['mhr_body_pose_cont'][:,254:].tobytes()!=self.fixed_parameters['mhr_body_pose_cont'][:,254:].tobytes():
            raise ValueError('Candidate changed fixed raw native internal translations')

    def tensors(self,p):
        validate_native_parameters(p,len(self.evidence.frame_index),require_shared_identity=True)
        self.validate_fixed(p)
        return {k:self.torch.tensor(v.copy(),dtype=self.torch.float32,device='cuda') for k,v in p.items()}

    def _loss(self,p,obj,gradient):
        # Full leaf construction/forward/backward scope, not only enable_grad:
        # enable_grad does not override a surrounding inference-mode scope.
        with self.torch.inference_mode(False):
            return self._loss_normal(p,obj,gradient)

    def _loss_normal(self,p,obj,gradient):
        self.check();torch=self.torch;i=self.instance;t=self.tensors(p)
        body=t['mhr_body_pose_cont'][:,:254].detach().clone().requires_grad_(gradient)
        ht=t['mhr_trans'].detach().clone().requires_grad_(gradient)
        ot=torch.tensor(obj.copy(),device='cuda',dtype=torch.float32,requires_grad=gradient)
        i.body_pose=body;i.human_translation=ht;i.object_translation=ot
        indices=torch.arange(len(obj),device='cuda',dtype=torch.int64)
        with (torch.enable_grad() if gradient else torch.no_grad()),torch.jit.optimized_execution(False):
            value,parts=i.loss(indices,OBJECTIVE_STAGE,include_diagnostics=False)
        if not torch.isfinite(value):raise ValueError('Nonfinite complete native stage181 objective')
        scalar=float(value.detach().cpu());metrics={k:float(v.detach().cpu()) for k,v in parts.items()}
        required={'loss_contact','loss_penetration','loss_silhouette','loss_human_pose_prior',
            'loss_temporal','loss_object_translation_prior','loss_joint_object_RGB','loss_joint_human_RGB',
            'loss_joint_human_translation_prior','loss_joint_human_translation_temporal'}
        if not required.issubset(metrics) or not all(np.isfinite(v) for v in metrics.values()):
            raise ValueError('Complete original native physics/image loss terms required')
        if gradient:
            derivatives=torch.autograd.grad(value,(body,ht,ot),allow_unused=False)
            arrays=[v.detach().cpu().numpy().astype(float) for v in derivatives]
        else:arrays=None
        self.check();return scalar,metrics,arrays

    def objective(self,p,obj,geometry):
        start=time.monotonic();value,metrics,_=self._loss(p,obj,False)
        self.completed(dict(kind='full_native_objective',seconds=time.monotonic()-start,stage=OBJECTIVE_STAGE,
            loss=value,terms=metrics,PEN_and_silhouette_omitted=False))
        return value

    def linearize(self,p,obj,geometry,e):
        if (e is not self.evidence):raise ValueError('Same immutable contact evidence required')
        start=time.monotonic();value,parts,derivatives=self._loss(p,obj,True)
        self.completed(dict(kind='full_native_objective_gradient',seconds=time.monotonic()-start,
            stage=OBJECTIVE_STAGE,loss=value,terms=parts,full_frames=len(obj),PEN_and_silhouette_omitted=False))
        gradient=np.c_[body_gradient_to_tangent(p['mhr_body_pose_cont'],derivatives[0]),derivatives[1],derivatives[2]]
        # Diagonal proximal metric in the exact same declared tangent/trust units,
        # not a claimed native Hessian or a new objective coefficient.
        metric=np.ones_like(gradient)
        jac=np.zeros((len(obj),2,3,ROTATION_DIM),float);torch=self.torch;maximum_error=0.
        with torch.inference_mode(False),torch.enable_grad(),torch.jit.optimized_execution(False):
            for start_frame in range(0,len(obj),self.chunk):
                self.check();sl=slice(start_frame,min(start_frame+self.chunk,len(obj)));count=sl.stop-sl.start
                t={k:torch.tensor(v[sl].copy(),dtype=torch.float32,device='cuda') for k,v in p.items()}
                body=t['mhr_body_pose_cont'].detach().clone().requires_grad_(True);t['mhr_body_pose_cont']=body
                vertices=self.layer.mhr_forward_vertices(t)
                error=float(np.linalg.norm(vertices.detach().cpu().numpy().astype(float)-geometry['human_vertices'][sl].astype(float),axis=-1).max())
                if error>1e-5:raise ValueError('Jacobian actual vertices route differs from full native geometry')
                maximum_error=max(maximum_error,error)
                ids=np.maximum(e.witness_source_indices[sl],0)
                selected=vertices[torch.arange(count,device='cuda')[:,None],torch.tensor(ids,device='cuda')]
                active=torch.tensor(e.activations[sl].copy(),device='cuda')
                for side in range(2):
                    for axis in range(3):
                        self.check();v=(selected[:,side,axis]*active[:,side]).sum()
                        derivative=torch.autograd.grad(v,body,retain_graph=not(side==1 and axis==2),allow_unused=False)[0]
                        raw=derivative.detach().cpu().numpy()[:,:254].astype(float)
                        jac[sl,side,axis]=body_gradient_to_tangent(p['mhr_body_pose_cont'][sl],raw)
                self.check();del vertices,selected,body,t
        if not np.isfinite(gradient).all() or not np.isfinite(jac).all():raise ValueError('Invalid actual native objective/Jacobian gradients')
        row=dict(kind='full_native_gradient_and_same_ID_VJP',seconds=time.monotonic()-start,stage=OBJECTIVE_STAGE,
            loss=value,terms=parts,full_frames=len(obj),VJPs_per_batch=6,chunk=16,
            dense_full_video_Jacobian_built=False,vertices_route_max_error_m=maximum_error)
        self.completed(row);return dict(gradient=gradient,metric_diagonal=metric,witness_jacobian_camera=jac,
            frame_index=e.frame_index,activations=e.activations,witness_source_indices=e.witness_source_indices)
