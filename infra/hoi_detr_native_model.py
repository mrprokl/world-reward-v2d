"""Neutral original HOI-DETR constructor, not a qualification or data loader.

The caller authenticates source/checkpoint/overlays/runtime, licenses and image
identity, installs nothing here, and supplies already-loaded Torch/NumPy. Its
deadline callback surrounds native work. No producer profile or globals change.
Original policy/schema/strict-load functions are reused byte-unchanged from
hoi_detr_model_qualify; no EMA swap, weight/key selection or inference fork.
"""
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
import sys

import hoi_detr_model_qualify as original


@dataclass(frozen=True,eq=False)
class OriginalHOIModel:
    model: object
    operations: object
    provenance: object  # Operation metadata only; not authenticated source proof.


def build_original_hoi(*, torch, np, runtime, deadline, source_root, checkpoint_path,
                       native_config, checkpoint_buffers):
    """Build once; return original FP32 CUDA eval model and native operations.

    Caller binds the supplied namespaces and exact source_root/checkpoint bytes
    before/after, selects the qualified import overlay BEFORE this call, and
    fixes the seed. This helper never reads a dataset, RGB, target, receipt or
    protocol profile. Use unchanged infer_hoi_detr_frame on subsequent RGBs.
    """
    runtime.check(deadline)
    original.require(torch.cuda.is_available() and torch.cuda.device_count()==1
        and torch.cuda.get_device_capability()==(9,0),'Original single H100 runtime required')
    original.require(not torch.is_autocast_enabled() and not torch.is_autocast_enabled('cpu'),
        'Caller must disable CPU/CUDA autocast before constructor')
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    from mmcv import Config
    from mmcv.parallel import collate,scatter
    from mmcv.ops import batched_nms
    from mmdet.models.builder import build_detector
    import projects.models  # Original registry side effects, not training APIs.
    from mmdet.datasets.pipelines import Compose
    from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy
    from mmdet.core.hook.ema import ExpMomentumEMAHook
    from world_reward.hoi_detr_observations import NativeHOIOperations
    original.require(not any(n=='mmdet.apis'or n.startswith('mmdet.apis.')for n in sys.modules)
        and not any(x.startswith(('/gpfs','/lus'))for x in sys.path),'No train API/foreign source imports')
    root=Path(source_root);path=root/native_config['path']
    original.require(root.is_absolute()and path.resolve().is_relative_to(root.resolve()),'Explicit native source/config root required')
    cfg=Config.fromfile(str(path),import_custom_modules=False)
    overrides=original.configuration_policy(cfg._cfg_dict,native_config)
    model=build_detector(cfg.model,test_cfg=cfg.get('test_cfg'));runtime.check(deadline)
    buffers=original.register_original_checkpoint_buffers(model,cfg,ExpMomentumEMAHook,checkpoint_buffers)
    original.require(buffers['original_state_keys']==buffers['registered_backup_keys']==898
        and buffers['full_state_keys']==1796,'Full original898+898 checkpoint schema required')
    checkpoint=torch.load(checkpoint_path,map_location='cpu',weights_only=True)
    loaded=original.strict_checkpoint(model,checkpoint,torch);del checkpoint
    original.require(loaded['keys']==1796,'All1796 original checkpoint fields required')
    model.to(device='cuda',dtype=torch.float32).eval();runtime.check(deadline)
    pipeline=[dict(x)for x in cfg.data.test.pipeline]
    original.require(pipeline and pipeline[0]==dict(type='LoadImageFromFile'),'Original RGB-only native test loader required')
    pipeline[0]=dict(type='LoadImageFromWebcam');transform=Compose(pipeline)
    def prepare(rgb):
        result=transform(dict(img=np.array(rgb[:,:,::-1],copy=True)))
        original.require(set(result)=={'img','img_metas'},'Native image-only preprocessing required')
        return result
    operations=NativeHOIOperations(prepare,collate,scatter,bbox_cxcywh_to_xyxy,batched_nms,torch,torch.device('cuda:0'))
    return OriginalHOIModel(model,operations,MappingProxyType(dict(configuration=overrides,checkpoint_buffer_schema=buffers,
        strict_checkpoint=loaded,model_loads=1,preprocessing='original_LoadImageFromWebcam_BGR_input_then_native_test_pipeline',
        decoder_queries=1500,native_postprocessing_unchanged=True,ema_swapped=False,AMP_used=False,TF32_used=False,
        source_authenticated=False,runtime_qualified=False,weights_authenticated=False,quality_verified=False,adoption=False)))
