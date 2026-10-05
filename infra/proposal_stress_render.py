"""New authored crowd/prop stress; RGB-only observation boundary, never a selector.

One unchanged MHR checkpoint load and one batched forward; original OpenCV
PyTorch3D camera/raster mechanics. No previous study, prediction or data input.
The recipe, entity inventory, geometry and all raster masks stay private.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
import mediapipe_cpu_runtime_verify as rt

ROOT = rt.ROOT
ENTRY = 'run_proposal_stress_render'
CONFIG = 'configs/proposal_stress_v1.json'
HELPERS = ('infra/proposal_stress_render.py', 'infra/run_proposal_stress_render.sh',
           'infra/camera_render.py', 'infra/mediapipe_cpu_runtime_verify.py',
           'infra/frontend_selected_assets.py', CONFIG, 'configs/frontend_asset_archive_pins.json')
DEST = Path('/srv/world-reward-data/frontend-assets-extracted-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b')
MANIFEST = DEST / 'world-reward-frontend-assets-manifest.json'
EXTRACTION = ROOT / 'results/frontend-asset-extract-75fdea08fb3b43f62f1c4b4f5e646674595cdd0b/report.json'
PUBLIC_SCHEMA = 'world_reward.rgb_proposal_inputs.v1'


def load_config(code):
    value = rt.strict((code / CONFIG).read_bytes())
    rt.require(set(value) == {'schema', 'seed', 'scenes', 'development_scenes', 'reserved_scenes',
        'frames_per_scene', 'width', 'height', 'focal_px', 'budget_seconds', 'output',
        'image_id_namespace', 'image', 'mhr_asset', 'pytorch3d_revision', 'shape_coefficients',
        'human_scales', 'object_forms', 'minimum_scene_foreground_pixels', 'minimum_visible_humans',
        'minimum_visible_human_pixels', 'resamples', 'scope'}, 'Exact frozen recipe fields required')
    expected = dict(schema='world_reward.proposal_stress_recipe.v1', seed=2026100517,
        scenes=16, development_scenes=8, reserved_scenes=8, frames_per_scene=2,
        width=640, height=480, focal_px=700., budget_seconds=300, output='validation/proposal_stress_v1',
        image_id_namespace='world_reward.proposal_stress_new_v1/',
        image='sha256:7ebfff18ba3b76dd919485c19115597d7531dfd3233f69461f1dce3f28a6c6d3',
        mhr_asset=dict(file='weights/cari4d/sam3d_body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt',
                       bytes=696110248, sha256='352e271a6c42729c68554ceaea0c955e866970160c31e35506d782dc0f7377bc'),
        pytorch3d_revision='33824be3cbc87a7dd1db0f6a9a9de9ac81b2d0ba',
        shape_coefficients=[-.3, .2, .4], human_scales=[.92, 1.04, .98],
        object_forms=['cuboid', 'cylinder', 'ellipsoid', 'duplicate'],
        minimum_scene_foreground_pixels=1024, minimum_visible_humans=2, minimum_visible_human_pixels=32,
        resamples=0, scope='synthetic_proposal_and_visible_mask_recall_cost_only')
    rt.require(value == expected and all(type(value[k]) is type(v) for k, v in expected.items()),
               'Frozen protocol differs; no runtime tuning or resampling')
    return value


def authenticate(root, code, revision):
    rt.require(root == ROOT and Path(__file__).resolve() == code / HELPERS[0], 'Actual immutable renderer required')
    source = rt.source(root, code, revision, ENTRY, HELPERS)
    config = load_config(code)
    import frontend_selected_assets as selected
    contract = selected.load_contract(root, MANIFEST)
    name = config['mhr_asset']['file']; row = contract['entries'].get(name)
    pin = {k: config['mhr_asset'][k] for k in ('bytes', 'sha256')}
    rt.require(type(row) is dict and row.get('type') == 'file'
               and all(row.get(k) == v for k, v in pin.items()), 'Independently selected MHR asset differs')
    asset = DEST / name
    rt.require(rt.identity(asset) == pin, 'Original native MHR bytes differ')
    return config, dict(source=source, config=rt.identity(code / CONFIG),
        manifest=contract['manifest_identity'], extraction=contract['extraction_receipt_identity'], model=pin)


def recipe(config):
    """Deterministic private plan, before models/QA; includes every frozen case."""
    import numpy as np
    rng = np.random.default_rng(config['seed'])
    result = []
    palette = np.array([[.22, .34, .48], [.48, .27, .20], [.25, .42, .30]])
    for scene in range(16):
        crowded = scene % 4 >= 2
        humans = []
        for person in range(3 if crowded else 2):
            x = [-.78, .78, .06][person]
            if scene % 4 == 1: x = [-.50, .42, .06][person]
            humans.append(dict(identity_coefficient=config['shape_coefficients'][person],
                scale=config['human_scales'][person], position=[x, -.04, 4.45 + .22*person],
                yaw=float(rng.uniform(-.32, .32)), motion=float(rng.uniform(-.06, .06)),
                garment=palette[(person + scene) % 3].tolist(), background_person=person == 2))
        objects = []
        positions = [[-.91, .43, 3.50], [-.18, .58, 3.65], [.57, .38, 3.55], [.97, .60, 3.85]]
        for obj in range(4):
            form = config['object_forms'][obj]
            if obj == 3:
                form = config['object_forms'][scene % 3]
                if scene % 2:
                    positions[obj] = [positions[scene % 3][0] + .17, positions[scene % 3][1] + .08,
                                      positions[scene % 3][2] + .18]
            material = palette[(obj + scene) % 3].tolist()
            if obj == 3 and scene % 2: material = palette[(scene % 3 + scene) % 3].tolist()
            objects.append(dict(form=form, position=positions[obj], scale=[.16, .20, .14],
                yaw=float(rng.uniform(-.6, .6)), material=material, motion=float(rng.uniform(-.045, .045))))
        result.append(dict(scene=scene, split='DEV' if scene < 8 else 'reserved',
            humans=humans, objects=objects, background_color=[.64+.025*(scene % 3), .66, .65]))
    return result


def frame_plan(config, plan):
    rows = []
    for scene in plan:
        for frame in range(2):
            key = config['image_id_namespace'] + str(scene['scene']) + '/' + str(frame)
            rows.append(dict(scene=scene['scene'], frame=frame, image_id=hashlib.sha256(key.encode()).hexdigest()[:32]))
    rows.sort(key=lambda r: r['image_id'])
    for index, row in enumerate(rows): row['file'] = f'image_{index:06d}.png'
    return rows


def rotation_y(angle):
    import numpy as np
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0., s], [0., 1., 0.], [-s, 0., c]])


def primitive(form):
    """Own compact closed meshes; no fetched object assets."""
    import numpy as np
    if form == 'cuboid':
        v = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
        f = np.array([[0,2,1],[0,3,2],[4,5,6],[4,6,7],[0,1,5],[0,5,4],
                      [3,7,6],[3,6,2],[0,4,7],[0,7,3],[1,2,6],[1,6,5]], np.int64)
    elif form == 'cylinder':
        angle = np.arange(24)*2*np.pi/24
        rings = [np.c_[np.cos(angle), np.full(24, y), np.sin(angle)] for y in (-1., 1.)]
        v = np.r_[rings[0], rings[1], [[0,-1,0],[0,1,0]]]; f = []
        for i in range(24):
            j = (i+1)%24; f.extend([[i,j,j+24],[i,j+24,i+24],[48,j,i],[49,i+24,j+24]])
        f = np.array(f, np.int64)
    elif form == 'ellipsoid':
        # Octahedron subdivided twice and normalized, no pole degeneracies.
        v = [[1,0,0],[-1,0,0],[0,1,0],[0,-1,0],[0,0,1],[0,0,-1]]
        f = [[0,2,4],[2,1,4],[1,3,4],[3,0,4],[2,0,5],[1,2,5],[3,1,5],[0,3,5]]
        for _ in range(2):
            cache = {}; refined = []
            def midpoint(a,b):
                key = tuple(sorted((a,b)))
                if key not in cache:
                    p = (np.array(v[a])+v[b])/2; p /= np.linalg.norm(p)
                    cache[key] = len(v); v.append(p.tolist())
                return cache[key]
            for a,b,c in f:
                ab,bc,ca = midpoint(a,b),midpoint(b,c),midpoint(c,a)
                refined.extend([[a,ab,ca],[ab,b,bc],[ca,bc,c],[ab,bc,ca]])
            f = refined
        v, f = np.array(v, float), np.array(f, np.int64)
    else: raise ValueError('Unknown authored primitive')
    return v, f


def controls(names, limits, plan):
    import numpy as np
    rt.require(len(names) == 249 and len(set(names)) == 249 and limits.shape == (249,2)
        and limits.dtype.kind=='f' and not np.isnan(limits).any()
        and np.all(limits[:,0] <= limits[:,1]), 'Native parameter ABI differs; unbounded native limits permitted')
    rt.require(np.all(limits[:204,0] <= 0) and np.all(limits[:204,1] >= 0), 'Native zero pose/scales outside limits')
    result, shapes, refs = [], [], []
    for scene in plan:
        for person, spec in enumerate(scene['humans']):
            for frame in range(2):
                row = np.zeros(204, np.float32)
                values = {'l_uparm_ry': .10+.025*person+.025*frame,
                          'r_uparm_ry': -.09-.020*person-.018*frame,
                          'l_elbow_bend': .24+.035*person+.040*frame,
                          'r_elbow_bend': .21+.025*person+.030*frame}
                for name, value in values.items():
                    rt.require(name in names[:136], 'Named native arm control missing')
                    col = names.index(name); rt.require(limits[col,0] <= value <= limits[col,1], 'Frozen arm pose outside limits')
                    row[col] = value
                shape = np.zeros(45, np.float32); shape[0] = spec['identity_coefficient']
                shape[1] = .08*(person-1)
                result.append(row); shapes.append(shape); refs.append((scene['scene'], person, frame))
    return np.stack(result), np.stack(shapes), refs


def combine_scene(scene, frame, humans, body_faces):
    import numpy as np
    vertices, faces, colors, labels, inventory = [], [], [], [], []; offset = 0
    for person, spec in enumerate(scene['humans']):
        raw = humans[(scene['scene'], person, frame)]
        neutral = humans[(scene['scene'], person, 0)]
        rt.require(np.isfinite(neutral).all() and np.ptp(neutral[:,1])>0, 'Native body height invalid')
        center = (neutral.min(0)+neutral.max(0))/2  # Authored fixed origin, reused in both frames.
        p = (raw-center)*spec['scale']; r = rotation_y(spec['yaw']+.04*frame)
        t = np.array(spec['position']); t[0] += spec['motion']*frame
        v = p@r.T+t
        c = np.tile([.68,.55,.45], (len(v),1)); band = (neutral[:,1]-neutral[:,1].min())/np.ptp(neutral[:,1])
        c[(band>.24)&(band<.80)] = spec['garment']
        entity = len(inventory); vertices.append(v); faces.append(body_faces+offset); colors.append(c)
        labels.append(np.full(len(body_faces),entity,np.int32)); offset += len(v)
        inventory.append(dict(kind='human', entity=entity, slot=person, vertices=len(v), faces=len(body_faces),
            rotation=r.tolist(), translation_m=t.tolist(), fixed_origin_m=center.tolist(), **spec))
    for obj, spec in enumerate(scene['objects']):
        raw, f = primitive(spec['form']); p = raw*np.array(spec['scale']); r = rotation_y(spec['yaw']+.10*frame)
        t = np.array(spec['position']); t[0] += spec['motion']*frame
        v = p@r.T+t; entity = len(inventory)
        vertices.append(v); faces.append(f+offset); colors.append(np.tile(spec['material'],(len(v),1)))
        labels.append(np.full(len(f),entity,np.int32)); offset += len(v)
        inventory.append(dict(kind='object', entity=entity, slot=obj, vertices=len(v), faces=len(f),
                              rotation=r.tolist(), translation_m=t.tolist(), **spec))
    v,f,c,label = np.concatenate(vertices),np.concatenate(faces),np.concatenate(colors),np.concatenate(labels)
    rt.require(np.isfinite(v).all() and np.all(v[:,2]>.05) and np.isfinite(c).all(), 'Authored geometry crosses near plane')
    return v,f,c,label,inventory


def render_rgb(torch, vertices, faces, colors, K, background):
    from camera_render import _opencv_camera
    from pytorch3d.renderer import (BlendParams, Materials, MeshRasterizer, PointLights,
        RasterizationSettings, SoftPhongShader, TexturesVertex)
    from pytorch3d.structures import Meshes
    mesh = Meshes(verts=[torch.as_tensor(vertices,device='cuda',dtype=torch.float32)],
        faces=[torch.as_tensor(faces,device='cuda',dtype=torch.int64)],
        textures=TexturesVertex(verts_features=torch.as_tensor(colors,device='cuda',dtype=torch.float32)[None]))
    camera = _opencv_camera(torch,K,640,480)
    settings = RasterizationSettings(image_size=(480,640),blur_radius=0.,faces_per_pixel=1,
        perspective_correct=True,cull_backfaces=False,clip_barycentric_coords=False,
        cull_to_frustum=False,z_clip_value=None,max_faces_per_bin=len(faces))
    fragments = MeshRasterizer(cameras=camera,raster_settings=settings)(mesh)
    lights = PointLights(device='cuda',location=((0.,-.8,0.),),ambient_color=((.35,)*3,),
                        diffuse_color=((.65,)*3,),specular_color=((0.,)*3,))
    shader = SoftPhongShader(device='cuda',cameras=camera,lights=lights,
        materials=Materials(device='cuda',specular_color=((0.,)*3,)),
        blend_params=BlendParams(background_color=tuple(background)))
    rgba = shader(fragments,mesh)
    rgb = torch.round(rgba[0,...,:3].clamp(0,1)*255).to(torch.uint8).cpu().numpy()
    return rgb,fragments.pix_to_face[0,...,0].cpu().numpy(),fragments.zbuf[0,...,0].cpu().numpy()


def visible_entities(face_index, face_labels, entities):
    import numpy as np
    rt.require(type(entities)is int and entities>0 and face_labels.ndim==1 and len(face_labels)>0
        and face_labels.dtype.kind in 'iu' and np.all((face_labels>=0)&(face_labels<entities)),
        'Complete original entity face inventory required')
    rt.require(face_index.shape == (480,640) and face_index.dtype.kind in 'iu'
               and np.all((face_index>=-1)&(face_index<len(face_labels))), 'Original raster face IDs invalid')
    label = np.full(face_index.shape,-1,np.int32); support = face_index>=0
    label[support] = face_labels[face_index[support]]
    masks = np.stack([label==i for i in range(entities)])
    return label, masks, masks.sum((1,2)).astype(np.int64)


def public_manifest(rows):
    rt.require(len(rows)==32 and len({r['image_id'] for r in rows})==32, 'All new frozen images required')
    keys = {'image_id','file','bytes','sha256','width','height'}
    rt.require(all(set(r)==keys and re.fullmatch('[0-9a-f]{32}',r['image_id'])
        and r['file']==f'image_{i:06d}.png' and type(r['bytes'])is int and r['bytes']>0
        and re.fullmatch('[0-9a-f]{64}',r['sha256']) and (r['width'],r['height'])==(640,480)
        for i,r in enumerate(rows)), 'RGB-only opaque manifest contract differs')
    return dict(schema=PUBLIC_SCHEMA, images=rows)


def persist_json(path, value):
    rt.write(path,(json.dumps(value,sort_keys=True,allow_nan=False)+'\n').encode())


def save_npz(path, **arrays):
    import numpy as np
    with path.open('xb')as stream:
        os.fchmod(stream.fileno(),0o400);np.savez_compressed(stream,**arrays)
        stream.flush();os.fsync(stream.fileno())


def installed_pytorch3d(distribution, revision):
    raw=distribution.read_text('direct_url.json')
    rt.require(type(raw)is str,'Installed PyTorch3D VCS metadata absent')
    actual=rt.strict(raw)
    expected=dict(url='https://github.com/facebookresearch/pytorch3d.git',
                  vcs_info=dict(commit_id=revision,requested_revision='v0.7.9',vcs='git'))
    rt.require(actual==expected,'Actual installed PyTorch3D commit/source differs')
    return actual


def native(root,code,revision,config,binding,started):
    import numpy as np
    rt.require(sys.platform=='linux' and os.geteuid()==0 and {p.name for p in Path('/sys/class/net').iterdir()}=={'lo'}
               and os.environ.get('WR_IMAGE_ID')==config['image'], 'Offline owned Azure image required')
    import torch
    import pytorch3d
    from importlib import metadata
    from PIL import Image
    from camera_render import PYTORCH3D_REVISION
    rt.require(torch.cuda.is_available() and torch.__version__=='2.5.1+cu124'
        and pytorch3d.__version__=='0.7.9' and PYTORCH3D_REVISION==config['pytorch3d_revision'], 'Qualified CUDA raster runtime required')
    p3d_source=installed_pytorch3d(metadata.distribution('pytorch3d'),config['pytorch3d_revision'])
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    out=root/config['output']; rt.require(out.is_dir() and not tuple(out.iterdir()), 'Exclusive empty stress output required')
    private=out/'eval_private'; private.mkdir(mode=0o700)
    public=out/'inputs'; public.mkdir(mode=0o755)
    report=dict(schema='world_reward.proposal_stress_render.v1',stage='proposal_stress_render',status='fail',phase='recipe',
        producer_revision=revision,script_sha256=binding['source']['helpers'][HELPERS[0]]['sha256'],
        source_binding=binding,images=[],scope=config['scope'],accuracy_verified=False,ownership_verified=False,
        real_fidelity_verified=False,stage_adoption=False,license_eligibility_verified=False,training_overlap_verified=False,
        resamples=0,models_loaded=0,mhr_forward_calls=0)
    def deadline(*_): raise TimeoutError('Frozen300s manufacture budget exceeded')
    previous=signal.signal(signal.SIGALRM,deadline); term=signal.signal(signal.SIGTERM,deadline)
    remaining=config['budget_seconds']-(time.monotonic()-started)
    rt.require(remaining>0,'Frozen budget exhausted during prerequisite authentication')
    signal.alarm(max(1,int(remaining)))
    try:
        plan=recipe(config); frames=frame_plan(config,plan)
        persist_json(private/'recipe.json',dict(config=config,scenes=plan,frame_mapping=frames))
        report['phase']='native_MHR'
        with torch.jit.optimized_execution(False):
            model=torch.jit.load(str(DEST/config['mhr_asset']['file']),map_location='cuda').float().eval()
        report['models_loaded']=1
        names=list(model.get_parameter_names()); joints=list(model.get_joint_names())
        limits=model.get_parameter_limits().detach().cpu().numpy()
        rt.require(len(joints)==127 and len(set(joints))==127
            and (model.get_num_identity_blendshapes(),model.get_num_face_expression_blendshapes())==(45,72), 'Native rig ABI differs')
        f=model.character_torch.mesh.faces.detach().cpu().numpy()
        rt.require(f.shape==(36874,3) and f.dtype.kind in 'iu' and np.all((f>=0)&(f<18439)), 'Native topology differs')
        params,shape,refs=controls(names,limits,plan)
        tp=torch.as_tensor(params,device='cuda'); ti=torch.as_tensor(shape,device='cuda'); te=torch.zeros(len(params),72,device='cuda')
        before=tp.clone()
        with torch.inference_mode(),torch.jit.optimized_execution(False): raw,skeleton=model(ti,tp,te,True)
        torch.cuda.synchronize(); report['mhr_forward_calls']=1
        rt.require(torch.equal(tp,before),'Native forward mutated recipe controls')
        raw,skeleton=raw.cpu().numpy(),skeleton.cpu().numpy()
        rt.require(raw.shape==(len(refs),18439,3) and skeleton.shape==(len(refs),127,8)
            and np.isfinite(raw).all() and np.isfinite(skeleton).all(),'Native full geometry invalid')
        canonical=raw@np.diag([1.,-1.,-1.])/100.
        humans=dict(zip(refs,canonical)); K=np.array([[700.,0,320.],[0,700.,240.],[0,0,1.]])
        save_npz(private/'rig.npz',faces=f,controls=params,shape=shape,skeleton_native_cm=skeleton,parameter_limits=limits)
        persist_json(private/'rig_names.json',dict(parameter_names=names,joint_names=joints,refs=refs))
        report.update(phase='rendering',torch=torch.__version__,pytorch3d=pytorch3d.__version__,
            pytorch3d_revision=PYTORCH3D_REVISION,installed_pytorch3d_direct_url=p3d_source,GPU=torch.cuda.get_device_name(),
            strict_bit_replay_claimed=False,private_metadata={name:rt.identity(private/name)
                for name in('recipe.json','rig.npz','rig_names.json')})
        rows=[]
        for record in frames:
            scene=plan[record['scene']]
            v,faces,c,labels,inventory=combine_scene(scene,record['frame'],humans,f)
            with torch.inference_mode(): rgb,face_index,depth=render_rgb(torch,v,faces,c,K,scene['background_color'])
            torch.cuda.synchronize()
            label,masks,counts=visible_entities(face_index,labels,len(inventory))
            rt.require(rgb.shape==(480,640,3) and rgb.dtype==np.uint8 and np.isfinite(depth[face_index>=0]).all()
                and np.all(depth[face_index>=0]>.05), 'Original raster RGB/depth contract failed')
            rt.require(np.count_nonzero(face_index>=0)>=config['minimum_scene_foreground_pixels']
                and sum(counts[i]>=config['minimum_visible_human_pixels'] for i,r in enumerate(inventory) if r['kind']=='human')
                >=config['minimum_visible_humans'], 'Frozen manufacture visibility gate failed; no resampling')
            truth=private/(record['image_id']+'.npz')
            save_npz(truth,vertices_camera_m=v,faces=faces,face_entity_ids=labels,camera_K=K,
                visible_entity_masks=masks,visible_entity_ids=label,visible_face_indices=face_index,
                scene_depth_m=depth,visible_pixels=counts)
            persist_json(private/(record['image_id']+'.json'),dict(**record,split=scene['split'],entities=inventory))
            path=public/record['file']
            with path.open('xb')as stream:
                os.fchmod(stream.fileno(),0o444); Image.fromarray(rgb).save(stream,format='PNG');stream.flush();os.fsync(stream.fileno())
            pin=rt.identity(path)
            rows.append(dict(image_id=record['image_id'],file=record['file'],**pin,width=640,height=480))
            report['images'].append(dict(**record,truth_identity=rt.identity(truth),
                inventory_identity=rt.identity(private/(record['image_id']+'.json')),rgb_identity=pin,
                decoded_RGB_sha256=hashlib.sha256(rgb.tobytes()).hexdigest(),visible_pixels=counts.tolist()))
        persist_json(public/'manifest.json',public_manifest(rows)); public.chmod(0o555)
        rt.require(authenticate(root,code,revision)[1]==binding,'Source/selected model changed during manufacture')
        report.update(status='pass',phase='complete',public_manifest_identity=rt.identity(public/'manifest.json'),
            source_and_asset_rehashed_after=True,full_geometry_retained=True,all_truth_private=True,
            images_rendered=32,no_QA_resampling=True,recipe_supplied_in_public_inputs=False)
    except Exception as exc:
        report['error']=dict(type=type(exc).__name__,message=str(exc)[:250]); raise
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous);signal.signal(signal.SIGTERM,term)
        report['elapsed_seconds']=time.monotonic()-started
        persist_json(private/'report.json',report)
        print(json.dumps({k:report[k]for k in('stage','status','phase','elapsed_seconds')}),flush=True)


def main():
    started=time.monotonic()
    parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true')
    parser.add_argument('--finalize',type=int);parser.add_argument('--expected-binding');args=parser.parse_args()
    root=rt.canonical(Path(os.environ['WR_ROOT']));code=rt.canonical(Path(os.environ['WR_CODE']))
    revision=os.environ['WR_CODE_REVISION'];config,binding=authenticate(root,code,revision)
    if args.verify:
        print(hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest());return
    if args.finalize is not None:
        rt.require(args.finalize==0 and re.fullmatch('[0-9a-f]{64}',str(args.expected_binding))
            and hashlib.sha256(json.dumps(binding,sort_keys=True).encode()).hexdigest()==args.expected_binding,
            'Native exit/source/asset after-check failed; no host PASS')
        out=root/config['output']; report_path=out/'eval_private/report.json'
        report_pin=rt.identity(report_path,2<<20);report=rt.strict(report_path.read_bytes())
        rt.require(report.get('status')=='pass' and report.get('phase')=='complete'
            and report.get('source_binding')==binding and report.get('images_rendered')==32
            and report.get('models_loaded')==1 and report.get('mhr_forward_calls')==1,
            'Genuine completed new renderer receipt required')
        manifest_path=out/'inputs/manifest.json';manifest=rt.strict(manifest_path.read_bytes())
        rt.require(manifest==public_manifest(manifest['images'])
            and rt.identity(manifest_path)==report['public_manifest_identity'], 'Actual public manifest differs')
        rt.require({p.name for p in(out/'inputs').iterdir()}=={'manifest.json'}|{r['file']for r in manifest['images']},
            'No private/non-RGB public files allowed')
        for row,private_row in zip(manifest['images'],report['images']):
            pin={k:row[k]for k in('bytes','sha256')}
            rt.require(rt.identity(out/'inputs'/row['file'])==pin==private_row['rgb_identity']
                and row['image_id']==private_row['image_id']
                and rt.identity(out/'eval_private'/(row['image_id']+'.npz'))==private_row['truth_identity']
                and rt.identity(out/'eval_private'/(row['image_id']+'.json'))==private_row['inventory_identity'],
                'Completed public/private outputs changed')
        metadata=report.get('private_metadata',{})
        rt.require(set(metadata)=={'recipe.json','rig.npz','rig_names.json'}
            and all(rt.identity(out/'eval_private'/name)==pin for name,pin in metadata.items()),
            'Original private recipe/rig metadata changed')
        rt.require(rt.identity(report_path,2<<20)==report_pin and authenticate(root,code,revision)[1]==binding,
            'Completed output/source final recheck failed')
        receipt=dict(schema='world_reward.proposal_stress_wrapper.v1',status='pass',producer_revision=revision,
            native_report_identity=report_pin,source_binding=binding,source_and_assets_rechecked_before_and_after=True,
            owned_container_cleanup_completed=True,accuracy_verified=False,ownership_verified=False,stage_adoption=False)
        persist_json(out/'eval_private/wrapper-report.json',receipt)
        print(json.dumps(dict(stage='proposal_stress_wrapper',status='pass',images=32,native_report_identity=report_pin)));return
    native(root,code,revision,config,binding,started)


if __name__=='__main__':main()
