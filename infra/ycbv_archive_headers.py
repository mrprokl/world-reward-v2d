"""Azure-only bounded ZIP central-directory diagnostic; never member payloads."""
import hashlib,json,os,re,signal,stat,struct,subprocess,time,urllib.parse,urllib.request
from pathlib import Path,PurePosixPath
ROOT=Path('/srv/scenesmith/world-reward');JOB='run_ycbv_archive_headers'
PINS='configs/ycbv_point_inventory_failed_pins.json';MAX=32*1024**2;SECONDS=300
REV='5c2c4aa229800355648cd268040aa814f8dc94f0'
FAILED_REV='facbf00a4ab01091629301d936c1ea38018bc0e5'
FAILED_SCRIPT='93c1f716548da6df93de8e8a9f597c6b2b13966f0c984c1292fd5b08c993ffce'
FAILURE_REPORT=dict(path='validation/ycbv_point_pose_v1/report.json',bytes=2280,sha256='5eea9045ae8038dd2eb6e20463900f7c2bf45dd5bd65014f10280b9eeecd177e')
STABLE=lambda s:(s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns,s.st_ctime_ns,s.st_nlink)
PIN_KEYS={'schema','producer_revision','script_sha256','report','status','phase','error_type','elapsed_seconds','source_rehashed_after','disposable_archives_removed','cleanup_completed','RGB_inputs_retained','private_files_retained','private_annotation_values_decoded','base_archive_layout_independently_verified','full_test_inventory_failure_cause_verified','further_acquisition_authorized'}
ARCHIVES={'ycbv_base.zip':(15805,'98440f8bd403100b21cf11a6729fabe8b3d5ce714472edc57a18b7f1fcd4bb18'),'ycbv_test_all.zip':(14969383039,'fea2ab5f18aba1857acd320827cec10d9dbf258e4940ea4b52f5dd51cb2356a7')}
PATTERN=r'test/0000(?:4[89]|5[0-9])/(?:scene_(?:camera|gt|gt_info)\.json|(?:rgb|depth)/[0-9]{6}\.png|(?:mask|mask_visib)/[0-9]{6}_[0-9]{6}\.png)'
BASE={'ycbv/camera_cmu.json','ycbv/camera_uw.json','ycbv/dataset_info.md','ycbv/test_targets_bop19.json'}
class ContractError(ValueError):
    def __init__(self,code,facts=None):super().__init__(code);self.facts=facts or {}
def need(ok,code):
    if not ok:raise ContractError(code)
def public_url(url,redirect=False):
    p=urllib.parse.urlsplit(url);h=p.hostname or ''
    need(p.scheme=='https' and not p.username and not p.password and p.port in (None,443) and not p.fragment and (h=='huggingface.co' or redirect and (h.endswith('.hf.co') or h.endswith('.huggingface.co'))) and (redirect or not p.query) and not any(k.lower()in ('token','access_token','authorization','api_key')for k,v in urllib.parse.parse_qsl(p.query)),'public_https')
class Redirect(urllib.request.HTTPRedirectHandler):
    def __init__(self,sha):self.sha=sha
    def http_error_302(self,req,fp,code,msg,headers):
        self.hops=getattr(self,'hops',0)+1;need(self.hops<=5,'redirect_bound');newurl=headers.get('Location')or headers.get('URI');need(newurl is not None,'redirect_location')
        new=self.redirect_request(req,fp,code,msg,headers,newurl);fp.close();return self.parent.open(new,timeout=req.timeout)
    http_error_301=http_error_303=http_error_307=http_error_308=http_error_302
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        public_url(newurl,True);linked=headers.get('X-Linked-ETag')
        if linked is not None:need(linked.strip('"')==self.sha,'linked_etag')
        return super().redirect_request(req,fp,code,msg,headers,newurl)
class Ranges:
    def __init__(self):self.bytes=0;self.requests=0;self.proofs=[]
    def get(self,name,start,end):
        size,sha=ARCHIVES[name];need(0<=start<=end<size and self.bytes+end-start+2<=MAX,'range_budget')
        url=f'https://huggingface.co/datasets/bop-benchmark/ycbv/resolve/{REV}/{name}';public_url(url)
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}),Redirect(sha))
        req=urllib.request.Request(url,headers={'Range':f'bytes={start}-{end}','Accept-Encoding':'identity'})
        with opener.open(req,timeout=30)as response:
            public_url(response.geturl(),True);h=response.headers;n=end-start+1
            if not (response.status==206 and h.get('Content-Range')==f'bytes {start}-{end}/{size}' and h.get('Content-Encoding','identity').lower()=='identity' and h.get('Content-Length')==str(n) and h.get('ETag','').strip('"')==sha):
                raise ContractError('range_headers',dict(status=response.status,requested_bytes=n,headers_sha256=hashlib.sha256(json.dumps({k:h.get(k)for k in ('Content-Range','Content-Length','Content-Encoding','ETag')},sort_keys=True).encode()).hexdigest()))
            data=response.read(n+1);self.bytes+=len(data);self.requests+=1;need(len(data)==n,'range_body')
        self.proofs.append(dict(archive=name,start=start,end=end,bytes=n,sha256=hashlib.sha256(data).hexdigest(),primary_etag=sha));return data

def directory(get,name):
    total,_=ARCHIVES[name];end=total-22;e=get(name,end,total-1)
    need(len(e)==22 and e[:4]==b'PK\x05\x06','eocd_no_comment_required')
    _,disk,cd_disk,n_disk,count,length,offset,comment=struct.unpack('<4s4H2IH',e)
    need(disk==cd_disk==comment==0 and n_disk==count,'single_disk_eocd')
    boundary=end
    if count==65535 or length==4294967295 or offset==4294967295:
        locator=get(name,end-20,end-1);need(locator[:4]==b'PK\x06\x07','zip64_locator')
        _,disk,where,disks=struct.unpack('<4sIQI',locator);need(disk==0 and disks==1 and 0<=where<=end-76,'zip64_locator_bounds')
        z=get(name,where,where+55);need(z[:4]==b'PK\x06\x06','zip64_eocd')
        _,record,_,_,disk,cd_disk,n_disk,count,length,offset=struct.unpack('<4sQ2H2I4Q',z)
        need(record==44 and where+56==end-20 and disk==cd_disk==0 and n_disk==count,'zip64_record');boundary=where
    if not (0<count<=1000000 and 46*count<=length<=MAX and 0<=offset and offset+length==boundary):
        raise ContractError('central_directory_bounds',dict(central_members=count,central_bytes=length,central_offset=offset,trailer_boundary=boundary,max_range_bytes=MAX,max_members=1000000))
    return get(name,offset,offset+length-1),count,offset

def central(data,count,offset):
    rows=[];pos=0
    for _ in range(count):
        need(pos+46<=len(data),'truncated_central')
        v=struct.unpack_from('<4s6H3I5H2I',data,pos);need(v[0]==b'PK\x01\x02','central_signature')
        flags,compression=v[3:5];compressed,size=v[8:10];n,x,c,disk=v[10:14];attrs,where=v[15:17];stop=pos+46+n+x+c
        need(stop<=len(data)and n>0,'central_lengths');raw=data[pos+46:pos+46+n];extra=data[pos+46+n:pos+46+n+x]
        try:name=raw.decode('utf-8'if flags&2048 else 'cp437')
        except UnicodeError:raise ContractError('filename_encoding')from None
        chunks={};i=0
        while i<len(extra):
            need(i+4<=len(extra),'extra_header');tag,length=struct.unpack_from('<HH',extra,i);i+=4
            need(i+length<=len(extra)and tag not in chunks,'extra_length_or_duplicate');chunks[tag]=extra[i:i+length];i+=length
        if 4294967295 in (size,compressed,where)or disk==65535:
            z=chunks.get(1,b'');i=0
            for key,sentinel,width in (('size',4294967295,8),('compressed',4294967295,8),('where',4294967295,8),('disk',65535,4)):
                value=locals()[key]
                if value==sentinel:
                    need(i+width<=len(z),'zip64_extra');value=int.from_bytes(z[i:i+width],'little');i+=width
                    if key=='size':size=value
                    elif key=='compressed':compressed=value
                    elif key=='where':where=value
                    else:disk=value
        need(disk==0 and where+30+compressed<=offset,'member_offset_bounds')
        rows.append(dict(filename=name,file_size=size,external_attr=attrs,flag_bits=flags,compress_type=compression,raw_filename_sha256=hashlib.sha256(raw).hexdigest()));pos=stop
    need(pos==len(data),'central_trailing_bytes');return rows

def audit(rows,name,budget):
    members={};names=set();failure=None
    def reject(guard,row=None):return dict(guard=guard,member=None if row is None else {k:row[k]for k in ('filename','file_size','external_attr','flag_bits','compress_type','raw_filename_sha256')},expanded_bytes=budget[0],member_count=budget[1])
    for row in rows:
        n=row['filename'];p=PurePosixPath(n);directory=n.endswith('/');kind=stat.S_IFMT(row['external_attr']>>16)
        if not n or '\\'in n or '\x00'in n or p.is_absolute()or str(p)!=n.rstrip('/')or any(x in ('','.','..')for x in n.rstrip('/').split('/'))or n.rstrip('/')in names or kind not in ((0,stat.S_IFDIR)if directory else(0,stat.S_IFREG))or row['flag_bits']&1 or row['compress_type']not in (0,8):return reject('Unsafe or duplicate ZIP member',row)
        names.add(n.rstrip('/'));budget[0]+=row['file_size'];budget[1]+=1
        if directory and row['file_size']or row['file_size']<0 or row['file_size']>2000000000 or budget[0]>60000000000 or budget[1]>1000000:return reject('ZIP expansion bound exceeded',row)
        if not directory:members[n]=row
    if not members:return reject('Empty ZIP or file ancestor collision')
    for n in members:
        if any(str(p)in members for p in PurePosixPath(n).parents if str(p)!='.'):return reject('Empty ZIP or file ancestor collision',members[n])
    if name=='ycbv_base.zip':
        if set(members)!=BASE:return reject('Base layout differs')
    else:
        for n in members:
            if not re.fullmatch(PATTERN,n):return reject('Full test root/layout differs',members[n])
        scenes=sorted({int(PurePosixPath(n).parts[1])for n in members})
        if scenes!=list(range(48,60)):return reject('Full native test scene set differs')
        if any(f'test/{s:06d}/scene_{k}.json'not in members for s in scenes for k in ('camera','gt','gt_info')):return reject('Full original scene metadata missing')
    return failure

def metadata(path,pin=None,readonly=True,allow_empty=False):
    need(path.is_absolute()and path.resolve()==path and not any(p.is_symlink()for p in (path,*path.parents)),'canonical_metadata');s=path.lstat()
    need(stat.S_ISREG(s.st_mode)and s.st_nlink==1 and (0 if allow_empty else 1)<=s.st_size<=200000 and (not readonly or not s.st_mode&0o222),'readonly_metadata')
    data=path.read_bytes();need(STABLE(s)==STABLE(path.lstat()),'changed_metadata');proof=dict(bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
    if pin is not None:need(proof=={k:pin[k]for k in ('bytes','sha256')},'metadata_pin')
    return data,proof

def json_value(data):
    def pairs(rows):
        d={}
        for k,v in rows:need(k not in d,'duplicate_json');d[k]=v
        return d
    return json.loads(data,object_pairs_hook=pairs,parse_constant=lambda _:(_ for _ in ()).throw(ContractError('nonfinite_json')))

def inactive(cid):
    def query(args):
        r=subprocess.run(args,stdin=subprocess.DEVNULL,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,timeout=5,env={'PATH':'/usr/bin:/bin','DOCKER_HOST':'unix://'+str(ROOT/'docker.sock')})
        need(r.returncode==0 and len(r.stdout)<=4096,'bounded_runtime_query');return r.stdout
    fields=('LoadState','ActiveState','SubState','MainPID','ExecMainStatus');text=query(['systemctl','show','--no-pager',*('--property='+k for k in fields),'world-reward-ycbv-point-acquire-technical-v2.service']);values={}
    for line in text.splitlines():
        k,sep,v=line.partition('=');need(sep and k in fields and k not in values,'unit_fields');values[k]=v
    need(values==dict(LoadState='loaded',ActiveState='failed',SubState='failed',MainPID='0',ExecMainStatus='1'),'original_failed_unit')
    need(not query(['docker','ps','-aq','--no-trunc','--filter','id='+cid]).strip(),'original_container_still_exists');return values

def bindings(code,revision):
    need(re.fullmatch('[0-9a-f]{40}',revision)and code==ROOT/'jobs'/revision/JOB/'code'and Path(__file__).resolve()==code/'infra/ycbv_archive_headers.py','actual_dispatch')
    files={str(p.relative_to(code)):metadata(p,allow_empty=True)[1]for p in sorted(code.rglob('*'))if p.is_file()}
    need(not any(p.is_symlink()or p.lstat().st_mode&0o222 or not(stat.S_ISDIR(p.lstat().st_mode)or stat.S_ISREG(p.lstat().st_mode))for p in (code,*code.rglob('*'))),'readonly_full_source')
    markers={n:metadata(code.parent/n,readonly=False)[1]for n in ('revision','source-sha256')}
    need((code.parent/'revision').read_bytes()==(revision+'\n').encode()and re.fullmatch(b'[0-9a-f]{64}\n',(code.parent/'source-sha256').read_bytes()),'dispatch_markers')
    # Historical paths below are source-hash evidence only, never imported/executed.
    pins=json_value(metadata(code/PINS)[0]);need(type(pins)is dict and set(pins)==PIN_KEYS and pins['producer_revision']==FAILED_REV and pins['script_sha256']==FAILED_SCRIPT and type(pins['elapsed_seconds'])is float and pins['elapsed_seconds']==1701.1018400439934 and pins['report']==FAILURE_REPORT and all(pins[k]is True for k in ('source_rehashed_after','disposable_archives_removed','cleanup_completed','base_archive_layout_independently_verified')) and all(pins[k]is False for k in ('private_annotation_values_decoded','full_test_inventory_failure_cause_verified','further_acquisition_authorized')) and all(type(pins[k])is int and pins[k]==0 for k in ('RGB_inputs_retained','private_files_retained')),'exact_closed_failure_pins');rev=pins['producer_revision'];old=ROOT/'jobs'/rev/'run_ycbv_point_acquire/code'
    need(pins['schema']=='world_reward.ycbv_acquisition_closed_failure.v1'and pins['status']=='fail'and pins['phase']=='inventory'and pins['error_type']=='ValueError'and pins['further_acquisition_authorized']is False and pins['base_archive_layout_independently_verified']is True,'closed_inventory_failure')
    original=ROOT/'validation'/'ycbv_point_pose_v1';need({p.name for p in original.iterdir()}=={'report.json','.container.cid'},'original_cleanup_inventory')
    cid,cid_identity=metadata(original/'.container.cid');need(re.fullmatch(b'[0-9a-f]{64}',cid),'original_cid');runtime=inactive(cid.decode())
    report,proof=metadata(ROOT/pins['report']['path'],pins['report']);report=json_value(report)
    need(report.get('status')=='fail'and report.get('phase')=='inventory'and report.get('error_type')=='ValueError'and report.get('producer_revision')==rev and report.get('script_sha256')==pins['script_sha256']and all(report.get(k)is True for k in ('source_rehashed_after','disposable_archives_removed','cleanup_completed'))and all(report.get(k)is None for k in ('selected_frames','public_manifest','retention_receipt','archive_members')),'original_failure')
    need(set(report['source_helpers'])=={'files','markers'} and set(report['source_helpers']['files'])=={'/'.join(('infra',n))for n in ('ycbv_point_acquire.py','run_ycbv_point_acquire.sh','tudl_acquire.py')}|{'/'.join(('configs','ycbv_point_protocol.json'))} and set(report['source_helpers']['markers'])=={'revision','source-sha256'},'original_source_inventory')
    historical={n:metadata(old/n,pin,allow_empty=True)[1]for n,pin in report['source_helpers']['files'].items()}
    historical_markers={n:metadata(old.parent/n,pin,False)[1]for n,pin in report['source_helpers']['markers'].items()}
    need((old.parent/'revision').read_bytes()==(rev+'\n').encode()and re.fullmatch(b'[0-9a-f]{64}\n',(old.parent/'source-sha256').read_bytes()),'original_dispatch_markers')
    need(historical['/'.join(('infra','ycbv_point_acquire.py'))]['sha256']==pins['script_sha256'],'original_script')
    protocol=json_value(metadata(old/'configs'/'ycbv_point_protocol.json')[0])
    need({k:(v['bytes'],v['sha256'])for k,v in protocol['archives'].items()}==ARCHIVES and protocol['limits']['expanded_bytes']==60000000000 and protocol['limits']['member_bytes']==2000000000 and protocol['limits']['members']==1000000 and protocol['selection']['split_prefix']=='test','original_inventory_constants')
    return dict(files=files,markers=markers,original_failure=proof,historical_sources=historical,historical_markers=historical_markers,original_cid_identity=cid_identity,original_unit=runtime)

def main():
    code=Path(os.environ['WR_CODE']);revision=os.environ['WR_CODE_REVISION']
    need(os.environ['WR_ROOT']==str(ROOT)and os.getuid()==0 and os.uname().sysname=='Linux'and os.uname().nodename=='world-reward-ncc-h100-02'and re.fullmatch('[0-9a-f]{40}',revision),'azure_host_only')
    start=time.monotonic();previous=signal.signal(signal.SIGALRM,lambda *_:(_ for _ in ()).throw(TimeoutError()));signal.alarm(SECONDS)
    out=ROOT/'results'/('ycbv-archive-header-diagnostic-'+revision);out.mkdir(mode=0o700);before=None;client=Ranges()
    report=dict(stage='ycbv_archive_central_directory_metadata_diagnostic',status='fail',phase='source_binding',producer_revision=revision,device='cpu',gpu_used=False,budget_seconds=SECONDS,max_range_bytes=MAX,whole_archive_SHA_verified=False,CRC_verified=False,acquisition_performed=False,inference_performed=False,private_annotation_values_read=False,member_payload_read=False,original_failure_unchanged=False)
    try:
        before=bindings(code,revision);report['source_bindings']=before;report['phase']='central_headers';budget=[0,0];report['archives']=[]
        for name in ARCHIVES:
            data,count,offset=directory(client.get,name);rows=central(data,count,offset);result=audit(rows,name,budget)
            need(name!='ycbv_base.zip'or result is None,'independent_base_inventory_disagrees')
            report['archives'].append(dict(archive=name,central_bytes=len(data),central_sha256=hashlib.sha256(data).hexdigest(),central_members=count,result=result))
            if result is not None:break
        report.update(status='pass',phase='complete',original_guard_reproduced=result is not None,first_rejection=result,expanded_bytes=budget[0],member_count=budget[1])
    except Exception as error:report.update(error_type=type(error).__name__,protocol_guard=str(error)if isinstance(error,ContractError)else'bounded_diagnostic_failed',protocol_facts=error.facts if isinstance(error,ContractError)else{})
    finally:
        try:report['original_failure_unchanged']=before is not None and bindings(code,revision)==before
        except Exception:report['original_failure_unchanged']=False
        if not report['original_failure_unchanged']:report['status']='fail'
        report.update(elapsed_seconds=time.monotonic()-start,range_bytes=client.bytes,range_requests=client.requests,range_proofs=client.proofs)
        report['whole_budget_respected']=report['elapsed_seconds']<=SECONDS
        if not report['whole_budget_respected']:report['status']='fail'
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        with os.fdopen(os.open(out/'report.json',os.O_WRONLY|os.O_CREAT|os.O_EXCL,0o400),'w')as f:json.dump(report,f,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        fd=os.open(out,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    if report['status']!='pass':raise SystemExit(1)
if __name__=='__main__':main()
