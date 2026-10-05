import pathlib,json,hashlib,urllib.request,urllib.parse,re,os,time,base64,concurrent.futures
D=pathlib.Path('/srv/world-reward-data/openimages_holds_census_v1')
b=(D/'census_v2.json').read_bytes();assert len(b)==10130 and hashlib.sha256(b).hexdigest()=='b4333ea57bb408b0eafdbc5eee51d2b3ee3e6e963ef9ec269178bbd45480ff4a'
rows=json.loads(b)['public_metadata'];assert len(rows)==16 and rows[0]['ImageID']=='2333ac90234d7d50'
O=pathlib.Path('/srv/world-reward-data/openimages_holds_fresh_acquisition_v1');assert not O.exists();O.mkdir(mode=0o755)
def pin(b):return dict(bytes=len(b),sha256=hashlib.sha256(b).hexdigest())
def write(p,b):
 with p.open('xb') as f:f.write(b);f.flush();os.fsync(f.fileno());os.fchmod(f.fileno(),0o444)
class ExactRedirect(urllib.request.HTTPRedirectHandler):
 def redirect_request(self,req,fp,code,msg,headers,newurl):
  raise ValueError('Unapproved redirect')
def fetch(url,maximum):
 op=urllib.request.build_opener(urllib.request.ProxyHandler({}),ExactRedirect())
 with op.open(urllib.request.Request(url,headers={'User-Agent':'Mozilla/5.0 WorldRewardRightsAudit'}),timeout=30) as r:
  v=r.read(maximum+1)
  if r.status!=200 or r.url!=url or len(v)>maximum:raise ValueError('Exact bounded response required')
  return v
def one(row):
 iid=row['ImageID'];record=dict(image_id=iid,status='unavailable',publisher_original_metadata=row,reference_geometry_read=False,training_overlap_verified=False)
 folder=O/iid;folder.mkdir()
 try:
  u=row['OriginalLandingURL'];assert re.fullmatch(r'https://www\.flickr\.com/photos/[^/?#]+/[0-9]+/?',u)
  raw=fetch(u,2<<20);blocks=re.findall(r'<script[^>]+type=["\x27]application/ld\+json["\x27][^>]*>(.*?)</script>',raw.decode(),re.S)
  nodes=[]
  for block in blocks:
   v=json.loads(block)
   for n in (v if isinstance(v,list) else [v]):
    if isinstance(n,dict):nodes.extend([n]+n.get('@graph',[]))
  matched=[x for x in nodes if isinstance(x,dict) and x.get('@type')=='ImageObject' and x.get('acquireLicensePage')==u]
  assert len(matched)==1
  obj=matched[0];assert obj['license']=='https://creativecommons.org/licenses/by/2.0/' and obj['author']['name']==row['Author']
  rights=dict(schema='world_reward.external_image_rights.v3',image_id=iid,census_pin=pin(b),creator_ld_json=obj,creator_page=dict(url=u,**pin(raw)),license='CC-BY-2.0',annotations_license='CC-BY-4.0',individual_creator_declaration_verified=True,attribution=row['Author']+' — '+row['Title']+' — '+u)
  rb=(json.dumps(rights,sort_keys=True)+'\n').encode();write(folder/'rights.json',rb)
  record['rights_pin']=pin(rb);record['creator_grant_verified']=True
  # Unknown publisher orientation is not guessed from pixels or reference boxes.
  if row['Rotation']!='0.0':
   record.update(status='unscorable_rotation',reason='publisher_rotation_unknown_or_nonzero_no_RGB_acquired')
  else:
   u=row['OriginalURL'];assert re.fullmatch(r'https://(?:farm[0-9]+|c[0-9]+|live)\.staticflickr\.com/[^?]+\.jpg',u)
   n=int(row['OriginalSize']);assert 0<n<=16<<20
   rgb=fetch(u,n);assert len(rgb)==n and base64.b64encode(hashlib.md5(rgb).digest()).decode()==row['OriginalMD5']
   write(folder/'rgb.jpg',rgb)
   record.update(status='acquired',image_pin=pin(rgb),publisher_md5_matched=True,rotation='0.0',image_transformation='none_original_file')
 except Exception as e:
  record.update(error_type=type(e).__name__,reason='exact_creator_grant_or_original_acquisition_not_qualified_no_replacement')
 rb=(json.dumps(record,sort_keys=True)+'\n').encode();write(folder/'record.json',rb);folder.chmod(0o555)
 return record
t=time.monotonic()
with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool: results=list(pool.map(one,rows[1:]))
v=dict(schema='world_reward.openimages_fresh_acquisition.v1',census_pin=pin(b),selection='fixed_original_16_metadata_cohort_excluding_one_runtime_QA_image_no_replacements',excluded_QA_image='2333ac90234d7d50',records=results,reference_geometry_read=False,challenge_inputs_used=False,quality_verified=False,training_overlap_verified=False,local_heavy_transfer=False,elapsed_seconds=time.monotonic()-t)
raw=(json.dumps(v,sort_keys=True)+'\n').encode();write(O/'manifest.json',raw);O.chmod(0o555)
print(json.dumps(dict(manifest=pin(raw),slots=len(results),status_counts={s:sum(r['status']==s for r in results) for s in sorted({r['status'] for r in results})},RGB_bytes=sum(r.get('image_pin',{}).get('bytes',0) for r in results),elapsed_seconds=v['elapsed_seconds'],reference_geometry_read=False,local_heavy_transfer=False)))
