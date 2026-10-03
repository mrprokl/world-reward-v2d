"""H99 public RGB-only contract. No renderer recipe, array or model imports."""
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import struct
import zlib


@dataclass(frozen=True)
class PublicCohort:
    base: str = "validation/factorial_rgb_v1"
    schema: str = "world-reward-factorial-rgb-v1"
    groups: int = 8
    frames: int = 3
    width: int = 1024
    height: int = 768
    prior_focal: float = 1280.

    def __post_init__(self):
        expected=("validation/factorial_rgb_v1","world-reward-factorial-rgb-v1",8,3,1024,768,1280.)
        actual=(self.base,self.schema,self.groups,self.frames,self.width,self.height,self.prior_focal)
        if any(type(a)is not type(b)or a!=b for a,b in zip(actual,expected)):
            raise ValueError("Exact preregistered public H99 contract required")

    @property
    def count(self):return self.groups*self.frames

    @property
    def fixed_K(self):
        return ((self.prior_focal,0.,self.width/2),(0.,self.prior_focal,self.height/2),(0.,0.,1.))

    def frame_name(self,index):
        if type(index)is not int or not 0<=index<self.count:raise ValueError("Original H99 frame index required")
        group,frame=divmod(index,self.frames)
        return f"group_{group:02d}_frame_{frame:03d}.png"


COHORT=PublicCohort()
BASE,SCHEMA=COHORT.base,COHORT.schema
GROUPS,FRAMES,WIDTH,HEIGHT=COHORT.groups,COHORT.frames,COHORT.width,COHORT.height
FIXED_K=COHORT.fixed_K


def identity(path):
    path=Path(path)
    if (path.resolve()!=path.absolute()or not path.is_file()or path.stat().st_mode&0o222
            or any(p.is_symlink()for p in(path,*path.parents))):
        raise ValueError("Canonical immutable public regular file required")
    digest=hashlib.sha256()
    with path.open("rb")as stream:
        for chunk in iter(lambda:stream.read(1<<20),b""):digest.update(chunk)
    return dict(sha256=digest.hexdigest(),bytes=path.stat().st_size)


def validate_png_header(path,cohort=COHORT):
    with Path(path).open("rb")as stream:header=stream.read(33)
    if (len(header)!=33 or header[:8]!=b"\x89PNG\r\n\x1a\n"or header[8:16]!=b"\x00\x00\x00\x0dIHDR"
            or struct.unpack(">I",header[29:33])[0]!=(zlib.crc32(header[12:29])&0xffffffff)):
        raise ValueError("Original PNG signature/IHDR required")
    width,height,depth,color,compression,filter_method,interlace=struct.unpack(">IIBBBBB",header[16:29])
    if (width,height,depth,color,compression,filter_method,interlace)!=(cohort.width,cohort.height,8,2,0,0,0):
        raise ValueError("Original 1024x768 eight-bit RGB PNG required")


def public_manifest(digests,cohort=COHORT):
    if (not isinstance(cohort,PublicCohort)or not isinstance(digests,list)or len(digests)!=cohort.count
            or any(not isinstance(d,str)or not re.fullmatch("[0-9a-f]{64}",d)for d in digests)):
        raise ValueError("All24 ordered public RGB hashes required")
    return dict(schema=cohort.schema,images=[dict(file=cohort.frame_name(i),sha256=d,width=cohort.width,height=cohort.height)
        for i,d in enumerate(digests)])


def public_inputs(directory,cohort=COHORT):
    """Read only the complete immutable manifest/RGB whitelist, never private data."""
    if not isinstance(cohort,PublicCohort):raise ValueError("Explicit public cohort required")
    directory=Path(directory)
    if (directory.resolve()!=directory.absolute()or not directory.is_dir()
            or any(p.is_symlink()for p in(directory,*directory.parents))):
        raise ValueError("Canonical public input directory required")
    path=directory/"manifest.json";receipt=identity(path);manifest=json.loads(path.read_text())
    if (not isinstance(manifest,dict)or set(manifest)!={"schema","images"}or manifest["schema"]!=cohort.schema
            or not isinstance(manifest["images"],list)or len(manifest["images"])!=cohort.count):
        raise ValueError("Exact24 H99 public RGB-only manifest required")
    records=[]
    for index,row in enumerate(manifest["images"]):
        if (not isinstance(row,dict)or set(row)!={"file","sha256","width","height"}or row["file"]!=cohort.frame_name(index)
                or not isinstance(row["sha256"],str)or not re.fullmatch("[0-9a-f]{64}",row["sha256"])
                or type(row["width"])is not int or type(row["height"])is not int
                or(row["width"],row["height"])!=(cohort.width,cohort.height)):
            raise ValueError("Ordered original RGB records only; private fields forbidden")
        image=directory/row["file"];actual=identity(image);validate_png_header(image,cohort)
        if actual["sha256"]!=row["sha256"]or actual["bytes"]<=33:raise ValueError("Public RGB bytes differ")
        group,frame=divmod(index,cohort.frames)
        records.append(row|dict(path=image,group_index=group,frame_index=frame))
    if {p.name for p in directory.iterdir()}!={"manifest.json",*[r["file"]for r in records]}:
        raise ValueError("Only24 original RGB files and manifest may be exposed")
    return records,receipt
