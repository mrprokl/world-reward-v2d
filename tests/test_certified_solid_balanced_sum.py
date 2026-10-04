"""Exact rational/source checks, not native CGAL qualification or HOI evidence."""
from fractions import Fraction
import hashlib
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'infra/certified_solid_query.cpp'
ORIGINAL_BYTES=13914
ORIGINAL_SHA256='72098be329146be0c48f32bf1473731c195120fb53b65511955d69e222c59ee4'
DETERMINANT=('a.x()*(b.y()*c.z()-b.z()*c.y())\n'
             '                           -a.y()*(b.x()*c.z()-b.z()*c.x())\n'
             '                           +a.z()*(b.x()*c.y()-b.y()*c.x())')


def restore_original(text):
    first=text.index('// Exact binary carries bound lazy addition depth')
    last=text.index('std::vector<Component> validate(',first)
    text=text[:first]+text[last:]
    text=text.replace('  std::vector<BalancedVolumeSum> volume_sums(input.components);\n','')
    text=text.replace('volume_sums[face.component].add('+DETERMINANT+');',
                      'volume6[face.component]+='+DETERMINANT+';')
    text=text.replace('  for(std::size_t i=0;i<input.components;++i) volume6[i]=volume_sums[i].sum();\n','')
    return text.encode()


def test_only_accumulation_changed_against_frozen_original_source():
    raw=restore_original(SOURCE.read_text())
    assert len(raw)==ORIGINAL_BYTES
    assert hashlib.sha256(raw).hexdigest()==ORIGINAL_SHA256


def test_native_ft_terms_origin_predicates_and_diagnostic_unchanged():
    text=SOURCE.read_text()
    assert 'std::vector<Kernel::FT> volume6(input.components,Kernel::FT(0));'in text
    assert 'volume_sums[face.component].add('+DETERMINANT+');'in text
    assert 'volume6[i]=volume_sums[i].sum();'in text
    helper=text[text.index('struct BalancedVolumeSum'):text.index('std::vector<Component> validate(')]
    assert 'carry=levels[level]+carry;'in helper
    assert 'levels[level]=Kernel::FT(0); occupied[level]=false;'in helper
    assert 'result=any ? levels[level]+result : levels[level];'in helper
    assert all(token not in helper for token in ('CGAL::exact','CGAL::to_double','double ','epsilon','sign(','sort('))
    assert text.index('CGAL::sign(volume6[i])')<text.index('CGAL::to_double(volume6[i]/Kernel::FT(6))')
    assert 'MAX_VERTICES = 1000000, MAX_FACES = 2000000, MAX_COMPONENTS = 256;'in text


def balanced(terms):
    """Independent rational/tree-depth oracle mirroring the declared carry order."""
    levels=[];occupied=[];seen=[]
    for term in terms:
        carry=(Fraction(term),0,1);level=0
        while level<len(levels)and occupied[level]:
            previous=levels[level]
            carry=(previous[0]+carry[0],max(previous[1],carry[1])+1,previous[2]+carry[2])
            levels[level]=None;occupied[level]=False;level+=1
        if level==len(levels):levels.append(carry);occupied.append(True)
        else:levels[level]=carry;occupied[level]=True
        seen.append(tuple((i,x[2])for i,x in enumerate(levels)if occupied[i]))
    result=(Fraction(0),0,0)
    for level in range(len(levels)):
        if occupied[level]:
            left=levels[level]
            result=(left[0]+result[0],max(left[1],result[1])+1,left[2]+result[2])if result[2]else left
    return result,seen


@pytest.mark.parametrize('count',[0,1,2,3,4,7,8,15,16,17,127,128,129,1023,1024,1025,8193])
def test_exact_dyadic_cancellation_and_logarithmic_tree(count):
    terms=[Fraction((-1)**i*((i*37)%23+1),2**((i*7)%41))for i in range(count)]
    result,steps=balanced(terms)
    assert result[0]==sum(terms,Fraction(0))and result[2]==count
    assert result[1]<=max(0,count.bit_length())
    for n,slots in enumerate(steps,1):
        assert sum(size for _,size in slots)==n
        assert all(size==2**level for level,size in slots)
        assert len(slots)==n.bit_count()


def test_reversed_orientation_and_term_permutations_preserve_exact_sign():
    values=[Fraction(2**35),Fraction(-2**35),Fraction(1,2**31),Fraction(-7,2**31),Fraction(9,2**31)]
    expected=sum(values,Fraction(0))
    for row in (values,list(reversed(values)),values[2:]+values[:2]):
        assert balanced(row)[0][0]==expected>0
        assert balanced([-x for x in row])[0][0]==-expected<0
    assert balanced([Fraction(1),Fraction(-1)]*64)[0][0]==0


def test_interleaved_components_keep_independent_terms_without_drop():
    labelled=[(i%3,Fraction((-1)**i*(i+1),2**(i%11)))for i in range(53)]
    groups=[[term for label,term in labelled if label==j]for j in range(3)]
    assert sum(len(row)for row in groups)==len(labelled)
    assert [balanced(row)[0][0]for row in groups]==[sum(row,Fraction(0))for row in groups]


def test_declaration_requires_native_qualification_and_preserves_failure_scope():
    text=(ROOT/'docs/balanced_solid_sum.md').read_text()
    for literal in ('15','four','O(F)','to_double','EP25','unqualified','No production replay'):
        assert literal in text
