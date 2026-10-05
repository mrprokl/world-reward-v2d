"""Bounded JSON syntax streams; skipped values are never semantically decoded.

Arrays yield one raw row. With ``id_key`` its top-level ID is decoded first,
regardless of field order; callers decide whether to decode the remaining row.
Object streams emit only requested arrays, lexically skipping all other values.
Every byte is UTF-8/syntax checked, including skipped values. Escaped surrogate
and finite-number checks additionally apply to semantically consulted values.
"""
import codecs
import json
import math
import re

_SPACE = re.compile(rb'[ \t\r\n]*')
_STRING_END = re.compile(rb'["\\\x00-\x1f]')
_NUMBER = re.compile(rb'-?(?:0|[1-9][0-9]*)(?:\.[0-9]+)?(?:[eE][+-]?[0-9]+)?\Z')


def strict_decode(raw):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result: raise ValueError('Duplicate JSON key')
            result[key] = value
        return result
    def number(value):
        result = float(value)
        if not math.isfinite(result): raise ValueError('Nonfinite consulted JSON number')
        return result
    value = json.loads(raw.decode('utf-8'), object_pairs_hook=pairs,
                       parse_float=number, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('Nonfinite JSON literal')))
    def strings(item):
        if isinstance(item, str): item.encode('utf-8', errors='strict')
        elif isinstance(item, list):
            for child in item: strings(child)
        elif isinstance(item, dict):
            for key, child in item.items(): strings(key); strings(child)
    strings(value)
    return value


class _Stream:
    def __init__(self, source, row_bytes, max_bytes, check, chunk_bytes):
        if any(type(v) is not int or v <= 0 for v in (row_bytes,max_bytes,chunk_bytes)):
            raise ValueError('Positive integer JSON byte limits required')
        self.source, self.row_bytes, self.maximum = source, row_bytes, max_bytes
        self.check, self.chunk_bytes = check, min(chunk_bytes,max_bytes)
        self.data, self.pos, self.total, self.eof = b'', 0, 0, False
        self.decoder = codecs.getincrementaldecoder('utf-8')('strict')
        self.captures = []

    def peek(self):
        if self.pos == len(self.data) and not self.eof:
            self.check(); self.data = self.source.read(self.chunk_bytes); self.pos = 0
            if type(self.data) is not bytes: raise ValueError('Binary JSON stream required')
            self.eof = not self.data
            self.decoder.decode(self.data, final=self.eof)
        return self.data[self.pos] if self.pos < len(self.data) else -1

    def take(self, count=1):
        raw = self.data[self.pos:self.pos+count]
        if len(raw) != count: raise ValueError('Unexpected JSON EOF')
        self.pos += count; self.total += count
        if self.total > self.maximum: raise ValueError('JSON stream byte limit exceeded')
        for buffer, limit in self.captures:
            if len(buffer)+count > limit: raise ValueError('JSON row/token byte limit exceeded')
            buffer.extend(raw)
        return raw

    def expect(self, byte):
        if self.peek() != byte: raise ValueError('Unexpected JSON syntax')
        self.take()

    def space(self):
        while self.peek() in (32, 9, 10, 13):
            end = _SPACE.match(self.data, self.pos).end()
            self.take(end-self.pos)

    def string(self, decode=False):
        buffer = bytearray()
        if decode: self.captures.append((buffer, 65536))
        try:
            self.expect(34)
            while True:
                if self.peek() == -1: raise ValueError('Unterminated JSON string')
                found = _STRING_END.search(self.data, self.pos)
                end = found.start() if found else len(self.data)
                if end > self.pos: self.take(end-self.pos)
                if not found: continue
                byte = self.peek(); self.take()
                if byte == 34: break
                if byte != 92: raise ValueError('JSON string control byte')
                escape = self.peek(); self.take()
                if escape == 117:
                    for _ in range(4):
                        if self.peek() not in b'0123456789abcdefABCDEF': raise ValueError('Invalid JSON Unicode escape')
                        self.take()
                elif escape not in b'"\\/bfnrt': raise ValueError('Invalid JSON escape')
            return strict_decode(bytes(buffer)) if decode else None
        finally:
            if decode: self.captures.pop()

    def value(self, depth=0, id_key=None):
        if depth > 128: raise ValueError('JSON nesting limit exceeded')
        self.space(); byte = self.peek(); identifier = None
        if byte == 34: self.string()
        elif byte in (123, 91):
            object_mode = byte == 123; close = 125 if object_mode else 93
            self.take(); self.space(); seen = set()
            if self.peek() == close: self.take(); return identifier
            while True:
                key = None
                if object_mode:
                    key = self.string(decode=True)
                    if key in seen: raise ValueError('Duplicate JSON key')
                    seen.add(key); self.space(); self.expect(58)
                if object_mode and key == id_key:
                    raw = bytearray(); self.captures.append((raw, self.row_bytes))
                    try: self.value(depth+1)
                    finally: self.captures.pop()
                    identifier = strict_decode(bytes(raw))
                else: self.value(depth+1)
                self.space()
                if self.peek() == close: self.take(); break
                self.expect(44); self.space()
        elif byte in (116, 102, 110):
            literal = {116:b'true', 102:b'false', 110:b'null'}[byte]
            for expected in literal: self.expect(expected)
        elif byte in b'-0123456789':
            raw = bytearray()
            while self.peek() in b'-+0123456789.eE':
                raw.extend(self.take())
                if len(raw) > 256: raise ValueError('JSON number token limit exceeded')
            if not _NUMBER.fullmatch(raw): raise ValueError('Invalid JSON number')
        else: raise ValueError('Invalid JSON value')
        return identifier

    def rows(self, id_key):
        self.space(); self.expect(91); self.space()
        if self.peek() == 93: self.take(); return
        while True:
            self.space()
            if self.peek() != 123: raise ValueError('Metadata array rows must be objects')
            raw = bytearray(); self.captures.append((raw, self.row_bytes))
            try: identifier = self.value(id_key=id_key)
            finally: self.captures.pop()
            yield (identifier, bytes(raw)) if id_key is not None else bytes(raw)
            self.space()
            if self.peek() == 93: self.take(); break
            self.expect(44)

    def finish(self):
        self.space()
        if self.peek() != -1: raise ValueError('Trailing JSON data')
        self.check()


def iter_array(source, *, id_key=None, row_bytes=16 << 20, max_bytes=1 << 30,
               check=lambda: None, chunk_bytes=65536):
    stream = _Stream(source, row_bytes, max_bytes, check, chunk_bytes)
    yield from stream.rows(id_key); stream.finish()


def iter_object_arrays(source, keys, *, row_bytes=16 << 20, max_bytes=1 << 30,
                       check=lambda: None, chunk_bytes=65536):
    """Yield (root key, raw object row); unrequested arrays never json.loads."""
    stream = _Stream(source, row_bytes, max_bytes, check, chunk_bytes)
    stream.space(); stream.expect(123); stream.space(); seen = set()
    if stream.peek() == 125:
        stream.take(); stream.finish()
        if keys: raise ValueError('Required JSON root arrays absent')
        return
    while True:
        key = stream.string(decode=True)
        if key in seen: raise ValueError('Duplicate JSON root key')
        seen.add(key); stream.space(); stream.expect(58)
        if key in keys:
            for raw in stream.rows(None): yield key, raw
        else: stream.value()
        stream.space()
        if stream.peek() == 125: stream.take(); break
        stream.expect(44); stream.space()
    stream.finish()
    if not set(keys) <= seen: raise ValueError('Required JSON root arrays absent')
