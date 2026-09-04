"""rw_struct 单测: decode 截断防护 + encode/struct_write/struct_read roundtrip."""
import os
import sys
from ctypes import sizeof

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracker.utils.rw_struct import Raw_Trk, Raw_TrkHead, struct_write, struct_read


def test_decode_short_buffer_raises():
    r = Raw_Trk()
    with pytest.raises(ValueError):
        r.decode(b'x' * (sizeof(Raw_Trk) - 1))          # 短 buffer 直调 → 越界读拦截


def test_decode_full_buffer_ok():
    r = Raw_Trk()
    payload = b'x' * sizeof(Raw_Trk)
    assert r.decode(payload) == sizeof(Raw_Trk)


def test_struct_roundtrip(tmp_path):
    r = Raw_Trk()
    r.id, r.x_m, r.heading_deg = 7, 1234, -9000
    h = Raw_TrkHead()
    h.version, h.frame_cnt, h.trk_num, h.reserved = 1, 42, 1, 0
    f = str(tmp_path / '0201.bin')
    fh = str(tmp_path / '0200.bin')
    struct_write(f, [r], heads=[h], head_filepath=fh)
    recs = struct_read(f, Raw_Trk)
    heads = struct_read(fh, Raw_TrkHead)
    assert len(recs) == 1 and len(heads) == 1
    assert recs[0].id == 7 and recs[0].x_m == 1234 and recs[0].heading_deg == -9000
    assert heads[0].frame_cnt == 42
