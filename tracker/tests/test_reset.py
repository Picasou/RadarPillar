# -*-coding:utf-8-*-
"""序列边界重置验证: reset() 清空 type_states / IMM _states; 剪枝删除死航迹类型后验。

用法: python tracker/tests/test_reset.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from types import SimpleNamespace

from tracker.updater import Updater


def build_cfg():
    imm = {'models': [{'type': 1, 'alpha': 0.85, 'beta': 0.2, 'r': 1.0},
                      {'type': 2, 'q_acc': 1.0, 'r': np.eye(4)}],
           'markov': [[0.95, 0.05], [0.05, 0.95]], 'dim': 2}
    para = SimpleNamespace(para_abf={'alpha': 0.85, 'beta': 0.2},
                           para_kf=SimpleNamespace(dim=2, q_acc=1.0, r=np.eye(4)),
                           para_ekf={}, para_imm=imm)
    flt = SimpleNamespace(type=4, para=para)
    man = SimpleNamespace(adapter={'smooth': 1, 'markov': 1, 'type_markov': {}})
    return SimpleNamespace(FILTER=flt, MANAGER=man)


def test_updater_reset():
    upd = Updater(build_cfg())
    upd.type_states[7] = np.array([0.2, 0.5, 0.3])
    upd.filter.filter._states[7] = {'probs': None, 'banks': None}
    upd.reset()
    ok = len(upd.type_states) == 0 and len(upd.filter.filter._states) == 0
    print('  %s reset() 清空 type_states + IMM _states' % ('✓' if ok else '✗'))
    return ok


def test_type_states_prune():
    upd = Updater(build_cfg())
    upd.type_states.update({1: np.array([1.0]), 2: np.array([2.0])})
    upd._prune_type_states([SimpleNamespace(id=1)])
    ok = 1 in upd.type_states and 2 not in upd.type_states
    print('  %s _prune_type_states 按存活 id 剪枝' % ('✓' if ok else '✗'))
    return ok


if __name__ == '__main__':
    results = [test_updater_reset(), test_type_states_prune()]
    print('RESULT:', 'PASS' if all(results) else 'FAIL')
    sys.exit(0 if all(results) else 1)
