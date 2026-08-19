"""matcher 测试 - 每个函数直接单测 (手算 oracle) + KM 确定性例 + run 集成."""
import os
import sys
import types
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from tracker.schemas import Trk, Obj, TrkHistory, Matches, Cfg
from tracker.matcher import Matcher


# ============================================================
# -------------------- fixture 工厂 --------------------
# ============================================================
def mk_trk(x=0.0, y=0.0, doppler=0.0, cov=None, id=1, length=4, width=2, heading_deg=0):
    return Trk(
        x_m=x, y_m=y, z_m=0, vx_mps=0, vy_mps=0, doppler_mps=doppler,
        ax_mps2=0, ay_mps2=0, heading_deg=heading_deg, yaw_rate_degs=0,
        id=id, width_m=width, height_m=1, length_m=length, lifetime_s=0,
        x_std_m=0, y_std_m=0, z_std_m=0, vx_std_mps=0, vy_std_mps=0,
        ax_std_mps2=0, ay_std_mps2=0, xy_pos_cov=0, xy_vel_cov=0, xy_acc_cov=0,
        width_std_m=0, height_std_m=0, length_std_m=0, heading_std_deg=0, yaw_rate_std_degs=0,
        type=0, type_confi=0, obstacle_prob=0, existence_prob=0,
        motion_status=0, measurement_status=0, passable_status=0, rel_vel=0, rel_acc=0,
        cov=np.eye(4) if cov is None else cov, history=TrkHistory(),
    )


def mk_obj(x=0.0, y=0.0, doppler=0.0, id=1):
    return Obj(x=x, y=y, doppler=doppler, id=id)


def mk_matcher(gap_type=1, gap_weight=None, thresh=1.0, gap_dim=3):
    cfg = types.SimpleNamespace(MATCH=types.SimpleNamespace(
        gap_type=gap_type,
        gap_weight=[1.0, 1.0, 1.0] if gap_weight is None else gap_weight,
        thresh=thresh,
        gap_dim=gap_dim,
    ))
    return Matcher(cfg)


# brute-force 最大权匹配 oracle: 枚举所有 injective 部分行→列赋值, 返回 (最优总权, {row:col})
def brute_force_max(W):
    n_trk, n_obj = W.shape
    best = [-1.0, None]

    def rec(ri, used, cur, total):
        if ri == n_trk:
            if total > best[0]:
                best[0] = total
                best[1] = dict(cur)
            return
        rec(ri + 1, used, cur, total)  # row ri 不匹配
        for j in range(n_obj):
            if j not in used and W[ri][j] > 0:
                cur[ri] = j
                used.add(j)
                rec(ri + 1, used, cur, total + W[ri][j])
                used.discard(j)
                del cur[ri]

    rec(0, set(), {}, 0.0)
    return best[0], best[1]


def solver_total(W, match):
    """由 match 阵 (match[j]=i 或 -1) 反算总权."""
    return sum(W[match[j]][j] for j in range(len(match)) if match[j] >= 0)


def brute_force_max_prefilled(W, prefill):
    """带预填的 brute-force oracle: 锁定 prefill={row:col}, 剩余行/列求最大权部分匹配."""
    n_trk, n_obj = W.shape
    used_cols = set(prefill.values())
    locked = sum(W[r][c] for r, c in prefill.items())
    rows = [r for r in range(n_trk) if r not in prefill]
    best = [locked]

    def rec(ri, used, total):
        if ri == len(rows):
            best[0] = max(best[0], total)
            return
        rec(ri + 1, used, total)
        r = rows[ri]
        for j in range(n_obj):
            if j in used or j in used_cols:
                continue
            if W[r][j] > 0:
                used.add(j)
                rec(ri + 1, used, total + W[r][j])
                used.discard(j)

    rec(0, set(), locked)
    return best[0]


# ============================================================
# ----------------- _KMSolver 单元 (算法核心) -----------------
# ============================================================
def km_solve(W, match):
    # 直接喂 W + 预填 match 跑 KM (thresh 不参与, 仅满足 Matcher 构造)
    return mk_matcher(thresh=1.0)._km_solve(W, match)


# ============================================================
# ----------------------- _gap 数值正确性 -----------------------
# ============================================================
class TestGapNumeric:
    def test_euclidean_manual(self):
        # weight=[2,3,4], diff=(1,1,1): pos=2*1+3*1=5, dpl=4*1=4 → 9
        m = mk_matcher(gap_type=1, gap_weight=[2.0, 3.0, 4.0])
        assert m._gap(mk_trk(x=0, y=0, doppler=0), mk_obj(x=1, y=1, doppler=1)) == pytest.approx(9.0)

    def test_euclidean_per_axis_isolated(self):
        # 各权重单独作用: y 变化只受 weight[1]
        m = mk_matcher(gap_type=1, gap_weight=[0.0, 3.0, 0.0])
        g = m._gap(mk_trk(x=5, y=2, doppler=9), mk_obj(x=5, y=0, doppler=9))
        assert g == pytest.approx(3.0 * 4)

    def test_euclidean_zero_distance(self):
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 1.0])
        assert m._gap(mk_trk(x=1, y=2, doppler=3), mk_obj(x=1, y=2, doppler=3)) == pytest.approx(0.0)

    def test_euclidean_negative_squares_out(self):
        # 负坐标/负多普勒: 平方消号, 与正坐标同结果
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 1.0])
        g_pos = m._gap(mk_trk(x=0, y=0, doppler=0), mk_obj(x=1, y=1, doppler=1))
        g_neg = m._gap(mk_trk(x=0, y=0, doppler=0), mk_obj(x=-1, y=-1, doppler=-1))
        assert g_pos == pytest.approx(g_neg)

    def test_mahalanobis_square_isotropic(self):
        # len=wid=2,h=0 ×1.2 → 圆半轴1.2, 逆=diag(1/1.44); 椭圆边界上 maha=1 (weight 不参与位置项)
        m = mk_matcher(gap_type=2, gap_weight=[9.9, 9.9, 1.0])
        g = m._gap(mk_trk(x=0, y=0, doppler=0, length=2, width=2), mk_obj(x=1.2, y=0, doppler=0))
        assert g == pytest.approx(1.0)

    def test_mahalanobis_axis_scaled(self):
        # len=4,wid=2,h=0 ×1.2 → 半轴 (2.4, 1.2); 沿轴边界点 maha=1
        m = mk_matcher(gap_type=2, gap_weight=[1.0, 1.0, 0.0])
        gx = m._gap(mk_trk(x=0, y=0, doppler=0, length=4, width=2), mk_obj(x=2.4, y=0, doppler=0))
        gy = m._gap(mk_trk(x=0, y=0, doppler=0, length=4, width=2), mk_obj(x=0, y=1.2, doppler=0))
        assert gx == pytest.approx(1.0) and gy == pytest.approx(1.0)

    def test_mahalanobis_weight_ignored_in_pos(self):
        # weight[0]/[1] 改变不影响马氏位置项 (设计契约), 只 dpl 项随 weight[2] 变
        m1 = mk_matcher(gap_type=2, gap_weight=[1.0, 1.0, 1.0])
        m2 = mk_matcher(gap_type=2, gap_weight=[100.0, 100.0, 1.0])
        base = mk_trk(x=0, y=0, doppler=0, length=4, width=2)
        obj = mk_obj(x=1, y=1, doppler=0)
        assert m1._gap(base, obj) == pytest.approx(m2._gap(base, obj))

    def test_mahalanobis_heading_90_swaps_axes(self):
        # h=90: 长轴(原x,半轴2.4)转到 y, 短轴(原y,半轴1.2)转到 x → 椭圆边界 maha=1
        m = mk_matcher(gap_type=2, gap_weight=[1.0, 1.0, 0.0])
        trk = mk_trk(x=0, y=0, doppler=0, length=4, width=2, heading_deg=90)
        assert m._gap(trk, mk_obj(x=1.2, y=0, doppler=0)) == pytest.approx(1.0)
        assert m._gap(trk, mk_obj(x=0, y=2.4, doppler=0)) == pytest.approx(1.0)

    def test_mahalanobis_heading_offdiag_matches_reference(self):
        # h=45: 协方差非对角项非零 → 锁住旋转数学; 对照独立 numpy 实现 (R→R·Rᵀ→inv) 逐元素
        def ref(length, width, heading_deg, scale=1.2):
            h = np.deg2rad(heading_deg)
            c, s = np.cos(h), np.sin(h)
            hx, hy = 0.5 * length * scale, 0.5 * width * scale
            R = np.array([[hx * c, -hy * s], [hx * s, hy * c]])
            return np.linalg.inv(R @ R.T)
        trk = mk_trk(x=0, y=0, doppler=0, length=4, width=2, heading_deg=45)
        inv = mk_matcher(gap_type=2)._maha_inv(trk)
        assert abs(inv[0][1]) > 1e-6                                   # 非对角项确被激活
        np.testing.assert_allclose(inv, ref(4, 2, 45))


# ============================================================
# ------------------- _build_cost_table -------------------
# ============================================================
class TestBuildCostTable:
    def test_weight_is_thresh_minus_gap(self):
        # thresh=1.0, weight=[1,0,0] (只看 dx); gap 由 dx² 决定
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 0.0, 0.0], thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0), mk_trk(x=0.5, y=0.0)]
        objs = [mk_obj(x=0.5, y=0.0), mk_obj(x=100.0, y=0.0)]
        W = m._build_cost_table(trks, objs)
        assert W.shape == (2, 2)
        # t0(0) vs o0(0.5): dx=0.5 → gap=0.25 → W=0.75
        assert W[0][0] == pytest.approx(0.75)
        # t0(0) vs o1(100): gap=10000 → W=0
        assert W[0][1] == pytest.approx(0.0)
        # t1(0.5) vs o0(0.5): dx=0 → gap=0 → W=1.0
        assert W[1][0] == pytest.approx(1.0)
        # t1(0.5) vs o1(100): gap=99.5² → 0
        assert W[1][1] == pytest.approx(0.0)

    def test_empty_inputs(self):
        m = mk_matcher()
        assert m._build_cost_table([], []).shape == (0, 0)
        assert m._build_cost_table([mk_trk()], []).shape == (1, 0)

    def test_all_below_thresh_is_zero(self):
        # thresh 极小 → 全 gap≥thresh → 全 0
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 0.0], thresh=0.0)
        W = m._build_cost_table([mk_trk(x=0.0)], [mk_obj(x=0.1, y=0.0)])
        assert W[0][0] == pytest.approx(0.0)


# ============================================================
# --------------------- _build_graph ---------------------
# ============================================================
class TestBuildGraph:
    def test_unique_edge_direct_match(self):
        # 每行每列只 1 条可连边 → 直接配对
        W = np.array([[0.9, 0.0], [0.0, 0.8]])
        match = mk_matcher()._build_graph(W)
        assert list(match) == [0, 1]

    def test_conflict_left_for_km(self):
        # 2x2 全可连 → 冲突, 全留 -1 交 KM
        W = np.array([[0.9, 0.8], [0.8, 0.9]])
        match = mk_matcher()._build_graph(W)
        assert list(match) == [-1, -1]

    def test_partial_direct_partial_conflict(self):
        # W=[[0.5,0.7],[0.6,0]]: Ti=[2,1] Tj=[2,1]
        # col1 只 row0 可连(Tj=1) 但 row0 两边可连(Ti=2) → 双向不唯一 → 全冲突
        W = np.array([[0.5, 0.7], [0.6, 0.0]])
        match = mk_matcher()._build_graph(W)
        assert list(match) == [-1, -1]

    def test_true_unique_direct_match(self):
        # 真双向唯一: W=[[0.5,0],[0,0.7]] → Ti=[1,1] Tj=[1,1] → 各自直接配对
        W = np.array([[0.5, 0.0], [0.0, 0.7]])
        match = mk_matcher()._build_graph(W)
        assert list(match) == [0, 1]

    def test_empty_matrix(self):
        match = mk_matcher()._build_graph(np.zeros((0, 0)))
        assert list(match) == []

    def test_all_zero_no_match(self):
        W = np.zeros((2, 2))
        match = mk_matcher()._build_graph(W)
        assert list(match) == [-1, -1]


# ============================================================
# --------------------- _post_result ---------------------
# ============================================================
class TestPostResult:
    def test_split_matched_unmatched(self):
        # match=[0,-1,1]: (t0,o0)+(t1,o2) matched, o1 unmatched, t2 unmatched
        trks = [mk_trk(id=1), mk_trk(id=2), mk_trk(id=3)]
        objs = [mk_obj(id=10), mk_obj(id=11), mk_obj(id=12)]
        res = mk_matcher()._post_result(trks, objs, np.array([0, -1, 1]))
        assert {(t.id, o.id) for t, o in res.matched} == {(1, 10), (2, 12)}
        assert [t.id for t in res.unmatched_trks] == [3]
        assert [o.id for o in res.unmatched_objs] == [11]

    def test_all_matched(self):
        trks = [mk_trk(id=1), mk_trk(id=2)]
        objs = [mk_obj(id=10), mk_obj(id=11)]
        res = mk_matcher()._post_result(trks, objs, np.array([0, 1]))
        assert len(res.matched) == 2
        assert res.unmatched_trks == [] and res.unmatched_objs == []

    def test_none_matched(self):
        trks = [mk_trk(id=1), mk_trk(id=2)]
        objs = [mk_obj(id=10), mk_obj(id=11)]
        res = mk_matcher()._post_result(trks, objs, np.array([-1, -1]))
        assert res.matched == []
        assert [t.id for t in res.unmatched_trks] == [1, 2]
        assert [o.id for o in res.unmatched_objs] == [10, 11]

    def test_empty(self):
        res = mk_matcher()._post_result([], [], np.array([], dtype=int))
        assert res.matched == [] and res.unmatched_trks == [] and res.unmatched_objs == []


class TestKMSolver:
    def test_square_conflict_optimal(self):
        # 反对角强边: 最优为 (0,1)+(1,0)
        W = np.array([[90.0, 5.0], [5.0, 90.0]])
        match = km_solve(W, np.array([-1, -1]))
        assert solver_total(W, match) == pytest.approx(180.0)
        bf, _ = brute_force_max(W)
        assert solver_total(W, match) == pytest.approx(bf)

    def test_single_col_contest(self):
        # 2 行争 1 列: row1 权高胜出, row0 落 pad 列 (未匹配)
        W = np.array([[5.0], [6.0]])
        match = km_solve(W, np.array([-1]))
        assert solver_total(W, match) == pytest.approx(6.0)
        assert match[0] == 1  # col0 归 row1

    def test_more_rows_than_cols(self):
        # Ni=3 > Nj=2: pad 列吸收多出的行
        W = np.array([[80.0, 70.0], [70.0, 80.0], [60.0, 60.0]])
        match = km_solve(W, np.array([-1, -1]))
        bf, _ = brute_force_max(W)
        assert solver_total(W, match) == pytest.approx(bf)

    def test_more_cols_than_rows(self):
        # Ni=2 < Nj=3: 多出的列自然未匹配
        W = np.array([[80.0, 70.0, 60.0], [60.0, 80.0, 70.0]])
        match = km_solve(W, np.array([-1, -1, -1]))
        bf, _ = brute_force_max(W)
        assert solver_total(W, match) == pytest.approx(bf)

    def test_prefilled_direct_passthrough(self):
        # 预填直接配对 (无冲突) → 原样透传
        W = np.array([[90.0, 0.0], [0.0, 80.0]])
        match = km_solve(W, np.array([0, 1]))
        assert list(match) == [0, 1]

    def test_empty_matrix(self):
        W = np.zeros((0, 0))
        match = km_solve(W, np.array([], dtype=int))
        assert list(match) == []

    @pytest.mark.parametrize("seed", range(60))
    def test_random_vs_bruteforce(self, seed):
        # 强性质: 60 个随机矩阵, solver 总权 == brute-force 最优
        rng = np.random.RandomState(seed)
        n_trk = rng.randint(1, 6)
        n_obj = rng.randint(1, 6)
        W = rng.uniform(0, 10, (n_trk, n_obj))
        W[rng.random((n_trk, n_obj)) < 0.3] = 0.0  # ~30% 不可连边
        match = km_solve(W, np.full(n_obj, -1, dtype=int))
        bf, _ = brute_force_max(W)
        assert solver_total(W, match) == pytest.approx(bf, abs=1e-6), \
            f"seed={seed} W=\n{W}\nsolver={solver_total(W, match)} bf={bf}"

    @pytest.mark.parametrize("seed", range(30))
    def test_random_vs_bruteforce_with_prefill(self, seed):
        # 带随机预填的 fuzz: 覆盖 prefill + KM 冲突混合路径 (Ci/Cj 索引与回填)
        rng = np.random.RandomState(seed + 1000)
        n_trk = rng.randint(2, 6)
        n_obj = rng.randint(2, 6)
        W = rng.uniform(0, 10, (n_trk, n_obj))
        W[rng.random((n_trk, n_obj)) < 0.3] = 0.0
        prefill = {}
        for r in rng.permutation(n_trk):
            cands = [j for j in range(n_obj)
                     if W[r][j] > 0 and j not in prefill.values() and r not in prefill]
            if cands and rng.rand() < 0.5:
                prefill[r] = int(rng.choice(cands))
        match = np.full(n_obj, -1, dtype=int)
        for r, c in prefill.items():
            match[c] = r
        out = km_solve(W, match)
        bf = brute_force_max_prefilled(W, prefill)
        assert solver_total(W, out) == pytest.approx(bf, abs=1e-6), \
            f"seed={seed} W=\n{W} prefill={prefill}\nsolver={solver_total(W, out)} bf={bf}"

    # ---- 确定性手算例: 独立于 brute-force, 验证 KM 原理 (顶标/增广/pad) ----
    def test_augment_path_flip(self):
        # 增广路翻转: W=[[5,4],[6,0]], match[j]=i 语义
        # 最优 = row1→col0(6) + row0→col1(4) = 10, 需 row0 让出 col0 改配 col1
        W = np.array([[5.0, 4.0], [6.0, 0.0]])
        match = km_solve(W, np.array([-1, -1]))
        assert match[0] == 1        # col0 归 row1
        assert match[1] == 0        # col1 归 row0
        assert solver_total(W, match) == pytest.approx(10.0)

    def test_label_relaxation_needed(self):
        # 两行都只争 col0, 相等子图无完美匹配 → 必须松弛顶标才能配出 9
        W = np.array([[5.0, 4.0], [5.0, 4.0]])
        match = km_solve(W, np.array([-1, -1]))
        assert solver_total(W, match) == pytest.approx(9.0)
        assert list(match) == [1, 0]

    def test_strong_prefilled_blocks_rematch(self):
        # 预填直接配对不参与增广 (build_graph 已透传), KM 只解 -1 位
        # match=[0, -1]: row0 已锁 col0, row1 需独立找; col1 空
        W = np.array([[5.0, 4.0], [6.0, 3.0]])
        match = km_solve(W, np.array([0, -1]))
        assert match[0] == 0        # 透传不变
        assert match[1] == 1        # row1 配 col1

    def test_pad_column_absorbs_extra_row(self):
        # Ni=3 > Nj=2: 最优只配 2 对, 最弱行落 pad 列 (未匹配)
        # W=[[10,1],[1,10],[9,9]]: 最优 = 10+10 = 20 (前两行各取强列), row2 落空
        W = np.array([[10.0, 1.0], [1.0, 10.0], [9.0, 9.0]])
        match = km_solve(W, np.array([-1, -1]))
        assert solver_total(W, match) == pytest.approx(20.0)
        assert np.sum(match >= 0) == 2     # 只配出 2 对

    def test_optimal_not_greedy(self):
        # 真·反贪心 3x2: W=[[8,7,0],[7,8,0],[9,6,0]]
        # 行最大贪心: row0→col0(8)+row1→col1(8), row2 落空 = 16 (次优)
        # 全局最优: row2→col0(9)+row1→col1(8), row0 落空 = 17 (KM 必须放弃 row0)
        W = np.array([[8.0, 7.0, 0.0], [7.0, 8.0, 0.0], [9.0, 6.0, 0.0]])
        match = km_solve(W, np.array([-1, -1, -1]))
        assert solver_total(W, match) == pytest.approx(17.0)
        assert match[0] == 2        # col0 归 row2
        assert match[1] == 1        # col1 归 row1

    def test_single_matchable_all_zero_weights(self):
        # 全 0 权 → 无可连边 → 全 -1, 不崩
        W = np.zeros((2, 2))
        match = km_solve(W, np.array([-1, -1]))
        assert list(match) == [-1, -1]


# ============================================================
# ----------------- Matcher.run 集成场景 -----------------
# ============================================================
class TestMatcherRun:
    def test_one_to_one_direct(self):
        # 单 trk/obj 近距 → 直接配对 (build_graph 直接匹配, 无需 KM)
        m = mk_matcher(thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0, id=1)]
        objs = [mk_obj(x=0.1, y=0.0, id=10)]
        res = m.run(trks, objs)
        assert len(res.matched) == 1
        assert res.matched[0][0].id == 1 and res.matched[0][1].id == 10
        assert res.unmatched_trks == [] and res.unmatched_objs == []

    def test_conflict_resolved_optimal(self):
        # 2x2 全可连, 强制走 KM: 最优 t0↔o0, t1↔o1
        m = mk_matcher(thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0, id=1), mk_trk(x=0.2, y=0.0, id=2)]
        objs = [mk_obj(x=0.0, y=0.0, id=10), mk_obj(x=0.15, y=0.0, id=11)]
        res = m.run(trks, objs)
        assert len(res.matched) == 2
        pair = {(t.id, o.id) for t, o in res.matched}
        assert pair == {(1, 10), (2, 11)}

    def test_mixed_prefill_and_km_conflict(self):
        # t0 唯一连 o0 (预填); t1/t2 争 o1/o2 (KM 冲突) → 混合路径
        m = mk_matcher(thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0, id=1), mk_trk(x=5.0, y=0.0, id=2), mk_trk(x=5.0, y=0.2, id=3)]
        objs = [mk_obj(x=0.1, y=0.0, id=10), mk_obj(x=5.1, y=0.0, id=11), mk_obj(x=5.1, y=0.2, id=12)]
        res = m.run(trks, objs)
        pair = {(t.id, o.id) for t, o in res.matched}
        assert pair == {(1, 10), (2, 11), (3, 12)}
        assert res.unmatched_trks == [] and res.unmatched_objs == []

    def test_km_no_phantom_match_on_zero_edge(self):
        # 回归: 0 权(不可连)边不得被 KM 当匹配 - B 不得配给 141m 外的 Y
        m = mk_matcher(thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0, id=1), mk_trk(x=0.5, y=0.0, id=2)]
        objs = [mk_obj(x=0.1, y=0.0, id=10), mk_obj(x=100.0, y=100.0, id=11)]
        res = m.run(trks, objs)
        assert [(t.id, o.id) for t, o in res.matched] == [(1, 10)]
        assert [t.id for t in res.unmatched_trks] == [2]
        assert [o.id for o in res.unmatched_objs] == [11]

    def test_unmatched_trk(self):
        # trk 远离所有 obj → unmatched_trks
        m = mk_matcher(thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0, id=1), mk_trk(x=100.0, y=100.0, id=2)]
        objs = [mk_obj(x=0.1, y=0.0, id=10)]
        res = m.run(trks, objs)
        assert len(res.matched) == 1
        assert [t.id for t in res.unmatched_trks] == [2]
        assert res.unmatched_objs == []

    def test_unmatched_obj(self):
        # obj 远离所有 trk → unmatched_objs
        m = mk_matcher(thresh=1.0)
        trks = [mk_trk(x=0.0, y=0.0, id=1)]
        objs = [mk_obj(x=0.1, y=0.0, id=10), mk_obj(x=100.0, y=100.0, id=11)]
        res = m.run(trks, objs)
        assert len(res.matched) == 1
        assert [o.id for o in res.unmatched_objs] == [11]
        assert res.unmatched_trks == []

    def test_both_empty(self):
        m = mk_matcher()
        res = m.run([], [])
        assert res.matched == [] and res.unmatched_trks == [] and res.unmatched_objs == []

    def test_all_disconnectable(self):
        # 全部 gap>=thresh → 全不可连, 全未匹配, 不崩
        m = mk_matcher(thresh=0.5)
        trks = [mk_trk(x=0.0, y=0.0, id=1)]
        objs = [mk_obj(x=10.0, y=10.0, id=10)]  # gap=200 >> thresh
        res = m.run(trks, objs)
        assert res.matched == []
        assert [t.id for t in res.unmatched_trks] == [1]
        assert [o.id for o in res.unmatched_objs] == [10]

    def test_returns_matches_type(self):
        m = mk_matcher()
        res = m.run([mk_trk(id=1)], [mk_obj(x=0.1, id=10)])
        assert isinstance(res, Matches)


# ============================================================
# ----------------------- gap 加固 -----------------------
# ============================================================
class TestGapHardening:
    def test_mahalanobis_degenerate_extent_rejected(self):
        # 长宽退化(长=0) → 逆不存在 → gap=inf 拒绝匹配 (不误算 0)
        m = mk_matcher(gap_type=2)
        g = m._gap(mk_trk(x=0.0, y=0.0, length=0.0, width=2.0), mk_obj(x=1.0, y=1.0))
        assert g == float('inf')

    def test_mahalanobis_degenerate_extent_no_cost(self):
        # 退化航迹建表: 该行全 0 (不可连), 不崩
        m = mk_matcher(gap_type=2, thresh=1.0)
        W = m._build_cost_table([mk_trk(x=0.0, y=0.0, length=0.0, width=2.0)], [mk_obj(x=1.0, y=1.0)])
        assert W[0][0] == pytest.approx(0.0)

    def test_isvalid_rejects_short_weight(self):
        # gap_dim=2 + len(gap_weight)=2 → isvalid 应拒绝 (强制 len>=3)
        import yaml
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'cfg', 'cfg.yaml')
        with open(cfg_path, 'r', encoding='utf-8') as f:
            raw = yaml.safe_load(f)
        raw['MATCH']['gap_weight'] = [1.0, 1.0]  # 只给 2 元素
        cfg = _build_cfg_from_raw(raw)
        with pytest.raises(ValueError):
            cfg.isvalid()


# 辅助: 从 raw dict 构建 Cfg (复用 Cfg.get_cfg 的逻辑但走内存)
def _build_cfg_from_raw(raw):
    import tempfile
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False, encoding='utf-8') as f:
        yaml = __import__('yaml')
        yaml.dump(raw, f)
        path = f.name
    return Cfg.get_cfg(path)


# ============================================================
# ----------------------- gap_dim 维度开关 -----------------------
# ============================================================
class TestGapDim:
    def test_dim2_drops_doppler_euclidean(self):
        # weight=[1,1,5], diff=(1,1,1): gap_dim=2 只位置=2, 不加 dpl
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 5.0], gap_dim=2)
        g = m._gap(mk_trk(x=0, y=0, doppler=0), mk_obj(x=1, y=1, doppler=1))
        assert g == pytest.approx(2.0)

    def test_dim3_adds_doppler_euclidean(self):
        # 同输入 gap_dim=3 → 位置 2 + dpl 5 = 7
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 5.0], gap_dim=3)
        g = m._gap(mk_trk(x=0, y=0, doppler=0), mk_obj(x=1, y=1, doppler=1))
        assert g == pytest.approx(7.0)

    def test_dim2_doppler_weight_ignored(self):
        # gap_dim=2: 改 dpl 权重不影响 gap
        m1 = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 1.0], gap_dim=2)
        m2 = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 99.0], gap_dim=2)
        base = mk_trk(x=0, y=0, doppler=0)
        obj = mk_obj(x=1, y=1, doppler=5)
        assert m1._gap(base, obj) == pytest.approx(m2._gap(base, obj))

    def test_dim2_drops_doppler_mahalanobis(self):
        # 马氏 len=wid=2(圆半轴1.2), diff=(1.2,1.2) → pos=2; doppler 不加
        m = mk_matcher(gap_type=2, gap_weight=[1.0, 1.0, 5.0], gap_dim=2)
        g = m._gap(mk_trk(x=0, y=0, doppler=0, length=2, width=2), mk_obj(x=1.2, y=1.2, doppler=1))
        assert g == pytest.approx(2.0)

    def test_run_dim2_doppler_absent_in_cost(self):
        # 集成: gap_dim=2 时两 obj 位置同/doppler 异 → 代价相同
        m = mk_matcher(gap_type=1, gap_weight=[1.0, 1.0, 5.0], thresh=1.0, gap_dim=2)
        trks = [mk_trk(x=0.0, y=0.0, doppler=0.0, id=1)]
        objs = [mk_obj(x=0.0, y=0.0, doppler=0.0, id=10),
                mk_obj(x=0.0, y=0.0, doppler=9.0, id=11)]
        W = m._build_cost_table(trks, objs)
        assert W[0][0] == pytest.approx(W[0][1])


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
