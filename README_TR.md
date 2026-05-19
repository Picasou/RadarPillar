<div align="center">

# RadarPillars: View-of-Delft üzerinde yeniden üretim

**Yalnızca radar girdisiyle 3B nesne tespiti — [Gillen ve ark., IROS 2024](https://arxiv.org/abs/2408.05020) çalışmasının OpenPCDet tabanlı reprodüksiyonu**

</div>

---

## Özet sonuçlar

| Yöntem | Araç | Yaya | Bisikletçi | mAP_3D (R11) |
|---|:---:|:---:|:---:|:---:|
| MAFF-Net (PV-RCNN, 2025) | 42.3 | 46.8 | 74.7 | 54.6 |
| SCKD (2025) | 41.9 | 43.5 | 70.8 | 52.1 |
| **Bizim — en iyi tohum** | **41.6** | **44.8** | 71.3 | **52.56** |
| Bizim — 3-tohum ortalama | 41.0 | 43.2 | 70.1 | 51.43 ± 0.99 |
| SMURF (2023) | 42.3 | 39.1 | 71.5 | 51.0 |
| **RadarPillars (orijinal makale)** | 41.1 | 38.6 | 72.6 | **50.70** |
| CenterPoint (taban) | 33.9 | 39.0 | 66.9 | 46.6 |
| PointPillars (taban) | 37.9 | 31.2 | 65.7 | 45.0 |

VoD doğrulama kümesinde 3B AP (R11), IoU eşikleri Araç=0.50, Yaya/Bisikletçi=0.25.

En iyi ağırlık dosyası: `output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth`
Ablasyon, tohum bazlı kayıtlar, hiperparametre tabloları → [`experiments/RESULTS.md`](experiments/RESULTS.md).

---

## Mimari

```
Radar nokta bulutu (N,7)
  → PillarVFE (voxelleştirme + Doppler ayrıştırması: vx, vy = atan2 ile)
  → PillarAttention (maskeli öz-dikkat, C=E=32)
  → PointPillarScatter (320×320×32 BEV)
  → BaseBEVBackbone (3 bloklu 2B CNN, sabit kanal C=32)
  → AnchorHeadSingle (Araç / Yaya / Bisikletçi)
```

Temel uygulama detayları:
- **Hız ayrıştırması** VFE içinde: `vx = v_r_comp·cos(φ)`, `vy = v_r_comp·sin(φ)`, `φ = atan2(y, x)`
- **Fizik-tutarlı veri artırma**: hız vektörleri, nokta koordinatlarıyla birlikte döndürülür/yansıtılır (OpenPCDet'in nuScenes sütun düzenini varsayan hatasını giderir)
- **PillarAttention** key-padding mask ile çalışır — boş pilarlar dikkat skorlarını kirletmez
- **`FFN_CHANNELS` konfig sürücülü** (`pillar_attention.py`); önceki sürümde `*2` hardcoded'du

---

## Kurulum

```bash
python -m venv .venv && source .venv/bin/activate
pip install -U pip
python setup.py develop
```

Gereksinimler: Python 3.8+, PyTorch 2.4+, CUDA 12.x, spconv 2.3.6.

---

## Veri

```
data/VoD/view_of_delft_PUBLIC/radar_5frames/
  ├── ImageSets/{train,val,test}.txt
  ├── training/{velodyne,label_2,calib,image_2}/
  └── testing/velodyne/
```

Info pkl + GT veritabanı üretimi:
```bash
python -m pcdet.datasets.vod.vod_dataset create_vod_infos \
    tools/cfgs/dataset_configs/vod_dataset_radar.yaml
```

---

## Eğitim

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --batch_size 8 --extra_tag <koşu_adı> --workers 4
```

3-tohum çoklu koşu (özet tablodaki sayıyı üreten komut):
```bash
bash experiments/chain_scripts/multiseed_v2.sh
```

---

## Değerlendirme

```bash
CUDA_VISIBLE_DEVICES=0 python tools/test.py \
  --cfg_file tools/cfgs/vod_models/vod_radarpillar_rot.yaml \
  --ckpt output/cfgs/vod_models/vod_radarpillar_rot/paper_faithful_rot_s3/ckpt/checkpoint_best.pth
```

---

## Konfigler

| Dosya | Açıklama |
|---|---|
| `tools/cfgs/vod_models/vod_radarpillar.yaml` | makale Section IV'e sadık temel hat (rotation yok) |
| `tools/cfgs/vod_models/vod_radarpillar_rot.yaml` | **rotation eklenmiş varyant — özet sonucu üreten konfig** |
---

## Atıf

```bibtex
@inproceedings{gillen2024radarpillars,
  title     = {RadarPillars: Efficient Object Detection from 4D Radar Point Clouds},
  author    = {Gillen, Julius and Bieder, Manuel and Stiller, Christoph},
  booktitle = {Proc. IEEE/RSJ Int. Conf. Intelligent Robots and Systems (IROS)},
  year      = {2024}
}

@misc{openpcdet2020,
  title  = {OpenPCDet: An Open-source Toolbox for 3D Object Detection from Point Clouds},
  author = {OpenPCDet Development Team},
  year   = {2020},
  url    = {https://github.com/open-mmlab/OpenPCDet}
}
```

---

## Lisans

Apache 2.0 License altında yayınlanmıştır — bkz. [LICENSE](LICENSE). Bu proje [OpenPCDet](https://github.com/open-mmlab/OpenPCDet) üzerine inşa edilmiştir; OpenPCDet'in lisansı da Apache 2.0'dır.
