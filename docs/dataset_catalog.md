# Illustrated dataset catalog (traffic signs)

`scripts/make_dataset_catalog.py` builds a Japanese catalog (HTML and PDF) that puts the definition
of every class or category of the representative sign datasets next to real samples drawn with a
fixed seed. It exists because the survey tables in `datasets.md` cannot show what the data looks
like: how small the signs are, how the label granularity differs between datasets, and how the
same sign type changes between countries. The built catalog lives under `runs/catalog/` (not
committed); the 2026-11-01 build was delivered to the maintainer as a PDF.

## Sections

| Section | Data used | What is shown |
|---|---|---|
| GTSRB | `data/gtsrb_hf` (official split) and the Synset Signset Germany GTSRB subset | all 43 classes with id, Japanese and English names, GTSDB category, train/test counts, median size, 6 test samples and 2 synthetic twins; size histogram; category definitions |
| GTSDB | `data/gtsdb_hf` (600 training frames, gt.txt) | 6 annotated frames, box size histogram, 14 crops per competition category, per-class counts |
| Panoramax classified DE | `data/panoramax_de/val` | all 134 classes with a Japanese gloss, counts, non-German padding, 6 samples each |
| Synset Signset Germany | `data/synset_signset/cycles/validation.parquet` | image / mask / segmentation triples and the distribution of the rendering metadata |
| TT100K | `data/tt100k_hf` (three parquet shards, `categories.csv`) | code-family legend, 4 frames, box size histogram, up to 3 crops for every category present in the shards |
| MTSD | `data/mtsd_official` (sample of official annotation JSON + images, see below) | taxonomy groups, 4 frames, size histogram, box attribute counts, one crop per label present in the sample |
| CURE-TSR | `data/cure_tsr/CURE-TSR.zip` (read in place) | 14 sign types with real and unreal samples, 12 challenge types at levels 1, 3, 5 |
| Japan (道路標識令) | `data/japan/mlit` (tiles cut from the MLIT sign list) | the five official categories with definitions and all 221 pictograms of the MLIT list; no public dataset exists, so this section is the class list for the Japan track |

## Building it

```bash
make gtsrb gtsdb replicate-panoramax   # GTSRB, GTSDB and the Panoramax validation crops (the replication data)
make catalog-data                      # Synset, TT100K shards, MTSD sample, CURE-TSR zip, MLIT list and tiles
make catalog                           # runs/catalog/catalog.html and catalog.pdf (playwright + Chromium for the PDF)
```

Sampling uses `random.Random("<seed>:<dataset>")`, so the same seed and data files give the same
pictures. Detection samples are cropped with a 15 percent margin around the ground-truth box and
scaled to 64 to 72 px; frames are scaled to 820 px width with boxes coloured by category. All
images are embedded as data URIs, so the HTML is self contained (about 20 MB).

## Notes on the sources

- TT100K and MTSD are non-commercial datasets; their samples appear in the catalog as research
  citations and are not part of anything OpenTSR redistributes.
- The `ThankGod/mapillary_traffic_sign_dataset*` parquet mirrors of MTSD carry integer categories
  whose `id2labels.txt` does not match the images (checked by eye and by matching every box of the
  cleaned test shard against the original mirror: the ids are identical in both, and both disagree
  with the label file). The catalog therefore samples the official annotation JSON files, which the
  `NgTMDuc/cl_dataset` mirror hosts together with the training images (`scripts/sample_mtsd_official.py`).
- Wikimedia Commons could not be used for the Japanese pictograms from the cloud session (the
  thumbnail service answers 429 for automated clients). The MLIT list (`ichiran.pdf`, one A3 page)
  is the official source anyway; `scripts/cut_mlit_sign_list.py` cuts it into tiles using the label
  positions from `pdftotext -bbox-layout`. A handful of guide-sign tiles (route-number stacks) are
  still partial because of the page layout.
- Japanese glosses for the Panoramax and TT100K class codes are the catalog author's
  translations; StVO and GB 5768 numbers are indicative.
