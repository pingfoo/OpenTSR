# Public datasets for traffic sign detection and classification

Survey compiled on 2026-10-31 from live web pages (claude.ai research session, pingfoo/OpenTSR).
Every factual claim carries the URL on which it was verified. Where a page could not be fetched
or did not state a value, the field says "unknown" or "not verified". Numbers that disagree
between sources are reported with both values. Section 7 is the summary table, section 8 the
recommendation for OpenTSR, section 9 the list of pages that could not be fetched (claims that
rest on secondary sources are marked there).

Conventions:
- "NC" = non-commercial restriction; "SA" = share-alike; "ND" = no derivatives
- "type labelled" = the fine-grained sign type (speed limit 50, no entry, ...) is annotated,
  as opposed to a single generic "traffic sign" box
- "public test labels" = the labels of the official test split can be downloaded
- Japanese sign categories: 規制 (regulatory), 指示 (indication/mandatory), 警戒 (warning),
  案内 (guide), 補助 (supplementary plates)
- Hugging Face metadata (download counts, split sizes, license tags) read from the JSON API
  on 2026-10-31

Quick orientation (details below):

| If you need | Use | Why |
|---|---|---|
| The standard classification benchmark, redistributable | GTSRB (43 classes, 39,209 / 12,630) | "free to use" per the official site; test labels public since 2011; several HF mirrors |
| A synthetic twin of GTSRB with a clean license | Synset Signset Germany (CC BY 4.0, 211 classes, 105,500 images) | reaches 98.3% on GTSRB when trained purely synthetic |
| Robustness / corruption test sets | CURE-TSR (Zenodo copy is CC BY 4.0), tanganke/gtsrb corrupted splits | 12 challenge types x 5 levels |
| The standard small detection benchmark with public test labels | GTSDB (900 images, 43 classes) | full annotations released 2013-07-15 |
| Large detection sets with fine types | MTSD (NC), TT100K (NC), DFG (NC-SA), ZOD (CC BY-SA 4.0, 156 sign classes) | ZOD is the only large fine-grained one with a commercial-friendly license |
| Japanese signs with any public labels | nothing public of research quality (section 3) | only tiny Roboflow community sets and one Apache-2.0 detector whose training data is withheld |
| Pictograms for template rendering | Wikimedia Commons SVGs: German signs public domain (official works), Japanese signs public domain (Art. 13 Copyright Act of Japan) | section 4 |

---

## 1. Classification benchmarks (cropped signs)

### 1.1 GTSRB, German Traffic Sign Recognition Benchmark (Germany) [IMPORTANT]

- Origin: Institut fuer Neuroinformatik, Ruhr-Universitaet Bochum. Competition at IJCNN 2011. Papers: Stallkamp, Schlipsing, Salmen, Igel, "The German Traffic Sign Recognition Benchmark: A multi-class classification competition", IJCNN 2011, pp. 1453-1460 (https://benchmark.ini.rub.de/gtsrb_dataset.html); Stallkamp et al., "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition", Neural Networks 2012 (https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf). The site asks: "Please cite this paper when when using or referring to the GTSRB dataset" (https://benchmark.ini.rub.de/).
- Geography: roads in Germany; about 10 hours of video recorded in daytime in March, October and November 2010 with a Prosilica GC 1380CH at 1360x1024, 25 fps (https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf).
- Size: 51,840 images total, 12,630 test images, 43 classes (https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf). Train 39,209 / test 12,630 per the HF mirror card (https://huggingface.co/datasets/bazyl/GTSRB/raw/main/README.md) and per Haloi 2015 ("39,209 training images" out of 51,839) (https://arxiv.org/pdf/1511.02992). The official page only says "More than 50,000 images in total" and "More than 40 classes" (https://benchmark.ini.rub.de/gtsrb_dataset.html).
- Image sizes: 15x15 to 250x250 pixels per the official page; images are not necessarily square, the sign is not always centred, each image has a border of 10% around the sign (at least 5 pixels) (https://benchmark.ini.rub.de/gtsrb_dataset.html). The journal paper gives 15x15 to 222x193 (https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).
- Construction: source material of about 133,000 labelled images of 2,416 sign instances in 70 classes; tracks of 30 images per physical sign; tracks with fewer than 30 images and classes with fewer than 9 tracks discarded (https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf). Test images are "consecutively numbered and shuffled to prevent deduction of class membership" (https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).
- Format: PPM (P6) images; semicolon-separated CSV with Filename, Width, Height, ROI.x1, ROI.y1, ROI.x2, ROI.y2, ClassId; one directory per class with GT-<ClassID>.csv; precomputed HOG (1568/1568/2916 dims), Haar (11,584) and hue histogram features (https://benchmark.ini.rub.de/gtsrb_dataset.html).
- Splits and their history: competition started 2010-12-01; online test set released 2011-01-19; class IDs for the online test set published 2011-01-25; final training set released 2011-07-12; final test set released with ground truth 2011-08-23; the submission site was discontinued in July 2020 (https://benchmark.ini.rub.de/gtsrb_news.html). The Phase I (online competition) training set had 26,640 images and the online test set 12,569 images (https://www.arxiv.org/pdf/1202.2745); 26,640 + 12,569 = 39,209, which is the size of the final training set.
- License (quoted as written): "The data is free to use. However, we cordially ask you to cite the following publication if you do." No formal license name is given (https://benchmark.ini.rub.de/gtsrb_dataset.html). The result analysis application tsr-analysis-src.zip is GPLv2 (same page). The 2011 call for papers states "The GTSRB data is freely available from http://benchmark.ini.rub.de" (https://benchmark.ini.rub.de/).
- Hosting: on 2019-05-08 the INI site announced it had to "temporarily shut down the download section" and that "The datasets have been transferred to an external hosting service and can be downloaded from there" (https://benchmark.ini.rub.de/). The external host is sid.erda.dk: torchvision downloads from base_url "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/" the files GTSRB-Training_fixed.zip (md5 513f3c79a4c5141765e10e952eaa2478), GTSRB_Final_Test_Images.zip (md5 c7e4e6327067d32654124b0fe9e82185) and GTSRB_Final_Test_GT.zip (md5 fe31e9c9270bbcd7b84b7f21a9d9d9e5), and reads GT-final_test.csv for the test split (https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/gtsrb.py). The sid.erda.dk archive page itself could not be fetched in this session (section 9).
- torchvision: `torchvision.datasets.GTSRB(root, split="train"|"test", download=...)`; the documentation page names only the benchmark website as source and states no license (https://docs.pytorch.org/vision/0.26/generated/torchvision.datasets.GTSRB.html).
- Hugging Face mirrors (API read 2026-10-31):
  - bazyl/GTSRB: parquet train/test, 2,295 downloads, last modified 2022-10-25, license tag "gpl-3.0" (a tag chosen by the uploader, not the INI wording), card says "Categories: 42, Train: 39209, Test: 12630" (the class count 42 is a card error, the data has class IDs 0..42) and links the Kaggle copy kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign (https://huggingface.co/api/datasets/bazyl/GTSRB?full=true, https://huggingface.co/datasets/bazyl/GTSRB/raw/main/README.md).
  - tanganke/gtsrb: 16,418 downloads, no license tag, 43 named classes, splits train 26,640 / test 12,630 plus seven corrupted copies of the test set (contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter, 12,630 each), download_size 1,027,074,522 bytes (https://huggingface.co/api/datasets/tanganke/gtsrb?full=true, https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md). Its 26,640-image train split differs from the official 39,209: it matches the Phase I training set size reported by Ciresan et al. (26,640 training / 12,569 online-test) (https://www.arxiv.org/pdf/1202.2745), so models trained on it see about 32% fewer training images than models trained on the official split. The card does not explain the split.
  - clip-benchmark/wds_gtsrb: WebDataset tar shards, train 6 shards / test 3 shards, classnames.txt with 43 lines, zeroshot_classification_templates.txt, no README, no license tag, 567 downloads (https://huggingface.co/api/datasets/clip-benchmark/wds_gtsrb?full=true, https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/raw/main/classnames.txt).
  - Others found by API search: Zzyy2000/GTSRB, dpdl-benchmark/gtsrb, ilee0022/GTSRB, SomeBottle/GTSRB_224x224_balanced, mteb/wds_gtsrb, mteb/GTSRBZeroShot (license tag "unknown", test 12,630), zjl123321/GTSRB, pymlstudio/gtsrb-cleaned (https://huggingface.co/api/datasets?search=gtsrb&limit=100).
- Other mirrors: Zenodo record 13741936 "GTSRB ... by Real-Time Computer Vision at Ruhr-Universitaet Bochum", uploaded by Mehran, Kimovski, Najafabadi Samani, Nikolov, Prodan (not the INI authors), data.zip 320.5 MB, license CC BY 4.0, DOI 10.5281/zenodo.13741936; the page does not claim to be official (https://zenodo.org/record/13741936). Derived Zenodo sets: GTSRB-Concepts additional labels and ontology (CC BY-NC-SA 4.0, record 16326403) and GTSRB with three backdoor patterns (CC BY 4.0, record 3716766, files GTSRB_backdoor_green_1.zip etc.) (https://zenodo.org/api/records?q=%22traffic%20sign%22%20OR%20%22traffic%20signs%22%20OR%20%22road%20signs%22&size=25&page=1&type=dataset, https://zenodo.org/api/records/3716766).
- Baselines and state of the art over time (all on the final 12,630-image test set unless stated):
  - IJCNN 2011 Phase I (online test set): INI-RTCV human performance 99.81%; IDSIA CNN + HOG3 committee 98.98%; sermanet 98.97%; VISICS IK-SVM + PHOG + HOG2 97.88%; noob HOG + LDA + VQ 96.87%; INI baselines HOG + LDA 96.32 / 94.73 / 94.51%; HOG + k-NN 72.81 to 73.89% (https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf).
  - Final round (2011-08-03 ranking): IDSIA committee of CNNs 99.46%; human performance 98.84%; sermanet multi-scale CNNs 98.31%; CAOR random forests 96.14% (https://benchmark.ini.rub.de/gtsrb_news.html). Journal table adds best individual human 99.22% and LDA on HOG 2 / HOG 1 / HOG 3 at 95.68 / 93.18 / 92.34%, with per-subset results (speed limits 99.47% for the committee, danger 99.07%, etc.) (https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf).
  - Ciresan, Meier, Schmidhuber, MCDNN with 25 columns, 48x48 inputs, 0.54% error = 99.46%, 37 hours on four GPUs (https://www.arxiv.org/pdf/1202.2745).
  - Sermanet and LeCun, IJCNN 2011: 98.97% in Phase I, later 99.17% with larger capacity and greyscale input (https://research.google/pubs/traffic-sign-recognition-with-multi-scale-convolutional-networks/).
  - Haloi 2015 (arXiv 1511.02992): spatial transformer + modified inception, 99.81% top-1, about 10.5 M parameters; table compares 99.46 (committee), 98.84 (human), 98.31 (multi-scale CNN), 98.14 (pLSA), 96.14 (random forest) (https://arxiv.org/pdf/1511.02992).
  - Arcos-Garcia, Alvarez-Garcia, Soria-Morillo, Neural Networks 99:158-165, 2018: CNN with spatial transformers, 99.71% (https://idus.us.es/handle/11441/80679); listed first on the official results page as "DeepKnowledge Seville, CNN with 3 Spatial Transformers, 99.71%", above IDSIA 99.46%, COSFIRE 98.97%, human 98.84% (https://benchmark.ini.rub.de/gtsrb_results.html). Code: https://github.com/aarcosg/tsr-torch (link seen in search results, not fetched).
  - Later: hybrid CNN/transformer with locality module 99.66% on GTSRB and 99.8% on the Persian Traffic Sign Dataset (https://arxiv.org/abs/2311.06651); EATFormer-style vision transformer paper on GTSRB and BelgiumTS, numbers not in the abstract (https://arxiv.org/abs/2404.19066). Claims above 99.8% exist in the literature (the 99.81% of Haloi is the highest verified here); a 2025 PMC article with a comparison table could not be fetched (section 9).
- Known problems: the test set is drawn from the same 2010 videos as training (same camera, same country, daytime only); images are tiny PPM crops with heavy motion blur and low contrast; the official site's "free to use" is not an OSI or CC license, so the exact redistribution rights are implicit; HF mirrors carry contradictory license tags (gpl-3.0, none, unknown); the popular tanganke mirror has a non-official train split.

### 1.2 BelgiumTS classification, BTSC (Belgium)

- Origin: Radu Timofte (ETH Zurich, formerly KU Leuven VISICS). Hosted at ETH since 2014-02-18, the older KU Leuven address is dead (https://btsd.ethz.ch/shareddata/). Papers to cite: Timofte, Zimmermann, Van Gool, MVA 2011 (introduces BelgiumTS); Timofte and Van Gool, BMVC 2011 (introduces BTSC); Mathias, Timofte, Benenson, Van Gool, IJCNN 2013 (BTSD and BTSC splits) (same page). Data acquired by GeoAutomation with a van carrying 8 roof-mounted cameras (https://btsd.ethz.ch/shareddata/, https://arxiv.org/pdf/1712.02463).
- Geography: Flanders, Belgium (https://hyper.ai/en/datasets/15948).
- Size: BTSC cropped sets BelgiumTSC_Training.zip (171.3 MB) and BelgiumTSC_Testing.zip (76.5 MB); reducedSetTS.txt lists "62 used traffic signs types" (https://btsd.ethz.ch/shareddata/, https://btsd.ethz.ch/shareddata/BelgiumTS/reducedSetTS.txt). Image counts: 4,575 training / 2,520 test / 7,095 total, sizes 11x10 to 562x438, 62 classes per the Total Recall paper (https://arxiv.org/pdf/1808.10524). The often quoted 4,591 / 2,534 split could not be verified from a primary source in this session (the IJCNN 2013 PDF link at people.ee.ethz.ch returned an HTML page, section 9).
- Full BelgiumTS material: 8-camera annotation tars (camera00 to camera07), BelgiumTSD_annotations.zip "provided for detection evaluation", background image sets, and four multi-camera test sequences (3,001 / 6,001 / 2,001 / 4,001 frames x 8 cameras) with poses and calibration (https://btsd.ethz.ch/shareddata/). DefinedTS has 210 official sign patterns (https://btsd.ethz.ch/shareddata/BelgiumTS/structure.txt).
- License: no license text; "If you use the material from this dataset, please cite one of the following publications" (https://btsd.ethz.ch/shareddata/). Redistribution rights unclear.
- Access: direct download, no registration (https://btsd.ethz.ch/shareddata/).
- Baselines: 99.17% top-1 (Total Recall, 2018) vs 98.34% HOG and 98.17% OneCNN (https://arxiv.org/pdf/1808.10524).

### 1.3 rMASTIF / FER-MASTIF TS2010 (Croatia)

- Origin: MASTIF project, FER University of Zagreb (Segvic, Brkic, Kalafatic et al.); three datasets TS2009 (around 6,000 signs, cropped images only), TS2010 (around 3,000 signs, full images + .seq annotations + annotated video), TS2011 (around 1,000 signs); "Each sign is annotated 4-5 times at different distances from the car", TS2009 is interlaced (https://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml).
- License (quoted): "The datasets may be freely used in academic research." (https://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml). Non-academic use not covered.
- Class count and exact split sizes of the "rMASTIF" classification variant: not verified (the CCVW 2014 paper page gives no numbers) (https://www.fer.unizg.hr/crv/ccvw.2014.0016, https://www.bib.irb.hr/857781).

### 1.4 TSRD, Chinese Traffic Sign Recognition Database (China)

- Origin: CAS Institute of Automation (NLPR) traffic data pages; the official pages http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html timed out in this session (section 9).
- Size per the HF mirror: about 4,170 training and 1,994 test images, about 58 classes, annotation format "filename;width;height;x1;y1;x2;y2;class" (https://huggingface.co/datasets/EvanDu/Traffic_Sign_Recogntion_Database/raw/main/README.md); the mirror carries an "mit" license tag chosen by the uploader (https://huggingface.co/api/datasets/EvanDu/Traffic_Sign_Recogntion_Database?full=true). Official license: not verified.

### 1.5 CURE-TSR, Challenging Unreal and Real Environments for Traffic Sign Recognition (Belgium real + Unreal Engine synthetic) [IMPORTANT]

- Origin: OLIVES Lab, Georgia Tech (Temel, Kwon, Prabhushankar, AlRegib), NeurIPS 2017 workshop on ML for ITS (https://arxiv.org/pdf/1712.02463, https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md).
- Content: crops from the CURE-TSD video dataset ("around 1.7 million real-world and simulator images with more than 2 million traffic sign instances"); "around 2.2 million traffic sign images in the CURE-TSR dataset"; real sequences come from BelgiumTS videos, synthetic ("unreal") sequences from Unreal Engine 4; challenge effects applied with Adobe After Effects (https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md). Paper table: 2,206,106 images (https://arxiv.org/pdf/1712.02463).
- Classes: 14 sign types: speed_limit, goods_vehicles, no_overtaking, no_stopping, no_parking, stop, bicycle, hump, no_left, no_right, priority_to, no_entry, yield, parking (https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md).
- Challenge types (12) x levels (5): decolorization, lens blur, codec error, darkening, dirty lens, exposure, Gaussian blur, noise, rain, shadow, snow, haze; file name "sequenceType_signType_challengeType_challengeLevel_Index.bmp" with sequenceType 01 real, 02 unreal (same README). Sign crops range from about 3x7 to 206x277 pixels (https://arxiv.org/pdf/1712.02463).
- Splits used in the paper: 49 video sequences, about 70/30 train/test by sign; 7,292 challenge-free real training images; 3,334 test images per challenge type and level (200,040 total) (https://arxiv.org/pdf/1712.02463).
- Baselines: 7 methods (intensity, colour, HoG features with softmax or RBF-SVM, plus a shallow CNN); codec error is the most damaging challenge (at least 30% drop at level 1); adding unreal challenge images improved accuracy in 72 of 84 method-challenge combinations (https://arxiv.org/pdf/1712.02463).
- License and access (three different answers): GitHub README: fill out a Google form "and agree the conditions of use" (https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md); the repository code is MIT, 22 stars, last commit September 2021 (https://img.shields.io/github/license/olivesgatech/CURE-TSR.json, https://img.shields.io/github/stars/olivesgatech/CURE-TSR.json, https://img.shields.io/github/last-commit/olivesgatech/CURE-TSR.json). IEEE DataPort: listed as Open Access, 4.6 GB zip, requires an IEEE account login, DOI 10.21227/n4xw-cg56, no license stated (https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition). Zenodo record 3903066 by the same four authors: CURE-TSR.zip 4,944 MB, license cc-by-4.0, DOI 10.5281/zenodo.3903066, published 2020-06-22 (https://zenodo.org/api/records/3903066). The Zenodo CC BY 4.0 copy is the one OpenTSR can rely on.
- Known problems: the "real" images are BelgiumTS crops, so BTSC and CURE-TSR are not independent; only 14 coarse classes (all speed limits collapsed into one class); bmp files, 4.9 GB.

### 1.6 ETSD, European Traffic Sign Dataset (Belgium, Croatia, France, Germany, Netherlands, Sweden)

- Origin: Serna and Ruichek, "Classification of Traffic Signs: The European Dataset", IEEE Access 2018, DOI 10.1109/ACCESS.2018.2884826 (https://raw.githubusercontent.com/citlag/European-Traffic-Sings/master/README.md, https://citlag.github.io/publication/etsd).
- Size: "more than 80 000 images divided in 164 classes" in four Vienna Convention categories, assembled from public datasets of six countries plus French signs captured in Belfort with the UTBM vehicle (https://raw.githubusercontent.com/citlag/European-Traffic-Sings/master/README.md).
- License/access (quoted): "The European Traffic Sign Dataset is free for academic purposes and available upon request" via a Google form (same README). The GitHub repo (code and weights) is Apache-2.0, 15 stars (https://img.shields.io/github/license/citlag/European-Traffic-Sings.json, https://img.shields.io/github/stars/citlag/European-Traffic-Sings.json). Not redistributable.

### 1.7 LISA Traffic Sign Dataset (USA)

- Origin: UC San Diego LISA lab (Mogelmose, Trivedi, Moeslund, IEEE T-ITS 13(4):1484-1497, 2012) (https://cvrr.ucsd.edu/lisa-traffic-signs-dataset).
- Size: 6,610 frames, 7,855 annotations, 47 US sign types, sign sizes 6x6 to 167x168 px, image sizes 640x480 to 1024x522, some colour some greyscale; annotations: sign type, position, size, occluded, on side road (same page; also 6,610 / 47 / 7,855 in the MTSD comparison table https://web3.arxiv.org/pdf/1909.04422v2).
- License (quoted): "released under academic license agreement" (https://cvrr.ucsd.edu/lisa-traffic-signs-dataset). The download page https://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html could not be reached (section 9). Note that LISA is a detection set with full frames; it is listed here because it is commonly used for cropped classification too. An "Augmented Lisa Traffic Sign Dataset for Rural Areas" exists on Zenodo under cc-by-nc-1.0 (record 17297772) (https://zenodo.org/api/records?q=%22traffic%20sign%22%20OR%20%22traffic%20signs%22%20OR%20%22road%20signs%22&size=25&page=1&type=dataset).

### 1.8 Regional classification and small sets (one or two lines each)

- Austria, ATSD (Austrian Highway Traffic Sign Data Set), Data (MDPI) 8(1):16, 2023: 7,454 scene images at 1596x1196 recorded in 2014 with 27,521 annotations in 108 classes; ATSD-Signs 20,683 patches, 60 classes used by the classifiers; public train 4,068 / test 1,443 scenes plus 1,943 internal images that are not downloadable; YOLOv4-tiny 85.39% mAP, classifier 97.61%; license quoted "Permissive license that allows to use the data set free of charge" with terms at contentportal.asfinag.at (terms PDF not fetched, section 9); DOI https://doi.org/10.53177/ATSD resolves to https://contentportal.asfinag.at/ATSD/v1_0 (https://www.mdpi.com/2306-5729/8/1/16, curl of the DOI).
- UK, "English roadsigns" (Zenodo 7831751, CC BY 4.0): 55,000 crops of 256x256 in 31 classes sampled from 213 phone videos around Manchester, 4.83 GB (https://zenodo.org/api/records/7831751). UK_Traffic_Sign_Inspection_Dataset on HF (MIT tag, 10K to 100K rows, no README body) (https://huggingface.co/api/datasets/PastaEvangelists/UK_Traffic_Sign_Inspection_Dataset?full=true).
- France, Panoramax/classified_fr_road_signs (HF, license etalab-2.0): "66000+ detected road signs" cropped from Panoramax street-level pictures, "250+ classes" named by official French type codes (Axx danger, Bxx restriction, Cxx information ...), plus fake classes for false positives (https://huggingface.co/datasets/Panoramax/classified_fr_road_signs/raw/main/README.md, https://huggingface.co/api/datasets/Panoramax/classified_fr_road_signs?full=true). Panoramax/fr_road_sign_subsign (etalab-2.0) and classified_nl / classified_be exist too (https://huggingface.co/api/datasets?search=road%20sign&limit=100).
- Germany, Panoramax/classified_de_road_signs (HF, CC BY-SA 4.0): "20000+ photos of DE road signs in 130+ classes", crops made with the Panoramax detector, some classes padded with signs from other European countries (file prefix BE_, FR_) (https://huggingface.co/datasets/Panoramax/classified_de_road_signs/raw/main/README.md, https://huggingface.co/api/datasets/Panoramax/classified_de_road_signs?full=true).
- Germany, German Traffic Sign Anomaly Detection Dataset (Zenodo 21513624, CC BY 4.0, 2026-07-24): 369 real smartphone photos of 71 sign categories from Schleswig-Holstein with binary anomaly labels (dirt, fading, stickers, graffiti, vegetation), plus trainval_synthetic.zip (3,051 MB) and templates.zip (https://zenodo.org/api/records/21513624). Relevant for robustness and adversarial-sticker work.
- India: kannanwisen/Indian-Traffic-Sign-Classification (HF, CC BY 4.0, 1K to 10K images) and Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset (license tag "other", under 1K files) (https://huggingface.co/api/datasets/kannanwisen/Indian-Traffic-Sign-Classification?full=true, https://huggingface.co/api/datasets/Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset?full=true). Indian Signboard Image Dataset (Zenodo 14864212, CC BY 4.0) (Zenodo search above).
- Persia/Iran: the Persian Traffic Sign Dataset is used as a benchmark (99.8% reported) in arXiv 2311.06651 (https://arxiv.org/abs/2311.06651); access and license not verified.
- Arabic-script countries: Moroccan traffic sign dataset on HF (afyfbadr/moroccan-traffic-sign-dataset, 12,869 files, no license tag) (https://huggingface.co/api/datasets/afyfbadr/moroccan-traffic-sign-dataset?full=true); IEEE DataPort entries "Moroccan Traffic Sign Recognition" and "TUROS-TS: a Tunisian road scene videos traffic sign dataset" (titles seen in search results only, not fetched).
- Vietnam: Kaggle maitam/vietnamese-traffic-signs is CC BY-SA 4.0 per the VQA derivative card (https://huggingface.co/datasets/Anakonkai/vietnamese-traffic-sign-vqa/raw/main/README.md); HF detection sets thanhhiepvos/vietnam_traffic_sign (cc-by-sa-3.0 tag, 6,437 files), star092304/Traffic-sign-detection-VietNam (cc-by-4.0 tag), ghbihuy/vietnam_traffic_sign (mit tag, 250 rows) (HF API).
- Turkey: no Turkish set surfaced in the HF API search or web searches run here; not verified.
- Russia: Dognellaf/russian-road-signs (HF, CC BY 4.0): 43,851 JPEG frames manually labelled in YOLO format, 89 classes (signs plus traffic lights) (https://huggingface.co/datasets/Dognellaf/russian-road-signs/raw/main/README.md). See also RTSD in section 2.
- Malaysia: Extended Malaysian Traffic Sign Dataset, EMTD (Zenodo 1217105, CC BY 4.0): 1,413 images, 66 categories, 7.47 GB (https://zenodo.org/api/records/1217105).
- Brazil: BR-RTSD40k (Zenodo 23025922, CC BY-SA 4.0, 11 GB, 2026-09-30), not inspected further; HumynLabs/Brazilian_Road_Signs_Dataset (HF, CC BY 4.0, under 1K files) (Zenodo search above; HF API).
- Korea: HumynLabs/Korean-Signs-Image-Dataset (HF, CC BY 4.0, under 1K files) (HF API); Roboflow picmin/koreanroadsign, 404 images, CC BY 4.0, 2025-02-19 (https://universe.roboflow.com/picmin/koreanroadsign). Larger Korean sets are in section 2.
- Canada: RDLTechworks/quebec-traffic-signs (HF, CC BY-NC-ND 4.0, under 1K) (HF API); CanDOR-99 in section 2.
- Lithuania, Indonesia, Cambodia: Ramuneri/lithuanian-road-signs (no license), ilmagita/indonesia-traffic-signs, Mongkolek10/cambodia-traffic-sign (tiny) (https://huggingface.co/api/datasets?search=traffic%20sign&limit=100).
- Cross-country: guoyang1/Cross-Regional_Traffic_Sign_Dataset (HF, MIT tag, 2 files, not inspected).

---

## 2. Detection benchmarks (full frames with boxes)

### 2.1 GTSDB, German Traffic Sign Detection Benchmark (Germany) [IMPORTANT]

- Origin: INI Bochum, IJCNN 2013 competition. Paper: Houben, Stallkamp, Salmen, Schlipsing, Igel, "Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark", IJCNN 2013 (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf). Cite per the site's 2013 notice (https://benchmark.ini.rub.de/).
- Geography and capture: near Bochum, Germany, spring and autumn 2010, daytime and dusk, urban, rural and highway; Prosilica GC 1380CH, 1360x1024 clipped to 1360x800 (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf).
- Size: 900 images, 600 training and 300 evaluation images, PPM, 0 to 6 signs per image, sign sizes 16x16 to 128x128 (https://benchmark.ini.rub.de/gtsdb_dataset.html). Sign count: 1,206 (846 training + 360 evaluation) per the paper (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf); 852 boxes per Dataset Ninja's parse of the released ground truth (prohibitory 396, other 186, danger 156, mandatory 114; 394 images have no annotation) (https://datasetninja.com/gtsdb) and 852 per the MTSD paper's comparison table (https://web3.arxiv.org/pdf/1909.04422v2). The two figures do not agree; the released gt.txt is what matters for OpenTSR.
- Categories: the competition scored three categories, prohibitive, danger and mandatory; signs outside them "may appear in images but aren't scored" (https://benchmark.ini.rub.de/gtsdb_dataset.html); the paper names a fourth category "other" (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf). The ground truth uses the 43 GTSRB class IDs (43 classes listed by Dataset Ninja, e.g. keep right, priority road, speed limit 50 ...) (https://datasetninja.com/gtsdb).
- Annotation format: semicolon CSV "filename; left; top; right; bottom; classID" (https://benchmark.ini.rub.de/gtsdb_dataset.html).
- Test ground truth: training data public 2012-12-01, test set public 2013-02-18, and on 2013-07-15 "We put together the test and training dataset with full annotations for download" (https://benchmark.ini.rub.de/gtsdb_news.html, https://datasetninja.com/gtsdb). Original files on sid.erda.dk: FullIJCNN2013.zip, TrainIJCNN2013.zip, TestIJCNN2013.zip, gt.txt (https://datasetninja.com/gtsdb); the sid.erda.dk page itself could not be fetched (section 9).
- License: the GTSDB page contains no license text (https://benchmark.ini.rub.de/gtsdb_dataset.html); Dataset Ninja labels it "custom" (https://datasetninja.com/gtsdb). The site-wide statement that the data is freely available and the request to cite (section 1.1) are the only terms.
- Competition results (AUC of precision-recall): wgy@HIT501 100% prohibitive / 99.91% danger / 100% mandatory; visics 100 / 100 / 96.98%; LITS1 100 / 98.85 / 92%; BolognaCVLab 99.98 / 98.72 / 95.76%; baselines Viola-Jones 90.81 / 46.26 / 44.87%, HOG + LDA 70.33 / 35.94 / 12.01%; 18 teams, over 110 submissions (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf).
- Hugging Face mirrors: GlowBond/GTSDB (one archive.zip, README is one line, no license tag, created 2026-03-18, 276 downloads) (https://huggingface.co/api/datasets/GlowBond/GTSDB?full=true, https://huggingface.co/datasets/GlowBond/GTSDB/raw/main/README.md); keremberke/german-traffic-sign-detection (Roboflow export of 545 images in COCO format, train 383 / valid 108 / test 54, 43 labels, card license "CC BY 4.0" assigned by the Roboflow uploader Mohamed Traore, created 2023-01-16, 644 downloads) (https://huggingface.co/datasets/keremberke/german-traffic-sign-detection/raw/main/README.md, https://huggingface.co/api/datasets/keremberke/german-traffic-sign-detection?full=true). A resized 128x128 variant is on Zenodo under CC BY 4.0 (record 3941507) (Zenodo search above). None of these CC tags were issued by INI.
- Known problems: tiny (900 images, 852 or 1,206 signs); the 300-image test set is far too small for per-class detection metrics on 43 classes; same source footage as GTSRB.

### 2.2 Tsinghua-Tencent 100K, TT100K (China) [IMPORTANT]

- Origin: Zhu, Liang, Zhang, Huang, Li, Hu, "Traffic-Sign Detection and Classification in the Wild", CVPR 2016; Tsinghua University graphics group with Tencent Street View (https://cg.cs.tsinghua.edu.cn/traffic-sign/).
- 2016 version: 100,000 images of 2048x2048 cut from 8192x2048 Tencent Street View panoramas; "Of these, 10000 contain 30000 traffic-signs in total"; each sign annotated with class label, bounding box and pixel mask (ellipse/polygon); classes with fewer than 100 instances ignored for training, leaving 45 classes (https://cg.cs.tsinghua.edu.cn/traffic-sign/0682.pdf). Annotation files list 221 categories (https://docs.ultralytics.com/datasets/detect/tt100k, https://web3.arxiv.org/pdf/1909.04422v2 Table 1: 221 classes, 26,349 signs). Splits: train 6,105 / test 3,071 / "other" 7,641 images (Ultralytics maps "other" to val); 16,817 images with annotations in the common redistribution (https://docs.ultralytics.com/datasets/detect/tt100k). The remaining roughly 82,000 "no sign" images are shipped as five extra parts of 16.8 to 17.9 GB each (16,555 / 16,638 / 16,582 / 16,632 / 15,690 images) (https://cg.cs.tsinghua.edu.cn/traffic-sign/).
- 2021 version: "Tsinghua-Tencent 100K Annotations 2021 (with more classification)", file tt100k_2021.zip; the site states "The 2021 version of the Tsinghua-Tencent 100K Annotation is a complement to the 2016 version. The 2021 version could NOT be used independently. It should be merged into the 2016 version, if more data is required." (https://cg.cs.tsinghua.edu.cn/traffic-sign/). Dataset Ninja's parse of the 2021 release: 16,817 images, 27,359 annotated objects, 232 classes, train 6,105 / test 3,071 / other 7,641, 6,219 images unlabeled (https://datasetninja.com/tt100k-2021).
- License (quoted from the official page): "[Tsinghua-Tencent 100K dataset is under Creative Commons Attribution-NonCommercial (CC-BY-NC) license.]" and "If you have any questions about the code or dataset, or request for commercial use, please contact Song-Hai Zhang (shz@tsinghua.edu.cn)". Download requires entering "name", "Email", "Your orgnization" in a form (https://cg.cs.tsinghua.edu.cn/traffic-sign/). Ultralytics and Dataset Ninja label it CC BY-NC 2.0 (https://docs.ultralytics.com/datasets/detect/tt100k, https://datasetninja.com/tt100k-2021); the official page does not give a version number.
- Baselines: CVPR 2016 Fast R-CNN style network with 81.6 mAP per the MTSD paper's citation; MTSD two-stage detector + classifier reaches 89.9 mAP on TT100K, 93.4 / 95.7 with MTSD pre-training (https://arxiv.org/pdf/1909.04422). The Ultralytics page gives no YOLO results (https://docs.ultralytics.com/datasets/detect/tt100k).
- Mirrors on Hugging Face: PrashantDixit0/TT-100K (license tag cc-by-2.0, which contradicts the official NC license; 1,117 downloads), Genius-Society/tt100k (cc-by-nc-4.0 tag; train 6,105 / validation 1,097 / test 3,071), finedet/tt100k (cc-by-nc-4.0), GlowBond/TT100K_Annotationed (no license) (https://huggingface.co/api/datasets?search=TT100K&limit=100 and per-dataset API calls). An augmentation method with "iteration version datasets based on TT100K" is described in arXiv 2303.18037 (https://arxiv.org/abs/2303.18037).
- Known problems: NC license; registration form; most of the 221 classes have very few instances (only 45 with more than 100); the "other" split is ambiguous; the 2021 annotations cannot be used standalone.

### 2.3 Mapillary Traffic Sign Dataset, MTSD (global) [IMPORTANT]

- Origin: Ertler, Mislej, Ollmann, Porzi, Neuhold, Kuang (Mapillary, now Meta), "The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale", ECCV 2020; announced 2019-06-27 (https://arxiv.org/pdf/1909.04422, https://blog.mapillary.com/update/2019/06/27/mapillary-traffic-sign-dataset.html, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf).
- Size: about 100K images; fully annotated 52,453 images with 257,543 boxes; partially annotated 47,547 images with 87,358 automatically generated labels; fully annotated splits train 36,589 / val 5,320 / test 10,544 with test annotations withheld; resolution from about 1 MP to over 16 MP, over 200 camera manufacturers, 1,138 360-degree panoramas (https://arxiv.org/pdf/1909.04422). Table 1 of the paper: 100,000 images, 313 classes, 325,172 boxes; train+val 41,907 images and 206,388 boxes (https://web3.arxiv.org/pdf/1909.04422v2).
- Classes: 313 classes plus "other-sign" in the arXiv version (https://arxiv.org/pdf/1909.04422); the ECCV camera-ready reports 400 classes derived from Wikimedia Commons templates grouped by appearance (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf). The released v2 annotations (mtsd_v2_fully_annotated) use the 400-class taxonomy: an HF mirror's id2labels.txt has 400 entries, 176 regulatory, 135 warning, 45 complementary, 44 information, named like "regulatory--maximum-speed-limit-10--g1" (https://huggingface.co/datasets/ThankGod/mapillary_traffic_sign_dataset/raw/main/id2labels.txt). Attributes: occluded, ambiguous, dummy, out-of-frame, included, exterior (https://arxiv.org/pdf/1909.04422). Annotation format: one JSON per image with width, height, ispano, objects[bbox, key, label, properties], panoramas with cross_boundary boxes; partially annotated set has correspondence keys (https://huggingface.co/datasets/NgTMDuc/cl_dataset/resolve/main/MTSD/annotations/mtsd_v2_fully_annotated/README.md).
- Geography: images sampled per country toward 20% North America, 20% Europe, 20% Asia, 15% South America, 15% Oceania, 10% Africa; "covering almost all habitable areas of the world"; no country list; Japan is not mentioned anywhere in the paper or the blog post (https://arxiv.org/pdf/1909.04422, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf, https://blog.mapillary.com/update/2019/06/27/mapillary-traffic-sign-dataset.html). Whether Japanese images are included is therefore not verified; the image keys are Mapillary image IDs, so geolocation could be recovered through the Mapillary API by a project that has the data.
- License: the LICENSE.txt shipped with the annotations reads in full: "This Mapillary Traffic Sign Dataset is provided under the Creative Commons Attribution NonCommercial Share Alike (CC BY-NC-SA) license available here: https://creativecommons.org/licenses/by-nc/4.0/legalcode" (the text says NC-SA but links the BY-NC legal code) (https://huggingface.co/datasets/NgTMDuc/cl_dataset/resolve/f8fa651855ae2cf72a47cdb7a45ef044e03ef8f8/MTSD/annotations/LICENSE.txt). The paper says "freely available for academic research" and that the underlying images are CC BY-SA on Mapillary (https://arxiv.org/pdf/1909.04422); the blog post says there is "a commercial edition and a research edition" (https://blog.mapillary.com/update/2019/06/27/mapillary-traffic-sign-dataset.html). The official page www.mapillary.com/dataset/trafficsign is JavaScript-rendered and could not be read (section 9).
- Access: sign in to Mapillary and download from the dataset page (not verified in this session, see section 9).
- Baselines: Faster R-CNN + FPN ResNet101 88.44 AP at 4000 px max side (81.80 at 2048 px); two-stage detector + classifier 83.4 mAP over 313 classes (https://arxiv.org/pdf/1909.04422).
- Hugging Face mirrors and their license status: ThankGod/mapillary_traffic_sign_dataset (3,813 train images, 3.2 GB, 400-class labels, no license tag, 647 downloads), ThankGod/mapillary_traffic_sign_dataset_cleaned (train 23,360 / test 1,297 / validation 1,297, 21.8 GB, no license tag), ThankGod/mapillary_traffic_sign_dataset_recommended (25,956 train images, 21.8 GB, license tag "mit") (https://huggingface.co/api/datasets/ThankGod/mapillary_traffic_sign_dataset?full=true and the two sibling API calls). The MIT tag contradicts the NC license of the source; all three redistribute MTSD images, which the NC-SA license allows only non-commercially and with the same license, so OpenTSR should treat them as "point to, do not re-host". sparshgarg57/mapillary_traffic_signs is a different thing: Mapillary Vistas validation images with fine-grained sign labels for VLM evaluation (arXiv 2508.02047), no license tag (https://huggingface.co/datasets/sparshgarg57/mapillary_traffic_signs/raw/main/README.md). NgTMDuc/cl_dataset contains a copy of the MTSD v2 annotations and LICENSE.txt (https://huggingface.co/api/datasets/NgTMDuc/cl_dataset/tree/main/MTSD/annotations).
- Known problems: NC license; test labels withheld; long-tailed taxonomy (66,138 of 257,543 boxes carry an in-taxonomy class per the paper); class names encode appearance groups (g1, g2 ...) rather than countries, so country-specific evaluation needs the image geolocation.

### 2.4 DFG Traffic Sign Dataset (Slovenia) [IMPORTANT]

- Origin: Tabernik and Skocaj, "Deep Learning for Large-Scale Traffic-Sign Detection and Recognition", IEEE T-ITS 2019, DOI 10.1109/TITS.2019.2913588; ViCoS Lab, University of Ljubljana, with DFG Consulting (https://vicos.si/resources/dfg).
- Size: 6,957 images (train 5,254 / test 1,703), 6,758 at 1920x1080 and 199 at 720x576; 13,239 tightly annotated polygon instances of at least 30 px plus 4,359 loosely annotated boxes under 30 px marked "ignore"; 200 categories with at least 20 instances each; augmentation set with 8,775 additional images and over 30,000 instances; COCO JSON; faces and plates blurred; revisions v1.01 (2019-04-25) and v1.02 (2019-06-14) (https://vicos.si/resources/dfg).
- License (quoted): "The dataset is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"; for commercial use contact Danijel Skocaj (https://vicos.si/resources/dfg).
- Access: direct links go.vicos.si/dfgannot, /dfgaugannot, /dfgaugannotdiff, /dfgimages (JPEGImages.tar.bz2), /dfgcat; file sizes not stated (https://vicos.si/resources/dfg).
- Baselines: the T-ITS paper uses Mask R-CNN; numbers not on the resource page (not verified).
- Known problems: NC-SA; Slovenian signs only; many of the 200 classes are guide and information panels with text.

### 2.5 Swedish Traffic Signs, STS (Sweden)

- Origin: Larsson and Felsberg, SCIA 2011 (LNCS 6688, pp. 238-249); Linkoeping University CVL (https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/).
- Size: "more than 20 000 images" from over 350 km of highway and city driving with a 1.3 MP dashboard camera; every fifth frame labelled; 3,488 traffic signs; 7 classes (pedestrian crossing, designated lane right, no standing or parking, priority road, give way, 50 kph, 30 kph) with visibility (occluded, blurred, visible) and road status (on the travelled road or a side road) attributes (https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/). The MTSD comparison table lists STS as 3,777 images, 20 classes, 5,582 signs (https://web3.arxiv.org/pdf/1909.04422v2); the figures differ.
- License: the old summary page says "Without prior approval from CVL, the Traffic Signs Database, in whole or in part, may not be modified or used for commercial purposes." (https://www.cvl.isy.liu.se/research/trafficSigns/); the current page only requires citation (https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/). The authors re-uploaded the data to Zenodo on 2026-04-20 as "traffic-signs_dataset.zip" (15.8 GB) under cc-by-4.0, DOI 10.5281/zenodo.19666339 (https://zenodo.org/api/records/19666339). The Zenodo CC BY 4.0 copy supersedes the old restriction for OpenTSR purposes (confirm with the authors before redistributing).
- Access: Set1/Set2 parts and annotations.txt from CVL, or the Zenodo zip (same URLs).

### 2.6 BelgiumTS detection, BTSD (Belgium)

- Same material as section 1.2; BelgiumTSD_annotations.zip (485,645 bytes) is "provided for detection evaluation, it uses the same material and splits"; annotation files training_GT.txt, testing_2D_GT.txt, testing_comparison_GT.txt are pole annotations relative to the 8-camera image folders (https://btsd.ethz.ch/shareddata/, https://btsd.ethz.ch/shareddata/BelgiumTS/structure.txt). The MTSD table lists BelgiumTS with 108 classes and 8,851 signs (https://web3.arxiv.org/pdf/1909.04422v2). Image counts of the detection split: not verified (section 9). License: citation request only (section 1.2).

### 2.7 RTSD, Russian Traffic Sign Dataset (Russia)

- Origin: Graphics and Media Lab, Moscow State University (Shakhuro and Konushin, Computer Optics 2016, in Russian) (https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html).
- Size: detection train 47,639 images / 80,277 signs, test 11,389 images / 25,232 signs; classification train 79,896 / test 25,613 samples; 205 classes in total, 106 in training, 99 appear only in the test set (rare classes); 1280x720 to 1920x1080 frames from a dashcam at 5 fps in all seasons and weather (https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html). The MTSD table gives 179,138 frames, 156 classes, 104,358 signs (https://web3.arxiv.org/pdf/1909.04422v2).
- License: none stated; download from a Yandex Disk link https://yadi.sk/d/TX5k2hkEm9wqZ (https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html). Redistribution rights unclear.

### 2.8 CCTSDB and CCTSDB2021, CSUST Chinese Traffic Sign Detection Benchmark (China)

- Origin: Zhang Jianming et al., Changsha University of Science and Technology; CCTSDB extends the earlier CTSD with more highway images (https://arxiv.org/pdf/1902.06857 related work). CCTSDB 2021 paper: Human-centric Computing and Information Sciences 12:23, 2022, DOI 10.22967/HCIS.2022.12.023 (https://raw.githubusercontent.com/csust7zhangjm/CCTSDB2021/master/README.md).
- CCTSDB2021 size: 17,856 images in training set plus positive test set; 16,356 training images, 1,500 positive test images (numbered 18992-20491), 500 negative images; three categories mandatory, prohibitory, warning; XML and TXT annotations; test subsets grouped by weather/lighting and by sign size (same README). Zenodo mirror: dataset.rar 1,555.6 MB, "approximately 40,000 annotated traffic signs", license gpl-3.0, DOI 10.5281/zenodo.17152727 (https://zenodo.org/api/records/17152727).
- License: the CCTSDB2021 GitHub repo is GPL-3.0 (102 stars) (https://img.shields.io/github/license/csust7zhangjm/CCTSDB2021.json, https://img.shields.io/github/stars/csust7zhangjm/CCTSDB2021.json); the original CCTSDB repo has no license (271 stars, last commit July 2024) and its README could not be fetched on master or main (https://img.shields.io/github/license/csust7zhangjm/CCTSDB.json, https://img.shields.io/github/stars/csust7zhangjm/CCTSDB.json). A GPL license on image data is unusual; treat as "ask the authors".
- Access: Google Drive and Baidu links in the README (same README).
- Known problems: only three super-classes, no fine type.

### 2.9 CTSD, Chinese Traffic Sign Dataset (China)

- 1,100 images (700 training / 400 test) in three classes (prohibitory, mandatory, danger); introduced by Yang et al. 2016 (https://arxiv.org/pdf/1804.10428, https://arxiv.org/pdf/1902.06857). Hosting and license: not verified (official NLPR page unreachable, section 9).

### 2.10 Korea

- KRID, Korean Road Infrastructure Dataset (Data, MDPI, 10(3):36, March 2025): 200,000 annotated frames selected from about 2 million at 3328x1872; 2,676,583 objects (2,031,633 boxes, 644,950 polygons) in 34 road facility classes including road signs, safety signs, road nameplates, guideposts, traffic signals; license CC BY-NC-SA per the data availability statement; distributed via AI Hub (Korean researchers) and GitHub kimhyeongbok/korean-road-infrastructure-dataset (https://mdpi.com/2306-5729/10/3/36). Sign type is not labelled at the fine level (the page gives no sign taxonomy).
- SAIT Traffic Landmark Dataset (Samsung Advanced Institute of Technology, IEEE DataPort, DOI 10.21227/gar1-z597): about 150,000 BMP images from Seoul, Suwon, Hwaseong, Yongin, Seongnam with about 470,000 landmarks annotated as reference points (not boxes); 6 classes: warning sign, prohibition sign, mandatory sign, supplementary sign, traffic light, road marking; 301.72 GB; requires IEEE DataPort subscription; license not stated (https://ieee-dataport.org/documents/sait-traffic-landmark-dataset).
- Kakao Mobility announced a Korea-specific autonomous driving training dataset in 2025 (title seen in search results only, not fetched).

### 2.11 ZOD, Zenseact Open Dataset (14 European countries)

- Origin: Zenseact AB (Alibeigi et al., ICCV 2023) (https://huggingface.co/datasets/Zenseact/ZOD/raw/main/README.md, https://arxiv.org/pdf/2305.02008).
- Traffic signs: class "TrafficSign" with 2D boxes (3D when visible in LiDAR too); taxonomy of 6 categories (Mandatory, Priority, Prohibitory, RoadType, Special, Warning) and 156 granular classes; attributes OcclusionRatio, IsForConstruction, IsElectronic, ContainsInnerSigns, ContentContainsText, IsForEgoRoad, IsForEgoLane, IsSticker; 446k unique labelled sign instances ("about twice the 206k in MTSD"); "annotations for traffic signs and ego-road are unavailable for some Frames" (https://arxiv.org/pdf/2305.02008).
- Geography: 14 European countries; Sweden, Germany, France, Italy named; no Japan (https://arxiv.org/pdf/2305.02008).
- License: CC BY-SA 4.0, which the paper describes as permitting research and commercial use (https://arxiv.org/pdf/2305.02008); HF card license tag cc-by-sa-4.0 (https://huggingface.co/api/datasets/Zenseact/ZOD?full=true). Access through the devkit at zod.zenseact.com (download flow not verified).
- Known problems: European signs only; sign labels missing on some frames; large multimodal download.

### 2.12 Canada, CanDOR-99 (Mapillary-derived)

- Zenodo record 20739255 (2026-06-18): 95,314 images with 103,521 COCO annotations in 99 classes "organized by Mapillary taxonomy" (37 regulatory, 45 warning, 14 complementary, 3 information), collected through the Mapillary API from Ottawa, Toronto, Vancouver, Montreal, Quebec City, Calgary, Victoria between September and December 2025, annotations verified or corrected manually or by a classifier; license cc-by-sa-4.0; 117.9 GB (https://zenodo.org/api/records/20739255). This is the template for how a Japanese set could be built from Mapillary (section 3.5).

### 2.13 Multi-purpose driving datasets: do they have a traffic sign class, and is the type labelled?

| Dataset | Sign class | Type labelled | License | Source |
|---|---|---|---|---|
| Argoverse 2 Sensor (USA, 6 cities, 1,000 x 15 s) | 3D cuboids: SIGN, STOP_SIGN, MOBILE_PEDESTRIAN_SIGN, MESSAGE_BOARD_TRAILER, TRAFFIC_LIGHT_TRAILER among 30 categories | only stop sign | "All datasets are released under the CC BY-NC-SA 4.0 license" | https://arxiv.org/pdf/2301.00493, https://argoverse.org/av2.html |
| Waymo Open Perception (USA, 2,030 x 20 s) | 3D LiDAR label class "Signs" (one of 4 classes, 12.6M boxes); "Sign" also in the 28-class 2D panoptic labels | no | "licensed for non-commercial use" | https://waymo.com/open/about |
| BDD100K (USA, 100k images) | detection class "traffic sign" (274,594 objects) among 10 classes | no (see BDD100K-MUTCD below) | custom: free "for educational, research, and not-for-profit purposes", commercial via BAIR Commons or UC Berkeley OTL | https://datasetninja.com/bdd100k |
| BDD100K-MUTCD (IEEE DataPort, 2026-07) | 38-class MUTCD taxonomy for 15,628 verified sign crops over 80,000 BDD100K images, Gemini-labelled then partly human-checked | yes (38 classes) | subscription; images must come from BDD100K | https://ieee-dataport.org/documents/bdd100k-mutcd-fine-grained-human-verified-traffic-sign-type-annotations-bdd100k-dataset |
| Cityscapes (Germany, 50 cities) | semantic class 'traffic sign' (id 20, trainId 7) | no | non-commercial; "That you do not distribute this dataset or modified versions" | https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py, https://www.cityscapes-dataset.com/license/ |
| Mapillary Vistas (global, 25,000 images, 124 classes in v2.0) | Traffic Sign (Front) 93,287 objects in 17,178 images, Traffic Sign (Back), Traffic Sign - Direction (Front/Back), Traffic Sign - Parking, Traffic Sign - Temporary (Front/Back), Traffic Sign - Ambiguous, Traffic Sign Frame | no (only sign function groups) | Dataset Ninja: CC BY-NC-SA 4.0 (research); the ICCV paper mentions a commercial edition; official page not fetched | https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/mmseg/datasets/mapillary.py, https://datasetninja.com/mapillary-vistas-dataset, https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf, https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html |
| nuImages / nuScenes (Boston, Singapore) | no traffic sign category in the annotation instructions (only "signboard" as an attached-object note) | no | not verified (terms page is JavaScript-only) | https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/docs/instructions_nuimages.md |
| A2D2 (Germany, 41,277 labelled frames) | semantic classes "Traffic sign 1-3" and "Traffic signal 1-3" | no | CC BY-ND 4.0 "permitting commercial use subject to the terms of the license" (no derivatives) | https://arxiv.org/abs/2004.06320 and PDF |
| IDD, India Driving Dataset (10,003 seg. images, 46,588 det. images) | label 'traffic sign' (id 24) and 'traffic light' | no | not fetched (site requires JavaScript) | https://raw.githubusercontent.com/AutoNUE/public-code/master/helpers/anue_labels.py, https://idd.insaan.iiit.ac.in/dataset/details/ |
| DDAD (TRI; USA + Tokyo/Odaiba) | none (depth benchmark only) | no | CC BY-NC-SA 4.0 | https://raw.githubusercontent.com/TRI-ML/DDAD/master/README.md |

### 2.14 aiMotive 3D Traffic Light and Traffic Sign Dataset (USA + Hungary)

- Origin: aiMotive (Budapest), paper "Accurate Automatic 3D Annotation of Traffic Lights and Signs for Autonomous Driving" (https://arxiv.org/html/2409.12620v3, https://raw.githubusercontent.com/aimotive/aimotive_tl_ts_dataset/main/README.md).
- Size: 220 sequences of about 15 s, about 50,000 auto-annotated frames, about 320,000 traffic light and about 550,000 traffic sign 3D boxes; California and Hungary; four cameras (Sony IMX490, 2896x1876); per-frame 3D boxes, light state, sign type, occlusion, sign text via Google Vision API (https://arxiv.org/html/2409.12620v3, README). Number of sign types: not stated (same).
- License: paper says CC BY-NC-SA 4.0; README says "made freely available for non-commercial research purposes only"; GitHub shows no license file (https://arxiv.org/html/2409.12620v3, README, https://img.shields.io/github/license/aimotive/aimotive_tl_ts_dataset.json). Download: 107 GB zip on S3 or Kaggle (README).

### 2.15 Hugging Face detection sets named in the task

- Ericzzzzh/STIC-Traffic-Sign (created 2026-10-05, CC BY 4.0 tag, 199 downloads): "US MUTCD-style traffic-sign detection in YOLO format", two Roboflow exports: TS_1 7,204 images / 21,282 boxes / 7 classes (Guide Signs-Information, Guide Signs-Route, Makers, Regulatory Signs-BW, Regulatory Signs-Red, School, Warning Signs) at 1920x1080 stretched, train 5,764 / val 1,440; TS_2 1,564 images / 3,374 boxes / 5 classes at 640x640 with a 50% horizontal-flip offline augmentation; sources Roboflow Universe test-rtrlj/ts-27jkg and test-rtrlj/520-smb96 (https://huggingface.co/datasets/Ericzzzzh/STIC-Traffic-Sign/raw/main/README.md, https://huggingface.co/api/datasets/Ericzzzzh/STIC-Traffic-Sign?full=true). Coarse MUTCD function groups only; the CC BY 4.0 is the Roboflow uploader's claim.
- emb-ai/traffic-sign-bench (created 2026-08-22, license odbl, 2,498 downloads): not an image dataset. It is "Official per-sign SUMO maps for TrafficSignBench: real Moscow OSM layouts, 25 signs, 2500 maps", 80 train + 20 test maps per sign, geometry derived from OpenStreetMap under ODbL 1.0; a traffic-simulation benchmark of sign semantics, not vision (https://huggingface.co/datasets/emb-ai/traffic-sign-bench/raw/main/README.md).
- KeenForgeAI/TrafficSigns_YOLO-corrected (created 2026-10-14, license tag "other"): a leak-safe re-split of the "Traffic Signs Dataset for YOLOv8-YOLOv11 benchmarking" (Reveles-Martinez et al., Technologies 2025): 6,000 images of 640x640 taken with an Intel RealSense D435i on an indoor FIRA 1:10-scale autonomous-vehicle track, 6 classes (left_turn, straight_ahead, right_turn, dead_end, no_entry, stop), one box per image; 2,526 images are near-identical augmented copies that the upstream split leaked across train/val/test, re-split to 4,259 / 858 / 883; "Upstream license: none stated" (https://huggingface.co/datasets/KeenForgeAI/TrafficSigns_YOLO-corrected/raw/main/README.md). Toy-track data, not road imagery.

### 2.16 Derived and attack-oriented collections

- "Datasets Resources of Paper 'Hidden in Plain Signs: Realistic Sticker Attacks on Production Traffic Sign Recognition Systems'" (Zenodo 23527395, 2026-10-21, anonymous authors, CC BY-NC-SA 4.0, 6.59 GB): Dataset A combines DFG (867 images), TT100K (3,221), GTSDB (305), ItalianSigns (324), BelgiumTS (178), "Road Signs Detection [Italy]" (64) and 6,717 proprietary images into 18 classes; Dataset B combines MTSD (13,662), DFG (2,360), TT100K (6,605), GTSDB (357), ItalianSigns (361) into 30,226 images of 17 classes (speed limits, stop, no vehicles, no stopping, no parking, no entry), 640x640 crops, YOLO labels, with per-source adapter scripts; "Academic Use Only" (https://zenodo.org/api/records/23527395). Useful to OpenTSR as a recipe for a cross-country speed-limit subset, and as prior art for the maintainer's adversarial work.
- GTSRB backdoor variants and GTSRB-Concepts: section 1.1.

---

## 3. Japan (日本の道路標識) [IMPORTANT]

Short answer: as of 2026-10-31 there is no public Japanese traffic sign dataset of research quality
(thousands of frames, fine-grained 規制/指示/警戒/案内/補助 labels, documented license). Everything
found is listed below so that the maintainer can judge for himself. Searches were run in English
and Japanese (道路標識 データセット, 交通標識 データセット, 日本 標識 認識 公開データ) against the
Hugging Face API (datasets and models), Zenodo API, Roboflow Universe, Kaggle (via search engines),
IEEE DataPort, GitHub, J-STAGE, IEICE and IPSJ pages.

### 3.1 Community datasets (Roboflow Universe)

- ritsumeikan/japanese-road-signs ("Japanese Road Signs Computer Vision Project", creator Ritsumeikan, citation year 2022, "Updated 2 years ago"): 5 classes Information, Mandatory, Prohibitory, Speed Limit 30, Warning; license CC BY 4.0; image count and annotation type not shown on the page; no description (https://universe.roboflow.com/ritsumeikan/japanese-road-signs). Category coverage: a mix of 案内 or 指示 ("Information", "Mandatory"), 規制 ("Prohibitory", "Speed Limit 30") and 警戒 ("Warning") as coarse groups; no 補助.
- test-u58bl/japan-roadsign (2024-09): 3 classes not_speed, speed, with_auxiliary_sign; CC BY 4.0; "This project has no images available to display" (https://universe.roboflow.com/test-u58bl/japan-roadsign). Covers only speed-limit vs other 規制 plus presence of a 補助標識.
- test-u58bl/japan-roadsign-second-2 (2024-12): 536 images, object detection, 16 classes mixing vehicles and sign content (car, truck, Bus, bicycle, Bike, Taxi, Finish, Handcarts, horn, No_crossing, No_overtaking, No_turning, Pedestrians, Section, Start, Two_seater); CC BY 4.0 (https://universe.roboflow.com/test-u58bl/japan-roadsign-second-2). Sibling projects japan-roadsign-ver.2 and japan-roadsign-second exist in the same workspace (URLs seen in search results, not fetched).
- media-traverse-co-ltd/japan-road-scene (v1, 2023-04-24): 1,306 images (914 train / 261 valid / 131 test), classes not listed, license not stated; the page does not mention traffic signs (https://universe.roboflow.com/media-traverse-co-ltd/japan-road-scene/dataset/1).
- Assessment: all are hobby-scale, under-documented, and the CC BY 4.0 tags are the uploaders' choice; the source imagery rights are unknown.

### 3.2 Models trained on Japanese signs whose data is not released

- infra-intelligence/road-sign-yolox-tiny (Hugging Face model, Apache-2.0, created 2026-10-22, 34 downloads): a YOLOX-Tiny detector (about 5.03 M parameters, input 1280x736) that finds Japanese sign plates and sorts them into the five 道路標識令 categories 案内 (guide), 警戒 (warning), 規制 (regulatory), 指示 (indication), 補助 (supplementary), plus a hard_negative class; it does not distinguish sign types ("標識の種類（「一時停止」「最高速度」など）までは分けません"). Training data: Mapillary images from inside Japan (CC BY-SA 4.0), 15 regions, train 4,585 images (3,499 with signs; boxes: 案内 1,833, 警戒 363, 規制 3,878, 指示 634, 補助 1,100, ignore 2,048) and validation 992 images; evaluation on 596 images from 7 held-out regions with 988 boxes: recall 0.857, precision 0.877, AP50 0.902 (class-agnostic), AP50-95 0.700, per class AP50 案内 0.594 / 警戒 0.958 / 規制 0.937 / 指示 0.893 / 補助 0.748. Labels were drafted from Mapillary's automatic sign detections and adjusted by an "AI assistant" following written rules; labels were not checked image by image; "画像とラベル（データセット）は、この公開物には含めていません" (the dataset is not included in the release) (https://huggingface.co/infra-intelligence/road-sign-yolox-tiny/raw/main/README.md, https://huggingface.co/api/models/infra-intelligence/road-sign-yolox-tiny). Covers all five categories at the category level only; weights are Apache-2.0 and the eval JSON is included.
- University of Tokyo, Sekimoto Lab (Yang, Maeda, Sekimoto), poster "Vison-based Traffic Sign Detection and Localization in Tokyo Metropolitan Area": YOLOv4 detection (AP 94% at IoU 0.5) and VGG19/ResNet101/Swin-B classification on their own ZED stereo driving data in Tokyo; 50 test images with laser distances; no dataset release stated (https://sekilab.iis.u-tokyo.ac.jp/wp-content/uploads/poster_YANG.pdf).
- Osaka Institute of Technology (Kato, Nishiguchi, Mizutani, Hashimoto), FIT2018 H-027: 303,750 synthetic training images (and masks) of 225 sign classes from the MLIT road sign list of September 2016, rendered in Unity on a 3D city model (1,350 variations per class), evaluated with YOLOv2 on only 30 real photos and Street View crops; not public (https://www.ieice.org/publications/conference-FIT-DVDs/FIT2018/data/pdf/H-027.pdf). The same group's IEICE PRMU2018-133 (March 2019) compares a 3DCG sign dataset with a geometrically augmented real-photo set; sizes and availability not stated (https://ken.ieice.org/ken/paper/20190317h1kE/).
- Tokyo Denki University (Sekizawa, Nakajima), IEICE PRMU2018-89 (December 2018): training an end-to-end sign detector on synthetic full road scenes; no dataset release (https://ken.ieice.org/ken/paper/2018121461IO/).
- Turing Inc. and NII (Iwamasa, Shiotsuka, Yamaguchi, Miwa, Aoki), JSAI 2023: 900+ hours of Japanese driving data and a 15,000-image traffic light dataset; traffic lights only, availability not stated (https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2M6GS1002/_article/-char/en).
- TIER IV: a NEDO-funded "large-scale dataset for autonomous driving" was announced on 2025-12-25 with automated labelling and multiple vehicle models; no size, sensor list, sign annotation, release date or license was given (https://www.fieldtechnologiesonline.com/doc/tier-iv-to-develop-large-scale-dataset-for-autonomous-driving-under-nedo-project-0001). Earlier TIER IV traffic light models are documented in the traffic light survey (traffic_lights/docs/oss_implementations.md), their training data is internal.

### 3.3 Commercial offerings (not open)

- Brycen Co., Ltd. (株式会社ブライセン), press release 2024-02-27: annotated "道路付属物" semantic segmentation dataset (guardrails, road signs, road lighting), with a "詳細版" that splits signs into 本標識 and 補助標識; 36 classes; free sample on request, full dataset for sale; image count, price and license not stated (https://kyodonewsprwire.jp/release/202402277052).
- Nexdata / Datatang advertise autonomous driving datasets with 2D/3D/4D annotations (news page seen in search results, not fetched); whether Japanese sign type labels are offered is not verified.
- rita-studio on BOOTH: photoreal 3D CG models (FBX, 4K textures) of Japanese roadside and overhead sign posts for 500 JPY; free for commercial and non-commercial projects, redistribution and resale prohibited (https://booth.pm/zh-cn/items/3981593). Usable for rendering synthetic Japanese scenes, not as a dataset.

### 3.4 Global datasets: do they contain Japanese signs?

- MTSD: target 20% Asia, no country list, Japan never mentioned; not verified either way (section 2.3).
- Mapillary Vistas: "spans all the continents except for Antarctica", no country list, Japan not mentioned (https://datasetninja.com/mapillary-vistas-dataset, https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf). Signs are labelled only as front/back/direction/temporary/parking groups.
- ZOD: 14 European countries only (https://arxiv.org/pdf/2305.02008). No Japan.
- DDAD (Toyota Research Institute): Tokyo and Odaiba scenes (Japan: 47 training scenes / 3,900 frames and 14 validation scenes / 950 frames), but it is a depth benchmark with no object labels, CC BY-NC-SA 4.0 (https://raw.githubusercontent.com/TRI-ML/DDAD/master/README.md). Japanese imagery exists, unlabelled, NC.
- Mapillary platform imagery itself is CC BY-SA 4.0 and has Japanese coverage (used by infra-intelligence, section 3.2); Mapillary's own sign detections follow the taxonomy "{category}--{name}--{appearance-group}" with values such as regulatory--stop--g1, 218 values mapped to MUTCD codes on the developer page (https://www.mapillary.com/developer/api-documentation/mutcd). CanDOR-99 (section 2.12) shows that a national sign dataset can be assembled from Mapillary and released under CC BY-SA 4.0.

### 3.5 Pictograms and official sources for Japan

- Wikimedia Commons SVG drawings of Japanese signs (e.g. File:Japan road sign 302.svg, 車両通行止め, author Monaneko, source "道路標識、区画線及び道路標示に関する命令別表第2") are tagged public domain: "This work is in the public domain in Japan according to Article 13 of the Copyright Act of Japan" (mirror page https://download.osmand.net/wiki/File:Japan_road_sign_302.svg; the Commons API returned HTTP 429 in this session, section 9). Related category names seen: "Road signs in Japan", "Diagrams of Japanese-language road signs", "White-on-green road signs in Japan"; also Japanese National Route Sign SVGs and expressway shields (search results, not individually verified).
- The legal catalogue is the 道路標識、区画線及び道路標示に関する命令 (MLIT/NPA ordinance) whose 別表第2 defines every 規制/指示/警戒/案内/補助 sign; the Osaka Institute of Technology work used the MLIT list of September 2016 with 225 classes (https://www.ieice.org/publications/conference-FIT-DVDs/FIT2018/data/pdf/H-027.pdf).

### 3.6 What this means for OpenTSR

- No existing public set covers Japanese 規制/指示/警戒/案内/補助 signs with fine types; the only artefacts are a 5-category Apache-2.0 detector trained on withheld Mapillary labels, a few hundred Roboflow images, and public-domain pictograms.
- A Japanese benchmark must be built. The two legal paths are (a) Mapillary imagery for Japan (CC BY-SA 4.0, attribution to contributors, share-alike on the redistributed images, as CanDOR-99 did) and (b) own dashcam collection released under Apache-2.0 or CC BY 4.0. Synthetic data from public-domain SVG pictograms (section 4) can seed the long tail of the 225-class catalogue.

---

## 4. Synthetic and template-based data

### 4.1 Synset Signset Germany (Fraunhofer IOSB / IPA) [most relevant]

- Full dataset FraunhoferIOSB/Synset-Signset-Germany: 105,500 images of 211 German sign classes (500 per class, including signs introduced in 2020), rendered with OCTAS through both OGRE (rasterization) and Cycles (path tracing), with per-image mask, segmentation and metadata (orientation, upper/lower signs, daytime, contrast, noise, motion blur, AEC error ...); GAN-generated dirt and wear textures; templates from the German Wikipedia sign table; IBL from 327 PolyHaven environment maps; license CC BY 4.0; created 2026-03-03 (https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany/raw/main/README.md, https://huggingface.co/api/datasets/FraunhoferIOSB/Synset-Signset-Germany?full=true).
- GTSRB subset FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset: the 43 GTSRB classes, 500 images each, 21,500 images per renderer, train/validation parquet for cycles and ogre, CC BY 4.0 with a requested attribution notice ("This work is based on the Synset Signset Germany Dataset ... (c) 2024 Fraunhofer IOSB ... Licence: CC BY 4.0"); paper Sielemann et al., ITSC 2024 (arXiv 2512.05936); the card warns against use in EU AI Act Annex III high-risk systems without further validation (https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/raw/main/README.md, https://huggingface.co/api/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset?full=true).
- Result: a ConvNeXt-Small trained only on Signset Cycles reaches 98.3% top-1 on real GTSRB (98.2% with OGRE), 1.2 points below in-domain; the reverse direction (GTSRB-trained, tested on Signset) gives 89.4% / 87.4% (https://arxiv.org/pdf/2512.05936).
- Siblings: FraunhoferIOSB/SynsetSignsetSensitivityAnalysisDataset, Synset-Background-Effect-Datasets, Synset-Boulevard (https://huggingface.co/api/datasets?search=synset&limit=30), not inspected.

### 4.2 Other synthetic or challenge-synthesis sets

- CURE-TSR unreal half (Unreal Engine 4 scenes, After Effects challenge effects), CC BY 4.0 on Zenodo (section 1.5).
- German Traffic Sign Anomaly Detection Dataset: trainval_synthetic.zip (3,051 MB) generated from templates, plus 369 real anomaly photos, CC BY 4.0 (section 1.8).
- "traffic sign synthetic dataset" (Gen3D, Sun et al., Zenodo 17815715, 2025-12-04, GPL-3.0-or-later): "Gen3D synthetic traffic sign dataset generated for research"; the record lists no files in the API response (https://zenodo.org/api/records/17815715). Not usable as found.
- tanganke/gtsrb corrupted test splits (contrast, Gaussian noise, impulse noise, JPEG, motion blur, pixelate, spatter; 12,630 each) (section 1.1).
- Japanese synthetic sets by Osaka Institute of Technology (Unity, 303,750 images, 225 classes) and Tokyo Denki University (synthetic road scenes): not released (section 3.2).
- CARLA, SYNTHIA, Virtual KITTI, GTA-based sets: no released traffic-sign-type-labelled dataset was found in the web and HF searches run here (an HF search for "carla traffic sign" returned nothing) (https://huggingface.co/api/datasets?search=carla%20traffic%20sign&limit=30). DriveLM-CARLA is a QA set, not a sign-type set (section 5). Not verified further.
- Diffusion-model or world-model generated sign datasets: none found with a public release in this session (TEMPO-Diffusion, arXiv 2606.26285, mentions traffic sign poisoning in search results only). Not verified.
- emb-ai/traffic-sign-bench: SUMO road networks per sign (ODbL), a simulation benchmark, no images (section 2.15).

### 4.3 Pictogram sources and their licenses

- Germany: Commons SVGs such as File:Zeichen 205.svg (Vorfahrt gewaehren) are public domain as official works: "it is part of a statute, ordinance, official decree or judgment (official work) issued by a German authority or court"; categories like "Diagrams of regulatory road signs of Germany (2017 set)" exist per StVO revision (mirror page https://download.osmand.net/wiki/File:Zeichen_205.svg). Synset Signset Germany used the German Wikipedia overview "Bildtafel der Verkehrszeichen in der Bundesrepublik Deutschland seit 2017" as template source (https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/raw/main/README.md).
- Japan: public domain under Article 13 of the Copyright Act of Japan (section 3.5).
- Belgium: BelgiumTS DefinedTS.tar.gz contains 210 official sign pattern images (https://btsd.ethz.ch/shareddata/, https://btsd.ethz.ch/shareddata/BelgiumTS/structure.txt); license as the dataset (citation only).
- Croatia: MASTIF page offers model images of the five Croatian sign categories A to E (https://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml).
- Sweden: swedishTemplates.zip "Synthetic sign templates" on the CVL page (https://www.cvl.isy.liu.se/research/trafficSigns/).
- MTSD's 400-class taxonomy was itself built from Wikimedia Commons template images grouped by appearance (https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf), so Commons is a defensible template source for a global catalogue; per-file license templates vary by country and must be checked file by file (Commons API access was rate-limited here, section 9).

---

## 5. Vision-language style datasets for signs

- TSI-CN, "Traffic Sign Interpretation in Real Road Scene" (Yang et al., arXiv 2311.10793, v2 2023-11-28): a task that turns interrelated texts, symbols and guide panels into natural-language driving instructions; Chinese highway and urban scenes with location labels for texts, symbols and guide panels plus description labels; image and sign counts not in the abstract; "will be publicly available after the revision process"; no license (https://arxiv.org/abs/2311.10793v2). Release not verified.
- Vietnamese traffic sign VQA on Hugging Face: Anakonkai/vietnamese-traffic-sign-vqa (CC BY-SA 4.0, built from Kaggle maitam/vietnamese-traffic-signs which is CC BY-SA 4.0; 2,736 images, 130,056 QA pairs, 12 question types yes_no, count, sign_type, color, shape, location, attribute, negative, spatial_rel, count_total, multi_object, context; train 2,193 / val 272 / test 271 images) (https://huggingface.co/datasets/Anakonkai/vietnamese-traffic-sign-vqa/raw/main/README.md); tungitachi/vietnamese-traffic-sign-vqa-dedup-1k (CC BY 4.0 tag; 1,000 QA pairs on 359 images, deduplicated, leakage-free, 46 of 47 sign classes, built from the same Kaggle source) (https://huggingface.co/datasets/tungitachi/vietnamese-traffic-sign-vqa-dedup-1k/raw/main/README.md); VuthilinhVn/traffic-signs-VQA (1,595 rows with image, question, choices, answer, reasoning; no license tag) and hanguyen1907/VN_traffic_signs_qa (8,844 rows including video_path and support_frames; no license tag) (https://huggingface.co/api/datasets/VuthilinhVn/traffic-signs-VQA?full=true, https://huggingface.co/api/datasets/hanguyen1907/VN_traffic_signs_qa?full=true).
- dnth/german-traffic-sign-vl-enriched (created 2024-08-08, 545 rows, no license tag, no README body): rows carry image_uri (CloudFront links), object_labels such as "speed limit 50" with bbox, image_label and image_issues fields; 545 is exactly the size of the Roboflow GTSDB export (section 2.1), so this is a GTSDB derivative enriched with VLM fields rather than a new dataset (https://huggingface.co/api/datasets/dnth/german-traffic-sign-vl-enriched?full=true, https://datasets-server.huggingface.co/rows?dataset=dnth/german-traffic-sign-vl-enriched&config=default&split=train&offset=0&length=1).
- DriveLM (OpenDriveLab): DriveLM-nuScenes (30k frames, 360k annotated QA pairs on perception, prediction, planning, behaviour) and DriveLM-CARLA; key objects chosen by annotators include "traffic signals, pedestrians crossing the road, other vehicles"; questions are about key objects, not a sign-type benchmark; code Apache-2.0, "The language data is under CC BY-NC-SA 4.0", nuScenes inherits its own license; 1.4k stars (https://raw.githubusercontent.com/OpenDriveLab/DriveLM/main/README.md, https://img.shields.io/github/stars/OpenDriveLab/DriveLM.json). Whether sign questions exist: not verified beyond "traffic signals".
- CODA-LM (DLUT): built on CODA2022 (train 4,884 images from CODA val, val 4,384 and test 500 from CODA test); the general perception task lists seven obstacle categories "vehicles, vulnerable road users (VRUs), traffic cones, traffic lights, traffic signs, barriers, and miscellaneous", and the annotation JSON has a "traffic signs" list; license not specified on GitHub (https://raw.githubusercontent.com/DLUT-LYZ/CODA-LM/main/README.md, https://img.shields.io/github/license/DLUT-LYZ/CODA-LM.json). Sign questions exist at the object-description level.
- LingoQA (Wayve): Scenery train 3.5k videos / 267.8k QA pairs, Action train 24.5k videos / 152.5k QA, Evaluation 100 videos / 1,000 QA, Google Drive links; license not identifiable in the README (https://raw.githubusercontent.com/wayveai/LingoQA/main/README.md, https://img.shields.io/github/license/wayveai/LingoQA.json). Sign-specific questions: not verified.
- NuScenes-QA (AAAI 2024): 460k QA pairs on 30k nuScenes frames generated from perception results; code MIT (https://raw.githubusercontent.com/OpenDriveLab/DriveLM/main/README.md table, https://raw.githubusercontent.com/qiantianwen/NuScenes-QA/main/README.md, https://img.shields.io/github/license/qiantianwen/NuScenes-QA.json). nuScenes has no traffic sign category (section 2.13), so sign questions are absent by construction (inference, not verified in the QA files).
- DrivingVQA (EPFL, Zenodo 19353376, MIT, 2026-03-31): 3,931 French driving-theory-exam style samples (image, MCQ with 2 to 4 options, expert explanations, relevant entities with boxes); road signs are part of the exam content; 826 MB images (https://zenodo.org/api/records/19353376).
- Indian Traffic VQA Dataset (Zenodo 17300841, CC BY 4.0, 0.05 GB) and AutoDrive-QA (arXiv 2503.15778): seen in search results only, not inspected.
- merve/roadsign-labeled-qwen and related "roadsign-judged" sets (2026-06): VLM-labelled road sign sets (HF API search), not inspected.

---

## 6. Related evaluations of VLMs on signs

- "Mapillary Vistas Validation for Fine-Grained Traffic Signs: A Benchmark Revealing Vision-Language Model Limitations" (Garg and Aich, arXiv 2508.02047, ICCV 2025 workshop submission), data on HF as sparshgarg57/mapillary_traffic_signs (COCO instances_default.json, one parquet, no license tag) (https://huggingface.co/datasets/sparshgarg57/mapillary_traffic_signs/raw/main/README.md). Inherits the Vistas license.
- BDD100K-MUTCD used Gemini Flash zero-shot to pre-label 125,731 crops into 38 MUTCD classes before human QC (section 2.13).

---

## 7. Summary table

Images = full frames unless the task is classification (then crops). "Redistributable" = may OpenTSR re-host the images under its own repository (yes / no / unclear). "Test labels" = official test split labels are public.

| Dataset | Country | Year | Images | Instances | Classes | Task | License (as found) | Redistributable | Test labels |
|---|---|---|---|---|---|---|---|---|---|
| GTSRB | DE | 2011 | 51,839 crops (39,209 / 12,630) | one per image | 43 | cls | "The data is free to use" + citation request | unclear (no formal license; widely mirrored) | yes (since 2011-08-23) |
| BTSC (BelgiumTS cls) | BE | 2011 | 7,095 crops (4,575 / 2,520 per one source) | one per image | 62 | cls | citation request only | unclear | yes |
| rMASTIF / FER-MASTIF | HR | 2009-2011 | TS2009 ~6,000, TS2010 ~3,000, TS2011 ~1,000 signs | same | not verified | cls/det | "freely used in academic research" | no | yes |
| TSRD | CN | ~2013 | ~6,164 crops (4,170 / 1,994) | one per image | 58 | cls | not verified | unclear | yes |
| CURE-TSR | BE real + synthetic | 2017 | 2,206,106 crops | same | 14 | cls (robustness) | Zenodo: CC BY 4.0; GitHub: form + conditions; DataPort: login | yes (Zenodo copy) | yes |
| ETSD | 6 EU countries | 2018 | >80,000 crops | same | 164 | cls | "free for academic purposes ... upon request" | no | yes (on request) |
| LISA TS | US | 2012 | 6,610 | 7,855 | 47 | det | "academic license agreement" | no | yes |
| ATSD | AT | 2023 | 7,454 scenes (public 5,511) | 27,521 | 108 (60 used) | det + cls | "Permissive license ... free of charge" (terms PDF not fetched) | unclear | yes (public test) + hidden internal set |
| English roadsigns | UK | 2023 | 55,000 crops | same | 31 | cls | CC BY 4.0 | yes | n/a (no split) |
| Panoramax classified FR / DE | FR / DE | 2024-2026 | 66,000+ / 20,000+ crops | same | 250+ / 130+ | cls | etalab-2.0 / CC BY-SA 4.0 | yes (with attribution / SA) | n/a |
| German TS Anomaly Detection | DE | 2026 | 369 real + synthetic | same | 71 | anomaly | CC BY 4.0 | yes | yes |
| Synset Signset Germany | DE (synthetic) | 2024 | 105,500 (43-class subset 21,500) | same | 211 | cls (synthetic) | CC BY 4.0 | yes | n/a |
| GTSDB | DE | 2013 | 900 (600 / 300) | 1,206 (paper) or 852 (gt.txt) | 43 (3 or 4 super) | det | none stated; citation request | unclear | yes (since 2013-07-15) |
| TT100K 2016 (+2021) | CN | 2016 / 2021 | 100,000 (10,000 with signs; 16,817 annotated in 2021 parse) | ~30,000 (26,349 or 27,359) | 221 (45 usable) / 232 | det + cls | "CC-BY-NC"; download form | no | yes |
| MTSD | global | 2020 | 52,453 fully + 47,547 partially annotated | 257,543 (+87,358) | 313 (v1) / 400 (v2) + other-sign | det + cls | CC BY-NC-SA per LICENSE.txt | no | no (withheld) |
| DFG | SI | 2019 | 6,957 (+8,775 aug.) | 13,239 (+4,359 ignore) | 200 | det + cls | CC BY-NC-SA 4.0 | no | yes |
| STS | SE | 2011 | >20,000 frames (~20% labelled; or 3,777) | 3,488 (or 5,582) | 7 (or 20) | det + cls | Zenodo: CC BY 4.0 (2026); old page: no commercial use | yes (Zenodo copy, confirm) | yes |
| BTSD (BelgiumTS det) | BE | 2011-2013 | not verified | 8,851 (MTSD table) | 108 | det | citation request only | unclear | yes |
| RTSD | RU | 2016 | 59,028 (47,639 / 11,389) or 179,138 | 105,509 or 104,358 | 205 or 156 | det + cls | none stated | unclear | yes |
| Dognellaf/russian-road-signs | RU | 2026 | 43,851 | n/a | 89 (incl. lights) | det | CC BY 4.0 | yes | n/a |
| CCTSDB2021 | CN | 2021 | 17,856 + 500 negatives | ~40,000 | 3 | det | GPL-3.0 (repo and Zenodo) | unclear (GPL on data) | yes |
| CTSD | CN | 2016 | 1,100 (700 / 400) | n/a | 3 | det | not verified | unclear | yes |
| KRID | KR | 2025 | 200,000 | 2,676,583 (all facilities) | 34 (several sign classes) | det + seg | CC BY-NC-SA | no | not verified |
| SAIT Traffic Landmark | KR | 2021 | ~150,000 | ~470,000 points | 6 | point annotation | not stated; DataPort subscription | no | n/a |
| ZOD | 14 EU countries | 2023 | Frames subset (count n/a) | 446k sign instances | 156 (6 categories) | det (2D/3D) + cls | CC BY-SA 4.0 | yes (SA) | not verified |
| CanDOR-99 | CA | 2026 | 95,314 | 103,521 | 99 | det + cls | CC BY-SA 4.0 | yes (SA) | n/a |
| aiMotive TL/TS 3D | US + HU | 2024 | ~50,000 | ~550,000 sign boxes | not stated | 3D det | CC BY-NC-SA 4.0 | no | n/a |
| Argoverse 2 Sensor | US | 2021 | 1,000 x 15 s | n/a | 30 (5 sign-related) | 3D det | CC BY-NC-SA 4.0 | no | no |
| Waymo Open Perception | US | 2019+ | 2,030 x 20 s | 12.6M 3D boxes (4 classes) | 4 (Signs) | 3D det | non-commercial | no | no |
| BDD100K (+MUTCD ext.) | US | 2018 (+2026) | 100,000 | 274,594 sign boxes | 1 (+38 MUTCD) | det | custom research/not-for-profit; ext. on DataPort | no | no (20k test withheld) |
| Cityscapes / Vistas / A2D2 / IDD | DE / global / DE / IN | 2016-2021 | 5,000+20,000 / 25,000 / 41,277 / 10,003 | n/a | 1 generic sign class (Vistas: 8 sign groups) | seg | NC / NC-SA (research) / CC BY-ND 4.0 / not fetched | no / no / no (ND) / unclear | no / no / yes / no |
| STIC-Traffic-Sign (HF) | US | 2026 | 7,204 + 1,564 | 21,282 + 3,374 | 7 / 5 | det | CC BY 4.0 (uploader) | unclear (Roboflow provenance) | n/a |
| KeenForgeAI/TrafficSigns_YOLO-corrected | toy track | 2026 | 6,000 | 6,000 | 6 | det | upstream "none stated" | no | yes |
| Japan: Roboflow community sets | JP | 2022-2024 | 536 (second-2); others unknown | n/a | 3 to 16 | det | CC BY 4.0 (uploader) | unclear | n/a |
| Japan: infra-intelligence YOLOX-Tiny | JP | 2026 | 4,585 + 992 + 596 (not released) | 7,808 train boxes | 5 categories | det (model only) | weights Apache-2.0; data withheld | model yes, data n/a | eval JSON only |
| Vietnamese TS VQA (Anakonkai) | VN | 2026 | 2,736 | 130,056 QA | 47 sign classes | VQA | CC BY-SA 4.0 | yes (SA) | yes |
| DrivingVQA | FR | 2025 | 3,931 | same | MCQ | VQA | MIT | yes | yes |
| Hidden in Plain Signs A/B | multi | 2026 | 30,226 (B) | n/a | 17 / 18 | det | CC BY-NC-SA 4.0 (derivative) | no | yes |

---

## 8. Recommendation for OpenTSR

### 8.1 What can be redistributed by OpenTSR (Apache-2.0 code, data under its own CC or Apache terms)

- Clean permissive licenses verified on a primary page: Synset Signset Germany and its GTSRB subset (CC BY 4.0, with the attribution notice the authors request); CURE-TSR Zenodo copy (CC BY 4.0); STS Zenodo copy (CC BY 4.0, uploaded by the authors in 2026, but confirm with CVL because their page still carries an older restriction); English roadsigns (CC BY 4.0); German Traffic Sign Anomaly Detection Dataset (CC BY 4.0); EMTD (CC BY 4.0); Dognellaf/russian-road-signs (CC BY 4.0); Panoramax FR crops (etalab-2.0, compatible with attribution licensing); Vietnamese TS VQA (CC BY-SA 4.0: redistributable, but derived labels must stay SA). Share-alike sets usable with care: ZOD and CanDOR-99 (CC BY-SA 4.0), Panoramax DE (CC BY-SA 4.0). CC BY-SA data cannot be relabelled and released under Apache-2.0 or CC BY; keep a separate SA-licensed data package if used.
- Permissive in practice but without a formal license: GTSRB ("free to use" + citation) and GTSDB (no text at all). Recommendation: mirror the official zips only after asking INI for written permission, or point to the sid.erda.dk archive and ship only OpenTSR's own split files, label conversions and md5 lists (the md5 values in torchvision make this verifiable). The HF mirrors' GPL/CC tags are not INI's and should not be relied on.

### 8.2 Point to, do not re-host

- TT100K (CC BY-NC + registration form), MTSD (CC BY-NC-SA, test labels withheld), DFG (CC BY-NC-SA), aiMotive (CC BY-NC-SA), Argoverse 2 (CC BY-NC-SA), Waymo (non-commercial), BDD100K (research/not-for-profit custom license), Cityscapes (non-commercial, no distribution), Vistas (NC-SA research edition), KRID (CC BY-NC-SA), DDAD (CC BY-NC-SA), Hidden in Plain Signs derivatives (CC BY-NC-SA). Models trained on NC data are a legal grey zone for a commercial-grade release; OpenTSR should keep NC data out of the released weights' training set, or publish separate "research-only" weights clearly labelled as such.
- Uncertain terms: BelgiumTS/BTSC/BTSD, rMASTIF ("academic research"), RTSD (no license), TSRD, CTSD, CCTSDB (GPL-3.0 on images), LISA ("academic license agreement"), ATSD (permissive wording in the paper, terms PDF unread). Use for evaluation and comparison, not in released training data, until the authors confirm.

### 8.3 Registration or request required

- ETSD (Google form, academic only), CURE-TSR via GitHub form or IEEE DataPort login (the Zenodo copy avoids this), TT100K (name/email/organisation form), MTSD (Mapillary account), SAIT Traffic Landmark and BDD100K-MUTCD (IEEE DataPort subscription), KRID via AI Hub (Korean researchers) or GitHub.

### 8.4 A fixed public benchmark for traffic signs (proposal)

Classification tier (cropped signs):
- C1 "GTSRB-official": the 12,630-image official test set with the 43 classes, evaluated as top-1 accuracy and per-subset accuracy (speed limits, other prohibitions, derestriction, mandatory, danger, unique) as in the Neural Networks 2012 table, so numbers are comparable to 99.46% (committee), 99.71% (STN CNN) and 98.84% (human). Train only on the official 39,209 images, not on the 26,640-image mirror split; publish the md5 of the zips used.
- C2 "GTSRB-corrupted": the seven tanganke corruption splits (12,630 each) plus CURE-TSR real challenge levels 1 to 5 for the 14 CURE classes; report mean accuracy per corruption, as robustness is where production TSR fails.
- C3 "Synthetic-to-real": train on Synset Signset Germany GTSRB subset only, test on C1 (reference 98.3% ConvNeXt-S); this is the tier that tells whether OpenTSR's planned generative data pipeline works.
- C4 "Cross-country crops" (redistributable only): English roadsigns (31 classes), Panoramax FR and DE crops, Dognellaf RU crops, Vietnamese crops from the CC BY-SA Kaggle set; evaluated per country with the country's own taxonomy, to measure how a shared backbone transfers.

Detection tier (full frames):
- D1 "GTSDB-full": 900 images, all 852 (or 1,206) boxes with 43-class labels, evaluated as COCO mAP@0.5 and the IJCNN 2013 AUC per super-category; the 300-image test split is too small alone, so also report 5-fold cross-validation over the 900 images.
- D2 "ZOD-signs": the ZOD Frames 2D sign boxes with the 156-class taxonomy as the only large fine-grained detection benchmark whose license permits commercial use; publish an OpenTSR split file and evaluation script, not the images.
- D3 "NC reference" (reported for comparison only, no redistribution): TT100K official train/test (45 classes with over 100 instances; references 81.6, 89.9, 95.7 mAP) and MTSD validation (5,320 images; reference 83.4 mAP two-stage), run by users who register themselves.
- D4 "Japan" (to be built): Mapillary Japan imagery under CC BY-SA 4.0 with OpenTSR boxes and 別表第2 sign IDs across 規制/指示/警戒/案内/補助, geographically split by prefecture (as infra-intelligence split by region), with the infra-intelligence YOLOX-Tiny (Apache-2.0) as the only public baseline; synthetic long-tail classes rendered from the public-domain Commons SVGs.

Metrics and protocol notes: fix the IoU threshold (0.5) and the ignore rules (DFG ignores boxes under 30 px, infra-intelligence ignores plates under 20 px and more than half occluded), report per-size bins (ZOD, infra-intelligence and DFG all do), and keep test sets geographically disjoint from training (ZOD, CanDOR-99, infra-intelligence); the GTSRB/GTSDB test sets are not geographically disjoint from their training data.

---

## 9. Pages that could not be fetched

Blocked or failed in this session (claims about them rest on the secondary sources cited inline):

- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html (WebFetch permission not granted; proxy refuses curl). File names and md5 values come from the torchvision source instead.
- https://www.mapillary.com/dataset/trafficsign (WebFetch permission not granted; curl returns a JavaScript shell). The MTSD license text was read from a LICENSE.txt copy on Hugging Face; the research/commercial edition terms are from the Mapillary blog.
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12585093 and https://pmc.ncbi.nlm.nih.gov/articles/PMC10900943/table/tbl5 (reCAPTCHA / permission). Recent GTSRB state-of-the-art tables therefore not verified beyond the papers cited in section 1.1.
- https://contentportal.asfinag.at/assets/pdf/Bedingungen_VZK%20ASFINAG_20210726_en.pdf (ATSD license terms; permission not granted). ATSD license wording taken from the MDPI paper.
- http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html and detection.html (TSRD and CTSD official pages; curl timed out, https certificate mismatch).
- https://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html (LISA download page; proxy 502). Summary page cvrr.ucsd.edu was fetched.
- https://doc.bdd100k.com/license.html (proxy 502); BDD100K license summarised from Dataset Ninja.
- https://www.a2d2.audi/a2d2/en/dataset.html (TLS error); A2D2 facts from the arXiv paper.
- https://www.nuscenes.org/terms-of-use and https://www.nuscenes.org/nuimages (JavaScript-only pages); nuImages class list taken from the devkit annotation instructions on GitHub.
- https://people.ee.ethz.ch/~timofter/publications/Mathias-IJCNN-2013.pdf (returns an HTML page, not the PDF); BTSC/BTSD split sizes therefore from secondary papers.
- https://pytorch.org/vision/stable/_modules/torchvision/datasets/gtsrb.html (robots disallowed); the GitHub source file was used instead.
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12585093/ (redirect only).
- Wikimedia Commons API (commons.wikimedia.org/w/api.php) returned HTTP 429 "You are making too many requests"; Japanese and German SVG license statements were read from the download.osmand.net mirror of the Commons file pages.
- https://raw.githubusercontent.com/csust7zhangjm/CCTSDB/master/README.md and .../main/README.md (404); original CCTSDB statistics not verified.
- https://raw.githubusercontent.com/mapillary/mapillary_vistas/master/LICENSE (404); Vistas license from Dataset Ninja and the ICCV paper.
- http://export.arxiv.org/api/query (returned empty through the proxy); arXiv abstracts were read from arxiv.org/abs pages and PDFs instead.
- Kaggle dataset pages (meowmeowmeowmeowmeow/gtsrb-german-traffic-sign, maitam/vietnamese-traffic-signs, tamasmatuszka/aimotive-3d-traffic-light-and-sign-dataset) were not fetched; their licenses are quoted from the Hugging Face cards that cite them.
- Roboflow projects test-u58bl/japan-roadsign-ver.2 and test-u58bl/japan-roadsign-second were not fetched (siblings of fetched projects).
- IEEE DataPort pages for TUROS-TS and Moroccan Traffic Sign Recognition, the Kakao Mobility dataset article, hyper.ai dataset pages 37148 / 41354 / 16402, and the Nexdata news page were not fetched.
