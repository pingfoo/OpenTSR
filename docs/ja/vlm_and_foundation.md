# 道路標識認識のための基盤モデル、VLM、ゼロショット / オープン語彙手法、生成データ

> 本文書は英語版 [docs/vlm_and_foundation.md](../vlm_and_foundation.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

2026-10-31に、その時点で閲覧可能なWebページから作成した調査（claude.aiのリサーチセッション、pingfoo/OpenTSR）。対象範囲：GTSRBにおけるCLIP型のゼロショットおよび線形プローブの結果、道路標識で評価された大規模マルチモーダルモデル（LMM / VLM）、標識に適用されたオープン語彙検出器およびプロンプト可能な検出器、自己教師あり学習と大規模事前学習（GTSRBの結果を報告するモデルマージ論文を含む）、標識の生成データ合成。古典的なCNNとYOLOの実装は別文書で扱っており、ここでは繰り返さない。事実に関する記述にはすべて、それを検証したURLを付す。値を検証できなかった欄には「不明」または「未検証」と記す。出典間で値が食い違う場合は両方の値を示す。G節に取得できなかったページを挙げる。

規約：
- 「GTSRB」＝German Traffic Sign Recognition Benchmark。CLIPの評価スイートで用いられる形の、43クラス、train 26,640 / test 12,630のクロップ（https://arxiv.org/pdf/2103.00020 の表9）。
- 「ゼロショット」＝GTSRBの学習画像を一切使わないこと。クラス名のみを使う（CLIP型）か、自由記述 / 多肢選択の質問を使う（VLM型）。
- 「線形プローブ」＝凍結した特徴量に対するロジスティック回帰。「FT」＝フルファインチューニング。
- 特に断らない限り、精度はtop-1をパーセントで示す。CSVファイルから写した数値は小数第1位に丸めている。
- 「NC」＝非商用の制限。

概要（詳細は後述）：

| 質問 | 要点 | 該当箇所 |
|---|---|---|
| GTSRBでのCLIPのゼロショット性能はどの程度か？ | CLIP論文では32.2（ViT-B/32）から52.3（ViT-L/14-336）。open_clipの結果表では、ダウンロード可能なモデルの最高値が67.8（DFN-5B ViT-H/14-378、apple-amlrライセンス）、寛容なライセンスのモデルの最高値が67.4（EVA02-E-14-plus、MIT） | A.1、A.3 |
| GTSRBでのCLIPの線形プローブやファインチューニングの性能はどの程度か？ | 線形プローブは85.3（ViT-B/32）から92.5（ViT-L/14）。フルFTは98.5（B/32）、99.0（B/16）、99.3（L/14） | A.1、D.2 |
| 現時点でVLMは細粒度の標識で専用の分類器を上回るか？ | いいえ。オープンなVLMはGTSRB型のタスクでゼロショットtop-1が0.32から0.68、GPT-4oは0.89。MVVの11クラス標識タスクではDINOv2の特徴マッチングがInternVL-3とGemma-3を上回る。教師ありCNNはGTSRBで99+を維持している | B.1、B.3、F.2 |
| 土台として使えるだけの強さを持つ、寛容なライセンスの基盤コンポーネントはどれか？ | SigLIP / SigLIP 2（Apache-2.0）、LAION / DataCompのopen_clip重み（MIT相当）、DINOv2（Apache-2.0）、OWLv2とGrounding DINO（Apache-2.0）、Florence-2（MIT）、SAM / SAM 2（Apache-2.0）、Qwen2-VL / Qwen3-VL（Apache-2.0） | F.2 |
| 合成データによる下流のTSRの改善は測定されているか？ | はい。Synset Signset Germany（CC BY 4.0）のみで学習すると実データのGTSRBテストで98.3。AMAPの拡散インペインティングはTT100Kの希少クラスのAP50を1.23倍から7.40倍に引き上げる | E.1、E.2 |

---

## A. GTSRBにおけるCLIP型のゼロショットおよび線形プローブの結果

### A.1 OpenAI CLIP論文（Radford et al., 2021, arXiv 2103.00020）

- 評価に用いられたデータセット：GTSRB、43クラス、train 26,640、test 12,630、評価指標は精度（https://arxiv.org/pdf/2103.00020 の表9、PDFの38ページ）。
- ゼロショット精度。表11（"Zero-shot performance of CLIP models over 27 datasets"）のGTSRB列（https://arxiv.org/pdf/2103.00020 の43ページ）：

| バックボーン | RN50 | RN101 | RN50x4 | RN50x16 | RN50x64 | ViT-B/32 | ViT-B/16 | ViT-L/14 | ViT-L/14-336px |
|---|---|---|---|---|---|---|---|---|---|
| GTSRBゼロショット | 35.2 | 38.3 | 35.7 | 39.6 | 47.9 | 32.2 | 43.3 | 50.3 | 52.3 |

- 線形プローブ精度。表10（"Linear probe performance of various pre-trained models over 27 datasets"）のGTSRB列（https://arxiv.org/pdf/2103.00020 の40ページ）：

| モデル | GTSRB線形プローブ |
|---|---|
| CLIP RN50 / RN101 / RN50x4 / RN50x16 / RN50x64 | 82.4 / 82.4 / 85.5 / 89.0 / 90.2 |
| CLIP ViT-B/32 / ViT-B/16 / ViT-L/14 / ViT-L/14-336px | 85.3 / 86.6 / 92.5 / 92.4 |
| LM RN50（言語モデルのベースライン） | 73.8 |
| EfficientNet B0からB8（ImageNet） | 80.0, 75.9, 76.4, 75.8, 78.3, 78.5, 78.1, 80.8, 80.4 |
| EfficientNet Noisy Student L2-475 / L2-800 | 73.4 / 77.7 |
| Instagram ResNeXt 32x48d / FixRes-v2 | 73.0 / 76.3 |
| BiT-M R152x4（ImageNet-21k） | 68.2 |
| ViT（ImageNet-21k）B/32 / B/16 / L/16 / H/14 | 68.1 / 63.1 / 70.4 / 73.9 |
| SimCLRv2 R152x3 | 78.4 |
| BYOL 200x2 / MoCo v2 / VirTex | 77.1 / 75.7 / 69.5 |
| ResNet-50 / 101 / 152（ImageNet教師あり） | 70.6 / 72.5 / 70.1 |

- 論文中の文脈：ゼロショットCLIPは"quite weak on several specialized, complex, or abstract tasks such as ... self-driving related tasks such as German traffic sign recognition (GTSRB)"と記述されており、著者らは"non-expert humans can robustly perform several of these tasks, such as ... traffic sign recognition"と述べている（https://arxiv.org/pdf/2103.00020 の第3.1.5節）。図5では、GTSRBにおいてゼロショットCLIPがImageNet ResNet-50特徴量上の線形プローブを18.4ポイント下回っている（https://arxiv.org/pdf/2103.00020 の図5）。図7によれば、GTSRBでゼロショット精度に並ぶのに、CLIP特徴量上の線形プローブはクラスあたりわずか1.6個のラベル付き例で足りる（図中で最も低い値の一つで、これより低いのはCLEVRCounts 1.5、RESISC45 1.5、EuroSAT 0.9、Flowers102 0.9のみ。中央値は5.4）。つまりゼロショットは、少数のラベルを使う場合よりわずかに良い程度にすぎない（https://arxiv.org/pdf/2103.00020 の図7）。線形プローブの比較では、最良のCLIPモデルがGTSRBでNoisy Student EfficientNet-L2を+14.7上回り、著者らはこれをImageNet-1kが"only a single label for all traffic and street signs"しか持たないためとしている（https://arxiv.org/pdf/2103.00020 の図11および第3.2節）。表12（YFCC100Mで学習したResNet-50と、同じサイズのWITサブセットで学習したものの比較）：GTSRBの線形プローブは66.8対72.5、ゼロショットは6.9対7.0（https://arxiv.org/pdf/2103.00020 の表12）。
- 標識に関係するプロンプトエンジニアリング上の注記："GTSRB, don't appear to include this mapping [class id to name] at all in their released versions"（公開版にクラスIDから名前への対応が含まれていない）ため、著者らは独自にクラスの記述を作成した。OCRデータセットについては"putting quotes around the text or number to be recognized improved performance"とされている（https://arxiv.org/pdf/2103.00020 の第3.1.4節）。
- コードと重み：MITライセンス（https://img.shields.io/github/license/openai/CLIP.json の値は"MIT"）。

### A.2 公式のGTSRBクラス名とテンプレート（openai/CLIP、data/prompts.md）

出典：https://raw.githubusercontent.com/openai/CLIP/main/data/prompts.md （"GTSRB"の節）。同じ43個の文字列が、CLIP_benchmark（https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/resolve/main/classnames.txt ）とfusion_bench（https://raw.githubusercontent.com/tanganke/fusion_bench/main/fusion_bench/tasks/clip_classification/gtsrb.py ）で使われ、Hugging Faceのtanganke/gtsrbデータセットのラベル名にもなっている（https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md ）。

テンプレート（3個）：`a zoomed in photo of a "{}" traffic sign.`、`a centered photo of a "{}" traffic sign.`、`a close up photo of a "{}" traffic sign.`

クラス名（GTSRBのラベル順で0から42）：`red and white circle 20 kph speed limit`、`red and white circle 30 kph speed limit`、`red and white circle 50 kph speed limit`、`red and white circle 60 kph speed limit`、`red and white circle 70 kph speed limit`、`red and white circle 80 kph speed limit`、`end / de-restriction of 80 kph speed limit`、`red and white circle 100 kph speed limit`、`red and white circle 120 kph speed limit`、`red and white circle red car and black car no passing`、`red and white circle red truck and black car no passing`、`red and white triangle road intersection warning`、`white and yellow diamond priority road`、`red and white upside down triangle yield right-of-way`、`stop`、`empty red and white circle`、`red and white circle no truck entry`、`red circle with white horizonal stripe no entry`（sic）、`red and white triangle with exclamation mark warning`、`red and white triangle with black left curve approaching warning`、`red and white triangle with black right curve approaching warning`、`red and white triangle with black double curve approaching warning`、`red and white triangle rough / bumpy road warning`、`red and white triangle car skidding / slipping warning`、`red and white triangle with merging / narrow lanes warning`、`red and white triangle with person digging / construction / road work warning`、`red and white triangle with traffic light approaching warning`、`red and white triangle with person walking warning`、`red and white triangle with child and person walking warning`、`red and white triangle with bicyle warning`（sic）、`red and white triangle with snowflake / ice warning`、`red and white triangle with deer warning`、`white circle with gray strike bar no speed limit`、`blue circle with white right turn arrow mandatory`、`blue circle with white left turn arrow mandatory`、`blue circle with white forward arrow mandatory`、`blue circle with white forward or right turn arrow mandatory`、`blue circle with white forward or left turn arrow mandatory`、`blue circle with white keep right arrow mandatory`、`blue circle with white keep left arrow mandatory`、`blue circle with white arrows indicating a traffic circle`、`white circle with gray strike bar indicating no passing for cars has ended`、`white circle with gray strike bar indicating no passing for trucks has ended`（https://raw.githubusercontent.com/openai/CLIP/main/data/prompts.md ）。

注：43個の名前のうち9個は数字（速度制限の20から120）だけが異なり、"white circle with gray strike bar"の3クラスは末尾の句だけが異なる。このため、クラスの3分の1について、分類器のテキスト側はほぼ縮退している。

### A.3 CLIP_benchmark / open_clipの結果（LAION、mlfoundations）

- CLIP_benchmark（https://github.com/LAION-AI/CLIP_benchmark ）は、Hugging Face上のwebdataset `clip-benchmark/wds_gtsrb` を用いてGTSRBを評価する。ライセンスはMIT（https://img.shields.io/github/license/LAION-AI/CLIP_benchmark.json ）、スター数は817（https://img.shields.io/github/stars/LAION-AI/CLIP_benchmark.json ）。ファイルは `classnames.txt`、`zeroshot_classification_templates.txt`、`train/0..5.tar`、`test/0..2.tar`。最終更新は2023-01-20。カードにライセンスのメタデータはない（cardDataはnull）（https://huggingface.co/api/datasets/clip-benchmark/wds_gtsrb?full=true ）。テストシャード0は41,308,160バイトである（https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/resolve/main/test/0.tar のx-linked-sizeヘッダ）。テンプレートは `{}` の代わりに `{c}` を使うが、それ以外はOpenAIのテンプレートと同じである（https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/resolve/main/zeroshot_classification_templates.txt ）。READMEは、38データセットの結果についてopen_clipの結果CSVを参照している（https://raw.githubusercontent.com/LAION-AI/CLIP_benchmark/main/README.md ）。
- open_clipの結果CSV（https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv 、121モデル行、列"GTSRB"）は、CLIP_benchmarkのプロトコルによるGTSRBのゼロショット精度を示す。抜粋した行（値はすべて同ファイルから取得し、割合をパーセントに換算した）：

| open_clipモデル | pretrainedタグ | GTSRBゼロショット | ImageNet-1k | 重みのライセンス（HFカード） |
|---|---|---|---|---|
| ViT-H-14-378-quickgelu | dfn5b | 67.8 | 84.4 | apple-amlr（https://huggingface.co/api/models/apple/DFN5B-CLIP-ViT-H-14-378 ） |
| ViT-H-14-quickgelu | dfn5b | 67.7 | 83.4 | apple-amlr |
| EVA02-E-14-plus | laion2b_s9b_b144k | 67.4 | 82.0 | MIT（https://huggingface.co/api/models/QuanSun/EVA-CLIP ） |
| EVA02-E-14 | laion2b_s4b_b115k | 67.3 | 82.0 | MIT |
| EVA01-g-14-plus | merged2b_s11b_b114k | 66.4 | 79.3 | MIT |
| ViT-bigG-14-CLIPA | datacomp1b | 64.6 | 82.7 | 未検証 |
| ViT-SO400M-14-SigLIP-384 | webli | 64.3 | 83.1 | googleのSigLIPカードはApache-2.0（https://huggingface.co/api/models/google/siglip-base-patch16-224 ） |
| ViT-L-16-SigLIP-384 | webli | 63.9 | 82.1 | Apache-2.0 |
| ViT-SO400M-14-SigLIP | webli | 63.8 | 82.0 | Apache-2.0 |
| ViT-H-14-quickgelu | metaclip_fullcc | 62.6 | 80.5 | cc-by-nc-4.0（https://huggingface.co/api/models/facebook/metaclip-h14-fullcc2.5b ） |
| ViT-bigG-14 | laion2b_s39b_b160k | 62.4 | 80.1 | MIT（laionのカード。例：https://huggingface.co/api/models/laion/CLIP-ViT-H-14-laion2B-s32B-b79K ） |
| ViT-L-14-quickgelu | dfn2b | 61.7 | 81.4 | apple-amlr（https://huggingface.co/api/models/apple/DFN2B-CLIP-ViT-L-14 ） |
| ViT-L-16-SigLIP-256 | webli | 61.2 | 80.5 | Apache-2.0 |
| ViT-L-14 | datacomp_xl_s13b_b90k | 58.5 | 79.2 | 未検証 |
| ViT-H-14 | laion2b_s32b_b79k | 58.3 | 78.0 | MIT |
| convnext_large_d_320 | laion2b_s29b_b131k_ft_soup | 58.2 | 76.9 | 未検証 |
| ViT-L-14 | laion2b_s32b_b82k | 56.2 | 75.3 | MIT |
| ViT-B-16 | datacomp_xl_s13b_b90k | 55.2 | 73.5 | 未検証 |
| ViT-L-14-336 | openai | 52.4 | 76.6 | MIT（github） |
| ViT-B-16-SigLIP-384 | webli | 52.0 | 78.5 | Apache-2.0 |
| ViT-B-32 | datacomp_xl_s13b_b90k | 51.9 | 69.2 | 未検証 |
| ViT-B-16-SigLIP | webli | 51.6 | 76.0 | Apache-2.0 |
| ViT-L-14 | openai | 50.6 | 75.5 | MIT（github） |
| ViT-B-32 | laion2b_s34b_b79k | 49.3 | 66.6 | MIT（https://huggingface.co/api/models/laion/CLIP-ViT-B-32-laion2B-s34B-b79K ） |
| ViT-B-16 | laion2b_s34b_b88k | 48.3 | 70.2 | MIT |
| RN50x64 | openai | 47.9 | 73.9 | MIT（github） |
| ViT-B-16 | dfn2b | 47.0 | 76.2 | apple-amlr |
| ViT-B-16 | openai | 43.4 | 68.3 | MIT（github） |
| ViT-B-32 | laion400m_e32 | 40.7 | 60.2 | MIT |
| RN50x16 / RN101 / RN50x4 / RN50 | openai | 39.8 / 37.5 / 36.2 / 35.1 | 70.7 / 62.3 / 66.3 / 59.8 | MIT（github） |
| ViT-B-32 | openai | 32.6 | 63.3 | MIT（github） |
| ViT-B-32 | commonpool_s_*（13Mサンプル） | 4.1から7.3 | 3から5 | 未検証 |

  （GTSRBとImageNetの値はすべて次による：https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv ）。所見：（1）2つのOpenAIのプロトコルは0.4ポイント以内で一致する（ViT-B/32で32.2対32.6、ViT-L/14で50.3対50.6）。（2）オープンなCLIP型モデルは、1Bから5Bパラメータ規模でもGTSRBのゼロショットで68%を超えるものがない。これに対し、教師ありCNNは99+である。（3）サイズよりデータのキュレーションが効く。DFN-5B ViT-H（67.8）はLAION-2B ViT-bigG（62.4）を上回る。
- open_clipのコードライセンス：著者名を記したMIT型の許諾表示（https://raw.githubusercontent.com/mlfoundations/open_clip/main/LICENSE 。GitHubはこれを自動判別できず、 https://img.shields.io/github/license/mlfoundations/open_clip.json は"not identifiable by github"を返す）。

### A.4 GTSRBの結果を報告しているその他の基盤モデル論文

- Reproducible scaling laws for open_clip（Cherti et al., arXiv 2212.07143）：表6（ゼロショット）、表7（ゼロショットヘッドを凍結したまま、視覚エンコーダを8タスクで同時にファインチューニング）、表8（同時パッチング）。GTSRB列（https://arxiv.org/pdf/2212.07143 の付録の表）：

| モデル（学習サンプル数、データ） | ゼロショット | ファインチューニング（同時） | パッチ適用 |
|---|---|---|---|
| ViT-B/32 (13B, CLIP-WIT) | 32.6 | 98.2 | 95.5 |
| ViT-B/32 (13B, LAION-400M) | 42.0 | 98.4 | 94.0 |
| ViT-B/32 (34B, LAION-2B) | 36.5 | 98.0 | 95.6 |
| ViT-B/16 (13B, CLIP-WIT) | 43.3 | 98.8 | 97.4 |
| ViT-B/16 (13B, LAION-400M) | 43.5 | 98.8 | 97.2 |
| ViT-L/14 (13B, CLIP-WIT) | 50.6 | 99.2 | 99.0 |
| ViT-L/14 (13B, LAION-400M) | 49.9 | 99.0 | 98.9 |
| ViT-L/14 (34B, LAION-2B) | 56.1 | 98.9 | 98.7 |
| ViT-H/14 (34B, LAION-2B) | 58.4 | 99.0 | 98.8 |
| ViT-g/14 (13B, LAION-2B) | 49.9 | 99.0 | 98.8 |

  同論文自身のLAION-2Bモデルのゼロショット表（34ページ）では、GTSRBは49.5（B/32、34B）、48.2（B/16）、56.1（L/14）、49.7（g/14、13B）、58.5（H/14）である（https://arxiv.org/pdf/2212.07143 ）。ViT-B/32 LAION-2Bの36.5と49.5は、異なる2つのチェックポイントの値である（open_clipのCSVではlaion2b_e16 = 36.6、laion2b_s34b_b79k = 49.3。https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv ）。
- MetaCLIP（Xu et al., arXiv 2309.16671）の表4（同一の学習設定でのMetaCLIP-400M、CLIP、OpenCLIP-LAION-400Mの比較）とスケーリングの表のGTSRB列（https://arxiv.org/pdf/2309.16671 の8ページ）：

| バックボーン | CLIP（WIT-400M、同論文での評価） | OpenCLIP（LAION-400M） | MetaCLIP 400M | MetaCLIP 1B | MetaCLIP 2.5B |
|---|---|---|---|---|---|
| ViT-B/32 | 32.6 | 42.0 | 35.8 | 40.7 | 39.2 |
| ViT-B/16 | 43.3 | 43.5 | 43.8 | 49.5 | 46.6 |
| ViT-L/14 | 50.6 | 49.9 | 47.5 | 58.7 | 56.3 |
| ViT-H/14 | なし | なし | なし | なし | 62.5 |

  MetaCLIPの重みはHugging Face上でCC-BY-NC 4.0であり（https://huggingface.co/api/models/facebook/metaclip-h14-fullcc2.5b ）、OpenTSRのリリースには使えない。
- EVA-CLIP（Sun et al., arXiv 2303.15389）の表2（27データセットでのゼロショット）のGTSRB列（https://arxiv.org/pdf/2303.15389 の3ページ）：OpenAI CLIP-B/16 43.5、OpenCLIP-B/16 48.3、EVA-02-CLIP-B/16 46.3。OpenAI CLIP-L/14 50.6（336 px版は52.4）、OpenCLIP-L/14 51.6、EVA-02-CLIP-L/14 57.1（336 px版は57.6）。OpenCLIP-H/14 58.4。EVA-01-CLIP-g/14 49.1と66.5（後者は"plus"データ版）。OpenCLIP-G/14 62.5。EVA-02-CLIP-E/14 67.1と67.7。EVA-CLIPの重みはHugging Face上でMITである（https://huggingface.co/api/models/QuanSun/EVA-CLIP, https://huggingface.co/api/models/timm/eva02_large_patch14_clip_224.merged2b ）。
- DINOv3（Simeoni et al., arXiv 2508.10104）の付録B.4 "Classification on OCR-Heavy Datasets"、表25（入力512 pxでのバックボーン凍結による分類）：GTSRBでDINOv2 ViT-g 78.2、PE-core ViT-G 94.8、DINOv3-7B 87.5（https://arxiv.org/pdf/2508.10104 の57ページ）。論文はこれらを"notoriously hard datasets for SSL"と呼んでおり、弱教師ありのPE-coreモデルとの差は7.3ポイントである。DINOv3の重みは独自の"dinov3-license"の下にあり、Hugging Faceでは手動承認のゲートが設けられている（https://huggingface.co/api/models/facebook/dinov3-vitb16-pretrain-lvd1689m ）。GitHub上のライセンスは"not identifiable by github"である（https://img.shields.io/github/license/facebookresearch/dinov3.json ）。DINOv2はApache-2.0である（https://img.shields.io/github/license/facebookresearch/dinov2.json, https://huggingface.co/api/models/facebook/dinov2-base ）。
- 報告なし：SigLIP論文（https://arxiv.org/pdf/2303.15343 ）、SigLIP 2論文（https://arxiv.org/pdf/2502.14786 ）、Data Filtering Networks論文（https://arxiv.org/pdf/2309.17425 ）、AIMv2論文（https://arxiv.org/pdf/2411.14402 ）、DINOv2論文（https://arxiv.org/pdf/2304.07193 ）にはGTSRBの数値がない（PDF全文をgrepし、いずれも0件）。SigLIP（v1）とDFNの数値は上記のopen_clipのCSVにしかない。SigLIP 2のGTSRBの数値はどこにも見つからなかった（open_clipのCSVにSigLIP 2の行はない。 https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv ）。DataComp論文はGTSRB（テスト画像12,630枚、43クラス）を38の評価タスクの一つとして挙げているが、図示しているだけである（https://arxiv.org/pdf/2304.14108 の41ページの表と56ページの図）。

### A.5 「数字が読めない」という失敗モード：出典が実際に述べていること

- 取得した出典のなかに、CLIPによる速度制限の数字の取り違えをクラス別の表で切り分けたものはない。根拠は間接的である。（1）GTSRBの43クラスのうち9クラスは速度制限の数字だけが異なり（A.2）、ゼロショット精度は32から68にとどまる一方、同じ特徴量上の線形プローブは85から93に達する（A.1、A.3）。これは、情報は画像埋め込みには含まれているが、テキスト埋め込み中の数字の文字列とは整合していないことを示す。（2）CLIP論文は、OCR型のデータセットについて"putting quotes around the text or number to be recognized improved performance"と報告しており（https://arxiv.org/pdf/2103.00020 の第3.1.4節）、GTSRBのテンプレートがクラス名を引用符で囲んでいるのはこのためである。（3）DINOv3の著者らは、GTSRBを自己教師あり特徴量にとって"notoriously hard"な"OCR-heavy"データセットに分類している（https://arxiv.org/pdf/2508.10104 の付録B.4）。（4）生成側では、AMAPのインペインティング論文が、汎用の拡散インペインティングは標識面で"distort[s] digits"（数字を歪める）と報告している（https://arxiv.org/pdf/2609.02348 の要旨）。（5）インドの教科書の標識28サンプルによる小規模なテストでは、CLIPの精度が0.39（ラベルの語のみ）から0.46（"sign"を付加）、0.54（"symbol"を付加）へと変化し、プロンプトへの依存が強いことを示している（https://arxiv.org/pdf/2511.13909 の表1と考察）。GTSRBの速度制限に対するCLIPのクラス別混同分析を、F.3の再現実験1として提案する。

### A.6 プロンプトエンジニアリングと、テンプレート / プロトタイプに基づくゼロショットTSR

- Prototypical Priors（Jetley et al., BMVC 2015, arXiv 1512.01192）：各標識の公式ピクトグラムのテンプレートを補助情報として用いる。GTSRB（"D1"、43クラス）では、プロトタイプにより分類精度が97.48から97.98に向上した。既知33クラス / 未知10クラスのゼロショット分割では、属性ベースのベースラインを平均5.48ポイント上回る（絶対値は図にのみ示されている）（https://arxiv.org/pdf/1512.01192 の表2および第5.2.2節）。
- クラスタリングの帰納バイアスとランダムニューラルネットワークによる少数ショットTSR（Zhou et al., Pattern Recognition 2020）と、中間レベル特徴マッチングによるゼロショットTSR（Gan et al., Sensors 2023）は、北海道大学のグループが用いる2つのテンプレートマッチングのベースラインである。CdMTとの比較では、Gan et al. 2023はtop-1で0.56（GTSRBサブセット）、0.67（BTSD）、0.12（TT-100K）、0.42（Sapporo）、0.27（Yokohama）に達する（https://arxiv.org/pdf/2409.01534 の表1および参考文献[22]、[23]）。
- 標識に対するプロンプト学習付きCLIP：GTSRBサブセットでのtop-1はCoOp 0.32、MaPLe 0.28、CLIP-Adapter 0.37で、素のCLIP ViT-B/32は0.24である（同じサブセット。https://arxiv.org/pdf/2409.01534 の表1）。TSCLIP（ICRA 2025, arXiv 2409.15077）は"traffic sign prompt engineering scheme"（クラスごとのシナリオと交通ルールの記述）と、ゼロショットの重みとの適応的な動的重みアンサンブルを導入している。CRTSデータセットのドイツ部分（GTSRB由来、31クラス）では、CLIPのゼロショットが27.75%、中国とスロベニアで学習した場合に線形プローブ30.56、フルFT 84.58、Wise-FT 85.54、TSCLIP 87.08である（https://arxiv.org/pdf/2409.15077 の表IおよびII）。使用したCLIPのバックボーンは、解析できた本文中には記載がない（未検証）。コードは"will be released soon"とされ、CRTSデータセット（190,000枚超、統一された46カテゴリ、10地域）はGoogle Driveのリンクで提供されている（https://raw.githubusercontent.com/guoyangzhao/TSCLIP/main/README.md ）。リポジトリのライセンスはMIT（https://img.shields.io/github/license/guoyangzhao/TSCLIP.json ）。
- LMMのプロンプトエンジニアリング（ピクトグラムのテンプレートから生成した文脈 / 特徴 / 差分の記述）は、CdMTと少数ショットのインコンテキスト学習論文の中核である。B.1とB.2を参照。

---

## B. 道路標識で評価された大規模マルチモーダルモデル

### B.1 Cross-domain Multi-step Thinking、CdMT（Gan et al., Hokkaido University; arXiv 2409.01534）

- タイトルの変遷：v1（2024）は"Think Twice Before Recognizing: Large Multimodal Models for General Fine-grained Traffic Sign Recognition"（http://export.arxiv.org/api/query?id_list=2409.01534v1 ）、v2（2025年7月23日）は"Cross-domain Multi-step Thinking: Zero-shot Fine-grained Traffic Sign Recognition in the Wild"（https://arxiv.org/pdf/2409.01534 ）。
- 手法：道路標識の抽出（ViT-Adapter検出器。代替の抽出器としてSAM 2も比較）に続き、LMMによる3つの「思考」ステップを行う。中心座標によるプロンプト最適化と"prior traffic sign hypotheses"を伴う文脈記述、きれいなピクトグラムのテンプレートからインコンテキスト学習で生成する特徴記述、類似した標識間の差分記述である。学習は行わない（https://arxiv.org/pdf/2409.01534 の要旨および第3節）。
- データセット：公開ベンチマークのGTSRB、BTSD（ベルギー）、TT-100Kに加え、非公開の日本のデータセット2件（札幌と横浜の市街地道路データセットであるSapporoとYokohama）。APIのレート制限のため、著者らは層化抽出によって全クラスを残したうえで"randomly used the subsets of GTSRB, BTSD, and TT-100K validation data"としている。v2にはサブセットのサイズが記載されていない（未検証。姉妹論文のB.2ではGTSRB 100枚とBTSD 125枚を使用）（https://arxiv.org/pdf/2409.01534 の第4.1節）。評価指標：5回の試行で平均したtop-k精度。
- 表1のtop-1（top-3）の結果（https://arxiv.org/pdf/2409.01534 の表1）：

| 手法 | GTSRB | BTSD | TT-100K | Sapporo | Yokohama |
|---|---|---|---|---|---|
| CLIP ViT-B/32 | 0.24 (0.35) | 0.20 | 0.29 | 0.27 | 0.14 |
| EVA-02 | 0.41 | 0.30 | 0.32 | 0.48 | 0.29 |
| LLaVA-1.5 | 0.32 (0.45) | 0.28 | 0.13 | 0.09 | 0.11 |
| LLaVA-NeXT | 0.39 (0.48) | 0.31 | 0.20 | 0.10 | 0.13 |
| VITA-1.5 | 0.45 | 0.39 | 0.25 | 0.18 | 0.20 |
| GPT-4V | 0.81 (0.85) | 0.70 | 0.72 | 0.32 | 0.22 |
| GPT-4o | 0.89 (0.89) | 0.83 | 0.74 | 0.57 | 0.49 |
| CdMT-LLaVA-1.5 | 0.55 | 0.48 | 0.45 | 0.35 | 0.36 |
| CdMT-LLaVA-NeXT | 0.60 | 0.51 | 0.48 | 0.37 | 0.41 |
| CdMT-VITA-1.5 | 0.68 | 0.65 | 0.52 | 0.45 | 0.61 |
| CdMT-GPT-4V | 0.91 | 0.89 | 0.90 | 0.77 | 0.83 |
| CdMT-GPT-4o | 0.93 (0.97) | 0.88 | 0.97 | 0.89 | 0.85 |

  GTSRBで学習せずにクロスドメインで評価した教師ありモデル（DenseNet-121 0.31、EfficientNet-B0 0.52、Li et al. 0.75、MobileViT 0.05、Swin V2 0.14、MAE 0.20、DeiT 0.27）も表1に含まれている。標識あたりの推論コスト：LLaVA-1.5 6.4 s、LLaVA-NeXT 6.0 s、GPT-4V 2.0 s、GPT-4o 1.6 s（https://arxiv.org/pdf/2409.01534 の表6）。
- OpenTSRにとっての意義：日本の標識でLMMを評価した論文として見つかった唯一のものである。多段階のプロンプトなしでは、オープンなLMMはCLIPの線形プローブと同等かそれ以下である。GPT-4oの数値は小さなサブセットでのものである。論文中にコードやデータの公開は見つからなかった（未検証）。

### B.2 クロスドメインの少数ショットインコンテキスト学習（Gan et al., arXiv 2407.05814）

- 同じグループによるもので、GPT-4VとGPT-4oのみを対象とし、テンプレートの標識から説明文を生成する。サブセット：GTSRB 100枚（43クラス）、BTSD 125枚（62クラス）、Sapporo 68枚（18種類の標識）、Yokohama 33枚（13種類）（https://arxiv.org/pdf/2407.05814 の第4節）。top-1 / top-3 / top-5（https://arxiv.org/pdf/2407.05814 の表1および表2）：

| 手法 | GTSRB | BTSD | Sapporo | Yokohama |
|---|---|---|---|---|
| ベースライン（GPT-4V、画像のみ） | 0.80 / 0.86 / 0.87 | 0.61 / 0.70 / 0.75 | 0.28 / 0.43 / 0.54 | 0.46 / 0.70 / 0.76 |
| 提案手法（GPT-4V） | 0.88 / 0.97 / 1.00 | 0.91 / 0.98 / 1.00 | 0.75 / 0.84 / 0.91 | 0.88 / 0.94 / 1.00 |
| ベースライン（GPT-4o） | 0.88 / 0.89 / 0.89 | 0.79 / 0.89 / 0.89 | 0.46 / 0.66 / 0.80 | 0.65 / 0.88 / 0.88 |
| 提案手法（GPT-4o） | 0.89 / 0.98 / 0.98 | 0.91 / 0.99 / 0.99 | 0.82 / 0.99 / 1.00 | 0.93 / 1.00 / 1.00 |

### B.3 Mapillary Vistas Validation for Traffic Signs、MVV（Garg and Aich, NEC Labs America; arXiv 2508.02047, ICCVW 2025）

- データセット：Mapillary Vistasの検証画像2,000枚を、専門のアノテータが11の標識カテゴリ（stop、speed limit、yield、do not enter、crosswalk、parking、no parking、roundabout、turn restriction、cycle lane、others）に再ラベル付けしたもの。元のインスタンスマスクを再利用しているため、ボックスとマスクの両方が利用できる（https://arxiv.org/pdf/2508.02047 の第1節および第3節）。リポジトリのREADMEは同じ11の名前を挙げている（ただし"12 categories"と記している）（https://raw.githubusercontent.com/nec-labs-ma/relabeling/main/README.md ）。ライセンス：GitHubは"not identifiable by github"と報告している（https://img.shields.io/github/license/nec-labs-ma/relabeling.json ）。スターは4（https://img.shields.io/github/stars/nec-labs-ma/relabeling.json ）。元のMapillary Vistasの画像には、Mapillary独自の研究用ライセンスが付いている（ここでは再検証していない）。
- プロトコル：粗いラベルを与えたうえでの各標識インスタンスの分類。入力モードは3種類（クロップ画像、赤いバウンディングボックスを描いた全体画像、SAMで精緻化したクロップ）。VLMはデフォルト設定のゼロショットで、実行は2回繰り返し、評価指標は精度。DINOv2は外部データセット（MapillaryとBDDは除外）から構築した特徴バンクに対する最近傍マッチング（https://arxiv.org/pdf/2508.02047 の第4.2節、第4.3節）。
- 道路標識タスクT3での結果（https://arxiv.org/pdf/2508.02047 の表3）：InternVL-3は1B 0.085（クロップ）、2B 0.323、8B 0.453、9B 0.468（VLMの最高値、クロップ）、14B 0.408。Gemma-3は4B 0.276、12B 0.428。教師ありのResNet-50 0.275、ResNet-101 0.310。DINOv2の特徴マッチング0.484（最高値）。車両（T1）ではDINOv2が0.935、VLMの最高値が0.768であり、差は全般的に見られるが、標識が最も難しいタスクである。
- 要点：実際の路上のクロップによる11クラスの標識タスクでは、14BまでのオープンなVLMはゼロショットで精度0.5未満にとどまり、言語を使わないDINOv2のプロトタイプマッチャーが最も強いゼロショット手法である。

### B.4 TS-1M（Zhao et al., HKUST(GZ); arXiv 2603.23034, March 2026）

- データセット："TS-1M"。公開データセットとWeb画像を集約した、標準化済みの454カテゴリ。論文の表Iではtrain 1,033,947、test 233,611のクロップとしている（https://arxiv.org/pdf/2603.23034 の表I）。プロジェクトページには"approximately 1.26 million images ... about 1.06M training images and 200K test images"とある（https://guoyangzhao.github.io/projects/ts1m ）。Baidu、Google Drive、Kaggleからダウンロードでき、"publicly available for research and academic use"とされるが、データセットのライセンス文は見つからなかった（ページのフッタはWebサイト自体をCC BY-SA 4.0でライセンスしている）（https://guoyangzhao.github.io/projects/ts1m, https://raw.githubusercontent.com/guoyangzhao/TS1M-Traffic-Sign/main/README.md ）。コードのリポジトリはMITで、スターは28（https://img.shields.io/github/license/guoyangzhao/TS1M-Traffic-Sign.json, https://img.shields.io/github/stars/guoyangzhao/TS1M-Traffic-Sign.json ）。TS-1MはGTSRB、TT100K、MTSD、RTSD、DFG、ARTSなどから構築されているため、元データのライセンス（いくつかはNC）が引き継がれる。研究用途に限るものとして扱う。
- プロトコル：すべてのモデルを、分類ヘッドを付けてTS-1Mで学習する（VLMはエンコーダとして用いる。CLIP ViT-B/32のデュアルエンコーダ、BLIP、BLIP-2、線形ヘッド付きのLLaVA-1.5-7Bであり、"without invoking multi-step reasoning"）（https://arxiv.org/pdf/2603.23034 の第IV.C節、第IV.D節、表III）。
- 表IVの全体のtop-1 / F1（https://arxiv.org/pdf/2603.23034 ）：ResNet50 0.9260 / 0.8570、ConvNeXt 0.9428 / 0.9043、DINO ViT-B/16 0.9190 / 0.8595、MoCoV3 0.9385、CLIP 0.9533 / 0.9113（最高値）、BLIP 0.9448、BLIP-2 0.9432、LLaVA 0.9406 / 0.8776（7,062.9 Mパラメータ、6,216.7 GFLOPs）。表Vの地域横断の精度：CLIPはGTSRB 0.9996、TT100K 0.9898、MTSD 0.9118、平均0.9578で、ResNet50は0.9150。表VIの希少クラス：CLIP 0.9291で、ResNet50ベースラインに対し相対で+9.10%。
- 要点：教師ありでは、CLIP ViT-B/32エンコーダがこのベンチマークで最良のバックボーンであり、分類器として使った7BのLLaVA（0.9406）は、パラメータ数が140分の1、FLOPsが約700分の1のConvNeXt（0.9428）を下回る（https://arxiv.org/pdf/2603.23034 の表IV）。

### B.5 VLADBench（Li et al., USTC and Huawei; arXiv 2503.21505, ICCV 2025 "Fine-Grained Evaluation of LVLMs in AD"）

- 5ドメイン、29タスクにわたる12Kの多肢選択問題。第3階層のタスク"Traffic Sign"は、標識の種類、状態、意味、OCRを扱う（https://arxiv.org/pdf/2503.21505 の第3.1節、第3.3節、表2）。リポジトリ：https://github.com/Depth2World/VLADBench 。ライセンスは"not specified"（https://img.shields.io/github/license/Depth2World/VLADBench.json ）、スターは43。
- Traffic Signタスクの精度、表10（https://arxiv.org/pdf/2503.21505 ）：Qwen2-VL-72B 71.18、GPT-4o 68.96、Qwen2.5-VL-72B 65.11、Gemini-1.5 Pro 64.05、Qwen2.5-VL-7B 61.71、InternVL2-76B 53.92、LLaVA-Video-7B 49.73、LLaVA-OneVision-7B 49.27、InternVL2-8B 47.45、Qwen2-VL-7B 40.94、MiniCPM-V-2.6 32.10、Qwen-VL-7B 33.58、InternVL2-4B 34.01。運転特化型のDriveLM 44.74、DriveMM 47.05、Dolphins 25.02。著者らがVLADBenchでファインチューニングしたInternVL2-4Bは61.00。

### B.6 IDKB、"Can LVLMs Obtain a Driver's License?"（Lu et al., ShanghaiTech; arXiv 2409.02914）

- "Driving Road Data"の部分は、CARLAでレンダリングした道路標識理解のセットである。複数の国の3D標識モデルを2つのCARLAマップに配置して約400,000フレームを収集し、フィルタリング後のQAサンプルは112,388件（https://arxiv.org/pdf/2409.02914 の第3節）。リポジトリはMIT（https://img.shields.io/github/license/4DVLab/IDKB.json ）。
- Driving Road Dataの多肢選択の精度、表2（https://arxiv.org/pdf/2409.02914 ）：GPT-4o 0.87、Gemini-1.5-flash 0.85、XComposer2 0.62、MiniCPM-Llama3-V2.5 0.58、Qwen-VL-chat 0.47、Monkey-chat 0.46、DeepSeek-VL-7B 0.45、ShareGPT4V-7B 0.44、CogVLM 0.40、LLaVA-1.5-7B 0.33、VisualGLM-6B 0.18、BLIP2 0.17。

### B.7 DriveQA（Wei, Liu, Ohn-Bar; arXiv 2508.21824, ICCV 2025）

- 米国の運転教本の知識を扱う、テキストと視覚のベンチマーク。DriveQA-VはMUTCDの標識を、視点、天候、シナリオを変えながらCARLAでレンダリングする。プロジェクトページ：https://driveqaiccv.github.io （ライセンスは未検証）。Signs部分での既製モデルの精度、表4（https://arxiv.org/pdf/2508.21824 ）：GPT-4oは規制標識93.75 / 警戒標識94.02 / 案内標識95.11 / 臨時標識94.35。Mini-InternVL-2B 64.06 / 55.34 / 65.82 / 45.04。LLaVA-1.6-mistral-7B 42.58 / 43.01 / 52.75 / 37.50。LLaVA-1.5-7B 23.51 / 26.61 / 22.31 / 21.10。VILA-1.5-8B 25.32 / 23.33 / 27.78 / 21.46。DriveQAでファインチューニングした後：Mini-InternVL-2B 93.79 / 92.19 / 91.08 / 96.51。モデル横断で最も難しい標識の種類には、Playground、Trauma Center、Golf Carts、Ground Clearance、No Stopping、No Parkingがある（表5）。

### B.8 SignReasonerとTrafficSignEval（Wang et al., Huawei Noah's Ark; arXiv 2604.10436, April 2026）

- 対象：複雑な中国の案内標識（多言語のテキスト、複合記号）。"Functional Structure Units"への分解と、Qwen2.5-VL-7BおよびPangu-MM-7Bに対する2段階の事後学習（Caption-FSU蒸留とFSU-GRPO）。手作業でアノテーションした標識5K件（https://arxiv.org/pdf/2604.10436 の要旨および第4節）。
- TrafficSignEvalでのキャプションプロトコルによるゼロショット精度、表1（https://arxiv.org/pdf/2604.10436 ）：InternVL3-8B 54.07、InternVL3-78B 68.02、Qwen2.5-VL-7B 48.26、Qwen2.5-VL-72B 65.12。SignReasoner（Qwen2.5-VL-7B）72.67（+24.41）。構造プロトコル：Qwen2.5-VL-7BはSignReasonerにより37.95から83.08に向上。図1では、Qwen2.5-VL-72B、InternVL3-78B、Gemini 2.5 Proのいずれもが車線案内標識を読み誤っている。PDF中にベンチマークの公開URLは見つからなかった（未検証）。

### B.9 SignRAG（Zhu et al., Ohio State and TRC; arXiv 2512.12885, Dec 2025）

- オハイオ州のMUTCD規制標識303デザインのゼロショット認識：Gemini 2.5 Flashが標識を記述し、その記述を埋め込み（OpenAIの3072次元埋め込み）、参照デザインのベクトルデータベースから候補を検索し、その候補に対してLLMが推論する（https://arxiv.org/pdf/2512.12885 の第III節）。
- 結果：理想的な参照画像での生成精度は95.58%（検索のtop-1は88.65）。オハイオ州の道路72マイルにわたって収集した181インスタンス / 20種類の実世界セットでは82.45（top-1 63.44、top-5 96.45）。303枚の参照画像に対してGemini 2.5 FlashとGPT-5に直接ゼロショットでプロンプトした場合の精度は"less than 10%"。平均レイテンシは3.99 s（https://arxiv.org/pdf/2512.12885 の第IV節）。

### B.10 道路標識解釈（TSI）の系譜：TSI-CNとSignEye

- "Traffic Sign Interpretation in Real Road Scene"（Yang et al., NWPU; arXiv 2311.10793）：TSI-CNデータセットは2160x3840の画像2,682枚と2,910件の自然言語記述からなり、記号7カテゴリと案内板155カテゴリを含み、分割は2,016 / 666。マルチタスクのTSIアーキテクチャ（LMMではない）は、記号F 79 / Acc 72、テキストF 77 / Acc 63、パネルF 85 / Acc 85、解釈のソフト精度48に達する。コードとデータは"will be publicly available after the revision process"とされている（https://arxiv.org/pdf/2311.10793 の第4節、第6節、表5）。公開は確認できていない。
- SignEye（Yang et al.; arXiv 2411.11507）：Traffic-CNは、中国の6省で撮影した1920x1080の一人称視点画像20k+枚。SigLIPの視覚エンコーダとQwen2デコーダをLoRAでファインチューニングしている。TSIのBLEU-4 56.2 / CIDEr 4.70に対し、Qwen2-VL 45.4 / 4.24、MiniCPM-2.6 48.6 / 4.05、LLaVA-1.5 53.9 / 4.20。交通案内支援の精度は87.0で、MiniCPM-2.6は79.7（https://arxiv.org/pdf/2411.11507 の表3および表4）。比較対象のVLMがファインチューニングされたかどうかは、抽出したテキストには明記されていない（未検証）。

### B.11 定性的評価と関連する評価

- "On the Road with GPT-4V(ision)"（OpenGVLab; arXiv 2311.05332）：定性的な評価のみ。GPT-4Vは"can recognize most road signs, including the nearby 'SLOW' and the distant height limit '4.5m', but it misrecognizes the 'Speed Bump' sign"とされ、"non-English traffic signs"に関する限界が指摘されている（https://arxiv.org/pdf/2311.05332 の第2.1節および第7節）。
- VehicleSec 2024 WIP（Aldeen et al., Clemson）：GTSRBの4クラスから生成した敵対的（"Natural Denoising Diffusion"）な偽の道路標識の検出器としてLMMを使う。"real sign or not"の2値分類の精度は、GPT-4V 84.06、Bard 85.42、LLaVA-7B 79.80、LLaVA-13B 80.81で、GTSRBで学習したモデルはResNeXt 17.17、MobileNet 18.67、YOLOv5 44.12、VGG16 71.05である（https://www.ndss-symposium.org/wp-content/uploads/vehiclesec2024-44-paper.pdf の表II）。
- "Mind the Gap"（Chalamalasetti, Potsdam; arXiv 2511.13909, AAAI 2026）：インドの教科書の標識28サンプル、ゼロショットの多肢選択：GPT-5 Image Mini 0.79、InternVL3-78B 0.71、Gemma-3-27B 0.71、Qwen3-VL-30B 0.68、CLIP 0.54（https://arxiv.org/pdf/2511.13909 の表1）。
- LVLMによるシナリオ理解（Rivera et al., TUM; arXiv 2501.17131, WACVW 2025）：シーンタグの一つとして、yes / noのタグ"traffic sign for ego-vehicle"がある。全体ではLLaVA-1.6-34Bが最高で、精度83.7（https://arxiv.org/pdf/2501.17131 の表1および表2）。細粒度の標識テストではない。
- Sah et al.（arXiv 2503.06313）：TSRはResNet-50（GTSRBで99.8）が担い、MLLMの部分は車線に関する推論にのみ使われる（https://arxiv.org/pdf/2503.06313 の要旨）。タイトルから誤解しないようにここに挙げる。
- TLS-Assist（Schmidt et al., Esslingen and Freiburg; arXiv 2511.14391）：VLMの評価ではなく、YOLO11による信号機と標識の検出結果をテキストとしてCARLA上のLLM運転エージェントに注入する、モジュール型の冗長化レイヤである。著者らのCARLA標識セットでのYOLO11l/xの標識検出mAP50-95は0.974。コード：https://github.com/iis-esslingen/TLS-Assist （https://arxiv.org/pdf/2511.14391 の表I）。
- 工事区間の速度に関するVLMパイプライン（Martinez-Sanchez et al., UC Merced; arXiv 2606.08860）：Jetson上で臨時の速度制限を扱う、検出器、OCR、意味的検証の組み合わせ。フレーム単位の状態の評価指標のみ（https://arxiv.org/pdf/2606.08860 ）。

### B.12 運転特化型VLM：標識を明示的に扱っているか？

各論文のPDF全文で"traffic sign"、"road sign"、"speed limit"、"stop sign"を検索して確認した（括弧内は件数）。

| モデル / データセット | arXiv | 標識の明示的な扱い | 根拠 |
|---|---|---|---|
| DriveLM（nuScenes / CARLA上のグラフVQA） | 2312.14150 | 見つからず（0件） | https://arxiv.org/pdf/2312.14150 |
| DriveGPT4 | 2310.01412 | 見つからず（0件） | https://arxiv.org/pdf/2310.01412 |
| Dolphins | 2312.00438 | 見つからず（0件） | https://arxiv.org/pdf/2312.00438 |
| LingoQA | 2312.14115 | 見つからず（0件） | https://arxiv.org/pdf/2312.14115 |
| EM-VLM4AD | 2403.19838 | 見つからず（0件） | https://arxiv.org/pdf/2403.19838 |
| OmniDrive | 2405.01533 | 見つからず（0件） | https://arxiv.org/pdf/2405.01533 |
| Senna | 2410.22313 | "Traffic Signal Detection"タスクはあるが"we primarily focus on the most critical one: traffic lights"（red / green / yellow / none） | https://arxiv.org/pdf/2410.22313 の知覚タスクの節 |
| DriveBench（Are VLMs Ready for AD?） | 2501.04003 | 標識タスクはなし。標識は回答例にのみ現れる | https://arxiv.org/pdf/2501.04003 |
| VLADBench | 2503.21505 | あり。専用のTraffic Signタスク（B.5） | https://arxiv.org/pdf/2503.21505 |
| IDKB | 2409.02914 | あり。CARLAの標識セット（B.6） | https://arxiv.org/pdf/2409.02914 |
| DriveQA | 2508.21824 | あり。CARLA上のMUTCD標識（B.7） | https://arxiv.org/pdf/2508.21824 |

結論：第1世代の運転VLM（2023年から2024年半ば）は、標識をタスクとしてモデル化していない。標識の理解が運転VLMのベンチマークに入ったのは知識テスト型のデータセット（IDKB、DriveQA、VLADBench）を通じてのみであり、いずれでもオープンな7B級モデルは標識でGPT-4oを大きく下回っている。

### B.13 Hugging Face上の道路標識向けにファインチューニングされたVLMチェックポイント

https://huggingface.co/api/models?search=traffic%20sign&limit=200 とその変種（"traffic-sign"、"trafficsign"、"road sign"）のクエリは、2026-10-31時点で重複を除いて146モデルを返し、そのほとんどがYOLO検出器であった。見つかったVLM / LoRAチェックポイント：

| モデル | ベース | 内容 | 根拠 |
|---|---|---|---|
| longhoang01/Qwen3-VL-8B-Instruct-Traffic-Signs | unsloth/Qwen3-VL-8B-Instruct | マージ済みのQwen3-VL-8B全体。カードはUnslothのテンプレートのみで、データセットも評価指標もない。apache-2.0、9ダウンロード | https://huggingface.co/api/models/longhoang01/Qwen3-VL-8B-Instruct-Traffic-Signs?full=true |
| RidaKo/qwen3-vl-road-signs-lt-lora | unsloth/Qwen3-VL-8B-Instruct-unsloth-bnb-4bit | リトアニアの道路標識のキャプション生成用LoRA。学習画像161枚、テスト画像50枚。ライセンスタグなし | https://huggingface.co/RidaKo/qwen3-vl-road-signs-lt-lora/raw/main/README.md |
| AKrasavcev/lora_gemma3_4b_lt_road_signs | google/gemma-3-4b-it | LoRA。リトアニアの道路標識のキャプション生成。gemmaライセンス | https://huggingface.co/api/models/AKrasavcev/lora_gemma3_4b_lt_road_signs?full=true |
| Anakonkai/vietnamese-traffic-sign-vqa-checkpoints | Qwen2.5-VL-3B | ベトナムの標識VQA用のQLoRAとDPOのアダプタ（独自のテストセットでVQA精度0.9379）、および標識生成用のStable Diffusion 1.5 LoRA。apache-2.0タグ | https://huggingface.co/Anakonkai/vietnamese-traffic-sign-vqa-checkpoints/raw/main/README.md |

これらのクエリでは、道路標識でファインチューニングしたFlorence-2、PaliGemma、LLaVA、SmolVLM、Ideficsのチェックポイントは見つからず、見つかったチェックポイントはいずれもGTSRB、TT100K、MTSDの精度を報告していない。したがって、標識の精度を上記の文献と比較できるオープンなVLMチェックポイントは存在しない。

---

## C. 標識に適用されたオープン語彙検出器とプロンプト可能な検出器

### C.1 標識のクラスを含む汎用検出ベンチマーク

| ベンチマーク | 標識関連のクラス | 根拠 |
|---|---|---|
| COCO (80) | `traffic light`（id 9）、`stop sign`（id 11）。汎用の道路標識クラスはなし | https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/coco.yaml |
| LVIS v1 (1203) | `stop_sign`（id 1019、頻度f）、`street_sign`（1026、f）、`signboard`（959、f）、`traffic_light`（1112、f）。汎用の道路標識クラスはなし | https://raw.githubusercontent.com/facebookresearch/detectron2/main/detectron2/data/datasets/lvis_v1_categories.py |
| Objects365 (365) | `Traffic Sign`（89）、`Stop Sign`（127）、`Speed Limit Sign`（267）、`Crosswalk Sign`（305）、`Traffic Light`（40）、`Traffic cone`（66） | https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/Objects365.yaml |
| Open Images V7 (601) | `Traffic sign`（549）、`Stop sign`（495）、`Traffic light`（548） | https://raw.githubusercontent.com/ultralytics/ultralytics/main/ultralytics/cfg/datasets/open-images-v7.yaml |

確認したGrounding DINO、OWL-ViT、YOLO-World、Deticの論文では、これらのクラスのクラス別APは公表されていない（表はデータセット単位のAPを報告している）。4つとも「未検証」とする。Objects365は、速度制限と横断歩道の標識クラスを持つ唯一の一般的な事前学習データセットである。Grounding DINO、YOLO-World、GLIP、Florence-2はいずれもObjects365で事前学習しているため、この点は重要である。

### C.2 標識に対するオープン語彙モデルやプロンプト可能なモデルの報告値

- 武漢の路側インフラのデータセットでのGrounding DINOのゼロショットとファインチューニング後の比較（Fu et al., Wuhan University; arXiv 2601.10551）：カテゴリ"Traffic sign"のゼロショットはmAP 28.4 / mAP@50 38.9、オープン語彙のファインチューニング後は63.8 / 77.7（上海のセットでは63.9 / 85.5）。信号機のゼロショットは33.3 / 57.9。その後、LoRAを適用したQwen-VLが属性（種類、形状、色、損傷）について推論する（https://arxiv.org/pdf/2601.10551 の、武漢 / 上海の結果を載せたページの表）。これは、見つかった中で"traffic sign"クラスに関する唯一の定量的なGrounding DINOの数値であり、GTSDB、TT100K、MTSDでの数値はない。
- Grounding DINO 1.5論文（https://arxiv.org/pdf/2405.10300 ）、Grounding DINO論文（検索で挙がったもの。https://arxiv.org/pdf/2303.05499 ）、YOLO-World（https://arxiv.org/abs/2401.17270 ）：道路標識に特化した数値はない（2405.10300では"traffic sign"は0件。他はgrepしておらず、未検証）。
- 道路標識セグメンテーションのためのSAM（Henninger et al., Fraunhofer IPM; Journal of Imaging 2024, 10(9), 220, CC BY）：SAM ViT-HとTinySAMのエンコーダに、学習したプロンプト埋め込みまたは畳み込みデコーダを組み合わせ、2,808枚で学習し、Cityscapes（標識画像3,281枚）、A2D2、ApolloScape、IDDA、Mapillary Vistasでクラス"traffic sign"について評価した。平均IoUの最高値は0.432（SAM-Conv-ViTH）で、Cityscapes、ApolloScape、MapillaryではなおMask R-CNNのほうが良い。学習画像が90枚の場合、SAM-Conv-ViTHのIoUの低下は0.15で、Mask R-CNNは0.26。インスタンス検出率は32.04%（SAM-Conv-ViTH）対22.60（Mask R-CNN）。結論：SAMはMask R-CNNを"does not clearly outperform"（明確には上回らない）（https://www.mdpi.com/2313-433X/10/9/220 ）。
- CdMT内の標識抽出器としてのSAM 2：ViT-Adapter検出器と比較されている。ViT-Adapterを用いた場合のtop-1はTT-100K / Sapporo / Yokohamaで0.93 / 0.84 / 0.80（https://arxiv.org/pdf/2409.01534 の表5）。
- NanoVerse YOLO + TSR-MCL（Lu et al.; arXiv 2507.23331）：YOLO-World型のRepVL-PAN検出器と、ViT / BERTによる対照学習の分類器の組み合わせ。TT100Kの全クラスのロングテール検出で78.4 mAP、精度91.8。このプレプリントにはまだプレースホルダーの引用（"[please supplement specific references]"）が残っている（https://arxiv.org/pdf/2507.23331 の要旨および第1節）。
- 使用したクエリ（2026-10-31のWeb検索）では、GTSDB、TT100K、MTSDでOWL-ViT / OWLv2、YOLO-World、YOLOE、Florence-2、Detic、GLIP、DINO-Xを報告した論文は見つからなかった。未解決の問題として扱い、F.3の再現実験7を参照のこと。

### C.3 オープン語彙検出器とセグメンテーションモデルのライセンス（OpenTSR向け）

| モデル | コードのライセンス | 重み（HFカード） | 根拠 |
|---|---|---|---|
| Grounding DINO | Apache-2.0 | IDEA-Research/grounding-dino-tiny apache-2.0 | https://img.shields.io/github/license/IDEA-Research/GroundingDINO.json, https://huggingface.co/api/models/IDEA-Research/grounding-dino-tiny |
| Grounding DINO 1.5 / DINO-X | Apache-2.0（APIクライアントのみ。重みは非公開） | なし | https://img.shields.io/github/license/IDEA-Research/Grounding-DINO-1.5-API.json, https://img.shields.io/github/license/IDEA-Research/DINO-X-API.json |
| MM-Grounding-DINO（mmdetection） | Apache-2.0 | なし | https://img.shields.io/github/license/open-mmlab/mmdetection.json |
| OWL-ViT / OWLv2 | Apache-2.0（scenic） | google/owlv2-base-patch16-ensemble apache-2.0, google/owlvit-base-patch32 apache-2.0 | https://img.shields.io/github/license/google-research/scenic.json, https://huggingface.co/api/models/google/owlv2-base-patch16-ensemble |
| YOLO-World | GPL-3.0 | なし | https://img.shields.io/github/license/AILab-CVC/YOLO-World.json |
| YOLOE | AGPL-3.0 | なし | https://img.shields.io/github/license/THU-MIG/yoloe.json |
| Florence-2 | MIT（重み） | microsoft/Florence-2-baseおよび-large mit | https://huggingface.co/api/models/microsoft/Florence-2-large |
| SAM / SAM 2 | Apache-2.0 | facebook/sam2-hiera-small apache-2.0 | https://img.shields.io/github/license/facebookresearch/segment-anything.json, https://img.shields.io/github/license/facebookresearch/sam2.json, https://huggingface.co/api/models/facebook/sam2-hiera-small |
| SAM 3 | "not identifiable by github"。HFの重みは"other"で、手動承認のゲートあり | | https://img.shields.io/github/license/facebookresearch/sam3.json, https://huggingface.co/api/models/facebook/sam3 |
| Detic | Apache-2.0 | なし | https://img.shields.io/github/license/facebookresearch/Detic.json |
| GLIP | MIT | なし | https://img.shields.io/github/license/microsoft/GLIP.json |

---

## D. 標識に関係する自己教師あり学習と大規模事前学習

### D.1 標識におけるDINOv2 / DINOv3とSSLエンコーダ

- DINOv2の最近傍特徴マッチングは、MVVの11クラス標識タスクで最良のゼロショット手法である（0.484、InternVL-3 9Bは0.468）（https://arxiv.org/pdf/2508.02047 の表3）。B.3を参照。
- DINOv3の表25（凍結した特徴量、512 px）：GTSRBでDINOv2 ViT-g 78.2、DINOv3-7B 87.5、PE-core ViT-G 94.8（https://arxiv.org/pdf/2508.10104 の付録B.4）。比較として、CLIP ViT-L/14の線形プローブは224 pxで92.5であり（https://arxiv.org/pdf/2103.00020 の表10）、このOCRに近いタスクでは言語で教師付けしたエンコーダがなお優位である。
- TS-1M（エンコーダ全体の教師ありファインチューニング）：DINO ViT-B/16 0.9190、MAE 0.9160、SimMIM 0.9403、MoCoV3 0.9385、SimCLR 0.9364に対し、CLIP 0.9533、ConvNeXt 0.9428（https://arxiv.org/pdf/2603.23034 の表IV）。著者らは、標識ではマスク画像型のSSLが"fall[s] below the supervised baseline"と結論づけている。
- GTSRBでのImageNet-21k ViTとBiT-Mの線形プローブは、CLIP（85から93）に比べて弱く（63から74）（https://arxiv.org/pdf/2103.00020 の表10）、ImageNetがすべての標識を1クラスにまとめていることと整合する。

### D.2 モデルマージとタスク算術の論文（GTSRBでファインチューニングしたCLIPの数値）

いずれもOpenAI CLIPのバックボーンを用い、ゼロショットのテキストヘッドを凍結して、2000イテレーション、バッチ128、lr 1e-5、コサインスケジュール、AdamW wd 0.1でファインチューニングしている（https://arxiv.org/pdf/2212.04089 の付録B.1）。

| バックボーン | ゼロショット（"Pre-trained"） | 個別にファインチューニング | 出典 |
|---|---|---|---|
| CLIP ViT-B/32 | 32.5 | 98.5 | https://arxiv.org/pdf/2212.04089 の表7 |
| CLIP ViT-B/16 | 43.4 | 99.0 | https://arxiv.org/pdf/2212.04089 の表6 |
| CLIP ViT-L/14 | 50.6 | 99.3 | https://arxiv.org/pdf/2212.04089 の表5 |
| CLIP ViT-B/32（TIES / AdaMergingの評価） | 32.6 | 98.7 | https://arxiv.org/pdf/2306.01708 の表11、https://arxiv.org/pdf/2310.02575 の表1 |
| CLIP ViT-L/14（TIES / AdaMergingの評価） | 50.5 | 99.2 | https://arxiv.org/pdf/2306.01708 の表12、https://arxiv.org/pdf/2310.02575 の表2 |

否定（忘却）の結果：GTSRBのタスクベクトルを差し引くと、GTSRBの精度は50.6から6.45に下がる（ViT-L/14）一方、対照のImageNetは72.2を保つ（https://arxiv.org/pdf/2212.04089 の表5）。8タスクをマージしたモデルのGTSRBでの精度：ViT-B/32ではTask Arithmetic 69.7、TIES 72.1、Layer-wise AdaMerging 91.9、AdaMerging++ 89.0（従来のマルチタスク学習は98.9）。ViT-L/14ではTask Arithmetic 86.7、TIES 83.3、AdaMerging 98.0、AdaMerging++ 97.5（https://arxiv.org/pdf/2310.02575 の表1および表2。TIESの値は次の表11および表12にもある：https://arxiv.org/pdf/2306.01708 ）。つまりGTSRBはマージによって最も性能が落ちるタスクであり、標識のヘッドを汎用のマージ済みCLIPに組み込む計画への警告となる。

これらの文献で使われているチェックポイントとデータ：
- Hugging Face上のtanganke/gtsrb：parquet形式で、train 26,640、test 12,630に加え、劣化を加えたテスト分割7種（contrast、gaussian_noise、impulse_noise、jpeg_compression、motion_blur、pixelate、spatter）が各12,630枚。ダウンロード数16,418。クラス名はOpenAI CLIPの文字列。カードにライセンスの欄はない（https://huggingface.co/api/datasets/tanganke/gtsrb?full=true, https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md ）。
- tanganke/clip-vit-base-patch32_gtsrb：Adam、lr 1e-5、4000ステップ、バッチ32で視覚エンコーダをファインチューニングしたもの。カードは事前学習時0.3043、ファインチューニング後0.9881と報告している（https://huggingface.co/tanganke/clip-vit-base-patch32_gtsrb/raw/main/README.md 。ダウンロード数は19,362：https://huggingface.co/api/models/tanganke/clip-vit-base-patch32_gtsrb?full=true ）。tanganke/clip-vit-large-patch14_gtsrb：事前学習時0.4382、ファインチューニング後0.9932（https://huggingface.co/tanganke/clip-vit-large-patch14_gtsrb/raw/main/README.md ）。tangankeの事前学習時の値（30.4、43.8）は、テンプレートがOpenAIのものであるにもかかわらず（https://raw.githubusercontent.com/tanganke/fusion_bench/main/fusion_bench/tasks/clip_classification/gtsrb.py ）、CLIP論文の値（32.2、50.3）より低い。原因（HFの前処理とOpenAIの前処理の違い）は未検証である。B/16のカードにはREADMEがない（https://huggingface.co/tanganke/clip-vit-base-patch16_gtsrb/raw/main/README.md は"Entry not found"を返す）。これらのモデルカードにライセンスタグはない。
- fusion_bench：MIT、スター244、最終コミットは2026-10-31までの1週間以内（https://img.shields.io/github/license/tanganke/fusion_bench.json, https://img.shields.io/github/stars/tanganke/fusion_bench.json, https://img.shields.io/github/last-commit/tanganke/fusion_bench.json ）。task_vectorsリポジトリ：ライセンスは"not specified"、スター552（https://img.shields.io/github/license/mlfoundations/task_vectors.json ）。

### D.3 ロバスト性と画像劣化

- AdaMergingの表4（4タスクでマージしたViT-B/32、GTSRB列）：Task Arithmetic / AdaMergingの順に、劣化なし75.1 / 96.3、motion blur 64.2 / 94.1、impulse noise 40.4 / 77.5、gaussian noise 49.4 / 76.0、pixelate 66.6 / 90.6、spatter 54.3 / 89.4、contrast 70.6 / 95.1、JPEG 60.0 / 90.1（https://arxiv.org/pdf/2310.02575 の表4）。劣化を加えた分割は、tanganke/gtsrbに同梱されているもの（D.2）である。
- Synset Signset Germanyは、感度分析のための画像ごとの環境と撮像効果のパラメータ、および別の"SynsetSignsetSensitivityAnalysisDataset"（サイズ区分1Mから10M、CC BY 4.0）を提供している（https://huggingface.co/api/datasets?search=synset&limit=50 ）。E.1を参照。
- CURE-TSR（チャレンジレベル別の劣化、2M+枚、合成によるバイアス）は、MVVで"Cls. only"、"Synthetic bias, no masks"として挙げられている（https://arxiv.org/pdf/2508.02047 の表1）。ここでは再検証していない。

---

## E. 標識の生成データ

### E.1 Synset Signset Germany（Fraunhofer IOSB / IPA, ITSC 2024; arXiv 2512.05936）

- 内容：GANで生成した汚れと摩耗のテクスチャを用いる、物理ベースレンダリングのパイプライン（OCTASプラットフォーム。CyclesパストレーサとOGREラスタライザ）。2020年に導入された標識を含むドイツの標識211クラス、クラスあたり500枚、レンダラごとに105,500枚（合計211,000枚）で、画像ごとにマスク、セグメンテーション画像、完全なメタデータ（環境、照明、撮像効果）が付く（https://arxiv.org/pdf/2512.05936 の要旨および第III節、https://synset.de/datasets/synset-signset-ger/ ）。
- 下流の結果（ConvNeXt-Small、top-1）：Signset Cyclesのみで学習し実データのGTSRBでテストすると98.3。Signset OGREで学習しGTSRBでテストすると98.2。GTSRBで学習しGTSRBでテストすると99.9。GTSRBで学習しSignset Cyclesでテストすると89.4。CATERED（以前の合成データセット）で学習しGTSRBでテストすると76.4。Signsetで学習したモデルはCATEREDで84.4に達する（https://arxiv.org/pdf/2512.05936 の表II）。つまり、合成データのみでの学習は、実データのテストセットにおいて実データでの学習より1.6ポイント低いだけであり、以前の合成データセットより約20ポイント良い。
- ライセンス："This work is licensed under Creative Commons Attribution 4.0"であり、再配布の際には引用とクレジット表記が求められている（https://synset.de/datasets/synset-signset-ger/ ）。Hugging Faceのミラーとして、FraunhoferIOSB/Synset-Signset-Germany（cc-by-4.0、cycles/ogreのtrainとvalidationのparquet）とFraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset（cc-by-4.0、GTSRBの43クラス、21,500枚で、GTSRBの"synthetic twin"。cycles/train.parquetは1,495,705,131バイト）がある（https://huggingface.co/api/datasets/FraunhoferIOSB/Synset-Signset-Germany?full=true, https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/raw/main/README.md, https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/resolve/main/cycles/train.parquet ）。同じグループによる関連のCC BY 4.0のセット：Synset-Boulevard、Synset-Background-Effect-Datasets（https://huggingface.co/api/datasets?search=synset&limit=50 ）。
- 意義：見つかった中で唯一の、寛容なライセンスで大規模かつ効果が測定された合成標識データセットである。パイプラインはテンプレート駆動（標識の形状はドイツの公式カタログに基づく）なので、OpenTSRが独自にテンプレートをレンダリングすれば同じ手法を日本のカタログにも適用できる。ただし、OCTASのパイプライン自体は公開されていない（未検証）。

### E.2 実シーンへの標識の拡散インペインティング（AMAP / Alibaba; arXiv 2609.02348, Sep 2026）

- 手法：Stable Diffusion 1.5のインペインティング用バックボーン（約1.4Bパラメータ）に、3つの事前情報を与える。JSONのテキストプロンプト（種類と値）、測定した主要色で塗った正面視のベクタテンプレート（IP-Adapter経由）、同じテンプレートをフィットさせた標識のシルエットにアフィン変換で重ねたもの（ControlNet経由）である。さらに物理的整合性の損失（CIELABの彩度とSobel勾配）を加える（https://arxiv.org/pdf/2609.02348 の要旨、https://raw.githubusercontent.com/52hz-whale/TrafficSignInpaint/main/README.md ）。
- 評価：AMAPの社内データで学習し、TT100K-2021の5,182領域で、FLUX.1 Fill [dev]（12B）を含む7つのインペインティングのベースラインに対してゼロショットで評価した。9つの画像評価指標のすべてで1位であり、レイテンシはFLUX Fillの123.55 sの1/14である（https://arxiv.org/pdf/2609.02348 の表2および第4節）。
- 下流のTSRの改善：TT100Kの希少クラスでYOLO26を用いた場合、10%から100%のデータ割合において、合成データは希少クラスのグループ単位でプールしたAP50を"by 1.23x to 7.40x over a real-data-only baseline"で引き上げる。Copy-Pasteは"reaches most of that improvement"とされ、拡散が勝るのは主に最も難しい記号クラスである（対象12クラスの平均AP50は0.770、hard pasteは0.606、Poisson pasteは0.481）。ヘッドクラスは変わらない（0.966 / 0.954 / 0.970 / 0.963）（https://arxiv.org/pdf/2609.02348 の第4.4節および表6）。
- コード：https://github.com/52hz-whale/TrafficSignInpaint 。Apache-2.0で、2026-10-31時点のスターは0（https://img.shields.io/github/license/52hz-whale/TrafficSignInpaint.json, https://img.shields.io/github/stars/52hz-whale/TrafficSignInpaint.json ）。なお、ベースとなるSD 1.5の重みはCreativeML OpenRAIL-Mである（https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5 ）。

### E.3 以前のGANおよびコピーペーストの研究

- Konushin, Faizov, Shakhuro（MSU; arXiv 2101.04927）：ロシアの希少な標識（RTSD。205クラスのうち99クラスはテストセットにのみ現れる）を、高解像度のアイコンを用いて実際の道路画像に埋め込む。VAEがもっともらしい標識の位置を予測し、GANに基づく3つの外観整合手法を単純な"Synt"の貼り付けと比較している。合成データと実データを組み合わせると、希少クラスについて"improves the accuracy of both classifier and detector"とされる（https://arxiv.org/pdf/2101.04927 の要旨および第4節）。正確な改善幅はPDFの表3から表8にある（ここには転記していない）。
- テンプレートを用いたコピーペーストは、依然として最も強力な低コストのベースラインである。AMAPの研究では、hardおよびPoissonのコピーペーストが、プールしたAP50における拡散による改善の大部分を取り戻している（https://arxiv.org/pdf/2609.02348 の第4.4節）。

### E.4 Hugging Face上の標識生成用コミュニティLoRA

同じHFの検索（B.13）から：Nyanmaru/TS_LoRA_sd3_uneven-road-signとTS_LoRA_sd3_slipper-road-condition-sign（ベースはstabilityai/stable-diffusion-3-medium-diffusers、タグはopenrail++）、CiroN2022/road-sign（ベースはSDXL、ライセンスは"other"）、Muapi/road-sign-erector-flux（ベースはFLUX.1-dev、タグはopenrail++）、およびAnakonkai/vietnamese-traffic-sign-vqa-checkpoints内のSD 1.5 LoRA（https://huggingface.co/api/models?search=road%20sign&limit=200 ）。いずれも下流のTSRの測定は報告していない。FLUX.1-devのベースは非商用である（E.6）。

### E.5 世界モデルと運転映像の生成器：標識を描画するか、どのライセンスか

| 生成器 | 論文 | 確認された標識の扱い | コードのライセンス | 重み |
|---|---|---|---|---|
| GAIA-1 (Wayve) | https://arxiv.org/pdf/2309.17080 | 標識関連の語は0件 | 非公開 | 非公開 |
| GAIA-2 (Wayve) | https://arxiv.org/pdf/2503.20523 | "Speed limits"、"pedestrian crossings, traffic lights and their states"、一方通行の表示を含むメタデータで条件付けする（メタデータによる条件付けの節）。標識認識の評価はなし | 非公開 | 非公開 |
| Cosmos (NVIDIA) | https://arxiv.org/pdf/2501.03575 | 標識や信号機に関する語は0件 | Apache-2.0（https://img.shields.io/github/license/nvidia-cosmos/cosmos-predict1.json, cosmos-predict2, cosmos-transfer1） | "nvidia-open-model-license"（https://huggingface.co/api/models/nvidia/Cosmos-Predict2-2B-Video2World ） |
| DriveDreamer | https://arxiv.org/pdf/2309.09777 | 0件 | Apache-2.0（https://img.shields.io/github/license/JeffWang987/DriveDreamer.json ） | 未検証 |
| Vista (OpenDriveLab) | https://arxiv.org/pdf/2405.17398 | 0件 | Apache-2.0（https://img.shields.io/github/license/OpenDriveLab/Vista.json ） | apache-2.0（https://huggingface.co/api/models/OpenDriveLab/Vista ） |
| GenAD / OpenDV (OpenDriveLab) | https://arxiv.org/pdf/2403.09630 | 0件 | Apache-2.0（https://img.shields.io/github/license/OpenDriveLab/DriveAGI.json ） | 未検証 |
| MagicDrive | https://arxiv.org/pdf/2310.02601 | 0件 | Apache-2.0（https://img.shields.io/github/license/cure-lab/MagicDrive.json ） | 未検証 |
| Panacea | https://arxiv.org/pdf/2311.16813 | 0件 | Apache-2.0（https://img.shields.io/github/license/wenyuqing/panacea.json ） | 未検証 |
| DrivingDiffusion | https://arxiv.org/pdf/2310.07771 | 0件 | MIT（https://img.shields.io/github/license/shalfun/DrivingDiffusion.json ） | 未検証 |
| SimGen | https://arxiv.org/pdf/2406.09386 | プロンプト例が1つ："A motorcyclist on a road with a speed limit sign" | "not specified"（https://img.shields.io/github/license/metadriverse/SimGen.json ） | 未検証 |

所見：運転の世界モデルの論文はいずれも、標識の判読性や下流のTSRを評価しておらず、速度制限で条件付けするのはGAIA-2（非公開）のみである。レイアウトで条件付けする生成器（MagicDrive、Panacea、DrivingDiffusion）はnuScenesのボックスと地図で条件付けするが、それらには標識クラスがないため、標識は制御されない背景としてのみ現れる。OpenTSRにとって、世界モデルは現時点ではせいぜい背景と天候の多様性に役立つ程度で、標識の内容を制御する用途には使えない。標識レベルの制御が可能なのは、インペインティング（E.2）とテンプレートのレンダリング（E.1）のみである。

### E.6 標識の合成に使える画像生成器のライセンス

| 生成器 | ライセンス（HFのタグ） | 根拠 |
|---|---|---|
| Stable Diffusion 1.5 | creativeml-openrail-m | https://huggingface.co/api/models/stable-diffusion-v1-5/stable-diffusion-v1-5 |
| SDXL base 1.0 | openrail++ | https://huggingface.co/api/models/stabilityai/stable-diffusion-xl-base-1.0 |
| Stable Diffusion 3 Medium | "stabilityai-ai-community"（other）、ゲートあり | https://huggingface.co/api/models/stabilityai/stable-diffusion-3-medium |
| FLUX.1-dev, FLUX.1-Fill-dev | "flux-1-dev-non-commercial-license"、ゲートあり | https://huggingface.co/api/models/black-forest-labs/FLUX.1-dev, https://huggingface.co/api/models/black-forest-labs/FLUX.1-Fill-dev |
| FLUX.1-schnell | apache-2.0 | https://huggingface.co/api/models/black-forest-labs/FLUX.1-schnell |
| Cosmos Predict 1 / 2の重み | nvidia-open-model-license | https://huggingface.co/api/models/nvidia/Cosmos-1.0-Diffusion-7B-Text2World |
| Stabilityのgenerative-modelsのコード | MIT。black-forest-labs/fluxのコードはApache-2.0 | https://img.shields.io/github/license/Stability-AI/generative-models.json, https://img.shields.io/github/license/black-forest-labs/flux.json |

OpenRAIL-MとOpenRAIL++は、用途に基づく制限付きで商用利用を認めている。生成した画像をApache-2.0やCC BYで再配布してよいかどうかは、モデルのライセンスと各地の法律に依存し、ここでは結論を出していない（未検証）。FLUX.1-devの出力は、その非商用ライセンスによって制限される。

---

## F. 総括

### F.1 まとめの表

| モデルまたは論文 | データセット | プロトコル | 評価指標と値 | オープンウェイト | ライセンス | 出典 |
|---|---|---|---|---|---|---|
| CLIP ViT-B/32 | GTSRBテスト | ゼロショット、テンプレート3個 | 32.2 top-1 | あり | MIT | https://arxiv.org/pdf/2103.00020 の表11 |
| CLIP ViT-L/14-336 | GTSRBテスト | ゼロショット | 52.3 | あり | MIT | 同上 |
| CLIP ViT-L/14 | GTSRB | 線形プローブ | 92.5 | あり | MIT | 同上、表10 |
| DFN-5B ViT-H/14-378 | GTSRBテスト | ゼロショット（CLIP_benchmark） | 67.8 | あり | apple-amlr | open_clipの結果CSV |
| EVA02-E-14-plus | GTSRB | ゼロショット | 67.4（CSV）/ 67.7（論文） | あり | MIT | CSV、https://arxiv.org/pdf/2303.15389 |
| SigLIP SO400M-384 | GTSRB | ゼロショット | 64.3 | あり | Apache-2.0 | open_clipの結果CSV |
| SigLIP ViT-B/16 | GTSRB | ゼロショット | 51.6 | あり | Apache-2.0 | open_clipの結果CSV |
| MetaCLIP ViT-H/14 2.5B | GTSRB | ゼロショット | 62.5 | あり | CC-BY-NC 4.0 | https://arxiv.org/pdf/2309.16671 |
| LAION-2B ViT-H/14 | GTSRB | ゼロショット / 同時FT / パッチ適用 | 58.4 / 99.0 / 98.8 | あり | MIT相当 | https://arxiv.org/pdf/2212.07143 |
| CLIP ViT-B/32、B/16、L/14（ファインチューニング済み） | GTSRB | フルFT、テキストヘッドは凍結 | 98.5 / 99.0 / 99.3 | あり（tanganke） | 記載なし | https://arxiv.org/pdf/2212.04089 |
| DINOv2 ViT-g / DINOv3-7B / PE-core G | GTSRB | 凍結した特徴量、512 px | 78.2 / 87.5 / 94.8 | あり / ゲートあり / あり | Apache-2.0 / 独自 / 未検証 | https://arxiv.org/pdf/2508.10104 の表25 |
| CdMT GPT-4o（北海道大学） | GTSRB、BTSD、TT-100Kのサブセット、Sapporo、Yokohama | 多段階プロンプトによるゼロショットLMM | top-1 0.93 / 0.88 / 0.97 / 0.89 / 0.85 | なし（API） | 該当なし | https://arxiv.org/pdf/2409.01534 |
| 素のGPT-4o / GPT-4V | 同上 | ゼロショット | 0.89 / 0.81（GTSRBサブセット） | なし | 該当なし | 同上 |
| LLaVA-1.5 / LLaVA-NeXT / VITA-1.5 | GTSRBサブセット | ゼロショット | 0.32 / 0.39 / 0.45 | あり | llama2ほか | 同上 |
| InternVL-3 9B / Gemma-3 12B / DINOv2 kNN | MVV（11クラス） | ゼロショット | 0.468 / 0.428 / 0.484 | あり | Apache-2.0（InternVL3）、gemma、Apache-2.0 | https://arxiv.org/pdf/2508.02047 |
| CLIP ViT-B/32エンコーダ + ヘッド | TS-1M（454クラス） | 教師あり | 0.9533 top-1 | あり | MIT | https://arxiv.org/pdf/2603.23034 |
| TSCLIP | CRTS（46クラス、10地域） | 中国+スロベニアでFT、ドイツでテスト | 0.8708（ゼロショット0.2775） | コードは公開予定 | MIT（リポジトリ） | https://arxiv.org/pdf/2409.15077 |
| Qwen2-VL-72B / GPT-4o / Qwen2.5-VL-7B | VLADBench Traffic Sign | 多肢選択 | 71.18 / 68.96 / 61.71 | あり / なし / あり | Qwenライセンス / 該当なし / Apache-2.0 | https://arxiv.org/pdf/2503.21505 |
| GPT-4o / Gemini-1.5-flash / オープンの最高値（XComposer2） | IDKB Driving Road Data（CARLAの標識） | 多肢選択 | 0.87 / 0.85 / 0.62 | なし / なし / あり | 該当なし | https://arxiv.org/pdf/2409.02914 |
| GPT-4o / Mini-InternVL-2B（ゼロショット / ファインチューニング後） | DriveQA-Vの規制標識（CARLA） | 多肢選択 | 93.75 / 64.06 / 93.79 | なし / あり | 該当なし | https://arxiv.org/pdf/2508.21824 |
| InternVL3-78B / Qwen2.5-VL-72B / SignReasoner-7B | TrafficSignEval（複雑な中国の標識） | キャプションプロトコル | 68.02 / 65.12 / 72.67 | あり / あり / 未検証 | | https://arxiv.org/pdf/2604.10436 |
| SignRAG（Gemini 2.5 Flash + RAG） | オハイオ州のMUTCDデザイン303件 / 実画像181件 | ゼロショット | 95.58 / 82.45 | なし | 該当なし | https://arxiv.org/pdf/2512.12885 |
| GPT-4V / LLaVA-13B対ResNeXt | NDDの敵対的標識（2値） | ゼロショット | 84.06 / 80.81対17.17 | | | VehicleSec 2024のPDF |
| Grounding DINO | 武漢の路側、traffic signクラス | ゼロショット / ファインチューニング後 | mAP@50 38.9 / 77.7 | あり | Apache-2.0 | https://arxiv.org/pdf/2601.10551 |
| SAM-Conv-ViTH対Mask R-CNN | Cityscapesほか4件、traffic sign | ファインチューニングしたデコーダ | 平均IoU 0.432。5件中3件でMask R-CNNが上回る | あり | Apache-2.0 | https://www.mdpi.com/2313-433X/10/9/220 |
| Synset Signset Germany（ConvNeXt-S） | 合成データで学習、GTSRBでテスト | 教師あり | 98.3 top-1 | データセットはあり | CC BY 4.0 | https://arxiv.org/pdf/2512.05936 |
| AMAPの拡散インペインティング（YOLO26） | TT100Kの希少クラス | 合成データによる拡張 | プールしたAP50でx1.23からx7.40 | コードはあり | コードはApache-2.0、SD1.5の重みはOpenRAIL-M | https://arxiv.org/pdf/2609.02348 |

### F.2 OpenTSRにとっての意味

1. VLMは、細粒度の標識において専用の分類器の代わりにはまだならない。GTSRB型のタスクにおけるオープンなVLMのゼロショットの数値はすべてtop-1で0.32から0.68の間にあり（B.1、B.3、B.8）、最良のクローズドモデル（GPT-4o）が0.89から0.93に達するのは、層化抽出した小さなサブセットで、かつ大掛かりなプロンプトの足場を組んだ場合に限られる（B.1、B.2）。一方、教師ありCNNやファインチューニングしたCLIPエンコーダは98.5から99.9にある（A.4、D.2、E.1）。路上レベルの11クラスのMVVタスクでは、言語を使わないDINOv2のマッチャーがInternVL-3とGemma-3を上回る（B.3）。OpenTSRにおけるVLMの妥当な役割は、（a）NECが行ったような、人手のレビューを伴う半自動のラベル付けと再ラベル付け（B.3）、（b）テキストの多い案内標識や補助標識の解釈（SignEye、SignReasoner、SignRAG。B.8からB.10）、（c）敵対的な標識や分布外の標識のスクリーニングである。（c）では、GPT-4V級のモデルがGTSRBで学習したCNNを大差で上回る（B.11）。
2. CLIP系のエンコーダは、寛容なライセンスで使える分類器の出発点として最良である。ファインチューニングしたCLIP ViT-B/32はGTSRBで98.5から98.8に達し（D.2）、CLIP ViT-B/32はTS-1Mで20のエンコーダ中最良であり（B.4）、このOCRに近いタスクでは言語で教師付けした特徴量がSSLの特徴量を圧倒する（D.1）。寛容なライセンスの選択肢：OpenAI CLIP（MIT）、LAION / DataCompのopen_clip重み（MIT）、SigLIPとSigLIP 2（Apache-2.0）、EVA-CLIP（MIT）。MetaCLIP（CC-BY-NC 4.0）は避け、DFN（apple-amlr）とDINOv3（独自のゲート付きライセンス）は、それらから派生した重みを公開する前に法的な確認が必要なものとして扱う（A.3、A.4）。
3. ゼロショットCLIPは、弱い事前分布として、あるいはクラスの発見にしか役立たない。43クラスで32%から68%にとどまり、速度制限についてはテキスト埋め込みがほぼ縮退しているため（A.2、A.5）、速度の値をラベル付けすることはできない。テンプレート駆動の手法（DINOv2またはCLIPの画像埋め込みによるピクトグラムのプロトタイプ。A.6、B.3）が、日本の新しいクラスに対する信頼できるゼロショットの道筋であり、OpenTSRではどのVLMよりも先にこれを測定すべきである。
4. オープン語彙検出器は、クラスに依存しない標識のボックスは出せるが、細粒度のクラスは出せない。Grounding DINOのゼロショットの"traffic sign"のmAP@50は路上レベルのセットで38.9であり、ファインチューニング後は77.7に跳ね上がる（C.2）。Objects365での事前学習により、これらの検出器は"Speed Limit Sign"や"Crosswalk Sign"をある程度理解している（C.1）。Grounding DINO、OWLv2、Florence-2はApache-2.0またはMITである。YOLO-World（GPL-3.0）とYOLOE（AGPL-3.0）は、そのコードを同梱するApache-2.0のリリースとは両立しない（C.3）。標識のマスクではSAMがMask R-CNNを明確には上回らないため（C.2）、マスクは労せずして得られる利点ではない。
5. 合成データは、OpenTSRが最も大きな成果を得られる領域である。Synset Signset Germany（CC BY 4.0）は、テンプレートからレンダリングした合成セットだけで、GTSRBで98.3%の分類器を学習できることを示しており、そのGTSRBサブセットはドメインギャップを調べるための既製の"synthetic twin"である（E.1）。日本についても同じレシピ（道路標識のカタログの公式ピクトグラム、物理ベースレンダリング、GANまたは拡散による摩耗テクスチャ）が、写真のライセンスの問題が生じないため、再配布可能なデータセットへの最も妥当な道筋である。実シーンへの標識のインペインティングについては、AMAPのレシピ（SD 1.5インペインティング + ControlNetによる幾何 + IP-Adapterによる色）が、Apache-2.0のコードとOpenRAIL-Mの重みで公開されており、FLUX Fillより14倍安価である。希少クラスで測定された改善は本物だが、その大部分はコピーペーストでも得られる（E.2）。運転の世界モデルは標識の内容を制御しないため、現時点では標識クラスの不足を埋めることを期待すべきではない（E.5）。
6. モデルマージの文献は、道具ではなく注意喚起である。CLIPの標識ヘッドを他のタスクとマージすると、GTSRBは一貫して最も性能が落ちるタスクである（D.2）。

### F.3 Hugging Faceにアクセスできる2コア、RAM 7 GBのCPUマシンで実施可能な再現実験

いずれも `clip-benchmark/wds_gtsrb` のテストシャード（tarファイル3個、各約41 MB、A.3）または `tanganke/gtsrb`（テストのparquetは約105 MB、D.2）を使う。おおよそのCPUコストは、2コアで、224 pxのViT-B/32が毎秒約15から25枚、ViT-L/14が毎秒約1枚と仮定している。これらは推定であり、測定値ではない。

1. GTSRBのテストセット全体（12,630枚）でのCLIP ViT-B/32のゼロショット。43個の公式クラス名と3個のテンプレートを使い（A.2）、OpenAIの重みをopen_clip経由またはHFの `openai/clip-vit-base-patch32` で読み込む。32.2（https://arxiv.org/pdf/2103.00020 の表11）、32.6（open_clipのCSV）、32.5（https://arxiv.org/pdf/2212.04089 の表7）、30.4（tangankeのカード）と比較する。数字に関する失敗モード（A.5）を定量化するため、9つの速度制限クラスについてクラス別の混同行列を加える。これは取得したどの論文も報告していない。所要時間の見積もりは15から20分。
2. 同じセットでのopen_clipのViT-B-32 laion2b_s34b_b79kとViT-B-16 datacomp_xl_s13b_b90kのゼロショット。49.3および55.2と比較する（https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv ）。見積もりはそれぞれ20から40分。
3. SigLIP `google/siglip-base-patch16-224` のゼロショット（シグモイドのロジット、同じテンプレート）。open_clipのCSVにあるViT-B-16-SigLIPの51.6と比較する。続いて同一のプロンプトで `google/siglip2-base-patch16-224` を実行する。GTSRBの公表値は存在しないため（A.4）、これは新しいデータ点になる。見積もりはそれぞれ30から60分。
4. 凍結したCLIP ViT-B/32の画像特徴量に対する線形プローブ：train 26,640枚とtest 12,630枚の特徴量を抽出し、scikit-learnのロジスティック回帰を当てはめる。85.3（https://arxiv.org/pdf/2103.00020 の表10）と比較する。`facebook/dinov2-small` または `dinov2-base` の特徴量でも繰り返し、文献ではViT-g、512 pxでしか報告されていないDINOv2のデータ点を得る（78.2、https://arxiv.org/pdf/2508.10104 の表25）。見積もりは1から2時間。
5. 公開されているファインチューニング済みエンコーダ `tanganke/clip-vit-base-patch32_gtsrb` を、凍結したゼロショットのテキストヘッドとともにtest分割で評価する。カードの0.9881（https://huggingface.co/tanganke/clip-vit-base-patch32_gtsrb/raw/main/README.md ）および98.5（https://arxiv.org/pdf/2212.04089 の表7）と比較する。tanganke/gtsrbの劣化を加えた7つの分割でも評価し、劣化なしから劣化ありへの低下幅をAdaMergingの単一タスクの数値と比較する（https://arxiv.org/pdf/2310.02575 の表4。ただしそれらはマージ済みモデルなので、低下幅はより小さくなると予想される）。見積もりは劣化なしで20分、劣化ありでさらに約2時間。
6. 層化抽出したGTSRBサブセット（クラスあたり10枚、計430枚）での小型VLMによる多肢選択：SmolVLM-256M（Apache-2.0）、Moondream2（Apache-2.0）、Florence-2-base（MIT。キャプションまたはOCRのタスクを使い、そのテキストをクラス名と照合する）、Qwen2-VL-2B-Instruct（Apache-2.0。CPU上でint8またはbf16、1枚あたり約10から30 s）。プロンプト：43個のクラス名を選択肢として示し、選択肢の文字で答えさせる。北海道大学グループのGTSRBサブセットでのLLaVA-1.5 0.32、LLaVA-NeXT 0.39、CLIP ViT-B/32 0.24、GPT-4o 0.89と比較する（https://arxiv.org/pdf/2409.01534 の表1）。数時間かかりそうなのはQwen2-VL-2Bだけで、他は1時間以内に収まる。
7. OWLv2（`google/owlv2-base-patch16-ensemble`、Apache-2.0）とGrounding DINO tiny（`IDEA-Research/grounding-dino-tiny`、Apache-2.0）による、GTSDBのテスト画像（300枚。ボックスはデータセット調査から入手可能）でのクラス非依存の標識検出。プロンプトは"traffic sign"とし、統合した単一クラスのAP50を報告する。どちらのモデルについてもGTSDBの公表値は存在しない（C.2）。最も近い参照値は、武漢の路上の標識でのGrounding DINOのゼロショットmAP@50 38.9である（https://arxiv.org/pdf/2601.10551 ）。見積もりは1から2時間（960 pxのOWLv2はCPUでは遅い。必要なら640 pxに縮小し、その旨を報告する）。
8. 凍結した特徴量による合成から実データへの転移：`FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset` を読み込み（cyclesのvalidationのparquet、またはtrainの数千行をストリーミング）、実験4のロジスティック回帰のプローブを合成データの特徴量のみで学習し、実データのGTSRBテストセットでテストする。論文の全体学習での結果（合成データのみ98.3対実データ99.9。https://arxiv.org/pdf/2512.05936 の表II）と定性的に比較する。凍結特徴量の版はそれより低くなるが、ほとんど計算なしでドメインギャップを切り分けられる。見積もりは1.5 GBのダウンロードを含めて1から2時間。

---

## G. 取得できなかったページ

- https://guoyangzhao.github.io/projects/ts1m （WebFetch経由。許可リクエストに応答がなかった）。代わりにcurlでページを取得したため、TS-1Mに関する記述はcurlで得たテキストとGitHubのREADMEに基づく。
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12193857/ とそのリダイレクト先（https://pmc.ncbi.nlm.nih.gov/articles/PMC12193857 ）（"SAM for Road Object Segmentation: Promising but Challenging"）：許可リクエストに応答がなかった。使用していない。
- https://doi.org/10.3390/ai7050153 （"Automated Synthetic Traffic Dataset Generation via Diffusion-Based Inpainting Pipeline"、MDPI AI）：時間の都合で取得していない。使用していない。
- https://huggingface.co/tanganke/clip-vit-base-patch16_gtsrb/raw/main/README.md ："Entry not found"（モデルカードなし）。そのため、このチェックポイントのB/16の事前学習時とファインチューニング後の値は不明である。
- https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/raw/main/README.md ："Entry not found"（カードなし）。そのため、データセットのライセンスは不明である。
- TSCLIP論文の本文からはCLIPのバックボーンのサイズが得られなかった（解析したのはPDFのみ）。未検証。
- CdMT（2409.01534）、少数ショットICL（2407.05814）、SignReasoner（2604.10436）、TSI-CN（2311.10793）のPDF中には、コードやデータのURLが見つからなかった。これらのデータセットの公開状況は不明である。
- Grounding DINO、OWL-ViT、YOLO-World、DeticのLVIS、Objects365、Open Imagesにおけるstop sign / traffic signのクラス別AP：各論文には見つからなかった。未検証。
- https://github.com/nec-labs-ma/relabeling ：READMEはraw.githubusercontent.com経由で取得した（出力の上限により"Getting Started"の節以降は切り詰められた）。LICENSEファイルの内容は取得しておらず、GitHubは"not identifiable by github"と報告している。
