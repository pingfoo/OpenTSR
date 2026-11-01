# サードパーティのアセット

> 本文書は英語版 [docs/third_party.md](../third_party.md) の日本語訳である（原文 PR #7 `phase0-skeleton` 時点）。内容に食い違いがある場合は英語版を正とする。

OpenTSRが基盤とする、または測定対象とするすべてのデータセット、モデル、コードベースを、そのライセンスと出典とともに示す。「OpenTSRでの利用」の列は、本プロジェクトのApache-2.0による公開方針（`docs/decisions.md`を参照）の下で、それをどのように利用することが認められるかを述べる。ライセンスは、記載された日付の時点で出典ページに書かれていたとおりに引用している。再配布の前には必ず再確認すること。

本ファイルは、両トラックで共有するアセットと、道路標識トラック（主トラック）のアセットを扱う。信号機トラックのみで用いるアセットは`traffic_lights/docs/third_party.md`に列挙している。

## 道路標識データセット

| アセット | 出典 | ライセンス（原文どおり） | OpenTSRでの利用 |
|---|---|---|---|
| GTSRB（German Traffic Sign Recognition Benchmark、IJCNN 2011） | 公式：<https://benchmark.ini.rub.de/gtsrb_dataset.html>（ファイルはsid.erda.dkでホストされており、クラウドセッションからは到達できない）。本プロジェクトで使用：Hugging Faceミラー<https://huggingface.co/datasets/bazyl/GTSRB>、リビジョン`a8093a9c7757b59d64702f892002542e8f3a1fb0`（Kaggle版のparquetエクスポート。両parquetファイルのsha256は`scripts/prepare_gtsrb.py`にある）、train 39,209 / test 12,630、公式の分割サイズと照合済み | 公式ページ："The data is free to use. However, we cordially ask you to cite the following publication if you do"（Stallkamp et al., IJCNN 2011）。ミラーのカードにはミラー作成者が付けた`gpl-3.0`タグがあるが、これはベンチマークのライセンスではない | 標識の最初の分類ベンチマーク（`docs/replication.md`）。派生ラベルとクロップの再配布は"free to use"の記述により認められる。論文を引用すること。 |
| GTSDB（German Traffic Sign Detection Benchmark、IJCNN 2013） | 公式：<https://benchmark.ini.rub.de/gtsdb_dataset.html>。本プロジェクトで使用：Hugging Faceミラー<https://huggingface.co/datasets/GlowBond/GTSDB>、リビジョン`0d1c1a77fce322ba23b57a5e3261fae934f5afe3`、`archive.zip`のsha256は`6144b38827acf3989bd7cd000a424c52cfd46a49da07e586439e25cf52995547`（gt.txt付きのTrainIJCNN2013と、正解データなしのTestIJCNN2013Download） | パッケージ内のReadMe.txt："You are free to use this data package for any purpose you like."。ウェブページにはライセンス文の記載はない | 学習フレームでのゼロショット検出ベースライン；852枚のクロップでの、撮影をまたいだ分類の確認。テストの正解データには公式のFullIJCNN2013.zipが必要（ローカルでダウンロードする）。 |
| Panoramax/classified_de_road_signs（ドイツの標識クロップ、134クラス、20,000枚以上） | <https://huggingface.co/datasets/Panoramax/classified_de_road_signs>、リビジョン`b4856947ed7cb6312587258acc90e8cf88a4aa13`；ダウンロードしたのは`val.zip`（29.7 MB、6,944クロップ）のみ | "cc-by-sa-4.0"（データセットカード）；Panoramaxのストリートレベル画像から切り出したクロップ | Panoramaxの分類器の評価（`docs/replication.md`）。継承（SA）：派生ラベルはCC BY-SAのままにする必要がある。 |
| Swedish Traffic Signs（STS、Larsson と Felsberg、SCIA 2011） | Zenodo <https://zenodo.org/records/19666339>（`traffic-signs_dataset.zip`、15.8 GB、著者が 2026-04-20 にアップロード）。ここで使用したアノテーションファイル：<https://www.cvl.isy.liu.se/research/trafficSigns/swedishSignsSummer/Set1/annotations.txt> と `.../Set2/annotations.txt`（sha256 は `make sts` 後の `data/sts/summary.json`） | Zenodo："cc-by-4.0"（2026-11-01 に Zenodo API で確認）。CVL の概要ページには、承認なしに「改変または商用利用してはならない」という文がまだある。OpenTSR が依拠するのは Zenodo のライセンスであり、著者には連絡する | 最初の L3（標識通過イベント）セット。アノテーションから派生したイベントファイルは `configs/sts/` にある。Set1 のアノテーション40行の抜粋をテストのフィクスチャにしている。Larsson と Felsberg、SCIA 2011 および IET CV 2011 を引用すること。 |
| CLIP_benchmarkの`clip-benchmark/wds_gtsrb`テストシャード | <https://huggingface.co/datasets/clip-benchmark/wds_gtsrb>（ライセンスのメタデータなし；公式GTSRBテストセットのwebdatasetエクスポート、ロスレスWebP） | カードに記載なし；元データはGTSRB（"free to use"） | ミラーの画像がCLIP_benchmarkプロトコルの画像とピクセル単位で同一であることの確認にのみ使用；再配布しない。 |
| Synset Signset Germany、GTSRBサブセット（Fraunhofer IOSB） | <https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset>；ダウンロードしたのは`cycles/validation.parquet`（マスクとセグメンテーション付きの4,300枚のレンダリング画像）のみ | "CC BY 4.0"、カードが求める帰属表示（"This work is based on the Synset Signset Germany Dataset ... (c) 2024 Fraunhofer IOSB"）を付ける | データセットカタログ（`docs/dataset_catalog.md`）におけるGTSRBクラスの合成版；学習データの候補。 |
| `finedet/tt100k`ミラー経由のTT100K（Tsinghua-Tencent 100K） | <https://huggingface.co/datasets/finedet/tt100k>（`categories.csv`付きのparquetエクスポート、232のカテゴリコード）；3シャードをダウンロード（test-00003、train-00005、train-00006；1,050フレーム） | 公式ページ："Creative Commons Attribution-NonCommercial (CC-BY-NC)"；ミラーのタグは`cc-by-nc-4.0` | カタログのサンプルのみ（研究利用）。公開候補ではなく、再配布しない。 |
| `NgTMDuc/cl_dataset`ミラーからサンプリングしたMTSD（Mapillary Traffic Sign Dataset）の公式アノテーションJSONと画像 | <https://huggingface.co/datasets/NgTMDuc/cl_dataset>（`MTSD/annotations/mtsd_v2_fully_annotated`、`MTSD/mtsd_fully_annotated_train_images`）；`scripts/sample_mtsd_official.py`で、シードを固定してサンプリングした数百個のキーを取得 | アノテーションに同梱の`LICENSE.txt`：CC BY-NC-SA（CC BY-NC 4.0のリーガルコードへのリンク） | カタログのサンプルのみ（研究利用）。`ThankGod/*`のparquetミラーは使用をやめた：その整数カテゴリが自身の`id2labels.txt`と一致しない（目視と、ミラー間でのボックスの照合により確認）。 |
| CURE-TSR（Zenodo版） | <https://zenodo.org/records/3903066>（`CURE-TSR.zip`、4.9 GB） | Zenodoのレコード：CC BY 4.0（GitHubのREADMEでは、代わりに利用条件のフォームの提出を求めている） | カタログのロバスト性の節（14種の標識×12種のチャレンジ×5レベル）；ロバスト性テストセットの候補。 |
| 国土交通省 道路局「道路標識一覧」（MLITの日本の道路標識一覧、A3判1ページ） | https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf | ピクトグラムは道路標識令の別表の図であり（著作権法 第 13 条により著作権の対象外）、PDF自体はMLITの公開文書である | 日本の公式ピクトグラムを`scripts/cut_mlit_sign_list.py`で221枚のタイルに切り出し、カタログ用および日本トラックのクラス一覧として用いる。名称は日本語版Wikipediaの記事「日本の道路標識」と照合した（`scripts/parse_ja_wikipedia_signs.py`、CC BY-SAのテキスト、番号と名称の対応表にのみ使用）。 |

調査（`docs/datasets.md`）で取り上げたその他の標識データセットは、ライセンスとともにそちらに列挙している。初めてダウンロードした時点で本表に追加する。

## 道路標識のモデルと重み

| アセット | 出典 | ライセンス（原文どおり） | OpenTSRでの利用 |
|---|---|---|---|
| poojahira/gtsrb-pytorchの公開チェックポイント`model/model_40.pth`（sha256 `157996e0f6baa5b5f0803dae03a74c61c31723c8768f7cdd0e6535df14bbae38`） | https://github.com/poojahira/gtsrb-pytorch | リポジトリにライセンスファイルなし | GTSRBの公式テスト分割で採点（`docs/replication.md`）。再配布しない；アーキテクチャは再実装している（後述）。 |
| OpenAI CLIP ViT-B/32、ViT-B/16、ViT-L/14（ゼロショットの重み、open_clipで`-quickgelu`の設定を用いて`pretrained="openai"`により読み込む） | <https://github.com/openai/CLIP>（重みはtimmによってHugging Face Hubに再ホストされている） | "MIT"（CLIPリポジトリ、公開された重みも対象） | GTSRBでのゼロショットベースライン。寛容なライセンス；バックボーンや教師モデルとしても使える可能性がある。 |
| FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb（リビジョン`50b332c27907`） | https://huggingface.co/FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb | "cc-by-4.0"（モデルカード）；ベースモデルのfacebook/convnextv2-atto-1k-224はApache-2.0 | GTSRBの公式テスト分割とGTSDBのクロップで採点。tanganke/gtsrbの分割で学習されている；CC BY 4.0であれば帰属表示付きでの再配布が可能である。 |
| bazyl/gtsrb-model（bazyl/GTSRBでファインチューニングしたViT-B/16 in21k、リビジョン`95f6c95c765e`） | https://huggingface.co/bazyl/gtsrb-model | "apache-2.0"（モデルカード） | 公式テスト分割での採点を待機中（`make replicate-gtsrb-released`）。 |
| google/owlv2-base-patch16-ensemble（リビジョン`cfd3195ba4ea9592eec887ded089f4c08eff231d`） | https://huggingface.co/google/owlv2-base-patch16-ensemble | "apache-2.0"（モデルカード） | GTSDBでのゼロショット・クラス非依存の標識検出ベースライン。寛容なライセンス；自動ラベル付けに使える可能性がある。 |
| HuggingFaceTB/SmolVLM-500M-Instruct（リビジョン`a7da5b986cb5`）、Qwen/Qwen2-VL-2B-Instruct（リビジョン`895c3a49bc3f`） | <https://huggingface.co/HuggingFaceTB/SmolVLM-500M-Instruct>、<https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct> | "apache-2.0"（両モデルカード） | GTSRBのサブセットに対するVLMの多肢選択プローブ。 |
| Panoramax/classify_de_road_signs（`classify_de_road_signs.pt`、26,282,865バイト、リビジョン`5360aa6f4ef6c7b1998044b18d00b4d0b1a5a790`） | https://huggingface.co/Panoramax/classify_de_road_signs | "etalab-2.0"（モデルカード）；Ultralyticsのチェックポイント形式 | 検証用クロップでのカード記載値0.986の再現；`scripts/eval_panoramax_de_classifier.py`の制限付きunpicklerを通してのみ読み込む。公開候補ではない（AGPL-3.0のツールチェーン）。 |
| Ultralytics YOLO-World v2（`yolov8s-worldv2.pt`） | <https://github.com/ultralytics/ultralytics>（重みはultralytics/assetsのリリースから取得） | "AGPL-3.0"（下記のYOLO11と同じ） | GTSDBでのみ用いるオープン語彙検出ベースライン；公開候補ではない。 |

## 共有データセット

| アセット | 出典 | ライセンス（原文どおり） | OpenTSRでの利用 |
|---|---|---|---|
| COCO 2017（COCOで事前学習したYOLOの重みを介して） | https://cocodataset.org | アノテーションは"CC BY 4.0"；画像の権利はFlickrのユーザーに帰属 | サードパーティの重みを介した事前学習のみ。信号機トラックでは、その"traffic light"クラスも直接用いる（`traffic_lights/docs/replication.md`）。 |

## 共有モデルとツール

| アセット | 出典 | ライセンス（原文どおり） | OpenTSRでの利用 |
|---|---|---|---|
| Ultralytics YOLO11（コードと、COCOで事前学習した重み`yolo11n.pt`、`yolo11s.pt`） | <https://github.com/ultralytics/ultralytics>、重みはultralytics/assetsのGitHubリリースから取得 | "AGPL-3.0"（商用向けにUltralytics Enterprise Licenseあり） | ベースラインと再現のみ（両トラック）。Ultralyticsでファインチューニングした重みは、`docs/decisions.md`のライセンスに関する決定が見直されるまで、OpenTSRの公開候補としない。 |

## コピーせずに再実装したコード

| アセット | 出典 | ライセンス | 取り入れたもの |
|---|---|---|---|
| PASCAL VOC devkitのAP計算（SCRDet++がS2TLDで用いており、大半の標識ベンチマークでも用いられている） | https://host.robots.ox.ac.uk/pascal/VOC/ | 該当なし（アルゴリズム） | 貪欲マッチングと全点 / 11点補間。`src/opentsr/metrics.py`に再実装。 |
| poojahira/gtsrb-pytorch（GTSRB向けのSTN + 3層畳み込みCNN、プライベートリーダーボードで99.809%） | https://github.com/poojahira/gtsrb-pytorch | リポジトリにライセンスファイルなし | アーキテクチャ（層のサイズ、正規化定数、データ拡張のリスト）を、READMEと`model.py`をもとに`src/opentsr/signs/models/gtsrb_cnn.py`に再実装；ソースコードはコピーしていない。 |
| Udacity Self-Driving Car Nanodegreeの"Traffic Sign Classifier"のLeNet-5レシピ | 数百の公開された受講生のリポジトリ（`docs/oss_implementations.md`を参照） | さまざま（大半はMIT） | LeNet-5の構成とグレースケール正規化を`src/opentsr/signs/models/gtsrb_cnn.py`に再実装。 |
| OpenAI CLIPのGTSRB用プロンプトとクラス名 | https://github.com/openai/CLIP/blob/main/data/prompts.md | MIT | 43クラスの説明文と3個のテンプレートを`src/opentsr/signs/datasets/gtsrb.py`と`scripts/eval_clip_zeroshot_gtsrb.py`にそのままコピーしている（MITはこれを許可している；帰属表示はdocstringに残している）。 |
| Ultralyticsの分類評価用変換（リサイズ、センタークロップ、ToTensor） | <https://github.com/ultralytics/ultralytics>（`ultralytics/data/augment.py`） | AGPL-3.0 | 3段階の前処理を、`scripts/eval_panoramax_de_classifier.py`でtorchvisionを用いて作り直している；ソースコードはコピーしていない。 |

## Pythonの依存関係

`requirements.txt`（道路標識トラック、ルート）と`traffic_lights/requirements.txt`（信号機トラック）でバージョンを固定している。主なライセンス：PyTorch（BSD-3-Clause）、transformers（Apache-2.0）、open_clip（MIT形式の表記）、onnxruntime（MIT）、OpenCV（Apache-2.0）、NumPy（BSD-3-Clause）、Ultralytics（AGPL-3.0、上記を参照）。
