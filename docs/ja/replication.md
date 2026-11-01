# オープンな道路標識認識実装の再現

> 本文書は英語版 [docs/replication.md](../replication.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

我々が自ら再実行した内容、使用したデータとプロトコル、および得られた結果をまとめる。本ファイルのすべての実行は、2026-10-31にclaude.aiのクラウドセッションで行った（AVX-512とAMXを備えたIntel XeonのCPU 2コア、RAM 7 GB、GPUなし、Python 3.13、PyTorch 2.15.0+cpu、transformers 5.22.1、open_clip 3.3.0、YOLO-World専用にインストールしたultralytics 8.4.194。バージョン固定は`requirements.txt`）。すべての数値は、`scripts/`以下のスクリプト、学習を伴う場合は`configs/gtsrb/`以下の設定ファイル、および`runs/`以下のJSONファイル（gitignore対象。第10節に挙げる`make`ターゲットで再生成できる）に対応する。これらの実装を選んだ理由は`docs/oss_implementations.md`、基盤モデルの参照値は`docs/vlm_and_foundation.md`、データについては`docs/datasets.md`を参照のこと。

状況一覧：

| # | 内容 | データ | 状況 | 主な結果 |
|---|---|---|---|---|
| 2 | 公開されたpoojahira/gtsrb-pytorchのチェックポイント`model_40.pth` | GTSRB公式テスト（12,630） | 完了 | 99.71 %（公称値99.809 %） |
| 3 | スクラッチから学習したCNN：Udacity LeNet-5（2種）、計算予算を縮小したpoojahira STN-CNNアーキテクチャ | GTSRB公式分割 | 完了 | 91.84 / 89.14 / 99.13 % |
| 4 | 公式テスト分割で評価したHubのチェックポイント：ConvNeXt V2 atto（Fraunhofer IOSB）、ViT-B/16（bazyl） | GTSRB公式テスト | 完了 | 99.81 %（ConvNeXt V2 atto、カード記載値99.8）、97.00 %（ViT-B/16、カード記載値は独自の検証分割での99.93。379件の誤りのうち200件は左右の鏡像の混同） |
| 5 | OpenAIのプロンプトによるCLIPとSigLIPのゼロショット | GTSRB公式テスト | 完了（B/32）、実行待ち（B/16、SigLIP、L/14） | 32.56 %（ViT-B/32、論文値32.2、open_clip 32.6）。open_clipのGELU不一致の落とし穴にはまると26.57 % |
| 6 | 小規模VLMによる43択の多肢選択 | GTSRBテストのサブセット（層化） | 完了 | 2.33 %（SmolVLM-500M、215枚）、23.26 %（Qwen2-VL-2B、215枚） |
| 7 | オープン語彙検出器、クラス非依存 | GTSDB学習パッケージ（公式gt.txt） | 完了 | OWLv2のAP50は150フレームで77.1、YOLO-World-sのAP50は54.2 |
| 8 | GTSRBで学習した分類器をGTSDBのクロップで評価（別撮影データでの確認） | GTSDBのクロップ852件 | 完了 | STN-CNN 99.53 %、ConvNeXt V2 atto 99.53 %、LeNet 98.36 % |
| 9 | Panoramaxのドイツ標識分類器（134クラス）を自身の検証用クロップで評価 | Panoramax/classified_de_road_signs val（6,944） | 完了 | 98.68 %（公称値0.986） |
| 10 | GTSDB / ZOD / TT100Kでの検出器学習、CLIPの線形プローブ、合成から実データへの転移、VLMのファインチューニング | GPU | 準備済み、未実施 | 第10節 |

読み方（数字は節番号）：2と9は公開された重みから公表値を再現したものである。4と5はHubと論文の主張を一つの固定したプロトコルで確認する。3は古典的なレシピを本環境で学習した場合に公式分割で得られる値を示す。6、7、8は、この形ではどの論文も報告していない測定である。

---

## 1. GTSRB実験に共通のデータとプロトコル

- データ：Hugging Faceの`bazyl/GTSRB`、リビジョン`a8093a9c7757b59d64702f892002542e8f3a1fb0`（GTSRBのKaggle版をparquet形式でエクスポートしたもので、公式のPPMファイルをPNGに変換している）。`scripts/prepare_gtsrb.py`は、他の処理を実行する前にリビジョンを固定し、両parquetファイルのsha256を確認し、クラスID 0から42の学習画像39,209枚とテスト画像12,630枚を検証する（`data/gtsrb_hf/VERIFIED.json`、`problems: []`）。公式ホストのsid.erda.dkはクラウドセッションから到達できない。ミラーの`gpl-3.0`タグはアップロード者が付けたものであり、ベンチマーク自体のものではない（"The data is free to use"、https://benchmark.ini.rub.de/gtsrb_dataset.html ）。
- `tanganke/gtsrb`を使わない理由：その学習分割は26,640枚で、IJCNN 2011 Phase Iのサブセットである（`docs/datasets.md`の第1.1節）。そのため、これで学習したモデルは公式の39,209枚のプロトコルと比較できない。テスト分割は同じ12,630枚である。
- 評価指標：12,630枚のテスト画像全体でのtop-1精度（公式リーダーボードが用いた唯一の指標）。加えて、各JSONにはクラスごとの再現率と完全な混同行列を記録する。テストセットでのモデル選択は行わない。学習したモデルについては、最終エポック後の重みを一度だけ評価する。
- 画像はGTSRBの画像全体（標識とその周囲約10%の余白）として入力する。比較対象のレシピはすべてこの方式である。アブレーション用に、スクリプトには`--use-roi`がある。
- クラス名：テキストベースのモデルにはOpenAI CLIPリポジトリの43個の説明文（`data/prompts.md`）を、表とVLMのプロンプトには通常の短い名前（Speed limit (20km/h), ..., End of no passing by vehicles over 3.5 metric tons）を用いる。どちらも`src/opentsr/signs/datasets/gtsrb.py`にある。

## 2. 公開されたpoojahira/gtsrb-pytorchチェックポイントの公式テスト分割での評価（完了）

- 出典：https://github.com/poojahira/gtsrb-pytorch のファイル`model/model_40.pth`（3,430,539バイト、sha256 `157996e0f6baa5b5f0803dae03a74c61c31723c8768f7cdd0e6535df14bbae38`）。READMEの主張では、NYU CV Fall 2018 Kaggle in-class competition（テスト画像はラベルを伏せた公式GTSRBテストセット）のprivateリーダーボードで99.809%、publicリーダーボードで99.746%であり、これはランダムなテスト時拡張で得られた値で、READMEによれば実行ごとにスコアが約0.1ポイント変動する。学習：40エポック、Adam 1e-4、バッチ64、10倍にオフライン拡張した学習セット（水平・垂直反転を含む392,090枚）を用い、学習画像と検証画像を合わせて学習した。
- 実施内容：READMEと`model.py`をもとにネットワークを`src/opentsr/signs/models/gtsrb_cnn.py`に再実装し（STNはconv7x7(8)、conv5x5(10)、FC32。CNNはconv5x5(100)、conv3x3(150)、conv3x3(250)、FC350、FC43。BatchNorm、Dropout2d、leaky ReLU）、デフォルトの`weights_only=True`でstate dictを読み込み、すべてのテスト画像を32x32にリサイズしてリポジトリの正規化定数を適用した。テスト時拡張は用いていない。このチェックポイントはPyTorch 1.3より前のものであり、PyTorch 1.3では`affine_grid` / `grid_sample`のデフォルトが`align_corners=True`から`False`に変わった。両方の設定で評価した。スクリプト：`scripts/eval_poojahira_weights_gtsrb.py`。
- 結果：`align_corners=True`で**99.71%**（12,630枚中36件の誤り）、`False`で99.71%（37件の誤り）。READMEの99.809%は、テスト時拡張なしで0.1ポイント以内まで再現できた。残る差は、README自身がTTAのランダム性によるとしている大きさである。これは我々が見つけた中で最良の公開PyTorch GTSRBチェックポイントであり、同じ画像で公式リーダーボードの99.71%（Arcos-Garcia、Torch7）を上回る。
- クラス別再現率が97%未満のクラス：Beware of ice/snow 96.7。

## 3. 公式分割でスクラッチから学習したCNN（完了）

設定は`configs/gtsrb/`以下、スクリプトは`scripts/train_gtsrb_classifier.py`である。データは32x32のuint8配列としてキャッシュし、設定で有効にした場合はバッチ上でサンプルごとのオンラインデータ拡張（ランダムアフィン：回転15度、平行移動10%、スケール0.9から1.1、せん断2度。明るさとコントラストのジッタ40%）を行う。Adamとコサイン減衰、シード0を用い、最終エポックの重みを一度だけ評価する。

| 設定 | ネットワーク（パラメータ数） | エポック数、バッチ、学習率 | データ拡張 | テストtop-1 | 誤り数 | 実時間（2コア） |
|---|---|---|---|---|---|---|
| `lenet5_gray_32_noaug` | LeNet-5、グレースケール32x32（64,511） | 30、128、1e-3（コサイン） | なし | **91.84 %** | 1030 | 3.4分 |
| `lenet5_gray_32` | 同上 | 同上 | アフィン + フォトメトリック | **89.14 %** | 1371 | 3.5分 |
| `stn_cnn_poojahira_32` | poojahira STN-CNN、RGB 32x32（855,487） | 20、64、1e-3（コサイン） | アフィン + フォトメトリック | **99.13 %** | 110 | 44分 |

- LeNet-5は、Udacity時代のリポジトリと同程度の値に落ち着く（91から97%。`docs/oss_implementations.md`のB節を参照）。データ拡張ありの版は、なしの版より低い。パラメータ数64k、グレースケール入力のネットワークは、40%の明るさジッタのもとではアンダーフィットする（最も悪いクラスは暗くて出現頻度の低い警戒標識で、Pedestriansの再現率は0.0%、Beware of ice/snowは32.0%）。古典的なレシピはこれをコントラスト正規化（SermanetのYUV + 局所コントラスト正規化、Udacityのヒストグラム平坦化）で解決していたが、この数値を素のレシピのベースラインとして保つため、ここではあえて追加していない。
- STN-CNNは、拡張前の学習セットで20エポック後に99.13%に達する。これは、10倍のオフライン拡張、反転、40エポック、TTAを用いた同じアーキテクチャの公開チェックポイント（第2節）より0.59ポイント低い。エポックごとの推移：エポック1で93.70、エポック2で96.89、エポック5で98.19、エポック10で98.54、エポック15で99.02、エポック20で99.13。この実行は2コアで44分かかったため、10倍のセットでの忠実な40エポックの実行はGPUで行うタスクである（第10節）。

## 4. Hubのチェックポイントの公式テスト分割での評価（完了）

スクリプト`scripts/eval_hf_classifier_gtsrb.py`：各チェックポイント自身の画像プロセッサを用い、出力インデックスiをGTSRBのクラスiとみなす（両チェックポイントの`id2label`は公式の順序に従っている。JSONに出力済み）。

| チェックポイント | ベースモデル、記載されている学習データ | カード記載値 | 公式テストtop-1 | 誤り数 | CPU時間 |
|---|---|---|---|---|---|
| `FraunhoferIOSB/convnextv2-atto-1k-224-finetuned-gtsrb`（CC BY 4.0） | ConvNeXt V2 atto、ImageNet-1k、最初の2ステージを凍結して`tanganke/gtsrb`（学習26,640枚）でファインチューニング | "99.8% on the GTSRB validation set" | **99.81 %** | 24 | 4分 |
| `bazyl/gtsrb-model`（Apache-2.0） | ViT-B/16 ImageNet-21k、`bazyl/GTSRB`（公式学習分割）で10エポックのファインチューニング | 0.9993 "on the evaluation set" | **97.00 %** | 379 | 45分 |

- Fraunhofer IOSBのConvNeXt V2 atto（3.7 Mパラメータ、ImageNet-1kで事前学習、最初の2ステージを凍結して26,640枚のミラー分割でファインチューニング）は、公式テスト画像でカード記載の99.8%を再現し、公開されたpoojahiraチェックポイント（99.71%）と2018年に凍結されたリーダーボード（99.71%）を上回る。誤り数は本セッションで学習したどのモデルよりも少ない。最も弱いクラスはBeware of ice/snow（再現率97.3%）である。GTSRBでは、事前学習と224 px入力の組み合わせは、古典的な32 pxのレシピのいずれよりも効果が大きい。bazylのViT-B/16のファインチューニング版（86 Mパラメータ、ImageNet-21kで事前学習）は公式テストで97.00%であり、カード記載値より2.9ポイント低い。カードの99.93%は学習セットから取り分けた一部で測定されたものである（ログにはエポックあたりバッチ8で4,166ステップとあり、これは39,209枚の学習画像のうち約33,300枚に相当する。したがって評価セットは残りの15%であり、学習画像とトラックおよび物理的な標識を共有している）。混同行列は鏡像のペアが大半を占める：Turn right aheadをTurn left aheadと読んだもの（70枚）、Keep leftをKeep rightと読んだもの（48）、Dangerous curve to the leftを右カーブと読んだもの（29）、Go straight or rightとGo straight or leftの取り違え（27と26）。379件の誤りのうちこの200件は、学習時のデータ拡張におけるランダムな水平反転と整合する（カードは自動生成されたもので、変換を列挙していない。Hugging Faceの`run_image_classification.py`の例はデフォルトで反転を行う）。水平反転は左向きの標識を右向きの標識に変えてしまう。これらの誤りがなければ、このチェックポイントは約98.6%になるはずである。残りの誤りは、よくある小さな速度制限の数字の誤読である（120を80や50と、60を50と読む）。したがって、CNNチェックポイントとの差はアーキテクチャの影響ではなく、レシピの影響（反転によるデータ拡張、ROIクロップなし、10エポック）である。`vlm_and_foundation.md`にあるファインチューニングしたCLIP ViTの数値（98.5から99.3%）は、標識を考慮したレシピでViTが到達できる水準を示している。

## 5. OpenAIのプロンプトによるCLIPとSigLIPのゼロショット（完了）

スクリプトは`scripts/eval_clip_zeroshot_gtsrb.py`で、CLIP論文のプロトコルに従う：`openai/CLIP/data/prompts.md`にある43個のクラス説明文と3個のテンプレートを用い、テキスト埋め込みをテンプレート間で平均し、コサイン類似度のargmaxをとり、12,630枚のテスト画像でtop-1を求める。各画像は、モデル自身のopen_clip変換でリサイズと中央クロップを行う。参照値：CLIP論文の表11（https://arxiv.org/pdf/2103.00020 ）とopen_clipの結果CSV（https://raw.githubusercontent.com/mlfoundations/open_clip/main/docs/openclip_results.csv ）。どちらも`docs/vlm_and_foundation.md`のA節にまとめてある。

| open_clipモデル | pretrained | 測定したtop-1 | CLIP論文 | open_clip CSV | CPU時間 |
|---|---|---|---|---|---|
| ViT-B-32 | openai | **32.56 %** | 32.2 | 32.6 | 10分 |
| ViT-B-16 | openai | 本セッションでは未実施（2コアで約50分。`make replicate-gtsrb-clip`） | 43.3 | 43.4 | |
| ViT-B-16-SigLIP | webli | 本セッションでは未実施（約50分。同じターゲット） | 該当なし | 51.6 | |
| ViT-L-14 | openai | 本セッションでは未実施（2コアで約1.5時間。`make replicate-gtsrb-clip`） | 50.3 | 50.6 | |

- ViT-B/32は、アーキテクチャを重みに合わせれば両方の参照値を再現する。本セッションの最初の実行では26.57%だった。これは、open_clip 3.3.0が`pretrained="openai"`を指定されたときに`ViT-B-32`を標準のGELUでインスタンス化し、`QuickGELU mismatch`の警告を出すだけだからである。OpenAIのチェックポイントはQuickGELUで学習されているため、正しいモデル名は`ViT-B-32-quickgelu`である（現在のスクリプトは接尾辞を自動で付加し、`--keep-gelu-mismatch`でこの落とし穴を再現できる。JSONは`clip_zeroshot_ViT-B-32_openai_gelu_mismatch.json`として残してある）。前処理は原因ではない。1,500枚のサブセットでは、open_clip変換、224x224への押し潰し、公式ROIへのクロップ、バイリニアリサイズのいずれも、不一致のモデルでは26.1から27.0%の間に収まる。また、CLIP_benchmarkの`wds_gtsrb`テストシャードはミラーとピクセル単位で同一（ロスレスWebP）であるため、6ポイントの差はすべて活性化関数によるものである。ゼロショットCLIPは形状と色で区別できるクラスをよく認識する（No entry 90.8%、Roundabout 90.0、Stop 88.9）が、名前が数字やピクトグラムの細部だけで異なるクラスでは失敗する（カーブ、追い越し禁止、制限解除のクラスはすべて0%）。
- 速度制限の混同（文献が記述しているものの、定量化したことのない失敗モード）：ViT-B/32では、数値を持つ8個の速度制限クラス（20から120 km/h、テスト画像4,170枚）は、68.3%の割合で*何らかの*速度制限として認識されるが、*正しい*値として認識されるのは44.4%にとどまる。クラスごとの再現率は(20km/h) 65.0、(30km/h) 10.0、(50km/h) 76.0、(60km/h) 4.0、(70km/h) 61.7、(80km/h) 27.3、(100km/h) 64.7、(120km/h) 62.4である。`vlm_and_foundation.md`のA.5節が予測するとおり、分類器のテキスト側は数字を区別できない。

## 6. 小規模VLMによる43択の多肢選択（完了）

スクリプト`scripts/eval_vlm_gtsrb_mc.py`：公式テスト分割の層化サブセット（クラスごとに5枚、シード0、215枚。両モデルで同じサブセット）を用いる。各クロップを224 pxに拡大し、画像、43個の短いクラス名の番号付きリスト、および"Answer with the number of the correct option only"を含む1回のユーザーターンを与え、貪欲デコーディングを行い、応答中の最初の整数を予測とする。これは、LMMを用いた道路標識の論文における閉じた語彙のゼロショット設定である（`vlm_and_foundation.md`のB.1節は、同様のプロトコルのもとGTSRBのサブセットでLLaVA-1.5 0.32、LLaVA-NeXT 0.39、CLIP ViT-B/32 0.24、GPT-4o 0.89を報告している）。

| モデル（ライセンス） | パラメータ数 | サブセット | Top-1 | 解析不能な応答 | 1枚あたりの秒数（2コア） |
|---|---|---|---|---|---|
| `HuggingFaceTB/SmolVLM-500M-Instruct`（Apache-2.0）、512 pxの単一タイル | 0.5 B | 215 | **2.33 %** | 0 | 3.7 |
| `Qwen/Qwen2-VL-2B-Instruct`（Apache-2.0）、bf16 | 2.2 B | 215 | **23.26 %** | 0 | 5.3 |

- SmolVLM-500Mはほぼすべての画像に"10"か"9"と答え（215件中212件の応答）、チャンスレベル（43分の1 = 2.3%）に落ち着く。0.5 Bのモデルは43択のリストをまったく扱えないため、その数値は標識認識について何も示さない。Qwen2-VL-2Bは画像を実際に読み取っている：8個の速度制限クラスのうち7個で60から100%の正解率であり（速度クラスの平均は70%、30 km/hとStopでは100%）、これはまさにCLIPにできないことである。しかし、標識が数字でない場合は常に選択肢1（Speed limit 30）に偏り、215件中147件の応答がそうなるため、ピクトグラムのクラスはほぼゼロとなり、全体のtop-1はCLIP ViT-B/32のゼロショットを下回る。CdMT論文のLLaVA-1.5（0.32）とGPT-4o（0.89）の数値は異なるサブセットとプロンプトによるものなので、比較できるのは順位だけである。Qwen2-VL-2BはAMXのbf16のおかげで、このCPU単独では1枚あたり2.7 s（競合時は5.3 s）で動作する。そのため、7 B級（Qwen2.5-VL-7B、InternVL 3）と選択肢順序のシャッフルは、第10節に挙げたGPUでの追加実験とする。ここで測定したオープンなVLMは、いずれも教師あり分類器から60ポイント以内に入らない。これは、VLMは分類器そのものではなくラベル付けと解釈の補助であるという調査の結論と一致する。VLMの数字の読み取り能力は残す価値のある部分である。

## 7. GTSDBにおけるクラス非依存のオープン語彙検出器（完了）

- データ：Hugging Faceのミラー`GlowBond/GTSDB`にあるGTSDBの学習パッケージ（リビジョン`0d1c1a77fce322ba23b57a5e3261fae934f5afe3`、`archive.zip`のsha256 `6144b38827acf3989bd7cd000a424c52cfd46a49da07e586439e25cf52995547`）：1360x800のフレーム600枚、公式の`gt.txt`には506フレームに852個のボックス（標識のないフレームは94枚）、標識の一辺は16から128 px（中央値38。32 px未満が297個、32から64が420個、それより大きいものが135個）。300枚のテストフレームは正解データなしで配布されており、それらには公式のFullIJCNN2013アーカイブが必要である（第10節）。カテゴリ別の件数：prohibitory 396、danger 156、mandatory 114、other 186。
- プロトコル（`scripts/eval_openvocab_gtsdb.py`）：プロンプトは"traffic sign"、"road sign"、"speed limit sign"、"round red and white prohibition sign"、"triangular warning sign"、"blue round mandatory sign"、"stop sign"、"yield sign"。すべての検出を一つのクラス"sign"にラベル付けし直す。スコア0.05以上のすべての検出に対し、全点補間（`src/opentsr/metrics.py`）によるIoU 0.5および0.3でのPASCAL VOC APを求める。動作閾値0.3で、再現率、適合率、カテゴリ別およびサイズ別の再現率、ネガティブフレーム（パッケージ中に94枚、ここで用いた150フレームのサブセット中に3枚）での誤検出を求める。ゼロショットモデルはGTSDBを一度も見ていないので、学習フレームで評価することは正当である。ただし、これらの数値はIJCNN 2013コンペティションのプロトコル（300枚のテストフレームでカテゴリごとにIoU 0.6で評価した）によるものではない。

| 検出器（ライセンス） | 入力 | フレーム数 | AP50 | AP30 | 閾値0.3での再現率 | 閾値0.3での適合率 | カテゴリ別再現率（proh / mand / dang / other） | サイズ別再現率（<32 / 32-64 / >=64 px） | ネガティブフレームでのFP | 1フレームあたりの秒数 |
|---|---|---|---|---|---|---|---|---|---|---|
| OWLv2 base patch16 ensemble（Apache-2.0） | 960x960（パディング） | 150（238ボックス） | **77.1** | 77.2 | 95.4 | 39.0 | 99 / 90 / 98 / 92 | 90 / 98 / 98 | 3 / 3 | 6.6 |
| YOLO-World v2 s（Ultralytics、AGPL-3.0） | 1280 | 150（238ボックス） | **54.2** | 54.5 | 64.7 | 57.5 | 87 / 88 / 45 / 33 | 70 / 53 / 82 | 2 / 3 | 1.6 |

- OWLv2はほぼすべての標識の位置を特定する（全検出に対するIoU 0.5での最大再現率は99.6%、動作閾値0.3では95.4%、32 px未満の標識でも90.5%）が、適合率は低い（閾値0.3で39.0%、238個の標識に対して618個のボックス。標識の裏面、ポール、店の看板にも反応する）。そのためAP50は77.1となる。IoU 0.3でのAPは0.5でのAPと同じなので、反応したときの位置精度は高い。用いた標識向けのプロンプトでは、mandatory（青色）の標識が最も弱いカテゴリである（再現率90.2%）。時間の都合上、どちらの実行も最初の150枚の学習フレーム（00000から00149：238ボックス、ネガティブフレーム3枚）を対象とした。`make replicate-gtsdb-openvocab`を実行すると600枚すべてを対象とする。1280 pxのYOLO-World v2 sは4倍高速であるが、これらのプロンプトではdangerとotherの標識の大半を見逃し（再現率44.7%と32.8%）、AP50は54.2にとどまる。また、UltralyticsはPPMのパスを受け付けないため、スクリプトはPIL画像を渡している。OpenTSRにとっての実務的な解釈：Apache-2.0のオープン語彙検出器は、標識分類器で絞り込むのであれば自動ラベル付けの候補ボックスを提案するには十分であるが、単独の検出器としては不十分である。

## 8. GTSRBで学習した分類器のGTSDBクロップでの評価（完了）

スクリプト`scripts/eval_gtsrb_models_on_gtsdb_crops.py`：GTSDBの学習フレームにある852個の正解ボックスを10%のマージン付きで切り出し（GTSRBの余白を模倣）、GTSRBしか見ていないモデルで43クラスに分類する。注意：GTSDBはGTSRBと同じグループが同じカメラ構成で収録したもの（Houben et al., IJCNN 2013）であるため、これは国をまたぐテストではなく、同一ドメインの別撮影データでのテストである。

| モデル | GTSRBテストtop-1 | GTSDBクロップtop-1（852） | カテゴリ別（proh / mand / dang / other） | サイズ別（<32 / 32-64 / >=64 px） |
|---|---|---|---|---|
| LeNet-5、データ拡張なし（第3節） | 91.84 | **98.36** | 98.5 / 96.5 / 97.4 / 100.0 | 97.5 / 98.6 / 99.3 |
| STN-CNN、縮小予算（第3節） | 99.13 | **99.53** | 99.5 / 100.0 / 98.7 / 100.0 | 99.6 / 99.3 / 100.0 |
| ConvNeXt V2 atto、Hubチェックポイント（第4節） | 99.81 | **99.53** | 99.5 / 100.0 / 98.7 / 100.0 | 99.6 / 99.3 / 100.0 |

- 縮小予算のSTN-CNNとConvNeXt V2 attoのチェックポイントは、まったく同じ4枚のクロップ（gt.txtの順でインデックス50、224、232、718）で失敗する：ピクトグラムが白飛びした露出過多の三角形の標識、ステッカーで汚損されたNo vehicles標識、19 pxのSpeed limit 50、そしてChildren crossing標識1枚である。4枚のうち3枚はモデルの弱点というより劣化した標識であり、`requirements_and_robustness.md`のB節にあるロバスト性ベンチマークが想定しているのはまさにこの種のケースである。LeNet-5の誤りは14件である。GTSDBのフレームは主に適正露出の標識を含む昼間の収録である一方、GTSRBのテストセットには暗いトラックやぼやけたトラックが残っているため、どのモデルでもGTSDBクロップでの精度はGTSRBテストセットより高い。したがって、これらのクロップはより難しい確認ではなく、より易しい同一ドメインでの確認である。

## 9. Panoramaxのドイツ標識分類器の検証用クロップでの評価（完了）

- 出典：`Panoramax/classify_de_road_signs`（etalab-2.0、`classify_de_road_signs.pt` 26,282,865バイト）と`Panoramax/classified_de_road_signs`（CC BY-SA 4.0、`val.zip` 29.7 MB、134個のクラスフォルダに1から247枚ずつ、計6,944枚のクロップ。Panoramaxのストリートレベル画像からプロジェクトのYOLO11l検出器で切り出したドイツの標識のクロップで、標識以外のものには"bad:*"と"other:*"のバケットがある）。モデルカード：精度0.986、"based on YOLOv26"（実際にはpickleはYOLO11形式のC3k2 / C2PSAブロックを参照している）。
- 読み込み：このチェックポイントはUltralyticsの完全なpickleであるため、`scripts/eval_panoramax_de_classifier.py`はまず`pickletools`で逆アセンブルし（見つかったglobalsはtorchのストレージとモジュール、torchvisionの変換、`ultralytics.nn.modules.{block,conv,head}`と`ultralytics.nn.tasks.ClassificationModel`のみで、それ以外はない）、それらのクラスだけを登録したうえで`torch.load(weights_only=True)`で読み込む。推論は素のPyTorchで行い、Ultralyticsの分類評価用変換（短辺を学習時のサイズにリサイズ、中央クロップ、ToTensor）を用いる。
- 結果：imgsz 224で、全6,944枚の検証用クロップに対して**top-1 98.68%**（top-5 99.77%）。実在の標識クラスの6,316枚のクロップ（bad/otherを除く）では98.77%。カードの0.986は再現できた（98.68 vs 98.6）。残る誤りは、何でも受け入れるバケット（other:multi、other）と検証用クロップの少ないクラス（oneway、n=23）に集中している。学習用クロップが公開されている（CC BY-SA 4.0）オープンな欧州の分類器として見つかったのはこれだけであり、そのためOpenTSRの欧州向け細粒度分類ヘッドにとって自然な参照となる。検証用クロップが20枚以上あるクラスのうち最も弱いもの：oneway 60.9（n=23）、other:multi 69.7（n=33）、no:other:dog 87.0（n=23）、hazard:other 92.0（n=25）、arrow_right 92.3（n=39）。

## 10. 準備済みのGPU実行と未解決事項

| 実行 | 理由 | コマンドまたは設定 | 想定コスト | 超えるべき参照値 |
|---|---|---|---|---|
| 第7節と第8節のための公式GTSDBテスト分割 | ミラーにはテストの正解データがない | ローカルでsid.erda.dkからFullIJCNN2013.zipをダウンロードし、`python scripts/eval_openvocab_gtsdb.py --detector owlv2 --full-gt <dir>/gt.txt --images <dir> --out runs/gtsdb/owlv2_test300.json`を実行 | CPUで1から2時間、GPUなら数分 | OWLv2については公表値なし |
| poojahiraの学習の忠実な再現 | 反転とTTAを伴う10倍拡張セットでの40エポック | `configs/gtsrb/stn_cnn_poojahira_32.yaml`を`epochs: 40`とし、オフラインでの10倍拡張を加える（未実装。CPUではオンラインデータ拡張で代用している） | GPU 1基で1から2時間 | 99.809（README）、99.71（第2節） |
| CLIPの線形プローブ | 論文の線形プローブの数値85.3（ViT-B/32）と92.5（ViT-L/14） | 39,209 + 12,630枚の画像についてopen_clipの特徴量を抽出し、scikit-learnのロジスティック回帰を行う | CPUで1時間（B/32） | 85.3 / 92.5（CLIP論文の表10） |
| 合成から実データへ | Synset Signset Germanyのみで学習して実際のGTSRBで98.3になるか（Fraunhofer IOSBの主張） | `FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset`（CC BY 4.0、21,500枚）で学習したConvNeXtまたはSTN-CNNを、`scripts/eval_hf_classifier_gtsrb.py`と同様のコードで公式テストにより評価 | GPU 1基で1から3時間 | 合成データのみで98.3、実データで99.9（arXiv 2512.05936） |
| GTSDB（43クラス、5-fold）とZODの標識での検出器のファインチューニング | OpenTSR最初の検出器ベースライン | `docs/decisions.md`でのフレームワーク決定後に、`configs/gtsdb/`、`configs/zod/`以下にYOLOX / RT-DETR / D-FINEの設定を追加予定 | GPU 1基で1実行あたり数時間 | GTSDBの3カテゴリmAP 95.77（aarcosgのFaster R-CNN）、ZOD論文の数値 |
| より大きなVLM | 第6節の多肢選択プロトコルでのQwen2.5-VL 7B / InternVL 3 / Gemma 3 | GPU上で`scripts/eval_vlm_gtsrb_mc.py --model <id> --dtype bfloat16` | 数分 | GPT-4o 0.89、LLaVA-NeXT 0.39（CdMT論文） |
| 日本向けベースライン | `infra-intelligence/road-sign-yolox-tiny`（Apache-2.0、ONNX）を自前の日本の映像で実行 | 同リポジトリの`examples/predict_onnx.py` | 数分 | 作者の非公開テストセットでAP50 0.902（定性的な参考のみ） |

作業中に見つかった未解決事項：

- `bazyl/GTSRB`と対になる`bazyl/gtsrb-model`には、出力ラベルが44個ある（インデックス43は未使用）。評価には影響しない。
- GTSDBのReadMeには学習パッケージのボックス数が852個と記載されている一方、IJCNN 2013の論文は全900枚で1,213個（または1,206個）と報告している。したがって、テストパッケージには約360個のボックスがある。

Makeターゲット：`make gtsrb`、`make gtsdb`、`make replicate-gtsrb-cnn`、`make replicate-gtsrb-released`、`make replicate-gtsrb-clip`、`make replicate-gtsrb-vlm`、`make replicate-gtsdb-openvocab`、`make replicate-panoramax`。
