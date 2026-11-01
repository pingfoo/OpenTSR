# 商用の道路標識認識システム：文書化されていることと推定されること

> 本文書は英語版 [docs/commercial_tsr.md](../commercial_tsr.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

調査メモ、取得日2026-11-01。メンテナからの二つの問いに答えるために書いた：量産TSRシステムが最も採用していそうなアーキテクチャは何か（コスト、精度、速度を総合して）、そしてそれらが検出・追跡・分類のパイプラインとして構築されていることを示す根拠として何があるか。A節の各主張には、読み取った出典を付けている。C節は推定であり、その旨を明記している。

## 要約

- 文書化されている量産TSRシステムの骨格は次のとおりである：単一フレームでの標識候補の検出、フレームをまたいだ候補の追跡、位置合わせしたクロップの分類、そしてトラック全体にわたって下す判定。これはMobileyeの2006年の特許、Siemens VDO / Continentalの2005年と2008年の論文、2012年のIEEE T-ITSのサーベイに明記されている。2023年のMobileye ISAの説明では、テキストのためのOCR、暗黙の制限のための道路種別ネットワーク、関連性判定のステップ（出口や側道の標識）が加わる。2025年のNDSSのブラックボックス研究では、認識した標識の長期間の"spatial memorization"（空間的記憶）が加わる。
- 現行製品の検出器はCNNである。Mobileyeのアクセラレータの説明は畳み込みネットワークを挙げている。TeslaのAI Dayで示されたスタックは、RegNetバックボーンとYOLO風のワンステージヘッドを用い、TransformerはマルチカメラのBEV融合に限って用いている。フロントカメラが動作する車載SoC（5から34 TOPS、数ワット）は成熟したCNNのツールチェーンを持つ一方、Transformerのサポートは部分的にとどまる。
- 検出と詳細分類が一つのマルチタスクネットワークなのか二つのモデルなのかは、どの現行製品についても公表されていない。2段構成の設計は、2006年から2012年の世代とオープンなスタック（Autowareの信号機、Panoramax）で確認されている。Teslaについては検出ヘッドしか説明されていない。

## A. 文書化された根拠

### A.1 特許

- Mobileye、US 8,064,643 B2 "Detecting and recognizing traffic signs"（優先日2006-12-06、出願2007-12-06、登録2011-11-22。発明者Stein, Shachar, Taieb, Wolfovitz）。特許に記載されたパイプライン：フレームはTSRと他の運転支援機能とで分け合う。検出は単一フレーム内で候補標識を見つける（"a candidate image may include a circle and detection may be performed based on a circle detection algorithm"。夜間は"detects light spots in the image, typically in a low resolution image"）。候補はフレームをまたいで対応付けられて"multi-frame structure"となり、これは"is verified over multiple frames"とされ、車両の動きに合わせて位置合わせされる。分類は"is preferably a multi-level classification that constructs a gating tree"であり、"the classifier is preferably a non-binary classifier"とされる。"a decision is made (step 370) based on the output of the classification (step 350) for each image in the multiple frame structure"とあり、この構造は標識が視野から外れるか遮蔽されると終了する。速度制限はOCRで読み取るのではなく分類される："classification proceeds (step 440) by the number of digits"とあり、2桁の候補は70、80、90と、3桁の候補は110、120、130と照合される（https://patents.google.com/patent/US8064643B2/en ）。

### A.2 サプライヤーの研究論文（最初の量産世代）

- Bahlmann, Zhu, Ramesh（Siemens Corporate Research）およびPellkofer, Koehler（Siemens VDO Automotive, Regensburg）、"A system for traffic sign detection, tracking, and recognition using color, shape, and motion information"、IEEE IV 2005。二つの構成要素からなる：検出と追跡（色チャネルをウェーブレットのパラメータとしたHaarウェーブレット上のAdaBoost、5スケール、単純な運動モデル、フレームごとの検出を時間方向に融合）と、分類（円形にクロップ、グレースケール化、ヒストグラム平坦化、固定サイズ化。25次元のLDA特徴量上でクラスごとに一つのガウス分布を用いるベイズ分類器。信頼度をフレーム間で融合）。ドイツの速度制限と追い越し禁止の23クラス。検出の偽陰性1.4%、偽陽性0.03%、パッチ分類の誤り6%、映像での認識の誤り15%、384x288の入力で2.8 GHzのXeon上10 fps。"many signs are smaller than 20 pixels in diameter"（https://lmbweb.informatik.uni-freiburg.de/people/bahlmann/data/ba_zh_ra_pe_ko_iv2005.pdf ）。
- Keller, Sprunk（Freiburg）、Bahlmann（Siemens CR）、Giebel, Baratoff（VDO Automotive AG, Lindau。所属住所はContinental）、"Real-time recognition of U.S. speed signs"、IEEE IV 2008。勾配投票による矩形検出器とAdaBoostによる棄却、位置合わせ（ヒストグラム投票による回転補正、矩形の再検出）、LDA特徴量とマハラノビス分類器、そしてトラック全体での多数決からなる。多数決は"proved to be very robust against accidental misclassifications with high confidence"とされる。673 sの映像中の80個の標識インスタンスで検出98.75%、分類97.5%、認識96.25%、誤警報は42 sに1回、2.16 GHzのノートPCで追跡込み11 fps（https://lmb.informatik.uni-freiburg.de/people/bahlmann/data/ke_sp_ba_ba_gi_iv08.pdf ）。どちらの論文も研究用プロトタイプを示したものであるが、著者らは最初の速度制限カメラを出荷したサプライヤー組織に属しており、構成はMobileyeの特許と一致する。

### A.3 サーベイ

- Møgelmose, Trivedi, Moeslund、"Vision-based traffic sign detection and analysis for intelligent driver assistance systems: perspectives and survey"、IEEE T-ITS 2012。TSRを検出（"concerned with locating signs in input images"。セグメンテーション、特徴抽出、検出）と分類からなるものと定義し、追跡を時間方向の層として位置付ける（"accept sign candidates as signs only if they have shown up on a number of consecutive frames"、またはカルマン予測器）。製品については、"speed limit detection systems have been included in top-of-the-line models from various manufacturers"とする一方で、"a more general sign detection solution and an integration into other vehicle systems have not yet materialized"としている。システムはドライバーの状態もドライバーの入力も用いておらず、検出した標識は自車が走行中の道路と結び付けられていないことが多い（https://vbn.aau.dk/ws/files/72574480/survey.pdf ）。

### A.4 現行の製品説明（機能レベル）

- Mobileye、カメラのみのIntelligent Speed Assist（ブログ、2023年。プレスリリース2023-07-18）：ソフトウェア更新によりEyeQ4とEyeQ6で動作する。標識の検出と分類には"signature-based classification"を用い、新しい種類の標識を更新で追加できる。形状では伝えられないテキスト（通学時間帯、重量区分、市町村入口の地名）にはOCRを用いる。標識がない、劣化している、または遮蔽されている場所では、道路種別を分類する深層ネットワークで制限速度を推定する。暗黙の制限（工事区間、スクールゾーン、高速道路の入口と出口）。一時的な標識と可変式のデジタル標識。車線との関連性（"offramp speed limits that shouldn't affect vehicles staying on the highway"）。学習データは"tens of millions of video clips"。EU加盟5か国の6機関によりGSRへの適合認証を受けている。精度の数値は公表されていない（https://www.mobileye.com/blog/intelligent-speed-assist-isa-computer-vision-adas-solution/ 、 https://ir.mobileye.com/news-releases/news-release-details/mobileye-launches-first-camera-only-intelligent-speed-assist ）。
- MobileyeのEyeQのページ：ADASフロントカメラ向けのEyeQ6 Liteは5 TOPS（int8、7 nm）、EyeQ6 Highは34 TOPS。XNNエンジンは"the main source of horse power for convolutional neural networks"であり、VMPとPMAは"short integral types common in computer vision and deep learning"を対象とする。Transformerへの言及はない（https://www.mobileye.com/technology/eyeq-chip/ ）。
- Boschの"Road sign information"（商用車向け）："round, triangular and rectangular"の標識を検出・分類し、標識上の数字を読み取る。マルチパーパスカメラに依存する（https://bosch-mobility.com/en/solutions/assistance-systems/road-sign-information-cv ）。
- Tesla AI Day 2021（二次的な要約）：1280x960、12ビットHDR、36 Hzのカメラ8台。RegNetバックボーン、BiFPNによるマルチスケール融合、"one-stage, YOLO-like detectors"と説明されるタスクヘッドを持つHydraNet。Transformerは、画像特徴を鳥瞰図の空間に写像してカメラ間を融合するためだけに用いられる（"one or more multi-head self-attention blocks"）。道路標識はそのベクトル空間の内容として現れる。独立した分類器やレイテンシの数値については何も述べられていない（https://kimbochen.github.io/blog/2021/11/09/tesla-ai-day-vision.html ）。

### A.5 2023年から2024年の量産車のブラックボックス研究

- Wang et al.、"Revisiting physical-world adversarial attack on traffic sign recognition: a commercial systems perspective"、NDSS 2025（UC Irvine）。米国の上位15ブランドの2023年式または2024年式の車両4台を匿名化して評価した。フロントウィンドシールドのカメラデータは"processed by DNN-based object detectors"とされる。認識した標識は"persistently memorized until the sign's reaction task is finished"とされる（時間ではなく位置によって消去される。4台中3台は60 sのテスト時間枠全体にわたって検出を保持し、通常の追跡では3 s未満であるのと対照的である）。速度制限の認識結果は、標識をおよそ半分通過した時点で現れた。テストした標識については、地図やGPSとの融合の根拠は見られなかった。公表済みの物理的攻撃のブラックボックス転移の成功率は30通りの組み合わせで6.67%であり（論文での主張は51.6%）、著者らはこれを、記憶によって隠蔽攻撃（hiding attack）が難しくなり、出現攻撃（appearing attack）が容易になるためとしている（https://arxiv.org/html/2409.09860v1 ）。

### A.6 内部を読めるオープンなスタック

- Autoware Universe（TIER IV、標識ではなく信号機）：HDマップから関心領域への投影、416x416のYOLOX-sによる精密検出器、224x224のMobileNetV2による状態分類器（矢印を含む11クラス）、加えて960x960で画像全体に適用するYOLOX-s。2026年以降、すべてのモデルがHugging Faceで公開されている（`traffic_lights/docs/oss_implementations.md`を参照）。
- Panoramax：YOLO11lの標識検出器と国別の分類器（DEは134クラス）が、学習用クロップとともに別々のモデルとして公開されている（`oss_implementations.md`を参照）。
- openpilot：カメラによるTSRはない（"openpilot does not detect speed limits at this time"）。フォークのFrogPilotは、地図データと融合するMNNの速度制限読み取り器を搭載している（`oss_implementations.md`のE3を参照）。

### A.7 組込みハードウェアとツールチェーン

- TI TDA4（TIDL）：SDK 9.1は"supports basic transformer (classification) ViT, DeiT (matmul, layernorm, softmax) and partial support of SwinT"とされる。DETR系の検出器は挙げられていない（https://e2e.ti.com/f/791/t/1329460 ）。
- Renesasの第4世代R-Car：V4Mは最大17 TOPS、"9 TOPS/W"、典型的な8 MPのスマートカメラで約5 W、2026年第1四半期から量産。ページは"dedicated deep learning and computer vision IP"を挙げているが、ネットワークの種類は示していない（https://www.renesas.com/en/about/newsroom/renesas-leads-adas-innovation-power-efficient-4th-generation-r-car-automotive-socs ）。
- Hailo-8：CNN指向のデータフロー型アクセラレータ。アテンションベースの検出器に関するベンダーのフォーラムスレッドは存在するが、全文を読むことはできなかった（https://community.hailo.ai/t/hailo-8-for-attention-based-object-detection-models/16105 ）。
- AMD Vitis AI 2.5のTT100K YOLOX-m：APはfloatで62.3、INT8（QAT）で62.1。つまりINT8化によるCNN検出器の損失は0.2ポイントである（https://raw.githubusercontent.com/Xilinx/Vitis-AI/v2.5/model_zoo/model-list/pt_yolox_TT100K_640_640_73G_2.5/model.yaml ）。
- Ultralyticsの比較（COCO 640、T4 TensorRT 10）：YOLO11nは39.5 mAP、1.5 ms、2.6 Mパラメータ、6.5 GFLOPs。YOLO11sは47.0、2.5 ms。YOLO11lは53.4、6.2 ms、87 GFLOPs。RT-DETRv2-sは48.1、5.03 ms、20 M、60 GFLOPs。RT-DETRv2-lは53.4、9.76 ms、136 GFLOPs。このページはRT-DETRの学習時メモリの多さと収束の遅さを指摘し、制約のあるエッジデバイスには"less accessible"であるとしている（https://docs.ultralytics.com/compare/rtdetr-vs-yolo11 ）。

## B. 設計の枠組みを決める数値

- GTSRBの分類は両系統とも飽和している：文献ではCNNが99.46から99.81%、ファインチューニングしたViTとCLIPのエンコーダが98.5から99.3%である（`vlm_and_foundation.md`のA.1、D.2）。我々自身の実行では、ConvNeXt V2 attoが3.7 Mパラメータで99.81%、bazylのViT-B/16が97.00%であり、後者の差はレシピの影響である（`replication.md`の第4節）。
- 我々が行ったTransformerとCNNの検出比較はゼロショットのものだけである：GTSDBのフレームでのクラス非依存のAP50は、OWLv2（ViT-B/16）が77.1、YOLO-World-sが54.2であった（`replication.md`の第7節）。したがって、我々の測定にはTransformerのほうが標識の検出性能が劣ることを示すものはない。ただしコストは高い。
- 計算予算：5 TOPSのフロントカメラ用SoCを利用率30から50%、30 fpsで使うと、全機能で使えるのは1フレームあたりおよそ50から80 GOPsである。1280 pxのnanoまたはsmallのCNN検出器（約25から90 GFLOPs）は収まる。640 pxで60 GFLOPsのRT-DETR-sを、標識に必要な解像度（16から48 pxの物体。`dataset_catalog.md`を参照）で動かすと、他の機能の余地が残らない。

## C. 推定アーキテクチャ（どの現行製品についても文書化されていない）

1. 検出器：粗い標識クラス（形状と色の系統、または単一の"sign"クラス）を持つ高解像度のCNNワンステージネットワークで、他のカメラ機能とバックボーンを共有する（TeslaのHydraNetがこの共有の文書化された例である）。
2. 追跡と空間的記憶：候補をフレームをまたいで確認し、認識した標識は車両が通過するまで位置とともに保持し（NDSS 2025）、トラック全体での投票で判定する（Mobileyeの特許、Keller 2008）。
3. クロップ上での分類：地域または国ごとの小さなネットワークで、検出器に手を加えずに差し替えられる（Mobileyeの"signature-based classification"の更新）。加えて、テキストと可変標識にはOCRを用いる（Mobileye ISA、Bosch）。
4. 関連性判定と融合：車線と出口への割り当て、道路種別からの暗黙の制限、任意の地図データとの融合（Mobileye ISA。オープンな同等物としてはFrogPilot。NDSSの車両では融合は見られなかった）。
5. Transformerが現れるとすれば、完全自動運転スタックのマルチカメラBEV融合と、OCRのシーケンスデコーダにおいてである。オフラインでは自動ラベル付けとデータ生成に用いられる。

これが最も可能性の高い形である理由：公開されている構成要素はすべてCNNである。5から34 TOPS級のアクセラレータとツールチェーンはCNN優先であり、INT8化による精度低下は無視できる程度である。分類タスクの精度の上限には小さなCNNで到達できる。機能の説明（更新可能な分類、OCR、関連性判定）はモジュール化されたパイプラインを前提としている。そして、ロングテールの要件（EU ISA、Euro NCAPの条件付き制限と暗黙の制限）は、バックボーンの選択ではなく、データとシステムのロジックで対処されている。

## D. OpenTSRにとっての意味

- 1280 pxまたはタイル分割した2048 pxで動作する粗いクラスのCNNワンステージ検出器を、INT8量子化後に評価したうえで学習・公開する。あわせて、数字用のOCRヘッドを持つ5 Mパラメータ未満のクロップ分類器も学習・公開する。二つのモデルは個別に差し替え可能にしておく。
- クロップだけでなく、追跡と関連性判定を含めた標識通過イベントで評価する（`requirements_and_robustness.md`のD.1）。ISAの距離ベースの90%という数値はイベントレベルの指標である。
- TransformerとVLMはオフラインで使う：候補ボックスにはOWLv2、数字の読み取りとラベル付けにはVLM、ロングテールには生成データを用いる（`vlm_and_foundation.md`）。
- コストに関する議論を仮定ではなく測定に基づくものにするため、ベンチマークにはRT-DETR / D-FINEのベースラインを、INT8での精度と組込み環境でのレイテンシの列とともに残しておく。

## E. 未解決の問い

- 詳細分類が独立したモデルかどうかを文書化している現行製品はない。積み重なった根拠は、2006年から2012年のものとオープンなスタックによるものである。
- EyeQ6 Highや同等の30 TOPS級SoCは融合段でアテンションブロックを実行している可能性があるが、それを標識認識と結び付ける公開情報はない。
- NDSSの研究では、米国の4台の車両で地図データとの融合は見られなかった。欧州のISA実装（Mobileyeのものは"without relying on a third-party map"）はOEMによって異なる可能性がある。
