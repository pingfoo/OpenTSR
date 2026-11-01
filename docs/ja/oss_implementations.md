# 道路標識認識のオープンソース実装（分類と検出）

> 本文書は英語版 [docs/oss_implementations.md](../oss_implementations.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

2026-10-31に作成した調査の草稿である（claude.aiのリサーチセッション、pingfoo/OpenTSR）。以下の各記述は、同日に公開中のページ、rawファイル、APIレスポンス、gitのshallow clone、またはリモートのzip一覧と照合して確認した。各記述を裏付けるURLはその直後の括弧内に示す。「不明」は事実を検証できなかったことを意味する。スター数と「最終コミット」の値はshields.ioのバッジ（https://img.shields.io/github/stars/OWNER/REPO.json, .../last-commit/..., .../license/...）から取得したもので、概数である。正確なHEADの日付は、2026-10-31に作成したshallow cloneの`git log -1`による。

対象範囲：道路標識のみ（信号機の調査は`traffic_lights/docs/oss_implementations.md`にある）。視覚言語モデル、CLIP形式のゼロショット分類器、オープン語彙検出器は別文書で扱い、本文書では参照先としてのみ示す（C4節とG節）。

本セッションでの検証上の制約：github.comのHTMLページは、そのURLが先にWeb検索結果に現れた場合にのみ読めた。api.github.comは利用できなかった。sid.erda.dk（公式GTSRBおよびGTSDBアーカイブのホスト）はサンドボックスから到達できない。universe.roboflow.com、doaj.org、pmc.ncbi.nlm.nih.govは自動リクエストを拒否またはブロックした。Kaggleのノートブックページには到達できなかった。Papers with Codeはオフラインであり、引用していない。

## 0. 要点

1. GTSRBのリーダーボードは2018年以降更新されていない。公式結果ページには、Arcos-Garcia et al.（DeepKnowledge Seville）の"CNN with 3 Spatial Transformers"が99.71%、IDSIAのCNNコミッティが99.46%、人間が98.84%、SermanetのマルチスケールCNNが98.31%、ランダムフォレストが96.14%、HOGに対するLDAが95.68%と記載されている(https://benchmark.ini.rub.de/gtsrb_results.html)。ダウンロード可能な重みを伴う公式コードがあるのは99.71%のエントリのみであり、それもTorch7/Luaで、CUDAの空間変換ネットワークモジュールが必須である(https://github.com/aarcosg/tsr-torch)。IDSIA、Sermanet、HOG/LDA、ランダムフォレストのエントリには公式コードがない。IDSIA形式のネットワークの再実装として最良のものはMoodstocks/gtsrb.torchである（2つのSTNで99.67%、Torch7）。
2. 現代的なフレームワークで、重みが手元にある状態でGTSRB分類を行う場合、実用的な選択肢はpoojahira/gtsrb-pytorch（STN + CNN、リポジトリ内に3.4 MBの`model_40.pth`、NYU Kaggleのprivateリーダーボードで99.809%）、hello2all/GTSRB_Keras_STN（Keras、13.9 MBの`weights.hdf5`、Udacityのpickle形式テストセットで99.59%）、およびHugging Face上のViTのファインチューニング済みモデル（kelvinandreas 98.46%、bazylは独自の評価用分割で99.93%）である。Udacityのpickle形式GTSRB（32x32でtrain 34,799 / valid 4,410 / test 12,630）は現在もCloudFrontからダウンロードでき、公式の12,630枚のテストセットはHugging Face（tanganke/gtsrb）にミラーされている。torchvisionの`datasets.GTSRB`はsid.erda.dkからダウンロードするため、この点は重要である。
3. 重みと公表された数値がある検出：aarcosg/traffic-sign-detection（TF1 Object Detection API、Google Drive上に8個のモデルzip、GTSDBの3スーパークラスでのmAPはFaster R-CNN Inception ResNet V2の95.77からSSD MobileNetの61.64まで）、skokec/detectron-traffic-signs（Caffe2 Detectron、644 MBのDFG Mask R-CNNの重み、DFGの200クラスでmAP50 95.5）、TT100K公式のCaffeモデル（801 MBの`code.zip`内に467 MBの`model.caffemodel`、45クラスでIoU 0.5におけるaccuracy 0.877 / recall 0.906）、およびTT100Kで学習したAMD/Xilinx Vitis AI 2.5のYOLOX-m（PyTorchのfloat、QAT、INT8 xmodelを含む594 MBの公開zip、TT100K valでAP 62.3）。4つとも古いフレームワークかGPU前提であり、クリーンな`yolo val`プロトコルを持つ2024〜2026年のYOLOは1つもない。
4. 2024〜2026年の2つのリリースはOpenTSRに直接役立つ。（a）Panoramax（フランスのオープンなストリートレベル画像プロジェクト）は、標識・ナンバープレート・顔のYOLO11l検出器（imgsz 2048で標識のAP50 0.898）と、フランス、ドイツ（130以上のクラス、accuracy 0.986）、ベルギー、オランダの標識のYOLO-cls分類器を、20k枚以上の画像からなる学習セットとともにEtalab-2.0 / CC BY-SA 4.0でHugging Face上に公開している。（b）Infra Intelligence（日本）は`road-sign-yolox-tiny`を公開している。これはApache-2.0のYOLOX-Tinyで、日本の標識板を道路標識令の5カテゴリ（guide/warning/regulatory/indication/supplementary）で検出する。ONNXとPyTorchの重み、日本語のモデルカードがあり、held-outテストでAP50 0.902、クラス非依存のrecall 0.857で、日本のMapillary画像で学習されている。評価画像は公開されていない。
5. 視覚ベースのTSRを公開している量産相当スタックはない。Autoware UniverseとBaidu Apolloには道路標識の認識モジュールがなく（信号機と、HDマップ上の一時停止/譲れ標識のみ）、openpilotは"openpilot does not detect speed limits at this time"と述べている。唯一の例外はopenpilotのフォークであるFrogPilotで、道路カメラの映像からCPU上で速度制限標識を読み取る12.6 MBのMNNモデル`speed_limit_reader.mnn`を同梱し、その出力を"Speed Limit Controller"とクラウドソースの"Speed Limit Filler"に渡している。このモデルのアーキテクチャと学習データは文書化されていない。sunnypilotは車両内蔵のTSR（CAN）とOpenStreetMapのデータのみを使う。

## A. 深層学習以前と初期の深層学習（2011〜2015年）

### A1. IJCNN 2011 GTSRBコンペティション（最終結果ページ）

- 公式の表（チーム、手法、最終テストセットでの正認識率）：DeepKnowledge Seville, CNN with 3 Spatial Transformers, 99.71; IDSIA, Committee of CNNs, 99.46; COSFIRE, color-blob-based COSFIRE filters, 98.97; INI-RTCV, Human Performance, 98.84; sermanet, Multi-Scale CNNs, 98.31; CAOR, Random Forests, 96.14; INI-RTCV, LDA on HOG 2, 95.68; LDA on HOG 1, 93.18; LDA on HOG 3, 92.34 (https://benchmark.ini.rub.de/gtsrb_results.html)。このページには参考文献のみが示され、コードへのリンクはない（同URL）。テストセットは12,630枚である(この枚数は、これを用いて評価するコミュニティのリポジトリに記載されている。例："a total of 12,630 images that we used for testing", https://raw.githubusercontent.com/navoshta/traffic-signs/master/README.md)。
- Ciresan, Meier, Masci, Schmidhuber（IDSIA）、"Multi-column deep neural network for traffic sign classification"(Neural Networks 2012。MCDNNのarXiv版はhttps://arxiv.org/abs/1202.2745)。公式コード：見つからなかった(arXiv論文は内製のGPUコードを記述しているが、PDFにリポジトリへのリンクはない。https://arxiv.org/pdf/1202.2745)。最もよく知られた再実装：Moodstocks/gtsrb.torch（`idsia_net.lua`）。IDSIA類似の単一ネットワークで98.85%、空間変換ネットワーク1つで99.46%、2つで99.67%と報告している(https://raw.githubusercontent.com/Moodstocks/gtsrb.torch/master/README.md)。リポジトリの情報：MIT、177スター、最終コミットMay 2016、Torch7/LuaJIT、`--no_cuda`はSTNモジュールなしでのみ動作する(同README; https://img.shields.io/github/stars/Moodstocks/gtsrb.torch.json)。
- SermanetとLeCun、"Traffic Sign Recognition with Multi-Scale Convolutional Networks"（IJCNN 2011）。論文によれば、ネットワークは"was implemented using the EBLearn C++"ライブラリ(http://eblearn.sf.net)であり、phase Iで98.97%、ネットワークを拡大した後に99.17%を報告している（http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf, 1ページと5ページ）。GTSRBのデモが現在もEBLearnのSourceForgeツリー内にあるかは未検証である。このアーキテクチャ（2段の畳み込みステージ、分類器に入力されるマルチスケール特徴、グレースケール入力）は、Udacity時代のリポジトリの大半がTensorFlowで再実装したものである（B節）。
- INI-RTCVのベースライン（LDA on HOG 1/2/3、92.34〜95.68%）：コードは公開されていない。HOG特徴量そのものは事前計算済みファイルとしてデータセットとともに配布された（GTSRB_Final_Training_HOG.zipなど、掲載先はhttps://benchmark.ini.rub.de/gtsrb_dataset.html; ホストsid.erda.dkに到達できなかったため、ファイル一覧は未検証）。
- ZakloutaとStanciulescu（CAOR、Mines ParisTech）、ランダムフォレスト96.14%、およびその後のHOGに対するK-d木 / ランダムフォレスト：公式コードは見つからなかった（結果ページ以上の検証はしていない）。
- COSFIRE（Gecer, Azzopardi, Petkov）、98.97%：著者らのCOSFIREのMATLABコードは一般には存在するが、GTSRB専用のリリースは未検証である。

### A2. IJCNN 2013 GTSDBコンペティション

- 完全な結果表（約280行、指標はカテゴリごとの"Area under Curve"と"Average overlap"）が公開されている(https://benchmark.ini.rub.de/gtsdb_results_all.html)。上位のエントリ：prohibitoryのAUC 100%はwgy@HIT501（ELL_SVM2、overlap 90.08%）、visics（boosted_intChn_ratio_scales、88.22%）、LITS1（intColorShapeAppearance、86.91%）、glc12125（robok、85.57%）、DSL_clusters（ToneMap+AcfChunks、83.56%）。mandatoryのAUC 100%はglc12125、DSL_clusters、wgy@HIT501（HOG_LDA_SVM2）。dangerのAUC 100%はvisics、glc12125、s_trafficsign（DL2、"danger 2, 9k aug TR"）、DSL_clusters。qcom VSSD3はmandatory 99.16% / danger 99.93%（同URL）。このページにはコードへのリンクもプロトコルの詳細もない（同URL）。コンペティションの論文はHouben et al.、"Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark"である(https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf)。
- 各エントリの公式コード：見つからなかった。visicsのエントリ（Mathias, Timofte, Benenson, Van Gool、"Traffic sign recognition, how far are we from the solution?"、IJCNN 2013）はDollarのツールボックス流のintegral channel featuresを用いた。道路標識専用のリリースは未検証である。
- コードと重みを伴いGTSDBで評価した後年の再実装はD節にある（aarcosg 2018、Signfeld 2020、hellojialee 2018）。

### A3. Haloi 2015（空間変換ネットワークとInception）

- Mrinal Haloi、"Traffic Sign Classification Using Deep Inception Based Convolutional Networks"、arXiv 1511.02992（2015）は、GTSRBで99.81%を報告している（https://arxiv.org/pdf/1511.02992v2.pdf, 1ページ）。PDFにはコードへのリンクがない（抽出したテキストを"github"と"code"でgrepして確認）。広く使われている再実装は見つからなかった（2026-10-31のWeb検索では論文アグリゲータのみがヒットした）。空間変換ネットワークを用い、コードを公開している後年の研究：Moodstocks/gtsrb.torch（2015〜2016、A1）、aarcosg/tsr-torch（2018、C1）、hello2all/GTSRB_Keras_STN（2018、C2）、poojahira/gtsrb-pytorch（2018、C2）。

## B. Udacity Self-Driving Car Nanodegree "Traffic Sign Classifier"の時代（2016〜2018年）

共通プロトコル：Udacityが公開したpickle形式の32x32のGTSRB。"train.p / valid.p / test.p"で、43クラス、学習34,799枚、検証4,410枚、テスト12,630枚である(https://raw.githubusercontent.com/hello2all/GTSRB_Keras_STN/master/capstone_report.md)。zipは現在もオンラインにある：https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip （2026-10-31時点でHTTP 200、Content-Length 123,524,425バイト）。特記のない限り、精度は12,630枚のテスト画像でのものである。ほぼすべてのコードはTensorFlow 0.12〜1.xであり、書き換えなしでは現行のPythonでは動かない。

| リポジトリ | スター数 / 最終コミット / ライセンス | 手法とフレームワーク | 報告されたテスト精度 | 重み |
|---|---|---|---|---|
| udacity/CarND-Traffic-Sign-Classifier-Project（テンプレート） | 372、Nov 2021、MIT (https://img.shields.io/github/stars/udacity/CarND-Traffic-Sign-Classifier-Project.json) | スターターノートブック、LeNet lab | なし（テンプレート） | なし |
| navoshta/traffic-signs | 148、Dec 2019（HEAD 2019-12-23）、Apache-2.0（shields、クローン） | TensorFlow。3層の畳み込み（5x5、32/64/128）+ FC 1層（1024）、スキップ接続、dropout、L2、データ拡張、拡張セットでの2段階の事前学習(https://raw.githubusercontent.com/navoshta/traffic-signs/master/README.md) | "99.33% accuracy on the test set"、"12,630 images"、誤りは"85 examples"（同README） | なし（ノートブックのみ。クローンのファイル一覧による） |
| kenshiro-o/CarND-Traffic-Sign-Classifier-Project | 145、Aug 2017（HEAD 2017-08-26）、ライセンスなし | TensorFlow、3x3畳み込みのモデル、グレースケール、dropout p-conv 0.6 / p-fc 0.5、imgaugによるデータ拡張(https://raw.githubusercontent.com/kenshiro-o/CarND-Traffic-Sign-Classifier-Project/master/writeup_template.md) | "test accuracy of 97.75%"（同writeup） | なし |
| mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow | 73、Dec 2019、MIT | TensorFlow 0.12.1、LeNetおよびVGG形式のネットワーク、グレースケール(https://raw.githubusercontent.com/mohamedameen93/German-Traffic-Sign-Classification-Using-TensorFlow/master/README.md) | "+99% validation accuracy, and a 97.6% testing accuracy"（同README） | なし |
| vxy10/p2-TrafficSigns（Vivek Yadav） | 67、Feb 2017（HEAD 2017-02-09）、ライセンスなし | TensorFlowのノートブック。READMEの表には、Haloi 99.81%と"Committee of CNNs 99.47%"の横に"Our model 98.8%"が記載されている(https://raw.githubusercontent.com/vxy10/p2-TrafficSigns/master/README.md) | 98.8% | なし |
| jeremy-shannon/CarND-Traffic-Sign-Classifier-Project | 59、Jan 2017（HEAD 2017-01-16）、ライセンスなし | TensorFlow、Sermanet/LeCun形式のマルチスケールモデル、グレースケール、クラスあたり800枚までのデータ拡張(https://raw.githubusercontent.com/jeremy-shannon/CarND-Traffic-Sign-Classifier-Project/master/Traffic_Sign_Classifier.md) | ログ上の検証精度は0.987。テストの数値はmarkdownエクスポートにない（同ファイル） | あり：TF 0.12のチェックポイント`lenet.data-00000-of-00001`、`lenet.index`、`lenet.meta`がリポジトリにコミットされている（クローンのファイル一覧） |
| chsasank/Traffic-Sign-Classification.keras | 68、Sep 2018、MIT | KerasのCNN、ブログのチュートリアルhttps://chsasank.github.io/keras-tutorial.html (https://raw.githubusercontent.com/chsasank/Traffic-Sign-Classification.keras/master/README.md) | READMEに記載なし（ブログは約98%と主張しているが未検証） | なし |
| MehdiSv/TrafficSignsRecognition | 58、Dec 2016、ライセンスなし、READMEファイルは取得していない | 不明 | 不明 | 不明 |
| vamsiramakrishnan/TrafficSignRecognition | 46、May 2017、ライセンスなし | TensorFlow、CLAHEによる正規化、10倍のデータ拡張(https://raw.githubusercontent.com/vamsiramakrishnan/TrafficSignRecognition/master/README.md) | "98.5% accuracy over the test set"（同README） | なし |
| pierluigiferrari/traffic_sign_classifier | 17、Nov 2017、ライセンスなし | TensorFlow 0.1xのCNN。pickle形式のデータはGoogle Drive上(https://raw.githubusercontent.com/pierluigiferrari/traffic_sign_classifier/master/README.md) | "97.9% on the official GTSRB test dataset"（同README） | なし |
| surmenok/GTSRB | 128、Nov 2019、MIT | fastai + PyTorch、ImageNetで事前学習済みのResNet-34(https://raw.githubusercontent.com/surmenok/GTSRB/master/README.md) | "99.22% accuracy on the validation set (random 20% subset of the training dataset)"であり、テストセットではない（同README） | なし |
| magnusja/GTSRB-caffe-model | 15、Oct 2019、ライセンスなし | Caffe AlexNet 256x256、"fill"方式のリサイズ(https://raw.githubusercontent.com/magnusja/GTSRB-caffe-model/master/README.md) | "roughly 97%"（同README） | git-lfsまたはNextcloudのリンクでcaffemodelを提供（同README。ダウンロードはしていない） |
| ndrplz/self-driving-car（Udacityの全プロジェクト） | 2.9k、Oct 2020、ライセンスなし（shields） | TensorFlowのノートブック。プロジェクト2が標識分類器 | 未確認 | なし |

現時点での実行可能性：TensorFlow 0.12 / 1.xのコードはすべて「困難」である（TF1にはPython 3.12以降向けのwheelがない。`tf.compat.v1`でfrozen graphは読み込めるが、これらのリポジトリはSavedModelではなくノートブックである）。データはCloudFrontのzipのおかげで「容易」である。このグループでコミットされている重みファイルはjeremy-shannonのチェックポイントのみであり、TF 0.12のチェックポイントを読み込むにはTF 1.xが必要である。

## C. 公開重みのあるPyTorchおよびKerasのGTSRB分類器

### C1. Arcos-Garcia, Alvarez-Garcia, Soria-Morillo：aarcosg/tsr-torch（GTSRBリーダーボード1位）

- 論文："Deep neural network for traffic sign recognition systems: An analysis of spatial transformers and stochastic optimisation methods"、Neural Networks 99 (2018) 158-165。3つの空間変換ネットワークを持つCNNで、"accuracy of 99.71%"(https://raw.githubusercontent.com/aarcosg/tsr-torch/master/README.md)。
- リポジトリ：MIT、41スター、最終コミットOct 2018（HEAD f21a9c1 2018-10-18）（https://img.shields.io/github/stars/aarcosg/tsr-torch.json; クローン）。Torch7（Lua）で、torch/nn、image、cutorch、cunn、cudnn、qassemoquab/stnbhwd、nninitを使用する（README）。事前学習済みモデル`gtsrb_cnn3st_pretrained.zip`はGoogle Drive上にある（id 1iqX1TZBxJSEmaEoReK2ZDko03JHqw1bp、README）。学習コマンドの例：`th main.lua -data GTSRB ... -netType cnn3st -globalNorm -localNorm -cNormConv`（README）。fb.resnet.torchとMoodstocks/gtsrb.torchを基にしている（README）。
- 現時点での実行可能性：困難（Torch7はメンテナンス停止。cutorch/cunn/cudnnとstnbhwdは古いCUDAを必要とし、STNにはCPUで動く経路がない）。

### C2. 重みがコミットされたPyTorchおよびKerasのリポジトリ

- poojahira/gtsrb-pytorch："CNN with 3 layers, 2 fully connected layers and 1 spatial transformer network layer"、特徴マップ3-100-150-250-350、フィルタ[5,3,3]。学習時とテスト時に強いデータ拡張を行う。"392,090 images for training"。NYU Kaggle InClassチャレンジ(https://www.kaggle.com/c/nyu-cv-fall-2018)で"99.809% on the Private Leaderboard and 99.746% on the Public Leaderboard"。モデルは"trained on combination of training + validation sets"であり、テスト時のランダムなデータ拡張のため、予測を繰り返すと99.699%から99.746%の間で変動する(https://raw.githubusercontent.com/poojahira/gtsrb-pytorch/master/README.md)。重み`model/model_40.pth`（3,430,539バイト）と`model/pred.csv`がリポジトリにある（クローンのgit ls-tree）。60スター、ライセンス未指定、HEAD b4c44b0 2026-02-19（https://img.shields.io/github/stars/poojahira/gtsrb-pytorch.json; クローン）。KaggleのテストセットはGTSRBのテストセットのラベルを非公開にしたもので、Kaggleによりpublic/privateに分割されている。正確な分割は不明である。テンプレートのリポジトリ：soumith/traffic-sign-detection-homework (50スター、Oct 2018、https://raw.githubusercontent.com/soumith/traffic-sign-detection-homework/master/README.md)。現時点での実行可能性：容易（素のPyTorch、torchvisionのtransforms、CPUで問題ない）。
- hello2all/GTSRB_Keras_STN："99.59% Accuracy on German Traffic Sign Recognition without Preprocessing nor Augmentation"。カスタムの空間変換ネットワーク層を持つKerasで、Udacityのpickle（34,799 / 4,410 / 12,630）で150エポック学習している。カテゴリ別の精度は99.60 / 99.53 / 100 / 99.81 / 99.87 / 99.69 / 99.17で、人間は99.72 / 98.67 / 98.89 / 98.00 / 99.93 / 97.63 / 100である(https://raw.githubusercontent.com/hello2all/GTSRB_Keras_STN/master/capstone_report.md)。重み`output/weights.hdf5`（13,904,448バイト）がリポジトリにある（クローン）。MIT、51スター、HEAD 4c40f72 2020-02-07（https://img.shields.io/github/stars/hello2all/GTSRB_Keras_STN.json; クローン）。現時点での実行可能性：中程度（カスタム層を含むKeras 2のHDF5。`tf.keras`のレガシーな読み込みかKeras 2.xが必要）。
- tomlawrenceuk/GTSRB-Dataloader：PyTorch 1.0の`Dataset`ラッパー。データはOneDriveのzip。20スター、Apr 2019、ライセンスなし、モデルなし(https://raw.githubusercontent.com/tomlawrenceuk/GTSRB-Dataloader/master/README.md)。
- hellojialee/Traffic_Sign_Recognition_Efficient_CNNs：LiとWang、"Real-time traffic sign recognition based on efficient CNNs in the wild"、IEEE T-ITS 20(3) 975-984の付属コード。分類器とMobileNet Faster R-CNN検出器のKerasノートブックで、GTSDBのサンプル画像がコミットされている。MIT、154スター、HEAD 6325032 2020-04-06（https://raw.githubusercontent.com/hellojialee/Traffic_Sign_Recognition_Efficient_CNNs/master/README.md; クローン）。READMEに精度の数値はない。重み：`config.pickle`のみで、ツリー内にモデルファイルは見当たらない（クローン、先頭40エントリ）。
- dineshresearch/Novel-Deep-Learning-Model-for-Traffic-Sign-Detection-Using-Capsule-Networks：Kerasのカプセルネットワーク、"state-of-the-art accuracy of 97.6% on GTSRB"（arXiv 1805.04424）、MIT、126スター、Jul 2018、READMEに重みは見当たらない(https://raw.githubusercontent.com/dineshresearch/Novel-Deep-Learning-Model-for-Traffic-Sign-Detection-Using-Capsule-Networks/master/README.md)。
- frogermcs/GTSRB-TensorFlow-Lite：MobileNet v1の転移学習とTFLiteへの変換、Colabノートブック、MIT、15スター、Sep 2020(https://raw.githubusercontent.com/frogermcs/GTSRB-TensorFlow-Lite/master/README.md)。
- Kaggleのノートブック：調査していない。サンドボックスからKaggleのコードページを取得できず、Web検索でもノートブックのURLは見つからなかった。Kaggleでよく使われるGTSRBのミラーはユーザーによるアップロードであるため、そこに投稿された数値は同じ12,630枚のテスト画像でのものだが、プロトコルは検証できない。

### C3. GTSRBで学習したHugging Face Hub上のモデル（APIクエリ`search=gtsrb`と`search=traffic sign`、2026-10-31）

2つのクエリはそれぞれ100件と89件のモデルを返した（`traffic-sign`、`tt100k`、`gtsdb`、`road sign`、`speed limit`、`mapillary`と合わせるとユニークで254件）。ヒットのうち約60件は`YusufBayram-Personal/shufflenet_GTSRB_krum_round_*`という連合学習のチェックポイントで、約25件はtext-to-imageのLoRAか無関係なモデルである。関連する分類器のエントリ（ダウンロード数はAPIが示す30日間の値）：

| モデル | 種類 / ベース | 主張されている精度とプロトコル | ダウンロード数 | ライセンス（カード） |
|---|---|---|---|---|
| bazyl/gtsrb-model | transformersのViT。google/vit-base-patch16-224-in21kをbazyl/GTSRBでファインチューニング(https://huggingface.co/bazyl/gtsrb-model/raw/main/README.md) | 10エポック後のmodel-indexのaccuracy 0.99932（"on the evaluation set"）。カードには評価セットが公式テスト分割であるとは書かれていない（同README） | 87 | apache-2.0 |
| kelvinandreas/vit-traffic-sign-GTSRB | ViT base patch16 224 in21k、safetensors 343,350,100バイト(https://huggingface.co/api/models/kelvinandreas/vit-traffic-sign-GTSRB?blobs=true) | GTSRBで"Accuracy: 0.9846, F1 Score: 0.9846"、分割は記載なし(https://huggingface.co/kelvinandreas/vit-traffic-sign-GTSRB/raw/main/README.md) | 171 | mit |
| cvetanovskaa/vit-base-patch16-224-in21k-gtsrb-tuned | ViT、データセットはbazyl/GTSRB | READMEに数値なし | 48 | apache-2.0 |
| dima806/traffic_sign_detection | ViT in21kの分類器（検出器ではない） | "about 94% accuracy"、9,252枚でのclassification reportのaccuracy 0.9388(https://huggingface.co/dima806/traffic_sign_detection/raw/main/README.md)。データセット名はGTSRBではない | 15 | apache-2.0 |
| xpatox08/gtsrb-traffic-signs-v1, gtsrb-vit, gtsrb-vit-model | ViT base patch16 224 | 数値なし | 12, 3, 3 | apache-2.0 |
| tzhao3/vit-GTSRB, vit-L-GTSRB, DeiT-GTSRB, DeiT-S-GTSRB | ViT / DeiT | READMEなし | 10, 7, 6, 16 | なし |
| sai-prakash-c/swinv2-tiny-patch4-window16-256-gtsrb-ft | SwinV2 tiny | READMEなし（結果のjsonのみ） | 6 | なし |
| Hydrogenium9/gtsrb-traffic-sign-models | Keras：best_convnexttiny.keras, best_custom_cnn.keras, best_efficientnetv2b0.keras | READMEなし | 28 | なし |
| jack-perlo/MobileNet-Gtsrb, MobileNetV2-Gtsrb, ResNet18-Gtsrb, ResNet34-Gtsrb | Keras .kerasとint8/uint8 .tflite | READMEなし | 21, 19, 11, 5 | なし |
| finnff/GTSRB_trained | torchvisionのEfficientNet / MaxViT / RegNetの.pth、10エポック | "Validation Accuracy: 99.83%"（EfficientNet）、分割は記載なし(https://huggingface.co/finnff/GTSRB_trained/raw/main/README.md) | 0 | mit |
| xieyuy/traffic-sign-adverse-weather | timm ConvNeXtV2-base、EfficientNetV2-M、SwinV2-B | "Macro-F1 = 0.945 (Evaluated on 2-Core CPU without GPU)"、独自の悪天候セット(https://huggingface.co/xieyuy/traffic-sign-adverse-weather/raw/main/README.md) | 0 | mit |
| tanganke/clip-vit-base-patch32_gtsrb, clip-vit-large-patch14_gtsrb, clip-vit-base-patch16_gtsrb（+ LoRA版）, crisostomi/ViT-*-GTSRB, gladia/ViT-B-16__*-gtsrb, hoffman-lab/KnOTS-*_gtsrb, yongxianwei/clip-base-32_gtsrb, EvaristeL/vitb32-GTSRB | モデルマージ / タスク算術の研究によるCLIPのファインチューニング済みモデル | VLMの文書を参照。Hub上で最もダウンロードされているGTSRBモデル（19,362 / 13,333 / 11,497） | | 記載なし |

これらのモデルが使用したデータセット（再現用）：tanganke/gtsrb (train 26,640 / test 12,630に加えてcorruption分割、parquet、16,418ダウンロード、ライセンスタグなし、https://huggingface.co/api/datasets/tanganke/gtsrb?full=true)、bazyl/GTSRB (parquetのtrain/test、ライセンスタグgpl-3.0、2,295ダウンロード、https://huggingface.co/api/datasets/bazyl/GTSRB?full=true)、FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset (合成データ、cc-by-4.0、152ダウンロード。論文はarXiv 2512.05936で、ドイツの211クラス、105,500枚、https://arxiv.org/abs/2512.05936)。torchvisionの`datasets.GTSRB`は`https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/`からmd5チェック付きでダウンロードする(https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/gtsrb.py)が、このホストには本サンドボックスから到達できない。

### C4. 参照先：VLM / CLIP / オープン語彙

CLIPのファインチューニング済みモデル（tanganke/*, crisostomi/*, gladia/*, hoffman-lab/KnOTS-*）、標識理解のためのQwen3-VLおよびGemma-3のLoRA（longhoang01/Qwen3-VL-8B-Instruct-Traffic-Signs、RidaKo/qwen3-vl-road-signs-lt-lora、AKrasavcev/lora_gemma3_4b_lt_road_signs、GuilhermeRatti/qwen2-7b-instruct-trl-sft-GTSDB_Detect）、およびVLM評価用にNEC LabsがラベルをつけなおしたMapillaryの検証セット(arXiv 2508.02047, https://github.com/nec-labs-ma/relabeling, 2,000枚、12カテゴリ、画像はhttps://mapillary-signs.s3.us-west-2.amazonaws.com/images.zip; 出典はhttps://raw.githubusercontent.com/nec-labs-ma/relabeling/master/README.md)は、別のVLM文書で扱う。

## D. GTSDB、TT100K、MTSD、DFG、CCTSDB、STSでの検出（2016〜2026年）

### D1. aarcosg/traffic-sign-detection（GTSDB、TensorFlow Object Detection API、2018）

- 論文：Arcos-Garcia, Alvarez-Garcia, Soria-Morillo、"Evaluation of deep neural networks for traffic sign detection systems"、Neurocomputing 316 (2018) 332-344 (https://raw.githubusercontent.com/aarcosg/traffic-sign-detection/master/README.md)。
- リポジトリ：MIT、335スター、最終コミットFeb 2020（HEAD 9949b3d 2020-02-13）。TensorFlow Object Detection API（TF1）と、YOLOv2用のdarkflow（shields、クローン、README）。ファイル：`Evaluation.ipynb`、`Run_models_on_new_images.ipynb`、`scripts/create_gtsdb_tf_record.py`、`scripts/create_gtsdb_3classes_tf_record.py`、ラベルマップ（クローン）。
- クラス：3つのスーパークラス`prohibitory`、`mandatory`、`danger`(https://raw.githubusercontent.com/aarcosg/traffic-sign-detection/master/scripts/gtsdb3_label_map.pbtxt)。
- 重みとデータ：Google Drive上のGTSDBのTFRecords（id 1hKdjTsiFm_vdtZPdto0QJihThGFOPEkq）と、Google Drive上の8個のモデルzip：faster_rcnn_inception_resnet_v2_atrous、faster_rcnn_inception_v2、faster_rcnn_resnet_101、faster_rcnn_resnet50、rfcn_resnet101、ssd_inception_v2、ssd_mobilenet_v1、yolo_v2（README。本セッションではDriveのリンクからダウンロードしていない）。
- 結果（READMEの表。APは`Evaluation.ipynb`内のVOC 2010+のarea法により、IoU 0.5でクラスごとに算出。同ノートブックにはCOCO形式の0.5:0.95の関数もある）：Faster R-CNN Resnet 50 mAP 91.52、Resnet 101 95.08、Inception V2 90.62、Inception Resnet V2 95.77、R-FCN Resnet 101 95.15、SSD Mobilenet 61.64、SSD Inception V2 66.10、YOLO V2 78.83。小 / 中 / 大の標識のmAPは、例えばFaster R-CNN Resnet 101で70.89 / 94.17 / 88.87、SSD Mobilenetで22.13 / 55.32 / 82.06である(同README; https://raw.githubusercontent.com/aarcosg/traffic-sign-detection/master/Evaluation.ipynb)。
- 現時点での実行可能性：中程度。TF1のfrozen graphはTF 2.xの`tf.compat.v1`で実行できる。ノートブックはTF1のAPIを対象としている。DriveのGTSDB TFRecordsを使えば、到達できないsid.erda.dkを回避できる。

### D2. TabernikとSkocaj、skokec/detectron-traffic-signs（DFG、Mask R-CNN、2019）

- 論文："Deep Learning for Large-Scale Traffic-Sign Detection and Recognition"、IEEE T-ITS 2019、arXiv 1904.00649。OHEM、小/大ROIの均等なサンプリング、サイズで重み付けした損失を用いたMask R-CNN。200カテゴリ、DFGの分割は学習5,254枚 / テスト1,703枚、アノテーションは30 px以上。表III：Faster R-CNNはmAP50 92.4 / mAP50:95 80.4 / 最大recall 93.8。Mask R-CNN ResNet-50は、改良なしで93.0 / 82.3 / 94.6、改良ありで95.2 / 82.0 / 96.5、改良とデータ拡張ありで95.5 / 84.4 / 96.5（https://arxiv.org/pdf/1904.00649, VI.D節と表III）。
- リポジトリ：Apache-2.0（Detectronのライセンス）、38スター、最終コミットApr 2026（デフォルトブランチ`villard`上のHEAD 0c11316 2026-04-21）。`caffe2-modules/`にカスタムの`WeightedSigmoidCrossEntropyLoss` opを持つCaffe2 Detectronのフォーク（https://raw.githubusercontent.com/skokec/detectron-traffic-signs/villard/README.md; クローン）。
- 重み：`https://data.vicos.si/skokec/villard/dfg_mask_rcnn_weights.tar.gz`（HTTP 200、643,851,795バイト）、YAML設定`detectron-model-yaml-DFG-dataset.zip`（1,842バイト）とそのデータ拡張版（README、2026-10-31のcurl -I）。READMEには"Due to slight update of the DFG dataset the resulty may vary from the ITS 2019 paper"という警告がある。
- 現時点での実行可能性：困難（Caffe2 Detectron、カスタムCUDA op、GPUのみ）。

### D3. TT100K公式コード（Zhu et al.、CVPR 2016）

- サイト：https://cg.cs.tsinghua.edu.cn/traffic-sign/ 、論文"Traffic-Sign Detection and Classification in the Wild"。ダウンロードリンクは直接リンクである：`data_model_code/code.zip`（801,430,483バイト）、`cvpr_fastrcnn.zip`（370,022,700バイト、"Fast R-CNN Model and Results"）、`data_model_code/data.zip`（19,152,969,603バイト）、`tt100k_2021.zip`（19,205,284,669バイト）、およびそれぞれ約17 GBの5つの"no sign"パート（2026-10-31のHEADリクエスト）。ページには"Tsinghua-Tencent 100K dataset is under Creative Commons Attribution-NonCommercial (CC-BY-NC) license"および"The 2021 version ... could NOT be used independently"と記載され、ダウンロード前に氏名、メールアドレス、所属組織の入力を求めるが、フォームなしでもリンクは機能する（同ページ）。
- `code.zip`の内容（セントラルディレクトリをリモートで読み取り）：`code/caffe/`（gitの履歴を含むCaffeのフォーク）、`code/model/model.caffemodel`（467,396,710バイト）、`model.prototxt`、`solver.prototxt`、`train_val.prototxt`、`code/python/anno_func.py`、`data_show.ipynb`、`deployer.ipynb`、`eval_check.ipynb`、`code/script/1_convert_lmdb.sh`、`2_calc_img_mean.sh`、および`ours_result.json`、`fastrcnn_result.json`とレポート類を含む`code/results/`。`code/README.md`："caffe: caffe code that we used; model: our caffe model file; python: some python utils; results: some results in our experiments"。
- 報告されている数値（`code/results/report_ours.txt`）："iou:0.5, size:[0,400), types:[... total 45 ...], accuracy:0.877, recall:0.906"。サイズ別では[0,32) 0.822 / 0.867、[32,96) 0.913 / 0.941、[96,400) 0.905 / 0.875。Fast R-CNNのベースライン（`report_fastrcnn.txt`）：全体でaccuracy 0.501、recall 0.555、32 px未満の標識ではrecall 0.238。（ここでの"accuracy"は、レポートに列挙された45クラスにおけるIoU 0.5での検出の適合率（precision）である。論文がこの45クラスを選んだ基準は本調査では再確認していない。）
- 現時点での実行可能性：困難（BVLC Caffeのフォーク、LMDB変換スクリプト、GPU、19 GBのデータ）。

### D4. MTSDのベースライン（Mapillary、2020）

- 論文：Ertler et al.、"The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale"、arXiv 1909.04422（ECCV 2020）。分割はtrain 36,589枚 / val 5,320枚 / test 10,544枚と、部分的にアノテーションされた47,547枚。"the annotations for the test set will not be released"。ベースライン：フル解像度の1000x1000クロップで学習したFaster R-CNN FPNの二値検出器と、別個の分類器。"Our classifier in combination with ResNet101 binary detector reaches 83.4 mAP over all 313 classes; the ResNet50 variant is only about 2 points lower"（https://arxiv.org/pdf/1909.04422, 第2.5節と第5節）。PDFにはコードリポジトリの名前がない（github.comでgrepしても何も見つからなかった）。データセットはhttps://www.mapillary.com/dataset/trafficsign （アブストラクトによれば学術研究目的では無料）。
- MTSDのクロップのコミュニティによるミラーはHub上にある(ThankGod/mapillary_traffic_sign_datasetとその派生、647 / 117 / 63ダウンロード、https://huggingface.co/api/datasets?search=mapillary%20traffic)が、MTSDで学習した検出器の重みはGitHubにもHubにも見つからなかった。

### D5. CCTSDBとCCTSDB2021（CSUST）

- csust7zhangjm/CCTSDB2021：データセットの公開のみ（Google DriveとBaiduのリンク）、GPL-3.0のLICENSEファイル、102スター、HEAD de484cd 2026-07-25。"16356 training set images"、"positive sample test set has 1500 images"、mandatory / prohibitory / warningの3クラス、負例500枚。READMEにはこのグループの検出器（Neurocomputing 2026のHQ-YOLO、IEEE TETCI 2024のアンカーフリーの1段階検出器、ReYOLO 2022、cascaded R-CNN 2020）が列挙されているが、コードや重みへのリンクはない（https://raw.githubusercontent.com/csust7zhangjm/CCTSDB2021/master/README.md; クローン）。
- csust7zhangjm/CCTSDB（2017年版）：アノテーションのテキストのみ、271スター、HEAD 2024-07-12（クローン）。
- Genius-Society/yolov4_tt100k（Hugging Face）：MIT、READMEのみ、"Testing on the CCTSDB 2021 dataset showed ... mAP of up to 92.77%"、"81 frames per second"。重みはHubではなく、次のModelScopeのミラーにある：https://www.modelscope.cn/models/Genius-Society/yolov4_tt100k (Hubのファイル一覧には`.gitattributes`と`README.md`しかない、https://huggingface.co/api/models/Genius-Society/yolov4_tt100k?full=true)。カード以上の検証はしていない。
- HaoqinHong/CCSPNet-Joint（IJCNN 2024、ノイズ除去とYOLOv5l（EfficientViT-b2バックボーン）の結合モデル、CCTSDB-AUG）：GPL-3.0、28スター、最終コミットJan 2026。重みはKaggleとBaidu上にある(https://raw.githubusercontent.com/HaoqinHong/CCSPNet-Joint/master/README.md)。

### D6. Swedish Traffic Signs（STS / STSD）

- LiとWang 2018（T-ITS、リポジトリはhellojialee、C2）と"Efficient Traffic-Sign Recognition with Scale-aware CNN"(arXiv 1805.12289, "99.88% precision and 96.61% recall on the Swedish Traffic Signs Dataset", https://arxiv.org/abs/1805.12289)がSTSの主要な結果である。後者のアブストラクトページにはコードへのリンクがない。本調査ではSTSで学習した重みは見つからなかった。

### D7. 重みのあるYOLO系道路標識検出器（GitHubとHugging Face）

- keremberke：道路標識のモデルは存在しない(作者でのクエリは86件のモデルを返し、標識関連のものはなかった。https://huggingface.co/api/models?author=keremberke&limit=200)。関連するHub上のエントリはデータセットのkeremberke/german-traffic-sign-detectionである(Roboflow UniverseからのGTSDBの再エクスポート、train 383枚 / valid 108枚 / test 54枚、43ラベル、"License: CC BY 4.0", https://huggingface.co/datasets/keremberke/german-traffic-sign-detection/raw/main/README.md)。
- 重みと明記された指標を持つHub上の検出器（すべてUltralytics形式の`.pt`チェックポイント。Ultralyticsの学習・推論コードはAGPL-3.0であり、以下のライセンスはアップロード者がカード上でチェックポイントについて宣言したものである）：
  - cvtechniques/TrafficSignDetection：LISA traffic sign datasetでのYOLO11n、自身の検証分割で"mAP@0.5 0.994, mAP@0.5-0.95 ~0.89, Best F1 0.99"、mit、153ダウンロード(https://huggingface.co/cvtechniques/TrafficSignDetection/raw/main/README.md)。
  - neitpm/traffic-sign-yolo11-benchmark：独自の5,668 / 1,259 / 1,151分割でYOLO11nのベースラインが"70.27% mAP@[0.5:0.95]"、加えてUIB-nano版と蒸留版、Dockerfile、mit、93ダウンロード(https://huggingface.co/neitpm/traffic-sign-yolo11-benchmark/raw/main/README.md)。
  - star092304/traffic-sign-detection-vietnam-yolo：ベトナムの標識、データセットはHub上（CC BY 4.0、サイズn<1K）、"mAP50 98.06%, mAP50-95 83.57%"、apache-2.0、93ダウンロード(https://huggingface.co/star092304/traffic-sign-detection-vietnam-yolo/raw/main/README.md)。
  - OrDinaD/belarus-road-signs-yolov8："mAP50 0.7252, mAP50-95 0.6106"、mit、30ダウンロード(https://huggingface.co/OrDinaD/belarus-road-signs-yolov8/raw/main/README.md)。
  - ankitjha07/Traffic-Sign-Detection-YOLOv8（mit、128ダウンロード、results.csvはあるがカードに数値なし）、yahyagul/traffic-sign-yolov8（Roboflowのデータ、mit、86）、JakobJFL/yolov8-dk-Traffic-Signs（デンマーク、apache-2.0）、Nughm3/yolov8-traffic-signs、Erpix3lt/traffic-sign-detection（RoboflowのセットでのYOLO26、.ptと.onnx、mit）、jcasdvasj/yolov12-traffic-sign-detection（zipのみ、mit）、jkot/pova-traffic-sign-recognition-models（1280/1920のYOLO二値検出器と分類器、カードなし）(https://huggingface.co/api/models?search=traffic%20sign&limit=100)。
  - youssefhazem/yolov8s-tt100k：空のリポジトリ(`.gitattributes`のみ、https://huggingface.co/api/models/youssefhazem/yolov8s-tt100k?full=true)。
- GitHub(`traffic-sign-detection`および`traffic-sign-recognition`のトピックページより、それぞれ102件と192件のリポジトリ、https://github.com/topics/traffic-sign-detection, https://github.com/topics/traffic-sign-recognition)：
  - MDhamani/Traffic-Sign-Recognition-Using-YOLO：GTSRBのクロップでのYOLOv5、">93% mAP"、重みはmega.nz上、81スター、Jan 2023、ライセンスなし(https://raw.githubusercontent.com/MDhamani/Traffic-Sign-Recognition-Using-YOLO/master/README.md)。
  - AmeyaWagh/traffic_sign_detection：LISAでのdarkflow tiny-YOLO、pb/metaはGoogle Drive上、MIT、75スター、Jan 2020(https://raw.githubusercontent.com/AmeyaWagh/traffic_sign_detection/master/README.md)。
  - wyfsh/YOLO-LISA：LISA-TSでのDarknet YOLOv2、26スター、Jun 2017(https://raw.githubusercontent.com/wyfsh/YOLO-LISA/master/README.md)。
  - fredotran/traffic-sign-detector-yolov4：877枚のKaggle "road-sign-detection"セット（4クラス）でのOpenCV DNNによるYOLOv4、32スター、Nov 2025(https://raw.githubusercontent.com/fredotran/traffic-sign-detector-yolov4/master/README.md)。
  - bhaskrr/traffic-sign-detection-using-yolov11：Roboflowの"Self-Driving Cars"セット（4,969枚、15クラス、416 px）でのYOLO11n、34スター、Oct 2024(https://raw.githubusercontent.com/bhaskrr/traffic-sign-detection-using-yolov11/master/README.md)。
  - MehmetOKUYAR/yolov7_tensorrt_test：独自セットでのYOLOv7 + TensorRT、MIT、19スター(https://github.com/topics/traffic-sign-dataset)。
  - Savatage7/Yolo5_TT100K（0スター、Jun 2022）とlunian/Improved-YOLOv8（1スター、Mar 2025）が、検索で見つかった唯一のTT100KのYOLOリポジトリである。どちらのREADMEも重みについて記載していない(https://raw.githubusercontent.com/Savatage7/Yolo5_TT100K/master/README.md, https://raw.githubusercontent.com/lunian/Improved-YOLOv8/master/README.md)。
- コード付きのTT100K/CCTSDB向け改良YOLO論文（2021〜2026年）：Hubの論文検索で"traffic sign detection"を検索すると、YOLO-TS（arXiv 2410.17144、TT100KとCCTSDB2021）とEMDFNet（arXiv 2408.14189）が返る。どちらのPDFにもGitHubへのリンクはない(https://arxiv.org/pdf/2410.17144, https://arxiv.org/pdf/2408.14189; https://huggingface.co/api/papers/search?q=traffic%20sign%20detection)。YOLO-CCA（arXiv 2412.04289）とETSR-YOLO（PLOS ONE 2023）はWeb検索で見つかったが、コードの有無は確認していない。結論：TT100Kについて、ダウンロード可能であることを検証できた現代的な重みは、Vitis AIのYOLOX-m（D9）とModelScopeでホストされているyolov4_tt100kのみである。

### D8. RT-DETR、D-FINE、DEIM、RF-DETRおよびDETR系の道路標識検出器

- merve/rfdetr-roadsign-agree1/agree2（およびlarge/medium、noaugの各版）：名称不明の道路標識セットでのRF-DETRのファインチューニング、apache-2.0、agree2で"Map 0.2815, Map 50 0.3324"、7〜25ダウンロード(https://huggingface.co/merve/rfdetr-roadsign-agree2/raw/main/README.md)。
- Jesse020202/detr_finetuned_roadsigns（+ ポイズニング版 / 比較用の版）：Conditional DETR ResNet-50のファインチューニング、apache-2.0、カードに指標なし（APIの一覧）。
- iamrizwan077/detr-finetuned-road-signs-v1、maier-s/detr_fine_tuned_traffic_signs：カードなし。
- RT-DETR、D-FINE、DEIMの道路標識チェックポイントは、Hub上にもトピックページにも見つからなかった。CSDETR（改良版RT-DETR）は信号機向けには存在するが、標識向けではない（信号機の調査を参照）。

### D9. ベンダーおよびフレームワークの設定：Vitis AI、MMDetection、Detectron2

- AMD/Xilinx Vitis AI 2.5のmodel zooのエントリ21 "ADAS Traffic sign Detection, pt_yolox_TT100K_640_640_73G_2.5, 0.623 float / 0.621 quantized, 640*640, 73G"(https://raw.githubusercontent.com/Xilinx/Vitis-AI/v2.5/model_zoo/README.md)。モデルのyamlには、公開ダウンロードhttps://www.xilinx.com/bin/public/openDownload?filename=pt_yolox_TT100K_640_640_73G_2.5.zip （リダイレクト先はhttps://download.amd.com/opendownload/xlnx/pt_yolox_TT100K_640_640_73G_2.5.zip, HTTP 200、594,249,291バイト）、ボード用xmodel`tsd_yolox_pt-*`、Vitis AIのLICENSE（Apache-2.0）への参照が記載されている(https://raw.githubusercontent.com/Xilinx/Vitis-AI/v2.5/model_zoo/model-list/pt_yolox_TT100K_640_640_73G_2.5/model.yaml)。zipの内部（セントラルディレクトリをリモートで読み取り）：`float/best_ckpt.pth`（202,928,501バイト）、`qat/epoch_58_ckpt.pth`、`quantized/YOLOX_0_int.xmodel`、`exps/example/custom/yolox_m_tt100k_float.py`を含むYOLOXのコードツリー、`run_train.sh`（4 GPU、バッチ64、fp16。評価は`tools/eval.py ... --conf 0.001`）、および`readme.md`："performance is evaluated on TT100k val"、"YOLOx-m-modified, 640x640, 72.96G FLOPs, AP 62.3; QAT INT8 AP 62.1"、前処理は"RGB(0~255), mean 0, std 1"。APの定義（YOLOXと同じCOCOの0.5:0.95か、AP50か）は記載されていない。このエントリはVitis AI 3.0および3.5のmodel zooのREADMEにはない(grepで"yolox"が見つからなかった：https://raw.githubusercontent.com/Xilinx/Vitis-AI/v3.5/model_zoo/README.md)。
- MMDetection / mmpretrain：open-mmlab/mmpretrainに道路標識の設定はない（HEAD ee7f2e8 2024-11-01のクローンのツリーをgtsrb/trafficでgrepしても何も見つからなかった）。TIER IVのAWML（mmdet/mmpretrainベース）は信号機のモデルのみを学習する（信号機の調査を参照）。
- Detectron2：moabitcoin/signfeld（"Signfeld in PyTorch"、中立的な画像に標識テンプレートを貼り付けた合成データによる学習）：RetinaNet ResNet-50 "Calzone"、"GTSDB mAP 67.23"、重み`DE-signs-169.pth`はGitHub releases上（HTTP 200、317,883,321バイト）、ドイツの標識169クラス、MIT、13スター、May 2020(https://raw.githubusercontent.com/moabitcoin/signfeld/master/README.md; curl -I https://github.com/moabitcoin/Signfeld/releases/download/v1.0/DE-signs-169.pth)。関連：LCAD-UFES/publications-tabelini-ijcnn-2019(テンプレートと任意の自然画像からのFaster R-CNNの学習、IJCNN 2019、endernewton/tf-faster-rcnnを使用、生成したGTSDB形式のデータはGoogle Drive上、18スター、Jul 2020、https://raw.githubusercontent.com/LCAD-UFES/publications-tabelini-ijcnn-2019/master/README.md)。
- sovit-123/fasterrcnn-pytorch-training-pipeline（torchvisionのFaster R-CNNトレーナー、MIT、275スター、Oct 2025）は、チュートリアルでGTSDBとともによく使われる汎用パイプラインだが、標識の重みは同梱していない(https://raw.githubusercontent.com/sovit-123/fasterrcnn-pytorch-training-pipeline/master/README.md)。

### D10. Roboflow Universe

- "Traffic Sign Detection"プロジェクトは数十件ある（例えばhttps://universe.roboflow.com/traffic-signs-ryni7/traffic-signs-detection-auxjz: 7クラス、画像数や指標の表示なし、"LICENSE: CC BY 4.0"、"Hosted Inference API and other deployment methods"）。ライセンスはアップロード者ごとに選択され（確認したページではCC BY 4.0がデフォルト）、データセットに適用される。一方、Roboflowで学習したモデルの重みは、ダウンロード可能なファイルとしてではなく、APIキーを用いたRoboflowのホスト型またはセルフホスト型の推論を通じて提供される（確認したページには重みのダウンロード手段がない。それ以外の点ではこの記述は未検証）。データセットはログイン後にYOLO/COCO形式でエクスポートできる。keremberke/german-traffic-sign-detectionがミラーしているのは、RoboflowでのGTSDBの再エクスポート（mohamed-traore-2ekkp/gtsdb---german-traffic-sign-detection-benchmark）である（D7）。

### D11. Panoramax（フランス）：オープンな検出器と国別の分類器（2024〜2026年）

- Panoramax/detect_face_plate_sign："trained for the Panoramax project in order to detect people face to blur them, licence plates to blur them, road signs to classify them with other models"。imgsz 2048のYOLO11l、300エポック。329枚での検証：全体でmAP50 0.815 / mAP50-95 0.412、標識はP 0.879 R 0.836 mAP50 0.898 mAP50-95 0.561（507インスタンス）、ナンバープレートは0.889 / 0.438、顔は0.657 / 0.237。ファイルは`yolo11l_panoramax.pt`と`yolov8s_panoramax.pt`。ライセンスはetalab-2.0(https://huggingface.co/Panoramax/detect_face_plate_sign/raw/main/README.md)。
- Panoramax/classify_de_road_signs："fine tuned version of classify_de_road_signs classification model (based on YOLOv26)"、"130+ types of road signs"、model-indexのaccuracy 0.986、`classify_de_road_signs.pt`（26,282,865バイト）、etalab-2.0、2026-03-29作成(https://huggingface.co/Panoramax/classify_de_road_signs/raw/main/README.md; https://huggingface.co/api/models/Panoramax/classify_de_road_signs?blobs=true)。学習セットPanoramax/classified_de_road_signs："20000+ photos of DE road signs in 130+ classes"、ドイツのPanoramax画像からdetect_face_plate_signで作成したクロップ、追加の"bad"/"other"クラス、CC BY-SA 4.0、`train.zip`と`val.zip`(https://huggingface.co/datasets/Panoramax/classified_de_road_signs/raw/main/README.md)。
- Panoramax/classify_fr_road_signs（accuracy 0.98717、2024-06-23）、classify_be_road_signs（2026-03-02）、classify_nl_road_signs（2026-01-04）、detect_fr_road_signs_subsigns（2024）、detect_nl_subsignsとclassify_nl_subsigns（2026-09、CC BY-SA 4.0）：特記のない限りすべてetalab-2.0で、Ultralyticsの`.pt`ファイルである(https://huggingface.co/api/models?author=Panoramax&limit=50; https://huggingface.co/api/models/Panoramax/classify_fr_road_signs?full=true)。APIが示すダウンロード数はすべて0であり、ほとんど知られていない。
- 重要な理由：本調査で見つかった欧州の標識向けの検出器 + 細粒度分類器の連鎖のうち、オープンで、メンテナンスされており、寛容なライセンスで、学習データも公開されているものはこれだけである。なお、Ultralyticsの学習コードはAGPL-3.0であり、`.pt`ファイルはUltralyticsのチェックポイントである。

### D12. 日本：Infra Intelligence road-sign-yolox-tiny（2026）

- Hub上のモデルinfra-intelligence/road-sign-yolox-tiny（2026-10-22作成、34ダウンロード、重みとコードのライセンスはapache-2.0、タグはja/en/japan/traffic-sign）(https://huggingface.co/api/models/infra-intelligence/road-sign-yolox-tiny?full=true)。モデルカード(日本語で英語の要約付き、https://huggingface.co/infra-intelligence/road-sign-yolox-tiny/raw/main/README.md)：
  - タスク：車載スマートフォンまたはドライブレコーダーの画像から日本の道路標識板を検出し、道路標識令に従って形状と色により5クラスに分類する：`guide`（案内）、`warning`（警戒）、`regulatory`（規制）、`indication`（指示）、`supplementary`（補助）。加えて6番目の`hard_negative`スコア列（標識板の裏面、店舗の看板、信号機、視線誘導標、情報板）があり、これは標識として出力されない。標識の種類（"一時停止"や"最高速度"）は識別しない。
  - アーキテクチャ：YOLOX-Tiny、約5.03 Mパラメータ、Apple Silicon（MPS）上でスクラッチから学習、バッチ8、AdamW 1e-3 cosine、SimOTA、mosaic。入力は`[1,3,736,1280]`のRGB 0〜255（正規化はモデル内部で行う）。出力は`boxes [200,4]`と`scores [200,6]`で、内部にNMSはない。推奨される後処理：6列でのargmax、クラスごとのNMS（IoU 0.65）、しきい値0.5。
  - ファイル：`road_sign_yolox_tiny_v9.pth`（20,373,347バイト）、`road_sign_yolox_tiny_v9.onnx`（20,426,324バイト、opset 17、1280x736固定）、`model.py`（torch + torchvision + Pillowのみ）、`examples/predict_onnx.py`、`config.json`、`MODEL_CONTRACT.md`、`eval/evaluation.json`、`eval/onnx_parity.json`、`LICENSE`、`NOTICE`（blobsの一覧）。
  - データ：日本のMapillary画像（CC BY-SA 4.0）、15地域からの学習4,585枚と検証992枚。Mapillaryの自動検出を下書きとし、文書化されたルールのもとで"AI assistant"がボックスを確認・再描画したもので、1枚ずつのレビューはされていない。データセットは公開されていない。
  - 結果：別の7地域からの596枚（988ボックス、標識なし画像117枚）のheld-outテストで、しきい値0.5においてrecall 0.857（847/988、クラス非依存）、precision 0.877、マッチした標識板でのクラス精度0.980、標識なし画像100枚あたりの誤検出12.0件、AP50 0.902、AP50-95 0.700。クラス別のAP50はguide 0.594、warning 0.958（9ボックス）、regulatory 0.937、indication 0.893、supplementary 0.748。サイズ別（長辺2048 pxでの幅）のAP50は0.759（32 px未満）、0.861、0.878、0.642（128 px以上）。検証（992枚）：AP50 0.925、recall 0.869、precision 0.906。作成者は公開された`model.py`と重みをCPUで再実行し、同じval AP50 0.925を得た（"torch 2.13.0 で動くことを確かめています"）。
  - 作成者が述べている注意点：テストラベルもAIが生成したものである。近くにある大きな標識板、小さな交差点名の標識板や補助標識板に弱い。夜間、雨、雪は評価していない。日本のみ。セーフティクリティカルな用途向けではない。
- 現時点での実行可能性：容易（ONNX RuntimeのCPU実行または素のPyTorch）。評価画像が非公開のため数値は再現できないが、このモデルはOpenTSRの日本向け検出器ステージにとって自然な定性的ベースラインであり、ラベル空間の参照先となる。

### D13. その他の確認されたデータセット固有のリポジトリ

- ituracingdriverless/TTVS（トルコの道路標識データセット、80スター、May 2026、ライセンスバッジなし）。
- ptran1203/traffic_sign_detection（Zalo AI 2020、タイリングを用いたRetinaNet、27スター）。
- lynguyenminh/CS114.M11.Vietnam-traffic-sign-detection（47スター）、Hoang42とMinh124689のHub上のモデル（ベトナム）、aresad/russian-road-signs、OrDinaD（ベラルーシ）、JakobJFL（デンマーク）、eugeneyan84(シンガポール、SSD MobileNet V2 FPNLite、32クラスの分類セットは公開、検出セットはPDPAを理由に非公開、GPL-3.0、https://raw.githubusercontent.com/eugeneyan84/Classification-and-Detection-of-Singapore-Road-Traffic-Signs/master/README.md)。(スター数の出典：https://github.com/topics/traffic-sign-detection, https://github.com/topics/traffic-sign-recognition)

## E. 量産スタックと産業界のスタック

### E1. Autoware Universe

- autowarefoundation/autoware_universe、1.8kスター、最終コミット"yesterday"（HEAD 1a411ff 2026-10-30）、Apache-2.0（shields、クローン）。クローンのツリーを`traffic_sign`、`trafficsign`、`road_sign`、`speed_limit`、`speedlimit`、`tsr`で検索すると、`perception/autoware_map_based_prediction/.../traffic_signal.*`と`planning/autoware_diffusion_planner/.../traffic_signals.*`しか見つからず、これらは信号機に関するものである。道路標識の検出や分類のノードはない。速度制限はカメラではなくLanelet2マップ（regulatory elements）から得られる。AutowareのHugging Faceの組織は信号機と物体検出のモデルのみを公開している（信号機の調査を参照）。

### E2. Baidu Apollo

- ApolloAuto/apollo、27kスター、HEAD d53aa3d 2026-04-16、Apache-2.0（shields、クローン）。HEADの`modules/perception/`ディレクトリには、barrier_recognition, camera_detection_*, lane_detection, lidar_*, radar_*, traffic_light_detection / recognition / region_proposal / trackingがあり、道路標識のモジュールはない（クローンのgit ls-tree）。標識はHDマップの要素としてのみ存在する：`modules/common_msgs/map_msgs/map_stop_sign.proto`、`map_yield_sign.proto`、`map_speed_control.proto`、`map_speed_bump.proto`。Dreamviewはこれらを描画する（`modules/dreamview/frontend/src/renderer/traffic_controls/traffic_signs.js`）（同クローン）。

### E3. comma.ai openpilotとそのフォーク

- commaai/openpilot、64kスター、MIT、HEAD afedebc 2026-10-30（shields、クローン）。`docs/LIMITATIONS.md`："In presence of traffic signs or stop lights, which are not detected by openpilot at this time."および"When the posted speed limit is below the user selected set speed. openpilot does not detect speed limits at this time."(https://raw.githubusercontent.com/commaai/openpilot/master/docs/LIMITATIONS.md)。ツリー内に標識のモデルはない（クローンをsign / speed_limit / mapdでgrepしても何も見つからなかった）。
- sunnypilot/sunnypilot、2.2kスター、MIT、HEAD 6f7a2c9 2026-10-29（shields、クローン）。"Speed Limit Assist"には2つのソースがある："Car State: Speed limit information provided by the vehicle's built-in sensors (e.g., Traffic Sign Recognition cameras)"と"Map Data: Speed limits from downloaded OpenStreetMap data"(https://community.sunnypilot.ai/t/249)。リゾルバのコードもこれを裏付けている：`SpeedLimitSource.car`は`sm['carStateSP'].speedLimit`を、`SpeedLimitSource.map`は`sm['liveMapDataSP']`を読む。ポリシーは`car_state_only`、`map_data_only`、`car_state_priority`、`map_data_priority`、`combined`である(https://raw.githubusercontent.com/sunnypilot/sunnypilot/master/openpilot/sunnypilot/selfdrive/controls/lib/speed_limit/speed_limit_resolver.py)。独自のカメラベースの標識認識はない。
- FrogAi/FrogPilot、568スター、MIT、ブランチ`FrogPilot-Staging`上のHEAD 51e1121 2026-10-31（shields、クローン）。これは実際に視覚ベースのTSRを同梱している：
  - README："It reads speed limit signs with the road camera and, on supported Ford, Genesis, Hyundai, Kia, Lexus, and Toyota models, your car's dashboard. It also uses maps from OpenStreetMap and the speed limits along your Navigate on openpilot route, and, with your own Mapbox key, fills gaps with Mapbox's online data."。ソースの優先順位はユーザーが設定できる（"Speed Limit Source Priority"）。"Speed Limit Filler ... learns the limit from the signs the camera reads, saves it for that road and direction ... keeps a cropped photo of the sign"。"Share FrogPilot Data"をオンにすると、学習された制限値と写真がFrogPilotにアップロードされる(https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/README.md)。
  - コード：`frogpilot/system/speed_limit_reader.py`は`speed_limit_reader.mnn`をMNNランタイムでCPU上に読み込み（`{"backend": "CPU", "numThread": 1}`）、道路カメラを購読し（`VisionIpcClient("camerad", VISION_STREAM_ROAD)`）、生のNV12バッファにテキストと数値のコンテキスト（マップマッチ、GPS、姿勢）を加えて入力し、`speedLimit`、`heldSignConfidence`、`heldSignLeft/Top/Right/Bottom`などのフィールドを持つ`frogpilotSignReading`をpublishする。確信度の高い標識のクロップは、方位/緯度/経度のJSONとともにJPEGとして保存される(https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/frogpilot/system/speed_limit_reader.py)。`frogpilot/system/speed_limit_reader_models.py`はbearerトークンを用いて`FROGPILOT_API + /v1/speed-limit-reader/<version>/<file>`から新しいモデルをダウンロードする。`SPEED_LIMIT_READER_MODEL_FILES = ("speed_limit_reader.mnn",)`(https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/frogpilot/system/speed_limit_reader_models.py; https://raw.githubusercontent.com/FrogAi/FrogPilot/FrogPilot-Staging/frogpilot/common/frogpilot_variables.py)。
  - 重み：`frogpilot/assets/models/speed_limit_reader.mnn`、12,601,448バイト、リポジトリにコミットされている（クローンのgit ls-tree -l）。アーキテクチャ、学習データ、精度、モデル自体のライセンスは、見つかった範囲ではどこにも文書化されていない（リポジトリはMIT）。openpilot外での実行可能性：困難（openpilot固有の入力を持つMNNグラフ）。
- TeslaとMobileye：TSRについてオープンソース化されたものはない（リポジトリは存在しない。これ以上は確認していない）。

### E4. ライブラリとフレームワークの対応状況

- torchvision：`torchvision.datasets.GTSRB(root, split="train"|"test", download=...)`は`GTSRB/Training`と、`GT-final_test.csv`を伴う`GTSRB/Final_Test/Images`を読み込む。ダウンロード元は`https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/`で、md5は513f3c79a4c5141765e10e952eaa2478 / c7e4e6327067d32654124b0fe9e82185 / fe31e9c9270bbcd7b84b7f21a9d9d9e5である(https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/gtsrb.py)。torchvisionに事前学習済みの標識モデルはない。
- TensorFlow Datasets：GTSRBは`visual_domain_decathlon`の中にのみ現れる（"gtsrb"、"German Traffic Signs"、43クラス、Decathlonの著者らがリサイズした画像、robots.ox.ac.uk / image-net.orgでホスト）(https://raw.githubusercontent.com/tensorflow/datasets/master/tensorflow_datasets/image_classification/visual_domain_decathlon.py)。単独のGTSRBやGTSDBのbuilderはない（クローンのツリーのgrep）。
- timm（huggingface/pytorch-image-models、HEAD 03b76b6 2026-10-30）、keras-io（HEAD 5a1293e 2026-10-28）、mmpretrain（HEAD ee7f2e8 2024-11-01）：GTSRBや道路標識の例も設定もない（クローンのツリーをgtsrb / traffic_sign / tt100k / gtsdbでgrep）。

## F. 組込みおよびエッジでのデプロイ

- コードのある古典的なOpenCVパイプライン：hoanglehaithanh/Traffic-Sign-Detection（292スター、MIT、Dec 2018）：動画に対する色によるセグメンテーションと輪郭ベースの候補抽出、自作データセットの12個のクロップフォルダに対するSVM分類器（`classification.py`、`data_svm.dat`）、密なオプティカルフローによる追跡、OpenCV 3。"A new version using SSD will be released"とあるが、新版は公開されなかった(https://raw.githubusercontent.com/hoanglehaithanh/Traffic-Sign-Detection/master/README.md)。Vicondrus/Roadster（古典的な抽出 + CNN、23スター）とPraveenterax/Traffic-sign-Detection-Recognition-Matlab-RandomForest（13スター、MATLAB）も同様である(https://github.com/topics/traffic-sign-detection)。いずれにも公開ベンチマークでの精度の数値はない。
- NVIDIA Jetson / TensorRT：dusty-nv/jetson-inference（HEAD 45da40a 2025-10-16）には道路標識のモデルも例もない（ツリーのgrep）。MehmetOKUYAR/yolov7_tensorrt_test（YOLOv7 TensorRT、独自データ、MIT）とfreicar-2022-1/freicar_project_sign_detect（ROS YOLOv7）はコミュニティのプロジェクトである(https://github.com/topics/traffic-sign-detection)。NVIDIA TAOやNGCの道路標識モデルは見つからなかった（網羅的には検証していない）。
- Raspberry Pi：GTSDBでのファインチューニングの論文arXiv 2003.03256は、Faster R-CNN Inception v2とTiny-YOLOv2を"on the host PC as well as Raspberry Pi 3 Model B+"で試験したが、コードへのリンクはない(https://arxiv.org/abs/2003.03256)。見つかったTFLiteの選択肢は、frogermcs/GTSRB-TensorFlow-Lite（MobileNet v1 TFLite、Colab）とjack-perlo/*-Gtsrb（Hub上のint8/uint8 .tflite）である。
- FPGA：ZCU102/ZCU104/KV260、VCK190、VCK5000、U50LV/U55C向けのVitis AI 2.5の`tsd_yolox_pt` xmodel（D9）。LaErre9/Zynq_Ultrascale_Vitis_AI_CNN_ZCU102(学生プロジェクト、ZCU102のDPUにデプロイしたLeNet/VGG16のGTSRB分類器、GPL-3.0、21スター、https://raw.githubusercontent.com/LaErre9/Zynq_Ultrascale_Vitis_AI_CNN_ZCU102/master/README.md)。
- 確認したが道路標識に関するものがなかったベンダーのサンプルアプリケーション（2026-10-31にshallow cloneのツリーをtraffic sign / tsd / gtsrb / tt100k / road sign / speed limitでgrep）：STMicroelectronics/stm32ai-modelzoo（HEAD 1423c78 2026-04-21。zooのフォルダは画像分類、物体検出、姿勢、セグメンテーション、音声で、標識のタスクはない）、renesas-rz/rzv_ai_sdk（HEAD ce4aed7 2026-10-08。アプリケーションはQ01〜Q13とR01で、標識向けのものはない）、ARM-software/ML-zoo（HEAD fec0bb5 2023-03-31）、quic/ai-hub-models（HEAD f4758c3 2026-10-31）、TexasInstruments/edgeai-modelzoo（HEAD 99f1bd6 2026-08-12）、openvinotoolkit/open_model_zoo（HEAD 2c9b5c5 2026-10-15）、nxp-imx/nxp-nnstreamer-examples（HEAD 37d3d86 2026-06-12）。NXPのeIQとQualcommのAI HubのWebカタログは取得していない（未検証）。

## G. リーダーボード、調査論文、ベンチマーク

- 稼働中のリーダーボード：INIのページのみ。GTSRBの最終結果（A1）とGTSDBの結果（A2）は2010年代から変わっておらず、どちらのページにもコードへのリンクはない。TT100K、MTSD、DFG、CCTSDBには公開リーダーボードがない（TT100Kのサイトは著者らの結果のみを掲載し、MTSDの論文はテストラベルを非公開とし、DFGの結果は論文中にある）。Papers with Codeはなくなっており、Hugging Faceの論文インデックスには道路標識のタスクリーダーボードがない（https://huggingface.co/api/papers/search?q=TT100K: 0件、"traffic sign detection"のクエリでは指標のない論文が返る）。
- 結果表を含む調査論文：
  - Lim, Lee, Lim, Ong, Alqahtani, Ali、"Recent Advances in Traffic Sign Recognition: Approaches and Datasets"、Sensors 23(10) 4674, 2023, doi 10.3390/s23104674。GTSRB（51,839枚、43クラス）、BTSD（7,095+、62）、TSRD（6,164、58）をレビューしている。GTSRBの精度として、最高99.99%（Li, Song and Wang 2022、色ヒストグラム + HOG + PCA）、99.94%（Shustanov and Yakimov 2017）、99.93%（Mamatkulovich 2022）、99.72%（DeepThin、Haque 2021）を引用している。"Data sharing not applicable"、コードへのリンクなし(https://www.mdpi.com/1424-8220/23/10/4674)。99.9x%のエントリは未検証の自己申告として扱うこと。
  - Pavlitska, Lambing, Zoellner、"Adversarial Attacks on Traffic Sign Recognition: A Survey"、arXiv 2307.08278（2023）：標識の分類器と検出器に対するデジタルおよび物理的な攻撃を調査している(https://arxiv.org/abs/2307.08278)。メンテナの研究に直接関係する。
  - Zhao, Qi, Zhang, Bi et al.、"Road Infrastructure Perception for Autonomous Driving: Taxonomy, Representations, Tasks, and Challenges"、arXiv 2610.33758（2026-10-24）：道路標識認識を"regulatory-semantic infrastructure"の項でベンチマークおよび評価とともに扱っている(https://arxiv.org/abs/2610.33758)。全文は読んでいない。
  - Roman et al.、"One-Stage Object Detectors in Autonomous Driving"、arXiv 2608.19014（2026-08-19）：YOLO/SSD/RetinaNet/FCOS/CenterNet/YOLOv10の調査で、道路標識を含む報告済みのベンチマーク性能を掲載している(https://arxiv.org/abs/2608.19014)。全文は読んでいない。
  - "Review. Machine learning techniques for traffic sign detection"、arXiv 1712.04391（2017）(https://export.arxiv.org/api/query?search_query=all:%22traffic%20sign%22%20AND%20(all:survey%20OR%20all:review))。
  - Wali et al. 2019（Sensors）とSaadna and Behloul 2017はよく引用されるが、取得していない（未検証）。
- プロトコル設計に関係するコード付きのベンチマーク論文：aarcosg 2018（D1、8つの検出器、単一のプロトコル）、Tabernik 2019（D2、最小30 pxでのmAP50とmAP50:95）、MTSD 2020（D4、検出器と分類器を分けた313クラスのmAP）、OPPOのACCV 2022ワークショップのベンチマーク"Evaluating and Bench-marking Object Detection Models for Traffic Sign and Traffic Light Datasets"（信号機の調査のB5節を参照）。

## H1. 一覧表

現時点での実行可能性：容易 = CPU上の現代的なPyTorch/ONNX/Ultralytics。中程度 = compat経由のTF1 frozen graph、Keras 2のHDF5、またはGPU専用だが現代的なもの。困難 = Torch7、Caffe、Caffe2 Detectron、MNN、または入手不能なデータ。スター数と最終コミットは、クローンのHEADの日付が示されていない限りshields.ioによる。

| 名称 | 年 | 手法 | フレームワーク | データセット | 報告された指標 | 重み | ライセンス | スター数 | 最終コミット | 実行可能性 |
|---|---|---|---|---|---|---|---|---|---|---|
| IDSIAのCNNコミッティ（公式コードなし） | 2011 | MCDNN | 内製のGPUコード | GTSRB | 99.46% | なし | 該当なし | 該当なし | 該当なし | 該当なし |
| SermanetのマルチスケールCNN（EBLearn） | 2011 | 2段のConvNet、マルチスケール | EBLearn C++ | GTSRB | 最終98.31%（論文では99.17%） | 不明 | 不明 | 該当なし | 該当なし | 困難 |
| INI LDA on HOG / CAORのランダムフォレスト | 2011 | HOG + LDA / RF | なし | GTSRB | 95.68% / 96.14% | なし | 該当なし | 該当なし | 該当なし | 該当なし |
| Moodstocks/gtsrb.torch | 2015 | IDSIA類似のCNN + 1〜2個のSTN | Torch7 | GTSRB | 98.85 / 99.46 / 99.67% | なし（学習レシピ） | MIT | 177 | May 2016 | 困難 |
| Haloi 2015 | 2015 | STN + inception | なし | GTSRB | 99.81% | なし | 該当なし | 該当なし | 該当なし | 該当なし |
| aarcosg/tsr-torch | 2018 | CNN + 3個のSTN | Torch7 + CUDA | GTSRB | 99.71%（リーダーボード1位） | Google Driveのzip | MIT | 41 | 2018-10-18 | 困難 |
| navoshta/traffic-signs | 2016 | 畳み込み3層 + FC、スキップ接続 | TF 1.x | GTSRB（Udacityのpickle） | テスト99.33% | なし | Apache-2.0 | 148 | 2019-12-23 | 困難 |
| kenshiro-o/CarND-... | 2017 | CNN 3x3 | TF 1.x | GTSRB（Udacity） | テスト97.75% | なし | なし | 145 | 2017-08-26 | 困難 |
| jeremy-shannon/CarND-... | 2017 | Sermanet形式 | TF 0.12 | GTSRB（Udacity） | val 0.987 | リポジトリ内にTFチェックポイント | なし | 59 | 2017-01-16 | 困難 |
| vxy10/p2-TrafficSigns | 2017 | CNN | TF 1.x | GTSRB（Udacity） | 98.8% | なし | なし | 67 | 2017-02-09 | 困難 |
| mohamedameen93/German-Traffic-Sign-... | 2017 | LeNet / VGG形式 | TF 0.12.1 | GTSRB（Udacity） | テスト97.6% | なし | MIT | 73 | Dec 2019 | 困難 |
| hello2all/GTSRB_Keras_STN | 2018 | CNN + STN | Keras 2 | GTSRB（Udacity） | テスト99.59% | リポジトリ内にweights.hdf5（13.9 MB） | MIT | 51 | 2020-02-07 | 中程度 |
| poojahira/gtsrb-pytorch | 2018 | CNN + STN、TTA | PyTorch | GTSRB（NYU Kaggle） | private LBで99.809% | リポジトリ内にmodel_40.pth（3.4 MB） | なし | 60 | 2026-02-19 | 容易 |
| surmenok/GTSRB | 2018 | ResNet-34のファインチューニング | fastai | GTSRB | trainの20%を用いたvalで99.22% | なし | MIT | 128 | Nov 2019 | 中程度 |
| hellojialee/Traffic_Sign_Recognition_Efficient_CNNs | 2018 | 効率的なCNN + MobileNet FRCNN | Keras | GTSRB、GTSDB、TT100K | READMEに記載なし | なし | MIT | 154 | 2020-04-06 | 困難 |
| bazyl/gtsrb-model（HF） | 2022 | ViT-B/16 in21k | transformers | GTSRB（bazylの分割） | eval 0.9993 | あり | apache-2.0 | 該当なし | 2023-05-05 | 容易 |
| kelvinandreas/vit-traffic-sign-GTSRB（HF） | 2024 | ViT-B/16 in21k | transformers | GTSRB | 0.9846 | あり | mit | 該当なし | 2025-01-02 | 容易 |
| aarcosg/traffic-sign-detection | 2018 | Faster R-CNN、R-FCN、SSD、YOLOv2 | TF1 OD API、darkflow | GTSDB（3スーパークラス） | mAP 61.64〜95.77 | Google Drive（8個のzip） | MIT | 335 | 2020-02-13 | 中程度 |
| moabitcoin/signfeld | 2020 | RetinaNet R50、合成データによる学習 | Detectron2 | GTSDB（ドイツの169クラス） | GTSDB mAP 67.23 | GitHub release（318 MB） | MIT | 13 | May 2020 | 中程度 |
| skokec/detectron-traffic-signs | 2019 | Mask R-CNN + OHEM | Caffe2 Detectron | DFG（200クラス） | mAP50 95.5、mAP50:95 84.4 | data.vicos.si（644 MB） | Apache-2.0 | 38 | 2026-04-21 | 困難 |
| TT100K公式（Zhu et al.） | 2016 | マルチタスクFCN + 分類器 | Caffeのフォーク | TT100K（45クラス） | IoU 0.5でacc 0.877 / recall 0.906 | code.zip（467 MBのcaffemodel） | CC BY-NC（データセット）、コードのライセンスは不明 | 該当なし | 2016 | 困難 |
| Vitis AI 2.5 pt_yolox_TT100K | 2022 | YOLOX-mの改変版 | PyTorch（YOLOXのフォーク） + Vitis AI | TT100K val | AP 62.3 float / 62.1 INT8 | AMDのzip（594 MB） | Apache-2.0 | 1.8k（Vitis-AI） | 2.5より後のリリースではモデル削除 | 中程度（GPU） |
| MTSDのベースライン（Mapillary） | 2020 | Faster R-CNN FPN + 分類器 | 不明 | MTSD（313クラス） | 83.4 mAP | なし | 該当なし（データセットは研究用途のみ） | 該当なし | 該当なし | 該当なし |
| Genius-Society/yolov4_tt100k（HF/ModelScope） | 2024 | YOLOv4 + attention | 不明 | CCTSDB2021 | mAP 92.77（自己申告） | ModelScopeのみ | MIT | 該当なし | 2025-10-30 | 不明 |
| HaoqinHong/CCSPNet-Joint | 2024 | YOLOv5l + EfficientViT + ノイズ除去器 | PyTorch | CCTSDB-AUG | 論文に記載 | Kaggle / Baidu | GPL-3.0 | 28 | Jan 2026 | 中程度 |
| cvtechniques/TrafficSignDetection（HF） | 2026 | YOLO11n | Ultralytics | LISA TS | mAP50 0.994（val） | best.pt | mit（カード） | 該当なし | 2026-03-16 | 容易 |
| neitpm/traffic-sign-yolo11-benchmark（HF） | 2026 | YOLO11n + UIB版 | Ultralytics | 独自（5,668/1,259/1,151） | mAP50-95 70.27 | .pt 3個 | mit | 該当なし | 2026-10-16 | 容易 |
| star092304/traffic-sign-detection-vietnam-yolo（HF） | 2026 | YOLO | Ultralytics | ベトナム（HFのデータセット） | mAP50 0.9806 | best.pt/.onnx | apache-2.0 | 該当なし | 2026-06-08 | 容易 |
| Panoramax/detect_face_plate_sign（HF） | 2024 | YOLO11l @2048 | Ultralytics | Panoramaxの画像 | 標識のmAP50 0.898（val 329枚） | yolo11l_panoramax.pt | etalab-2.0 | 該当なし | 2024-11-16 | 容易 |
| Panoramax/classify_de_road_signs（HF） | 2026 | YOLO26-cls | Ultralytics | PanoramaxのDEクロップ（130以上のクラス） | accuracy 0.986 | .pt（26 MB） | etalab-2.0 | 該当なし | 2026-06-13 | 容易 |
| infra-intelligence/road-sign-yolox-tiny（HF） | 2026 | スクラッチから学習したYOLOX-Tiny | PyTorch / ONNX | Mapillaryの日本画像（非公開ラベル） | AP50 0.902、recall 0.857（test） | .pth + .onnx（20 MB） | Apache-2.0 | 該当なし | 2026-10-22 | 容易 |
| FrogPilot speed_limit_reader | 2026 | 不明（MNNグラフ） | MNN | 不明 | 公表なし | リポジトリ内に.mnn（12.6 MB） | MIT（リポジトリ） | 568 | 2026-10-31 | 困難 |
| hoanglehaithanh/Traffic-Sign-Detection | 2017 | 色 + 形状 + SVM | OpenCV 3 | 自作の12クラス | なし | data_svm.dat | MIT | 292 | Dec 2018 | 中程度 |

## H2. 再現候補（順位付き）

制約：2コア、7 GB RAMのCPUのみのクラウドセッションで、Hugging FaceとGitHub releasesからのダウンロードは可能、sid.erda.dkには到達不可（したがって公式のGTSRB/GTSDBアーカイブもtorchvisionのダウンロードも使えない）。各候補について比較対象の数値を示す。

### CPUのみのセッション

1. 公式の12,630枚のテストセットでのpoojahira/gtsrb-pytorch（STN + CNN）。理由：重みがリポジトリ内にあり（3.4 MB）、素のPyTorchで、ラベルはtanganke/gtsrbから入手できる。目標：99.809%（Kaggle private）/ 99.746%（public）。モデルはtrain+valで学習されているため、テスト時のデータ拡張ありで99.7〜99.8の範囲、なしではそれよりやや低い値が見込まれる。
   ```
   python3 -m venv ~/venv-ts && . ~/venv-ts/bin/activate
   pip install torch torchvision datasets pillow numpy
   git clone https://github.com/poojahira/gtsrb-pytorch.git ~/gtsrb-pytorch
   python - <<'EOF'
   from datasets import load_dataset
   ds = load_dataset("tanganke/gtsrb", split="test")     # ラベル付きの12,630枚の画像
   ds.save_to_disk("~/gtsrb_test")
   EOF
   # 30行程度の評価スクリプトを書く：model.py（Net）をimportし、model/model_40.pthを
   # torch.load(map_location="cpu")で読み込み、data.pyと同じtransforms（32x32へのリサイズ、normalize、
   # 必要に応じてTTAのtransforms）を適用し、argmaxをとってlabel列と比較し、精度を報告する。
   ```
   数値を信用する前にクラスインデックスの順序を確認すること。tanganke/gtsrbのlabel列が元の順序のGTSRBクラスID 0〜42を使っているか、またリポジトリの学習用フォルダ（NYUの課題のレイアウトでは`data/train_images/000XX`）が同じIDに対応するかは、本セッションでは検証していない。まず既知のクラス（速度制限30 = クラス1）でいくつかの予測を比較すること。

2. 同じテストセットでのHugging FaceのViT分類器（kelvinandreas/vit-traffic-sign-GTSRB、次いでbazyl/gtsrb-modelとcvetanovskaa/vit-base-patch16-224-in21k-gtsrb-tuned）。理由：`transformers`のpipelineで済み、移植すべきコードがない。コスト：2コアでの224 pxのViT-B/16は1枚あたり約0.2〜0.4秒で、12,630枚で約1〜1.5時間かかる。まず2,000枚のランダムなサブセットで実行すること。目標：0.9846（kelvinandreas、分割の記載なし）、0.9993（bazyl、独自の評価用分割。公式テストセットではより低い値が見込まれる）。
   ```
   pip install transformers datasets torch pillow
   python - <<'EOF'
   from transformers import pipeline; from datasets import load_dataset
   ds = load_dataset("tanganke/gtsrb", split="test")
   clf = pipeline("image-classification", model="kelvinandreas/vit-traffic-sign-GTSRB", device=-1)
   # モデルのid2labelをGTSRBのIDに対応付け（config.jsonを確認）、ループしてargmax == labelの数を数える
   EOF
   ```

3. Panoramax/classified_de_road_signsのval分割でのPanoramax/classify_de_road_signs（YOLO26-cls）。理由：重み（26 MB）と検証用の正確なzipがともにHub上にある。Ultralyticsの`val`でtop-1が直接得られる。ラベル空間（130以上のドイツの標識の種類）はOpenTSRが欧州向けに必要とするものに近い。目標：accuracy 0.986（model-index）。ライセンス：モデルはEtalab-2.0、データはCC BY-SA 4.0。
   ```
   pip install ultralytics huggingface_hub
   hf download Panoramax/classify_de_road_signs classify_de_road_signs.pt --local-dir ~/pano_de
   hf download Panoramax/classified_de_road_signs --repo-type dataset --local-dir ~/pano_de_data
   cd ~/pano_de_data && unzip -q train.zip && unzip -q val.zip   # フォルダ名を確認する。train/とval/を想定
   yolo classify val model=~/pano_de/classify_de_road_signs.pt data=~/pano_de_data device=cpu workers=2
   # metrics/accuracy_top1を0.986と比較する
   ```

4. Udacityのpickle形式テストセットでのhello2all/GTSRB_Keras_STN。理由：重みがリポジトリ内にあり（13.9 MB）、test.pは32x32の12,630枚のセットで、目標は99.59%。コスト：CPUで数分。リスク：カスタムの`SpatialTransformer`層を含むKeras 2のHDF5である。`pip install "tensorflow<2.16" "keras<3"`または`tf_keras`を使い、`custom_objects`を指定して読み込むこと。
   ```
   git clone https://github.com/hello2all/GTSRB_Keras_STN.git ~/stn_keras
   curl -L -o ~/traffic-signs-data.zip https://d17h27t6h515a5.cloudfront.net/topher/2017/February/5898cd6f_traffic-signs-data/traffic-signs-data.zip
   unzip -q ~/traffic-signs-data.zip -d ~/udacity_gtsrb          # train.p valid.p test.p
   pip install "tensorflow-cpu<2.16" "keras<3" numpy
   # test.pyを改変する：test.pをpickle.loadし、conv_model.pyのモデルを構築し、model.load_weights("output/weights.hdf5")を実行して評価する
   ```

5. 日本のドライブレコーダーのフレームでのinfra-intelligence/road-sign-yolox-tiny（定性評価、および自作の小規模テストセット）。理由：唯一のオープンな日本の標識検出器であり、ONNX RuntimeのCPU実行で動き、20 MBである。公開された数値は再現できない（評価画像が非公開）ため、成果物は数百枚の日本のフレーム（例えば埼玉のMapillary画像）での実行と目視確認であり、後にOpenTSR独自のラベル付き日本フレームでAP50を算出し、作成者の0.902（test）/ 0.925（val）と比較する。
   ```
   pip install onnxruntime numpy pillow huggingface_hub
   hf download infra-intelligence/road-sign-yolox-tiny --local-dir ~/jp_yolox
   python ~/jp_yolox/examples/predict_onnx.py ~/jp_yolox/road_sign_yolox_tiny_v9.onnx frame.jpg
   # 16:9のフレームは1280x736にレターボックス化される。しきい値0.5。クラス順はguide, warning, regulatory, indication, supplementary, hard_negative
   ```

6. GTSDBのテストTFRecordsでのaarcosg/traffic-sign-detectionのSSD MobileNet v1とFaster R-CNN ResNet 101。理由：リポジトリのGoogle Drive上のTFRecordsが、到達できないGTSDBアーカイブの代わりになる。目標は3スーパークラスでのIoU 0.5におけるmAP 61.64（SSD MobileNet）と95.08（FRCNN R101）。コスト：テスト画像は300枚。SSDは数秒、1360x800のFaster R-CNN R101は2コアで1枚あたり約5〜10秒。リスク：TF1のfrozen graphには、`tf.compat.v1.GraphDef`でのimportを伴う`tensorflow` 2.xと、ノートブックのVOC APコードの再実装（短い）が必要である。
   ```
   pip install gdown tensorflow-cpu numpy
   gdown 1hKdjTsiFm_vdtZPdto0QJihThGFOPEkq -O ~/gtsdb_tfrecords.zip      # TFRecords
   gdown 1U31RhUvE1Urr5Q92AJynMvl-oFBVRxxg -O ~/ssd_mobilenet_v1.zip     # frozen graphと検出結果を含むモデルzip
   # zipには著者らの検出結果ファイルも含まれるため、推論を行う前にmAPを再計算できる
   ```

7. star092304/traffic-sign-detection-vietnam-yolo。理由：小さなデータセット（n<1K）と重みがHub上にあり、`yolo val`のプロトコルで、目標はmAP50 0.9806 / mAP50-95 0.8357。対象地域外だが、CPU上でのUltralyticsの経路の10分間の動作確認になる。

8. cvtechniques/TrafficSignDetection（YOLO11n、LISA traffic signs）。目標は自身のval分割でmAP50 0.994。UCSDのLISA TSデータセットと作成者の分割が必要だが、これらの所在は確認できなかった（未検証）ため、最下位とする。

### GPUマシン

1. TT100KでのVitis AI 2.5 `pt_yolox_TT100K_640_640_73G_2.5`。理由：floatのチェックポイント、YOLOXのフォーク、評価スクリプトを含む594 MBの公開zipがある。目標は"TT100k val"でAP 62.3（expファイル`exps/example/custom/yolox_m_tt100k_float.py`が分割を定義している。それが2016年のテスト分割か独自の分割かを確認すること。分割のサイズは本調査では検証していない）。データ：cg.cs.tsinghua.edu.cn（直接リンクが機能する）、またはHub上のミラーfinedet/tt100kとGenius-Society/tt100k（CC BY-NC 4.0）からのTT100K `data.zip`（19.2 GB）。手順：
   ```
   curl -L -o pt_yolox_tt100k.zip https://download.amd.com/opendownload/xlnx/pt_yolox_TT100K_640_640_73G_2.5.zip && unzip -q pt_yolox_tt100k.zip
   cd pt_yolox_TT100K_640_640_73G_2.5/code && pip install -r ../requirements.txt && pip install -v -e .
   curl -L -o data.zip https://cg.cs.tsinghua.edu.cn/traffic-sign/data_model_code/data.zip && unzip -q data.zip -d datasets/   # 19 GB
   bash datasets/convert_label.sh
   python tools/eval.py -f exps/example/custom/yolox_m_tt100k_float.py -c ../float/best_ckpt.pth -b 32 -d 1 --conf 0.001
   # APを62.3と比較する
   ```
   データセットのライセンスがCC BY-NCである点に注意すること。ベンチマークには問題ないが、これで学習したモデルをOpenTSRのリリースとすることはできない。

2. DFGでのskokec/detectron-traffic-signs。目標はmAP50 95.5 / mAP50:95 84.4（改良とデータ拡張ありのMask R-CNN R50。READMEは、データセットの更新後は結果が異なる可能性があると警告している）。カスタムopを`modules/detectron`にコピーしたCaffe2のビルド、CUDA、およびhttps://www.vicos.si/Downloads/DFGTSD （ダウンロードページへの301リダイレクト。利用条件は未確認）からのDFGデータセットが必要である。重み：https://data.vicos.si/skokec/villard/dfg_mask_rcnn_weights.tar.gz （644 MB）。工数：数日。3つの改良をDetectron2で再実装する方が、Caffe2をビルドするよりおそらく安上がりである。

3. TT100K公式のCaffeモデル。目標：インスタンス数100以上の45クラスで、IoU 0.5におけるprecision 0.877 / recall 0.906（code/results/report_ours.txt）。手順：code.zip（801 MB）をダウンロードし、`code/caffe`（BVLC Caffeのフォーク、Python 2時代）をビルドし、`model/model.caffemodel`を用いて`code/python/deployer.ipynb`と`eval_check.ipynb`を実行する。`results/ours_result.json`ファイルにより、推論を行う前に評価コードを確認できる。

4. GTSDBでのmoabitcoin/signfeldのRetinaNet。目標はGTSDB mAP 67.23。GPU上のDetectron2なので手順は単純である。重みはGitHub releasesから入手する（318 MB）。GTSDBの画像が必要である（Roboflowの再エクスポートkeremberke/german-traffic-sign-detectionはスモークテストにのみ使うこと。その分割とラベルは300枚の公式テストセットと異なるため）。

5. aarcosg/tsr-torch（99.71%）とMoodstocks/gtsrb.torch（99.67%）：CUDA 8〜10時代のドライバを伴うTorch7のDockerイメージ内でのみ実行できる。事前学習済みのzipはGoogle Drive上にある。優先度は低いが、リーダーボード首位の数値に至る唯一の経路である。

6. 公開されたレシピ（navoshta 99.33%、hello2all 99.59%、poojahira 99.8%）を再現するためのGPU 1台でのスクラッチからの学習は、それぞれ数時間かかり、クリーンなプロトコル（train 39,209 / test 12,630）を持つ現代的なPyTorchのベースラインをOpenTSRにもたらす。最初に推奨するのは、poojahiraのデータ拡張リストを用いた64x64でのtorchvisionのResNet-18またはConvNeXt-Tのファインチューニングで、CPU候補1のスクリプトで評価する。

## 取得できなかったページ

- https://benchmark.ini.rub.de/gtsrb_dataset.html, およびsid.erda.dkのアーカイブディレクトリ(https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/)：サンドボックスからホストに到達できない。したがってGTSRB/GTSDBのファイル一覧とHOG特徴量のファイルは未検証である。
- https://github.com/topics/traffic-sign-detection?o=desc&s=stars （ソート版）とhttps://pmc.ncbi.nlm.nih.gov/articles/PMC10223536/ ：WebFetchの許可が間に合わなかった。代わりにソートなしのトピックページとMDPIのページを使用した。
- https://doaj.org/article/b286c95037b941c6acfbc56c05543989, および他の3件のDOAJレコード（最近の調査論文とYOLOの論文）：Cloudflareのチャレンジページが返された。
- https://universe.roboflow.com/* （curl経由）：HTTP 403。WebFetchで1ページを読んだ。
- Kaggleのノートブックとデータセットのページ（www.kaggle.com/code、www.kaggle.com/datasets）：Web検索で見つからず、取得していない。Kaggle APIには認証情報が必要である。
- aarcosg/traffic-sign-detection、aarcosg/tsr-torchのGoogle Drive上の重みファイル、およびMDhamani（mega.nz）：リンクの存在はREADMEに基づく。ファイルはダウンロードしていない。
- https://www.vicos.si/Downloads/DFGTSD/ （DFGの利用条件ページ）：301リダイレクトのみを確認した。
- https://www.mapillary.com/dataset/trafficsign （MTSDの利用条件）：取得していない。
- NXP eIQ、Qualcomm AI Hub、NVIDIA NGCのWebカタログ：取得していない。GitHubのツリーのみを確認した。
- Papers with Code：オフライン。引用していない。
