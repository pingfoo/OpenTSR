# 道路標識の検出・分類のための公開データセット

> 本文書は英語版 [docs/datasets.md](../datasets.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

2026-10-31に、公開中のWebページをもとにまとめた調査である（claude.aiのリサーチセッション、pingfoo/OpenTSR）。事実に関する記述にはすべて、確認に用いたURLを付す。ページを取得できなかった場合や値が記載されていなかった場合は、その項目を「不明」または「未検証」とする。情報源の間で食い違う数値は両方の値を示す。第7節は要約表、第8節はOpenTSRへの推奨、第9節は取得できなかったページの一覧である（二次情報源に依拠する記述はそこに明記する）。

表記の約束：
- 「NC」＝非商用の制限、「SA」＝継承、「ND」＝改変禁止
- 「種別ラベル付き」＝汎用的な単一の"traffic sign"ボックスではなく、細粒度の標識種別（速度制限50、進入禁止、...）がアノテーションされていること
- 「テストラベル公開」＝公式テスト分割のラベルをダウンロードできること
- 日本の標識カテゴリ：規制（regulatory）、指示（indication/mandatory）、警戒（warning）、案内（guide）、補助（supplementary plates）
- Hugging Faceのメタデータ（ダウンロード数、分割サイズ、ライセンスタグ）は2026-10-31にJSON APIから取得した

早見表（詳細は後述）：

| 必要なもの | 使うもの | 理由 |
|---|---|---|
| 標準的な分類ベンチマークで、再配布可能なもの | GTSRB（43クラス、39,209 / 12,630） | 公式サイトでは"free to use"、テストラベルは2011年から公開、HFミラーが複数ある |
| ライセンス上の問題がないGTSRBの合成版 | Synset Signset Germany（CC BY 4.0、211クラス、105,500画像） | 合成データのみで学習してGTSRBで98.3%に達する |
| ロバスト性 / 劣化のテストセット | CURE-TSR（ZenodoのコピーはCC BY 4.0）、tanganke/gtsrbの劣化分割 | 12種類のチャレンジ×5レベル |
| テストラベルが公開された標準的な小規模検出ベンチマーク | GTSDB（900画像、43クラス） | 完全なアノテーションを2013-07-15に公開 |
| 細粒度の種別を持つ大規模検出セット | MTSD（NC）、TT100K（NC）、DFG（NC-SA）、ZOD（CC BY-SA 4.0、156標識クラス） | 商用利用しやすいライセンスを持つ大規模な細粒度セットはZODのみ |
| 何らかの公開ラベルが付いた日本の標識 | 研究水準の公開データは存在しない（第3節） | Roboflowの小規模なコミュニティセットと、学習データが非公開のApache-2.0検出器が1つあるのみ |
| テンプレートレンダリング用のピクトグラム | Wikimedia CommonsのSVG：ドイツの標識はパブリックドメイン（公的著作物）、日本の標識はパブリックドメイン（著作権法第13条） | 第4節 |

---

## 1. 分類ベンチマーク（クロップ済み標識）

### 1.1 GTSRB、German Traffic Sign Recognition Benchmark（ドイツ）[重要]

- 作成元：Institut fuer Neuroinformatik, Ruhr-Universitaet Bochum。IJCNN 2011のコンペティション。論文：Stallkamp, Schlipsing, Salmen, Igel, "The German Traffic Sign Recognition Benchmark: A multi-class classification competition", IJCNN 2011, pp. 1453-1460 (https://benchmark.ini.rub.de/gtsrb_dataset.html)、Stallkamp et al., "Man vs. computer: Benchmarking machine learning algorithms for traffic sign recognition", Neural Networks 2012 (https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)。サイトには"Please cite this paper when when using or referring to the GTSRB dataset"とある(https://benchmark.ini.rub.de/)。
- 地域：ドイツの道路。2010年3月、10月、11月の日中に、Prosilica GC 1380CHにより1360x1024、25 fpsで撮影した約10時間の映像(https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)。
- 規模：計51,840画像、うちテスト画像12,630、43クラス(https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)。HFミラーのカード(https://huggingface.co/datasets/bazyl/GTSRB/raw/main/README.md)およびHaloi 2015（51,839のうち"39,209 training images"）(https://arxiv.org/pdf/1511.02992)によれば、train 39,209 / test 12,630。公式ページには"More than 50,000 images in total"と"More than 40 classes"としか書かれていない(https://benchmark.ini.rub.de/gtsrb_dataset.html)。
- 画像サイズ：公式ページによれば15x15から250x250ピクセル。画像は正方形とは限らず、標識が中央にあるとも限らない。各画像には標識の周囲に10%（最低5ピクセル）の余白がある(https://benchmark.ini.rub.de/gtsrb_dataset.html)。論文誌版では15x15から222x193とされている(https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)。
- 構築方法：元素材は70クラス、2,416個の標識インスタンスを写した約133,000枚のラベル付き画像。物理的な標識1つにつき30画像のトラックを作り、30画像未満のトラックと9トラック未満のクラスは除外した(https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)。テスト画像は"consecutively numbered and shuffled to prevent deduction of class membership"である(https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)。
- 形式：PPM（P6）画像。Filename, Width, Height, ROI.x1, ROI.y1, ROI.x2, ROI.y2, ClassIdを列とするセミコロン区切りCSV。クラスごとに1ディレクトリで、GT-<ClassID>.csvを含む。事前計算済みのHOG（1568/1568/2916次元）、Haar（11,584）、色相ヒストグラムの特徴量もある(https://benchmark.ini.rub.de/gtsrb_dataset.html)。
- 分割とその経緯：コンペティション開始2010-12-01、オンラインテストセット公開2011-01-19、オンラインテストセットのクラスID公開2011-01-25、最終学習セット公開2011-07-12、最終テストセットを正解データ付きで公開2011-08-23、投稿サイトは2020年7月に終了(https://benchmark.ini.rub.de/gtsrb_news.html)。Phase I（オンライン競技）の学習セットは26,640画像、オンラインテストセットは12,569画像であった(https://www.arxiv.org/pdf/1202.2745)。26,640 + 12,569 = 39,209であり、これは最終学習セットの規模に一致する。
- ライセンス（原文どおり引用）："The data is free to use. However, we cordially ask you to cite the following publication if you do."（データは自由に使用できるが、使用する場合は次の文献の引用を求める）。正式なライセンス名は示されていない(https://benchmark.ini.rub.de/gtsrb_dataset.html)。結果分析アプリケーションtsr-analysis-src.zipはGPLv2である（同ページ）。2011年の論文募集には"The GTSRB data is freely available from http://benchmark.ini.rub.de"とある(https://benchmark.ini.rub.de/)。
- ホスティング：2019-05-08、INIのサイトは"temporarily shut down the download section"せざるを得なくなったこと、および"The datasets have been transferred to an external hosting service and can be downloaded from there"であることを告知した(https://benchmark.ini.rub.de/)。外部ホストはsid.erda.dkである。torchvisionはbase_url "https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/"からGTSRB-Training_fixed.zip（md5 513f3c79a4c5141765e10e952eaa2478）、GTSRB_Final_Test_Images.zip（md5 c7e4e6327067d32654124b0fe9e82185）、GTSRB_Final_Test_GT.zip（md5 fe31e9c9270bbcd7b84b7f21a9d9d9e5）をダウンロードし、テスト分割にはGT-final_test.csvを読む(https://raw.githubusercontent.com/pytorch/vision/main/torchvision/datasets/gtsrb.py)。sid.erda.dkのアーカイブページ自体は本セッションでは取得できなかった（第9節）。
- torchvision：`torchvision.datasets.GTSRB(root, split="train"|"test", download=...)`。ドキュメントページは出典としてベンチマークのWebサイトのみを挙げ、ライセンスは記載していない(https://docs.pytorch.org/vision/0.26/generated/torchvision.datasets.GTSRB.html)。
- Hugging Faceミラー（2026-10-31にAPIで取得）：
  - bazyl/GTSRB：parquet形式のtrain/test、ダウンロード数2,295、最終更新2022-10-25、ライセンスタグ"gpl-3.0"（INIの文言ではなくアップロード者が選んだタグ）。カードには"Categories: 42, Train: 39209, Test: 12630"とあり（クラス数42はカードの誤りで、データのクラスIDは0..42）、Kaggleのコピーkaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-signへリンクしている(https://huggingface.co/api/datasets/bazyl/GTSRB?full=true, https://huggingface.co/datasets/bazyl/GTSRB/raw/main/README.md)。
  - tanganke/gtsrb：ダウンロード数16,418、ライセンスタグなし、名前付きの43クラス。分割はtrain 26,640 / test 12,630に加え、テストセットを劣化させた7種のコピー（contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter、各12,630）。download_size 1,027,074,522バイト(https://huggingface.co/api/datasets/tanganke/gtsrb?full=true, https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md)。26,640画像のtrain分割は公式の39,209と異なり、Ciresanらが報告したPhase Iの学習セットの規模（学習26,640 / オンラインテスト12,569）と一致する(https://www.arxiv.org/pdf/1202.2745)。したがって、これで学習したモデルは、公式分割で学習したモデルより学習画像が約32%少ない。カードには分割の説明がない。
  - clip-benchmark/wds_gtsrb：WebDataset形式のtarシャード（train 6シャード / test 3シャード）、43行のclassnames.txt、zeroshot_classification_templates.txt、READMEなし、ライセンスタグなし、ダウンロード数567(https://huggingface.co/api/datasets/clip-benchmark/wds_gtsrb?full=true, https://huggingface.co/datasets/clip-benchmark/wds_gtsrb/raw/main/classnames.txt)。
  - APIの検索で見つかったその他のミラー：Zzyy2000/GTSRB、dpdl-benchmark/gtsrb、ilee0022/GTSRB、SomeBottle/GTSRB_224x224_balanced、mteb/wds_gtsrb、mteb/GTSRBZeroShot（ライセンスタグ"unknown"、test 12,630）、zjl123321/GTSRB、pymlstudio/gtsrb-cleaned(https://huggingface.co/api/datasets?search=gtsrb&limit=100)。
- その他のミラー：Zenodoレコード13741936 "GTSRB ... by Real-Time Computer Vision at Ruhr-Universitaet Bochum"。アップロード者はMehran, Kimovski, Najafabadi Samani, Nikolov, Prodan（INIの著者ではない）、data.zip 320.5 MB、ライセンスCC BY 4.0、DOI 10.5281/zenodo.13741936。ページは公式であるとは主張していない(https://zenodo.org/record/13741936)。Zenodo上の派生セット：GTSRB-Conceptsの追加ラベルとオントロジー（CC BY-NC-SA 4.0、レコード16326403）、および3種のバックドアパターンを加えたGTSRB（CC BY 4.0、レコード3716766、ファイルGTSRB_backdoor_green_1.zipなど）(https://zenodo.org/api/records?q=%22traffic%20sign%22%20OR%20%22traffic%20signs%22%20OR%20%22road%20signs%22&size=25&page=1&type=dataset, https://zenodo.org/api/records/3716766)。
- ベースラインと最高性能の推移（特記しない限り、すべて12,630画像の最終テストセットでの値）：
  - IJCNN 2011 Phase I（オンラインテストセット）：INI-RTCVによる人間の性能99.81%、IDSIAのCNN + HOG3コミッティ98.98%、sermanet 98.97%、VISICSのIK-SVM + PHOG + HOG2 97.88%、noobのHOG + LDA + VQ 96.87%、INIのベースラインHOG + LDA 96.32 / 94.73 / 94.51%、HOG + k-NN 72.81から73.89%(https://www.ini.rub.de/upload/file/1470692848_f03494010c16c36bab9e/StallkampEtAl_GTSRB_IJCNN2011.pdf)。
  - 最終ラウンド（2011-08-03時点の順位）：IDSIAのCNNコミッティ99.46%、人間の性能98.84%、sermanetのマルチスケールCNN 98.31%、CAORのランダムフォレスト96.14%(https://benchmark.ini.rub.de/gtsrb_news.html)。論文誌の表には、個人として最良の人間99.22%、HOG 2 / HOG 1 / HOG 3上のLDA 95.68 / 93.18 / 92.34%が加わり、サブセットごとの結果もある（コミッティでは速度制限99.47%、危険99.07%など）(https://www.ini.rub.de/upload/file/1470692859_c57fac98ca9d02ac701c/stallkampetal_gtsrb_nn_si2012.pdf)。
  - Ciresan, Meier, Schmidhuber：25カラムのMCDNN、入力48x48、誤り率0.54% = 99.46%、4基のGPUで37時間(https://www.arxiv.org/pdf/1202.2745)。
  - Sermanet and LeCun, IJCNN 2011：Phase Iで98.97%、のちに容量を増やしグレースケール入力にして99.17%(https://research.google/pubs/traffic-sign-recognition-with-multi-scale-convolutional-networks/)。
  - Haloi 2015（arXiv 1511.02992）：spatial transformer + 改良版inception、top-1 99.81%、パラメータ数約10.5 M。表では99.46（コミッティ）、98.84（人間）、98.31（マルチスケールCNN）、98.14（pLSA）、96.14（ランダムフォレスト）と比較している(https://arxiv.org/pdf/1511.02992)。
  - Arcos-Garcia, Alvarez-Garcia, Soria-Morillo, Neural Networks 99:158-165, 2018：spatial transformerを用いたCNNで99.71%(https://idus.us.es/handle/11441/80679)。公式結果ページでは"DeepKnowledge Seville, CNN with 3 Spatial Transformers, 99.71%"として首位に掲載され、IDSIA 99.46%、COSFIRE 98.97%、人間98.84%を上回る(https://benchmark.ini.rub.de/gtsrb_results.html)。コード： https://github.com/aarcosg/tsr-torch （検索結果で見たリンク、未取得）。
  - その後：局所性モジュールを持つCNN/transformerハイブリッドがGTSRBで99.66%、Persian Traffic Sign Datasetで99.8%(https://arxiv.org/abs/2311.06651)。GTSRBとBelgiumTSを扱うEATFormer風のvision transformerの論文もあるが、アブストラクトに数値はない(https://arxiv.org/abs/2404.19066)。文献には99.8%を超える主張もある（本調査で検証できた最高値はHaloiの99.81%）。比較表を含む2025年のPMC論文は取得できなかった（第9節）。
- 既知の問題：テストセットは学習データと同じ2010年の映像から取られている（同じカメラ、同じ国、日中のみ）。画像は強いモーションブラーと低コントラストを伴う非常に小さなPPMクロップである。公式サイトの"free to use"はOSIやCCのライセンスではないため、厳密な再配布の権利は暗黙的である。HFミラーのライセンスタグは互いに矛盾している（gpl-3.0、なし、unknown）。広く使われているtangankeミラーのtrain分割は非公式のものである。

### 1.2 BelgiumTS分類、BTSC（ベルギー）

- 作成元：Radu Timofte（ETH Zurich、以前はKU Leuven VISICS）。2014-02-18以降ETHでホストされており、旧KU Leuvenのアドレスはリンク切れである(https://btsd.ethz.ch/shareddata/)。引用すべき論文：Timofte, Zimmermann, Van Gool, MVA 2011（BelgiumTSを導入）、Timofte and Van Gool, BMVC 2011（BTSCを導入）、Mathias, Timofte, Benenson, Van Gool, IJCNN 2013（BTSDとBTSCの分割）（同ページ）。データはGeoAutomationが屋根に8台のカメラを載せたバンで取得した(https://btsd.ethz.ch/shareddata/, https://arxiv.org/pdf/1712.02463)。
- 地域：ベルギーのフランダース地方(https://hyper.ai/en/datasets/15948)。
- 規模：BTSCのクロップセットはBelgiumTSC_Training.zip（171.3 MB）とBelgiumTSC_Testing.zip（76.5 MB）。reducedSetTS.txtには"62 used traffic signs types"とある(https://btsd.ethz.ch/shareddata/, https://btsd.ethz.ch/shareddata/BelgiumTS/reducedSetTS.txt)。画像数：Total Recall論文によれば学習4,575 / テスト2,520 / 計7,095、サイズ11x10から562x438、62クラス(https://arxiv.org/pdf/1808.10524)。よく引用される4,591 / 2,534の分割は、本セッションでは一次情報源で検証できなかった（people.ee.ethz.chのIJCNN 2013のPDFリンクはHTMLページを返した。第9節）。
- BelgiumTSの全素材：8台のカメラ分のアノテーションtar（camera00からcamera07）、"provided for detection evaluation"とされるBelgiumTSD_annotations.zip、背景画像セット、姿勢とキャリブレーション付きの4つのマルチカメラテストシーケンス（3,001 / 6,001 / 2,001 / 4,001フレーム×8カメラ）(https://btsd.ethz.ch/shareddata/)。DefinedTSには210種の公式標識パターンがある(https://btsd.ethz.ch/shareddata/BelgiumTS/structure.txt)。
- ライセンス：ライセンス文はなく、"If you use the material from this dataset, please cite one of the following publications"とあるのみ(https://btsd.ethz.ch/shareddata/)。再配布の権利は不明確である。
- 入手方法：直接ダウンロード、登録不要(https://btsd.ethz.ch/shareddata/)。
- ベースライン：top-1 99.17%（Total Recall, 2018）、比較対象はHOG 98.34%、OneCNN 98.17%(https://arxiv.org/pdf/1808.10524)。

### 1.3 rMASTIF / FER-MASTIF TS2010（クロアチア）

- 作成元：MASTIFプロジェクト、FER University of Zagreb（Segvic, Brkic, Kalafatic et al.）。3つのデータセットTS2009（約6,000標識、クロップ画像のみ）、TS2010（約3,000標識、全体画像 + .seqアノテーション + アノテーション付き動画）、TS2011（約1,000標識）がある。"Each sign is annotated 4-5 times at different distances from the car"とあり、TS2009はインターレースである(https://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml)。
- ライセンス（引用）："The datasets may be freely used in academic research."(https://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml)。学術以外の利用は対象外である。
- 分類用の派生版「rMASTIF」のクラス数と正確な分割サイズ：未検証（CCVW 2014の論文ページには数値がない）(https://www.fer.unizg.hr/crv/ccvw.2014.0016, https://www.bib.irb.hr/857781)。

### 1.4 TSRD、Chinese Traffic Sign Recognition Database（中国）

- 作成元：CAS Institute of Automation（NLPR）の交通データのページ。公式ページ http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html は本セッションではタイムアウトした（第9節）。
- HFミラーによる規模：学習画像約4,170、テスト画像約1,994、約58クラス、アノテーション形式"filename;width;height;x1;y1;x2;y2;class"(https://huggingface.co/datasets/EvanDu/Traffic_Sign_Recogntion_Database/raw/main/README.md)。ミラーにはアップロード者が選んだ"mit"ライセンスタグが付いている(https://huggingface.co/api/datasets/EvanDu/Traffic_Sign_Recogntion_Database?full=true)。公式のライセンス：未検証。

### 1.5 CURE-TSR、Challenging Unreal and Real Environments for Traffic Sign Recognition（ベルギーの実画像 + Unreal Engineによる合成）[重要]

- 作成元：OLIVES Lab, Georgia Tech（Temel, Kwon, Prabhushankar, AlRegib）、NeurIPS 2017のworkshop on ML for ITS(https://arxiv.org/pdf/1712.02463, https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md)。
- 内容：CURE-TSD動画データセット（"around 1.7 million real-world and simulator images with more than 2 million traffic sign instances"）から切り出したクロップ。"around 2.2 million traffic sign images in the CURE-TSR dataset"とある。実シーケンスはBelgiumTSの動画、合成（"unreal"）シーケンスはUnreal Engine 4に由来し、チャレンジ効果はAdobe After Effectsで付加した(https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md)。論文の表：2,206,106画像(https://arxiv.org/pdf/1712.02463)。
- クラス：14の標識種別（speed_limit, goods_vehicles, no_overtaking, no_stopping, no_parking, stop, bicycle, hump, no_left, no_right, priority_to, no_entry, yield, parking）(https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md)。
- チャレンジ種別（12）×レベル（5）：脱色、レンズぼけ、コーデックエラー、暗化、レンズの汚れ、露出、ガウスぼかし、ノイズ、雨、影、雪、霞。ファイル名は"sequenceType_signType_challengeType_challengeLevel_Index.bmp"で、sequenceTypeは01が実画像、02が合成（unreal）である（同README）。標識クロップは約3x7から206x277ピクセルである(https://arxiv.org/pdf/1712.02463)。
- 論文で用いた分割：49の動画シーケンス、標識単位で約70/30のtrain/test。チャレンジなしの実画像による学習画像7,292、チャレンジ種別とレベルごとにテスト画像3,334（計200,040）(https://arxiv.org/pdf/1712.02463)。
- ベースライン：7手法（輝度、色、HoGの各特徴量にsoftmaxまたはRBF-SVMを組み合わせたものと、浅いCNN）。最も影響の大きいチャレンジはコーデックエラーである（レベル1で少なくとも30%低下）。合成のチャレンジ画像を加えると、手法とチャレンジの組合せ84のうち72で精度が向上した(https://arxiv.org/pdf/1712.02463)。
- ライセンスと入手方法（3通りの答えがある）：GitHubのREADMEでは、Googleフォームに記入して"and agree the conditions of use"とされる(https://raw.githubusercontent.com/olivesgatech/CURE-TSR/master/README.md)。リポジトリのコードはMIT、スター22、最終コミット2021年9月(https://img.shields.io/github/license/olivesgatech/CURE-TSR.json, https://img.shields.io/github/stars/olivesgatech/CURE-TSR.json, https://img.shields.io/github/last-commit/olivesgatech/CURE-TSR.json)。IEEE DataPort：Open Accessとして掲載、4.6 GBのzip、IEEEアカウントでのログインが必要、DOI 10.21227/n4xw-cg56、ライセンスの記載なし(https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition)。同じ4名の著者によるZenodoレコード3903066：CURE-TSR.zip 4,944 MB、ライセンスcc-by-4.0、DOI 10.5281/zenodo.3903066、公開2020-06-22(https://zenodo.org/api/records/3903066)。OpenTSRが依拠できるのはZenodoのCC BY 4.0のコピーである。
- 既知の問題："real"画像はBelgiumTSのクロップなので、BTSCとCURE-TSRは独立ではない。粗い14クラスのみである（速度制限はすべて1クラスにまとめられている）。bmpファイルで4.9 GBある。

### 1.6 ETSD、European Traffic Sign Dataset（ベルギー、クロアチア、フランス、ドイツ、オランダ、スウェーデン）

- 作成元：Serna and Ruichek, "Classification of Traffic Signs: The European Dataset", IEEE Access 2018, DOI 10.1109/ACCESS.2018.2884826 (https://raw.githubusercontent.com/citlag/European-Traffic-Sings/master/README.md, https://citlag.github.io/publication/etsd)。
- 規模："more than 80 000 images divided in 164 classes"で、ウィーン条約の4カテゴリに分かれる。6か国の公開データセットを集め、UTBMの車両によりBelfortで撮影したフランスの標識を加えたもの(https://raw.githubusercontent.com/citlag/European-Traffic-Sings/master/README.md)。
- ライセンス / 入手方法（引用）："The European Traffic Sign Dataset is free for academic purposes and available upon request"で、Googleフォームから申請する（同README）。GitHubリポジトリ（コードと重み）はApache-2.0、スター15(https://img.shields.io/github/license/citlag/European-Traffic-Sings.json, https://img.shields.io/github/stars/citlag/European-Traffic-Sings.json)。再配布不可。

### 1.7 LISA Traffic Sign Dataset（米国）

- 作成元：UC San DiegoのLISA lab（Mogelmose, Trivedi, Moeslund, IEEE T-ITS 13(4):1484-1497, 2012）(https://cvrr.ucsd.edu/lisa-traffic-signs-dataset)。
- 規模：6,610フレーム、7,855アノテーション、米国の標識種別47、標識サイズ6x6から167x168 px、画像サイズ640x480から1024x522、一部はカラー、一部はグレースケール。アノテーション：標識種別、位置、サイズ、遮蔽の有無、側道上か否か(同ページ。6,610 / 47 / 7,855という値はMTSD論文の比較表にもある https://web3.arxiv.org/pdf/1909.04422v2)。
- ライセンス（引用）："released under academic license agreement"(https://cvrr.ucsd.edu/lisa-traffic-signs-dataset)。ダウンロードページ https://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html には到達できなかった（第9節）。なお、LISAは全体フレームを持つ検出セットであるが、クロップによる分類にもよく使われるためここに挙げる。Zenodoにはcc-by-nc-1.0の"Augmented Lisa Traffic Sign Dataset for Rural Areas"もある（レコード17297772）(https://zenodo.org/api/records?q=%22traffic%20sign%22%20OR%20%22traffic%20signs%22%20OR%20%22road%20signs%22&size=25&page=1&type=dataset)。

### 1.8 地域別の分類セットと小規模セット（各1、2行）

- オーストリア、ATSD（Austrian Highway Traffic Sign Data Set）、Data (MDPI) 8(1):16, 2023：2014年に撮影した1596x1196のシーン画像7,454枚、108クラスで27,521アノテーション。ATSD-Signsは20,683パッチで、分類器が用いるのは60クラス。公開されているのはtrain 4,068 / test 1,443シーンで、ほかにダウンロードできない内部画像1,943枚がある。YOLOv4-tinyでmAP 85.39%、分類器で97.61%。ライセンスは"Permissive license that allows to use the data set free of charge"と引用され、条件はcontentportal.asfinag.atにある（条件のPDFは未取得、第9節）。DOI https://doi.org/10.53177/ATSD は https://contentportal.asfinag.at/ATSD/v1_0 に解決される(https://www.mdpi.com/2306-5729/8/1/16, DOIのcurl)。
- 英国、"English roadsigns"（Zenodo 7831751、CC BY 4.0）：マンチェスター周辺で撮影したスマートフォン動画213本から抽出した256x256のクロップ55,000枚、31クラス、4.83 GB(https://zenodo.org/api/records/7831751)。HF上のUK_Traffic_Sign_Inspection_Dataset（MITタグ、10Kから100K行、READMEは本文なし）(https://huggingface.co/api/datasets/PastaEvangelists/UK_Traffic_Sign_Inspection_Dataset?full=true)。
- フランス、Panoramax/classified_fr_road_signs（HF、ライセンスetalab-2.0）：Panoramaxのストリートレベル画像から切り出した"66000+ detected road signs"。フランスの公式種別コード（Axxは危険、Bxxは制限、Cxxは情報 ...）で命名された"250+ classes"があり、さらに誤検出用の偽クラスがある(https://huggingface.co/datasets/Panoramax/classified_fr_road_signs/raw/main/README.md, https://huggingface.co/api/datasets/Panoramax/classified_fr_road_signs?full=true)。Panoramax/fr_road_sign_subsign（etalab-2.0）とclassified_nl / classified_beもある(https://huggingface.co/api/datasets?search=road%20sign&limit=100)。
- ドイツ、Panoramax/classified_de_road_signs（HF、CC BY-SA 4.0）："20000+ photos of DE road signs in 130+ classes"。クロップはPanoramaxの検出器で作成し、一部のクラスは他の欧州諸国の標識で補っている（ファイル接頭辞BE_、FR_）(https://huggingface.co/datasets/Panoramax/classified_de_road_signs/raw/main/README.md, https://huggingface.co/api/datasets/Panoramax/classified_de_road_signs?full=true)。
- ドイツ、German Traffic Sign Anomaly Detection Dataset（Zenodo 21513624、CC BY 4.0、2026-07-24）：シュレースヴィヒ＝ホルシュタイン州で撮影した71標識カテゴリの実スマートフォン写真369枚に、二値の異常ラベル（汚れ、退色、ステッカー、落書き、植生）を付けたもの。ほかにtrainval_synthetic.zip（3,051 MB）とtemplates.zipがある(https://zenodo.org/api/records/21513624)。ロバスト性や敵対的ステッカーの研究に関係する。
- インド：kannanwisen/Indian-Traffic-Sign-Classification（HF、CC BY 4.0、1Kから10K画像）とDataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset（ライセンスタグ"other"、1K未満のファイル）(https://huggingface.co/api/datasets/kannanwisen/Indian-Traffic-Sign-Classification?full=true, https://huggingface.co/api/datasets/Dataclusterlabspvtltd/Indian_Traffic_Sign_Image_Dataset?full=true)。Indian Signboard Image Dataset（Zenodo 14864212、CC BY 4.0）（前掲のZenodo検索）。
- ペルシャ/イラン：arXiv 2311.06651でPersian Traffic Sign Datasetがベンチマークとして使われている（99.8%と報告）(https://arxiv.org/abs/2311.06651)。入手方法とライセンスは未検証。
- アラビア文字圏の国：HF上のモロッコの道路標識データセット（afyfbadr/moroccan-traffic-sign-dataset、12,869ファイル、ライセンスタグなし）(https://huggingface.co/api/datasets/afyfbadr/moroccan-traffic-sign-dataset?full=true)。IEEE DataPortの"Moroccan Traffic Sign Recognition"と"TUROS-TS: a Tunisian road scene videos traffic sign dataset"（検索結果でタイトルを見たのみ、未取得）。
- ベトナム：VQA派生セットのカードによれば、Kaggleのmaitam/vietnamese-traffic-signsはCC BY-SA 4.0である(https://huggingface.co/datasets/Anakonkai/vietnamese-traffic-sign-vqa/raw/main/README.md)。HFの検出セット：thanhhiepvos/vietnam_traffic_sign（cc-by-sa-3.0タグ、6,437ファイル）、star092304/Traffic-sign-detection-VietNam（cc-by-4.0タグ）、ghbihuy/vietnam_traffic_sign（mitタグ、250行）（HF API）。
- トルコ：本調査で実施したHF APIの検索やWeb検索ではトルコのセットは見つからなかった。未検証。
- ロシア：Dognellaf/russian-road-signs（HF、CC BY 4.0）：YOLO形式で手作業によりラベル付けした43,851枚のJPEGフレーム、89クラス（標識と信号機）(https://huggingface.co/datasets/Dognellaf/russian-road-signs/raw/main/README.md)。第2節のRTSDも参照。
- マレーシア：Extended Malaysian Traffic Sign Dataset、EMTD（Zenodo 1217105、CC BY 4.0）：1,413画像、66カテゴリ、7.47 GB(https://zenodo.org/api/records/1217105)。
- ブラジル：BR-RTSD40k（Zenodo 23025922、CC BY-SA 4.0、11 GB、2026-09-30）、詳細は未調査。HumynLabs/Brazilian_Road_Signs_Dataset（HF、CC BY 4.0、1K未満のファイル）（前掲のZenodo検索、HF API）。
- 韓国：HumynLabs/Korean-Signs-Image-Dataset（HF、CC BY 4.0、1K未満のファイル）（HF API）。Roboflowのpicmin/koreanroadsign、404画像、CC BY 4.0、2025-02-19(https://universe.roboflow.com/picmin/koreanroadsign)。より大規模な韓国のセットは第2節にある。
- カナダ：RDLTechworks/quebec-traffic-signs（HF、CC BY-NC-ND 4.0、1K未満）（HF API）。CanDOR-99は第2節。
- リトアニア、インドネシア、カンボジア：Ramuneri/lithuanian-road-signs（ライセンスなし）、ilmagita/indonesia-traffic-signs、Mongkolek10/cambodia-traffic-sign（ごく小規模）(https://huggingface.co/api/datasets?search=traffic%20sign&limit=100)。
- 国をまたぐもの：guoyang1/Cross-Regional_Traffic_Sign_Dataset（HF、MITタグ、2ファイル、未調査）。

---

## 2. 検出ベンチマーク（バウンディングボックス付きの全体フレーム）

### 2.1 GTSDB、German Traffic Sign Detection Benchmark（ドイツ）[重要]

- 作成元：INI Bochum、IJCNN 2013のコンペティション。論文：Houben, Stallkamp, Salmen, Schlipsing, Igel, "Detection of Traffic Signs in Real-World Images: The German Traffic Sign Detection Benchmark", IJCNN 2013 (https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf)。引用はサイトの2013年の告知に従う(https://benchmark.ini.rub.de/)。
- 地域と撮影：ドイツのボーフム近郊、2010年の春と秋、日中と夕暮れ、市街地、地方部、高速道路。Prosilica GC 1380CHを用い、1360x1024を1360x800に切り詰めた(https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf)。
- 規模：900画像（学習600、評価300）、PPM、1画像あたり0から6標識、標識サイズ16x16から128x128(https://benchmark.ini.rub.de/gtsdb_dataset.html)。標識数：論文によれば1,206（学習846 + 評価360）(https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf)。Dataset Ninjaが公開済みの正解データを解析した結果では852ボックス（prohibitory 396、other 186、danger 156、mandatory 114。394画像はアノテーションなし）(https://datasetninja.com/gtsdb)、MTSD論文の比較表でも852である(https://web3.arxiv.org/pdf/1909.04422v2)。2つの数値は一致しない。OpenTSRにとって重要なのは公開されているgt.txtである。
- カテゴリ：コンペティションではprohibitive、danger、mandatoryの3カテゴリを採点し、それ以外の標識は"may appear in images but aren't scored"とされた(https://benchmark.ini.rub.de/gtsdb_dataset.html)。論文は4つ目のカテゴリ"other"を挙げている(https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf)。正解データはGTSRBの43クラスIDを用いる（Dataset Ninjaには43クラスが掲載されている。例：keep right、priority road、speed limit 50 ...）(https://datasetninja.com/gtsdb)。
- アノテーション形式：セミコロン区切りのCSV "filename; left; top; right; bottom; classID"(https://benchmark.ini.rub.de/gtsdb_dataset.html)。
- テストの正解データ：学習データ公開2012-12-01、テストセット公開2013-02-18、2013-07-15に"We put together the test and training dataset with full annotations for download"(https://benchmark.ini.rub.de/gtsdb_news.html, https://datasetninja.com/gtsdb)。sid.erda.dk上の元ファイル：FullIJCNN2013.zip、TrainIJCNN2013.zip、TestIJCNN2013.zip、gt.txt(https://datasetninja.com/gtsdb)。sid.erda.dkのページ自体は取得できなかった（第9節）。
- ライセンス：GTSDBのページにライセンス文はない(https://benchmark.ini.rub.de/gtsdb_dataset.html)。Dataset Ninjaは"custom"と分類している(https://datasetninja.com/gtsdb)。条件といえるのは、データを自由に利用できるというサイト全体の記述と、引用の依頼（第1.1節）のみである。
- コンペティション結果（precision-recallのAUC）：wgy@HIT501 prohibitive 100% / danger 99.91% / mandatory 100%、visics 100 / 100 / 96.98%、LITS1 100 / 98.85 / 92%、BolognaCVLab 99.98 / 98.72 / 95.76%、ベースラインはViola-Jones 90.81 / 46.26 / 44.87%、HOG + LDA 70.33 / 35.94 / 12.01%。18チーム、110件超の投稿があった(https://www.ini.rub.de/upload/file/1470692848_e7f65a6c27d4c0ec0c71/HoubenEtAl_GTSDB_IJCNN2013.pdf)。
- Hugging Faceミラー：GlowBond/GTSDB（archive.zipが1つ、READMEは1行、ライセンスタグなし、作成2026-03-18、ダウンロード数276）(https://huggingface.co/api/datasets/GlowBond/GTSDB?full=true, https://huggingface.co/datasets/GlowBond/GTSDB/raw/main/README.md)。keremberke/german-traffic-sign-detection（COCO形式の545画像をRoboflowからエクスポートしたもの、train 383 / valid 108 / test 54、43ラベル、カードのライセンス"CC BY 4.0"はRoboflowのアップロード者Mohamed Traoreが付与、作成2023-01-16、ダウンロード数644）(https://huggingface.co/datasets/keremberke/german-traffic-sign-detection/raw/main/README.md, https://huggingface.co/api/datasets/keremberke/german-traffic-sign-detection?full=true)。128x128にリサイズした版がZenodoにCC BY 4.0で置かれている（レコード3941507）（前掲のZenodo検索）。これらのCCタグはいずれもINIが付与したものではない。
- 既知の問題：非常に小さい（900画像、852または1,206標識）。300画像のテストセットは、43クラスのクラス別検出指標を求めるには小さすぎる。元映像はGTSRBと同じである。

### 2.2 Tsinghua-Tencent 100K、TT100K（中国）[重要]

- 作成元：Zhu, Liang, Zhang, Huang, Li, Hu, "Traffic-Sign Detection and Classification in the Wild", CVPR 2016。Tsinghua Universityのグラフィックスグループが、Tencent Street Viewと共同で作成した(https://cg.cs.tsinghua.edu.cn/traffic-sign/)。
- 2016年版：Tencent Street Viewの8192x2048パノラマから切り出した2048x2048画像100,000枚。"Of these, 10000 contain 30000 traffic-signs in total"とある。各標識にクラスラベル、バウンディングボックス、ピクセルマスク（楕円/多角形）が付く。インスタンスが100未満のクラスは学習で無視され、45クラスが残る(https://cg.cs.tsinghua.edu.cn/traffic-sign/0682.pdf)。アノテーションファイルには221カテゴリが列挙されている(https://docs.ultralytics.com/datasets/detect/tt100k, https://web3.arxiv.org/pdf/1909.04422v2 の表1：221クラス、26,349標識)。分割：train 6,105 / test 3,071 / "other" 7,641画像（Ultralyticsは"other"をvalに割り当てる）。一般に再配布されている版でアノテーション付きの画像は16,817(https://docs.ultralytics.com/datasets/detect/tt100k)。残りの約82,000枚の「標識なし」画像は、それぞれ16.8から17.9 GBの追加パート5つとして配布される（16,555 / 16,638 / 16,582 / 16,632 / 15,690画像）(https://cg.cs.tsinghua.edu.cn/traffic-sign/)。
- 2021年版："Tsinghua-Tencent 100K Annotations 2021 (with more classification)"、ファイルtt100k_2021.zip。サイトには"The 2021 version of the Tsinghua-Tencent 100K Annotation is a complement to the 2016 version. The 2021 version could NOT be used independently. It should be merged into the 2016 version, if more data is required."とある（2021年版は2016年版の補完であり単独では使えず、2016年版に統合して使う）(https://cg.cs.tsinghua.edu.cn/traffic-sign/)。Dataset Ninjaによる2021年版の解析：16,817画像、27,359のアノテーション付きオブジェクト、232クラス、train 6,105 / test 3,071 / other 7,641、ラベルなし画像6,219(https://datasetninja.com/tt100k-2021)。
- ライセンス（公式ページから引用）："[Tsinghua-Tencent 100K dataset is under Creative Commons Attribution-NonCommercial (CC-BY-NC) license.]"、および"If you have any questions about the code or dataset, or request for commercial use, please contact Song-Hai Zhang (shz@tsinghua.edu.cn)"。ダウンロードにはフォームで"name"、"Email"、"Your orgnization"を入力する必要がある(https://cg.cs.tsinghua.edu.cn/traffic-sign/)。UltralyticsとDataset NinjaはCC BY-NC 2.0としている(https://docs.ultralytics.com/datasets/detect/tt100k, https://datasetninja.com/tt100k-2021)。公式ページはバージョン番号を示していない。
- ベースライン：MTSD論文の引用によれば、CVPR 2016のFast R-CNN風ネットワークで81.6 mAP。MTSDの2段階の検出器 + 分類器はTT100Kで89.9 mAP、MTSDで事前学習すると93.4 / 95.7に達する(https://arxiv.org/pdf/1909.04422)。UltralyticsのページにYOLOの結果はない(https://docs.ultralytics.com/datasets/detect/tt100k)。
- Hugging Face上のミラー：PrashantDixit0/TT-100K（ライセンスタグcc-by-2.0で、公式のNCライセンスと矛盾する。ダウンロード数1,117）、Genius-Society/tt100k（cc-by-nc-4.0タグ、train 6,105 / validation 1,097 / test 3,071）、finedet/tt100k（cc-by-nc-4.0）、GlowBond/TT100K_Annotationed（ライセンスなし）(https://huggingface.co/api/datasets?search=TT100K&limit=100 および各データセットのAPI呼び出し)。"iteration version datasets based on TT100K"を用いる拡張手法がarXiv 2303.18037に記述されている(https://arxiv.org/abs/2303.18037)。
- 既知の問題：NCライセンス。登録フォームがある。221クラスの大半はインスタンスが非常に少ない（100を超えるのは45クラスのみ）。"other"分割の扱いが曖昧である。2021年版のアノテーションは単独では使えない。

### 2.3 Mapillary Traffic Sign Dataset、MTSD（全世界）[重要]

- 作成元：Ertler, Mislej, Ollmann, Porzi, Neuhold, Kuang（Mapillary、現Meta）, "The Mapillary Traffic Sign Dataset for Detection and Classification on a Global Scale", ECCV 2020。2019-06-27に発表された(https://arxiv.org/pdf/1909.04422, https://blog.mapillary.com/update/2019/06/27/mapillary-traffic-sign-dataset.html, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf)。
- 規模：約100K画像。完全アノテーション済みの52,453画像に257,543ボックス、部分アノテーションの47,547画像に自動生成ラベル87,358。完全アノテーション分の分割はtrain 36,589 / val 5,320 / test 10,544で、テストのアノテーションは非公開。解像度は約1 MPから16 MP超、カメラメーカー200社超、360度パノラマ1,138枚(https://arxiv.org/pdf/1909.04422)。論文の表1：100,000画像、313クラス、325,172ボックス。train+valは41,907画像、206,388ボックス(https://web3.arxiv.org/pdf/1909.04422v2)。
- クラス：arXiv版では313クラスと"other-sign"(https://arxiv.org/pdf/1909.04422)。ECCVのカメラレディ版は、Wikimedia Commonsのテンプレートを外観でグループ化した400クラスを報告している(https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf)。公開されたv2アノテーション（mtsd_v2_fully_annotated）は400クラスの分類体系を用いる。HFミラーのid2labels.txtには400エントリがあり、regulatory 176、warning 135、complementary 45、information 44で、"regulatory--maximum-speed-limit-10--g1"のように命名されている(https://huggingface.co/datasets/ThankGod/mapillary_traffic_sign_dataset/raw/main/id2labels.txt)。属性：occluded、ambiguous、dummy、out-of-frame、included、exterior(https://arxiv.org/pdf/1909.04422)。アノテーション形式：画像ごとに1つのJSONで、width、height、ispano、objects[bbox, key, label, properties]を持ち、パノラマにはcross_boundaryボックスがある。部分アノテーションのセットには対応付けのキーがある(https://huggingface.co/datasets/NgTMDuc/cl_dataset/resolve/main/MTSD/annotations/mtsd_v2_fully_annotated/README.md)。
- 地域：画像は国ごとにサンプリングされ、北米20%、欧州20%、アジア20%、南米15%、オセアニア15%、アフリカ10%を目標とする。"covering almost all habitable areas of the world"とあるが、国の一覧はなく、論文にもブログ記事にも日本への言及はない(https://arxiv.org/pdf/1909.04422, https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf, https://blog.mapillary.com/update/2019/06/27/mapillary-traffic-sign-dataset.html)。したがって日本の画像が含まれるかは未検証である。画像キーはMapillaryの画像IDなので、データを持つプロジェクトであればMapillary APIを通じて位置情報を復元できる。
- ライセンス：アノテーションに同梱のLICENSE.txtの全文は次のとおりである："This Mapillary Traffic Sign Dataset is provided under the Creative Commons Attribution NonCommercial Share Alike (CC BY-NC-SA) license available here: https://creativecommons.org/licenses/by-nc/4.0/legalcode" （本文はNC-SAとしながら、リンク先はBY-NCのリーガルコードである）(https://huggingface.co/datasets/NgTMDuc/cl_dataset/resolve/f8fa651855ae2cf72a47cdb7a45ef044e03ef8f8/MTSD/annotations/LICENSE.txt)。論文は"freely available for academic research"とし、元画像はMapillary上でCC BY-SAであるとする(https://arxiv.org/pdf/1909.04422)。ブログ記事には"a commercial edition and a research edition"があると書かれている(https://blog.mapillary.com/update/2019/06/27/mapillary-traffic-sign-dataset.html)。公式ページ www.mapillary.com/dataset/trafficsign はJavaScriptで描画されるため読めなかった（第9節）。
- 入手方法：Mapillaryにサインインしてデータセットページからダウンロードする（本セッションでは未検証、第9節参照）。
- ベースライン：Faster R-CNN + FPN ResNet101で、長辺最大4000 pxのとき88.44 AP（2048 pxでは81.80）。2段階の検出器 + 分類器で、313クラスに対し83.4 mAP(https://arxiv.org/pdf/1909.04422)。
- Hugging Faceミラーとそのライセンス状況：ThankGod/mapillary_traffic_sign_dataset（train 3,813画像、3.2 GB、400クラスのラベル、ライセンスタグなし、ダウンロード数647）、ThankGod/mapillary_traffic_sign_dataset_cleaned（train 23,360 / test 1,297 / validation 1,297、21.8 GB、ライセンスタグなし）、ThankGod/mapillary_traffic_sign_dataset_recommended（train 25,956画像、21.8 GB、ライセンスタグ"mit"）(https://huggingface.co/api/datasets/ThankGod/mapillary_traffic_sign_dataset?full=true および兄弟データセット2件のAPI呼び出し)。MITタグは元データのNCライセンスと矛盾する。3つともMTSDの画像を再配布しているが、NC-SAライセンスがそれを認めるのは非商用かつ同一ライセンスの場合のみなので、OpenTSRはこれらを「参照のみ、再ホストしない」ものとして扱うべきである。sparshgarg57/mapillary_traffic_signsは別物で、VLM評価のために細粒度の標識ラベルを付けたMapillary Vistasの検証用画像である（arXiv 2508.02047）。ライセンスタグはない(https://huggingface.co/datasets/sparshgarg57/mapillary_traffic_signs/raw/main/README.md)。NgTMDuc/cl_datasetにはMTSD v2のアノテーションとLICENSE.txtのコピーが含まれる(https://huggingface.co/api/datasets/NgTMDuc/cl_dataset/tree/main/MTSD/annotations)。
- 既知の問題：NCライセンス。テストラベルは非公開。分類体系はロングテールである（論文によれば257,543ボックスのうち分類体系内のクラスを持つのは66,138）。クラス名は国ではなく外観グループ（g1、g2 ...）を表すため、国別の評価には画像の位置情報が必要である。

### 2.4 DFG Traffic Sign Dataset（スロベニア）[重要]

- 作成元：Tabernik and Skocaj, "Deep Learning for Large-Scale Traffic-Sign Detection and Recognition", IEEE T-ITS 2019, DOI 10.1109/TITS.2019.2913588。ViCoS Lab, University of LjubljanaとDFG Consultingによる(https://vicos.si/resources/dfg)。
- 規模：6,957画像（train 5,254 / test 1,703）、うち6,758が1920x1080、199が720x576。30 px以上で厳密にアノテーションされたポリゴンインスタンス13,239に加え、30 px未満で大まかにアノテーションされ"ignore"とされたボックス4,359。各20インスタンス以上の200カテゴリ。8,775枚の追加画像と30,000超のインスタンスを含む拡張セット。COCO JSON。顔とナンバープレートはぼかし済み。改訂はv1.01（2019-04-25）とv1.02（2019-06-14）(https://vicos.si/resources/dfg)。
- ライセンス（引用）："The dataset is licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License"。商用利用についてはDanijel Skocajに連絡する(https://vicos.si/resources/dfg)。
- 入手方法：直接リンクgo.vicos.si/dfgannot、/dfgaugannot、/dfgaugannotdiff、/dfgimages（JPEGImages.tar.bz2）、/dfgcat。ファイルサイズの記載なし(https://vicos.si/resources/dfg)。
- ベースライン：T-ITS論文はMask R-CNNを用いる。リソースページに数値はない（未検証）。
- 既知の問題：NC-SA。スロベニアの標識のみ。200クラスの多くは文字を含む案内・情報板である。

### 2.5 Swedish Traffic Signs、STS（スウェーデン）

- 作成元：Larsson and Felsberg, SCIA 2011（LNCS 6688, pp. 238-249）。Linkoeping University CVL(https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/)。
- 規模：1.3 MPのダッシュボードカメラで高速道路と市街地を350 km以上走行して得た"more than 20 000 images"。5フレームに1枚をラベル付けし、道路標識は3,488。7クラス（横断歩道、右側の指定車線、駐停車禁止、優先道路、譲れ、50 kph、30 kph）で、可視性（遮蔽、ぼけ、可視）と道路の状態（走行中の道路上か側道か）の属性を持つ(https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/)。MTSD論文の比較表ではSTSを3,777画像、20クラス、5,582標識としており(https://web3.arxiv.org/pdf/1909.04422v2)、数値が異なる。
- ライセンス：旧概要ページには"Without prior approval from CVL, the Traffic Signs Database, in whole or in part, may not be modified or used for commercial purposes."（CVLの事前承認なしに、全体・一部を問わず改変や商用利用をしてはならない）とある(https://www.cvl.isy.liu.se/research/trafficSigns/)。現行のページは引用のみを求めている(https://www.cvl.isy.liu.se/en/research/datasets/traffic-signs-dataset/)。著者らは2026-04-20に、データを"traffic-signs_dataset.zip"（15.8 GB）としてcc-by-4.0でZenodoに再アップロードした。DOI 10.5281/zenodo.19666339(https://zenodo.org/api/records/19666339)。OpenTSRの目的においては、ZenodoのCC BY 4.0のコピーが旧来の制限に優先する（再配布の前に著者に確認すること）。
- 入手方法：CVLからSet1/Set2の各パートとannotations.txtを入手するか、Zenodoのzipを使う（URLは同上）。

### 2.6 BelgiumTS検出、BTSD（ベルギー）

- 第1.2節と同じ素材。BelgiumTSD_annotations.zip（485,645バイト）は"provided for detection evaluation, it uses the same material and splits"である。アノテーションファイルtraining_GT.txt、testing_2D_GT.txt、testing_comparison_GT.txtは、8カメラの画像フォルダを基準とした支柱単位のアノテーションである(https://btsd.ethz.ch/shareddata/, https://btsd.ethz.ch/shareddata/BelgiumTS/structure.txt)。MTSDの表ではBelgiumTSは108クラス、8,851標識とされる(https://web3.arxiv.org/pdf/1909.04422v2)。検出用分割の画像数：未検証（第9節）。ライセンス：引用の依頼のみ（第1.2節）。

### 2.7 RTSD、Russian Traffic Sign Dataset（ロシア）

- 作成元：Graphics and Media Lab, Moscow State University（Shakhuro and Konushin, Computer Optics 2016、ロシア語）(https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html)。
- 規模：検出用はtrain 47,639画像 / 80,277標識、test 11,389画像 / 25,232標識。分類用はtrain 79,896 / test 25,613サンプル。計205クラスで、学習に含まれるのは106、99クラスはテストセットにのみ現れる（希少クラス）。あらゆる季節と天候のもとで5 fpsのドライブレコーダーにより撮影した1280x720から1920x1080のフレーム(https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html)。MTSDの表では179,138フレーム、156クラス、104,358標識とされる(https://web3.arxiv.org/pdf/1909.04422v2)。
- ライセンス：記載なし。Yandex Diskのリンク https://yadi.sk/d/TX5k2hkEm9wqZ からダウンロードする(https://graphics.cs.msu.ru/projects/traffic-sign-recognition.html)。再配布の権利は不明確である。

### 2.8 CCTSDBとCCTSDB2021、CSUST Chinese Traffic Sign Detection Benchmark（中国）

- 作成元：Zhang Jianming et al., Changsha University of Science and Technology。CCTSDBは先行するCTSDに高速道路の画像を追加して拡張したものである(https://arxiv.org/pdf/1902.06857 の関連研究)。CCTSDB 2021の論文：Human-centric Computing and Information Sciences 12:23, 2022, DOI 10.22967/HCIS.2022.12.023 (https://raw.githubusercontent.com/csust7zhangjm/CCTSDB2021/master/README.md)。
- CCTSDB2021の規模：学習セットと正例テストセットを合わせて17,856画像。学習画像16,356、正例テスト画像1,500（番号18992-20491）、負例画像500。mandatory、prohibitory、warningの3カテゴリ。XMLとTXTのアノテーション。テストのサブセットは天候/照明別と標識サイズ別に分かれる（同README）。Zenodoミラー：dataset.rar 1,555.6 MB、"approximately 40,000 annotated traffic signs"、ライセンスgpl-3.0、DOI 10.5281/zenodo.17152727(https://zenodo.org/api/records/17152727)。
- ライセンス：CCTSDB2021のGitHubリポジトリはGPL-3.0（スター102）(https://img.shields.io/github/license/csust7zhangjm/CCTSDB2021.json, https://img.shields.io/github/stars/csust7zhangjm/CCTSDB2021.json)。元のCCTSDBリポジトリにはライセンスがなく（スター271、最終コミット2024年7月）、READMEはmasterでもmainでも取得できなかった(https://img.shields.io/github/license/csust7zhangjm/CCTSDB.json, https://img.shields.io/github/stars/csust7zhangjm/CCTSDB.json)。画像データにGPLライセンスを付けるのは異例であり、「著者に確認する」ものとして扱う。
- 入手方法：README内のGoogle DriveとBaiduのリンク（同README）。
- 既知の問題：上位クラスが3つのみで、細かい種別はない。

### 2.9 CTSD、Chinese Traffic Sign Dataset（中国）

- 3クラス（prohibitory、mandatory、danger）の1,100画像（学習700 / テスト400）。Yang et al. 2016で導入された(https://arxiv.org/pdf/1804.10428, https://arxiv.org/pdf/1902.06857)。ホスティングとライセンス：未検証（NLPRの公式ページに到達できない、第9節）。

### 2.10 韓国

- KRID、Korean Road Infrastructure Dataset（Data, MDPI, 10(3):36, 2025年3月）：3328x1872の約200万フレームから選んだ、アノテーション付きの200,000フレーム。道路標識、安全標識、道路名板、視線誘導標、信号機などを含む34の道路施設クラスで2,676,583オブジェクト（ボックス2,031,633、ポリゴン644,950）。データ可用性の記述によればライセンスはCC BY-NC-SA。AI Hub（韓国の研究者向け）とGitHubのkimhyeongbok/korean-road-infrastructure-datasetで配布されている(https://mdpi.com/2306-5729/10/3/36)。標識種別は細粒度ではラベル付けされていない（ページに標識の分類体系はない）。
- SAIT Traffic Landmark Dataset（Samsung Advanced Institute of Technology、IEEE DataPort、DOI 10.21227/gar1-z597）：Seoul, Suwon, Hwaseong, Yongin, Seongnamで撮影した約150,000枚のBMP画像に、約470,000個のランドマークを（ボックスではなく）基準点としてアノテーションしたもの。6クラス：警戒標識、禁止標識、指示標識、補助標識、信号機、路面標示。301.72 GB。IEEE DataPortのサブスクリプションが必要。ライセンスの記載なし(https://ieee-dataport.org/documents/sait-traffic-landmark-dataset)。
- Kakao Mobilityは2025年に韓国向けの自動運転学習用データセットを発表した（検索結果でタイトルを見たのみ、未取得）。

### 2.11 ZOD、Zenseact Open Dataset（欧州14か国）

- 作成元：Zenseact AB（Alibeigi et al., ICCV 2023）(https://huggingface.co/datasets/Zenseact/ZOD/raw/main/README.md, https://arxiv.org/pdf/2305.02008)。
- 道路標識：クラス"TrafficSign"、2Dボックス（LiDARでも見える場合は3D）。分類体系は6カテゴリ（Mandatory, Priority, Prohibitory, RoadType, Special, Warning）と156の細粒度クラス。属性はOcclusionRatio、IsForConstruction、IsElectronic、ContainsInnerSigns、ContentContainsText、IsForEgoRoad、IsForEgoLane、IsSticker。ラベル付きの一意な標識インスタンスは446k（"about twice the 206k in MTSD"）。"annotations for traffic signs and ego-road are unavailable for some Frames"とある(https://arxiv.org/pdf/2305.02008)。
- 地域：欧州14か国。明記されているのはスウェーデン、ドイツ、フランス、イタリア。日本は含まれない(https://arxiv.org/pdf/2305.02008)。
- ライセンス：CC BY-SA 4.0で、論文はこれを研究利用と商用利用の両方を認めるものと説明している(https://arxiv.org/pdf/2305.02008)。HFカードのライセンスタグはcc-by-sa-4.0(https://huggingface.co/api/datasets/Zenseact/ZOD?full=true)。入手はzod.zenseact.comのdevkitを通じて行う（ダウンロード手順は未検証）。
- 既知の問題：欧州の標識のみ。一部のフレームで標識ラベルが欠けている。マルチモーダルでダウンロード量が大きい。

### 2.12 カナダ、CanDOR-99（Mapillary由来）

- Zenodoレコード20739255（2026-06-18）：95,314画像、99クラスで103,521のCOCOアノテーション。クラスは"organized by Mapillary taxonomy"（regulatory 37、warning 45、complementary 14、information 3）。2025年9月から12月にかけて、Mapillary APIを通じてオタワ、トロント、バンクーバー、モントリオール、ケベック・シティ、カルガリー、ビクトリアの画像を収集し、アノテーションは手作業または分類器で検証・修正した。ライセンスcc-by-sa-4.0、117.9 GB(https://zenodo.org/api/records/20739255)。これは、Mapillaryから日本のセットを構築する方法の手本となる（第3.5節）。

### 2.13 多目的の走行データセット：道路標識クラスはあるか、種別はラベル付けされているか

| データセット | 標識クラス | 種別ラベル付き | ライセンス | 出典 |
|---|---|---|---|---|
| Argoverse 2 Sensor（米国、6都市、1,000 x 15 s） | 3Dキューボイド：30カテゴリ中のSIGN、STOP_SIGN、MOBILE_PEDESTRIAN_SIGN、MESSAGE_BOARD_TRAILER、TRAFFIC_LIGHT_TRAILER | 一時停止標識のみ | "All datasets are released under the CC BY-NC-SA 4.0 license" | https://arxiv.org/pdf/2301.00493, https://argoverse.org/av2.html |
| Waymo Open Perception（米国、2,030 x 20 s） | 3D LiDARラベルのクラス"Signs"（4クラスの1つ、12.6Mボックス）。28クラスの2Dパノプティックラベルにも"Sign"がある | なし | "licensed for non-commercial use" | https://waymo.com/open/about |
| BDD100K（米国、100k画像） | 10クラス中の検出クラス"traffic sign"（274,594オブジェクト） | なし（下のBDD100K-MUTCDを参照） | 独自：無料で"for educational, research, and not-for-profit purposes"、商用はBAIR CommonsまたはUC Berkeley OTL経由 | https://datasetninja.com/bdd100k |
| BDD100K-MUTCD（IEEE DataPort、2026-07） | 80,000枚のBDD100K画像から得た検証済み標識クロップ15,628個に対する38クラスのMUTCD分類体系。Geminiでラベル付けした後、一部を人手で確認 | あり（38クラス） | サブスクリプション。画像はBDD100Kから入手する必要がある | https://ieee-dataport.org/documents/bdd100k-mutcd-fine-grained-human-verified-traffic-sign-type-annotations-bdd100k-dataset |
| Cityscapes（ドイツ、50都市） | セマンティッククラス'traffic sign'（id 20、trainId 7） | なし | 非商用。"That you do not distribute this dataset or modified versions" | https://raw.githubusercontent.com/mcordts/cityscapesScripts/master/cityscapesscripts/helpers/labels.py, https://www.cityscapes-dataset.com/license/ |
| Mapillary Vistas（全世界、25,000画像、v2.0で124クラス） | Traffic Sign (Front)（17,178画像に93,287オブジェクト）、Traffic Sign (Back)、Traffic Sign - Direction (Front/Back)、Traffic Sign - Parking、Traffic Sign - Temporary (Front/Back)、Traffic Sign - Ambiguous、Traffic Sign Frame | なし（標識の機能グループのみ） | Dataset Ninja：CC BY-NC-SA 4.0（研究用）。ICCV論文は商用版に言及。公式ページは未取得 | https://raw.githubusercontent.com/open-mmlab/mmsegmentation/main/mmseg/datasets/mapillary.py, https://datasetninja.com/mapillary-vistas-dataset, https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf, https://blog.mapillary.com/update/2021/01/18/vistas-2-dataset.html |
| nuImages / nuScenes（ボストン、シンガポール） | アノテーション指示書に道路標識のカテゴリはない（付属物の注記として"signboard"があるのみ） | なし | 未検証（利用規約ページはJavaScriptのみ） | https://raw.githubusercontent.com/nutonomy/nuscenes-devkit/master/docs/instructions_nuimages.md |
| A2D2（ドイツ、ラベル付き41,277フレーム） | セマンティッククラス"Traffic sign 1-3"と"Traffic signal 1-3" | なし | CC BY-ND 4.0 "permitting commercial use subject to the terms of the license"（改変禁止） | https://arxiv.org/abs/2004.06320 とそのPDF |
| IDD、India Driving Dataset（セグメンテーション画像10,003、検出画像46,588） | ラベル'traffic sign'（id 24）と'traffic light' | なし | 未取得（サイトはJavaScriptが必要） | https://raw.githubusercontent.com/AutoNUE/public-code/master/helpers/anue_labels.py, https://idd.insaan.iiit.ac.in/dataset/details/ |
| DDAD（TRI。米国 + 東京/お台場） | なし（深度ベンチマークのみ） | なし | CC BY-NC-SA 4.0 | https://raw.githubusercontent.com/TRI-ML/DDAD/master/README.md |

### 2.14 aiMotive 3D Traffic Light and Traffic Sign Dataset（米国 + ハンガリー）

- 作成元：aiMotive（ブダペスト）、論文"Accurate Automatic 3D Annotation of Traffic Lights and Signs for Autonomous Driving"(https://arxiv.org/html/2409.12620v3, https://raw.githubusercontent.com/aimotive/aimotive_tl_ts_dataset/main/README.md)。
- 規模：約15 sのシーケンス220本、自動アノテーションされた約50,000フレーム、信号機の3Dボックス約320,000と道路標識の3Dボックス約550,000。カリフォルニアとハンガリー。カメラ4台（Sony IMX490、2896x1876）。フレームごとの3Dボックス、灯火の状態、標識種別、遮蔽、Google Vision APIによる標識の文字(https://arxiv.org/html/2409.12620v3, README)。標識種別の数：記載なし（同上）。
- ライセンス：論文ではCC BY-NC-SA 4.0、READMEでは"made freely available for non-commercial research purposes only"。GitHubにはライセンスファイルがない(https://arxiv.org/html/2409.12620v3, README, https://img.shields.io/github/license/aimotive/aimotive_tl_ts_dataset.json)。ダウンロード：S3またはKaggle上の107 GBのzip（README）。

### 2.15 タスクで名前が挙げられたHugging Faceの検出セット

- Ericzzzzh/STIC-Traffic-Sign（作成2026-10-05、CC BY 4.0タグ、ダウンロード数199）："US MUTCD-style traffic-sign detection in YOLO format"。Roboflowからのエクスポート2つから成る。TS_1は7,204画像 / 21,282ボックス / 7クラス（Guide Signs-Information, Guide Signs-Route, Makers, Regulatory Signs-BW, Regulatory Signs-Red, School, Warning Signs）で、1920x1080に引き伸ばし、train 5,764 / val 1,440。TS_2は1,564画像 / 3,374ボックス / 5クラスで、640x640、確率50%の水平反転によるオフライン拡張を施している。出典はRoboflow Universeのtest-rtrlj/ts-27jkgとtest-rtrlj/520-smb96(https://huggingface.co/datasets/Ericzzzzh/STIC-Traffic-Sign/raw/main/README.md, https://huggingface.co/api/datasets/Ericzzzzh/STIC-Traffic-Sign?full=true)。MUTCDの粗い機能グループのみである。CC BY 4.0はRoboflowのアップロード者の主張である。
- emb-ai/traffic-sign-bench（作成2026-08-22、ライセンスodbl、ダウンロード数2,498）：画像データセットではない。"Official per-sign SUMO maps for TrafficSignBench: real Moscow OSM layouts, 25 signs, 2500 maps"であり、標識ごとにtrain 80 + test 20のマップがある。ジオメトリはODbL 1.0のもとでOpenStreetMapから得ている。視覚ではなく、標識の意味を扱う交通シミュレーションのベンチマークである(https://huggingface.co/datasets/emb-ai/traffic-sign-bench/raw/main/README.md)。
- KeenForgeAI/TrafficSigns_YOLO-corrected（作成2026-10-14、ライセンスタグ"other"）："Traffic Signs Dataset for YOLOv8-YOLOv11 benchmarking"（Reveles-Martinez et al., Technologies 2025）を、リークが起きないように再分割したもの。屋内のFIRA 1:10スケール自動運転車トラックでIntel RealSense D435iにより撮影した640x640の6,000画像、6クラス（left_turn, straight_ahead, right_turn, dead_end, no_entry, stop）、1画像につき1ボックス。2,526画像はほぼ同一の拡張コピーで、上流の分割ではtrain/val/testをまたいでリークしていたため、4,259 / 858 / 883に再分割した。"Upstream license: none stated"(https://huggingface.co/datasets/KeenForgeAI/TrafficSigns_YOLO-corrected/raw/main/README.md)。玩具的なトラックのデータであり、道路画像ではない。

### 2.16 派生コレクションと攻撃研究向けコレクション

- "Datasets Resources of Paper 'Hidden in Plain Signs: Realistic Sticker Attacks on Production Traffic Sign Recognition Systems'"（Zenodo 23527395、2026-10-21、匿名著者、CC BY-NC-SA 4.0、6.59 GB）：Dataset AはDFG（867画像）、TT100K（3,221）、GTSDB（305）、ItalianSigns（324）、BelgiumTS（178）、"Road Signs Detection [Italy]"（64）と独自画像6,717を18クラスにまとめたもの。Dataset BはMTSD（13,662）、DFG（2,360）、TT100K（6,605）、GTSDB（357）、ItalianSigns（361）を17クラス（速度制限、一時停止、車両通行止め、駐停車禁止、駐車禁止、進入禁止）の30,226画像にまとめたもので、640x640のクロップ、YOLOラベル、情報源ごとの変換スクリプトを含む。"Academic Use Only"(https://zenodo.org/api/records/23527395)。OpenTSRにとっては、国をまたぐ速度制限サブセットを作る手順として、またメンテナの敵対的攻撃研究の先行事例として有用である。
- GTSRBのバックドア版とGTSRB-Concepts：第1.1節。

---

## 3. 日本（日本の道路標識）[重要]

結論から言えば、2026-10-31時点で、研究水準（数千フレーム、規制/指示/警戒/案内/補助の細粒度ラベル、明文化されたライセンス）を満たす日本の道路標識の公開データセットは存在しない。メンテナが自ら判断できるよう、見つかったものはすべて以下に挙げる。検索は英語と日本語（道路標識 データセット、交通標識 データセット、日本 標識 認識 公開データ）で、Hugging Face API（データセットとモデル）、Zenodo API、Roboflow Universe、Kaggle（検索エンジン経由）、IEEE DataPort、GitHub、J-STAGE、IEICE、IPSJのページを対象に実施した。

### 3.1 コミュニティのデータセット（Roboflow Universe）

- ritsumeikan/japanese-road-signs（"Japanese Road Signs Computer Vision Project"、作成者Ritsumeikan、引用年2022、"Updated 2 years ago"）：5クラス（Information, Mandatory, Prohibitory, Speed Limit 30, Warning）。ライセンスCC BY 4.0。ページには画像数もアノテーションの種類も示されておらず、説明文もない(https://universe.roboflow.com/ritsumeikan/japanese-road-signs)。カテゴリの網羅性：粗いグループとして案内または指示（"Information"、"Mandatory"）、規制（"Prohibitory"、"Speed Limit 30"）、警戒（"Warning"）が混在し、補助はない。
- test-u58bl/japan-roadsign（2024-09）：3クラス（not_speed, speed, with_auxiliary_sign）。CC BY 4.0。"This project has no images available to display"とある(https://universe.roboflow.com/test-u58bl/japan-roadsign)。扱うのは速度制限とその他の規制の区別、および補助標識の有無のみである。
- test-u58bl/japan-roadsign-second-2（2024-12）：536画像、物体検出、車両と標識の内容が混在する16クラス（car, truck, Bus, bicycle, Bike, Taxi, Finish, Handcarts, horn, No_crossing, No_overtaking, No_turning, Pedestrians, Section, Start, Two_seater）。CC BY 4.0(https://universe.roboflow.com/test-u58bl/japan-roadsign-second-2)。同じワークスペースに兄弟プロジェクトjapan-roadsign-ver.2とjapan-roadsign-secondがある（検索結果でURLを見たのみ、未取得）。
- media-traverse-co-ltd/japan-road-scene（v1、2023-04-24）：1,306画像（train 914 / valid 261 / test 131）、クラスの一覧なし、ライセンスの記載なし。ページは道路標識に言及していない(https://universe.roboflow.com/media-traverse-co-ltd/japan-road-scene/dataset/1)。
- 評価：いずれも趣味の規模で文書化が不十分であり、CC BY 4.0のタグはアップロード者が選んだものである。元画像の権利は不明である。

### 3.2 日本の標識で学習され、データが公開されていないモデル

- infra-intelligence/road-sign-yolox-tiny（Hugging Faceのモデル、Apache-2.0、作成2026-10-22、ダウンロード数34）：日本の標識板を検出し、道路標識令の5カテゴリ案内（guide）、警戒（warning）、規制（regulatory）、指示（indication）、補助（supplementary）とhard_negativeクラスに分類するYOLOX-Tiny検出器（パラメータ数約5.03 M、入力1280x736）。標識の種別は区別しない（"標識の種類（「一時停止」「最高速度」など）までは分けません"）。学習データ：日本国内のMapillary画像（CC BY-SA 4.0）、15地域。train 4,585画像（標識ありは3,499。ボックス数：案内1,833、警戒363、規制3,878、指示634、補助1,100、ignore 2,048）、validation 992画像。評価は、学習から除外した7地域の596画像（988ボックス）で行い、recall 0.857、precision 0.877、AP50 0.902（クラス非依存）、AP50-95 0.700、クラス別AP50は案内0.594 / 警戒0.958 / 規制0.937 / 指示0.893 / 補助0.748。ラベルはMapillaryの自動標識検出結果をもとに下書きし、文書化されたルールに従って"AI assistant"が調整した。ラベルは画像ごとには確認されていない。"画像とラベル（データセット）は、この公開物には含めていません"とある(https://huggingface.co/infra-intelligence/road-sign-yolox-tiny/raw/main/README.md, https://huggingface.co/api/models/infra-intelligence/road-sign-yolox-tiny)。5カテゴリすべてをカバーするが、カテゴリ単位のみである。重みはApache-2.0で、評価JSONが含まれている。
- 東京大学関本研究室（Yang, Maeda, Sekimoto）、ポスター"Vison-based Traffic Sign Detection and Localization in Tokyo Metropolitan Area"：東京で独自に収集したZEDステレオカメラの走行データで、YOLOv4による検出（IoU 0.5でAP 94%）とVGG19/ResNet101/Swin-Bによる分類を行った。レーザー距離付きのテスト画像50枚。データセット公開の記載はない(https://sekilab.iis.u-tokyo.ac.jp/wp-content/uploads/poster_YANG.pdf)。
- 大阪工業大学（Kato, Nishiguchi, Mizutani, Hashimoto）、FIT2018 H-027：2016年9月の国土交通省の道路標識一覧にある225クラスについて、3D都市モデル上でUnityによりレンダリングした合成学習画像303,750枚（とマスク）（1クラスあたり1,350バリエーション）。評価はYOLOv2で、実写真30枚とStreet Viewのクロップのみで行った。非公開(https://www.ieice.org/publications/conference-FIT-DVDs/FIT2018/data/pdf/H-027.pdf)。同グループのIEICE PRMU2018-133（2019年3月）は、3DCGの標識データセットと、幾何学的に拡張した実写真セットを比較している。規模と入手可否は記載なし(https://ken.ieice.org/ken/paper/20190317h1kE/)。
- 東京電機大学（Sekizawa, Nakajima）、IEICE PRMU2018-89（2018年12月）：合成した道路シーン全体でエンドツーエンドの標識検出器を学習した。データセットの公開はない(https://ken.ieice.org/ken/paper/2018121461IO/)。
- Turing Inc.とNII（Iwamasa, Shiotsuka, Yamaguchi, Miwa, Aoki）、JSAI 2023：900時間以上の日本の走行データと、15,000画像の信号機データセット。信号機のみで、入手可否は記載なし(https://www.jstage.jst.go.jp/article/pjsai/JSAI2023/0/JSAI2023_2M6GS1002/_article/-char/en)。
- TIER IV：NEDOの助成による"large-scale dataset for autonomous driving"が2025-12-25に発表された。自動ラベル付けと複数の車種を用いるとされるが、規模、センサ構成、標識のアノテーション、公開日、ライセンスは示されていない(https://www.fieldtechnologiesonline.com/doc/tier-iv-to-develop-large-scale-dataset-for-autonomous-driving-under-nedo-project-0001)。TIER IVの以前の信号機モデルは信号機の調査（traffic_lights/docs/oss_implementations.md）に記載されており、その学習データは社内のものである。

### 3.3 商用の提供物（非公開）

- Brycen Co., Ltd.（株式会社ブライセン）、2024-02-27のプレスリリース："道路付属物"（ガードレール、道路標識、道路照明）のアノテーション付きセマンティックセグメンテーションデータセット。標識を本標識と補助標識に分ける"詳細版"がある。36クラス。無料サンプルは申請により提供、完全版は販売。画像数、価格、ライセンスは記載なし(https://kyodonewsprwire.jp/release/202402277052)。
- Nexdata / Datatangは2D/3D/4Dアノテーション付きの自動運転データセットを宣伝している（検索結果でニュースページを見たのみ、未取得）。日本の標識種別ラベルが提供されているかは未検証である。
- BOOTH上のrita-studio：日本の路側式と頭上式の標識柱のフォトリアルな3D CGモデル（FBX、4Kテクスチャ）を500円で販売。商用・非商用を問わずプロジェクトで無料で利用できるが、再配布と転売は禁止(https://booth.pm/zh-cn/items/3981593)。日本の合成シーンのレンダリングには使えるが、データセットではない。

### 3.4 全世界のデータセット：日本の標識を含むか

- MTSD：アジア20%が目標、国の一覧なし、日本への言及なし。含む・含まないのいずれも未検証（第2.3節）。
- Mapillary Vistas："spans all the continents except for Antarctica"、国の一覧なし、日本への言及なし(https://datasetninja.com/mapillary-vistas-dataset, https://openaccess.thecvf.com/content_ICCV_2017/papers/Neuhold_The_Mapillary_Vistas_ICCV_2017_paper.pdf)。標識はfront/back/direction/temporary/parkingのグループとしてのみラベル付けされている。
- ZOD：欧州14か国のみ(https://arxiv.org/pdf/2305.02008)。日本は含まれない。
- DDAD（Toyota Research Institute）：東京とお台場のシーンがある（日本：学習47シーン / 3,900フレーム、検証14シーン / 950フレーム）。ただし物体ラベルのない深度ベンチマークで、CC BY-NC-SA 4.0である(https://raw.githubusercontent.com/TRI-ML/DDAD/master/README.md)。日本の画像はあるが、ラベルなしでNCである。
- Mapillaryプラットフォームの画像自体はCC BY-SA 4.0で、日本もカバーしている（infra-intelligenceが利用、第3.2節）。Mapillary自身の標識検出は"{category}--{name}--{appearance-group}"という分類体系に従い、値はregulatory--stop--g1のような形である。開発者ページでは218の値がMUTCDコードに対応付けられている(https://www.mapillary.com/developer/api-documentation/mutcd)。CanDOR-99（第2.12節）は、一国の標識データセットをMapillaryから構築し、CC BY-SA 4.0で公開できることを示している。

### 3.5 日本のピクトグラムと公式資料

- Wikimedia Commonsにある日本の標識のSVG図（例：File:Japan road sign 302.svg、車両通行止め、作者Monaneko、出典"道路標識、区画線及び道路標示に関する命令別表第2"）には、パブリックドメインのタグが付いている："This work is in the public domain in Japan according to Article 13 of the Copyright Act of Japan"(ミラーページ https://download.osmand.net/wiki/File:Japan_road_sign_302.svg; 本セッションではCommons APIがHTTP 429を返した、第9節)。確認できた関連カテゴリ名："Road signs in Japan"、"Diagrams of Japanese-language road signs"、"White-on-green road signs in Japan"。ほかに国道標識のSVGや高速道路のシールド標識もある（検索結果による、個別には未検証）。
- 法的な一覧は道路標識、区画線及び道路標示に関する命令（国土交通省・警察庁の命令）であり、その別表第2が規制/指示/警戒/案内/補助のすべての標識を定める。大阪工業大学の研究は、2016年9月の国土交通省の一覧（225クラス）を用いた(https://www.ieice.org/publications/conference-FIT-DVDs/FIT2018/data/pdf/H-027.pdf)。

### 3.6 OpenTSRにとっての意味

- 日本の規制/指示/警戒/案内/補助標識を細粒度の種別でカバーする既存の公開セットはない。あるのは、非公開のMapillaryラベルで学習した5カテゴリのApache-2.0検出器、数百枚のRoboflow画像、パブリックドメインのピクトグラムのみである。
- 日本のベンチマークは自ら構築する必要がある。法的に可能な道は2つあり、(a) 日本のMapillary画像（CC BY-SA 4.0。CanDOR-99と同様に、投稿者へのクレジット表示と、再配布する画像への継承条件が伴う）と、(b) 独自のドライブレコーダー収集データをApache-2.0またはCC BY 4.0で公開する方法である。パブリックドメインのSVGピクトグラムから作る合成データ（第4節）は、225クラスの一覧のロングテールを補う足がかりになる。

---

## 4. 合成データとテンプレートベースのデータ

### 4.1 Synset Signset Germany（Fraunhofer IOSB / IPA）[最も関連が深い]

- 完全版FraunhoferIOSB/Synset-Signset-Germany：ドイツの標識211クラスの105,500画像（1クラス500枚、2020年に導入された標識を含む）。OCTASを用い、OGRE（ラスタライズ）とCycles（パストレーシング）の両方でレンダリングしている。画像ごとにマスク、セグメンテーション、メタデータ（向き、上下の標識、時間帯、コントラスト、ノイズ、モーションブラー、AEC誤差 ...）が付く。汚れや摩耗のテクスチャはGANで生成。テンプレートはドイツ語版Wikipediaの標識一覧から取得。IBLには327のPolyHaven環境マップを使用。ライセンスCC BY 4.0、作成2026-03-03(https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany/raw/main/README.md, https://huggingface.co/api/datasets/FraunhoferIOSB/Synset-Signset-Germany?full=true)。
- GTSRBサブセットFraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset：GTSRBの43クラス、各500画像、レンダラごとに21,500画像、cyclesとogreそれぞれのtrain/validationのparquet。CC BY 4.0で、指定の帰属表示が求められている（"This work is based on the Synset Signset Germany Dataset ... (c) 2024 Fraunhofer IOSB ... Licence: CC BY 4.0"）。論文はSielemann et al., ITSC 2024（arXiv 2512.05936）。カードは、追加の検証なしにEU AI Act附属書IIIの高リスクシステムで使用しないよう警告している(https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/raw/main/README.md, https://huggingface.co/api/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset?full=true)。
- 結果：Signset Cyclesのみで学習したConvNeXt-Smallは、実データのGTSRBでtop-1 98.3%（OGREでは98.2%）に達し、ドメイン内学習より1.2ポイント低いだけである。逆方向（GTSRBで学習しSignsetでテスト）では89.4% / 87.4%となる(https://arxiv.org/pdf/2512.05936)。
- 兄弟データセット：FraunhoferIOSB/SynsetSignsetSensitivityAnalysisDataset、Synset-Background-Effect-Datasets、Synset-Boulevard(https://huggingface.co/api/datasets?search=synset&limit=30)。未調査。

### 4.2 その他の合成セットとチャレンジ合成セット

- CURE-TSRのunreal側の半分（Unreal Engine 4のシーン、After Effectsによるチャレンジ効果）、ZenodoではCC BY 4.0（第1.5節）。
- German Traffic Sign Anomaly Detection Dataset：テンプレートから生成したtrainval_synthetic.zip（3,051 MB）と、異常のある実写真369枚。CC BY 4.0（第1.8節）。
- "traffic sign synthetic dataset"（Gen3D、Sun et al.、Zenodo 17815715、2025-12-04、GPL-3.0-or-later）："Gen3D synthetic traffic sign dataset generated for research"。APIの応答ではレコードにファイルが1つも挙がっていない(https://zenodo.org/api/records/17815715)。現状では使えない。
- tanganke/gtsrbの劣化テスト分割（contrast、Gaussian noise、impulse noise、JPEG、motion blur、pixelate、spatter。各12,630）（第1.1節）。
- 大阪工業大学（Unity、303,750画像、225クラス）と東京電機大学（合成道路シーン）による日本の合成セット：未公開（第3.2節）。
- CARLA、SYNTHIA、Virtual KITTI、GTAベースのセット：本調査のWeb検索とHF検索では、標識種別のラベルが付いた公開データセットは見つからなかった（HFで"carla traffic sign"を検索しても結果はなかった）(https://huggingface.co/api/datasets?search=carla%20traffic%20sign&limit=30)。DriveLM-CARLAはQAセットであり、標識種別のセットではない（第5節）。それ以上は未検証。
- 拡散モデルや世界モデルで生成した標識データセット：本セッションでは公開されたものは見つからなかった（TEMPO-Diffusion、arXiv 2606.26285が道路標識のポイズニングに言及しているが、検索結果で見たのみ）。未検証。
- emb-ai/traffic-sign-bench：標識ごとのSUMO道路ネットワーク（ODbL）。シミュレーションのベンチマークであり、画像はない（第2.15節）。

### 4.3 ピクトグラムの入手元とライセンス

- ドイツ：File:Zeichen 205.svg（Vorfahrt gewaehren）などのCommonsのSVGは、公的著作物としてパブリックドメインである："it is part of a statute, ordinance, official decree or judgment (official work) issued by a German authority or court"。StVOの改正ごとに"Diagrams of regulatory road signs of Germany (2017 set)"のようなカテゴリがある(ミラーページ https://download.osmand.net/wiki/File:Zeichen_205.svg)。Synset Signset Germanyは、ドイツ語版Wikipediaの一覧"Bildtafel der Verkehrszeichen in der Bundesrepublik Deutschland seit 2017"をテンプレートの入手元とした(https://huggingface.co/datasets/FraunhoferIOSB/Synset-Signset-Germany-GTSRB-Subset/raw/main/README.md)。
- 日本：著作権法第13条によりパブリックドメイン（第3.5節）。
- ベルギー：BelgiumTSのDefinedTS.tar.gzには210種の公式標識パターン画像が含まれる(https://btsd.ethz.ch/shareddata/, https://btsd.ethz.ch/shareddata/BelgiumTS/structure.txt)。ライセンスはデータセットと同じ（引用のみ）。
- クロアチア：MASTIFのページで、クロアチアの標識の5カテゴリAからEの見本画像が提供されている(https://www.zemris.fer.hr/~ssegvic/mastif/datasets.shtml)。
- スウェーデン：CVLのページにあるswedishTemplates.zip "Synthetic sign templates"(https://www.cvl.isy.liu.se/research/trafficSigns/)。
- MTSDの400クラスの分類体系自体が、Wikimedia Commonsのテンプレート画像を外観でグループ化して構築されたものであり(https://www.ecva.net/papers/eccv_2020/papers_ECCV/papers/123680069.pdf)、Commonsは全世界の一覧に対するテンプレートの入手元として正当化しやすい。ただしファイルごとのライセンステンプレートは国によって異なるため、1ファイルずつ確認する必要がある（本調査ではCommons APIへのアクセスにレート制限がかかった、第9節）。

---

## 5. 標識のための視覚言語系データセット

- TSI-CN、"Traffic Sign Interpretation in Real Road Scene"（Yang et al., arXiv 2311.10793, v2 2023-11-28）：相互に関連する文字、記号、案内板を自然言語の運転指示に変換するタスク。中国の高速道路と市街地のシーンで、文字・記号・案内板の位置ラベルと説明ラベルを持つ。画像数と標識数はアブストラクトにない。"will be publicly available after the revision process"とあり、ライセンスはない(https://arxiv.org/abs/2311.10793v2)。公開は未検証。
- Hugging Face上のベトナムの道路標識VQA：Anakonkai/vietnamese-traffic-sign-vqa（CC BY-SA 4.0。CC BY-SA 4.0であるKaggleのmaitam/vietnamese-traffic-signsから構築。2,736画像、130,056のQAペア、12の質問タイプyes_no, count, sign_type, color, shape, location, attribute, negative, spatial_rel, count_total, multi_object, context。train 2,193 / val 272 / test 271画像）(https://huggingface.co/datasets/Anakonkai/vietnamese-traffic-sign-vqa/raw/main/README.md)。tungitachi/vietnamese-traffic-sign-vqa-dedup-1k（CC BY 4.0タグ。359画像に対する1,000のQAペア、重複除去済みでリークなし、47標識クラスのうち46、同じKaggleのデータから構築）(https://huggingface.co/datasets/tungitachi/vietnamese-traffic-sign-vqa-dedup-1k/raw/main/README.md)。VuthilinhVn/traffic-signs-VQA（image, question, choices, answer, reasoningを持つ1,595行、ライセンスタグなし）とhanguyen1907/VN_traffic_signs_qa（video_pathとsupport_framesを含む8,844行、ライセンスタグなし）(https://huggingface.co/api/datasets/VuthilinhVn/traffic-signs-VQA?full=true, https://huggingface.co/api/datasets/hanguyen1907/VN_traffic_signs_qa?full=true)。
- dnth/german-traffic-sign-vl-enriched（作成2024-08-08、545行、ライセンスタグなし、READMEは本文なし）：各行はimage_uri（CloudFrontのリンク）、bbox付きの"speed limit 50"のようなobject_labels、image_labelとimage_issuesのフィールドを持つ。545はRoboflowのGTSDBエクスポート（第2.1節）の規模とちょうど一致するので、これは新しいデータセットではなく、VLM用のフィールドを加えたGTSDBの派生物である(https://huggingface.co/api/datasets/dnth/german-traffic-sign-vl-enriched?full=true, https://datasets-server.huggingface.co/rows?dataset=dnth/german-traffic-sign-vl-enriched&config=default&split=train&offset=0&length=1)。
- DriveLM（OpenDriveLab）：DriveLM-nuScenes（30kフレーム、知覚・予測・計画・行動に関するアノテーション付きQAペア360k）とDriveLM-CARLA。アノテータが選ぶ重要オブジェクトには"traffic signals, pedestrians crossing the road, other vehicles"が含まれる。質問は重要オブジェクトに関するもので、標識種別のベンチマークではない。コードはApache-2.0、"The language data is under CC BY-NC-SA 4.0"、nuScenesは独自のライセンスを引き継ぐ。スター1.4k(https://raw.githubusercontent.com/OpenDriveLab/DriveLM/main/README.md, https://img.shields.io/github/stars/OpenDriveLab/DriveLM.json)。標識に関する質問の有無："traffic signals"以上のことは未検証。
- CODA-LM（DLUT）：CODA2022を基にする（trainはCODAのvalから4,884画像、valの4,384画像とtestの500画像はCODAのtestから）。一般知覚タスクには7つの障害物カテゴリ"vehicles, vulnerable road users (VRUs), traffic cones, traffic lights, traffic signs, barriers, and miscellaneous"があり、アノテーションJSONには"traffic signs"のリストがある。GitHubにライセンスの指定はない(https://raw.githubusercontent.com/DLUT-LYZ/CODA-LM/main/README.md, https://img.shields.io/github/license/DLUT-LYZ/CODA-LM.json)。標識に関する質問はオブジェクト記述のレベルで存在する。
- LingoQA（Wayve）：Scenery train 3.5k動画 / 267.8k QAペア、Action train 24.5k動画 / 152.5k QA、Evaluation 100動画 / 1,000 QA、Google Driveのリンク。READMEからはライセンスを特定できない(https://raw.githubusercontent.com/wayveai/LingoQA/main/README.md, https://img.shields.io/github/license/wayveai/LingoQA.json)。標識に特化した質問：未検証。
- NuScenes-QA（AAAI 2024）：知覚結果から生成した、nuScenesの30kフレームに対する460kのQAペア。コードはMIT(https://raw.githubusercontent.com/OpenDriveLab/DriveLM/main/README.md の表, https://raw.githubusercontent.com/qiantianwen/NuScenes-QA/main/README.md, https://img.shields.io/github/license/qiantianwen/NuScenes-QA.json)。nuScenesには道路標識のカテゴリがない（第2.13節）ので、構成上、標識に関する質問は存在しない（推論であり、QAファイルでは未検証）。
- DrivingVQA（EPFL、Zenodo 19353376、MIT、2026-03-31）：フランスの運転免許学科試験形式のサンプル3,931件（画像、選択肢2から4個の多肢選択問題、専門家による解説、ボックス付きの関連エンティティ）。道路標識は試験内容の一部である。画像は826 MB(https://zenodo.org/api/records/19353376)。
- Indian Traffic VQA Dataset（Zenodo 17300841、CC BY 4.0、0.05 GB）とAutoDrive-QA（arXiv 2503.15778）：検索結果で見たのみ、未調査。
- merve/roadsign-labeled-qwenと関連する"roadsign-judged"セット（2026-06）：VLMでラベル付けした道路標識セット（HF APIの検索）、未調査。

---

## 6. 標識に関するVLMの関連評価

- "Mapillary Vistas Validation for Fine-Grained Traffic Signs: A Benchmark Revealing Vision-Language Model Limitations"（Garg and Aich, arXiv 2508.02047、ICCV 2025ワークショップへの投稿）。データはHF上のsparshgarg57/mapillary_traffic_signs（COCOのinstances_default.json、parquet 1つ、ライセンスタグなし）(https://huggingface.co/datasets/sparshgarg57/mapillary_traffic_signs/raw/main/README.md)。Vistasのライセンスを引き継ぐ。
- BDD100K-MUTCDは、人手による品質確認の前に、Gemini Flashをゼロショットで用いて125,731個のクロップを38のMUTCDクラスに事前ラベル付けした（第2.13節）。

---

## 7. 要約表

「画像」は、タスクが分類の場合を除き全体フレームを指す（分類の場合はクロップ）。「再配布可能」＝OpenTSRが自身のリポジトリで画像を再ホストしてよいか（可 / 不可 / 不明確）。「テストラベル」＝公式テスト分割のラベルが公開されているか。

| データセット | 国 | 年 | 画像 | インスタンス | クラス | タスク | ライセンス（確認した記載） | 再配布可能 | テストラベル |
|---|---|---|---|---|---|---|---|---|---|
| GTSRB | DE | 2011 | 51,839クロップ（39,209 / 12,630） | 1画像に1つ | 43 | 分類 | "The data is free to use" + 引用の依頼 | 不明確（正式なライセンスなし、広くミラーされている） | あり（2011-08-23以降） |
| BTSC（BelgiumTS分類） | BE | 2011 | 7,095クロップ（1つの情報源による4,575 / 2,520） | 1画像に1つ | 62 | 分類 | 引用の依頼のみ | 不明確 | あり |
| rMASTIF / FER-MASTIF | HR | 2009-2011 | TS2009 ~6,000、TS2010 ~3,000、TS2011 ~1,000標識 | 同左 | 未検証 | 分類/検出 | "freely used in academic research" | 不可 | あり |
| TSRD | CN | ~2013 | ~6,164クロップ（4,170 / 1,994） | 1画像に1つ | 58 | 分類 | 未検証 | 不明確 | あり |
| CURE-TSR | BE実画像 + 合成 | 2017 | 2,206,106クロップ | 同左 | 14 | 分類（ロバスト性） | Zenodo：CC BY 4.0、GitHub：フォーム + 条件、DataPort：ログイン | 可（Zenodoのコピー） | あり |
| ETSD | EU 6か国 | 2018 | >80,000クロップ | 同左 | 164 | 分類 | "free for academic purposes ... upon request" | 不可 | あり（申請により） |
| LISA TS | US | 2012 | 6,610 | 7,855 | 47 | 検出 | "academic license agreement" | 不可 | あり |
| ATSD | AT | 2023 | 7,454シーン（公開5,511） | 27,521 | 108（使用は60） | 検出 + 分類 | "Permissive license ... free of charge"（条件のPDFは未取得） | 不明確 | あり（公開テスト）+ 非公開の内部セット |
| English roadsigns | UK | 2023 | 55,000クロップ | 同左 | 31 | 分類 | CC BY 4.0 | 可 | n/a（分割なし） |
| Panoramax classified FR / DE | FR / DE | 2024-2026 | 66,000+ / 20,000+クロップ | 同左 | 250+ / 130+ | 分類 | etalab-2.0 / CC BY-SA 4.0 | 可（クレジット表示 / SA条件付き） | n/a |
| German TS Anomaly Detection | DE | 2026 | 実画像369 + 合成 | 同左 | 71 | 異常検知 | CC BY 4.0 | 可 | あり |
| Synset Signset Germany | DE（合成） | 2024 | 105,500（43クラスのサブセットは21,500） | 同左 | 211 | 分類（合成） | CC BY 4.0 | 可 | n/a |
| GTSDB | DE | 2013 | 900（600 / 300） | 1,206（論文）または852（gt.txt） | 43（上位カテゴリは3または4） | 検出 | 記載なし、引用の依頼 | 不明確 | あり（2013-07-15以降） |
| TT100K 2016（+2021） | CN | 2016 / 2021 | 100,000（標識あり10,000。2021年版の解析ではアノテーション付き16,817） | ~30,000（26,349または27,359） | 221（使用可能45）/ 232 | 検出 + 分類 | "CC-BY-NC"、ダウンロードフォーム | 不可 | あり |
| MTSD | 全世界 | 2020 | 完全アノテーション52,453 + 部分アノテーション47,547 | 257,543（+87,358） | 313（v1）/ 400（v2）+ other-sign | 検出 + 分類 | LICENSE.txtによればCC BY-NC-SA | 不可 | なし（非公開） |
| DFG | SI | 2019 | 6,957（+拡張8,775） | 13,239（+ignore 4,359） | 200 | 検出 + 分類 | CC BY-NC-SA 4.0 | 不可 | あり |
| STS | SE | 2011 | >20,000フレーム（~20%にラベル。または3,777） | 3,488（または5,582） | 7（または20） | 検出 + 分類 | Zenodo：CC BY 4.0（2026）、旧ページ：商用利用不可 | 可（Zenodoのコピー、要確認） | あり |
| BTSD（BelgiumTS検出） | BE | 2011-2013 | 未検証 | 8,851（MTSDの表） | 108 | 検出 | 引用の依頼のみ | 不明確 | あり |
| RTSD | RU | 2016 | 59,028（47,639 / 11,389）または179,138 | 105,509または104,358 | 205または156 | 検出 + 分類 | 記載なし | 不明確 | あり |
| Dognellaf/russian-road-signs | RU | 2026 | 43,851 | n/a | 89（信号機を含む） | 検出 | CC BY 4.0 | 可 | n/a |
| CCTSDB2021 | CN | 2021 | 17,856 + 負例500 | ~40,000 | 3 | 検出 | GPL-3.0（リポジトリとZenodo） | 不明確（データにGPL） | あり |
| CTSD | CN | 2016 | 1,100（700 / 400） | n/a | 3 | 検出 | 未検証 | 不明確 | あり |
| KRID | KR | 2025 | 200,000 | 2,676,583（全施設） | 34（標識クラスを複数含む） | 検出 + セグメンテーション | CC BY-NC-SA | 不可 | 未検証 |
| SAIT Traffic Landmark | KR | 2021 | ~150,000 | ~470,000点 | 6 | 点アノテーション | 記載なし、DataPortのサブスクリプション | 不可 | n/a |
| ZOD | EU 14か国 | 2023 | Framesサブセット（件数n/a） | 標識インスタンス446k | 156（6カテゴリ） | 検出（2D/3D）+ 分類 | CC BY-SA 4.0 | 可（SA） | 未検証 |
| CanDOR-99 | CA | 2026 | 95,314 | 103,521 | 99 | 検出 + 分類 | CC BY-SA 4.0 | 可（SA） | n/a |
| aiMotive TL/TS 3D | US + HU | 2024 | ~50,000 | 標識ボックス~550,000 | 記載なし | 3D検出 | CC BY-NC-SA 4.0 | 不可 | n/a |
| Argoverse 2 Sensor | US | 2021 | 1,000 x 15 s | n/a | 30（標識関連5） | 3D検出 | CC BY-NC-SA 4.0 | 不可 | なし |
| Waymo Open Perception | US | 2019+ | 2,030 x 20 s | 3Dボックス12.6M（4クラス） | 4（Signs） | 3D検出 | 非商用 | 不可 | なし |
| BDD100K（+MUTCD拡張） | US | 2018（+2026） | 100,000 | 標識ボックス274,594 | 1（+38 MUTCD） | 検出 | 独自の研究/非営利ライセンス、拡張はDataPort | 不可 | なし（テスト20kは非公開） |
| Cityscapes / Vistas / A2D2 / IDD | DE / 全世界 / DE / IN | 2016-2021 | 5,000+20,000 / 25,000 / 41,277 / 10,003 | n/a | 汎用の標識クラス1つ（Vistas：標識グループ8つ） | セグメンテーション | NC / NC-SA（研究用）/ CC BY-ND 4.0 / 未取得 | 不可 / 不可 / 不可（ND）/ 不明確 | なし / なし / あり / なし |
| STIC-Traffic-Sign（HF） | US | 2026 | 7,204 + 1,564 | 21,282 + 3,374 | 7 / 5 | 検出 | CC BY 4.0（アップロード者） | 不明確（Roboflow由来） | n/a |
| KeenForgeAI/TrafficSigns_YOLO-corrected | 玩具トラック | 2026 | 6,000 | 6,000 | 6 | 検出 | 上流は"none stated" | 不可 | あり |
| 日本：Roboflowのコミュニティセット | JP | 2022-2024 | 536（second-2）、その他は不明 | n/a | 3から16 | 検出 | CC BY 4.0（アップロード者） | 不明確 | n/a |
| 日本：infra-intelligence YOLOX-Tiny | JP | 2026 | 4,585 + 992 + 596（未公開） | 学習ボックス7,808 | 5カテゴリ | 検出（モデルのみ） | 重みはApache-2.0、データは非公開 | モデルは可、データはn/a | 評価JSONのみ |
| Vietnamese TS VQA（Anakonkai） | VN | 2026 | 2,736 | 130,056 QA | 47標識クラス | VQA | CC BY-SA 4.0 | 可（SA） | あり |
| DrivingVQA | FR | 2025 | 3,931 | 同左 | 多肢選択 | VQA | MIT | 可 | あり |
| Hidden in Plain Signs A/B | 複数 | 2026 | 30,226（B） | n/a | 17 / 18 | 検出 | CC BY-NC-SA 4.0（派生） | 不可 | あり |

---

## 8. OpenTSRへの推奨

### 8.1 OpenTSRが再配布できるもの（コードはApache-2.0、データはOpenTSR自身のCCまたはApacheの条件）

- 一次情報源のページで確認できた、問題のない寛容なライセンス：Synset Signset GermanyとそのGTSRBサブセット（CC BY 4.0、著者が求める帰属表示を付ける）、CURE-TSRのZenodoのコピー（CC BY 4.0）、STSのZenodoのコピー（CC BY 4.0、2026年に著者がアップロードしたものだが、CVLのページには旧来の制限が残っているためCVLに確認すること）、English roadsigns（CC BY 4.0）、German Traffic Sign Anomaly Detection Dataset（CC BY 4.0）、EMTD（CC BY 4.0）、Dognellaf/russian-road-signs（CC BY 4.0）、PanoramaxのFRクロップ（etalab-2.0、帰属表示型のライセンスと互換）、Vietnamese TS VQA（CC BY-SA 4.0：再配布可能だが、派生ラベルはSAのままにする必要がある）。注意すれば使える継承（SA）セット：ZODとCanDOR-99（CC BY-SA 4.0）、PanoramaxのDE（CC BY-SA 4.0）。CC BY-SAのデータはラベルを付け直してApache-2.0やCC BYで公開することはできない。使う場合は、SAライセンスの別のデータパッケージとして保持する。
- 実質的には寛容だが正式なライセンスがないもの：GTSRB（"free to use" + 引用）とGTSDB（文言がまったくない）。推奨：INIに書面で許可を求めた後にのみ公式のzipをミラーするか、sid.erda.dkのアーカイブを参照先とし、OpenTSR独自の分割ファイル、ラベル変換、md5の一覧のみを配布する（torchvisionにあるmd5値によりこれを検証できる）。HFミラーのGPL/CCタグはINIが付けたものではなく、依拠すべきではない。

### 8.2 参照のみ、再ホストしない

- TT100K（CC BY-NC + 登録フォーム）、MTSD（CC BY-NC-SA、テストラベル非公開）、DFG（CC BY-NC-SA）、aiMotive（CC BY-NC-SA）、Argoverse 2（CC BY-NC-SA）、Waymo（非商用）、BDD100K（研究/非営利向けの独自ライセンス）、Cityscapes（非商用、配布不可）、Vistas（NC-SAの研究版）、KRID（CC BY-NC-SA）、DDAD（CC BY-NC-SA）、Hidden in Plain Signsの派生物（CC BY-NC-SA）。NCデータで学習したモデルは、商用水準のリリースにとって法的なグレーゾーンである。OpenTSRは公開する重みの学習セットからNCデータを除くか、そのことを明示した「研究専用」の重みを別途公開すべきである。
- 条件が不確かなもの：BelgiumTS/BTSC/BTSD、rMASTIF（"academic research"）、RTSD（ライセンスなし）、TSRD、CTSD、CCTSDB（画像にGPL-3.0）、LISA（"academic license agreement"）、ATSD（論文の文言は寛容だが、条件のPDFは未読）。著者が確認するまでは評価と比較に使い、公開する学習データには含めない。

### 8.3 登録または申請が必要なもの

- ETSD（Googleフォーム、学術用途のみ）、GitHubのフォームまたはIEEE DataPortのログインを経由するCURE-TSR（Zenodoのコピーならこれを回避できる）、TT100K（氏名/メールアドレス/所属のフォーム）、MTSD（Mapillaryのアカウント）、SAIT Traffic LandmarkとBDD100K-MUTCD（IEEE DataPortのサブスクリプション）、AI Hub（韓国の研究者向け）またはGitHub経由のKRID。

### 8.4 道路標識のための固定された公開ベンチマーク（提案）

分類ティア（クロップ済み標識）：
- C1「GTSRB-official」：43クラスの12,630画像からなる公式テストセット。Neural Networks 2012の表と同様にtop-1精度とサブセット別の精度（速度制限、その他の禁止、制限解除、指示、危険、固有（unique））で評価し、数値を99.46%（コミッティ）、99.71%（STN CNN）、98.84%（人間）と比較できるようにする。学習には公式の39,209画像のみを使い、26,640画像のミラーの分割は使わない。使用したzipのmd5を公開する。
- C2「GTSRB-corrupted」：tangankeの7つの劣化分割（各12,630）に加え、CUREの14クラスについてCURE-TSRの実画像のチャレンジレベル1から5。量産TSRが失敗するのはロバスト性の面であるため、劣化ごとの平均精度を報告する。
- C3「Synthetic-to-real」：Synset Signset GermanyのGTSRBサブセットのみで学習し、C1でテストする（参考値はConvNeXt-Sの98.3%）。OpenTSRが計画している生成データのパイプラインが機能するかを判断するのはこのティアである。
- C4「Cross-country crops」（再配布可能なもののみ）：English roadsigns（31クラス）、PanoramaxのFRとDEのクロップ、DognellafのRUのクロップ、CC BY-SAのKaggleセットによるベトナムのクロップ。共有バックボーンがどの程度転移するかを測るため、国ごとにその国の分類体系で評価する。

検出ティア（全体フレーム）：
- D1「GTSDB-full」：900画像、43クラスのラベルが付いた全852（または1,206）ボックス。COCOのmAP@0.5と、IJCNN 2013の上位カテゴリ別AUCで評価する。300画像のテスト分割だけでは小さすぎるため、900画像での5分割交差検証の結果も報告する。
- D2「ZOD-signs」：ZOD Framesの2D標識ボックスと156クラスの分類体系。ライセンスが商用利用を認める唯一の大規模な細粒度検出ベンチマークである。画像ではなく、OpenTSRの分割ファイルと評価スクリプトを公開する。
- D3「NC reference」（比較のための報告のみ、再配布なし）：TT100Kの公式train/test（インスタンスが100を超える45クラス。参考値は81.6、89.9、95.7 mAP）とMTSDの検証セット（5,320画像。参考値は2段階方式の83.4 mAP）。利用者が自ら登録して実行する。
- D4「Japan」（今後構築）：CC BY-SA 4.0の日本のMapillary画像に、OpenTSRのボックスと、規制/指示/警戒/案内/補助にわたる別表第2の標識IDを付ける。都道府県単位で地理的に分割し（infra-intelligenceが地域単位で分割したのと同様）、唯一の公開ベースラインとしてinfra-intelligenceのYOLOX-Tiny（Apache-2.0）を用いる。ロングテールのクラスは、パブリックドメインのCommonsのSVGからレンダリングした合成データで補う。

評価指標とプロトコルに関する注記：IoUの閾値（0.5）と無視ルール（DFGは30 px未満のボックスを無視し、infra-intelligenceは20 px未満の標識板と半分以上遮蔽された標識板を無視する）を固定する。サイズ区分ごとの結果を報告する（ZOD、infra-intelligence、DFGはいずれもそうしている）。テストセットは学習データと地理的に重ならないようにする（ZOD、CanDOR-99、infra-intelligence）。GTSRB/GTSDBのテストセットは学習データと地理的に分離されていない。

---

## 9. 取得できなかったページ

本セッションでブロックされた、または取得に失敗したページ（これらに関する記述は、本文中に示した二次情報源に依拠する）：

- https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/published-archive.html （WebFetchの権限が付与されず、プロキシがcurlを拒否）。ファイル名とmd5値は代わりにtorchvisionのソースから得た。
- https://www.mapillary.com/dataset/trafficsign （WebFetchの権限が付与されず、curlはJavaScriptの外枠のみを返す）。MTSDのライセンス文はHugging Face上のLICENSE.txtのコピーから読んだ。研究版/商用版の条件はMapillaryのブログによる。
- https://pmc.ncbi.nlm.nih.gov/articles/PMC12585093 と https://pmc.ncbi.nlm.nih.gov/articles/PMC10900943/table/tbl5 （reCAPTCHA / 権限）。そのため、最近のGTSRBの最高性能の表は、第1.1節で引用した論文以上には検証していない。
- https://contentportal.asfinag.at/assets/pdf/Bedingungen_VZK%20ASFINAG_20210726_en.pdf （ATSDのライセンス条件。権限が付与されず）。ATSDのライセンスの文言はMDPIの論文から取った。
- http://www.nlpr.ia.ac.cn/pal/trafficdata/recognition.html とdetection.html（TSRDとCTSDの公式ページ。curlがタイムアウト、httpsは証明書の不一致）。
- https://cvrr-nas.ucsd.edu/LISA/lisa-traffic-sign-dataset.html （LISAのダウンロードページ。プロキシが502を返した）。概要ページcvrr.ucsd.eduは取得できた。
- https://doc.bdd100k.com/license.html （プロキシが502を返した）。BDD100KのライセンスはDataset Ninjaの情報から要約した。
- https://www.a2d2.audi/a2d2/en/dataset.html （TLSエラー）。A2D2に関する事実はarXivの論文による。
- https://www.nuscenes.org/terms-of-use と https://www.nuscenes.org/nuimages （JavaScriptのみのページ）。nuImagesのクラス一覧はGitHub上のdevkitのアノテーション指示書から取った。
- https://people.ee.ethz.ch/~timofter/publications/Mathias-IJCNN-2013.pdf （PDFではなくHTMLページを返す）。そのためBTSC/BTSDの分割サイズは二次的な論文による。
- https://pytorch.org/vision/stable/_modules/torchvision/datasets/gtsrb.html （robotsで禁止）。代わりにGitHubのソースファイルを用いた。
- https://www.ncbi.nlm.nih.gov/pmc/articles/PMC12585093/ （リダイレクトのみ）。
- Wikimedia Commons API（commons.wikimedia.org/w/api.php）はHTTP 429 "You are making too many requests"を返した。日本とドイツのSVGのライセンス表記は、Commonsのファイルページのdownload.osmand.netのミラーから読んだ。
- https://raw.githubusercontent.com/csust7zhangjm/CCTSDB/master/README.md と .../main/README.md （404）。元のCCTSDBの統計は未検証。
- https://raw.githubusercontent.com/mapillary/mapillary_vistas/master/LICENSE （404）。VistasのライセンスはDataset NinjaとICCVの論文による。
- http://export.arxiv.org/api/query （プロキシ経由では空の応答）。arXivのアブストラクトは代わりにarxiv.org/absのページとPDFから読んだ。
- Kaggleのデータセットページ（meowmeowmeowmeowmeow/gtsrb-german-traffic-sign、maitam/vietnamese-traffic-signs、tamasmatuszka/aimotive-3d-traffic-light-and-sign-dataset）は取得しなかった。それらのライセンスは、それらを引用しているHugging Faceのカードから引いた。
- Roboflowのプロジェクトtest-u58bl/japan-roadsign-ver.2とtest-u58bl/japan-roadsign-secondは取得しなかった（取得済みプロジェクトの兄弟）。
- TUROS-TSとMoroccan Traffic Sign RecognitionのIEEE DataPortページ、Kakao Mobilityのデータセットの記事、hyper.aiのデータセットページ37148 / 41354 / 16402、Nexdataのニュースページは取得しなかった。
