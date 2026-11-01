# 図解データセットカタログ（道路標識）

> 本文書は英語版 [docs/dataset_catalog.md](../dataset_catalog.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

`scripts/make_dataset_catalog.py`は、代表的な標識データセットのすべてのクラスまたはカテゴリの定義を、固定したシードで抽出した実際のサンプルと並べて示す日本語のカタログ（HTMLとPDF）を生成する。このカタログを作るのは、`datasets.md`の調査表ではデータの見た目を示せないからである：標識がどれほど小さいか、ラベルの粒度がデータセット間でどう異なるか、同じ種類の標識が国によってどう変わるか。生成したカタログは`runs/catalog/`以下に置く（コミットしない）。2026-11-01のビルドはPDFとしてメンテナに渡した。

## セクション

| セクション | 使用データ | 表示内容 |
|---|---|---|
| GTSRB | `data/gtsrb_hf`（公式分割）とSynset Signset GermanyのGTSRBサブセット | 全43クラスについてID、日本語名と英語名、GTSDBのカテゴリ、train/testの件数、サイズの中央値、テストサンプル6枚と合成版2枚。サイズのヒストグラム。カテゴリの定義 |
| GTSDB | `data/gtsdb_hf`（学習フレーム600枚、gt.txt） | アノテーション付きフレーム6枚、ボックスサイズのヒストグラム、コンペティションのカテゴリごとに14個のクロップ、クラスごとの件数 |
| Panoramax classified DE | `data/panoramax_de/val` | 全134クラスについて日本語訳、件数、ドイツ以外の画像による補充、各6サンプル |
| Synset Signset Germany | `data/synset_signset/cycles/validation.parquet` | 画像 / マスク / セグメンテーションの3つ組と、レンダリングのメタデータの分布 |
| TT100K | `data/tt100k_hf`（parquetシャード3個、`categories.csv`） | コード系統の凡例、フレーム4枚、ボックスサイズのヒストグラム、シャードに存在するすべてのカテゴリについて最大3個のクロップ |
| MTSD | `data/mtsd_official`（公式アノテーションJSONと画像のサンプル。後述） | 分類体系のグループ、フレーム4枚、サイズのヒストグラム、ボックス属性の件数、サンプルに存在するラベルごとに1個のクロップ |
| CURE-TSR | `data/cure_tsr/CURE-TSR.zip`（展開せずにそのまま読み込む） | real（実写）とunreal（合成）のサンプルを含む14種類の標識、レベル1、3、5での12種類のチャレンジタイプ |
| 日本（道路標識令） | `data/japan/mlit`（MLITの標識一覧から切り出したタイル） | 定義付きの五つの公式カテゴリと、MLITの一覧にある221個のピクトグラムすべて。公開データセットが存在しないため、このセクションが日本トラックのクラス一覧となる |

## ビルド方法

```bash
make gtsrb gtsdb replicate-panoramax   # GTSRB、GTSDB、Panoramaxの検証用クロップ（再現用データ）
make catalog-data                      # Synset、TT100Kシャード、MTSDサンプル、CURE-TSRのzip、MLITの一覧とタイル
make catalog                           # runs/catalog/catalog.htmlとcatalog.pdf（PDFの生成にはplaywright + Chromiumを使用）
```

サンプリングには`random.Random("<seed>:<dataset>")`を用いるため、同じシードとデータファイルからは同じ画像が得られる。検出データのサンプルは、正解ボックスの周囲に15%のマージンを付けてクロップし、64から72 pxにスケーリングする。フレームは幅820 pxにスケーリングし、ボックスはカテゴリごとに色分けする。すべての画像はdata URIとして埋め込むため、HTMLは単体で完結する（約20 MB）。

## 出典に関する注記

- TT100KとMTSDは非商用のデータセットである。それらのサンプルは研究上の引用としてカタログに掲載しているものであり、OpenTSRが再配布するものには一切含まれない。
- MTSDのparquetミラーである`ThankGod/mapillary_traffic_sign_dataset*`は整数のカテゴリを持つが、その`id2labels.txt`は画像と一致しない（目視と、クリーニング済みテストシャードのすべてのボックスを元のミラーと照合することで確認した：IDは両者で同一であり、どちらもラベルファイルと食い違う）。そのため、カタログは公式のアノテーションJSONファイルからサンプリングしている。このファイルは、`NgTMDuc/cl_dataset`ミラーが学習画像とともにホストしている（`scripts/sample_mtsd_official.py`）。
- クラウドセッションからは、日本のピクトグラムにWikimedia Commonsを利用できなかった（サムネイルサービスが自動化クライアントに429を返す）。いずれにせよ、MLITの一覧（`ichiran.pdf`、A3で1ページ）が公式の出典である。`scripts/cut_mlit_sign_list.py`は、`pdftotext -bbox-layout`で得たラベル位置を用いてこれをタイルに切り分ける。ページレイアウトの都合で、いくつかの案内標識のタイル（路線番号が積み重なったもの）はまだ一部が欠けている。
- PanoramaxとTT100Kのクラスコードに付けた日本語訳はカタログ作成者による翻訳である。StVOとGB 5768の番号は目安である。
