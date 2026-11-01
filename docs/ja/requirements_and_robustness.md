# 「商用水準」の道路標識認識とは何か、およびロバスト性とセキュリティに関する資料

> 本文書は英語版 [docs/requirements_and_robustness.md](../requirements_and_robustness.md) の日本語訳である（原文 commit 52837a8 時点）。内容に食い違いがある場合は英語版を正とする。

2026-10-31に公開中のWebページから作成した調査（claude.aiリサーチセッション、pingfoo/OpenTSR）。事実に関する記述にはすべて、それを検証したURLを付す。「未検証」または「不明」は、本セッションでその事実を確認できなかったことを意味する。情報源間で値が食い違う場合は両方の値を示す。データセット全般、古典的なオープンソース実装、VLMによる手法は、同じディレクトリの`datasets.md`、`oss_implementations.md`、`vlm_and_foundation.md`で扱う。本ファイルでデータセットに触れるのは、それがロバスト性ベンチマークまたは劣化ベンチマークである場合に限る（B節）。D.1節はOpenTSRの目標仕様案、D.2節は攻撃・防御コードベースの一覧表、D.3節は取得できなかったページの一覧である。

表記規則：
- 「ISA」= Intelligent Speed Assistance（インテリジェント速度アシスタンス、EUの型式認証対象機能）、「SLIF」= speed limit information function（速度制限情報機能）、「TP_D」= Regulation (EU) 2021/1958のAnnex Iで定義されるISAの真陽性距離比率。
- 「明示標識」（explicit sign）= 数値の速度制限標識、「暗黙標識」（implicit sign）= 数値なしで制限を示す標識（市街地入口、高速道路の始点、制限解除、ゾーン標識）。
- 「物理」= 印刷物、投影、または照明された物体を実カメラの前に置いて攻撃を実証したもの、「デジタル」= 画素のみの攻撃。
- GitHubリポジトリのスター数とライセンスは2026-10-31に取得したshields.ioバッジによるもので、概数である。「識別不可」（not identifiable）は、分類できないLICENSEファイルに対するGitHub自身の表現である。

## 0. 要点

1. 「商用水準」のTSRについて、拘束力があり定量的な定義として見つかったのはEUのISA委任規則のみである。400 kmの実路走行（市街地、郊外、高速道路がそれぞれ25%以上、暗所が15%以上）において、システムが認識した速度制限は全体で距離の90%以上、各道路種別で80%以上正しくなければならず、新しい制限は標識通過後2.0秒以内（20 km/h未満では10 m以内）に判定されなければならない。対象は明示的な数値標識、暗黙標識（市街地境界、高速道路、ゾーン、制限解除）、可変表示標識であり、条件付き補助標識は無視して"most common condition"（最も一般的な条件）を優先してよい（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。新型車には2022年7月6日から、すべての新規登録車には2024年7月7日から適用される（<https://www.legislation.gov.uk/eur/2019/2144/annex/II/2020-12-31>）。
2. Euro NCAPは法規より踏み込んでいる。2023年以降、条件付き（雨、雪、時間、距離、車線矢印、車両区分）、暗黙（高速道路、市街地入口/出口、住宅地）、動的（道路工事を含む）の速度制限を採点しており（<https://cdn.euroncap.com/media/77301/euro-ncap-assessment-protocol-sa-safe-driving-v102.pdf>）、2026年からは3〜5か国にわたる2000 kmの走行でSLIFの精度を、距離ベースKPI（> 80%）とイベントベースKPIの両方で測定する（<https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Safe_Driving_Vehicle_Assistance_v1_2_b6fb486fa6.pdf>）。Thatchamによる独立したイベントベースの試験では、型式認証済みの車両が距離ベースで91.3%だったにもかかわらず、イベントベースでは74.3%にとどまった（<https://www.autocar.co.uk/node/375029>）。このため、OpenTSRの主要指標はイベントベースの指標とすべきである。
3. 日本にはTSRやISAの要件がない。JNCAP 2025には標識に関する項目がなく（<https://www.mlit.go.jp/jidosha/anzen/02assessment/data/pamphlet_car_r8.pdf>）、MLITのASV推進検討会は2019年12月にISAの「基本設計書」を公表しただけである（<https://www1.mlit.go.jp:8088/report/press/content/001320703.pdf>）。日本の標識体系（標識令、2026-10-31取得のe-Gov XML）は5カテゴリ163種類（案内41、警戒26、規制59、指示14、補助23）を定めており、A/B/C様式を数えると237の番号付き派生となる。図柄は著作権法第13条により著作権の対象外であり、このためWikimedia CommonsがSVGとして掲載している（<https://download.osmand.net/wiki/File:Japan_road_sign_302.svg>）。
4. Wang et al.（UC Irvine）のNDSS 2025の研究によれば、学術的な物理攻撃は量産TSRにはほとんど転移しない。2023-2024年型の4車種に対する成功率は6.67%で、元論文の51.6%を大きく下回る。主な理由は、商用システムが検出した標識を最大60秒間記憶に保持する（"spatial memorization"、空間的記憶）ことであり、これにより隠蔽攻撃は難しく、出現攻撃は容易になる（<https://arxiv.org/pdf/2409.09860>）。OpenTSRは同様の時間的持続ロジックを提供し、攻撃をフレーム単位ではなくシステムレベル（SysHA/SysAA）で評価すべきである。
5. OpenTSRがすぐに再利用できる公開ロバスト性ツール：CURE-TSR（14クラス、12チャレンジ種別×5段階、ZenodoでCC BY 4.0、<https://zenodo.org/record/3903066>）、Hugging Face上のGTSRBに対するImageNet-C形式の7種類の劣化分割（tanganke/gtsrb、ライセンスタグなし、<https://huggingface.co/api/datasets/tanganke/gtsrb?full=true>）、REAP（現実的なパッチ評価のための幾何変換と照明変換を備えた14,000枚の実Mapillary標識、ICCV 2023、<https://arxiv.org/abs/2212.05680>）、ShadowAttack（CVPR 2022、MIT、敵対的学習済みのGTSRB/LISAモデルを含む）、SLAP（USENIX Security 2021、GPL-3.0、GTSRB-CNNとLISA-CNNおよびその敵対的学習済み版を含む）、そして公開された重みのバックドア検査用のNeural Cleanse（S&P 2019、MIT、感染済みGTSRBモデルを同梱）。

---

## A. 法規および消費者テストの要件

### A1. EU General Safety Regulation（一般安全規則）、Regulation (EU) 2019/2144

- Article 6(2)はインテリジェント速度アシスタンスの最低要件を定める。（a）アクセル操作系またはその他の適切なフィードバックを通じて、速度制限を超過していることをドライバーに認識させる。（b）システムをオフにできなければならず、その場合も速度制限情報は表示してよく、システムはマスタースイッチを起動するたびに通常動作に戻る。（c）フィードバックは道路標識と信号、インフラからの信号、電子地図データのいずれか、またはその両方に基づく。（d）"it shall not affect the possibility, for the drivers, of exceeding the system's prompted vehicle speed"（システムが示す車速をドライバーが超える可能性に影響を与えてはならない）。（e）性能目標は実走行条件における誤り率の回避または最小化を目指す（<https://legislation.gov.uk/eur/2019/2144/chapter/II/adopted/data.pdf>。逐語引用は（d）のみで、（a）、（b）、（c）、（e）は取得ツールによる条文の言い換えである）。
- Article 6(6)は、詳細な試験手順と技術要件を定める委任法令を欧州委員会が採択し、Annex IIの日付の少なくとも15か月前に公表することを求めている（<https://legislation.gov.uk/eur/2019/2144/chapter/II/adopted/data.pdf>）。
- Article 19：本規則は"shall apply from 6 July 2022"とされる（<https://legislation.gov.uk/eur/2019/2144/article/19/2020-12-31/data.pdf>）。
- Annex IIの行D8 "Intelligent speed assistance"はM1、M2、M3、N1、N2、N3に適用され、注記Bとして"Date for refusal to grant EU type-approval: 6 July 2022"および"Date for the prohibition of the registration of vehicles ... 7 July 2024"が付されている（<https://www.legislation.gov.uk/eur/2019/2144/annex/II/2020-12-31>。EUR-LexのHTMLページが本文を返さなかったため、EU法文の英国ミラーを使用した。D.3参照）。

### A2. Delegated Regulation (EU) 2021/1958（ISA試験手順）

出典：官報（Official Journal）のPDF（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）および2023-09-21の統合版（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf>）。取得ツールは各PDFの先頭約94,000文字を読み込んだ。これはArticle 1〜6、Annex Iの全体、Annex IIの冒頭（説明注記とベルギー）に相当し、Annex IIの残りの加盟国の表は読んでいない（D.3参照）。統合版には改正法令として2023年6月26日のDelegated Regulation (EU) 2023/1675（OJ L 216, 1.9.2023）が1件記載されているが、その改正内容は本セッションでは検証できなかった。

日付と適用範囲
- Article 6：2022年7月6日から適用。Article 4(2)：認可当局は2024年7月7日に、その後2年間は6か月ごとに、集計データを欧州委員会に報告する。recital 9：フィードバック手法の評価は2025年12月31日までに完了する（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。

Annex I、point 1の定義（番号は取得ツールの報告による）
- 1.3 "speed limit information function (SLIF)"（速度制限情報機能）：速度制限判定システムとHMIの組み合わせ。
- 1.6 "perceived speed limit"（認識速度制限）："the applicable speed limit as obtained by the speed limit determination system"（速度制限判定システムが取得した適用速度制限）。
- 1.9 "catalogue of road signs"（道路標識カタログ）：標識種別および可変表示標識種別の国別・地域別の変種であり、Annex IIに定められる（Article 3）。
- 1.10 "applicable road sign"（適用対象の道路標識）：少なくとも1車線に適用されるカタログ上の標識。通常の標識と可変表示標識を含み、路面標示は除く。
- 1.11 "explicit numerical speed limit sign"（明示的数値速度制限標識）："an applicable road sign which shows a temporary or permanent numerical value"（一時的または恒久的な数値を示す適用対象の道路標識）。
- 1.13 "implicit speed limit sign"（暗黙速度制限標識）：数値を示さないもの（暗黙・非数値）、または"shows a strikethrough numerical value"（取り消し線付きの数値を示す）もの（暗黙・数値）。
- 1.14 "national speed limit"（国の法定速度制限）：加盟国における道路種別ごとの既定の最高速度。
- "conditional speed limit"（条件付き速度制限）や"supplementary plate"（補助標識）の定義はない。最も近い概念はpoint 3.4.2.3.2の"special variable conditions"（特殊な可変条件：トレーラー、時間帯、天候、乗員）であり、point 3.4.1.2.1は"Display of additional sub-sign information is permitted in all cases"と定めている（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>、<https://lexaris.de/book/version/documentflat/head/2056920>）。

認識すべき対象（Annex I、point 3.4.2）
- 3.4.2.1 運行国：自動検出またはドライバーによる選択。地域対応が限定的な車両には"ISA functionality is limited to country or region of operation"という注記が付される。
- 3.4.2.2 明示的数値標識：製造者が申告した基準点が標識を通過してから"at the latest 2.0 seconds"（遅くとも2.0秒後）までに制限を判定しなければならない。20 km/h未満では、その基準点から後方10 mまでの間に判定すればよい。標識の意匠、設置、状態は国の基準に適合していなければならず、標識は少なくとも1.0秒間遮られていてはならず、視界を損なう天候（霧、豪雨、雪）は除外される（3.4.2.2.3(d)）。
- 3.4.2.3 暗黙標識：システムはAnnex IIカタログ内のすべての適用対象標識から、同じ2.0秒/10 mのタイミングで制限を判定しなければならない。"need not account for special variable conditions (such as trailer status or time of day)"（トレーラーの有無や時間帯などの特殊な可変条件を考慮する必要はない）とされ、それ以外の場合は想定される最も一般的な条件を既定とする（3.4.2.3.2）。
- 3.4.2.5 実環境での信頼性：TP_Dは総距離の90%以上、3つの道路種別それぞれで80%以上。信頼性は少なくとも14年間維持すること。地図ベースのデータ更新は製造後7年まで、少なくとも年1回、無償で行うこと（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf>）。
- 3.2.3：情報と警告を発する際、認識制限値に対して最大3.0%の許容誤差が認められる（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。

試験手順（Annex I、point 4）
- 4.2 暗黙標識の試験：技術機関が選定した少なくとも3種類の異なる暗黙標識を用いる。その他のカタログ標識への適合は製造者が文書で示す。公道では速度計の速度は標識値より最大20%低くてもよく、テストトラックでは少なくとも10%高くする。合格基準は、期待されるフィードバック値が2.0秒以内（20 km/h未満では10 m以内）に表示されることである。適合確認は実環境試験と組み合わせてもよい（4.2.4.2）。
- 4.3 実環境試験：製造者、技術機関、認可当局の間で合意したEU域内の公道で行う。連続した1周回。市街地、非市街地、高速道路がそれぞれ経路の25%以上。暗所が距離の15%以上。距離は400 kmで、最後の50 kmでTP_Dの変動が±5.0%以内であれば300 km以降で早期終了してよい。TP_D = d_correct / d_total x 100%であり、d_totalは標識または国の法定制限が適用された距離である。複数の加盟国での走行を求める要件は見つからなかった。
- 4.4 テストトラックでの速度制限警告機能：初期の認識制限は試験制限より38%以上高くする。制限超過の速度帯は1-8、11-18、21-28、31-38%の4つ。視覚と音響/触覚を組み合わせた警告は、判定時間にそれぞれ6.0、5.0、4.0、3.0秒を加えた時間内に開始しなければならない。視覚警告は1.5秒以内。
- 4.5 速度制御機能：安定速度は市街地45-50 km/h、都市間75-80、高速道路125-130。制限設定後1.5秒以内に介入する（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。
- Point 5によるTP_D計算からの除外：一部が遮られた標識や向きが誤った標識（5.3.1）、欠落した標識や設置位置が曖昧な標識（5.3.2）、車両区分、時間帯、天候に関する矛盾または曖昧な情報（5.3.3）、静止状態での誤検出を引き起こす非常に本物らしい適用対象外の標識（5.3.4）、試験前12か月以内に変更または追加された標識（5.3.5）。該当する距離はd_totalとd_correctの両方から除外される（5.3.6）（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf>）。

Annex IIのカタログ（読み取れた範囲）
- 説明注記：「N」= 道路区分に対する国の法定制限。「V」= 可変表示標識で、国の表にある任意の明示標識または暗黙標識を表示でき、"refresh rate of the signs shall be at least 1 000 Hz"とされる。「S」= points 3.5.6または3.6.3によりフィードバックを停止。市街地境界の入口標識と出口標識は"share recognisable characteristics but may vary in shape, size and wording"とされ、システムはこれらに対応しなければならない。3.5 t未満のM2は、特に記載がない限りM1に従う（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。
- ベルギー（読み取れた唯一の表）：明示的数値標識C43（30〜120 km/h）について、車両区分M1、M2、M3、N1、N2、N3ごとに期待されるフィードバックが定められている。ほかに暗黙・数値標識C45（「N」）、暗黙・非数値標識、ゾーンF4a/F4b（30 km/h）、ZC43/ZC45（50 km/hと70 km/hのゾーン）、交通抑制区域F12a/F12b（20 km/h）、F113a/F113b（30 km/h）、高速道路F5（120 km/h）とF7（終点、「N」）、市街地境界F1（50 km/h）があり、自動車専用道路（expressway）は"None"である。大型車の行は7.5 t以下と7.5 t超で異なる（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。
- ドイツおよびその他25の加盟国：本セッションでは読んでいない（未検証）。ベルギーの構成から導かれる標識クラスが、OpenTSRがEU向けにサポートすべきクラスである。すなわち明示的数値、制限解除（取り消し線付き）、ゾーン始点/終点、交通静穏化区域始点/終点、高速道路始点/終点、自動車専用道路始点/終点、市街地入口/出口、国の法定制限へのフォールバックである。

### A3. Euro NCAP Speed Assist Systems

2023〜2025年スキーム：Assessment Protocol Safety Assist, Safe Driving v10.2（2023年6月、2023年実施）、第4節「Speed Assist Systems」（<https://cdn.euroncap.com/media/77301/euro-ncap-assessment-protocol-sa-safe-driving-v102.pdf>）：
- 4.1は法規上の背景を記す。ISAは新型車には2022年7月から、全車両には2024年7月から義務化される。4.5.3は2026年に導入する速度計の要件-3/+0 km/hを予告している。
- 4.4.1 SLIFの一般要件：制限は初期化中を除き常に、ドライバーの直接視野内（メーターパネルまたはヘッドアップディスプレイ）に道路標識として表示する。明示的な条件付き制限については、適用される制限を表示するか、条件なしの制限の横に条件付き制限が存在することを示さなければならない。警告機能は既定でONとする。
- 4.4 入力ソース："using all relevant system inputs, for example camera input and electronic map based or a combination of both"であるSLIFはすべて対象となるため、カメラのみ、地図のみ、融合のいずれも認められる。
- 4.4.2 高度な機能については、オーストリア、フランス、ドイツ、イタリア、ルクセンブルク、オランダ、スペイン、スウェーデン、英国（2024年からはノルウェーも）における典型的な公道走行の80%以上、またはApplication Areaの少なくとも半分で機能が動作することの証拠が求められる。
- 4.4.2.1 採点対象の標識カテゴリ（内部20点満点）：雨/路面湿潤（暗黙を含む、2点、正しい制限を表示）、雪/凍結（2、警告のみ、無関係なら無視）、時間/季節（3）、距離（for/in）（1）、矢印/車線関連性（1、表示するか無関係なら無視）、その他の車両区分または重量区分（1）、暗黙の高速道路（highway/motorway）（2）、暗黙の市街地入口/出口（3）、暗黙の住宅地ゾーン（2）、道路工事を含む動的標識（3）。明示的数値標識は採点カテゴリではなくSLIFの基本要件である。トレーラーとキャンピングトレーラーの標識は個別には挙げられていない（車両/重量区分に含まれる）。
- 4.4.2.2 道路特徴（内部10点）：カーブ、ラウンドアバウト、優先権のない交差点、信号機、一時停止標識、譲れ標識、進入禁止標識。4.4.2.3 局所的な危険（内部10点）。4.4.2.4 システム更新：四半期ごとの更新で5点、継続的（日次/週次）な更新で10点。
- 4.6 配点：SASはSafety Assistボックス内で合計3.00点。SLIF 1.50（基本0.50、高度な制限0.25、道路特徴0.25、局所的な危険0.25、システム更新0.25）、Speed Control Function 1.50（SLF 0.50、既定ONでないISL 1.00、iACCまたは既定ONのISL 1.50）。
- 本プロトコルにおけるタイミング：iACCは制限変更から5秒以内に、調整後の設定速度を採用または提示しなければならない（4.5.2.2）。速度制限の実施は30秒以内（4.5.2.1）。2秒/10 mの標識表示ルールはここには現れない。
- v10.3（<https://www.euroncap.com/media/79883/euro-ncap-assessment-protocol-sa-safe-driving-v103.pdf>）は取得できなかった（クライアントエラー）。代わりにv10.2を使用した。

2026年スキーム：Protocol Safe Driving, Vehicle Assistance（版表記が不一致：表紙は"Version 1.2 July 2026"、タイトル欄は"Version 1.1.0 July 2026"。2026年1月実施）（<https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Safe_Driving_Vehicle_Assistance_v1_2_b6fb486fa6.pdf>。v1.1は<https://www.euroncap.com/media/91705/euro-ncap-protocol-safe-driving-vehicle-assistance-v11.pdf>）：
- Speed Assistanceは合計20点：SLIF 12（精度4、高度な制限3、局所的な危険3、システム更新2）、Speed Control Function 8（ISL 5、iACC 8。満点には速度計精度-3/+0 km/hが必要で、-5/+0では半減）。
- 1.2.1 SLIFの精度："KPIDistance = Dcorrect / Dtotal"で、Dcorrectは"Total distance with correct speed limit displayed (km), applicable to ALL speed limit elements"である。2点の閾値は"> 80%*"。"KPIEvent = Ecorrect / Etotal"で、Ecorrectは"Total number of correctly identified events of the claimed speed limit elements"である。2点の閾値は"> 80 -10%*"と印字されている（v1.1とv1.2のどちらの抽出結果でも文字が崩れている。70〜80%の範囲の閾値と解釈し、引用前にPDFで確認すること）。脚注："Assuming perfect ground truth"。
- 経路："a total length of at least 2000 km, covering between three and five countries within the Euro NCAP Application Area"、各国で距離の20%以上、"A minimum of 10% of the total mileage shall be conducted during nighttime"、各国で市街地5-10%、都市間20-30%、高速道路60%以上。試験機関が実路で評価する。
- 1.2.1 タイミング：表示速度が20 km/hを超える場合、SLIFの表示変更は制限変更から2秒以内に行わなければならない。20 km/h以下では2秒以内または10 m以内（v1.2の抽出結果による。v1.1の抽出結果ではこのルールが現れなかったため、PDFで確認すること）。
- 1.2.2 高度な制限（3点）：条件付き2（雨/路面湿潤、雪/凍結、時間/季節、距離（for/in）、車線に関連する矢印と関連しない矢印、キャンピングトレーラーを含む車両区分）、暗黙0.5（高速道路（motorway/highway）、市街地入口/出口、住宅地ゾーン）、動的0.5（道路工事を含む）。適格条件：オーストリア、フランス、ドイツ、イタリア、ルクセンブルク、オランダ、スペイン、スウェーデン、英国、ノルウェーにおける典型的な走行の80%以上、またはApplication Area諸国の少なくとも半分で高度な情報を提供すること。製造者は約400 kmの実路走行の証拠を含む書類SD-301を提出する。実路評価が実施できない場合はHiLまたはテストトラックのデータも認められる。
- 地図との融合は必須ではなく、カメラ、地図、またはその両方が対象となる。頭上ガントリーや補助標識はそれとして名指しされていない。
- 雨は条件付き標識のカテゴリとしてのみ登場する。どちらのEuro NCAPプロトコルにもISA規則にも、雨天時の独立した試験条件はない（ISA規則は視界を損なう天候を除外している）。

### A4. 量産ISAの独立した実環境検証

- Thatcham Research（英国）は3つのISAシステム（新車に搭載されるシステムの約10%）をイベントベースの指標で試験した。MG ZS（2024）は距離ベースで91.3%と"just above the EU's 90% threshold"だったが、イベントベースでは74.3%（制限変更の約4回に1回が誤り）であった。BMW i5は距離ベースで98.39%、イベントベースで90.3%であった。3台すべてが5、10、15、100 mphなど英国では法的に存在しない制限を表示し、ThatchamはこれがACC作動時に意図しない制動や加速を引き起こしうると述べている。記事には試験日や経路の記載はなく、EUの認証試験は250マイル、Euro NCAPの試験は60マイルであるとしている（<https://www.autocar.co.uk/node/375029>。What Car?版の<https://www.whatcar.com/news/intelligent-speed-assist-systems-fail-to-read-road-signs-25-of-the-time-research-finds/n29166>は取得できなかった）。
- Euro NCAP Assisted Driving評価（2024年10月）：BMW i5はAssistance Competence 76%、Safety Backup 90。Mercedes C-Classは85 / 97。Volvo EC40は66 / 78（"good"）。VW ID.7は70 / 80（"good"）。BYD Atto 3は55 / 35（"not recommended"）で、その速度アシスタンスは"not interpreting road signs correctly"とされた（<https://paultan.org/2024/10/24/euro-ncap-grades-5-cars-for-assisted-driving-bmw-i5-mercedes-c-class-volvo-ec40-vw-id-7-byd-atto-3/>。二次情報源であり、Euro NCAPのデータシートは取得していない）。

### A5. UNECE

- TSRやISAに関するUN Regulationは見つからなかった。ISAの情報文書テンプレートを含むEU実施規則案は、速度関連の項目についてはUN Regulation No 39（速度計および走行距離計）のみを、イベントデータレコーダーについてはUN R160を参照しており、ISAや速度制限情報に関するUNECEの法令は挙げていない（<https://members.wto.org/crnattachments/2024/TBT/EEC/24_01641_01_e.pdf>）。UN R157（ALKS）やR89（速度制限装置）が標識読み取りの要件を課しているという主張は、本セッションでは検証していない。

### A6. 米国

- 連邦レベルのTSRやISAの要件はない。NTSB（2023年11月14日の理事会、2022年1月29日のNorth Las Vegasでの事故、死者9名）は、NHTSAに対して"Require ISA systems that, at a minimum, warn a driver a vehicle is speeding"こと、およびNCAPを通じてISAを奨励すること（2017年の勧告の再確認）を、また17の製造者に対して"Install ISA in all new passenger vehicles that, at a minimum, warns drivers when a vehicle is speeding"ことを勧告した。発表資料はISAを、GPSと速度制限データベースに車載カメラも併用するものと説明しており、性能要件は示していない（<https://www-s.ntsb.gov/news/press-releases/Pages/NR20231114.aspx>）。
- 2024年4月3日のNHTSAの回答：規則制定の約束はない。FY2024に"assess the capabilities and limitations of technologies to accurately determine the speed limit across a wide range of roadway types"ための技術調査を行い、NCAPの最終決定通知を2024年に予定し、Guideline 19を2025年に更新する（<https://www.nhtsa.gov/document/nhtsa-ntsb-response-april-2024-intelligent-speed-assistance-isa-interlock-speeding-ncap>）。2024年のNCAP決定でISAが追加されたかどうかは検証していない。
- カリフォルニア州のSB 961（Scott Wiener州上院議員）は、2030年から、GPSと州の速度制限リストを用いて制限を10 mph以上超えた際に1回の視覚・音声警告を義務付ける内容であったが、連邦法の優越とNHTSAが進めるISA評価を理由に2024年9月28日に拒否権が行使された（<https://timesofsandiego.com/politics/2024/09/28/governor-vetoes-senate-bill-calling-for-new-cars-to-alert-drivers-when-speeding/>）。
- IIHS "Intelligent Speed Assistance Test and Rating Protocol"、DRAFT Version I、2026年6月：閉鎖コース、昼間のみ（2,000 lux超）、mph表記の数値標識のみ（24 x 30 in、MUTCD 11th ed. Table 2B-1に従い車線端から約12 ft、高さ5 ft）。Assessment 2では500 ft以上離れた2枚の標識を高い制限から順に用い、3回走行する。Assessments 3と4では25 mphと45 mphの制限に対して5-9、10-14、15+ mph超過の速度帯を用いる。"the rating system will be determined after an initial round of testing"とされる。夜間、暗黙標識、条件付き標識は含まない（<https://www.iihs.org/media/4e417587-ab67-489c-a3bb-7fe1dd87fe1b/COWTLw/Ratings/Protocols/future%20programs/Intelligent_speed_assistance_protocol.pdf>）。
- MUTCD 11th Edition：2023年12月19日にFederal RegisterでFinal Ruleが公示され、2024年1月18日に発効した。各州は"within two years from the effective date"に採用しなければならない（<https://mutcd.fhwa.dot.gov/mutcd_news.htm>）。

### A7. 中国

- C-NCAP 2024（C-NCAP MANAGEMENT REGULATION 2024 EDITION）には"Appendix L Active safety ADAS test protocol"がある（<https://www.c-ncap.org.cn/cncap-guicheng-detail?id=8>）が、PDF本体（<https://cncap.obs.cn-north-4.myhuaweicloud.com/cms/files/909314048149975040.pdf>）は取得できなかった。C-NCAP 2024に関するLeddarTechのホワイトペーパーは、アクティブセーフティ項目として"Intelligent speed limit system (ISLS)"（速度が規定に適合しない場合に警告する）と、"road feature recognition (RFR)"の下位項目としての道路標識認識を挙げているが、配点は示していない（<https://leddartech.com/app/uploads/dlm_uploads/2024/06/White-Paper_A-Comprehensive-Overview-of-C-NCAP-2024-Regulations_V1.0_EN-5.pdf>。二次情報源であり、配点の重みは未検証）。
- C-ICAP 2024（intelligent connected vehicle index）の付属書は運転支援、ナビゲーション支援、駐車、コックピット、プライバシーを対象としており、ページの本文にTSRの項目は見当たらない（<https://www.c-ncap.org.cn/cncap-guicheng-detail?id=15>）。i-VISTAのTSR試験仕様：未検証（一次情報のページが見つからなかった）。
- 標識規格：GB 5768.2-2022 "Road traffic signs and markings, Part 2: Road traffic signs"。SAMRとSACが2022年3月15日に発行し、2022年10月1日に施行された。強制規格は中華人民共和国著作権法第5条により著作権の対象外となる"technical standards with legal natures"であるため、Wikimedia Commonsでは規格本文がパブリックドメインとして扱われている（National Copyright Administration 权司[1999]第50号）（<https://download.osmand.net/wiki/File:GB_5768.2-2022.pdf>）。

### A8. 日本

要件
- JNCAP（自動車アセスメント）2025の予防安全項目：衝突被害軽減ブレーキ（対歩行者の昼間と夜間（街灯ありおよび街灯なし）、対自転車、右折時の対向車、横断歩行者がいる場合の右左折）、車線逸脱抑制、高機能前照灯、ペダル踏み間違い時加速抑制。満点は予防安全85.8、衝突安全100、事故自動緊急通報8、合計193.8。標識認識、速度制限、ISAの項目はない（<https://www.mlit.go.jp/jidosha/anzen/02assessment/data/pamphlet_car_r8.pdf>。ファイル名はr8だが、パンフレットには2025と記載されている）。
- MLITの先進安全自動車推進検討会「自動速度制御装置（ISA）基本設計書」、2019年12月17日（令和元年12月）策定：目的は"ドライバーの不注意（漫然運転）や誤操作による速度超過を抑制することを目的とした装置とする"。速度制限の情報源は、カメラによる道路標識の認識、ナビゲーション地図データ、通信である（2.1）。上限速度は制限変更から5秒以内に設定または変更し、より低い制限が設定された場合は30秒以内に制御を開始する（"上限速度が現在の車速より低い速度に設定された場合は、設定された後遅くとも 30 秒以内に、新しい上限速度に合わせて車速の制御を開始すること"、2.3.3.1）。オフセットは最大±10 km/h。四輪車のみが対象で、導入期限はない（<https://www1.mlit.go.jp:8088/report/press/content/001320703.pdf>）。2024年の記事は、日本のISAが"is still at the study stage"であることを確認している（<https://kuruma-news.jp/post/794694>）。本調査時点で日本にTSRやISAの要件は存在しない（それを定める一次情報は見つからなかった）。
- 暗黙の制限に関わる法定速度の変更：改正道路交通法施行令により、2026年9月1日から生活道路（中央線や中央分離帯のない道路）の法定速度が60 km/hから30 km/hに引き下げられる。標識による指定はこれに優先する。中央線のある道路と自動車専用道路では60 km/hのままである（<https://web.pref.hyogo.lg.jp/kf13/press/20260528.html>）。したがって日本向けの暗黙制限モデルには、標識だけでなく「中央線あり」を特徴量として与える必要がある。

日本の標識体系（道路標識、区画線及び道路標示に関する命令、標識令）
- 法的根拠：昭和三十五年総理府・建設省令第三号。道路法第45条第2項および道路交通法第9条第3項に基づき1960年12月17日に公布された。第1条第2項："本標識は、案内標識、警戒標識、規制標識及び指示標識とする"。現行の条文はe-Gov法令APIからXMLとしてダウンロードした（法令ID 335M50004002003、<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>、2026-10-31取得）。最後に記載されている2件の改正省令は令和八年四月一日内閣府・国土交通省令第二号と令和八年六月五日内閣府・国土交通省令第三号である。
- 同XMLの別表第一から集計した件数（名称は種類欄による。「派生」はA/B/Cおよび「の2」形式の枝番を展開したもの）：

| カテゴリ | 異なる名称数（種類） | 基本番号数 | 番号付き派生数 |
|---|---|---|---|
| 案内標識（guide、101-126） | 41 | 41 | 89 |
| 警戒標識（warning、201-215） | 26 | 23 | 27 |
| 規制標識（regulatory、301-332） | 59 | 59 | 74 |
| 指示標識（indication、401-409） | 14 | 14 | 16 |
| 補助標識（supplementary、501-514） | 23 | 23 | 31 |
| 合計 | 163 | 160 | 237 |

  （<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>）。MLITの1ページの一覧表をOCRで数えると、同じカテゴリについて約77 / 26 / 73 / 16 / 31（合計約220）となり、OCRの誤差を考慮すれば上記の派生数と整合する（<https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf>）。
- 速度に関係する規制標識：323 最高速度、323の2 特定の種類の車両の最高速度、324 最低速度、325 自動車専用（高速道路の法定制限を示唆する）、329 徐行。規制標識の全リスト（301 通行止めから332 歩行者等横断禁止まで。303 車両進入禁止、311-AからF 指定方向外進行禁止、313 転回禁止、314/314の2 追越し関係、315/316 駐車関係、320/321/322 重量/高さ/幅、326 一方通行、327系列の車線規制、330 一時停止を含む）はXMLにある。速度標識の意味を変える補助標識：501 距離・区域、502 日・時間、503-AからD 車両の種類、505 始まり、506 区間内、506の2 区域内、507-AからD 終わり（規制の終わり）、509の2 踏切注意など（<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>）。一覧で確認できる最近の追加：特定小型原動機付自転車の標識（309、325の2、326の2）と503の2 遠隔操作型小型車。
- 別表第二の備考にある公式の色規定（合成データとVMSの扱いに有用）：速度標識と寸法関係の標識は"文字及び記号を青色、斜めの帯及び枠を赤色、縁及び地を白色"（青の数字、赤の環、白の地）、灯火により表示する最高速度 / 最低速度は"文字及び記号を白色又は黄色、地を黒色とすることができる"（黒地に白または黄の数字。日本の可変速度標識の通常の外観）。警戒標識は"縁線、文字及び記号を黒色、縁及び地を黄色"、一時停止は"文字及び縁線を白色、縁及び地を赤色"、徐行と前方優先道路は"文字を青色、枠を赤色、縁及び地を白色"、車両進入禁止は"帯及び縁を白色、地を赤色"である（<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>）。
- 標識の英語表記：2017年4月の標識令改正（2017年7月1日施行）により、日本語の下に"SLOW"を併記した徐行の様式が追加された（"「徐行」という日本字の下に「SLOW」という英字を併記した様式を追加します"）。警察庁（NPA）は、都道府県公安委員会が設置する一時停止標識と前方優先道路標識について同様の変更を行った（<https://www.mlit.go.jp/report/press/road01_hh_000829.html>）。道路上では両方の様式が併存するため、日本のクラスには両方が必要である。
- 図柄付きの公式またはオープンライセンスの一覧の入手先：（1）MLIT道路局の1ページのPDF（<https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf>。日付やライセンスの記載なし。MLITのページは一般に政府標準利用規約の下にあるが、ここでは未検証）。（2）前述のe-Gov XMLは様式の図を図の参照として含むのみで、ベクター画像は含まない。（3）Wikimedia Commonsは"Japan road sign NNN.svg"という名前のSVGを掲載しており（例：302 車両通行止め、作者Monaneko、出典"道路標識、区画線及び道路標示に関する命令別表第2"）、テンプレート"This work is in the public domain in Japan according to Article 13 of the Copyright Act of Japan"が付されている。この規定は法令を著作権の対象外とするものである（<https://download.osmand.net/wiki/File:Japan_road_sign_302.svg>。Commons APIはHTTP 429を返し、commons.wikipedia.comは404を返したため、237の派生すべてが揃っているかは検証していない）。したがって、標識令で定められた標識の意匠は、ライセンス上の問題なく合成学習データに使用できる。標識の写真は別問題である。

### A9. グローバルモデルにとって重要な標識規格

- Vienna Convention on Road Signs and Signals（1968年11月8日、1978年6月6日発効）：署名国35、締約国75。締約国にはドイツ（1978年8月3日批准）、フランス、イタリア、オーストリア、オランダ、スウェーデン、インドが含まれる。スペイン、英国、ブラジル、韓国は署名したが批准していない。日本、米国、中国、オーストラリア、カナダは一切記載がない（<https://treaties.un.org/doc/Publication/MTDSG/Volume%20I/Chapter%20XI/XI-B-20.en.pdf>）。したがって、OpenTSRの3つの主要市場（日本、EU、米国）はそれぞれ異なる体系に従う。EUはウィーン条約方式、米国はMUTCD、日本は独自の命令である（<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>の色規定による赤い環の禁止標識と黄地に黒の警戒標識。ひし形の警戒標識と逆三角形の一時停止標識はMLITの一覧<https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf>で確認できるが、形状は機械的には検証していない）。
- MUTCD 11th Edition（FHWA）：2024年1月18日発効（<https://mutcd.fhwa.dot.gov/mutcd_news.htm>）。米国の速度制限はmph表記の文字標識であり（"SPEED LIMIT 50"、R2-1）、IIHSのプロトコル草案はこれのみを試験する（<https://www.iihs.org/media/4e417587-ab67-489c-a3bb-7fe1dd87fe1b/COWTLw/Ratings/Protocols/future%20programs/Intelligent_speed_assistance_protocol.pdf>）。
- 中国：GB 5768.2-2022、2022-10-01施行（<https://download.osmand.net/wiki/File:GB_5768.2-2022.pdf>）。
- 上記の情報源に記録されている国間の混同：（a）単位：Thatchamは英国の車両が英国には存在しない5、10、15、100 mphの制限を表示するのを観測した。すなわち数字が妥当性チェックなしで読み取られている（<https://www.autocar.co.uk/node/375029>）。km/hのモデルを英国や米国で使ったり、mphのモデルを日本で使ったりすると、誤った単位が黙って出力されるため、運行国/地域はISAの明示的な要件となっている（point 3.4.2.1、<https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf>）。（b）市街地境界標識はEU各国で"shape, size and wording"が異なるが、それでも対応が求められる（Annex IIの注記）。（c）日本の可変速度標識は黒地に白または黄の数字であり、静的な白地に青の標識とは異なるが、標識令は両方を認めている（<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>）。（d）日本の最低速度（324）と最高速度（323）の標識は標識令で同一の色規定（青の数字、赤の環、白の地、<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>）を共有し、小さな意匠要素のみが異なる（最低速度の意匠では数字の下に下線がある。この詳細は<https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf>の図柄によるもので、機械的には検証していない）。これは速度制限のOCRが保持しなければならない細粒度の区別である。（e）ベルギーのISAの表は、同じ明示標識でも大型車（7.5 t以下と7.5 t超）では期待されるフィードバックが異なることを示しており、「正しい」ラベルは標識だけでなく車両区分にも依存する（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>）。フォントの違い（ドイツのDIN 1451、米国のFHWA Series、日本の標識書体）は、本セッションでは一次情報源から検証していない。

### A10. 商用システムの主張と既知の失敗事例

- Mobileye（2023年7月18日のプレスリリース）："vision-only Intelligent Speed Assist (ISA) solution"であり、ソフトウェア更新によりEyeQ4とEyeQ6で動作する。"city entrance identification for European-style signs"にOCRを用い、"static and dynamic message speed-limit signage"に対応する。EU全27か国とイスラエル、ノルウェー、スイス、トルコで認証済み。"Six independent labs across five different European countries have tested and confirmed"として適合を示している。精度の数値はなく、OEM名も明かされていない（<https://ir.mobileye.com/news-releases/news-release-details/mobileye-launches-first-camera-only-intelligent-speed-assist>）。
- Bosch "Road sign information"（商用車向け）："round, triangular and rectangular road signs"、速度制限（"including the start and ends of stretches where speed limits are in force, or where passing is prohibited"）、"relevant supplementary signs, such as time limits"、車種と右左折矢印の補助標識、進入禁止、一時停止、優先道路、道路工事を検出・分類する。"reliably detects road signs whether they are sited on actual signs, variable message systems or gantries"、"This function meets the requirement of the General Safety Regulation (GSR)"とされ、"In combination with a navigation system or the connected horizon"で性能が向上する。数値は示されていない（<https://bosch-mobility.com/en/solutions/assistance-systems/road-sign-information-cv>）。
- Teslaの取扱説明書（Model 3、日本向け英語版）、Speed Assistの制限事項："Visibility is poor and speed limit signs are not clearly visible"、"Bright light (such as from oncoming headlights or direct sunlight) is interfering with the view of the camera(s)"、"Speed limit signs are concealed by objects"、"The speed limits stored in the map database are incorrect or outdated"、"Traffic signs that do not conform to standard recognizable formats, such as digital or temporary speed signs"、"A road or a speed limit has recently changed"。また"Do not rely on Speed Assist to determine the appropriate speed limit or driving speed"とされている（<https://www.tesla.cn/ownersmanual/model3/en_jp/GUID-5D3D4014-4E98-45D7-8BBC-F76BCA9CEC05.html>）。少なくとも1つの量産システムでは、デジタル標識と仮設標識が明示的に対象外とされている。
- Continental/Aumovio、ZF、Denso：製品ページを取得できなかったか、汎用ページにリダイレクトされた（D.3参照）。検証済みの主張はない。
- 公開されている失敗事例の報告：（1）McAfee ATR、2020年2月19日：35 mphの標識に小さな黒いテープを1片貼ることで、2016年型Tesla Model S/X（hardware pack 1）のMobileye EyeQ3カメラが85 mphと読み取り、クルーズコントロールがその速度に向けて加速した。"MobilEye did indicate that the more recent version(s) of the camera system address these use cases"とされ、2020年型の車両はごく限られた試験で"did not appear to be susceptible"であった（<https://mcafee.com/blogs/other-blogs/mcafee-labs/model-hacking-adas-to-pave-safer-roads-for-autonomous-vehicles>）。（2）Phantom of the ADAS（Nassi et al.、CCS 2020。CACM、AutoSec 2021にも掲載）：投影された、またはビルボードに埋め込まれた幻影標識がTesla Model X（HW2.5およびHW3）とMobileye 630を欺いた。125 msだけ表示した一時停止標識でAutopilotが停止した。対策は"GhostBusters"（光、文脈、表面、奥行きに関する4つのCNNエキスパート。FPRゼロでTPR 0.994）。プロジェクトページにはコードやデータへのリンクはない（<https://nassiben.com/phantoms>）。（3）Wang et al.、NDSS 2025：2023-2024年型の米国市場向け4車種（候補リストはTesla Model 3 2023、Toyota Camry 2023、Nissan Sentra 2023、Mazda CX-30 2023、Hyundai Tucson 2024。C1-C4との対応は非公開）。C1はSTOPのみ、C2はSTOPと速度制限、C3とC4は速度制限のみを読み取る。結果は後述のC1節を参照（<https://arxiv.org/pdf/2409.09860>）。（4）Thatchamのイベントベースの結果と、Euro NCAPのBYD Atto 3に関する所見（前述のA4）。標識の誤読に関するNHTSAのリコールは、本セッションでは確認していない。

---

## B. 標識向けロバスト性ベンチマークと劣化データセット

### B1. CURE-TSRとCURE-TSD（Georgia Tech OLIVES）

- 出自：Dogancan Temel, Gukyeong Kwon, Mohit Prabhushankar, Ghassan AlRegib（Georgia Institute of Technology）。NIPS 2017ワークショップ論文（arXiv 1712.02463）。検出版のCURE-TSDはTemel, Chen, AlRegib, "Traffic Sign Detection Under Challenging Conditions: A Deeper Look into Performance Variations and Spectral Characteristics", IEEE T-ITS 2019, DOI 10.1109/TITS.2019.2931429, arXiv 1908.11262に記述されている（<https://zenodo.org/record/3903066>、<https://alregib.ece.gatech.edu/software-and-datasets/cure-tsd-challenging-unreal-and-real-environments-for-traffic-sign-detection>）。
- 地域と出典：実フレームはBelgiumTSの動画シーケンスから取得。合成（"unreal"）フレームはUnreal Engine 4でレンダリング。チャレンジ効果はAdobe After Effectsで付加（<https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition>）。
- CURE-TSDの規模：各300フレームの実ベースシーケンス49本と合成（unreal）ベースシーケンス49本。実データ×12チャレンジ種別×5段階+ベース = 2,989シーケンス、合成データ×11種別×5段階+ベース = 2,744シーケンス。合計5,733シーケンス、約172万フレーム。70/30で分割し、ドメインごとに学習用34本、テスト用15本の動画とする。ファイル名は`sequenceType_sequenceNumber_challengeSourceType_challengeType_challengeLevel.mp4`（<https://alregib.ece.gatech.edu/software-and-datasets/cure-tsd-challenging-unreal-and-real-environments-for-traffic-sign-detection>）。解像度とインスタンス数：同ページには記載なし（IEEE DataPortは"more than 2 million traffic sign instances"としている。<https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition>）。
- CURE-TSR：CURE-TSDからのクロップで"over 2 million images"、14クラス：speed_limit, goods_vehicles, no_overtaking, no_stopping, no_parking, stop, bicycle, hump, no_left, no_right, priority_to, no_entry, yield, parking。12のチャレンジ種別とコード01-12：decolorization, lens blur, codec error, darkening, dirty lens, exposure, gaussian blur, noise, rain, shadow, snow, haze（00 = チャレンジなし）。5段階で01（最も軽い）から05（最も深刻）。ファイル名は`sequenceType_signType_challengeType_challengeLevel_Index.bmp`。zipは4.6 GB（DataPort）または4.9 GB（Zenodo）（<https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition>、<https://zenodo.org/record/3903066>）。train/testの分割サイズ：どちらのページにも記載なし。
- ライセンス：Zenodoレコード3903066（v1、2020年6月22日）："Creative Commons Attribution 4.0 International"（<https://zenodo.org/record/3903066>）。IEEE DataPortはライセンスを明記しておらず、Open Accessのファイルには無料のIEEEアカウントが必要としつつ、"How to access"欄では購読に言及している（<https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition>）。再配布にはZenodo版（DOI 10.5281/zenodo.3903066）を用いること。
- 入手方法：Zenodo（CURE-TSR.zip、md5 49c2f9b24e5e186cac6df136a14d28e0）、IEEE DataPort（DOI 10.21227/n4xw-cg56）。コードは<https://github.com/olivesgatech/CURE-TSR>と<https://github.com/olivesgatech/CURE-TSD>（GitHubページは未取得、D.3）。
- ベースライン：未検証（T-ITS論文は読んでいない）。
- 既知の問題：14の粗いクラスのみ（速度値なし）、ベルギーの標識のみ、天候は合成。それでも段階的な深刻度レベルを持つ唯一の標識ベンチマークであり、学習データとしてよりもロバスト性の回帰テストとして有用である。

### B2. Hugging Face上のGTSRB劣化版（tanganke/gtsrb）

- 内容：データセット`tanganke/gtsrb`（作者Anke Tang、最終更新2024-05-07、取得時点で16,418ダウンロード）。GTSRBの43クラスで、分割はtrain 26,640、test 12,630に加え、テスト分割の劣化版7種（各12,630枚）：contrast, gaussian_noise, impulse_noise, jpeg_compression, motion_blur, pixelate, spatter。parquet形式、ダウンロードサイズ1.03 GB。ライセンス欄は空（<https://huggingface.co/api/datasets/tanganke/gtsrb?full=true>、<https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md>）。26,640 / 12,630の分割は、元の39,209枚の学習セットではなくtorchvisionのGTSRB分割と一致する（torchvisionの数値はここでは再検証していない）。
- 劣化の出所：カードはAdaMerging（Yang et al.、arXiv 2310.02575）と、Tang et al.によるモデルマージの論文2本（arXiv 2312.06173、2402.00433）を引用している。AdaMergingは"Following Hendrycks & Dietterich (2019), we created 7 corruption test data"と述べ、GTSRB、Cars、EuroSAT、RESISC45について"Motion Blur, Impulse Noise, Gaussian Noise, Pixelate, Spatter, Contrast and JPEG Compression"を列挙している（<https://arxiv.org/pdf/2310.02575>、ロバスト性に関する節とAppendix B）。したがってこれらの分割は、モデルマージのコミュニティがImageNet-Cのレシピ（Hendrycks and Dietterich、ICLR 2019）をGTSRBに適用したものである。深刻度レベルはカードに記録されていない（不明）。GTSRBにおけるAdaMergingの数値例（ViT-B/32、Task Arithmetic対AdaMerging）：clean 75.1対96.3、impulse noise 40.4対77.5、gaussian noise 49.4対76.0、pixelate 66.6対90.6、spatter 54.3対89.4、contrast 70.6対（表が途切れている）、JPEG 60.0対（途切れている）（<https://arxiv.org/pdf/2310.02575>）。
- ライセンス上の注意：カードにはライセンスがない。元のGTSRB画像はINIのベンチマークに由来し、その利用条件は`datasets.md`で扱っている。GTSRBの利用条件が確定するまでは、評価専用のミラーとして扱うこと。

### B3. 標識ラベル付きの夜間・天候データソース（参照先のみ。詳細は`datasets.md`）

- Zenseact Open Dataset（ICCV 2023）：欧州14か国の100kフレーム。156の細分類クラスに446kの道路標識インスタンス。天候は晴れ61.2%、曇り19.0、雨15.7、霧2.1、雪2.0。昼77.3%、夜19.0、薄暮3.6。ResNet-50のTSRベースラインはF1-macro 78.5、F1-micro 95.4、上位10クラス93.4%、下位10クラス65.4%。"released under the permissive CC BY-SA 4.0 license"とされる（<https://arxiv.org/pdf/2305.02008>）。見つかった中で、夜間と雨天の標識クロップの公開ソースとしては最大である。
- TS-1M（HKUST Guangzhou、2026）：1,267,558枚（train 1,033,947 / test 233,611）、Restriction、Prohibition、Warning、Guidanceにわたる454の標準化カテゴリ。GTSRB、TT100K、RTSD、MTSD、インターネット画像を統合している。雨、霧、夜間、逆光の画像を含むが、条件別の精度は報告していない。ライセンスは記載なし。プロジェクトページは<https://guoyangzhao.github.io/projects/ts1m>（<https://arxiv.org/pdf/2603.23034>）。

### B4. ドメインシフトと国間比較の研究

- TS-1Mベンチマーク：TS-1Mで学習したモデルを10の外部地域データセットで評価。ファインチューニングしたCLIPの平均は0.9578（GTSRB 0.9996、BelgiumTS 0.9921、RTSD 0.9968、MTSD 0.9118、ARTS 0.9719）、ConvNeXt 0.9353、ResNet50 0.9150。希少クラス（< 40サンプル）ではCLIP 0.9291対ResNet50 0.8381、低鮮明度（< 30 x 30 px）では0.9313対0.8768。個々のソースデータセット間で、Xで学習しYでテストするペアごとの実験はない（<https://arxiv.org/pdf/2603.23034>）。
- Gan et al.（北海道大学）、"Cross-domain Few-shot In-context Learning for Enhancing Traffic Sign Recognition"、arXiv 2407.05814（2024年7月）：テンプレート標識の記述をMLLMへのプロンプトとする。GTSRB、BelgiumTS、日本の実環境データセット2つで評価。アブストラクトに数値はない（<https://arxiv.org/abs/2407.05814v1>）。日本の標識をターゲットドメインとして扱っている点で関連がある。
- Wu et al.（2026）、"Efficient Cross-Country Data Acquisition Strategy for ADAS via Street-View Imagery"、arXiv 2602.01836：ZODとMapillaryのストリートビュー画像を対応付けて、対象国での収集地点を選定する。"a task particularly sensitive to cross-country variations in sign appearance"である道路標識検出において、対象ドメインのデータの半分でランダムサンプリングと同等の性能を得た（<https://export.arxiv.org/api/query?id_list=2602.01836>）。
- 量産システムにおけるドメインシフトの証拠：Thatchamの英国での結果（5、10、15、100 mphの制限が表示された）と、Euro NCAPのBYD Atto 3に関する所見（A4）。ISA規則自体の対策は、運行国設定の義務化である（A2）。

---

## C. TSRに対する敵対的攻撃と防御

以下のarXiv論文のメタデータ（発表先、著者、日付）は2026-10-31にarXiv APIから取得した（<https://export.arxiv.org/api/query?id_list=...>）。リポジトリのスター数、ライセンス、最終コミットはshields.ioから、リポジトリの内容はREADMEの生データから取得した。「コード：見つからず」は、本セッションで公式リポジトリを特定できなかったことを意味し、存在しないことを意味するものではない。

### C1. 標識分類器・検出器に対する物理攻撃とデジタル攻撃

RP2、Eykholt et al.、"Robust Physical-World Attacks on Deep Learning Visual Classification"、CVPR 2018（arXiv 1707.08945、初版2017年7月、"Accepted to CVPR 2018"）
- 対象：LISA-CNN（米国の標識、"achieves 91% accuracy"）とGTSRB-CNN（ドイツの標識で、一時停止標識をLISAの米国のものに置き換えたもの）、およびImageNetのInception V3版。物理：一時停止標識に印刷したポスターとステッカーを貼付。データセット：LISA、GTSRB。
- コード：<https://github.com/evtimovi/robust_physical_perturbations>、MIT、スター数112、最終コミット2019年12月。README：フォルダ`lisa-cnn-attack`、`gtsrb-cnn-attack`、`imagenet-attack`。古いcleverhansのコピー（MIT）を同梱（<https://raw.githubusercontent.com/evtimovi/robust_physical_perturbations/master/README.md>、<https://img.shields.io/github/stars/evtimovi/robust_physical_perturbations.json>）。
- 検出器への拡張：Eykholt et al.、"Physical Adversarial Examples for Object Detectors"、USENIX WOOT 2018（arXiv 1807.07769）：YOLOv2に対する消失攻撃。実験室の動画フレームの85%超で一時停止標識が見逃され、屋外では72.5%（ポスター）と63.5%（ステッカー）。Faster R-CNNへ転移。WOOT版のコード：Wang et al.（NDSS 2025）は、RP2（検出器版）、SIB、FTEのコードを入手できず、著者の協力を得て再実装したと報告している（<https://arxiv.org/pdf/2409.09860>）。

ShapeShifter、Chen, Cornelius, Martin, Chau、ECML-PKDD 2018（arXiv 1804.05810、journal_ref "Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 52-68, 2018"）
- 対象：Faster R-CNN（TensorFlow Object Detection API）の一時停止標識。標的型（一時停止標識を人物として検出させる）と非標的型。物理：印刷した一時停止標識。後に2D/3Dの人物への攻撃が追加された。
- コード：<https://github.com/shangtse/robust-physical-attack>、BSD-3-Clause、スター数170、最終コミット2020年1月。ノートブックとColab、`shapeshifter2d.py` / `shapeshifter3d.py`、`2d_stopsign_targeted_attack`などのMakefileターゲット（<https://raw.githubusercontent.com/shangtse/robust-physical-attack/master/README.md>）。

Rogue Signs / DARTS、Sitawarin, Bhagoji, Mosenia, Chiang, Mittal（Princeton INSPIRE）、2018
- Rogue Signs：第1回Deep Learning and Security Workshop（DLS、IEEE S&P 2018併設）、arXiv 1801.02780："Sign Embedding attack"により、広告やロゴを、選んだ標識として分類される入力に変える。物理的な走行中（drive-by）テストで成功率は"in excess of 95%"。DARTS：arXiv 1802.06430（"Submitted to ACM CCS 2018"、拡張版）：Out-of-Distribution攻撃とLenticular Printing攻撃。敵対的学習済みの分類器に対しては、OOD攻撃が分布内攻撃を上回る。
- 対象：GTSRB上のKerasのマルチスケールCNNと通常のCNN（リポジトリ内の重み`weights_mltscl_dataaug.hdf5`、`weights_cnn_dataaug.hdf5`）。物理：印刷した標識。コード：<https://github.com/inspire-group/advml-traffic-sign>、MIT、スター数37、最終コミット2018年4月。`OptProjTran.py`（ロバスト最適化）、Carlini-Wagner、ランダムな透視変換、pickle形式で生成済みの敵対的標識を含む（<https://raw.githubusercontent.com/inspire-group/advml-traffic-sign/master/README.md>）。

Seeing isn't Believing（SIB）、Zhao et al.、ACM CCS 2019（arXiv 1812.10217）
- YOLOv3とFaster R-CNNに対する隠蔽攻撃（特徴干渉の強化、現実的な制約の強化）と出現攻撃（遠距離と近距離に対する"nested AE"）。1〜25 m、-60〜60度で成功率は最大92.4%。物理。コード：見つからず。Wang et al.は元のコードなしで再現した（<https://arxiv.org/pdf/2409.09860>）。

SLAP、Lovisotto, Turner, Sluganovic, Strohmeier, Martinovic、USENIX Security 2021（arXiv 2007.04137）
- プロジェクタによる一時停止標識（およびボトル、譲れ標識）への短時間の摂動。対象はYOLOv3、Mask R-CNN、LISA-CNN、GTSRB-CNNで、敵対的学習済みの"usenix21adv"版もある。所与の照度（lux）で投影可能な色をプロファイリングする。物理（屋内および屋外）。
- コード：<https://github.com/ssloxford/short-lived-adversarial-perturbations>、GPL-3.0、スター数28、最終コミット2021年10月。リリース`usenix21`の`data.zip`にモデル（GTSRBとLISAそれぞれのcvpr18、cvpr18iyswim、usenix21、usenix21adv）、プロファイリングデータ、投影パターン、GTSRB/LISAの再学習セットが含まれる。Docker HubにDockerイメージあり。TensorFlow 1.15（<https://raw.githubusercontent.com/ssloxford/short-lived-adversarial-perturbations/master/README.md>）。GPL-3.0であるため、OpenTSRはこれをツールとして実行できるが、コードをApache-2.0のソースに取り込む（vendorする）べきではない。

Adversarial Camera Stickers、Li, Schmidt, Kolter、ICML 2019（arXiv 1904.00759、PMLR 97:3896-3904）
- カメラレンズに貼る半透明のステッカー。ImageNetのResNet-50。"street sign"を"guitar pick"に誤認させるのが代表例で、実際の一時停止標識がguitar pickと分類された。標的型の誤認成功率は49.6%（アブストラクト）、実動画では52%。street signの精度は64%から18%に低下。物理。コード：見つからず（論文にはデモ動画へのリンクのみ）（<https://ar5iv.labs.arxiv.org/html/1904.00759>）。

ローリングシャッター攻撃と光による攻撃
- Sayles et al.、"Invisible Perturbations: Physical Adversarial Examples Exploiting the Rolling Shutter Effect"、CVPR 2021（arXiv 2011.13375）：変調したLED照明が縞模様を生じさせ、ImageNet分類器を欺く。標的型の成功率は最大84%。物理。コード：見つからず。
- Köhler et al.、"They See Me Rollin'"、2021（arXiv 2101.10011）：レーザー変調によるローリングシャッターへの注入攻撃を、BDD100KとVIRAT上でFaster R-CNNとSSDを用いて評価。コード：<https://github.com/ssloxford/they-see-me-rollin>、GPL-3.0、スター数8（<https://raw.githubusercontent.com/ssloxford/they-see-me-rollin/master/README.md>）。
- Yan et al.、"Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition"、USENIX Security 2022（arXiv 2204.02675）：標識ではなく信号機（LIGHT）の認識に対するレーザー攻撃（<https://www.usenix.org/conference/usenixsecurity22/presentation/yan>）。タスクで言及されていたため掲載したが、TSRへの攻撃ではない。
- Duan et al.、"Adversarial Laser Beam"（AdvLB）、CVPR 2021（arXiv 2103.06504）：ImageNet分類器に対してレーザービームのパラメータを最適化。夜間テストあり。敵対的学習済みのResNet50では攻撃成功率が95.10%から77.20%に低下。コード：<https://github.com/RjDuan/Advlight>、MIT、スター数34（<https://raw.githubusercontent.com/RjDuan/Advlight/master/README.md>）。TSR固有ではない。
- Gnanasambandam, Sherman, Chan、"Optical Adversarial Attack"（OPAD）、ICCV Workshop 2021（arXiv 2108.06247）：プロジェクタ-カメラモデルを攻撃の最適化に組み込み、実際の3D物体に対してホワイトボックス攻撃とブラックボックス攻撃を行う。コード：見つからず。
- 赤外線攻撃（Zhou et al. "Invisible Mask"、顔認識）は本セッションでは検証しておらず、TSRの研究でもない。

Shadows、Zhong, Liu, Tang, Xu, Li、"Shadows can be Dangerous"、CVPR 2022（arXiv 2203.03818）
- GTSRBとLISAのクロップに対する、ブラックボックス（粒子群最適化）による影の形状の摂動。デジタルおよび物理。リポジトリには`model_gtsrb.pth`、`model_lisa.pth`と敵対的学習済みの`adv_model_gtsrb.pth`、`adv_model_lisa.pth`、標識形状（円、逆三角形、八角形、五角形、長方形、ひし形、三角形）のマスクが含まれる。PyTorch 1.9。
- コード：<https://github.com/hncszyq/ShadowAttack>、MIT、スター数64、最終コミット2025年9月（<https://raw.githubusercontent.com/hncszyq/ShadowAttack/master/README.md>）。逆三角形のマスクがあるため、日本の一時停止と徐行にそのまま再利用できる。

TPatch、Zhu, Ji, Yan et al.（Zhejiang USSLab）、USENIX Security 2023
- カメラへの音響信号注入攻撃によってトリガーされるまでは無害なままのパッチ。隠蔽、生成、改変の各攻撃。YOLOv3/v5、Faster R-CNN、8つのImageNet分類器を用い、ImageNet、COCO、KITTI、BDD100K（標識データセットではない）で評価。物理デモあり。
- コード：<https://github.com/USSLab/TPatch>、ライセンス指定なし、スター数24、最終コミット2023年8月（<https://raw.githubusercontent.com/USSLab/TPatch/master/README.md>）。発表先はリポジトリにある論文PDFの参照情報から取得した。USENIXのページは取得していない。

Phantom of the ADAS、Nassi, Mirsky, Nassi, Ben Netanel, Elovici、CCS 2020（A10参照）：Tesla Model XとMobileye 630に対する、物理的な投影およびビルボードによる幻影。コード：見つからず（<https://nassiben.com/phantoms>）。

DeepBillboard、Zhou et al.、2018（arXiv 1812.10812。ICSE 2020での発表は未検証）
- エンドツーエンド運転モデル（Daveなど）の操舵を変化させる敵対的ビルボード。デジタルおよび物理。標識分類器への攻撃ではない。コード：<https://github.com/deepbillboard/DeepBillboard>、ライセンス指定なし、スター数11（<https://raw.githubusercontent.com/deepbillboard/DeepBillboard/master/README.md>）。

Translucent Patch、Zolfi, Kravchik, Elovici, Shabtai、CVPR 2021（arXiv 2012.12528）
- 対象クラスの全インスタンスを隠すレンズパッチ。一時停止標識インスタンスの42.27%の検出を阻止しつつ、他クラスの検出は約80%を維持。物理。コード：見つからず。

Invisible for both Camera and LiDAR（MSF-ADV）、Cao et al.、IEEE S&P 2021（arXiv 2106.09249）
- Apolloのカメラ-LiDAR融合に対する3Dプリント物体。標識ではない。コード：<https://github.com/ASGuard-UCI/MSF-ADV>、MIT、スター数84。関連するレーンセンタリング攻撃"Dirty Road Can Attack"（Sato et al.、USENIX Security 2021、arXiv 2009.06701）、コード：<https://github.com/ASGuard-UCI/DRP-attack>、MIT、スター数38。いずれもTSRへの攻撃としてではなく、自動運転の認識系の脅威モデルを網羅するために掲載した。

GhostImage、Man, Li, Gerdes、USENIX RAID 2020（arXiv 2001.07792）
- プロジェクタでレンズフレア/ゴーストと自動露出を悪用し、遠隔からパターンを注入する。3つのデータセットの1つとしてLISA分類器を含む。成功率は距離に応じて最大100%。物理。
- コード：<https://github.com/Harry1993/GhostImage>、MIT、スター数3。TF 2.1、学習済みLISA分類器のダウンロードあり（<https://raw.githubusercontent.com/Harry1993/GhostImage/master/README.md>）。

FTE、Jia et al.、"Fooling the Eyes of Autonomous Vehicles"、NDSS 2022（arXiv 2201.06192）
- ぼかしと解像度の変換を考慮した、YOLOv5のTSRパイプラインに対するロバストな物理的敵対的サンプル（AE）。隠蔽/出現/非標的/標的の各攻撃、晴天、曇天、夜間。コード：見つからず。論文は商用システムでの成功を主張しているが、Wang et al.はFTEの成功率が4台の商用車すべてで0%であったと報告している（<https://arxiv.org/pdf/2409.09860>）。

Revisiting Physical-World Adversarial Attack on TSR: A Commercial Systems Perspective、Wang, Xie, Sato, Luo, Xu, Chen（UC Irvine、Drexel）、NDSS 2025（arXiv 2409.09860）
- 設定：RP2、SIB、FTEを再実装。代理モデルはYOLOv5（mAP 0.831）とFaster R-CNN ResNet-50-FPN-V2（0.871）。STOPにはCOCO、25 mph標識にはARTSを使用。屋外の屋上、25,000-30,000 lux、5 mph、3回反復。
- 結果：30の攻撃・車両の組み合わせのうち2つが、C2のSTOP標識で成功した（3/3）（Faster R-CNNを代理モデルとするRP2、YOLOv5によるSIB）。全体では6.67%で、元論文の"51.6% on average"を下回る（p < 0.02）。"Spatial memorization"（空間的記憶）：C2-C4は検出した標識を最大60秒間保持し、標識の位置を通過した後に消去する。約1秒の検出で記憶されるのに十分である。システムレベルの指標では、ホワイトボックスでのフレーム単位の隠蔽成功率56.8%（RP2）と47.5%（FTE）が6.6%と5.2%に低下する（SIBは87.1から45.1）。ブラックボックスでは平均47から約12.6%に低下。出現攻撃は逆の傾向を示す（SysAA 98.2〜100%）。
- コードとデータ："will be made available at our website"（<https://sites.google.com/view/av-ioat-sec/commercial-tsr-test>、未取得）。論文：<https://arxiv.org/pdf/2409.09860>。

2023〜2026年の研究
- DiffSign、Ma, Wang, Zheng, Guo, Wang, Chen, Shen（Xi'an Jiaotong、UC Irvine、Nankai、Wuhan）、arXiv 2511.12956、2025年11月17日：Stable Diffusion 2.1をベースに、CLIP損失とマスク付きプロンプトを用いたtext-to-imageの出現攻撃。対象はYOLOv3/v4/v5/v11、Faster R-CNN、Mask R-CNN、DETR、SSD。データはCOCO（STOP）とTT100K（速度制限5/40/60 km/h、No Entry、Children Crossing）。デジタルのAASRは8つの検出器の平均で82.7%。物理では平均83.3%（STOP：晴天85.4、曇天74.7、夜間47.8。5 km/h：97.7 / 98.8 / 96.6）。名称非公開の商用TSRシステム2つに対して97%で攻撃成功。PASS/CARLAのエンドツーエンド評価では100%（ベースラインは0）。コードは"will publicly release"とされるが、URLはまだない（<https://arxiv.org/html/2511.12956v1>）。
- FIGhost、Yuan, Xu, Li, Zhang et al.、arXiv 2505.12045（2025年5月）：UV光で活性化する蛍光インクのトリガーを用いた物理バックドア。検出器とVLMが対象。コード：不明（<https://export.arxiv.org/api/query?id_list=2505.12045>）。
- 運転向けVision-LLMに対するタイポグラフィ攻撃、Chung et al.、arXiv 2405.14169（2024年5月）：交通シーン中に自動生成した誤誘導テキストを用い、LLaVA、Qwen-VL、VILA、Impを攻撃。物理的な実現可能性も検討。コードは"upon acceptance"（<https://arxiv.org/abs/2405.14169>）。
- CARLAにおけるVLMへのパッチ攻撃：Fernandez, MohajerAnsari, Salarpour et al.、"Comparative Analysis of Patch Attack on VLM-Based Autonomous Driving Architectures"、IEEE IV 2025（arXiv 2603.08897）および"Understanding Adversarial Transferability in VLMs for Autonomous Driving"、SAE WCX 2026（arXiv 2604.27414）：道路脇のインフラに設置したパッチが、Dolphins、OmniDrive、LeapVADの間で73-91%の割合で転移する（<https://export.arxiv.org/api/query?id_list=2603.08897,2604.27414>）。

### C2. TSRに関連する防御、認証、ロバスト学習

- GTSRBとLISAの重みを提供する敵対的学習のベースライン：SLAPの`usenix21adv`モデル（GPL-3.0のリポジトリ）とShadowAttackの`adv_model_gtsrb.pth` / `adv_model_lisa.pth`（MIT）。C1参照。LAMDA、MohajerAnsari, Salarpour, Pesé、IROS 2026（arXiv 2608.08815）：凍結したOpenCLIPテキストエンコーダによる、言語をアンカーとするプロトタイプ損失。GTSRBとLISAで、4つのバックボーンと3種類の物理攻撃（影、自然光、印刷パッチ）を用いて評価。すべての組み合わせでロバスト性を向上させた"the only method among ten evaluated"であり、向上幅は最大+12.5 pp。コードは不明（<https://export.arxiv.org/api/query?id_list=2608.08815>）。V2LMs（同グループ、IEEE IV 2026、arXiv 2506.11472）：TSR、レーンセンタリング、車両検出向けにファインチューニングしたVLMは、DNNが33-74%低下する攻撃下でも低下が8%未満にとどまる。コードは不明。
- SentiNet、Chou, Tramèr, Pellegrino、DLS workshop 2020（arXiv 1812.00292）：顕著性とオーバーレイ検査により、局所的なユニバーサル攻撃（パッチ、トロイ、ポイズニング）を検知する。学習は不要。物理的に実現可能なパッチに対して評価。公式コード：見つからず。
- Feature Squeezing、Xu, Evans, Qi、NDSS 2018（arXiv 1704.01155）：ビット深度の削減と平滑化を検知器として用いる。コード：<https://github.com/mzweilin/EvadeML-Zoo>、MIT、スター数187。アブストラクトにはGTSRBでの評価は挙げられておらず、「GTSRBでのFeature Squeezing」の結果はいずれも第三者によるものである（未検証）。
- 認証付きパッチ防御（いずれもアブストラクトではGTSRBで評価していない）：Chiang et al.、ICLR 2020（arXiv 2003.06693）、コード：<https://github.com/Ping-C/certifiedpatchdefense>、BSD-2-Clause。Levine and Feizi、(De)Randomized Smoothing、NeurIPS 2020（arXiv 2002.10733）、CIFAR-10とImageNet、コード：<https://github.com/alevine0/patchSmoothing>、ライセンス識別不可。PatchGuard、Xiang et al.、USENIX Security 2021（arXiv 2005.10884）、ImageNet/ImageNette/CIFAR-10、コード：<https://github.com/inspire-group/PatchGuard>、MIT、スター数74。PatchGuardや(De)Randomized SmoothingのGTSRBへの移植は新規の研究となる。
- REAPベンチマーク、Hingun, Sitawarin, Li, Wagner、ICCV 2023（arXiv 2212.05680）：14,000を超えるMapillary Vistasの道路標識それぞれに幾何変換と照明変換が付与されており、デジタルパッチを現実的に適用できる。知見："adversarial patch attacks may present a smaller threat than previously believed"であり、単純なデジタルでの成功は実用上の有効性について"is not predictive"とされる。コード：<https://github.com/wagner-group/reap-benchmark>（ライセンス識別不可、スター数34、最終コミット2023年10月）。データと重みはKaggleの`csitawarin/reap-benchmark`（REAP `100`、REAP-Shape `no_color`、合成セット）。detectron2、kornia、detrex、YOLOF、YOLOv7に依存（<https://raw.githubusercontent.com/wagner-group/reap-benchmark/master/README.md>）。画像にはMapillary Vistasのライセンスが適用される（`datasets.md`参照）。

### C3. 道路標識を用いたバックドアおよびポイズニング研究

- BadNets、Gu, Dolan-Gavitt, Garg、2017（arXiv 1708.06733）：ステッカーがあると一時停止標識を速度制限とラベル付けする、バックドア入りの米国道路標識分類器。バックドアは転移学習後も残り、トリガー下で平均25%の精度低下を引き起こす。オリジナルのコード：<https://github.com/Kooscii/BadNets>（Caffe、py-faster-rcnn、LISA/VIVAチャレンジ由来の"US Traffic Signs (usts)"データセット、黄色の正方形/爆弾/花のトリガー）、ライセンス指定なし、スター数107、最終コミット2018年5月（<https://raw.githubusercontent.com/Kooscii/BadNets/master/README.md>）。コミュニティによるPyTorchの再実装：<https://github.com/verazuo/badnets-pytorch>、MIT、スター数197（データセットは未検証）。
- TrojanNN、Liu et al.、NDSS 2018：内部ニューロンから生成したトロイのトリガー。顔認識、音声、年齢、文の態度、自動運転のモデル。コード：<https://github.com/PurduePAML/TrojanNN>（Caffe、Theano）、ライセンス指定なし、スター数192（<https://raw.githubusercontent.com/PurduePAML/TrojanNN/master/README.md>）。READMEにGTSRBのモデルは記載されていない。
- Neural Cleanse、Wang et al.、IEEE S&P 2019：トリガーのリバースエンジニアリングと緩和。リポジトリには感染済みのGTSRBモデル（`models/gtsrb_bottom_right_white_4_target_33.h5`）とテストデータ（`data/gtsrb_dataset_int.h5`）、およびGTSRB向けの注入コードが含まれる。Keras 2.2.2 / TF 1.10。コード：<https://github.com/bolunwang/backdoor>、MIT、スター数326（<https://raw.githubusercontent.com/bolunwang/backdoor/master/README.md>）。
- ベンチマーク：BackdoorBench（SCLBD）はCIFAR-10、CIFAR-100、GTSRB、Tiny ImageNetに対応。CC BY-NC 4.0（非商用のため評価用途のみ）、スター数619、最終コミット2025年7月（<https://raw.githubusercontent.com/SCLBD/BackdoorBench/master/README.md>）。TrojanZoo：<https://github.com/ain-soph/trojanzoo>、MIT、スター数304（GTSRB対応は未検証）。

### C4. シミュレータとシステムレベル評価ツール

- SafeBench、Xu et al.、NeurIPS 2022 Datasets and Benchmarks（arXiv 2206.09682）：CARLA 0.9.13ベースのプラットフォームで、認識と制御の評価、Scenicによるシナリオ生成、Dockerイメージを備える。MIT、スター数164、最終コミット2023年9月（<https://raw.githubusercontent.com/trust-ai/SafeBench/master/README.md>）。認識トラックに敵対的な標識テクスチャが含まれるかは検証していない。
- PASS / CARLAによるエンドツーエンド評価はDiffSignで使用されている（ASR-S指標）（<https://arxiv.org/html/2511.12956v1>）。PASSシミュレータ自体は取得していない。
- Wang et al.のシステムレベル指標SysHA（標識通過前の区間ごとの隠蔽成功率の積）とSysAA（1から区間ごとの失敗率の積を引いたもの）は、時間的持続性を持つシステムにとって適切な評価単位である（<https://arxiv.org/pdf/2409.09860>）。
- CARLAに組み込まれた速度制限標識と一時停止標識のアセット、およびそのクラス網羅範囲は検証していない。

---

## D. 成果

### D.1 OpenTSR目標仕様草案（1ページ）

A〜C節から導出した。[reg]の付いた項目は法規またはプロトコルから写したもの、[proposed]の付いた項目は外部の情報源を持たないOpenTSRとしての選択である。

標識カテゴリの範囲
- 日本[proposed、カタログは<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>による]：規制標識全59種類（74派生）。323 / 323の2 / 324の速度標識を含み、その数値を属性として持つ。325 自動車専用は暗黙の高速道路制限のトリガーとし、330 一時停止と329 徐行は日本語のみの様式と2言語併記の様式の両方を含める。警戒標識全26種類。指示標識全14種類。補助標識23種類は親標識に付随する属性とする（501 距離・区域、502 日・時間、503 車両の種類、505 始まり、506 区間内、507 終わり）。案内標識は高速道路の入口/出口系（103 入口の方向、104 入口の予告、109 出口の予告、110 方面及び出口の予告、112 方面及び出口、113 出口）と路線番号（118 国道番号、118の2 都道府県道番号、118の3 高速道路番号）に限定する[proposed]。黒地に白または黄の数字を表示する可変（発光）速度標識は速度クラスに含めなければならない（<https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003>）。暗黙の法定制限：一般道60 km/h、2026-09-01以降は中央線のない道路で30 km/h、高速道路100 km/h。このため認識スタックは「中央線あり」も出力しなければならない（<https://web.pref.hyogo.lg.jp/kf13/press/20260528.html>）。
- EU[reg、<https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf>のAnnex IIの構成]：明示的数値制限5〜130 km/h、暗黙・数値（取り消し線付きの制限解除）、暗黙・非数値（市街地入口/出口、高速道路始点/終点、自動車専用道路始点/終点、ゾーン始点/終点、交通静穏化区域始点/終点、全規制解除）、これらのいずれかを表示する可変表示標識。さらにEuro NCAPの条件付き系統（雨/路面湿潤、雪/凍結、時間/季節、距離（for/in）、車線矢印、トレーラーを含む車両・重量区分）と動的/道路工事の制限（<https://cdn.euroncap.com/media/77301/euro-ncap-assessment-protocol-sa-safe-driving-v102.pdf>）。最小クラスセット：GTSRBの43クラス（<https://huggingface.co/api/datasets/tanganke/gtsrb?full=true>）に上記の暗黙系統と補助系統を加えたもの[proposed]。
- 米国[reg、<https://www.iihs.org/media/4e417587-ab67-489c-a3bb-7fe1dd87fe1b/COWTLw/Ratings/Protocols/future%20programs/Intelligent_speed_assistance_protocol.pdf>]：単位処理を伴うmph表記の数値速度制限（R2-1）。次の段階として一時停止、譲れ、進入禁止（do not enter）、逆走（wrong way）、スクールゾーン、制限解除[proposed]。
- 出力契約[proposed]：標識ごとにクラス、単位付きの数値、補助属性、車線関連フラグ、信頼度。持続ロジックを適用できるよう、フレームごとに追跡ID。

性能閾値
- 法的下限[reg]：暗所15%以上を含む400 kmの混合経路で、TP_Dが全体で距離の90%以上、道路種別ごとに80%以上。新しい制限は標識通過後2.0秒以内（20 km/h未満では10 m以内）に利用可能。運行国が既知であること（<https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf>）。
- 消費者テスト水準[reg]：3〜5か国にわたり夜間10%を含む2000 kmの走行で、Euro NCAPの距離KPIが80%超、イベントKPIが印字された閾値超（<https://www.euroncap.com/media/91705/euro-ncap-protocol-safe-driving-vehicle-assistance-v11.pdf>）。
- OpenTSRのリリースゲート[proposed]：国ごとに、イベントベースの精度（ThatchamやEuro NCAPと同様に、各標識通過イベントを1回として数える）が明示標識で95%以上、暗黙標識と条件付き標識で90%以上であり、非妥当な値がゼロであること（英国型の失敗のような5/10/15/100 mphがないこと、<https://www.autocar.co.uk/node/375029>）。フレーム単位の指標（mAP、top-1）は報告するがゲートには使わない。レイテンシ予算は高速道路の速度で2.0秒ルールに適合すること。

リリースごとに提供するロバスト性評価[proposed]
- CURE-TSR形式の深刻度曲線：12種別×5段階（<https://zenodo.org/record/3903066>）と7種類のGTSRB劣化分割（<https://huggingface.co/api/datasets/tanganke/gtsrb?full=true>）について、チャレンジ種別と段階ごとの精度を、リリースごとに表として報告する。
- 夜間と雨天のスライス：評価走行の15%以上を夜間とする[reg]。ZOD形式の条件別内訳（晴れ、曇り、雨、霧、雪。昼、薄暮、夜）（<https://arxiv.org/pdf/2305.02008>）。
- 国間テスト：TS-1Mの地域間プロトコルに従い、EUで学習して日本と米国でテストし、その逆も行う（<https://arxiv.org/pdf/2603.23034>）。

リリースごとに提供するセキュリティ評価[proposed、C節に基づく]
- REAPによる現実的な変換下でのパッチ攻撃（Mapillaryの標識。デジタルでの成功はリスクを過大評価するという論文の知見と比較する）（<https://arxiv.org/abs/2212.05680>）。公開する検出器に対するホワイトボックスでのRP2とShapeShifterの再現（MITとBSD-3のコード）。
- 自然現象型攻撃：日本の一時停止標識と徐行標識に対する、逆三角形マスクを用いたShadowAttack（MIT）（<https://raw.githubusercontent.com/hncszyq/ShadowAttack/master/README.md>）。
- システムレベル指標：接近区間におけるSysHAとSysAAを報告し、時間的持続（記憶）のポリシーを文書化する。このポリシーが、隠蔽攻撃と出現攻撃のどちらに晒されやすいかを決めるためである（<https://arxiv.org/pdf/2409.09860>）。
- 出現攻撃と幻影：100〜500 msの間だけ投影またはスクリーン表示した標識で試験し、制限を採用する前に最小持続時間を要求する（<https://nassiben.com/phantoms>）。
- タイポグラフィとVLM経路：VLMが標識の文字を読む場合は、タイポグラフィ攻撃セット（誤った数字や"STOP"を書いた文字ステッカー）を含める（<https://arxiv.org/abs/2405.14169>）。
- サプライチェーン衛生：公開する重みをNeural Cleanse（MIT、GTSRBの例を同梱）で検査し、BackdoorBenchのGTSRBスイート（評価用途のみ、CC BY-NC 4.0）を実行する。ポイズニングを監査できるよう、学習データのハッシュを公開する（<https://raw.githubusercontent.com/bolunwang/backdoor/master/README.md>、<https://raw.githubusercontent.com/SCLBD/BackdoorBench/master/README.md>）。
- モデルカード：脅威モデル、実施した攻撃、その成功率、既知の未防御ケース（Teslaの取扱説明書にあるようなデジタル標識と仮設標識）を明記する。

### D.2 攻撃・防御コードベースの一覧表

| 名称 | 年 | 発表先 | 対象 | 物理 / デジタル | コードURL | ライセンス（shields.io） | スター数 | 最終コミット |
|---|---|---|---|---|---|---|---|---|
| RP2 | 2018 | CVPR | LISA-CNN、GTSRB-CNN、Inception V3 | 物理 | https://github.com/evtimovi/robust_physical_perturbations | MIT | 112 | 2019年12月 |
| ShapeShifter | 2018 | ECML-PKDD | Faster R-CNN（一時停止標識） | 物理 | https://github.com/shangtse/robust-physical-attack | BSD-3-Clause | 170 | 2020年1月 |
| Rogue Signs / DARTS | 2018 | DLSワークショップ / arXiv | GTSRB上のKeras CNN | 物理 | https://github.com/inspire-group/advml-traffic-sign | MIT | 37 | 2018年4月 |
| SIB | 2019 | ACM CCS | YOLOv3、Faster R-CNN | 物理 | 見つからず | 該当なし | 該当なし | 該当なし |
| GhostImage | 2020 | RAID | LISA分類器、CIFAR、ImageNet | 物理（プロジェクタ） | https://github.com/Harry1993/GhostImage | MIT | 3 | 2020年6月 |
| Phantom of the ADAS | 2020 | CCS | Tesla Model X、Mobileye 630 | 物理 | 見つからず | 該当なし | 該当なし | 該当なし |
| SLAP | 2021 | USENIX Security | YOLOv3、Mask R-CNN、LISA-CNN、GTSRB-CNN | 物理（プロジェクタ） | https://github.com/ssloxford/short-lived-adversarial-perturbations | GPL-3.0 | 28 | 2021年10月 |
| AdvLB | 2021 | CVPR | ImageNet分類器 | 物理（レーザー） | https://github.com/RjDuan/Advlight | MIT | 34 | 2021年8月 |
| They See Me Rollin' | 2021 | arXiv | BDD100K上のFaster R-CNN、SSD | 物理（レーザー） | https://github.com/ssloxford/they-see-me-rollin | GPL-3.0 | 8 | 2021年10月 |
| Invisible Perturbations | 2021 | CVPR | ImageNet分類器 | 物理（LED） | 見つからず | 該当なし | 該当なし | 該当なし |
| OPAD | 2021 | ICCVW | 3D物体に対する分類器 | 物理（プロジェクタ） | 見つからず | 該当なし | 該当なし | 該当なし |
| Translucent Patch | 2021 | CVPR | 物体検出器（一時停止標識） | 物理（レンズ） | 見つからず | 該当なし | 該当なし | 該当なし |
| Camera Stickers | 2019 | ICML | ResNet-50 ImageNet | 物理（レンズ） | 見つからず | 該当なし | 該当なし | 該当なし |
| MSF-ADV | 2021 | IEEE S&P | カメラ-LiDAR融合（標識ではない） | 物理 | https://github.com/ASGuard-UCI/MSF-ADV | MIT | 84 | 2021年8月 |
| DRP (Dirty Road) | 2021 | USENIX Security | レーンセンタリング（標識ではない） | 物理 | https://github.com/ASGuard-UCI/DRP-attack | MIT | 38 | 2021年8月 |
| FTE | 2022 | NDSS | YOLOv5 TSR | 物理 | 見つからず | 該当なし | 該当なし | 該当なし |
| ShadowAttack | 2022 | CVPR | GTSRB、LISAのCNN | 物理 | https://github.com/hncszyq/ShadowAttack | MIT | 64 | 2025年9月 |
| TPatch | 2023 | USENIX Security | YOLOv3/v5、Faster R-CNN、分類器8種 | 物理（音響トリガー） | https://github.com/USSLab/TPatch | 指定なし | 24 | 2023年8月 |
| REAPベンチマーク | 2023 | ICCV | Mapillary標識上の検出器 | デジタル（現実的変換） | https://github.com/wagner-group/reap-benchmark | 識別不可 | 34 | 2023年10月 |
| Wang et al.（商用TSR） | 2025 | NDSS | 量産車4台 | 物理 | <https://sites.google.com/view/av-ioat-sec/commercial-tsr-test>（未取得） | 不明 | 該当なし | 該当なし |
| DiffSign | 2025 | arXiv | 検出器8種、商用TSR 2種 | 物理 | 公開予定 | 不明 | 該当なし | 該当なし |
| DeepBillboard | 2018 | arXiv（ICSE 2020は未検証） | 操舵モデル（標識ではない） | 物理 | https://github.com/deepbillboard/DeepBillboard | 指定なし | 11 | 2019年3月 |
| Feature Squeezing | 2018 | NDSS | 検知型防御 | デジタル | https://github.com/mzweilin/EvadeML-Zoo | MIT | 187 | 2018年12月 |
| SentiNet | 2020 | DLSワークショップ | パッチ/トロイ検知型防御 | 物理パッチに対するデジタル | 見つからず | 該当なし | 該当なし | 該当なし |
| 認証付きパッチ防御（Chiang） | 2020 | ICLR | MNIST、CIFAR-10 | デジタル | https://github.com/Ping-C/certifiedpatchdefense | BSD-2-Clause | 34 | 2020年9月 |
| (De)Randomized Smoothing | 2020 | NeurIPS | CIFAR-10、ImageNet | デジタル | https://github.com/alevine0/patchSmoothing | 識別不可 | 17 | 2022年8月 |
| PatchGuard | 2021 | USENIX Security | ImageNet、CIFAR-10 | デジタル | https://github.com/inspire-group/PatchGuard | MIT | 74 | 2022年7月 |
| BadNets（オリジナル） | 2017 | arXiv | 米国道路標識上のFaster R-CNN | デジタル（ステッカートリガー） | https://github.com/Kooscii/BadNets | 指定なし | 107 | 2018年5月 |
| BadNets（PyTorch） | 2022 | コミュニティ | 分類器 | デジタル | https://github.com/verazuo/badnets-pytorch | MIT | 197 | 2022年9月 |
| TrojanNN | 2018 | NDSS | 顔、音声、運転のモデル | デジタル | https://github.com/PurduePAML/TrojanNN | 指定なし | 192 | 2022年3月 |
| Neural Cleanse | 2019 | IEEE S&P | バックドア検知（GTSRBモデルを含む） | デジタル | https://github.com/bolunwang/backdoor | MIT | 326 | 2020年2月 |
| BackdoorBench | 2022+ | NeurIPS D&B（未検証） | CIFAR、GTSRB、Tiny ImageNet | デジタル | https://github.com/SCLBD/BackdoorBench | CC BY-NC 4.0（README） | 619 | 2025年7月 |
| TrojanZoo | 2020+ | EuroS&P（未検証） | バックドアツールボックス | デジタル | https://github.com/ain-soph/trojanzoo | MIT | 304 | 2025年8月 |
| SafeBench | 2022 | NeurIPS D&B | CARLA運転エージェント | シミュレーション | https://github.com/trust-ai/SafeBench | MIT | 164 | 2023年9月 |

スター数と最終コミットの値：<https://img.shields.io/github/stars/OWNER/REPO.json>および.../last-commit/...から2026-10-31に取得。ライセンス：BackdoorBench（READMEの記述）を除き<https://img.shields.io/github/license/OWNER/REPO.json>から取得。

### D.3 取得できなかったページ

- <https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng>と<https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng>（HTMLページはメタデータのみを返した）。<https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32021R1958>およびHTML版へのcurlは、空の本文とともにHTTP 202を返した（ボット対策）。PDFは取得ツールで読んだが約94,000文字までに限られたため、ベルギー以降のAnnex II（ドイツおよびその他25の加盟国）と、改正規則Regulation (EU) 2023/1675の内容は読んでいない。
- <https://lexaris.de/book/version/documentflat/head/2056920>（同様にベルギーで途切れた）。
- <https://www.euroncap.com/media/79883/euro-ncap-assessment-protocol-sa-safe-driving-v103.pdf>（クライアントエラー。代わりにv10.2を使用）。
- <https://www.whatcar.com/news/intelligent-speed-assist-systems-fail-to-read-road-signs-25-of-the-time-research-finds/n29166>（クライアントエラー。Autocar版を使用）。
- <https://cncap.obs.cn-north-4.myhuaweicloud.com/cms/files/909314048149975040.pdf>（C-NCAP 2024 Appendix L。許可が得られなかった）。C-NCAPのISLSの詳細はLeddarTechのホワイトペーパーに依拠している。
- i-VISTAのTSR試験仕様：一次情報のページが見つからなかった。
- <https://ja.wikipedia.org/wiki/日本の道路標識>（許可が得られなかった）。
- <https://commons.wikipedia.com/wiki/File:Japan_road_sign_319.svg>（404）とWikimedia Commons API（HTTP 429のレート制限）。Commonsのファイルデータはosmand.netのミラーから取得した。
- <https://docs.nvidia.com/drive/driveworks-3.5/signnet_JP_v1_0.html>と<https://docs.nvidia.com/drive/archive/driveworks-3.0/signnet_mainsection.html>（汎用のドキュメントページにリダイレクトされた。NVIDIAの日本向けSignNetのクラス一覧は読めなかった）。
- <https://github.com/olivesgatech/CURE-TSR>（許可が得られなかった。ZenodoとIEEE DataPortを使用）。
- <https://www.continental-automotive.com/en/components/cameras/multi-function-mono-camera-mfc526.html>（<https://www.aumovio.com/...>にリダイレクトされ、許可が得られなかった）。ZFやDensoのTSRページは取得していない。
- <https://sites.google.com/view/av-ioat-sec/commercial-tsr-test>（許可が得られなかった。Wang et al. NDSS 2025のコードの公開状況は不明）。
- <https://laws.e-gov.go.jp/api/1/lawdata/335M50000002003>は"0 results"を返した（誤った法令ID）。正しいID 335M50004002003を法令一覧APIで見つけて使用した。
- TPatchとSLAPのUSENIXページは取得していない。発表先はリポジトリとarXivによる。
- 時間不足のため試みなかったもの：Euro NCAPの車種別Speed Assistanceデータシート、標識の誤読に関するNHTSAリコールデータベースの検索、UN R157/R89の条文、IEEE T-ITSのCURE-TSD論文、torchvisionのGTSRB分割のドキュメント、PASSシミュレータとCARLAの標識アセット一覧。
