# What "commercial-grade" traffic sign recognition means, and robustness and security resources

Survey compiled on 2026-10-31 from live web pages (claude.ai research session, pingfoo/OpenTSR).
Every factual claim carries the URL on which it was verified; "not verified" or "unknown" means
the fact could not be checked in this session. Where sources disagree both values are given.
Datasets in general, classic open-source implementations and VLM approaches are covered by the
sibling files `datasets.md`, `oss_implementations.md` and
`vlm_and_foundation.md`; this file only touches datasets when they are robustness or
corruption benchmarks (section B). Section D.1 is the proposed OpenTSR target specification,
D.2 the table of attack and defence code bases, D.3 the list of pages that could not be fetched.

Conventions:
- "ISA" = Intelligent Speed Assistance (EU type-approval function); "SLIF" = speed limit
  information function; "TP_D" = the ISA true-positive distance ratio defined in Annex I of
  Regulation (EU) 2021/1958.
- "explicit sign" = a numerical speed limit sign; "implicit sign" = a sign that implies a limit
  without a number (city entry, motorway start, end of limit, zone signs).
- "physical" = the attack was demonstrated with a printed, projected or illuminated object in
  front of a real camera; "digital" = pixels only.
- Star counts and licenses of GitHub repositories come from shields.io badges fetched on
  2026-10-31 and are approximate; "not identifiable" is GitHub's own wording for a LICENSE file
  it cannot classify.

## 0. Key takeaways

1. The only binding, quantified definition of "commercial-grade" TSR found is the EU ISA
   delegated regulation: on a real-world drive of 400 km (urban, rural, motorway each at least
   25 percent, darkness at least 15 percent) the speed limit perceived by the system must be
   correct for at least 90 percent of the distance overall and at least 80 percent on each road
   type, and a new limit must be determined within 2.0 s of passing the sign (10 m when slower
   than 20 km/h). It covers explicit numerical signs, implicit signs (city limits, motorway,
   zones, end of limit) and variable message signs; conditional plates may be ignored in favour
   of the "most common condition" (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf).
   It applies to new vehicle types from 6 July 2022 and to all new registrations from 7 July 2024
   (https://www.legislation.gov.uk/eur/2019/2144/annex/II/2020-12-31).
2. Euro NCAP goes further than the law: since 2023 it scores conditional (rain, snow, time,
   distance, lane arrows, vehicle class), implicit (motorway, city entry/exit, residential) and
   dynamic (incl. roadworks) limits
   (https://cdn.euroncap.com/media/77301/euro-ncap-assessment-protocol-sa-safe-driving-v102.pdf),
   and from 2026 it measures SLIF accuracy on a 2000 km drive over 3 to 5 countries with both a
   distance-based KPI (> 80 percent) and an event-based KPI
   (https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Safe_Driving_Vehicle_Assistance_v1_2_b6fb486fa6.pdf).
   Independent event-based testing by Thatcham found a type-approved car at 91.3 percent by
   distance but only 74.3 percent by event (https://www.autocar.co.uk/node/375029), which is why an
   event-based metric should be the headline metric for OpenTSR.
3. Japan has no TSR or ISA requirement: JNCAP 2025 has no sign item
   (https://www.mlit.go.jp/jidosha/anzen/02assessment/data/pamphlet_car_r8.pdf) and MLIT's ASV
   committee published only an ISA "basic design document" in December 2019
   (https://www1.mlit.go.jp:8088/report/press/content/001320703.pdf). The Japanese catalogue
   (標識令, e-Gov XML fetched 2026-10-31) defines 163 sign kinds in 5 categories (案内 41, 警戒 26,
   規制 59, 指示 14, 補助 23) or 237 numbered variants when A/B/C forms are counted; the pictograms
   are exempt from copyright under Article 13 of the Copyright Act, which is how Wikimedia Commons
   hosts them as SVG (https://download.osmand.net/wiki/File:Japan_road_sign_302.svg).
4. The NDSS 2025 study by Wang et al. (UC Irvine) shows that academic physical attacks transfer
   poorly to production TSR: 6.67 percent success against four 2023-2024 vehicles versus 51.6
   percent in the original papers, largely because commercial systems keep a detected sign in
   memory for up to 60 s ("spatial memorization"), which makes hiding attacks hard and appearing
   attacks easy (https://arxiv.org/pdf/2409.09860). OpenTSR should ship the same temporal
   persistence logic and evaluate attacks at system level (SysHA/SysAA), not per frame.
5. Public robustness tooling that OpenTSR can reuse immediately: CURE-TSR (14 classes, 12
   challenge types x 5 levels, CC BY 4.0 on Zenodo, https://zenodo.org/record/3903066), the
   seven ImageNet-C style corruption splits of GTSRB on Hugging Face (tanganke/gtsrb, no license
   tag, https://huggingface.co/api/datasets/tanganke/gtsrb?full=true), REAP (14,000 real
   Mapillary signs with geometric and lighting transforms for realistic patch evaluation, ICCV
   2023, https://arxiv.org/abs/2212.05680), ShadowAttack (CVPR 2022, MIT, includes adversarially
   trained GTSRB/LISA models), SLAP (USENIX Security 2021, GPL-3.0, includes GTSRB-CNN and LISA-CNN
   with adversarially trained variants) and Neural Cleanse (S&P 2019, MIT, ships an infected
   GTSRB model) for backdoor scanning of released weights.

---

## A. Regulatory and consumer-test requirements

### A1. EU General Safety Regulation, Regulation (EU) 2019/2144

- Article 6(2) sets the minimum requirements for intelligent speed assistance: (a) the driver
  is made aware, through the accelerator control or other appropriate feedback, that the speed
  limit is exceeded; (b) it must be possible to switch the system off, speed limit information
  may still be shown, and the system returns to normal operation at each master-switch
  activation; (c) feedback is based on road signs and signals, infrastructure signals or
  electronic map data, or both; (d) "it shall not affect the possibility, for the drivers, of
  exceeding the system's prompted vehicle speed"; (e) performance targets aim to avoid or
  minimise the error rate in real driving conditions
  (https://legislation.gov.uk/eur/2019/2144/chapter/II/adopted/data.pdf; only (d) is a verbatim
  quote, (a), (b), (c) and (e) are the fetch tool's paraphrase of the article).
- Article 6(6) requires the Commission to adopt delegated acts with detailed test procedures and
  technical requirements, published at least 15 months before the Annex II dates
  (https://legislation.gov.uk/eur/2019/2144/chapter/II/adopted/data.pdf).
- Article 19: the Regulation "shall apply from 6 July 2022"
  (https://legislation.gov.uk/eur/2019/2144/article/19/2020-12-31/data.pdf).
- Annex II row D8 "Intelligent speed assistance" applies to M1, M2, M3, N1, N2, N3 with note B:
  "Date for refusal to grant EU type-approval: 6 July 2022" and "Date for the prohibition of the
  registration of vehicles ... 7 July 2024"
  (https://www.legislation.gov.uk/eur/2019/2144/annex/II/2020-12-31; the UK mirror of the EU
  text was used because EUR-Lex HTML pages returned no body text, see D.3).

### A2. Delegated Regulation (EU) 2021/1958 (the ISA test procedure)

Source: the Official Journal PDF, https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf,
and the consolidated version of 2023-09-21,
https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf. The fetch tool read roughly
the first 94,000 characters of each PDF, which covers Articles 1 to 6, the whole of Annex I and
the start of Annex II (explanatory notes and Belgium); the remaining Member State tables of Annex
II were not read (see D.3). The consolidated version lists one amending act, Delegated
Regulation (EU) 2023/1675 of 26 June 2023 (OJ L 216, 1.9.2023); what it changed could not be
verified in this session.

Dates and scope
- Article 6: applies from 6 July 2022; Article 4(2): approval authorities report aggregated data
  to the Commission on 7 July 2024 and every six months for two years; recital 9: assessment of
  feedback methodologies to be completed by 31 December 2025
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf).

Definitions in Annex I, point 1 (numbering as reported by the fetch tool)
- 1.3 "speed limit information function (SLIF)": speed limit determination system plus HMI.
- 1.6 "perceived speed limit": "the applicable speed limit as obtained by the speed limit
  determination system".
- 1.9 "catalogue of road signs": national and regional variants of sign types and variable
  message sign types, set out in Annex II (Article 3).
- 1.10 "applicable road sign": a catalogue sign that applies to at least one lane; includes
  conventional and variable message signs, excludes pavement markings.
- 1.11 "explicit numerical speed limit sign": "an applicable road sign which shows a temporary
  or permanent numerical value".
- 1.13 "implicit speed limit sign": shows no number (implicit non-numerical) or "shows a
  strikethrough numerical value" (implicit numerical).
- 1.14 "national speed limit": the default maximum for the road type in a Member State.
- There is no definition of "conditional speed limit" or "supplementary plate"; the nearest
  concept is "special variable conditions" (trailer, time of day, weather, passengers) in point
  3.4.2.3.2, and point 3.4.1.2.1 states "Display of additional sub-sign information is permitted
  in all cases" (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf;
  https://lexaris.de/book/version/documentflat/head/2056920).

What must be recognised (Annex I, point 3.4.2)
- 3.4.2.1 country of operation: detected automatically or selected by the driver; vehicles with
  limited regional capability carry the remark "ISA functionality is limited to country or
  region of operation".
- 3.4.2.2 explicit numerical signs: the limit must be determined "at the latest 2.0 seconds"
  after the manufacturer-declared reference point passes the sign; below 20 km/h the limit may be
  determined up to 10 m rearward of that point. Sign design, placement and condition must conform
  to national standards, the sign must be unobstructed for at least 1.0 s, and weather that
  impairs visibility (fog, heavy rain, snow) is excluded (3.4.2.2.3(d)).
- 3.4.2.3 implicit signs: the system must determine limits from all applicable signs in the
  Annex II catalogue with the same 2.0 s / 10 m timing; it "need not account for special
  variable conditions (such as trailer status or time of day)" and otherwise defaults to the
  assumed most common condition (3.4.2.3.2).
- 3.4.2.5 real-world reliability: TP_D at least 90 percent of total distance and at least 80
  percent on each of the three road types; reliability to be maintained for at least 14 years;
  map-based data updates at least annually, free of charge, until 7 years after manufacture
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf).
- 3.2.3: a tolerance of up to 3.0 percent on the perceived limit is allowed when triggering
  information and warnings (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf).

Test procedure (Annex I, point 4)
- 4.2 test for implicit signs: a minimum of three different implicit signs selected by the
  technical service; the manufacturer documents compliance with the other catalogue signs; on
  public roads the speedometer speed may be at most 20 percent below the sign value, on test
  tracks at least 10 percent above; pass criterion is the expected feedback value displayed
  within 2.0 s (10 m below 20 km/h); compliance may be combined with the real-world test
  (4.2.4.2).
- 4.3 real-world test: public roads in the EU agreed between manufacturer, technical service and
  approval authority; one consecutive loop; urban, non-urban and motorway each at least 25
  percent of the route; darkness at least 15 percent of the distance; 400 km, which may end early
  after 300 km if TP_D varies by at most plus or minus 5.0 percent over the final 50 km; TP_D =
  d_correct / d_total x 100 percent where d_total is the distance over which a sign or national
  limit applied. No requirement to drive in more than one Member State was found.
- 4.4 speed limit warning function on the test track: initial perceived limit at least 38
  percent above the test limit; four speed bands 1-8, 11-18, 21-28 and 31-38 percent above the
  limit; combined visual plus acoustic/haptic warning must start within 6.0, 5.0, 4.0 and 3.0 s
  respectively plus the determination time; visual warning within 1.5 s.
- 4.5 speed control function: stabilised speeds 45-50 km/h urban, 75-80 inter-urban, 125-130
  motorway; intervention within 1.5 s after the limit is set
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf).
- Point 5 exclusions from the TP_D calculation: partly obstructed or wrongly oriented signs
  (5.3.1), missing or ambiguously located signs (5.3.2), conflicting or ambiguous information
  about vehicle categories, time of day or weather (5.3.3), very life-like non-applicable signs
  that cause a stationary false positive (5.3.4), signs changed or added within 12 months before
  the test (5.3.5); the affected distances are removed from both d_total and d_correct (5.3.6)
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf).

Annex II catalogue (what was readable)
- Explanatory notes: "N" = national limit for the road class; "V" = variable message signs,
  which may display any explicit or implicit sign of the national table and whose "refresh rate
  of the signs shall be at least 1 000 Hz"; "S" = feedback suspended under points 3.5.6 or 3.6.3;
  city-limit entry and exit signs "share recognisable characteristics but may vary in shape, size
  and wording" and the system must handle these; M2 under 3.5 t follows M1 unless stated
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf).
- Belgium (the only table read): explicit numerical C43 at 30 to 120 km/h with expected feedback
  per vehicle category M1, M2, M3, N1, N2, N3; implicit numerical C45 ("N"); implicit
  non-numerical signs; zones F4a/F4b (30 km/h), ZC43/ZC45 (50 and 70 km/h zones);
  traffic-reduced areas F12a/F12b (20 km/h), F113a/F113b (30 km/h); motorway F5 (120 km/h) and
  F7 (end, "N"); city limits F1 (50 km/h); expressway "None". Heavy-vehicle rows differ for
  at most 7.5 t vs above 7.5 t (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf).
- Germany and the other 25 Member States: not read in this session (not verified). The sign
  classes implied by the Belgian structure are the ones OpenTSR must support for the EU: explicit
  numeric, end-of-limit (struck-through), zone start/end, traffic-calmed area start/end,
  motorway start/end, expressway start/end, city entry/exit, and the national-limit fallback.

### A3. Euro NCAP Speed Assist Systems

2023 to 2025 scheme: Assessment Protocol Safety Assist, Safe Driving v10.2 (June 2023,
implementation 2023), section 4 Speed Assist Systems
(https://cdn.euroncap.com/media/77301/euro-ncap-assessment-protocol-sa-safe-driving-v102.pdf):
- 4.1 records the regulatory context: ISA mandatory for new types from July 2022 and all vehicles
  from July 2024; 4.5.3 announces the 2026 speedometer requirement of -3/+0 km/h.
- 4.4.1 SLIF general requirements: limits shown as a traffic sign in the driver's direct field of
  view (instrument cluster or head-up display) at all times except initialisation; for explicit
  conditional limits the system must either show the applicable limit or indicate that a
  conditional limit exists next to the non-conditional one; the warning function is default ON.
- 4.4 input sources: any SLIF "using all relevant system inputs, for example camera input and
  electronic map based or a combination of both" is eligible, so camera-only, map-only and
  fusion are all allowed.
- 4.4.2 advanced functions require evidence that the feature works during at least 80 percent of
  typical public-road driving in Austria, France, Germany, Italy, Luxembourg, the Netherlands,
  Spain, Sweden and the UK (Norway from 2024), or in at least half of the Application Area.
- 4.4.2.1 scored sign categories (internal 20-point scale): rain/wetness incl. implicit (2 pts,
  show correct limit); snow/ice (2, warning only, ignore if irrelevant); time/season (3);
  distance for/in (1); arrows / lane relevance (1, show or ignore if irrelevant); other vehicle or
  weight categories (1); implicit highway/motorway (2); implicit city entry/exit (3); implicit
  residential zones (2); dynamic signs incl. roadworks (3). Explicit numerical signs are the basic
  SLIF requirement rather than a scored category. Trailer and caravan signs are not named
  separately (covered by vehicle/weight categories).
- 4.4.2.2 road features (10 internal points): curves, roundabouts, junctions without right of
  way, traffic lights, stop, yield and no-entry signs. 4.4.2.3 local hazards (10 internal
  points). 4.4.2.4 system updates: quarterly updates 5 points, continuous (daily/weekly) 10.
- 4.6 points: SAS total 3.00 of the Safety Assist box; SLIF 1.50 (basic 0.50, advanced limits
  0.25, road features 0.25, local hazards 0.25, system updates 0.25); Speed Control Function 1.50
  (SLF 0.50, ISL not default ON 1.00, iACC or ISL default ON 1.50).
- Timing in this protocol: iACC must adopt or offer an adjusted set speed within 5 s of a limit
  change (4.5.2.2); limiting within 30 s (4.5.2.1); no 2 s / 10 m sign-display rule appears here.
- v10.3 (https://www.euroncap.com/media/79883/euro-ncap-assessment-protocol-sa-safe-driving-v103.pdf)
  could not be fetched (client error); v10.2 was used instead.

2026 scheme: Protocol Safe Driving, Vehicle Assistance (version labels inconsistent: cover
"Version 1.2 July 2026", title block "Version 1.1.0 July 2026"; implementation January 2026)
(https://cdn.euroncap.com/cars/assets/Euro_NCAP_Protocol_Safe_Driving_Vehicle_Assistance_v1_2_b6fb486fa6.pdf;
v1.1 at https://www.euroncap.com/media/91705/euro-ncap-protocol-safe-driving-vehicle-assistance-v11.pdf):
- Speed Assistance total 20 points: SLIF 12 (accuracy 4, advanced limits 3, local hazards 3,
  system updates 2), Speed Control Function 8 (ISL 5, iACC 8; full score needs speedometer
  accuracy -3/+0 km/h, halved at -5/+0).
- 1.2.1 SLIF accuracy: "KPIDistance = Dcorrect / Dtotal" with Dcorrect the "Total distance with
  correct speed limit displayed (km), applicable to ALL speed limit elements"; threshold "> 80%*"
  for 2 points. "KPIEvent = Ecorrect / Etotal" with Ecorrect the "Total number of correctly
  identified events of the claimed speed limit elements"; threshold printed as "> 80 -10%*" for
  2 points (the text is garbled in both v1.1 and v1.2 extractions; read it as a threshold in the
  70 to 80 percent range and verify against the PDF before quoting). Footnote: "Assuming perfect
  ground truth".
- Route: "a total length of at least 2000 km, covering between three and five countries within
  the Euro NCAP Application Area", at least 20 percent of distance per country, "A minimum of 10%
  of the total mileage shall be conducted during nighttime", per country 5-10 percent urban,
  20-30 percent interurban, at least 60 percent highway. Evaluated on-road by the test laboratory.
- 1.2.1 timing: if the indicated speed is above 20 km/h the SLIF change must occur within 2 s of
  the limit change; at 20 km/h or below within 2 s or 10 m (v1.2 extraction; the v1.1 extraction
  did not surface this rule, so verify in the PDF).
- 1.2.2 advanced limits (3 points): conditional 2 (rain/wetness, snow/icy, time/season, distance
  for/in, arrows lane-relevant and not, vehicle categories incl. caravan), implicit 0.5
  (motorway/highway, city entry/exit, residential zones), dynamic 0.5 (incl. roadworks).
  Eligibility: advanced information during at least 80 percent of typical driving in Austria,
  France, Germany, Italy, Luxemburg, the Netherlands, Spain, Sweden, United Kingdom and Norway,
  or in at least half of the Application Area countries. Manufacturer dossier SD-301 with about
  400 km of on-road evidence; HiL or test-track data allowed when on-road evaluation is not
  feasible.
- Map fusion is not required; camera, map or both are eligible. Overhead gantries and
  supplementary plates are not named as such.
- Rain appears only as a conditional sign category; there is no separate rain test condition in
  either Euro NCAP protocol or in the ISA regulation (the latter excludes visibility-impairing
  weather).

### A4. Independent real-world checks of production ISA

- Thatcham Research (UK) tested three ISA systems (about 10 percent of systems in new cars) with
  an event-based metric: MG ZS (2024) 91.3 percent by distance, "just above the EU's 90%
  threshold", but 74.3 percent by event (about one in four limit changes wrong); BMW i5 98.39
  percent by distance, 90.3 percent by event. All three cars displayed limits that are not legal
  in the UK, including 5, 10, 15 and 100 mph, which Thatcham says could cause unwanted braking or
  acceleration with ACC. The article gives no test date or route; it states the EU approval test
  covers 250 miles and Euro NCAP's 60 miles (https://www.autocar.co.uk/node/375029; the What Car?
  version https://www.whatcar.com/news/intelligent-speed-assist-systems-fail-to-read-road-signs-25-of-the-time-research-finds/n29166
  could not be fetched).
- Euro NCAP Assisted Driving gradings (October 2024): BMW i5 Assistance Competence 76 percent,
  Safety Backup 90; Mercedes C-Class 85 / 97; Volvo EC40 66 / 78 ("good"); VW ID.7 70 / 80
  ("good"); BYD Atto 3 55 / 35 ("not recommended") with its speed assistance "not interpreting
  road signs correctly" (https://paultan.org/2024/10/24/euro-ncap-grades-5-cars-for-assisted-driving-bmw-i5-mercedes-c-class-volvo-ec40-vw-id-7-byd-atto-3/;
  secondary source, Euro NCAP datasheets not fetched).

### A5. UNECE

- No UN Regulation on TSR or ISA was found. The draft EU implementing regulation with ISA
  information-document templates references only UN Regulation No 39 (speedometer and odometer)
  for speed items and UN R160 for the event data recorder; it names no UNECE act on ISA or speed
  limit information (https://members.wto.org/crnattachments/2024/TBT/EEC/24_01641_01_e.pdf).
  Any claim that UN R157 (ALKS) or R89 (speed limiters) imposes sign-reading requirements was not
  verified in this session.

### A6. United States

- No federal TSR or ISA requirement. NTSB (board meeting 14 November 2023, North Las Vegas crash
  of 29 January 2022, nine fatalities) recommended that NHTSA "Require ISA systems that, at a
  minimum, warn a driver a vehicle is speeding", incentivise ISA through NCAP (reiterating a 2017
  recommendation), and that 17 manufacturers "Install ISA in all new passenger vehicles that, at a
  minimum, warns drivers when a vehicle is speeding". The release describes ISA as GPS plus a
  speed-limit database, also using onboard cameras; no performance requirement
  (https://www-s.ntsb.gov/news/press-releases/Pages/NR20231114.aspx).
- NHTSA's reply of 3 April 2024: no rulemaking commitment; a technology scan in FY2024 to "assess
  the capabilities and limitations of technologies to accurately determine the speed limit across
  a wide range of roadway types", an NCAP final decision notice expected in 2024, Guideline 19
  update in 2025 (https://www.nhtsa.gov/document/nhtsa-ntsb-response-april-2024-intelligent-speed-assistance-isa-interlock-speeding-ncap).
  Whether the 2024 NCAP decision added ISA was not verified.
- California SB 961 (Sen. Scott Wiener) would have required, from 2030, a one-time visual and
  audio alert at 10 mph or more over the limit using GPS and a state speed-limit list; vetoed on
  28 September 2024 citing federal preemption and NHTSA's ongoing ISA evaluation
  (https://timesofsandiego.com/politics/2024/09/28/governor-vetoes-senate-bill-calling-for-new-cars-to-alert-drivers-when-speeding/).
- IIHS "Intelligent Speed Assistance Test and Rating Protocol", DRAFT Version I, June 2026:
  closed track, daylight only (above 2,000 lux), only numerical mph signs (24 x 30 in, about 12 ft
  from the lane edge and 5 ft high per MUTCD 11th ed. Table 2B-1); Assessment 2 uses two signs at
  least 500 ft apart, higher limit first, three runs; Assessments 3 and 4 use bands of 5-9, 10-14
  and 15+ mph over 25 and 45 mph limits; "the rating system will be determined after an initial
  round of testing"; no night, no implicit or conditional signs
  (https://www.iihs.org/media/4e417587-ab67-489c-a3bb-7fe1dd87fe1b/COWTLw/Ratings/Protocols/future%20programs/Intelligent_speed_assistance_protocol.pdf).
- MUTCD 11th Edition: Final Rule in the Federal Register 19 December 2023, effective 18 January
  2024, States must adopt it "within two years from the effective date"
  (https://mutcd.fhwa.dot.gov/mutcd_news.htm).

### A7. China

- C-NCAP 2024 (C-NCAP MANAGEMENT REGULATION 2024 EDITION) has an "Appendix L Active safety ADAS
  test protocol" (https://www.c-ncap.org.cn/cncap-guicheng-detail?id=8); the PDF itself
  (https://cncap.obs.cn-north-4.myhuaweicloud.com/cms/files/909314048149975040.pdf) could not be
  fetched. A LeddarTech white paper on C-NCAP 2024 lists "Intelligent speed limit system (ISLS)"
  (warns when speed is non-compliant) and traffic sign recognition as a sub-item of "road feature
  recognition (RFR)" among the active safety items, with no scoring shown
  (https://leddartech.com/app/uploads/dlm_uploads/2024/06/White-Paper_A-Comprehensive-Overview-of-C-NCAP-2024-Regulations_V1.0_EN-5.pdf;
  secondary source, weights not verified).
- C-ICAP 2024 (intelligent connected vehicle index) appendices cover driving assistance,
  navigation assistance, parking, cockpit and privacy; the page text mentions no TSR item
  (https://www.c-ncap.org.cn/cncap-guicheng-detail?id=15). i-VISTA TSR test specifications: not
  verified (no primary page found).
- Sign standard: GB 5768.2-2022 "Road traffic signs and markings, Part 2: Road traffic signs",
  issued 15 March 2022 by SAMR and SAC, effective 1 October 2022; the standard text is treated as
  public domain on Wikimedia Commons because mandatory standards are "technical standards with
  legal natures" outside copyright under Article 5 of the PRC Copyright Law (National Copyright
  Administration 权司[1999]第50号) (https://download.osmand.net/wiki/File:GB_5768.2-2022.pdf).

### A8. Japan

Requirements
- JNCAP (自動車アセスメント) 2025 preventive safety items: 衝突被害軽減ブレーキ (pedestrian day, night
  with and without street lights, cyclist, right-turn oncoming vehicle, turning with crossing
  pedestrian), 車線逸脱抑制, 高機能前照灯, ペダル踏み間違い時加速抑制; preventive safety full score 85.8,
  collision safety 100, 事故自動緊急通報 8, total 193.8. No sign recognition, speed limit or ISA item
  (https://www.mlit.go.jp/jidosha/anzen/02assessment/data/pamphlet_car_r8.pdf; the pamphlet is
  labelled 2025 although the file name says r8).
- MLIT 先進安全自動車推進検討会, 自動速度制御装置（ISA）基本設計書, formulated 17 December 2019
  (令和元年12月): purpose "ドライバーの不注意（漫然運転）や誤操作による速度超過を抑制することを目的とした装置と
  する"; speed limit sources are camera recognition of road signs, navigation map data and
  communication (2.1); upper limit set or changed within 5 s of a limit change, control begins
  within 30 s when a lower limit is set ("上限速度が現在の車速より低い速度に設定された場合は、設定された後遅く
  とも 30 秒以内に、新しい上限速度に合わせて車速の制御を開始すること", 2.3.3.1); offset up to plus or minus
  10 km/h; four-wheeled vehicles only; no deployment deadline
  (https://www1.mlit.go.jp:8088/report/press/content/001320703.pdf). A 2024 article confirms
  ISA in Japan "is still at the study stage" (https://kuruma-news.jp/post/794694). No Japanese
  TSR or ISA requirement exists as of this survey (no primary source stating one was found).
- Statutory speed change relevant to implicit limits: from 1 September 2026 the 法定速度 on
  生活道路 (roads without a centre line or median) drops from 60 to 30 km/h under the amended
  道路交通法施行令; posted signs override it; 60 km/h remains on roads with a centre line and on
  motorways (https://web.pref.hyogo.lg.jp/kf13/press/20260528.html). An implicit-limit model for
  Japan therefore needs "centre line present" as a feature, not only signs.

The Japanese sign catalogue (道路標識、区画線及び道路標示に関する命令, 標識令)
- Legal basis: 昭和三十五年総理府・建設省令第三号, promulgated 17 December 1960 under 道路法第45条第2項 and
  道路交通法第9条第3項; Article 1-2: "本標識は、案内標識、警戒標識、規制標識及び指示標識とする". The current
  text was downloaded as XML from the e-Gov law API (law ID 335M50004002003,
  https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003, fetched 2026-10-31); the last two
  amending ordinances listed are 令和八年四月一日内閣府・国土交通省令第二号 and 令和八年六月五日内閣府・国土交通省令第三号.
- Counts parsed from 別表第一 of that XML (names in the 種類 column; "variants" expands A/B/C and
  の2 style sub-numbers):

| Category | Distinct names (種類) | Base numbers | Numbered variants |
|---|---|---|---|
| 案内標識 (guide, 101-126) | 41 | 41 | 89 |
| 警戒標識 (warning, 201-215) | 26 | 23 | 27 |
| 規制標識 (regulatory, 301-332) | 59 | 59 | 74 |
| 指示標識 (indication, 401-409) | 14 | 14 | 16 |
| 補助標識 (supplementary, 501-514) | 23 | 23 | 31 |
| Total | 163 | 160 | 237 |

  (https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003). The MLIT one-page list gives OCR
  counts of about 77 / 26 / 73 / 16 / 31 (total about 220) for the same categories, consistent
  with the variant counts above given OCR noise (https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf).
- 規制標識 relevant to speed: 323 最高速度, 323の2 特定の種類の車両の最高速度, 324 最低速度, 325 自動車専用
  (implies the motorway national limit), 329 徐行; the full regulatory list (301 通行止め to 332
  歩行者等横断禁止, including 303 車両進入禁止, 311-A to F 指定方向外進行禁止, 313 転回禁止, 314/314の2
  overtaking, 315/316 parking, 320/321/322 weight/height/width, 326 一方通行, 327 series lane rules,
  330 一時停止) is in the XML. 補助標識 that change the meaning of a speed sign: 501 距離・区域, 502 日・時間,
  503-A to D 車両の種類, 505 始まり, 506 区間内, 506の2 区域内, 507-A to D 終わり (end of restriction), 509の2
  踏切注意 etc. (https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003). Recent additions visible in
  the list: 特定小型原動機付自転車 signs (309, 325の2, 326の2) and 503の2 遠隔操作型小型車.
- Official colour rules in 別表第二 備考 (useful for synthetic data and for VMS handling): speed and
  dimension signs "文字及び記号を青色、斜めの帯及び枠を赤色、縁及び地を白色" (blue numerals, red ring, white
  ground), and for 最高速度 / 最低速度 displayed by lights "文字及び記号を白色又は黄色、地を黒色とすることができる"
  (white or yellow numerals on black, the usual appearance of Japanese variable speed signs);
  warning signs "縁線、文字及び記号を黒色、縁及び地を黄色"; 一時停止 "文字及び縁線を白色、縁及び地を赤色"; 徐行 and
  前方優先道路 "文字を青色、枠を赤色、縁及び地を白色"; 車両進入禁止 "帯及び縁を白色、地を赤色"
  (https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003).
- English on signs: the 標識令 amendment of April 2017 (effective 1 July 2017) added a 徐行 form
  with "SLOW" under the Japanese text ("「徐行」という日本字の下に「SLOW」という英字を併記した様式を追加します");
  the NPA made the parallel change for 一時停止 and 前方優先道路 signs set by prefectural public safety
  commissions (https://www.mlit.go.jp/report/press/road01_hh_000829.html). Both forms coexist on
  the road, so a Japanese class needs both.
- Where to get an official or openly licensed list with pictograms: (1) MLIT 道路局 one-page PDF
  https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf (no date or license printed; MLIT
  pages are generally under the Government of Japan standard terms, not verified here); (2) the
  e-Gov XML above contains the 様式 drawings only as figure references, not as vector art; (3)
  Wikimedia Commons hosts SVGs named "Japan road sign NNN.svg" (for example 302 車両通行止め, author
  Monaneko, source "道路標識、区画線及び道路標示に関する命令別表第2") under the template "This work is in the
  public domain in Japan according to Article 13 of the Copyright Act of Japan", which exempts
  laws and regulations (https://download.osmand.net/wiki/File:Japan_road_sign_302.svg; the
  Commons API returned HTTP 429 and commons.wikipedia.com returned 404, so coverage of all 237
  variants was not verified). Sign designs defined in the ordinance are therefore usable for
  synthetic training data without a license problem; photographs of signs are a separate matter.

### A9. Sign standards that matter for a global model

- Vienna Convention on Road Signs and Signals (8 November 1968, in force 6 June 1978): 35
  signatories, 75 parties. Parties include Germany (ratified 3 Aug 1978), France, Italy, Austria,
  the Netherlands, Sweden and India; Spain, the United Kingdom, Brazil and the Republic of Korea
  signed but never ratified; Japan, the United States, China, Australia and Canada are not
  listed at all (https://treaties.un.org/doc/Publication/MTDSG/Volume%20I/Chapter%20XI/XI-B-20.en.pdf).
  So the three OpenTSR home markets (Japan, EU, US) follow three different systems: EU
  Vienna-style, US MUTCD, Japan its own ordinance (red-ringed prohibitory signs and black-on-yellow
  warning signs per the colour rules in https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003; the
  diamond warning shape and the inverted-triangle stop sign are visible in the MLIT list
  https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf but shapes were not machine-verified).
- MUTCD 11th Edition (FHWA): effective 18 January 2024 (https://mutcd.fhwa.dot.gov/mutcd_news.htm).
  US speed limits are text signs in mph ("SPEED LIMIT 50", R2-1); the IIHS draft protocol tests
  only these (https://www.iihs.org/media/4e417587-ab67-489c-a3bb-7fe1dd87fe1b/COWTLw/Ratings/Protocols/future%20programs/Intelligent_speed_assistance_protocol.pdf).
- China: GB 5768.2-2022, effective 2022-10-01 (https://download.osmand.net/wiki/File:GB_5768.2-2022.pdf).
- Known cross-country confusions documented in the sources above: (a) units: Thatcham observed UK
  cars showing 5, 10, 15 and 100 mph limits that do not exist in the UK, i.e. numerals read
  without plausibility checks (https://www.autocar.co.uk/node/375029); a km/h model used in the
  UK or US, or an mph model in Japan, produces wrong units silently, so country/region of
  operation is an explicit ISA requirement (point 3.4.2.1,
  https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf); (b) city-limit signs vary
  in "shape, size and wording" across EU countries and are nevertheless required (Annex II
  notes); (c) Japanese variable speed signs are white or yellow digits on black, unlike the static
  blue-on-white sign, and the ordinance allows both (https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003);
  (d) the Japanese 最低速度 (324) and 最高速度 (323) signs share one colour specification in the
  ordinance (blue digits, red ring, white ground, https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003)
  and differ only by a small design element (an underline bar under the digits in the minimum
  speed design; this detail is from the pictograms in https://www.mlit.go.jp/road/sign/sign/douro/ichiran.pdf
  and was not machine-verified), a fine-grained distinction that speed-limit OCR must preserve; (e) Belgium's ISA table shows that the same explicit sign
  implies different expected feedback for heavy vehicles (at most 7.5 t vs above), so the
  "correct" label depends on vehicle category, not only on the sign
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf). Font differences (DIN 1451 in
  Germany, FHWA Series in the US, Japanese sign typefaces) were not verified from a primary
  source in this session.

### A10. What commercial systems claim, and known failures

- Mobileye (press release 18 July 2023): "vision-only Intelligent Speed Assist (ISA) solution",
  runs on EyeQ4 and EyeQ6 by software update, uses OCR for "city entrance identification for
  European-style signs", handles "static and dynamic message speed-limit signage", certified for
  all 27 EU countries plus Israel, Norway, Switzerland and Turkey; "Six independent labs across
  five different European countries have tested and confirmed" compliance; no accuracy figure;
  OEM not named (https://ir.mobileye.com/news-releases/news-release-details/mobileye-launches-first-camera-only-intelligent-speed-assist).
- Bosch "Road sign information" (commercial vehicles): detects and classifies "round, triangular
  and rectangular road signs", speed limits "including the start and ends of stretches where speed
  limits are in force, or where passing is prohibited", "relevant supplementary signs, such as
  time limits", vehicle-type and turn-arrow plates, no entry, stop, right of way, road works;
  "reliably detects road signs whether they are sited on actual signs, variable message systems
  or gantries"; "This function meets the requirement of the General Safety Regulation (GSR)";
  performance improves "In combination with a navigation system or the connected horizon"; no
  numbers (https://bosch-mobility.com/en/solutions/assistance-systems/road-sign-information-cv).
- Tesla owner's manual (Model 3, Japan English), Speed Assist limitations: "Visibility is poor
  and speed limit signs are not clearly visible", "Bright light (such as from oncoming headlights
  or direct sunlight) is interfering with the view of the camera(s)", "Speed limit signs are
  concealed by objects", "The speed limits stored in the map database are incorrect or outdated",
  "Traffic signs that do not conform to standard recognizable formats, such as digital or
  temporary speed signs", "A road or a speed limit has recently changed"; "Do not rely on Speed
  Assist to determine the appropriate speed limit or driving speed"
  (https://www.tesla.cn/ownersmanual/model3/en_jp/GUID-5D3D4014-4E98-45D7-8BBC-F76BCA9CEC05.html).
  Digital and temporary signs are explicitly out of scope for at least one production system.
- Continental/Aumovio, ZF and Denso: product pages could not be fetched or redirected to generic
  pages (see D.3); no verified claims.
- Public failure reports: (1) McAfee ATR, 19 February 2020: a small piece of black tape on a
  35 mph sign made the Mobileye EyeQ3 camera of a 2016 Tesla Model S/X (hardware pack 1) read
  85 mph and the cruise control accelerate toward it; "MobilEye did indicate that the more recent
  version(s) of the camera system address these use cases"; a 2020 vehicle "did not appear to be
  susceptible" in very limited testing
  (https://mcafee.com/blogs/other-blogs/mcafee-labs/model-hacking-adas-to-pave-safer-roads-for-autonomous-vehicles).
  (2) Phantom of the ADAS (Nassi et al., CCS 2020; also CACM, AutoSec 2021): projected or
  billboard-embedded phantom signs fooled Tesla Model X (HW2.5 and HW3) and Mobileye 630; a stop
  sign shown for 125 ms made Autopilot stop; countermeasure "GhostBusters" (four CNN experts on
  light, context, surface, depth; TPR 0.994 at zero FPR); no code or data links on the project
  page (https://nassiben.com/phantoms). (3) Wang et al., NDSS 2025: four 2023-2024 US-market
  vehicles (candidate list Tesla Model 3 2023, Toyota Camry 2023, Nissan Sentra 2023, Mazda CX-30
  2023, Hyundai Tucson 2024; mapping to C1-C4 not disclosed); C1 reads only STOP, C2 STOP and
  speed limits, C3 and C4 speed limits only; see C1 below for results
  (https://arxiv.org/pdf/2409.09860). (4) Thatcham event-based results and Euro NCAP BYD Atto 3
  finding, A4 above. No NHTSA recall concerning sign misreading was verified in this session.

---

## B. Robustness benchmarks and corruption datasets for signs

### B1. CURE-TSR and CURE-TSD (Georgia Tech OLIVES)

- Origin: Dogancan Temel, Gukyeong Kwon, Mohit Prabhushankar, Ghassan AlRegib, Georgia Institute
  of Technology; NIPS 2017 workshop paper (arXiv 1712.02463); the detection counterpart CURE-TSD
  is described in Temel, Chen, AlRegib, "Traffic Sign Detection Under Challenging Conditions: A
  Deeper Look into Performance Variations and Spectral Characteristics", IEEE T-ITS 2019, DOI
  10.1109/TITS.2019.2931429, arXiv 1908.11262
  (https://zenodo.org/record/3903066; https://alregib.ece.gatech.edu/software-and-datasets/cure-tsd-challenging-unreal-and-real-environments-for-traffic-sign-detection).
- Geography and source: real frames from BelgiumTS video sequences; synthetic ("unreal") frames
  rendered in Unreal Engine 4; challenge effects added with Adobe After Effects
  (https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition).
- CURE-TSD size: 49 real and 49 unreal base sequences of 300 frames each; real x 12 challenge
  types x 5 levels + base = 2,989 sequences, unreal x 11 types x 5 levels + base = 2,744; total
  5,733 sequences, about 1.72 million frames; 70/30 split into 34 training and 15 test videos per
  domain; file name `sequenceType_sequenceNumber_challengeSourceType_challengeType_challengeLevel.mp4`
  (https://alregib.ece.gatech.edu/software-and-datasets/cure-tsd-challenging-unreal-and-real-environments-for-traffic-sign-detection).
  Resolution and instance counts: not stated on that page (IEEE DataPort says "more than 2
  million traffic sign instances", https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition).
- CURE-TSR: crops from CURE-TSD, "over 2 million images", 14 classes: speed_limit,
  goods_vehicles, no_overtaking, no_stopping, no_parking, stop, bicycle, hump, no_left, no_right,
  priority_to, no_entry, yield, parking; 12 challenge types with codes 01-12: decolorization,
  lens blur, codec error, darkening, dirty lens, exposure, gaussian blur, noise, rain, shadow,
  snow, haze (00 = no challenge); 5 levels 01 (least) to 05 (most severe); file name
  `sequenceType_signType_challengeType_challengeLevel_Index.bmp`; zip 4.6 GB (DataPort) or 4.9 GB
  (Zenodo) (https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition;
  https://zenodo.org/record/3903066). Train/test split sizes: not stated on either page.
- License: Zenodo record 3903066 (v1, 22 June 2020): "Creative Commons Attribution 4.0
  International" (https://zenodo.org/record/3903066). IEEE DataPort states no license and says
  Open Access files need a free IEEE account while its "How to access" box mentions a
  subscription (https://ieee-dataport.org/open-access/cure-tsr-challenging-unreal-and-real-environments-traffic-sign-recognition).
  Use the Zenodo copy (DOI 10.5281/zenodo.3903066) for redistribution.
- Access: Zenodo (CURE-TSR.zip, md5 49c2f9b24e5e186cac6df136a14d28e0), IEEE DataPort (DOI
  10.21227/n4xw-cg56), code at https://github.com/olivesgatech/CURE-TSR and
  https://github.com/olivesgatech/CURE-TSD (GitHub pages not fetched, D.3).
- Baselines: not verified (the T-ITS paper was not read).
- Known problems: only 14 coarse classes (no speed values), Belgian signs only, synthetic
  weather; still the only sign benchmark with graded severity levels, so useful as a robustness
  regression test rather than as training data.

### B2. GTSRB corruption variants on Hugging Face (tanganke/gtsrb)

- Content: dataset `tanganke/gtsrb` (author Anke Tang, last modified 2024-05-07, 16,418 downloads
  at fetch time) with 43 GTSRB classes in splits train 26,640, test 12,630 and seven corrupted
  copies of the test split, each 12,630 images: contrast, gaussian_noise, impulse_noise,
  jpeg_compression, motion_blur, pixelate, spatter; parquet, 1.03 GB download; license field
  empty (https://huggingface.co/api/datasets/tanganke/gtsrb?full=true;
  https://huggingface.co/datasets/tanganke/gtsrb/raw/main/README.md). The 26,640 / 12,630 split
  matches the torchvision GTSRB split rather than the original 39,209-image training set (the
  torchvision figure was not re-verified here).
- Where the corruptions come from: the card cites AdaMerging (Yang et al., arXiv 2310.02575) and
  two model-merging papers by Tang et al. (arXiv 2312.06173, 2402.00433); AdaMerging states
  "Following Hendrycks & Dietterich (2019), we created 7 corruption test data" and lists exactly
  "Motion Blur, Impulse Noise, Gaussian Noise, Pixelate, Spatter, Contrast and JPEG Compression"
  for GTSRB, Cars, EuroSAT and RESISC45 (https://arxiv.org/pdf/2310.02575, Section on robustness
  and Appendix B). So these splits are the ImageNet-C recipe (Hendrycks and Dietterich, ICLR
  2019) applied to GTSRB by the model-merging community; severity level is not recorded in the
  card (unknown). Example AdaMerging numbers on GTSRB (ViT-B/32, Task Arithmetic vs AdaMerging):
  clean 75.1 vs 96.3, impulse noise 40.4 vs 77.5, gaussian noise 49.4 vs 76.0, pixelate 66.6 vs
  90.6, spatter 54.3 vs 89.4, contrast 70.6 vs (table truncated), JPEG 60.0 vs (truncated)
  (https://arxiv.org/pdf/2310.02575).
- License caution: the card carries no license; the underlying GTSRB images come from the INI
  benchmark whose terms are covered in `datasets.md`. Treat this as an evaluation-only
  mirror until the GTSRB terms are settled.

### B3. Night and weather sources with sign labels (pointers only; details in `datasets.md`)

- Zenseact Open Dataset (ICCV 2023): 100k frames from 14 European countries; 446k traffic sign
  instances in 156 granular classes; weather clear 61.2 percent, cloudy 19.0, rain 15.7, fog 2.1,
  snow 2.0; day 77.3 percent, night 19.0, twilight 3.6; ResNet-50 TSR baseline F1-macro 78.5,
  F1-micro 95.4, top-10 classes 93.4 percent, bottom-10 classes 65.4 percent; "released under the
  permissive CC BY-SA 4.0 license" (https://arxiv.org/pdf/2305.02008). The largest public source
  of night and rain sign crops found.
- TS-1M (HKUST Guangzhou, 2026): 1,267,558 images (1,033,947 train / 233,611 test), 454
  standardized categories in Restriction, Prohibition, Warning, Guidance, merged from GTSRB,
  TT100K, RTSD, MTSD and internet images; includes rain, fog, night and backlit images but
  reports no per-condition accuracy; license not stated; project page
  https://guoyangzhao.github.io/projects/ts1m (https://arxiv.org/pdf/2603.23034).

### B4. Domain shift and cross-country studies

- TS-1M benchmark: models trained on TS-1M evaluated on ten external regional datasets; CLIP
  fine-tuned averages 0.9578 (GTSRB 0.9996, BelgiumTS 0.9921, RTSD 0.9968, MTSD 0.9118, ARTS
  0.9719), ConvNeXt 0.9353, ResNet50 0.9150; rare classes (< 40 samples) CLIP 0.9291 vs ResNet50
  0.8381; low clarity (< 30 x 30 px) 0.9313 vs 0.8768. No pairwise train-on-X test-on-Y
  experiments between individual source datasets (https://arxiv.org/pdf/2603.23034).
- Gan et al. (Hokkaido University), "Cross-domain Few-shot In-context Learning for Enhancing
  Traffic Sign Recognition", arXiv 2407.05814 (July 2024): template-sign descriptions prompt an
  MLLM; evaluated on GTSRB, BelgiumTS and two real-world Japanese datasets; numbers not in the
  abstract (https://arxiv.org/abs/2407.05814v1). Relevant because it treats Japanese signs as a
  target domain.
- Wu et al. (2026), "Efficient Cross-Country Data Acquisition Strategy for ADAS via Street-View
  Imagery", arXiv 2602.01836: pairs ZOD with Mapillary street-view images to pick target-country
  collection sites; on traffic sign detection, "a task particularly sensitive to cross-country
  variations in sign appearance", matches random sampling with half the target-domain data
  (https://export.arxiv.org/api/query?id_list=2602.01836).
- Production evidence of domain shift: Thatcham's UK results (limits of 5, 10, 15, 100 mph
  displayed) and the Euro NCAP BYD Atto 3 finding (A4). The ISA regulation's own answer is the
  mandatory country-of-operation setting (A2).

---

## C. Adversarial attacks and defences on TSR

Metadata (venue, authors, dates) for the arXiv papers below was taken from the arXiv API on
2026-10-31 (https://export.arxiv.org/api/query?id_list=...); repository stars, licenses and last
commits from shields.io; repository contents from raw READMEs. "Code: none found" means no
official repository was identified in this session, not that none exists.

### C1. Physical and digital attacks on sign classifiers and detectors

RP2, Eykholt et al., "Robust Physical-World Attacks on Deep Learning Visual Classification",
CVPR 2018 (arXiv 1707.08945, first version July 2017, "Accepted to CVPR 2018")
- Target: LISA-CNN (US signs, "achieves 91% accuracy") and GTSRB-CNN (German signs with stops
  replaced by US ones from LISA), plus an ImageNet Inception V3 variant. Physical: printed
  posters and stickers on stop signs. Dataset: LISA, GTSRB.
- Code: https://github.com/evtimovi/robust_physical_perturbations, MIT, 112 stars, last commit
  December 2019; README: folders `lisa-cnn-attack`, `gtsrb-cnn-attack`, `imagenet-attack`; bundles
  an old cleverhans copy (MIT) (https://raw.githubusercontent.com/evtimovi/robust_physical_perturbations/master/README.md;
  https://img.shields.io/github/stars/evtimovi/robust_physical_perturbations.json).
- Extension to detectors: Eykholt et al., "Physical Adversarial Examples for Object Detectors",
  USENIX WOOT 2018 (arXiv 1807.07769): disappearance attack on YOLOv2, stop sign missed in over
  85 percent of lab video frames, 72.5 percent (poster) and 63.5 percent (sticker) outdoors;
  transfer to Faster R-CNN. Code for the WOOT version: Wang et al. (NDSS 2025) report they could
  not obtain code for RP2 (detector version), SIB or FTE and re-implemented them with the
  authors' help (https://arxiv.org/pdf/2409.09860).

ShapeShifter, Chen, Cornelius, Martin, Chau, ECML-PKDD 2018 (arXiv 1804.05810, journal_ref
"Joint European Conference on Machine Learning and Knowledge Discovery in Databases, pp. 52-68,
2018")
- Target: Faster R-CNN (TensorFlow Object Detection API) stop sign, targeted (stop sign detected
  as person) and untargeted; physical printed stop signs; later 2D/3D person attacks added.
- Code: https://github.com/shangtse/robust-physical-attack, BSD-3-Clause, 170 stars, last commit
  January 2020; notebook plus Colab, `shapeshifter2d.py` / `shapeshifter3d.py`, Makefile targets
  such as `2d_stopsign_targeted_attack` (https://raw.githubusercontent.com/shangtse/robust-physical-attack/master/README.md).

Rogue Signs / DARTS, Sitawarin, Bhagoji, Mosenia, Chiang, Mittal (Princeton INSPIRE), 2018
- Rogue Signs: 1st Deep Learning and Security Workshop (DLS, with IEEE S&P 2018), arXiv
  1801.02780: "Sign Embedding attack" turns ads and logos into inputs classified as a chosen
  sign, success "in excess of 95%" in physical drive-by tests. DARTS: arXiv 1802.06430
  ("Submitted to ACM CCS 2018", extended version): Out-of-Distribution attacks and a Lenticular
  Printing attack; OOD attacks beat in-distribution attacks against adversarially trained
  classifiers.
- Target: Keras multi-scale CNN and plain CNN on GTSRB (weights `weights_mltscl_dataaug.hdf5`,
  `weights_cnn_dataaug.hdf5` in the repo); physical printed signs. Code:
  https://github.com/inspire-group/advml-traffic-sign, MIT, 37 stars, last commit April 2018;
  includes `OptProjTran.py` (robust optimisation), Carlini-Wagner, random perspective transforms,
  generated adversarial signs in pickle format (https://raw.githubusercontent.com/inspire-group/advml-traffic-sign/master/README.md).

Seeing isn't Believing (SIB), Zhao et al., ACM CCS 2019 (arXiv 1812.10217)
- Hiding attack (feature-interference reinforcement, enhanced realistic constraints) and
  appearing attack ("nested AE" for long and short distance) against YOLOv3 and Faster R-CNN;
  success up to 92.4 percent from 1 to 25 m and -60 to 60 degrees; physical. Code: none found;
  Wang et al. reproduced it without the original code (https://arxiv.org/pdf/2409.09860).

SLAP, Lovisotto, Turner, Sluganovic, Strohmeier, Martinovic, USENIX Security 2021 (arXiv
2007.04137)
- Projector-based short-lived perturbations on a stop sign (and bottle, give way); targets
  YOLOv3, Mask R-CNN, LISA-CNN and GTSRB-CNN, with "usenix21adv" adversarially trained variants;
  profiling of projectable colours at given lux. Physical (indoor and outdoor).
- Code: https://github.com/ssloxford/short-lived-adversarial-perturbations, GPL-3.0, 28 stars,
  last commit October 2021; release `usenix21` with `data.zip` holding the models (cvpr18,
  cvpr18iyswim, usenix21, usenix21adv for both GTSRB and LISA), profiling data, projections and
  the GTSRB/LISA retraining sets; Docker image on Docker Hub; TensorFlow 1.15
  (https://raw.githubusercontent.com/ssloxford/short-lived-adversarial-perturbations/master/README.md).
  GPL-3.0 means OpenTSR can run it as a tool but should not vendor the code into Apache-2.0
  sources.

Adversarial Camera Stickers, Li, Schmidt, Kolter, ICML 2019 (arXiv 1904.00759, PMLR 97:3896-3904)
- Translucent sticker on the camera lens; ImageNet ResNet-50; "street sign" to "guitar pick" is a
  headline example and a real stop sign was classified as guitar pick; targeted fooling 49.6
  percent (abstract), 52 percent on real video; street-sign accuracy from 64 to 18 percent.
  Physical. Code: none found (the paper links only a demo video)
  (https://ar5iv.labs.arxiv.org/html/1904.00759).

Rolling-shutter and light attacks
- Sayles et al., "Invisible Perturbations: Physical Adversarial Examples Exploiting the Rolling
  Shutter Effect", CVPR 2021 (arXiv 2011.13375): modulated LED illumination creates striping
  that fools an ImageNet classifier, targeted rates up to 84 percent; physical; code: none found.
- Köhler et al., "They See Me Rollin'", 2021 (arXiv 2101.10011): laser-modulated rolling-shutter
  injection evaluated on BDD100K and VIRAT with Faster R-CNN and SSD; code:
  https://github.com/ssloxford/they-see-me-rollin, GPL-3.0, 8 stars
  (https://raw.githubusercontent.com/ssloxford/they-see-me-rollin/master/README.md).
- Yan et al., "Rolling Colors: Adversarial Laser Exploits against Traffic Light Recognition",
  USENIX Security 2022 (arXiv 2204.02675): laser against traffic LIGHT recognition, not signs
  (https://www.usenix.org/conference/usenixsecurity22/presentation/yan). Listed because the task
  mentioned it; not a TSR attack.
- Duan et al., "Adversarial Laser Beam" (AdvLB), CVPR 2021 (arXiv 2103.06504): laser beam
  parameters optimised against ImageNet classifiers; night tests; adversarially trained ResNet50
  drops attack success from 95.10 to 77.20 percent; code https://github.com/RjDuan/Advlight, MIT,
  34 stars (https://raw.githubusercontent.com/RjDuan/Advlight/master/README.md). Not TSR specific.
- Gnanasambandam, Sherman, Chan, "Optical Adversarial Attack" (OPAD), ICCV Workshop 2021 (arXiv
  2108.06247): projector-camera model inside the attack optimisation, white-box and black-box
  attacks on real 3D objects; code: none found.
- Infrared attacks (Zhou et al. "Invisible Mask", face recognition) were not verified in this
  session and are not TSR work.

Shadows, Zhong, Liu, Tang, Xu, Li, "Shadows can be Dangerous", CVPR 2022 (arXiv 2203.03818)
- Black-box (particle swarm) shadow-shaped perturbations on GTSRB and LISA crops, digital and
  physical; repo ships `model_gtsrb.pth`, `model_lisa.pth` and adversarially trained
  `adv_model_gtsrb.pth`, `adv_model_lisa.pth`, masks for circle, inverse triangle, octagon,
  pentagon, rectangle, rhombus, triangle sign shapes, PyTorch 1.9.
- Code: https://github.com/hncszyq/ShadowAttack, MIT, 64 stars, last commit September 2025
  (https://raw.githubusercontent.com/hncszyq/ShadowAttack/master/README.md). The inverse-triangle
  mask makes it directly reusable for Japanese 一時停止 and 徐行.

TPatch, Zhu, Ji, Yan et al. (Zhejiang USSLab), USENIX Security 2023
- Patch that stays benign until an acoustic signal-injection attack on the camera triggers it;
  hiding, creating and altering attacks; evaluated with YOLOv3/v5 and Faster R-CNN and eight
  ImageNet classifiers on ImageNet, COCO, KITTI and BDD100K (not a sign dataset); physical demos.
- Code: https://github.com/USSLab/TPatch, license not specified, 24 stars, last commit August
  2023 (https://raw.githubusercontent.com/USSLab/TPatch/master/README.md). Venue taken from the
  repository's paper PDF reference; the USENIX page was not fetched.

Phantom of the ADAS, Nassi, Mirsky, Nassi, Ben Netanel, Elovici, CCS 2020 (see A10): physical
projection and billboard phantoms against Tesla Model X and Mobileye 630; code: none found
(https://nassiben.com/phantoms).

DeepBillboard, Zhou et al., 2018 (arXiv 1812.10812; ICSE 2020 publication not verified)
- Adversarial billboards that change steering of end-to-end driving models (Dave etc.), digital
  and physical; not a sign classifier attack. Code: https://github.com/deepbillboard/DeepBillboard,
  license not specified, 11 stars (https://raw.githubusercontent.com/deepbillboard/DeepBillboard/master/README.md).

Translucent Patch, Zolfi, Kravchik, Elovici, Shabtai, CVPR 2021 (arXiv 2012.12528)
- Lens patch that hides a target class across all instances; prevented detection of 42.27 percent
  of stop sign instances while keeping about 80 percent detection of other classes; physical.
  Code: none found.

Invisible for both Camera and LiDAR (MSF-ADV), Cao et al., IEEE S&P 2021 (arXiv 2106.09249)
- 3D-printed objects against camera-LiDAR fusion in Apollo; not signs. Code:
  https://github.com/ASGuard-UCI/MSF-ADV, MIT, 84 stars. Related lane-centering attack "Dirty
  Road Can Attack" (Sato et al., USENIX Security 2021, arXiv 2009.06701), code
  https://github.com/ASGuard-UCI/DRP-attack, MIT, 38 stars. Both listed for completeness of the
  AV-perception threat model, not as TSR attacks.

GhostImage, Man, Li, Gerdes, USENIX RAID 2020 (arXiv 2001.07792)
- Projector exploits lens flare/ghosting and auto-exposure to inject patterns remotely; LISA
  classifier among the three datasets; success up to 100 percent depending on distance; physical.
- Code: https://github.com/Harry1993/GhostImage, MIT, 3 stars; TF 2.1, pre-trained LISA classifier
  download (https://raw.githubusercontent.com/Harry1993/GhostImage/master/README.md).

FTE, Jia et al., "Fooling the Eyes of Autonomous Vehicles", NDSS 2022 (arXiv 2201.06192)
- Robust physical AEs against a YOLOv5 TSR pipeline with blur and resolution transformations,
  hiding / appearance / non-target / target attacks, sunny, cloudy and night. Code: none found;
  Wang et al. report 0 percent success of FTE on all four commercial vehicles despite the
  paper's claimed commercial success (https://arxiv.org/pdf/2409.09860).

Revisiting Physical-World Adversarial Attack on TSR: A Commercial Systems Perspective, Wang,
Xie, Sato, Luo, Xu, Chen (UC Irvine, Drexel), NDSS 2025 (arXiv 2409.09860)
- Setup: RP2, SIB and FTE re-implemented; surrogates YOLOv5 (mAP 0.831) and Faster R-CNN
  ResNet-50-FPN-V2 (0.871); COCO for STOP, ARTS for a 25 mph sign; outdoor rooftop,
  25,000-30,000 lux, 5 mph, three repeats.
- Results: 2 of 30 attack-vehicle combinations succeeded (3/3) on C2's STOP sign (RP2 with Faster
  R-CNN surrogate, SIB with YOLOv5); overall 6.67 percent vs "51.6% on average" in the original
  papers (p < 0.02). "Spatial memorization": C2-C4 keep a detected sign for up to 60 s and clear
  it after passing the sign's position; a detection of about 1 s is enough to be memorised. With
  system-level metrics, white-box per-frame hiding success of 56.8 (RP2) and 47.5 (FTE) percent
  falls to 6.6 and 5.2 percent (SIB 87.1 to 45.1); black-box average 47 to about 12.6 percent;
  appearing attacks are the opposite (SysAA 98.2 to 100 percent).
- Code and data: "will be made available at our website"
  (https://sites.google.com/view/av-ioat-sec/commercial-tsr-test, not fetched). Paper:
  https://arxiv.org/pdf/2409.09860.

2023 to 2026 work
- DiffSign, Ma, Wang, Zheng, Guo, Wang, Chen, Shen (Xi'an Jiaotong, UC Irvine, Nankai, Wuhan),
  arXiv 2511.12956, 17 November 2025: text-to-image appearance attack built on Stable Diffusion
  2.1 with a CLIP loss and masked prompts; targets YOLOv3/v4/v5/v11, Faster R-CNN, Mask R-CNN,
  DETR, SSD; COCO (STOP) and TT100K (speed limits 5/40/60 km/h, No Entry, Children Crossing);
  digital AASR 82.7 percent averaged over eight detectors; physical 83.3 percent average (STOP:
  sunny 85.4, cloudy 74.7, night 47.8; 5 km/h: 97.7 / 98.8 / 96.6); two unnamed commercial TSR
  systems attacked at 97 percent; PASS/CARLA end-to-end 100 percent vs 0 for baselines; code
  "will publicly release", no URL yet (https://arxiv.org/html/2511.12956v1).
- FIGhost, Yuan, Xu, Li, Zhang et al., arXiv 2505.12045 (May 2025): physical backdoor with
  fluorescent-ink triggers activated by UV light, against detectors and VLMs; code: unknown
  (https://export.arxiv.org/api/query?id_list=2505.12045).
- Typographic attacks on Vision-LLMs for driving, Chung et al., arXiv 2405.14169 (May 2024):
  auto-generated misleading text in traffic scenes against LLaVA, Qwen-VL, VILA, Imp, with
  physical realisability study; code "upon acceptance" (https://arxiv.org/abs/2405.14169).
- VLM patch attacks in CARLA: Fernandez, MohajerAnsari, Salarpour et al., "Comparative Analysis of
  Patch Attack on VLM-Based Autonomous Driving Architectures", IEEE IV 2025 (arXiv 2603.08897)
  and "Understanding Adversarial Transferability in VLMs for Autonomous Driving", SAE WCX 2026
  (arXiv 2604.27414): patches on roadside infrastructure transfer across Dolphins, OmniDrive and
  LeapVAD at 73-91 percent (https://export.arxiv.org/api/query?id_list=2603.08897,2604.27414).

### C2. Defences, certification and robust training relevant to TSR

- Adversarial training baselines that ship weights for GTSRB and LISA: SLAP's `usenix21adv`
  models (GPL-3.0 repo) and ShadowAttack's `adv_model_gtsrb.pth` / `adv_model_lisa.pth` (MIT), see
  C1. LAMDA, MohajerAnsari, Salarpour, Pesé, IROS 2026 (arXiv 2608.08815): language-anchored
  prototype losses from a frozen OpenCLIP text encoder, evaluated on GTSRB and LISA with four
  backbones and three physical attack types (shadow, natural light, printed patch); "the only
  method among ten evaluated" that improves robustness in all combinations, up to +12.5 pp; code
  unknown (https://export.arxiv.org/api/query?id_list=2608.08815). V2LMs (same group, IEEE IV
  2026, arXiv 2506.11472): fine-tuned VLMs for TSR, lane centering and vehicle detection lose
  under 8 percent under attack where DNNs drop 33-74 percent; code unknown.
- SentiNet, Chou, Tramèr, Pellegrino, DLS workshop 2020 (arXiv 1812.00292): detects localized
  universal attacks (patches, trojans, poisoning) via saliency and overlay testing, no training
  needed; evaluated against physically realisable patches; official code: none found.
- Feature Squeezing, Xu, Evans, Qi, NDSS 2018 (arXiv 1704.01155): bit-depth reduction and
  smoothing as detectors; code https://github.com/mzweilin/EvadeML-Zoo, MIT, 187 stars. The
  abstract names no GTSRB evaluation; any "Feature Squeezing on GTSRB" result is third-party (not
  verified).
- Certified patch defences (none evaluated on GTSRB in their abstracts): Chiang et al., ICLR 2020
  (arXiv 2003.06693), code https://github.com/Ping-C/certifiedpatchdefense, BSD-2-Clause;
  Levine and Feizi, (De)Randomized Smoothing, NeurIPS 2020 (arXiv 2002.10733), CIFAR-10 and
  ImageNet, code https://github.com/alevine0/patchSmoothing, license not identifiable; PatchGuard,
  Xiang et al., USENIX Security 2021 (arXiv 2005.10884), ImageNet/ImageNette/CIFAR-10, code
  https://github.com/inspire-group/PatchGuard, MIT, 74 stars. A GTSRB port of PatchGuard or
  (De)Randomized Smoothing would be new work.
- REAP benchmark, Hingun, Sitawarin, Li, Wagner, ICCV 2023 (arXiv 2212.05680): over 14,000
  Mapillary Vistas traffic signs, each with geometric and lighting transforms so a digital patch
  can be applied realistically; finding: "adversarial patch attacks may present a smaller threat
  than previously believed" and simple digital success "is not predictive" of practical
  effectiveness. Code https://github.com/wagner-group/reap-benchmark (license not identifiable,
  34 stars, last commit October 2023); data and weights on Kaggle `csitawarin/reap-benchmark`
  (REAP `100`, REAP-Shape `no_color`, synthetic sets); depends on detectron2, kornia, detrex,
  YOLOF, YOLOv7 (https://raw.githubusercontent.com/wagner-group/reap-benchmark/master/README.md).
  The Mapillary Vistas license applies to the images (see `datasets.md`).

### C3. Backdoor and poisoning work that uses traffic signs

- BadNets, Gu, Dolan-Gavitt, Garg, 2017 (arXiv 1708.06733): backdoored US street sign classifier
  that labels stop signs as speed limits when a sticker is present; backdoor persists through
  transfer learning with a 25 percent average accuracy drop under trigger. Original code:
  https://github.com/Kooscii/BadNets (Caffe, py-faster-rcnn, "US Traffic Signs (usts)" dataset
  from the LISA/VIVA challenge, yellow square / bomb / flower triggers), license not specified,
  107 stars, last commit May 2018 (https://raw.githubusercontent.com/Kooscii/BadNets/master/README.md).
  Community PyTorch re-implementation https://github.com/verazuo/badnets-pytorch, MIT, 197 stars
  (datasets not verified).
- TrojanNN, Liu et al., NDSS 2018: trojan triggers generated from internal neurons; models for
  face recognition, speech, age, sentence attitude and autonomous driving; code
  https://github.com/PurduePAML/TrojanNN (Caffe, Theano), license not specified, 192 stars
  (https://raw.githubusercontent.com/PurduePAML/TrojanNN/master/README.md). No GTSRB model listed
  in the README.
- Neural Cleanse, Wang et al., IEEE S&P 2019: trigger reverse engineering and mitigation; the repo
  ships an infected GTSRB model (`models/gtsrb_bottom_right_white_4_target_33.h5`) and the test
  data (`data/gtsrb_dataset_int.h5`), plus injection code for GTSRB; Keras 2.2.2 / TF 1.10; code
  https://github.com/bolunwang/backdoor, MIT, 326 stars
  (https://raw.githubusercontent.com/bolunwang/backdoor/master/README.md).
- Benchmarks: BackdoorBench (SCLBD) supports CIFAR-10, CIFAR-100, GTSRB and Tiny ImageNet; CC
  BY-NC 4.0 (non-commercial, so evaluation use only), 619 stars, last commit July 2025
  (https://raw.githubusercontent.com/SCLBD/BackdoorBench/master/README.md). TrojanZoo
  https://github.com/ain-soph/trojanzoo, MIT, 304 stars (GTSRB support not verified).

### C4. Simulator and system-level evaluation tools

- SafeBench, Xu et al., NeurIPS 2022 Datasets and Benchmarks (arXiv 2206.09682): CARLA 0.9.13
  based platform with perception and control evaluation, Scenic scenario generation, Docker
  image; MIT, 164 stars, last commit September 2023
  (https://raw.githubusercontent.com/trust-ai/SafeBench/master/README.md). Whether its perception
  track includes adversarial sign textures was not verified.
- PASS / CARLA end-to-end evaluation is used by DiffSign (ASR-S metric)
  (https://arxiv.org/html/2511.12956v1); the PASS simulator itself was not fetched.
- The Wang et al. system-level metrics SysHA (product of per-segment hiding success before the
  sign is passed) and SysAA (one minus the product of per-segment failures) are the right unit of
  evaluation for a system with temporal persistence (https://arxiv.org/pdf/2409.09860).
- CARLA's built-in speed limit and stop sign assets and their class coverage were not verified.

---

## D. Outcomes

### D.1 Target specification draft for OpenTSR (one page)

Derived from sections A to C. Items marked [reg] are copied from a regulation or protocol;
items marked [proposed] are OpenTSR choices without an external source.

Scope of sign categories
- Japan [proposed, catalogue from https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003]:
  all 59 規制標識 kinds (74 variants) including 323 / 323の2 / 324 speed signs with their numeric
  value as an attribute, 325 自動車専用 as an implicit motorway-limit trigger, 330 一時停止 and 329 徐行
  in both Japanese-only and bilingual forms; all 26 警戒標識 kinds; all 14 指示標識 kinds; the 23
  補助標識 kinds as attributes attached to a parent sign (501 distance/area, 502 day/time, 503
  vehicle type, 505 start, 506 within, 507 end); 案内標識 limited to the 高速道路 entry/exit family
  (103 入口の方向, 104 入口の予告, 109 出口の予告, 110 方面及び出口の予告, 112 方面及び出口, 113 出口) and the
  route-number shields (118 国道番号, 118の2 都道府県道番号, 118の3 高速道路番号) [proposed]. Variable (light-emitting)
  speed signs with white or yellow digits on black must be in the speed class
  (https://laws.e-gov.go.jp/api/1/lawdata/335M50004002003). Implicit national limits: 60 km/h
  general, 30 km/h on roads without a centre line from 2026-09-01, 100 km/h motorway, so the
  perception stack must also expose "centre line present" (https://web.pref.hyogo.lg.jp/kf13/press/20260528.html).
- EU [reg, Annex II structure of https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng/pdf]:
  explicit numerical limits 5 to 130 km/h, implicit numerical (struck-through end-of-limit),
  implicit non-numerical (city entry/exit, motorway start/end, expressway start/end, zone
  start/end, traffic-calmed area start/end, end of all restrictions), variable message signs
  showing any of these, plus the Euro NCAP conditional families (rain/wet, snow/ice, time/season,
  distance for/in, lane arrows, vehicle and weight categories incl. trailer) and dynamic/roadworks
  limits (https://cdn.euroncap.com/media/77301/euro-ncap-assessment-protocol-sa-safe-driving-v102.pdf).
  Minimum class set: the 43 GTSRB classes (https://huggingface.co/api/datasets/tanganke/gtsrb?full=true)
  plus the implicit and supplementary families above [proposed].
- US [reg, https://www.iihs.org/media/4e417587-ab67-489c-a3bb-7fe1dd87fe1b/COWTLw/Ratings/Protocols/future%20programs/Intelligent_speed_assistance_protocol.pdf]:
  numerical mph speed limit (R2-1) with unit handling; stop, yield, do not enter, wrong way,
  school zone and end-of-limit as the next tier [proposed].
- Output contract [proposed]: per sign a class, numeric value with unit, supplementary
  attributes, lane relevance flag, and a confidence; per frame a track id so that persistence
  logic can be applied.

Performance thresholds
- Legal floor [reg]: TP_D at least 90 percent of distance overall and 80 percent per road type
  on a 400 km mixed route with at least 15 percent darkness; new limit available within 2.0 s of
  passing the sign (10 m below 20 km/h); country of operation known
  (https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng/pdf).
- Consumer-test level [reg]: Euro NCAP distance KPI above 80 percent and the event KPI above its
  printed threshold on 2000 km over 3 to 5 countries with 10 percent night
  (https://www.euroncap.com/media/91705/euro-ncap-protocol-safe-driving-vehicle-assistance-v11.pdf).
- OpenTSR release gate [proposed]: event-based accuracy (every sign passing event counted once,
  as Thatcham and Euro NCAP do) of at least 95 percent on explicit signs and 90 percent on
  implicit and conditional signs per country, with zero implausible values (no 5/10/15/100 mph
  in the UK style failure, https://www.autocar.co.uk/node/375029); per-frame metrics (mAP,
  top-1) reported but not used as the gate; latency budget compatible with the 2.0 s rule at
  motorway speed.

Robustness evaluation to ship with every release [proposed]
- CURE-TSR style severity curves: accuracy per challenge type and level on the 12 types x 5
  levels (https://zenodo.org/record/3903066) and the seven GTSRB corruption splits
  (https://huggingface.co/api/datasets/tanganke/gtsrb?full=true), reported as a table per release.
- Night and rain slices: at least 15 percent night in the evaluation drive [reg]; ZOD-style
  per-condition breakdown (clear, cloudy, rain, fog, snow; day, twilight, night)
  (https://arxiv.org/pdf/2305.02008).
- Cross-country test: train on EU, test on Japan and US, and the reverse, following TS-1M's
  cross-region protocol (https://arxiv.org/pdf/2603.23034).

Security evaluation to ship with every release [proposed, from section C]
- Patch attacks under realistic transforms with REAP (Mapillary signs; compare with the paper's
  finding that digital success overstates risk) (https://arxiv.org/abs/2212.05680); white-box RP2
  and ShapeShifter reproductions on the released detector (MIT and BSD-3 code).
- Natural-phenomenon attacks: ShadowAttack with the inverse-triangle mask for Japanese stop and
  slow signs (MIT) (https://raw.githubusercontent.com/hncszyq/ShadowAttack/master/README.md).
- System-level metrics: report SysHA and SysAA over the approach segment, and document the
  temporal persistence (memorisation) policy, because it decides hiding vs appearing exposure
  (https://arxiv.org/pdf/2409.09860).
- Appearing attacks and phantoms: test with projected or screen-displayed signs of 100 to 500 ms
  duration and require a minimum persistence before a limit is adopted (https://nassiben.com/phantoms).
- Typographic and VLM paths: if a VLM reads sign text, include a typographic attack set (text
  stickers with wrong numbers or "STOP") (https://arxiv.org/abs/2405.14169).
- Supply-chain hygiene: scan released weights with Neural Cleanse (MIT, ships a GTSRB example)
  and run BackdoorBench's GTSRB suite (evaluation only, CC BY-NC 4.0); publish training data
  hashes so poisoning can be audited (https://raw.githubusercontent.com/bolunwang/backdoor/master/README.md;
  https://raw.githubusercontent.com/SCLBD/BackdoorBench/master/README.md).
- Model card: threat model, the attacks run, their success rates, and known unprotected cases
  (digital and temporary signs, as in Tesla's manual) stated explicitly.

### D.2 Summary table of attack and defence code bases

| Name | Year | Venue | Target | Physical / digital | Code URL | License (shields.io) | Stars | Last commit |
|---|---|---|---|---|---|---|---|---|
| RP2 | 2018 | CVPR | LISA-CNN, GTSRB-CNN, Inception V3 | physical | https://github.com/evtimovi/robust_physical_perturbations | MIT | 112 | Dec 2019 |
| ShapeShifter | 2018 | ECML-PKDD | Faster R-CNN stop sign | physical | https://github.com/shangtse/robust-physical-attack | BSD-3-Clause | 170 | Jan 2020 |
| Rogue Signs / DARTS | 2018 | DLS workshop / arXiv | Keras CNNs on GTSRB | physical | https://github.com/inspire-group/advml-traffic-sign | MIT | 37 | Apr 2018 |
| SIB | 2019 | ACM CCS | YOLOv3, Faster R-CNN | physical | none found | n/a | n/a | n/a |
| GhostImage | 2020 | RAID | LISA classifier, CIFAR, ImageNet | physical (projector) | https://github.com/Harry1993/GhostImage | MIT | 3 | Jun 2020 |
| Phantom of the ADAS | 2020 | CCS | Tesla Model X, Mobileye 630 | physical | none found | n/a | n/a | n/a |
| SLAP | 2021 | USENIX Security | YOLOv3, Mask R-CNN, LISA-CNN, GTSRB-CNN | physical (projector) | https://github.com/ssloxford/short-lived-adversarial-perturbations | GPL-3.0 | 28 | Oct 2021 |
| AdvLB | 2021 | CVPR | ImageNet classifiers | physical (laser) | https://github.com/RjDuan/Advlight | MIT | 34 | Aug 2021 |
| They See Me Rollin' | 2021 | arXiv | Faster R-CNN, SSD on BDD100K | physical (laser) | https://github.com/ssloxford/they-see-me-rollin | GPL-3.0 | 8 | Oct 2021 |
| Invisible Perturbations | 2021 | CVPR | ImageNet classifier | physical (LED) | none found | n/a | n/a | n/a |
| OPAD | 2021 | ICCVW | classifiers on 3D objects | physical (projector) | none found | n/a | n/a | n/a |
| Translucent Patch | 2021 | CVPR | object detectors (stop sign) | physical (lens) | none found | n/a | n/a | n/a |
| Camera Stickers | 2019 | ICML | ResNet-50 ImageNet | physical (lens) | none found | n/a | n/a | n/a |
| MSF-ADV | 2021 | IEEE S&P | camera-LiDAR fusion (not signs) | physical | https://github.com/ASGuard-UCI/MSF-ADV | MIT | 84 | Aug 2021 |
| DRP (Dirty Road) | 2021 | USENIX Security | lane centering (not signs) | physical | https://github.com/ASGuard-UCI/DRP-attack | MIT | 38 | Aug 2021 |
| FTE | 2022 | NDSS | YOLOv5 TSR | physical | none found | n/a | n/a | n/a |
| ShadowAttack | 2022 | CVPR | GTSRB, LISA CNNs | physical | https://github.com/hncszyq/ShadowAttack | MIT | 64 | Sep 2025 |
| TPatch | 2023 | USENIX Security | YOLOv3/v5, Faster R-CNN, 8 classifiers | physical (acoustic trigger) | https://github.com/USSLab/TPatch | not specified | 24 | Aug 2023 |
| REAP benchmark | 2023 | ICCV | detectors on Mapillary signs | digital, realistic | https://github.com/wagner-group/reap-benchmark | not identifiable | 34 | Oct 2023 |
| Wang et al. commercial TSR | 2025 | NDSS | 4 production vehicles | physical | https://sites.google.com/view/av-ioat-sec/commercial-tsr-test (not fetched) | unknown | n/a | n/a |
| DiffSign | 2025 | arXiv | 8 detectors, 2 commercial TSR | physical | to be released | unknown | n/a | n/a |
| DeepBillboard | 2018 | arXiv (ICSE 2020 not verified) | steering models (not signs) | physical | https://github.com/deepbillboard/DeepBillboard | not specified | 11 | Mar 2019 |
| Feature Squeezing | 2018 | NDSS | detection defence | digital | https://github.com/mzweilin/EvadeML-Zoo | MIT | 187 | Dec 2018 |
| SentiNet | 2020 | DLS workshop | patch/trojan detection defence | digital on physical patches | none found | n/a | n/a | n/a |
| Certified patch defence (Chiang) | 2020 | ICLR | MNIST, CIFAR-10 | digital | https://github.com/Ping-C/certifiedpatchdefense | BSD-2-Clause | 34 | Sep 2020 |
| (De)Randomized Smoothing | 2020 | NeurIPS | CIFAR-10, ImageNet | digital | https://github.com/alevine0/patchSmoothing | not identifiable | 17 | Aug 2022 |
| PatchGuard | 2021 | USENIX Security | ImageNet, CIFAR-10 | digital | https://github.com/inspire-group/PatchGuard | MIT | 74 | Jul 2022 |
| BadNets (original) | 2017 | arXiv | Faster R-CNN on US traffic signs | digital (sticker trigger) | https://github.com/Kooscii/BadNets | not specified | 107 | May 2018 |
| BadNets (PyTorch) | 2022 | community | classifiers | digital | https://github.com/verazuo/badnets-pytorch | MIT | 197 | Sep 2022 |
| TrojanNN | 2018 | NDSS | face, speech, driving models | digital | https://github.com/PurduePAML/TrojanNN | not specified | 192 | Mar 2022 |
| Neural Cleanse | 2019 | IEEE S&P | backdoor detection incl. GTSRB model | digital | https://github.com/bolunwang/backdoor | MIT | 326 | Feb 2020 |
| BackdoorBench | 2022+ | NeurIPS D&B (not verified) | CIFAR, GTSRB, Tiny ImageNet | digital | https://github.com/SCLBD/BackdoorBench | CC BY-NC 4.0 (README) | 619 | Jul 2025 |
| TrojanZoo | 2020+ | EuroS&P (not verified) | backdoor toolbox | digital | https://github.com/ain-soph/trojanzoo | MIT | 304 | Aug 2025 |
| SafeBench | 2022 | NeurIPS D&B | CARLA driving agents | simulation | https://github.com/trust-ai/SafeBench | MIT | 164 | Sep 2023 |

Stars and last-commit values: https://img.shields.io/github/stars/OWNER/REPO.json and
.../last-commit/... fetched 2026-10-31; licenses: https://img.shields.io/github/license/OWNER/REPO.json
except BackdoorBench (README text).

### D.3 Pages that could not be fetched

- https://eur-lex.europa.eu/eli/reg_del/2021/1958/oj/eng and
  https://eur-lex.europa.eu/eli/reg_del/2021/1958/2023-09-21/eng (HTML pages returned metadata
  only); curl to https://eur-lex.europa.eu/legal-content/EN/TXT/PDF/?uri=CELEX:32021R1958 and the
  HTML variants returned HTTP 202 with an empty body (bot gate). The PDFs were read through the
  fetch tool but only up to about 94,000 characters, so Annex II beyond Belgium (Germany and the
  other 25 Member States) and the content of amending Regulation (EU) 2023/1675 were not read.
- https://lexaris.de/book/version/documentflat/head/2056920 (same truncation at Belgium).
- https://www.euroncap.com/media/79883/euro-ncap-assessment-protocol-sa-safe-driving-v103.pdf
  (client error; v10.2 used instead).
- https://www.whatcar.com/news/intelligent-speed-assist-systems-fail-to-read-road-signs-25-of-the-time-research-finds/n29166
  (client error; Autocar version used).
- https://cncap.obs.cn-north-4.myhuaweicloud.com/cms/files/909314048149975040.pdf (C-NCAP 2024
  Appendix L; permission not granted). C-NCAP ISLS details rest on the LeddarTech white paper.
- i-VISTA TSR test specification: no primary page found.
- https://ja.wikipedia.org/wiki/日本の道路標識 (permission not granted).
- https://commons.wikipedia.com/wiki/File:Japan_road_sign_319.svg (404) and the Wikimedia
  Commons API (HTTP 429 rate limit); Commons file data taken from the osmand.net mirror.
- https://docs.nvidia.com/drive/driveworks-3.5/signnet_JP_v1_0.html and
  https://docs.nvidia.com/drive/archive/driveworks-3.0/signnet_mainsection.html (redirect to a
  generic documentation page; NVIDIA's Japanese SignNet class list could not be read).
- https://github.com/olivesgatech/CURE-TSR (permission not granted; Zenodo and IEEE DataPort used).
- https://www.continental-automotive.com/en/components/cameras/multi-function-mono-camera-mfc526.html
  (redirects to https://www.aumovio.com/..., permission not granted); no ZF or Denso TSR page
  was fetched.
- https://sites.google.com/view/av-ioat-sec/commercial-tsr-test (permission not granted; code
  availability of Wang et al. NDSS 2025 unknown).
- https://laws.e-gov.go.jp/api/1/lawdata/335M50000002003 returned "0 results" (wrong law ID);
  the correct ID 335M50004002003 was found through the law list API and used.
- USENIX pages for TPatch and SLAP were not fetched; venues come from the repositories and arXiv.
- Not attempted for lack of time: Euro NCAP per-model Speed Assistance datasheets, NHTSA recall
  database searches for sign misreading, UN R157/R89 texts, the IEEE T-ITS CURE-TSD paper, the
  torchvision GTSRB split documentation, PASS simulator and CARLA sign asset lists.
