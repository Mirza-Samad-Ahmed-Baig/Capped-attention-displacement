# references.bib: verification log

- **Date of retrieval:** 2026-09-27
- **Result:** 48 entries, all verified. None are UNVERIFIED.
- **Test build:** BibTeX with `elsarticle-harv.bst` (tectonic, scratch copy, `\nocite{*}`). It ran without errors. It gave 11 "empty pages" warnings, and all 11 are for sources that do not publish page numbers (see the pages note below).

## Method

No entry was written from memory. Each field comes from a record fetched on the date above. The sources were:

- **DOI content negotiation:** `curl -H "Accept: application/x-bibtex"` and `application/vnd.citationstyles.csl+json` against `https://doi.org/<DOI>`.
- **arXiv:** the arXiv API, `https://export.arxiv.org/api/query?id_list=...`.
- **ACL Anthology:** the `.bib` files at `https://aclanthology.org/<ID>.bib`.
- **NeurIPS:** the proceedings abstract pages and their BibTeX files at proceedings.neurips.cc.
- **ICLR 2024/2025:** the proceedings abstract pages and their BibTeX at proceedings.iclr.cc.
- **ICLR 2019-2022:** the official schedule and virtual-poster pages at iclr.cc.
- **PMLR:** the proceedings pages.
- **CVF:** the open-access page.
- **JMLR:** the paper page and its `.bib` file.
- **COLM 2024:** the accepted-papers page.
- **ICML 2002:** the official conference site at icml.cc/Conferences/2002.

**Access limits during retrieval.** openreview.net (both the API and the forum pages), dblp.org and its mirrors, and dl.acm.org all returned bot-challenge pages (HTTP 403 or Anubis). No data could be read from them.
- For ICLR 2019-2022, OpenReview forum IDs were therefore taken from the links printed on iclr.cc's own schedule and poster pages. Titles and authors were checked against both iclr.cc and arXiv.
- Web search was not available.

**Field conventions:**
- `doi` is given whenever the publisher registered one. `url` is used only when there is no DOI; this avoids links that duplicate each other in the output.
- arXiv-only works are `@misc` entries with `eprint`, `archivePrefix` and `primaryClass`.
- Non-ASCII characters are LaTeX-escaped.
- Acronyms and proper nouns in titles are braced, because `elsarticle-harv` sentence-cases titles.

**Style note (not a data problem).** In `elsarticle-harv.bst` the `inproceedings` function has `format.bvolume` commented out. As a result, the `volume` field of `@inproceedings` entries (NeurIPS 26-38, PMLR 139/151, ICLR 2024/2025) is kept in the .bib but is not printed.

**Pages note.** The following sources publish no page numbers, so `pages` was left empty:
- NeurIPS 2013, 2017 and 2019;
- ICLR 2019-2022 (OpenReview-only proceedings);
- COLM 2024.

`lin2025twilight` has no pages for a different reason, explained in its row.

## Per-key log

| Key | Source used | Verified | Discrepancy / note |
|---|---|---|---|
| vaswani2017attention | https://proceedings.neurips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html (+ `-Bibtex.bib`); arXiv 1706.03762 | title, 8 authors, NeurIPS 2017 = NIPS vol. 30, year | The proceedings title is "Attention is All you Need"; the arXiv and PDF title is "Attention Is All You Need". The proceedings string is used, and both print identically after sentence-casing. The proceedings give no pages. |
| srivastava2014dropout | https://jmlr.org/papers/v15/srivastava14a.bib and .html (citation meta) | title, 5 authors, JMLR 15(56):1929-1958, 2014 | none |
| hinton2015distilling | arXiv API id 1503.02531 | title, 3 authors, 2015, stat.ML | The arXiv comment says "NIPS 2014 Deep Learning Workshop". It is kept as an arXiv preprint, as requested. |
| lin2019dropattention | arXiv API id 1907.11065 | title, 6 authors, 2019, cs.CL | arXiv writes the first author family-name first, as "Lin Zehui". The entry uses `Lin, Zehui`. |
| li2023dropkey | DOI 10.1109/CVPR52729.2023.02174 (Crossref); https://openaccess.thecvf.com/content/CVPR2023/html/Li_DropKey_for_Vision_Transformer_CVPR_2023_paper.html | title, 7 authors, CVPR 2023, pp. 22700-22709 | The arXiv title (2208.02646) is only "DropKey". The CVPR/CVF title "DropKey for Vision Transformer" is used. |
| zhou2020drophead | https://aclanthology.org/2020.findings-emnlp.178.bib | title, 5 authors, Findings of EMNLP 2020, pp. 1971-1980, DOI | none |
| lohrenz2021relaxed | DOI 10.1109/ASRU51503.2021.9688298 (Crossref); arXiv 2107.01275 | title, 4 authors, ASRU 2021, pp. 177-184 | none. The arXiv record says "Accepted at ASRU 2021". |
| lohrenz2023relaxed | DOI 10.1109/IJCNN54540.2023.10191643 (Crossref); arXiv 2209.09735 | title, 4 authors (Lohrenz, Moeller, Li, Fingscheidt), IJCNN 2023, pp. 1-10 | none |
| liang2021rdrop | https://proceedings.neurips.cc/paper_files/paper/2021/hash/5a66b9200f29ac3fa0ae244cc2a51b39-Abstract.html (+ BibTeX); arXiv 2106.14448 | title, 9 authors, NeurIPS 34, pp. 10890-10905 | The NeurIPS BibTeX lower-cases the first author ("liang, xiaobo"). Corrected to `Liang, Xiaobo` per arXiv. |
| lee2020mixout | https://iclr.cc/virtual_2020/poster_HkgaETNtDB.html (links openreview.net/forum?id=HkgaETNtDB); arXiv 1909.11299 | title, 3 authors, ICLR 2020 | OpenReview was blocked, so the forum ID was taken from the iclr.cc page. arXiv says "Published as a conference paper at ICLR 2020". |
| mosbach2021stability | https://iclr.cc/Conferences/2021/Schedule?type=Poster (card links forum?id=nzpLWnVAyah); arXiv 2006.04884 | title, 3 authors, ICLR 2021 | none |
| zhang2021revisiting | https://iclr.cc/Conferences/2021/Schedule?type=Poster (card links forum?id=cO1IH43yUF); arXiv 2006.05987 | title, 5 authors, ICLR 2021 | iclr.cc shows the name as "Kilian Weinberger". The entry uses "Kilian Q. Weinberger" from arXiv. |
| dodge2020finetuning | arXiv API id 2002.06305 | title, 6 authors, 2020, cs.CL | none. There is no peer-reviewed version on the arXiv record. |
| jin2019pubmedqa | https://aclanthology.org/D19-1259.bib | title, 5 authors, EMNLP-IJCNLP 2019, pp. 2567-2577, DOI | none |
| gu2021domain | DOI 10.1145/3458754 (Crossref BibTeX + CSL) | title, 9 authors, ACM Trans. Comput. Healthcare 3(1):1-23, 2021 | none |
| yasunaga2022linkbert | https://aclanthology.org/2022.acl-long.551.bib | title, 3 authors, ACL 2022 (Long), pp. 8003-8016, DOI | none |
| tinn2023finetuning | DOI 10.1016/j.patter.2023.100729 (Crossref); arXiv 2112.07869 | title, 8 authors, Patterns 4(4):100729, 2023 | The arXiv title is Title-Cased; the journal title (sentence case) is used. |
| wadden2020scifact | https://aclanthology.org/2020.emnlp-main.609.bib | title, 7 authors, EMNLP 2020, pp. 7534-7550, DOI | none |
| rohanian2023compact | DOI 10.1093/bioinformatics/btad103 (Crossref BibTeX + CSL) | title, 4 authors, Bioinformatics 39(3), 2023 | Crossref gives no page range, only the article number btad103, which is used as `pages`. |
| clark2019bert | https://aclanthology.org/W19-4828.bib | title, 4 authors, BlackboxNLP 2019 (ACL workshop), pp. 276-286, DOI | The official title has a lower-case "at": "What Does BERT Look at? An Analysis of BERT's Attention". |
| kobayashi2020attention | https://aclanthology.org/2020.emnlp-main.574.bib | title, 4 authors, EMNLP 2020, pp. 7057-7075, DOI | none |
| xiao2024sinks | https://proceedings.iclr.cc/paper_files/paper/2024/hash/5e5fd18f863cbe6d8ae392a93fd271c9-Abstract-Conference.html (+ BibTeX); arXiv 2309.17453 | title, 5 authors, ICLR 2024, pp. 21875-21895 | A trailing space in "Lewis, Mike " in the source BibTeX was removed. |
| darcet2024registers | https://proceedings.iclr.cc/paper_files/paper/2024/hash/0b408293619f725fd30162af057e531a-Abstract-Conference.html (+ BibTeX); arXiv 2309.16588 | title, 4 authors, ICLR 2024, pp. 2632-2652 | none |
| sun2024massive | https://colmweb.org/2024/AcceptedPapers.html (links openreview.net/forum?id=F7aAhfitX6); arXiv 2402.17762 | title, 4 authors, COLM 2024 | **Venue added:** the paper is peer-reviewed at COLM 2024 (arXiv comment: "First Conference on Language Modeling (COLM), 2024"). No pages are published. |
| gu2025sinks | https://proceedings.iclr.cc/paper_files/paper/2025/hash/f1b04face60081b689ba740d39ea8f37-Abstract-Conference.html (+ BibTeX); arXiv 2410.10781 | title, 8 authors, ICLR 2025, pp. 97114-97144 | none. arXiv says "ICLR 2025 (Spotlight)". |
| abadi2016deep | DOI 10.1145/2976749.2978318 (Crossref) | title, 7 authors, CCS '16, pp. 308-318, ACM | none |
| schulman2017ppo | arXiv API id 1707.06347 | title, 5 authors, 2017, cs.LG | none |
| kakade2002approximately | https://icml.cc/Conferences/2002/paper_list.txt and https://icml.cc/Conferences/2002 (official ICML 2002 site); author CV https://hunch.net/~jl/resume/resume.html; OpenAlex W1575592356 | title, 2 authors, ICML 2002 = "The Nineteenth International Conference on Machine Learning", UNSW Sydney, 2002 | **Partly secondary.** The pages 267-274 come only from OpenAlex (a secondary index), because dblp and the ACM DL were blocked and the ICML site gives no pages. The `url` is the author-hosted PDF linked from Langford's CV (HTTP 200). No DOI exists. |
| muennighoff2023scaling | https://proceedings.neurips.cc/paper_files/paper/2023/hash/9d89448b63ce1e2e8dc7af72c984c196-Abstract-Conference.html (+ BibTeX); DOI 10.52202/075280-2191 (Crossref) | title, 9 authors, NeurIPS 36, pp. 50358-50376 | Author order differs. arXiv v5 has "Piktus, Tazi"; NeurIPS and the DOI record have "Tazi, Piktus". The peer-reviewed order (Tazi before Piktus) is used. |
| bishop1995training | DOI 10.1162/neco.1995.7.1.108 (Crossref) | title, author (Chris M. Bishop), Neural Computation 7(1):108-116, 1995 | none |
| wager2013dropout | https://proceedings.neurips.cc/paper_files/paper/2013/hash/38db3aed920cf82ab059bfccbd02be6a-Abstract.html (+ BibTeX) | title, 3 authors, NIPS 26, 2013 | The proceedings give no pages. |
| gururangan2018annotation | https://aclanthology.org/N18-2017.bib | title, 6 authors, NAACL-HLT 2018 Vol. 2 (Short), pp. 107-112, DOI | none |
| poliak2018hypothesis | https://aclanthology.org/S18-2023.bib | title, 5 authors, *SEM 2018 (Seventh Joint Conf. on Lexical and Computational Semantics), pp. 180-191, DOI | none |
| kaushik2018reading | https://aclanthology.org/D18-1546.bib | title, 2 authors, EMNLP 2018, pp. 5010-5015, DOI | **Title corrected:** the full official title adds the subtitle "A Critical Investigation of Popular Benchmarks". |
| muller2019label | https://proceedings.neurips.cc/paper_files/paper/2019/hash/f1748d6b0fd9d439f71450117eba2725-Abstract.html (+ BibTeX); arXiv 1906.02629 | title, 3 authors, NeurIPS 32, 2019 | The proceedings title is "When does label smoothing help?" (it prints the same after sentence-casing). The proceedings give no pages. |
| loshchilov2019decoupled | https://iclr.cc/Conferences/2019/Schedule?type=Poster (card links forum?id=Bkg6RiCqY7); arXiv 1711.05101 | title, 2 authors, ICLR 2019 | none |
| hu2022lora | https://iclr.cc/Conferences/2022/Schedule?type=Poster (card links forum?id=nZeVKeeFYf9); arXiv 2106.09685 | title, 8 authors, ICLR 2022 | iclr.cc gives "Edward Hu" and "yelong shen". The entry uses the arXiv names "Edward J. Hu" and "Yelong Shen". |
| baig2025attentiondrop | arXiv API id 2504.12088 (v2, updated 2025-09-19) | title, 5 authors (Baig, Gillani, A. A. Khan, Shah, M. O. Khan), 2025, cs.CV | none. There is no journal reference on arXiv. |
| sander2022sinkformers | https://proceedings.mlr.press/v151/sander22a.html (embedded BibTeX + citation meta) | title, 4 authors, AISTATS 2022, PMLR 151:3515-3530 | Leading and trailing spaces in the PMLR title were trimmed. |
| lin2025twilight | https://proceedings.neurips.cc/paper_files/paper/2025/hash/ad2f57116d62a9d8dcfaba6a168715e4-Abstract-Conference.html (+ BibTeX); DOI 10.52202/085713-3986 (Crossref); arXiv 2502.02770 | title, 9 authors, NeurIPS 2025 (vol. 38) | **Venue upgraded:** the paper is now in the NeurIPS 2025 proceedings (arXiv v5: "NeurIPS 2025 (spotlight)"), so the peer-reviewed version replaces the arXiv one. **Page conflict:** the NeurIPS site BibTeX gives pp. 119406-119432, while the DOI record gives pp. 132387-132413. Pages are left out and the DOI is kept. |
| maekaku2022attention | DOI 10.21437/Interspeech.2022-11441 (Crossref) | title, 4 authors, Interspeech 2022, pp. 1071-1075, ISCA | none |
| jain2019attention | https://aclanthology.org/N19-1357.bib | title, 2 authors, NAACL-HLT 2019 Vol. 1, pp. 3543-3556, DOI | none |
| nadeau2003inference | DOI 10.1023/A:1024068626366 (Crossref) | title, 2 authors, Machine Learning 52(3):239-281, 2003 | none |
| kim2021lipschitz | https://proceedings.mlr.press/v139/kim21i.html (embedded BibTeX + citation meta); arXiv 2006.04710 | title, 3 authors, ICML 2021, PMLR 139:5562-5571 | none |
| dao2022flashattention | https://proceedings.neurips.cc/paper_files/paper/2022/hash/67d57c32e20fd0a7a302cb81d36e40d5-Abstract-Conference.html (+ BibTeX); DOI 10.52202/068431-1189 (Crossref); arXiv 2205.14135 | title, 5 authors, NeurIPS 35, pp. 16344-16359 | The proceedings and the DOI record give "Fu, Dan"; arXiv gives "Daniel Y. Fu". The proceedings form is used. |
| fan2020reducing | https://iclr.cc/virtual_2020/poster_SylO2yStDr.html (links openreview.net/forum?id=SylO2yStDr); arXiv 1909.11556 | title, 3 authors, ICLR 2020 | none |
| wu2021unidrop | https://aclanthology.org/2021.naacl-main.302.bib (ID found from https://aclanthology.org/volumes/2021.naacl-main/); arXiv 2104.04946 | title, 8 authors, NAACL-HLT 2021, pp. 3865-3878, DOI | A first probe of the ID 2021.naacl-main.392 returned an unrelated paper (Zeng & Nie), which was not used. The correct ID is .302. |
| xiong2024mirage | https://aclanthology.org/2024.findings-acl.372.bib; arXiv 2402.13178 | title, 4 authors, Findings of ACL 2024, pp. 6233-6251, DOI | none. "MIRAGE" is the benchmark's name and is not part of the title. |

## Skipped keys (as instructed)

- `gu2020kl?`
- `lan2025bootstrapped?`
- `xu2020layerdrop`, which is replaced by `fan2020reducing`.

## UNVERIFIED

None.

The one field that could not be matched to a primary source is the page range of `kakade2002approximately`. Its title, authors, venue and year were all confirmed against primary sources. If a fully primary record is needed, the pages can be removed, or checked against the printed ICML 2002 proceedings (Morgan Kaufmann).

---

# Second batch: 35 keys appended on 2026-09-27

- **Date of retrieval:** 2026-09-27.
- **Result:** all 35 requested keys were verified and appended. None are UNVERIFIED. The file now has 83 entries (48 + 35).
- **Existing entries:** none of the 48 earlier entries was changed. A byte comparison confirmed that the first 524 lines are identical to the file before the append. No new key duplicates an existing key.
- **Test build:** tectonic with `elsarticle-harv.bst` and hyperref, as in `main.tex`, on a scratch copy with `\nocite{*}`.
  - It produced 83 `\bibitem`s and no errors.
  - BibTeX gave 16 "empty pages" warnings: the 11 earlier ones, plus 5 new ones for sources that publish no page numbers (`ghiasi2018dropblock`, `lopezpaz2016unifying`, `plappert2018parameter`, `wang2022eliminating`, `zhang2024raft`).
- **Method:** the same as the first batch. No entry was written from memory. The sources were:
  - DOI content negotiation (`Accept: application/x-bibtex`) and the Crossref REST API (`api.crossref.org/works/<DOI>`), with no mailto or other identifier sent;
  - the arXiv API;
  - ACL Anthology `.bib` files;
  - NeurIPS proceedings pages and their `-Bibtex.bib` files;
  - PMLR pages (embedded BibTeX);
  - the iclr.cc schedule pages and the ICLR 2016 accepted-papers page;
  - the COLM 2024 accepted-papers page;
  - the ISCA archive, the AAAI OJS page, Project Euclid, the Springer book page and its free front-matter and preview PDFs, the Variance journal page (with its JATS XML and PDF), the JITIE journal page, and the CAS-hosted scan of the ASTIN Bulletin article.
- **Access limits:** OpenReview still returns "ChallengeRequiredError" (HTTP 403), so the ICLR and COLM forum IDs were taken from the links printed on iclr.cc and colmweb.org.
- **Discovery searches:** OpenAlex title searches were used only to find out whether a peer-reviewed version exists. Every venue found that way was then confirmed from the primary record (for example, the ISIT 2026 DOI for `tzachristas2025topk`).

## Per-key log (second batch)

| Key | Source used | Verified | Discrepancy / note |
|---|---|---|---|
| tzachristas2025topk | DOI 10.1109/ISIT62367.2026.11653946 (content negotiation and Crossref); arXiv API 2512.07647 | title, 5 authors, 2026 IEEE ISIT (Guangzhou, 28 Jun - 3 Jul 2026), pp. 1-6, IEEE | **Venue upgraded:** a peer-reviewed ISIT 2026 version exists, so it is used instead of arXiv v1 (2025-12-08). **Year:** the entry has `year = 2026`, while the key still reads "2025". The key was kept as requested; only its label is out of date. |
| luo2021dropout | DOI 10.21437/Interspeech.2021-1066 (content negotiation and Crossref); https://www.isca-archive.org/interspeech_2021/luo21b_interspeech.html (embedded BibTeX); arXiv 2107.04227 | title, 4 authors, Interspeech 2021, pp. 1169-1173, ISCA | none. The arXiv comment is "will be presented in INTERSPEECH 2021". |
| utkin2022attention | DOI 10.1016/j.neunet.2022.07.029 (content negotiation and Crossref) | title, 2 authors (Utkin, Konstantinov), Neural Networks 154:346-359, Oct 2022 | none |
| wang2024loradropout | https://aclanthology.org/2024.findings-acl.119.bib (ID from https://aclanthology.org/volumes/2024.findings-acl/); arXiv 2403.00812 | title, 6 authors, Findings of ACL 2024, pp. 1995-2008, DOI 10.18653/v1/2024.findings-acl.119 | none |
| wu2021notall | arXiv API 2104.04692 (v3) | title, 3 authors (Hongqiu Wu, Hai Zhao, Min Zhang), 2021, cs.CL | No peer-reviewed version was found (arXiv has no journal reference; OpenAlex and Crossref title searches found none). It is kept as `@misc`. |
| yang2022addrop | https://proceedings.neurips.cc/paper_files/paper/2022/hash/4fdf8d49476a8001c91f9e9e90530e13-Abstract-Conference.html (+ `-Bibtex-Conference.bib`); DOI 10.52202/068431-0894 (Crossref); arXiv 2210.05883 | title, 5 authors, NeurIPS 35, pp. 12310-12324 | **Author typo fixed:** the NeurIPS BibTeX has "Deng, JInghao". The DOI record and arXiv have "Jinghao", which is used. |
| fan2020bam | https://proceedings.neurips.cc/paper_files/paper/2020/hash/bcff3f632fd16ff099a49c2f0932b47a-Abstract.html (+ `-Bibtex.bib`); arXiv 2010.10604 | title, 4 authors, NeurIPS 33, pp. 16362-16376 | none. There is no DOI; a `url` is given, as for `liang2021rdrop`. |
| amirahmadi2025anaa | DOI 10.3389/frai.2025.1663484 (content negotiation and Crossref) | title, 3 authors (Amirahmadi, Etminani, Ohlsson), Front. Artif. Intell. 8, article 1663484, published 2025-09-17 | Frontiers has no page range. The article number is used as `pages`, as for `rohanian2023compact`. |
| kitada2021attention | DOI 10.1109/ACCESS.2021.3093456 (content negotiation and Crossref) | title, 2 authors (Kitada, Iyatomi), IEEE Access 9:92974-92985, 2021 | none |
| shi2026trustmass | arXiv API 2608.25230 (v2, 2026-08-28) | title, 2 authors (Jack Shi, Jerry Gu), 2026, cs.LG | No peer-reviewed version was found. v2 note: "revised wording in 2.3 for increased accuracy (main results unchanged)". |
| emadi2026exact | arXiv API 2602.18849 (v2, 2026-08-17) | title, 1 author (Seyed Morteza Emadi), 2026, cs.LG | No peer-reviewed version was found. **Check what you cite:** v2 is a "major revision". It "removed an unsupported path-length scaling theorem and its DeepNorm derivation" and "corrected the uniform-attention parity case"; the exact softmax sensitivity theorem is unchanged. Any claim cited from v1 should be re-checked against v2. |
| koloskova2023revisiting | https://proceedings.mlr.press/v202/koloskova23a.html (embedded BibTeX + citation meta); arXiv 2305.01588 | title, 3 authors, ICML 2023, PMLR 202:17343-17363 | PMLR writes "Stich, Sebastian U" without a period. The arXiv form "Sebastian U. Stich" is used. |
| schulman2015trpo | https://proceedings.mlr.press/v37/schulman15.html (embedded BibTeX); the PMLR PDF http://proceedings.mlr.press/v37/schulman15.pdf; arXiv 1502.05477 | title, 5 authors, ICML 2015, PMLR 37:1889-1897, Lille | **Author order conflict:** the PMLR metadata lists Schulman, Levine, Abbeel, Jordan, Moritz. The paper's own title block in the PMLR PDF, and arXiv, give Schulman, Levine, Moritz, Jordan, Abbeel. The PDF order is used. The PDF footer says "31st International Conference", which is a typo; the PMLR page says 32nd, which is used. |
| plappert2018parameter | https://iclr.cc/Conferences/2018/Schedule?type=Poster (the card links openreview.net/forum?id=ByBAl2eAZ); arXiv 1706.01905 | title, 9 authors, ICLR 2018 | iclr.cc gives "Richard Chen". The arXiv form "Richard Y. Chen" is used. arXiv v2 says "Updated to camera-ready ICLR submission". No pages are published. |
| andrew2021adaptive | https://proceedings.neurips.cc/paper_files/paper/2021/hash/91cff01af640a24e7f9f7a5ab407889f-Abstract.html (+ `-Bibtex.bib`); arXiv 1905.03871 | title, 4 authors, NeurIPS 34, pp. 17455-17466 | The proceedings give "McMahan, Brendan". The arXiv form "H. Brendan McMahan" is used, which also matches `abadi2016deep`. |
| ghiasi2018dropblock | https://proceedings.neurips.cc/paper_files/paper/2018/hash/7edcfb2d8f6a659ef4cd1e6c9b6d7079-Abstract.html (+ `-Bibtex.bib`); arXiv 1810.12890 | title, 3 authors, NeurIPS 31, 2018 | The NeurIPS BibTeX has "Le, Quoc V"; "Quoc V." (from the page meta and arXiv) is used. The proceedings give no pages. |
| halliwell2012excess | https://variancejournal.org/article/77490-the-mathematics-of-excess-losses (citation meta); JATS XML https://variancejournal.org/article/77490.xml; PDF https://variancejournal.org/article/77490.pdf; DOI 10.66573/001c.77490 (Crossref; it resolves to the journal page) | title, author (Leigh J. Halliwell), Variance 6(1), pp. 32-47, 2012, Casualty Actuarial Society | **DOI found:** the journal has registered a DOI (10.66573/001c.77490), so it is used instead of a casact.org PDF link. Crossref has no pages. The page range 32-47 comes from the JATS `fpage`/`lpage` and matches the printed folios (32 and 47) in the PDF. |
| bartlett2005local | DOI 10.1214/009053605000000282 (content negotiation and Crossref); https://projecteuclid.org/journals/annals-of-statistics/volume-33/issue-4/Local-Rademacher-complexities/10.1214/009053605000000282.full (citation meta); arXiv math/0508275 (journal-ref) | title, 3 authors, Ann. Statist. 33(4):1497-1537, Aug 2005 | Crossref has no pages. The pages come from Project Euclid and agree with the arXiv journal-ref. |
| wang2022eliminating | https://iclr.cc/Conferences/2022/Schedule?type=Poster (the card links openreview.net/forum?id=B3Nde6lvab); arXiv 2102.04297 | title, 3 authors (Wang, Oh, Rhee), ICLR 2022 | none. The arXiv comment is "Proceedings of 2022 International Conference on Learning Representations". No pages are published. |
| wen2024characterizing | https://aclanthology.org/2024.findings-emnlp.197.bib (ID from https://aclanthology.org/volumes/2024.findings-emnlp/) | title, 3 authors (Wen, Howe, Wang), Findings of EMNLP 2024, pp. 3437-3450, DOI 10.18653/v1/2024.findings-emnlp.197 | none |
| lala2023paperqa | arXiv API 2312.07559 (v2) | title, 6 authors, 2023, cs.CL | No peer-reviewed version was found. The accent in "L\'{a}la" is LaTeX-escaped. |
| sadanandan2026cot | arXiv API 2603.25960 (v1) | title, 2 authors (Sadanandan, Behzadan), 2026, cs.CL | A Research Square copy also exists (DOI 10.21203/rs.3.rs-8759042/v1, Crossref type `posted-content`). It is a preprint, not a peer-reviewed version, so the arXiv version is kept. |
| zhang2024raft | https://colmweb.org/2024/AcceptedPapers.html (links openreview.net/forum?id=rzQGHXNReU); arXiv 2403.10131 | title, 7 authors, COLM 2024 | **Venue added:** the paper was accepted at COLM 2024, so it is cited as the COLM paper (formatted like `sun2024massive`). The COLM page gives "Shishir G Patil"; the arXiv form "Shishir G. Patil" is used. No pages are published. |
| xiao2025kailin | arXiv API 2504.19565 (v3, 2025-12-18) | title, 6 authors, 2025, cs.CL | No peer-reviewed version was found. "KAILIN" is not part of the title. |
| huang2025selfreflective | https://journal.jci.co.id/jitie/article/view/820 (citation and DC meta) | title, 5 authors (Tao Huang, Ashley Lewis, Hui Zhang, Steven Walker, Ling Gao), J. Information Technology and Informatics Engineering 1(2):23-34, published 2025-10-31, ISSN 3123-7827 | The journal page shows no DOI and Crossref has no record, so only a `url` is given; no DOI was guessed. **Venue-quality caution:** this is the first volume of a new journal. Its reference list includes unrelated works from "J. Adv. Comput. Syst." (for example, HDFS log anomaly detection and trade-receivable disclosure QA). The bibliographic data is verified, but it is worth checking that this source is acceptable for a Q1 venue before relying on it. |
| fries2022bigbio | https://proceedings.neurips.cc/paper_files/paper/2022/hash/a583d2197eafc4afdd41f5b8765555c5-Abstract-Datasets_and_Benchmarks.html (+ `-Bibtex-Datasets_and_Benchmarks.bib`); DOI 10.52202/068431-1870 (Crossref); arXiv 2206.15076 | 43 authors, NeurIPS 35 Datasets and Benchmarks Track, pp. 25792-25806 | **Title:** the proceedings title is "BigBio: A Framework for Data-Centric Biomedical Natural Language Processing". arXiv writes "BigBIO"; the requested short form "...Biomedical NLP" is not the official title. The proceedings title is used. **Names:** the peer-reviewed author list is used. It differs from arXiv in several names: "Sunny Kang" vs "Myungsun Kang", "Rosaline Su" vs "Ruisi Su", and middle names or initials dropped for Posada, Giorgi, Broad and Deshmukh. "(Datasets and Benchmarks Track)" is added to `booktitle`, based on the proceedings URL. |
| theimerlienhard2026openmeditron | arXiv API 2605.16215 (v2, 2026-05-29) | title, 8 authors, 2026, cs.AI | No peer-reviewed version was found. The arXiv comment is "Preprint". |
| luo2022biogpt | DOI 10.1093/bib/bbac409 (content negotiation and Crossref) | title, 7 authors, Brief. Bioinform. 23(6), article bbac409, 2022 | The article number bbac409 is used as `pages`. It was published online 2022-09-24, in the print issue of 2022-11-19. |
| hu2023seat | DOI 10.1609/aaai.v37i11.26517 (content negotiation and Crossref); https://ojs.aaai.org/index.php/AAAI/article/view/26517 (DC meta) | title, 6 authors, AAAI 2023 = Proc. AAAI 37(11):12907-12915, "AAAI Technical Track on Speech & Natural Language Processing" | none |
| ohlin1969class | DOI 10.1017/S0515036100008102 (content negotiation and Crossref); full-text scan at https://www.casact.org/sites/default/files/database/astin_vol5no2_249.pdf | title, author (Jan Ohlin, Stockholm), ASTIN Bulletin 5(2):249-266, May 1969, CUP | none. The content claim is confirmed; see below. |
| kaas2008modern | DOI 10.1007/978-3-540-70998-5 (content negotiation and Crossref); https://link.springer.com/book/10.1007/978-3-540-70998-5 (bibliographic panel); front matter https://link.springer.com/content/pdf/bfm:978-3-540-70998-5/1 | 4 authors, title "Modern Actuarial Risk Theory", subtitle "Using R", 2nd edition, Springer Berlin, Heidelberg, 2008, hardcover ISBN 978-3-540-70992-3 (eBook 978-3-540-70998-5), XVIII + 382 pp. | Crossref's title lacks the subtitle; the subtitle comes from the Springer page. The content claim is partly confirmed; see below. |
| errami2008tale | DOI 10.1038/451397a (content negotiation and Crossref) | title, 2 authors (Mounir Errami, Harold Garner), Nature 451(7177):397-399, Jan 2008 | none. The content (duplicate publication in MEDLINE) was not checked; the request asked only for bibliographic verification. |
| wu2023tlm | https://aclanthology.org/2023.emnlp-main.871.bib (ID from https://aclanthology.org/volumes/2023.emnlp-main/); arXiv 2310.18738 | 6 authors, EMNLP 2023 (main), pp. 14099-14111, DOI 10.18653/v1/2023.emnlp-main.871 | **Title:** the official title has a prefix: "TLM: Token-Level Masking for Transformers". |
| vapnik2009new | DOI 10.1016/j.neunet.2009.06.042 (content negotiation and Crossref) | title, 2 authors (Vapnik, Vashist), Neural Networks 22(5-6):544-557, Jul 2009 | none |
| lopezpaz2016unifying | https://iclr.cc/archive/www/doku.php%3Fid=iclr2016:accepted-main.html (ICLR 2016 accepted papers, conference track, poster presentations; the title links to arXiv 1511.03643); arXiv API 1511.03643 (v3) | title, 4 authors (Lopez-Paz, Bottou, Sch\"{o}lkopf, Vapnik), ICLR 2016 | ICLR 2016 published its papers through arXiv, with no OpenReview forum or pages, so the `url` is the arXiv abstract page. The arXiv journal-ref reads "Proceedings of the International Conference on Learning Representations (2016) 1-10"; the "1-10" is a page count, not a proceedings page range, so it is not used. |

## Content check: stop-loss (truncation) retention is minimal in convex order

**The claim checked.** The paper cites these sources for the following: suppose Y satisfies 0 <= Y <= X pathwise and E[Y] is fixed. Equivalently, the retained risk is X - I(X), with 0 <= I(X) <= X and fixed mean. Then the truncation min(X, t), which is the stop-loss or excess-of-loss retention, is minimal in convex (stop-loss) order. Equivalently, it minimises the variance and every convex risk measure.

### ohlin1969class: CONFIRMED

The quotes below are transcribed from the text layer of the CAS-hosted scan. Obvious OCR errors were corrected; the page numbers are the journal folios. The text layer does not reliably tell strict from non-strict inequality signs (< vs <=), so the inequality signs below follow the mathematics, not the OCR.

- **Setup (Sect. 0, p. 249):**
  - "An insurer, whose total claims for a certain period may be regarded as a random variable x with expected value Ex = m, wishes to cede part of his business to a reinsurer."
  - "[W]e will describe a reinsurance contract by a transformation (or function) T that for a given x yields the amount Tx borne by the cedent."
- **Admissible class A (Sect. 4, p. 259):** "A measurable transformation T is said to be admissible under A if 0 <= Tx <= x, c = int (x - Tx) dF(x), where c is a fixed constant such that 0 < c < m = Ex."
- **The stop-loss transformation (p. 259):** "T*x = x for x <= n0, T*x = n0 for x > n0". This is min(x, n0). The OCR of the condition that fixes n0 is garbled, but p. 260 states: "It is easily seen that both T* and [the quota transformation] are admissible under all four sets of conditions". So E(x - T*x) = c.
- **Theorem (p. 260):** "Let T_A, T_B, T_C and T_D be any transformations admissible under A, B, C and D respectively. Let T* and [T~] be the stop loss and quota transformations defined above. Then the following statements are true: (1) T*x is less dispersed than T_A x ..." It adds: "(Note: (1) is an extension to any m.o.d. W_phi of the Borch-Kahn theorem on minimum variance ...)".
- **Proof of (1) (p. 260):**
  - "F_A(t) >= F(t), because F_A(t) = P(T_A x <= t) = P(x <= t) + P(T_A x <= t < x) >= F(t)."
  - "F*(t) = F(t) for t < n0, F*(t) = 1 for t >= n0. Hence F*(t) <= F_A(t) for t < n0, F*(t) >= F_A(t) for t >= n0. Since E T*x = E T_A x = m - c, we may apply Lemma 2 and conclude that (1) is true."
- **Lemma 2 (p. 256), the "Ohlin lemma", which gives convex order directly:** "Let x1 and x2 be two r.v:s with the same finite mean Ex1 = Ex2, and F1 and F2 their c.d.f:s. If there exists a t0 such that F1(t) <= F2(t) for t < t0, F1(t) >= F2(t) for t > t0, then Ef(x1) <= Ef(x2) for any c.c.f. f." Here c.c.f. means a function that is continuous and convex on the whole real line (p. 253).
- **Lemma 1 and the class of dispersion measures (pp. 251-255):**
  - Lemma 1: "If Ef(x) <= Ef(y) for any c.c.f. f, then x is less dispersed than y".
  - The dispersion measures are W_phi(x) = inf_mu E phi(x - mu), for any convex phi >= 0 with phi(0) = 0. This class includes the variance and the mean absolute deviation.
  - p. 252: the optimal contract "minimizes not only the variance and the mean deviation but any measure of dispersion of this type."

**What is confirmed.** The proof of statement (1), through Lemma 2, shows that E f(min(X, n0)) <= E f(T_A X) for every continuous convex f and every admissible T_A. With equal means, this is exactly min(X, n0) <=_cx T_A X, that is, convex order. It then follows for the variance and for every W_phi.

**Three caveats on the wording:**

- **(a) Y as a function of X.** Ohlin states the result for Y = T(X), a measurable function of X. The proof of (1) uses only two facts: Y <= X a.s., which gives F_Y >= F_X, and E[Y] = E[min(X, n0)]. This is our reading of the proof, not a statement in the paper. If the manuscript's Y is not a function of X (for example, randomised dropping), it should say "by the argument of Ohlin (1969, proof of the Theorem, statement (1))" and not claim that Ohlin states it for general Y.
- **(b) Fixed ceded amount.** Ohlin fixes the ceded amount c = E(X - Y), with 0 < c < m. This is the same as fixing E[Y] in (0, m).
- **(c) "Every convex risk measure."** Ohlin proves the result for every expected convex function, and hence for the variance and every W_phi. He does not prove it for "every convex risk measure". Extending it to risk measures needs one more fact: the risk measure is law-invariant and consistent with convex order. Neither source checked here states this. Suggested wording: "minimal in convex order, i.e. it minimises E f(Y) for every convex f, in particular the variance". Otherwise, cite a separate consistency result, which was not verified here.

### kaas2008modern: PARTLY CONFIRMED (chapter and section level)

Confirmed from the publisher's free front matter and chapter previews:

- **Table of contents (front matter):**
  - "1.4 Stop-loss reinsurance ... 8";
  - "7.3.2 Stop-loss order ... 159";
  - "7.4.4 Optimal reinsurance ... 168".
- **Preface (pp. x-xi):** "Also, we can prove that the stop-loss insurance, demonstrated to be optimal as regards the variance of the retained risk in Chapter 1, is also preferable, other things being equal, in the eyes of all risk averse decision makers."
- **Chapter 1, pp. 1-2 (Springer preview):** "If a reinsurer is insuring the total claim amount of an insurer's portfolio of risks, insurance with a fixed maximal retained risk is called a stop-loss reinsurance. From the theory of ordering of risks, we will see that this type of reinsurance is optimal for risk averse decision makers. In this chapter we prove that a stop-loss reinsurance results in the smallest variance of the retained risk."
- **Chapter 7, p. 150 (Springer preview):** "In Chapter 1, we showed that stop-loss reinsurance is optimal in the sense that it gives the lowest variance for the retained risk when the mean is fixed. In this chapter we are able to prove the stronger assertion that stop-loss reinsurance leads to a retained loss that is preferable for any risk averse decision maker."

**Not verified.** The body of Sect. 1.4 (pp. 8-13) and Sect. 7.4.4 (p. 168) is paywalled. So the exact theorem or example numbers, and the exact admissibility conditions on I(x), were not seen. Theorem numbers should therefore not be cited. "Preferable for any risk averse decision maker" at a fixed mean is the stop-loss order with equal means, which is convex order.

### Citation to use

- **Primary:** Ohlin (1969, Sect. 4, Theorem, statement (1); proof via Lemma 2) for the convex-order minimality of min(X, t) among 0 <= Y <= X with fixed mean.
- **Textbook:** Kaas et al. (2008), Sect. 1.4 for the minimum-variance statement and Sect. 7.4.4 for the stop-loss (convex) order statement.
- **Historical:** Ohlin credits the variance-only result to Borch (Trans. XVIth Int. Congress of Actuaries, Vol. 2, pp. 597-610) and Kahn (ASTIN Bulletin Vol. 1, Part V, pp. 265-272). These were not fetched here and are not in the .bib.

## UNVERIFIED (second batch)

None.
