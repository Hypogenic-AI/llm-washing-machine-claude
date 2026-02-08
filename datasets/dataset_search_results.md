# Dataset Search Results: Compound Nouns and Multi-Word Expressions

**Research Goal**: Find datasets suitable for studying how compound concepts like "washing machine" are stored in LLMs.

**Search Date**: February 7, 2026

**Search Coverage**:
- HuggingFace datasets
- Papers with Code
- Academic publications
- Web search for dataset repositories

---

## Executive Summary

This search identified **15+ major datasets** spanning compound nouns, multiword expressions, compositionality judgments, and idiomatic expressions. Key findings:

- **Largest compound noun dataset**: Tratz & Hovy (19,158 compounds with 37 semantic relations)
- **Most comprehensive compositionality dataset**: Ramisch et al. (multilingual: 190 EN, 180 FR, 180 PT compounds with human judgments)
- **Largest idiom dataset**: MAGPIE (56,622 instances covering 1,756 idiom types)
- **Most directly relevant to LLM probing**: Ormerod et al.'s Kitchen Chair paper uses Gagné (2001) dataset (300 compounds organized by thematic relations)

**Key Gap Identified**: While many datasets exist for compound nouns and MWEs, few are specifically designed for probing LLM representations. Most focus on traditional NLP tasks (classification, parsing) rather than mechanistic interpretability.

---

## 1. Compound Noun Datasets with Semantic Relations

### 1.1 Tratz & Hovy Dataset (2010)

**Description**: The largest publicly-available annotated noun compound dataset for automatic semantic interpretation.

**Size**:
- Original: 17,509 compounds with 43 semantic relations
- Revised (Tratz 2011): 19,158 compounds with 37 semantic relations

**Annotations Include**:
- 37 fine-grained semantic relations (revised taxonomy)
- Derived from established linguistic resources including:
  - Penn Treebank
  - NomBank
  - Prague Czech-English Dependency Treebank 2.0 (PCEDT)

**Download**:
- Official site: http://www.isi.edu/publications/licensed-sw/
- Requires license agreement

**Relevance to Research**: ★★★★☆
- Excellent for understanding semantic relations in compounds
- Large scale allows statistical analysis
- Could be used to test whether LLMs encode specific relation types differently
- **Application**: Test if "washing machine" (INSTRUMENT relation) has different representation patterns than other relation types

**Key Papers**:
- Tratz & Hovy (2010). "A Taxonomy, Dataset, and Classifier for Automatic Noun Compound Interpretation." ACL
- Available at: https://aclanthology.org/P10-1070/
- Paper PDF: https://www.cs.cmu.edu/~hovy/papers/10ACL-nounnoun-compound-rels.pdf

---

### 1.2 Gagné (2001) Dataset

**Description**: Dataset for investigating thematic relations in noun-noun compounds, used in psycholinguistic experiments and LLM probing research.

**Size**: 300 English noun-noun compounds organized into 60 groups of 5 compounds each

**Annotations Include**:
- 16 thematic relation types
- Organized for relational priming experiments
- Each compound categorized by dominant thematic relation

**Download**:
- Extension dataset: https://osf.io/gvc2w/ (108 novel compounds with active/passive interpretation)
- Original Gagné dataset: Contact authors or check psycholinguistics repositories

**Relevance to Research**: ★★★★★
- **DIRECTLY USED** in the "Kitchen Chair" paper (Ormerod et al., 2024) for probing transformer representations
- Psycholinguistically validated
- Smaller size makes it manageable for detailed analysis
- **Application**: Direct replication and extension of Ormerod et al.'s probing methodology

**Key Papers**:
- Gagné & Shoben (2001). "Influence of Thematic Relations on the Comprehension of Modifier-Noun Combinations."
- Ormerod, Martinez-del-Rincon, & Devereux (2024). "How Is a 'Kitchen Chair' like a 'Farm Horse'? Exploring the Representation of Noun-Noun Compound Semantics in Transformer-based Language Models." Computational Linguistics
- Available at: https://direct.mit.edu/coli/article/50/1/49/118133/

**Extension Dataset Details**:
- Title: "A Dataset of 108 Novel Noun-Noun Compound Words with Active and Passive Interpretation"
- Authors: Open Psychology Data
- URL: https://openpsychologydata.metajnl.com/articles/10.5334/jopd.93
- Includes lexical-semantic features for interpretation

---

### 1.3 Fares (2016) Dataset

**Description**: Dataset for joint noun-noun compound bracketing and semantic interpretation.

**Size**: Sizeable collection derived from established linguistic resources

**Annotations Include**:
- Syntactic bracketing (for multi-word compounds like "computer science department")
- Two different taxonomies of semantic relations per compound
- Draws from:
  - Penn Treebank
  - NomBank
  - Prague Czech-English Dependency Treebank 2.0

**Download**:
- Paper: https://aclanthology.org/P16-3011/
- Dataset availability: Check Papers with Code or contact authors (availability was being investigated at time of publication)

**Relevance to Research**: ★★★☆☆
- Useful for understanding structural composition in longer compounds
- Dual annotation scheme allows cross-taxonomy comparison
- Less directly applicable to single two-word compounds like "washing machine"

**Key Papers**:
- Fares (2016). "A Dataset for Joint Noun-Noun Compound Bracketing and Interpretation." ACL Student Research Workshop

---

### 1.4 Kim & Baldwin (2005) Dataset

**Description**: Early influential dataset for noun compound semantic relation classification.

**Size**: Not specified in search results

**Annotations Include**:
- 20 semantic relations

**Download**: Check through University of Melbourne NLP group or contact authors

**Relevance to Research**: ★★☆☆☆
- Historical importance but smaller and less comprehensive than Tratz & Hovy
- Useful for comparison with more recent datasets

---

### 1.5 IIT Bombay Noun Compound Datasets

**Description**: Collection of datasets for noun compound interpretation research, hosted by IIT Bombay.

**Size**: Multiple datasets available

**Annotations Include**:
- Various semantic relation taxonomies
- Compositionality information

**Download**: https://www.cse.iitb.ac.in/~girishp/nc-dataset/

**Relevance to Research**: ★★★☆☆
- Centralized repository for multiple NC datasets
- Good for comparative studies
- Check site for specific dataset details

---

## 2. Compositionality Datasets

### 2.1 Reddy et al. (2011) & Ramisch et al. (2019) - NC Compositionality Dataset

**Description**: Gold-standard dataset for noun compound compositionality with human judgments across multiple languages.

**Size**:
- English: 190 nominal compounds
- French: 180 nominal compounds
- Brazilian Portuguese: 180 nominal compounds
- Total: 550 compounds across 3 languages

**Annotations Include**:
- Compositionality scores from 1 (fully idiomatic) to 5 (fully compositional)
- Averaged over 10-20 annotators per language
- Synonyms and similar expressions provided by annotators
- Raw individual annotations and filtered/averaged data
- Quality metrics and distribution graphics
- Gender, number, and example sentences for compounds

**Download**:
- **Primary source**: http://pageperso.lis-lab.fr/carlos.ramisch/?page=downloads/compounds
- Package includes:
  - Raw annotation files
  - Averaged filtered/unfiltered data
  - Scripts for quality estimation
  - Compound lists with metadata
  - MTurk/HTML data collection interfaces

**Relevance to Research**: ★★★★★
- **GOLD STANDARD** for compositionality research
- Multilingual coverage allows cross-linguistic comparison
- Human judgments provide ground truth for model evaluation
- Used as basis for Garcia et al. (2021) idiomaticity probing paper
- **Application**: Test whether LLM representations of compounds correlate with human compositionality judgments

**Key Papers**:
- Reddy, McCarthy, & Manandhar (2011). "An Empirical Study on Compositionality in Compound Nouns." IJCNLP
- Ramisch, Cordeiro, et al. (2019). "Unsupervised Compositionality Prediction of Nominal Compounds." Computational Linguistics
- Available at: https://direct.mit.edu/coli/article/45/1/1/1621/
- Garcia, Araujo, & Rademaker (2021). "Probing for idiomaticity in vector space models." EACL

---

### 2.2 Noun Compound Senses (NCS) Dataset

**Description**: Extended dataset based on Reddy et al. with full sentence contexts for each compound.

**Size**:
- 280 English noun compounds
- 180 Portuguese noun compounds
- Total: 9,220 sentences (5,620 English, 3,600 Portuguese)

**Annotations Include**:
- Human compositionality scores (0-5 scale)
- Two context conditions:
  - **NAT (Naturalistic)**: Real corpus sentences (avg 23.39 words EN, 13.03 PT)
  - **NEU (Neutral)**: "This is a/an <NC>" sentences (5 words)
- Variants with:
  - Synonyms of whole NC
  - Synonyms of components
  - Individual components isolated

**Download**:
- GitHub: https://github.com/marcospln/noun_compound_senses
- Associated with Garcia et al. (2021) paper

**Relevance to Research**: ★★★★★
- **HIGHLY RELEVANT**: Specifically designed for probing contextualized embeddings
- Controlled comparison between naturalistic and neutral contexts
- Multiple sentences per compound reduce variance
- Includes substitution variants for systematic testing
- **Application**: Perfect for testing context-dependence of "washing machine" representations

**Key Papers**:
- Garcia, Araujo, & Rademaker (2021). "Probing for idiomaticity in vector space models." EACL
- Paper: https://aclanthology.org/2021.eacl-main.201/

---

### 2.3 HuggingFace Compositionality Datasets

**2.3.1 mehdidc/compositionality**

**Description**: Compositionality dataset available on HuggingFace.

**Download**: https://huggingface.co/datasets/mehdidc/compositionality

**Details**: Limited information available - requires exploration of dataset card

**Relevance to Research**: ★★☆☆☆

---

**2.3.2 mehdidc/compositionality_hpsv1**

**Description**: Compositionality dataset (version 1) on HuggingFace.

**Download**: https://huggingface.co/datasets/mehdidc/compositionality_hpsv1

**Details**: Limited information available - requires exploration of dataset card

**Relevance to Research**: ★★☆☆☆

---

**2.3.3 chiayewken/bamboogle**

**Description**: Dataset for "Measuring and Narrowing the Compositionality Gap in Language Models" paper.

**Download**: https://huggingface.co/datasets/chiayewken/bamboogle

**Annotations Include**:
- Designed to test compositionality in language models
- Focus on compositional reasoning gaps

**Relevance to Research**: ★★★★☆
- Specifically designed for LLM evaluation
- May include compound concepts or multi-hop reasoning
- **Application**: Could complement traditional compound noun datasets

---

## 3. Multiword Expression (MWE) Datasets

### 3.1 DiMSUM (SemEval 2016 Task 10)

**Description**: Detecting Minimal Semantic Units and their Meanings - comprehensive MWE identification dataset.

**Size**:
- Test set: 16,500 words in 1,000 English sentences
- Training combines STREUSLE 2.1, Ritter Twitter, and Lowlands datasets

**Annotations Include**:
- MWE segmentation
- Supersense labels for semantic classes
- Broad-coverage lexical semantics representation

**Data Sources**:
- Online reviews (TrustPilot corpus)
- Tweets (Tweebank corpus)
- TED talk transcripts (IWSLT)

**Download**:
- GitHub: https://github.com/dimsum16/dimsum-data
- Official page: http://dimsum16.github.io/
- CMU Lexical Semantics Resources: https://www.cs.cmu.edu/~ark/LexSem/

**Relevance to Research**: ★★★☆☆
- Broad MWE coverage (not limited to noun compounds)
- Diverse text genres
- Good for general MWE detection evaluation
- Less focused on specific compound noun properties

**Key Papers**:
- Schneider et al. (2016). "SemEval-2016 Task 10: Detecting Minimal Semantic Units and their Meanings (DiMSUM)."
- Paper: http://www.cs.cmu.edu/~nschneid/dimsum.pdf

---

### 3.2 PARSEME Shared Task Datasets

**Description**: Multilingual datasets for verbal multiword expression identification and paraphrasing.

**Size**:
- PARSEME 1.x: 19+ languages, focused on verbal MWEs
- PARSEME 2.0 (2025-2026): All syntactic types of MWEs

**Annotations Include**:
- Edition 1.0, 1.1, 1.2: Verbal MWE identification
- Edition 2.0:
  - All MWE types (including noun compounds)
  - Paraphrasing annotations to remove idiomaticity

**Download**:
- GitLab: https://gitlab.com/parseme/sharedtask-data/-/tree/master/2.0
- Main site: https://typo.uni-konstanz.de/parseme/

**Timeline**: PARSEME 2.0 workshop planned for EACL 2026 (March 24-28, Morocco)

**Relevance to Research**: ★★★★☆
- **NEW**: PARSEME 2.0 includes noun compounds
- Multilingual coverage (19+ languages)
- Paraphrasing annotations unique for studying idiomaticity
- **Application**: Cross-linguistic study of compound representations

**Key Papers**:
- Savary et al. (2017). "The PARSEME Shared Task on Automatic Identification of Verbal Multiword Expressions."
- Ramisch et al. (2018). "Edition 1.1 of the PARSEME Shared Task on Automatic Identification of Verbal Multiword Expressions."

---

### 3.3 Multiword Expressions Dataset for Indian Languages

**Description**: MWE annotation dataset for Hindi and Marathi.

**Size**:
- Hindi: 3,178 compound nouns, 2,556 light verb constructions
- Marathi: 1,003 compound nouns, 2,416 light verb constructions

**Annotations Include**:
- Two MWE types: compound nouns and light verb constructions
- POS-tagged corpus
- IndoWordNet synset mappings

**Download**: Check Papers with Code or contact authors

**Relevance to Research**: ★★☆☆☆
- Cross-linguistic comparison (Indo-European vs Germanic)
- Less directly applicable unless studying multilingual representations

**Key Papers**:
- "Multiword Expressions Dataset for Indian Languages" (2016)
- Paper: https://aclanthology.org/L16-1369.pdf
- Papers with Code: https://paperswithcode.com/paper/multiword-expressions-dataset-for-indian

---

## 4. Idiom and Non-Compositional Expression Datasets

### 4.1 MAGPIE (A Large Corpus of Potentially Idiomatic Expressions)

**Description**: Largest sense-annotated corpus of potentially idiomatic expressions based on British National Corpus.

**Size**:
- 56,622 instances
- 1,756 different idiom types
- All with crowdsourced meaning labels

**Annotations Include**:
- Binary sense labels (literal vs. idiomatic)
- Annotation confidence levels
- POS tags
- Sentence contexts from BNC
- Pointers to full BNC-XML contexts

**Download**:
- **GitHub**: https://github.com/hslh/magpie-corpus
  - Three jsonl files:
    - MAGPIE_unfiltered.jsonl (complete set)
    - MAGPIE_filtered_split_*.jsonl (filtered, 75%+ confidence, binary labels)
- **HuggingFace**: https://huggingface.co/datasets/gsarti/magpie
- **Sketch Engine**: Available as corpus

**License**: Creative Commons 4.0 (CC-BY-4.0)

**Relevance to Research**: ★★★★☆
- Largest idiom dataset available
- High-quality annotations with confidence scores
- Includes literal uses of idiomatic expressions for comparison
- **Application**: Compare compositional compounds like "washing machine" to non-compositional ones from MAGPIE

**Key Papers**:
- Haagsma, Bos, & Nissim (2020). "MAGPIE: A Large Corpus of Potentially Idiomatic Expressions."
- Paper: https://aclanthology.org/2020.lrec-1.35/
- PDF: https://aclanthology.org/2020.lrec-1.35.pdf

---

### 4.2 VNC-Tokens Dataset

**Description**: English verb-noun combination usage dataset with literal vs. idiomatic annotations.

**Size**: 2,984 English verb-noun combination tokens

**Annotations Include**:
- Binary classification: literal or idiomatic
- Pointers to BNC-XML full context
- Sentence-level context

**Download**:
- Originally from Paul Cook's site (may be deprecated)
- Check LREC 2008 MWE workshop proceedings

**Relevance to Research**: ★★☆☆☆
- Focused on VNCs rather than noun compounds
- Smaller scale than MAGPIE
- Historical dataset, may be superseded

**Key Papers**:
- Cook, Fazly, & Stevenson (2008). "The VNC-Tokens Dataset." LREC Workshop on Multiword Expressions

---

### 4.3 EPIE Dataset

**Description**: Corpus for Possible Idiomatic Expressions.

**Size**: Details limited in search results

**Download**: Check arXiv paper at https://arxiv.org/abs/2006.09479

**Relevance to Research**: ★★☆☆☆
- Alternative to MAGPIE
- Less widely adopted

**Key Papers**:
- "EPIE Dataset: A Corpus For Possible Idiomatic Expressions" (2020)

---

### 4.4 MultiCoPIE

**Description**: Multilingual corpus of potentially idiomatic expressions.

**Size**: Cross-lingual coverage

**Languages**: Catalan, Italian, Russian

**Annotations Include**:
- Potentially idiomatic expressions
- Cross-lingual transfer annotations

**Download**: Check MWE 2025 workshop proceedings

**Relevance to Research**: ★★★☆☆
- Multilingual coverage useful for cross-linguistic studies
- Less established than MAGPIE

---

### 4.5 IdiomsResearch Repository

**Description**: GitHub repository aggregating data and links for research on idiomatic expressions.

**Download**: https://github.com/maafiah/IdiomsResearch

**Relevance to Research**: ★★★☆☆
- Meta-resource pointing to multiple datasets
- Good starting point for idiom-related datasets

---

## 5. Datasets for Probing LLM Representations

### 5.1 Frame Representation Hypothesis Dataset (2025)

**Description**: Dataset specifically designed for analyzing multi-token concept representations in LLMs.

**Key Innovation**: Extends beyond single-token analysis to multi-token words

**Relevance to Research**: ★★★★★
- **HIGHLY RELEVANT**: Directly addresses multi-token concepts
- Grounded in Linear Representation Hypothesis
- Published in TACL 2025

**Download**: Check paper supplementary materials

**Key Papers**:
- "Frame Representation Hypothesis: Multi-Token LLM Interpretability and Concept-Guided Text Generation" (2025)
- TACL: https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.48/133800/

**Application**: Could provide methodology for probing "washing machine" as multi-token concept

---

### 5.2 Aljaafari et al. (2024) - Constituent-Aware Pooling Dataset

**Description**: Dataset and methodology from "Interpreting token compositionality in LLMs: A robustness analysis."

**Datasets Used**:
- Inverse Definition Modelling (IDM)
- Synonym Prediction (SP)
- Hypernym Prediction (HP)
- All derived from WordNet

**Relevance to Research**: ★★★★☆
- Methodology (CAP) more important than specific dataset
- Demonstrates how to test compositional processing in LLMs
- **Application**: Adapt CAP methodology for compound noun analysis

**Key Papers**:
- Aljaafari et al. (2024). "Interpreting token compositionality in LLMs: A robustness analysis."
- Corpus ID: 273404240

---

### 5.3 NYTK/HuSST (HuggingFace)

**Description**: Dataset referencing "Recursive Deep Models for Semantic Compositionality Over a Sentiment Treebank."

**Download**: https://huggingface.co/datasets/NYTK/HuSST

**Relevance to Research**: ★★☆☆☆
- Sentiment-focused rather than compound nouns
- May have useful compositionality structure

---

## 6. Cross-Domain and Multimodal Datasets

### 6.1 TIGER-Lab/ImagenWorld

**Description**: Multimodal compositionality benchmark.

**Size**: Six tasks across six domains

**Download**: https://huggingface.co/datasets/TIGER-Lab/ImagenWorld

**Annotations Include**:
- Model compositionality assessment
- Instruction following
- Multimodal reasoning

**Relevance to Research**: ★★★☆☆
- Visual grounding of concepts like "washing machine"
- Tests if visual and textual representations align
- **Application**: Compare text-only vs. vision-language representations

---

## 7. Related Resources Not Directly Dataset-Focused

### 7.1 MWE 2025 Workshop

**Description**: 21st Workshop on Multiword Expressions at NAACL 2025

**URL**: https://multiword.org/mwe2025/

**Proceedings**: https://aclanthology.org/2025.mwe-1.pdf

**Relevance**:
- Source of latest MWE datasets
- "Gathering Compositionality Ratings of Ambiguous Noun-Adjective Multiword Expressions in Galician" (240 ambiguous MWEs)

---

### 7.2 Papers with Code - Noun Datasets

**Description**: Aggregated datasets page for noun-related tasks.

**URL**: https://paperswithcode.com/datasets?q=noun

**Relevance**: Meta-resource for finding additional datasets

---

## Summary Table: Top Datasets by Research Priority

| Rank | Dataset | Size | Type | Availability | Direct Relevance |
|------|---------|------|------|--------------|------------------|
| 1 | **Gagné (2001)** | 300 compounds | Semantic relations | OSF (extension) | ★★★★★ Used in Kitchen Chair paper |
| 2 | **Reddy/Ramisch NC Compositionality** | 550 compounds (multilingual) | Compositionality scores | Freely available | ★★★★★ Gold standard |
| 3 | **Noun Compound Senses (NCS)** | 9,220 sentences | Contextualized compounds | GitHub | ★★★★★ Probing-ready |
| 4 | **Tratz & Hovy** | 19,158 compounds | Semantic relations | Licensed | ★★★★☆ Largest scale |
| 5 | **MAGPIE** | 56,622 instances | Idiomaticity | GitHub, HF | ★★★★☆ Idiom comparison |
| 6 | **PARSEME 2.0** | 19+ languages | All MWE types | GitLab | ★★★★☆ Multilingual, paraphrasing |
| 7 | **DiMSUM** | 16,500 words | Broad MWEs | GitHub | ★★★☆☆ General MWE |
| 8 | **Frame Rep. Hypothesis** | TBD | Multi-token probing | Paper supp. | ★★★★★ LLM-specific |
| 9 | **Aljaafari CAP** | WordNet-derived | Token compositionality | Paper supp. | ★★★★☆ Methodology |
| 10 | **Bamboogle** | TBD | Compositionality gap | HuggingFace | ★★★☆☆ LLM evaluation |

---

## Recommendations for "Washing Machine" Research

### Immediate Next Steps

1. **Download Priority 1 Datasets**:
   - Gagné (2001) dataset (extension from OSF)
   - Reddy/Ramisch NC Compositionality dataset
   - Noun Compound Senses (NCS) from GitHub

2. **Licensing/Access for Priority 2**:
   - Request Tratz & Hovy dataset license
   - Clone MAGPIE from GitHub

3. **Exploration**:
   - Check Frame Representation Hypothesis supplementary materials
   - Explore HuggingFace compositionality datasets

### Dataset Augmentation Strategy

Since "washing machine" specifically may not appear in all datasets, create custom augmentation:

1. **Household Objects Compound Set**:
   - Collect 100-200 household object compounds similar to "washing machine"
   - Examples: "coffee machine", "vacuum cleaner", "microwave oven", "dish washer"
   - Get human compositionality ratings using same scale as Reddy et al.

2. **Substitution Variants** (following Garcia et al.):
   - Synonyms: "laundry machine", "washer", "washing apparatus"
   - Component synonyms: "cleaning machine", "washing device"
   - Related: "dryer machine", "washing station"

3. **Cross-Domain Compounds**:
   - Compare household ("washing machine") vs. abstract ("grey matter") vs. technical ("computer science")
   - Test if domain affects representation structure

### Analysis Pipeline

1. **Human Baselines**:
   - Use Reddy/Ramisch dataset to establish human compositionality judgments
   - Correlate with LLM representations

2. **Probing Methodology**:
   - Apply Garcia et al.'s 4 probing measures (P1-P4)
   - Apply Aljaafari et al.'s CAP methodology
   - Combine approaches for comprehensive analysis

3. **Cross-Linguistic**:
   - Use multilingual datasets (Ramisch, PARSEME) to test universality
   - Compare "washing machine" to "máquina de lavar" (Portuguese), "machine à laver" (French)

4. **Idiomaticity Contrast**:
   - Compare MAGPIE idiomatic expressions to compositional compounds
   - Test if LLMs represent "washing machine" (compositional) differently from "piece of cake" (idiomatic)

---

## Key Gaps and Future Dataset Needs

### Identified Gaps

1. **Few LLM-Specific Probing Datasets**: Most datasets designed for traditional NLP, not mechanistic interpretability

2. **Limited Household Object Coverage**: Domain-specific compound collections lacking

3. **Sparse Multi-Token Analysis**: Frame Rep. Hypothesis (2025) is pioneering but rare

4. **Limited Layer-by-Layer Analysis**: Few datasets designed for internal representation analysis

5. **No Standard "Washing Machine" Benchmark**: Need standardized test set of household/common compounds

### Proposed New Datasets

1. **HouseHold Compounds 1000 (HHC-1000)**:
   - 1,000 common household object compounds
   - Compositionality ratings
   - Frequency information
   - Cross-linguistic translations
   - Example sentences in multiple contexts

2. **LLM Compound Probing Benchmark (LCPB)**:
   - Designed specifically for layer-wise probing
   - Includes:
     - Compounds with controlled tokenization patterns
     - Matched compositional vs. idiomatic pairs
     - Gradient of compositionality (not binary)
     - Multiple semantic relation types

3. **Multi-Token Concept Representations (MTCR)**:
   - Extension of Frame Rep. Hypothesis
   - Focused on 2-4 token concepts
   - Includes compounds, idioms, technical terms
   - Annotations for expected integration points in transformer layers

---

## Contact Information for Dataset Access

| Dataset | Primary Contact/Source | Alternative Access |
|---------|----------------------|-------------------|
| Tratz & Hovy | http://www.isi.edu/publications/licensed-sw/ | Email ISI licensing |
| Gagné Extension | https://osf.io/gvc2w/ | Open Science Framework |
| Ramisch Compositionality | http://pageperso.lis-lab.fr/carlos.ramisch/ | Direct download |
| NCS | https://github.com/marcospln/noun_compound_senses | GitHub clone |
| MAGPIE | https://github.com/hslh/magpie-corpus | HuggingFace mirror |
| DiMSUM | https://github.com/dimsum16/dimsum-data | CMU Lex Semantics |
| PARSEME 2.0 | https://gitlab.com/parseme/sharedtask-data | GitLab clone |

---

## Citation Tracker

### Most Cited Papers Referenced

1. **Tratz & Hovy (2010)**: Foundational taxonomy, widely cited baseline
2. **Reddy et al. (2011)**: Gold standard for compositionality, 100+ citations
3. **Ramisch et al. (2019)**: Multilingual extension, recent standard
4. **Gagné & Shoben (2001)**: Psycholinguistic foundation, cognitive science basis
5. **Ormerod et al. (2024)**: Most directly relevant to LLM probing

### Recent (2024-2026) Papers

- Frame Representation Hypothesis (2025) - TACL
- Aljaafari et al. (2024) - Token compositionality robustness
- MWE 2025 Workshop - Latest MWE research trends
- PARSEME 2.0 (2025-2026) - Ongoing shared task

---

## Sources

### HuggingFace and Dataset Platforms
- [HuggingFace Datasets Hub](https://huggingface.co/datasets)
- [Papers with Code - Noun-Noun Compound Dataset](https://paperswithcode.com/dataset/noun-ainoun-compound-dataset)
- [Papers with Code - Multiword Expressions Dataset](https://paperswithcode.com/paper/multiword-expressions-dataset-for-indian)
- [mehdidc/compositionality_hpsv1](https://huggingface.co/datasets/mehdidc/compositionality_hpsv1)
- [mehdidc/compositionality](https://huggingface.co/datasets/mehdidc/compositionality)
- [chiayewken/bamboogle](https://huggingface.co/datasets/chiayewken/bamboogle)
- [gsarti/magpie](https://huggingface.co/datasets/gsarti/magpie)
- [TIGER-Lab/ImagenWorld](https://huggingface.co/datasets/TIGER-Lab/ImagenWorld)
- [NYTK/HuSST](https://huggingface.co/datasets/NYTK/HuSST)

### Academic Papers and Resources
- [Tratz & Hovy (2010) - ACL Anthology](https://aclanthology.org/P10-1070/)
- [Tratz & Hovy (2010) - PDF](https://www.cs.cmu.edu/~hovy/papers/10ACL-nounnoun-compound-rels.pdf)
- [Gagné Dataset Extension - Journal of Open Psychology Data](https://openpsychologydata.metajnl.com/articles/10.5334/jopd.93)
- [Ormerod et al. (2024) - Computational Linguistics](https://direct.mit.edu/coli/article/50/1/49/118133/)
- [Reddy et al. (2011) - IJCNLP](https://aclanthology.org/I11-1024/)
- [Ramisch et al. (2019) - Computational Linguistics](https://direct.mit.edu/coli/article/45/1/1/1621/)
- [Frame Representation Hypothesis (2025) - TACL](https://direct.mit.edu/tacl/article/doi/10.1162/TACL.a.48/133800/)
- [Garcia et al. (2021) - EACL](https://aclanthology.org/2021.eacl-main.201/)

### Dataset Repositories
- [Carlos Ramisch's Dataset Page](https://pageperso.lis-lab.fr/carlos.ramisch/?page=downloads/compounds)
- [Noun Compound Senses GitHub](https://github.com/marcospln/noun_compound_senses)
- [IIT Bombay NC Datasets](https://www.cse.iitb.ac.in/~girishp/nc-dataset/)
- [MAGPIE Corpus GitHub](https://github.com/hslh/magpie-corpus)
- [DiMSUM Data GitHub](https://github.com/dimsum16/dimsum-data)
- [DiMSUM Official Site](http://dimsum16.github.io/)
- [PARSEME Shared Task Data](https://gitlab.com/parseme/sharedtask-data/-/tree/master/2.0)
- [Tratz & Hovy Dataset](http://www.isi.edu/publications/licensed-sw/)
- [Open Science Framework - Gagné Extension](https://osf.io/gvc2w/)
- [IdiomsResearch Repository](https://github.com/maafiah/IdiomsResearch)

### Workshop and Conference Proceedings
- [MWE 2025 Workshop](https://multiword.org/mwe2025/)
- [MWE 2025 Proceedings](https://aclanthology.org/2025.mwe-1.pdf)
- [PARSEME Shared Task Results](https://typo.uni-konstanz.de/parseme/index.php/results/shared-task)
- [Fares (2016) - ACL Student Workshop](https://aclanthology.org/P16-3011/)

### Semantic Scholar and ArXiv
- [Semantics of Multiword Expressions Survey - MIT Press](https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00657/120831/)
- [Predicting Compositionality of MWEs - Semantic Scholar](https://www.semanticscholar.org/paper/Predicting-the-Compositionality-of-Multiword-Using-Salehi-Cook/6bf6563c30ef2df9a0ddbe0c794885e8b57c7ce5)
- [Hermann Compositionality Paper](http://www.cs.ox.ac.uk/files/4792/Hermann:StarSEM:12b.pdf)
- [VNC-Tokens Dataset - Semantic Scholar](https://www.semanticscholar.org/paper/The-VNC-Tokens-Dataset-Cook-Fazly/b4402efcbdaceb282d73c59a01b81a9de66ca3e3)
- [EPIE Dataset - ArXiv](https://arxiv.org/abs/2006.09479)
- [Irish Noun Compounds - ArXiv](https://arxiv.org/html/2502.10061)

### Additional Resources
- [CMU Lexical Semantics Resources](https://www.cs.cmu.edu/~ark/LexSem/)
- [Sketch Engine - MAGPIE Corpus](https://www.sketchengine.eu/magpie-sense-annotated-corpus/)
- [ResearchGate - Gagné Dataset](https://www.researchgate.net/publication/373500551_A_Dataset_of_108_Novel_Noun-Noun_Compound_Words_with_Active_and_Passive_Interpretation)
- [Zenodo - PARSEME Analysis](https://zenodo.org/records/1469557)

---

## Document Metadata

- **Compiled by**: Dataset search for LLM washing machine storage research
- **Date**: February 7, 2026
- **Version**: 1.0
- **Total Datasets Catalogued**: 15+ major datasets plus numerous sub-collections
- **Primary Focus**: Compound nouns, multiword expressions, compositionality, and idiomaticity
- **Geographic Coverage**: Primarily English, with significant multilingual resources (French, Portuguese, Spanish, Hindi, Marathi, Catalan, Italian, Russian, and 19+ languages in PARSEME)
- **Temporal Coverage**: Datasets from 2001-2026 (25 years of research)
- **Next Update**: After accessing and evaluating Priority 1 datasets
