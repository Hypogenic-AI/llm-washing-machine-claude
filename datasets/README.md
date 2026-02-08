# Datasets for Compound Concept Representation in LLMs

This directory contains datasets for studying how large language models represent and process compound concepts, with a focus on compositionality and idiomaticity.

## Downloaded Datasets

### 1. Noun Compound Senses (NCS) Dataset
**Location:** `noun_compound_senses/`
**Source:** https://github.com/marcospln/noun_compound_senses
**Citation:** Garcia et al. (2021), EACL

**Description:**
- 280 English noun compounds with varying degrees of idiomaticity
- 180 Portuguese noun compounds
- Based on Reddy et al. (2011) compositionality dataset
- Contains 5,620 test items for English
- Three sentence variants (P1, P2, P3) for each compound:
  - P1: Compound replaced by synonym
  - P2: Compound split into head and dependent
  - P3: Each component replaced by synonym

**Data Format:**
CSV files in `dataset/en/` and `dataset/pt/` directories with columns:
- `compound`: The noun compound (e.g., "washing machine")
- `compositionality`: Compositionality score
- `sentence1`, `sentence2`, `sentence3`: Naturalistic corpus sentences
- Additional P1, P2, P3 variant files in `neutral/` subdirectories

**Files:**
- `dataset/en/neutral/P1_sents.csv`: English neutral contexts with synonym replacements
- `dataset/en/neutral/P2_sents.csv`: English neutral contexts with component splitting
- `dataset/en/neutral/P3_sents.csv`: English neutral contexts with component synonyms
- `input/sentids_en.csv`: Contains compounds with compositionality ratings (derived from Reddy et al. 2011)

### 2. Reddy et al. (2011) Compositionality Ratings
**Location:** `noun_compound_senses/input/sentids_en.csv`
**Source:** Included in NCS dataset
**Citation:** Reddy, McCarthy & Manandhar (2011), IJCNLP

**Description:**
- 90 English noun-noun compounds with human compositionality ratings
- Compositionality scores based on human judgments
- Standard benchmark for compositionality research

**Data Format:**
CSV with columns:
- `compound`: Noun-noun compound
- `compositionality`: Numerical compositionality score
- `sentence1`, `sentence2`, `sentence3`: Example sentences from ukWaC corpus

### 3. MAGPIE Idiom Dataset
**Location:** `magpie/`
**Source:** https://github.com/hslh/magpie-corpus
**Citation:** Haagsma et al. (2020), LREC

**Description:**
- Large sense-annotated corpus of potentially idiomatic expressions (PIEs)
- Based on British National Corpus (BNC)
- 56,622 instances covering 1,756 different idiom types
- All instances have crowdsourced meaning labels
- Includes both idiomatic and literal uses

**Data Format:**
JSON Lines format with fields:
- `id`: Instance identifier
- `idiom`: The potentially idiomatic expression
- `context`: List of surrounding sentences
- `label`: Sense label (i=idiomatic, l=literal, f=figurative, o=other, ?=unknown)
- `confidence`: Annotation confidence (0.0-1.0)
- `label_distribution`: Distribution of annotator judgments
- `offsets`: Character offsets in sentence
- `split`: train/dev/test split
- `genre`: BNC genre
- `document_id`: Source document

**Files:**
- `MAGPIE_unfiltered.jsonl`: Complete dataset with all annotations
- `MAGPIE_filtered_split_random.jsonl`: Filtered (75% confidence, binary labels), random split
- `MAGPIE_filtered_split_typebased.jsonl`: Filtered, type-based split (no idiom overlap between splits)

### 4. Custom Compound Nouns Test Set
**Location:** `compound_nouns_test.jsonl`
**Source:** Hand-curated for this project

**Description:**
- 35 diverse compound nouns varying in compositionality
- Includes fully compositional, partially compositional, and non-compositional (idiomatic) examples
- Designed to test LLM representations across the compositionality spectrum

**Categories:**
- Fully compositional (rating 5): "coffee table", "brick house", "mountain cabin", etc.
- Partially compositional (rating 3-4): "washing machine", "swimming pool", "parking lot", etc.
- Weakly compositional (rating 2): "shooting star", "guinea pig", "strawberry", etc.
- Non-compositional/idiomatic (rating 1): "hot dog", "butterfly", "deadline", etc.

**Data Format:**
JSON Lines with fields:
- `compound`: The compound noun
- `word1`: First component
- `word2`: Second component
- `compositionality_rating`: 1-5 scale (1=non-compositional, 5=fully compositional)
- `notes`: Explanation of compositionality rating

## Usage Notes

### NCS Dataset
To obtain the full naturalistic sentences from ukWaC corpus:
1. Download ukWaC corpus (requires registration)
2. Run: `python3 get_sentences.py --lang en --corpus UKWAC_full.xml`
3. This creates `original_sents.csv` in `dataset/en/naturalistic/`

### MAGPIE Dataset
For most experiments, use the filtered versions:
- `MAGPIE_filtered_split_random.jsonl`: For general training/testing
- `MAGPIE_filtered_split_typebased.jsonl`: For evaluating generalization to new idiom types

### Custom Test Set
Can be loaded with:
```python
import json
data = [json.loads(line) for line in open('compound_nouns_test.jsonl')]
```

## References

1. Garcia, M., Vieira, T. K., Scarton, C., Idiart, M., & Villavicencio, A. (2021). Probing for idiomaticity in vector space models. In Proceedings of EACL 2021.

2. Reddy, S., McCarthy, D., & Manandhar, S. (2011). An empirical study on compositionality in compound nouns. In Proceedings of IJCNLP 2011, pages 210-218.

3. Haagsma, H., Bos, J., & Nissim, M. (2020). MAGPIE: A Large Corpus of Potentially Idiomatic Expressions. In Proceedings of LREC 2020.

4. Cordeiro, S., Villavicencio, A., Idiart, M., & Ramisch, C. (2019). Unsupervised compositionality prediction of nominal compounds. Computational Linguistics, 45(1):1-57.

## Dataset Statistics Summary

| Dataset | Size | Language | Annotation Type | Source |
|---------|------|----------|----------------|--------|
| NCS | 280 compounds (EN) | English, Portuguese | Compositionality + Variants | ukWaC, brWaC |
| Reddy et al. | 90 compounds | English | Compositionality ratings | Human judgments |
| MAGPIE (filtered) | ~50K instances | English | Idiomaticity (binary) | BNC |
| MAGPIE (full) | 56,622 instances | English | Idiomaticity (5-way) | BNC |
| Custom Test | 35 compounds | English | Compositionality ratings | Hand-curated |

## License Information

- **NCS Dataset**: Check repository for license details
- **MAGPIE**: CC BY 4.0 (see LICENSE file in magpie/)
- **Custom Test Set**: Created for this project, free to use

## Contact

For questions about dataset usage in this project, refer to the main project README.
