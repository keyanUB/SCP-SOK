# Evaluation Metrics for Secure Coding Practices Understanding

### BLEU (Bilingual Evaluation Understudy)
Measures how many n-grams (word sequences) from the LLM's response match the reference explanation. Originally designed for machine translation, it focuses on precision - rewarding outputs that use the same phrases as the reference. Higher scores indicate more word-level overlap, but BLEU can miss semantically similar responses that use different wording.
**Range:** 0 to 1 (often reported as 0-100)
**Interpretation:**
    - Poor: < 0.3
    - Fair: 0.3 - 0.5
    - Good: 0.5 - 0.7
    - Excellent: > 0.7


### ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
Identifies the longest sequence of words that appear in the same order in both the LLM output and reference text, though not necessarily consecutively. Unlike BLEU which focuses on precision, ROUGE-L emphasizes recall - measuring how much of the reference content is captured. It's particularly good at detecting whether key information is preserved, even if expressed differently.
**Range:** 0 to 1 (reports Precision, Recall, and F1)
**Interpretation (F1 score):**
    - Poor: < 0.2
    - Fair: 0.2 - 0.4
    - Good: 0.4 - 0.6
    - Very Good: > 0.6

### METEOR (Metric for Evaluation of Translation with Explicit ORdering)
An enhanced metric that goes beyond exact word matching by considering synonyms, word stems, and paraphrases. For example, it recognizes that "validate" and "validation" are related, or that "user input" and "user data" convey similar meanings. METEOR provides a more nuanced evaluation than BLEU by accounting for linguistic variations while maintaining word order importance.
**Range:** 0 to 1
**Interpretation:**
    - Poor: < 0.2
    - Fair: 0.2 - 0.4
    - Good: 0.4 - 0.6
    - Very Good: > 0.6

### BERTScore
Uses deep learning (BERT embeddings) to measure semantic similarity between texts at the word and phrase level. Instead of requiring exact word matches, it understands contextual meaning - recognizing that "prevent SQL injection" and "protect against SQL injection attacks" are semantically equivalent. BERTScore captures the meaning behind the words, making it more robust to paraphrasing than traditional metrics.
**Range:** Approximately 0 to 1 (reports Precision, Recall, and F1)
**Interpretation (F1 score):**
    - Poor: < 0.7
    - Fair: 0.7 - 0.8
    - Good: 0.8 - 0.9
    - Very Good: > 0.9

### SBERT Cosine Similarity (Sentence-BERT Cosine Similarity)
Encodes entire responses into dense vector representations (embeddings) and compares them using cosine similarity. This provides a holistic measure of semantic similarity between the LLM's complete explanation and the reference. It's excellent for capturing overall meaning and conceptual alignment, even when the responses are structured or worded quite differently.
**Range:** -1 to 1 (typically 0 to 1 for sentence embeddings)
**Interpretation:**
    - Poor: < 0.5
    - Fair: 0.5 - 0.7
    - Good: 0.7 - 0.9
    - Excellent: > 0.9

### Semantic Key Concept Coverage (Based on SBERT)
Identifies whether specific key concepts from the reference appear in the LLM's output, even when paraphrased. Unlike overall semantic similarity which compares entire texts, this metric checks if each individual critical concept (e.g., "validate input length," "escape special characters") is present somewhere in the explanation. It uses SBERT embeddings and cosine similarity to match concepts semantically - recognizing that "sanitize user input" and "validate user data" convey similar ideas. This ensures that essential security principles aren't omitted from the LLM's explanation.
**Range:** 0 to 1 (fraction of key concepts covered)
**Interpretation:**
Poor: < 0.4 (missing most key concepts)
Fair: 0.4 - 0.6 (covers some concepts)
Good: 0.6 - 0.8 (covers most concepts)
Excellent: > 0.8 (comprehensive coverage)


## References
- Papineni, K., et al. (2002). BLEU: a method for automatic evaluation of machine translation.
- Lin, C. Y. (2004). ROUGE: A package for automatic evaluation of summaries.
- Banerjee, S., & Lavie, A. (2005). METEOR: An automatic metric for MT evaluation.
- Zhang, T., et al. (2019). BERTScore: Evaluating text generation with BERT.
- Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks.
