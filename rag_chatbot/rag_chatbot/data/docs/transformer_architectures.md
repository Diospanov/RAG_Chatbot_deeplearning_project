# Transformer Architectures: BERT and GPT-2

## The Transformer Foundation

Both BERT and GPT-2 are based on the Transformer architecture introduced by Vaswani et al. in "Attention Is All You Need" (2017). The core building block of a Transformer is the self-attention mechanism, which allows every position in a sequence to attend to every other position in a single operation. This replaces the sequential processing of RNNs and enables massive parallelization during training.

A standard Transformer consists of an encoder, a decoder, or both. BERT uses only the encoder stack. GPT-2 uses only the decoder stack. Many later models (T5, BART) use both.

## BERT: Bidirectional Encoder Representations from Transformers

BERT was introduced by Devlin et al. at Google in 2018. It is an encoder-only Transformer pretrained on large text corpora using two self-supervised objectives.

### Architecture

BERT uses stacked transformer encoder blocks. The base variant has 12 layers, 768 hidden dimensions, 12 attention heads, and 110 million parameters. The large variant has 24 layers, 1024 hidden dimensions, 16 attention heads, and 340 million parameters.

The critical design choice is bidirectional self-attention. In each encoder layer, every token attends to every other token in the sequence simultaneously. There is no masking of future positions. This allows BERT to build deeply contextual representations where the representation of each token is influenced by its full context — both the words to its left and right.

### Pre-training Tasks

**Masked Language Modeling (MLM)**: During pretraining, 15% of input tokens are randomly replaced with a [MASK] token. The model is trained to predict the original token at each masked position. Because the model sees context on both sides of the mask, it is forced to develop rich bidirectional representations. This is the primary source of BERT's representational power.

**Next Sentence Prediction (NSP)**: The model is given two sentences and trained to predict whether the second sentence follows the first in the original document. This objective helps BERT understand inter-sentence relationships. However, subsequent research showed NSP provides limited benefit, and models like RoBERTa removed it.

### Strengths and Weaknesses

BERT's bidirectional attention makes it excellent at understanding tasks: question answering, named entity recognition, text classification, and semantic similarity. The same word gets a different embedding depending on its context, which is crucial for disambiguation.

However, BERT cannot generate text autoregressively. To generate token t_n, a model must condition on t_1 through t_{n-1}, but BERT sees the full sequence simultaneously. Masked generation is possible but awkward and not competitive with decoder-based models.

### Connection to Sentence-Transformers

The `sentence-transformers` library fine-tunes BERT variants using contrastive learning objectives that optimize for semantic similarity. The model `all-MiniLM-L6-v2` is a 6-layer, 384-dimensional BERT variant trained on over 1 billion sentence pairs. It produces embeddings where semantically similar sentences are close in cosine distance — making it ideal for RAG retrieval.

## GPT-2: Generative Pre-trained Transformer 2

GPT-2 was introduced by Radford et al. at OpenAI in 2019. It demonstrated that large-scale autoregressive language modeling produces models capable of coherent long-form text generation.

### Architecture

GPT-2 uses stacked transformer decoder blocks with causal (unidirectional) self-attention. The small variant has 12 layers, 768 hidden dimensions, 12 attention heads, and 117 million parameters. Larger variants (medium, large, XL) scale up to 1.5 billion parameters.

The defining architectural choice is the causal attention mask — a lower-triangular matrix that prevents each position from attending to future positions. Token t_i can attend to t_1, t_2, …, t_i but not to t_{i+1}, t_{i+2}, and so on. This enforces the autoregressive property.

### Pre-training Task

GPT-2 is trained on a single objective: next-token prediction. Given a sequence of tokens, the model is trained to maximize the probability of each token given all previous tokens:

    P(t_1, t_2, ..., t_n) = ∏ P(t_i | t_1, ..., t_{i-1})

This is equivalent to minimizing cross-entropy loss on the token prediction at each position. The training signal comes entirely from predicting the natural continuation of text — no manual labels are required.

### Strengths and Weaknesses

GPT-2's autoregressive design makes it a natural text generator. At inference time, the model generates one token at a time, appending each generated token to the context before predicting the next. This enables open-ended generation of arbitrary length.

However, GPT-2's causal masking limits its representational power for understanding tasks. When processing token t_i, it cannot access the context of tokens t_{i+1}, t_{i+2}, …. This asymmetry makes it weaker than BERT for tasks requiring full bidirectional context.

### Scaling

GPT-2 established that decoder-only language model performance scales predictably with model size, dataset size, and compute. This insight led directly to GPT-3 (175B parameters), GPT-4, Claude, and Llama — all of which use the same decoder-only transformer architecture as GPT-2, simply scaled much larger.

## Comparison Table

| Property | BERT | GPT-2 |
|---|---|---|
| Architecture type | Encoder-only | Decoder-only |
| Attention pattern | Bidirectional (full) | Causal (unidirectional) |
| Primary pre-training task | MLM + NSP | Next-token prediction |
| Output | Contextual token embeddings | Token probability distribution |
| Natural use case | Understanding, retrieval | Generation |
| Can generate autoregressively? | No | Yes |
| Role in RAG | Retriever (via sentence-transformers) | Generator |

## Why Each Architecture Fits Its Role in RAG

In a RAG pipeline, BERT-style models are used for encoding and retrieval, while GPT-style models are used for generation. This division is not arbitrary — it follows directly from the architectural properties.

**BERT for retrieval**: Dense retrieval requires encoding both documents and queries into a shared vector space where semantic similarity correlates with cosine distance. BERT's bidirectional attention, by giving each token full context, produces the richest possible sentence-level representations. Fine-tuned with contrastive objectives, BERT variants are the state-of-the-art approach for dense passage retrieval.

**GPT for generation**: Answering a question in natural language requires producing a coherent sequence of tokens, each conditioned on all previous tokens. The autoregressive, left-to-right generation process is exactly what the causal decoder architecture was designed for. The same architecture, scaled to billions of parameters, now powers the most capable language models available.

Attempting to swap the roles — using GPT-2 for retrieval or BERT for generation — would yield substantially worse results in both cases. The architectural choices are closely matched to the task requirements.
