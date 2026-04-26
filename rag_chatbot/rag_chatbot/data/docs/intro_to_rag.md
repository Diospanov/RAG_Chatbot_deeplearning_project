# Introduction to Retrieval-Augmented Generation (RAG)

## What is RAG?

Retrieval-Augmented Generation (RAG) is the dominant architecture for production LLM applications. It solves the fundamental problem of LLMs: they are static snapshots of training data and have no access to private, recent, or domain-specific information. RAG fixes this by retrieving relevant document passages at query time and injecting them into the LLM's context as grounding evidence.

A RAG system combines two distinct capabilities: a retriever that finds relevant documents, and a generator that produces an answer conditioned on those documents. The retriever is typically a dense vector search system powered by sentence embeddings, while the generator is a large language model such as Claude, GPT-4, or Llama.

## Why RAG?

Large language models are trained on fixed datasets with a knowledge cutoff. After training, they have no way to access new information, private documents, or enterprise knowledge bases. RAG addresses this limitation without requiring expensive model fine-tuning.

The key advantages of RAG are:
- **Grounding**: Answers are derived from retrieved evidence, reducing hallucination
- **Attribution**: Sources can be cited, making answers verifiable
- **Updatability**: The knowledge base can be updated independently of the model
- **Privacy**: Private documents never enter the model's training data

## The Five Components of a RAG System

### 1. Document Ingestion

The ingestion pipeline loads raw documents from various file formats. Supported formats typically include PDF, HTML, plain text, Markdown, and DOCX. For each document, metadata is extracted and stored, including the source filename or URL, document title, and date. This metadata must be carried through the entire pipeline and surfaced in responses so users can verify the source of information.

### 2. Chunking and Preprocessing

Because LLM context windows are limited, documents must be split into smaller passages called chunks before embedding. Two common strategies exist:

**Fixed-size chunking** divides text into windows of N tokens with some overlap between consecutive windows. It is simple and predictable but often splits sentences at arbitrary positions.

**Sentence-aware chunking** respects sentence boundaries, accumulating sentences until a target size is reached. This preserves semantic units and consistently outperforms fixed-size chunking on retrieval benchmarks.

Typical chunk sizes range from 100 to 512 tokens, with an overlap of 10 to 25 percent of the chunk size. Overlap prevents information at chunk boundaries from being lost.

### 3. Embedding and Vector Storage

Each chunk is encoded into a dense vector representation using a pretrained embedding model. The most widely used models are fine-tuned BERT variants from the `sentence-transformers` library, such as `all-MiniLM-L6-v2`, which produces 384-dimensional vectors and runs efficiently on CPU.

These vectors are stored in a vector database alongside the chunk text and metadata. Popular vector databases include ChromaDB, FAISS, Qdrant, and Pinecone. The database indexes vectors for approximate nearest neighbor (ANN) search, enabling sub-millisecond retrieval over millions of chunks.

### 4. Retrieval

At query time, the user's question is embedded using the same model used to encode the documents. The resulting query vector is compared against all stored chunk vectors using cosine similarity. The top-k most similar chunks (typically k=5) are returned as the retrieved context.

Cosine similarity measures the angle between two vectors, making it robust to differences in text length. Chunks from multiple different source documents may be retrieved for a single query, which is expected and desirable.

### 5. Generation and Grounding

The retrieved chunks are assembled into a context block and injected into the LLM's system prompt along with grounding instructions. These instructions direct the LLM to:
1. Answer only using the provided context passages
2. Cite the source document by name for every factual claim
3. Respond with "I cannot find this in the provided documents" when the answer is absent

This grounding prompt is what separates a RAG system from a simple LLM chatbot. Without it, the LLM would freely mix retrieved evidence with its pretrained knowledge, making citation impossible and hallucination likely.

## Evaluation

RAG systems are evaluated on both retrieval and generation dimensions.

**Retrieval metrics** measure whether relevant passages were found:
- **Precision@k**: Fraction of top-k retrieved chunks that contain the answer
- **Recall@k**: Fraction of all relevant passages that appeared in the top-k results

**Generation metrics** measure answer quality:
- **Faithfulness**: Fraction of answer sentences that are supported by the retrieved context
- **Answer Relevance**: Semantic similarity between the generated answer and the ground-truth answer
- **Citation Rate**: Fraction of answers that include a source citation

The RAGAS framework provides automated implementations of these metrics using an LLM as a judge.

## Limitations

RAG is not a perfect solution. Common failure modes include:

**Chunking boundary failures**: When a key fact spans two adjacent chunks, neither chunk alone may be sufficiently informative for retrieval to succeed.

**Vocabulary mismatch**: Dense retrieval works by semantic similarity in embedding space. Highly technical queries may not match document embeddings if the embedding model was trained on different vocabulary.

**Multi-hop reasoning**: Questions that require combining information from multiple non-adjacent passages are difficult for standard RAG because a single retrieval step may not surface all necessary context.

**Context window overflow**: With large top-k or verbose chunks, the retrieved context can approach the LLM's token limit, forcing truncation of some passages.

Despite these limitations, RAG remains the most practical architecture for building production LLM applications over custom knowledge bases.
