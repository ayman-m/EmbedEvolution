# EmbedEvolution: A Journey Through Text Embedding Techniques

Embark on an educational journey through the evolution of word and sentence embeddings, from early sequential models like RNNs to modern instruction-aware architectures. This repository provides hands-on Jupyter notebooks with code snippets, visualizations, and concise summaries for each key stage. Our goal is to demonstrate the strengths, limitations, and core concepts of these models in capturing semantic relationships and context. Perfect for NLP enthusiasts, students, and learners!

## 🚀 About This Repository

The field of Natural Language Processing (NLP) has been revolutionized by the ability to represent text in a way that machines can understand and learn from. Text embeddings – dense vector representations of words, sentences, or documents – are at the heart of this revolution. This repository, "EmbedEvolution," is designed to walk you through the major milestones in the development of these embedding techniques.

Each stage is presented as a self-contained Jupyter notebook that:
* Explains the **intuition and core logic** behind the model.
* Provides **Python code examples** (primarily using common libraries like TensorFlow/Keras, Gensim, and Hugging Face Transformers/Sentence-Transformers).
* Includes **practical tests** such as semantic similarity comparisons or text generation/autocompletion.
* Visualizes embeddings using **PCA plots**.
* Summarizes the **strengths, weaknesses, and example applications** of the technique, often leading into why the next stage of evolution was necessary.

## ✨ Stages of Evolution Explored

The repository currently covers the following key embedding methodologies, each in its dedicated notebook:

1.  **Stage 1: Recurrent Neural Networks (RNNs)**
    * **Focus:** Early attempts to capture sequential context.
    * **Notebook Demonstrates:** How RNNs process sequences, generate basic contextual hidden states, and perform tasks like next-word prediction (used for autocompletion). Limitations regarding long-range dependencies and coherence are explored.

2.  **Stage 2: Word2Vec**
    * **Focus:** Efficient learning of static word embeddings from local context windows (CBOW & Skip-gram).
    * **Notebook Demonstrates:** Training Word2Vec, exploring word similarities, analogies, and visualizing the static nature of these word vectors.

3.  **Stage 3: GloVe (Global Vectors for Word Representation)**
    * **Focus:** Learning static word embeddings by leveraging global word-word co-occurrence statistics.
    * **Notebook Demonstrates:** Using pre-trained GloVe vectors, examining word similarities and analogies, and comparing its global approach to Word2Vec's local context.

4.  **Stage 4: BERT (Bidirectional Encoder Representations from Transformers)**
    * **Focus:** Deeply contextual *word/token* embeddings using the Transformer architecture.
    * **Notebook Demonstrates:** How BERT generates different embeddings for the same word in different contexts, showcasing its bidirectional understanding and the power of attention mechanisms.

5.  **Stage 5: Sentence Transformers (SBERT)**
    * **Focus:** Fine-tuning Transformer models (like BERT) to produce semantically meaningful *sentence embeddings* suitable for direct similarity comparison.
    * **Notebook Demonstrates:** Generating fixed-size sentence vectors and using cosine similarity to find semantically similar sentences, an improvement over naive pooling of BERT token outputs for this specific task.

6.  **Stage 6: GTE (General Text Embeddings - e.g., BGE, E5 models)**
    * **Focus:** State-of-the-art general-purpose text embedding models designed for robust performance across a wide range of tasks.
    * **Notebook Demonstrates:** Using a GTE model (like `BAAI/bge-base-en-v1.5`) to generate high-quality sentence/text embeddings and comparing their semantic similarity capabilities.

7.  **Stage 7: Instruction-Aware Embeddings (e.g., Nomic Embed, Instructor)**
    * **Focus:** Guiding embedding generation with task-specific instructions or prefixes to produce more tailored and effective representations.
    * **Notebook Demonstrates:** Using a model like `nomic-ai/nomic-embed-text-v1.5` to show how different task prefixes for the same input text result in different embeddings, optimized for specific downstream uses like search, clustering, or classification.

## 🔧 Inside Each Notebook

Each notebook typically follows this structure:
* **Introduction** to the specific embedding technique and its place in the evolution.
* **Setup and Imports** of necessary libraries.
* **Model Loading/Definition** and training (where applicable, or loading pre-trained models).
* **Corpus Definition** (using consistent example text like snippets from "Alice in Wonderland" for relatable comparisons).
* **Embedding Generation** for words or sentences.
* **Analysis:**
    * Cosine similarity tests.
    * Text autocompletion/generation (for sequence models like RNNs).
    * PCA visualization of the embedding space.
* **Discussion** of results, strengths, and limitations of the technique.
* **Summary Text** often included directly on plots for quick takeaways.

## 🛠️ Getting Started

1.  **Clone the Repository:**
    ```bash
    git clone [https://github.com/ayman-m/EmbedEvolution.git](https://github.com/ayman-m/EmbedEvolution.git)
    cd EmbedEvolution
    ```
2.  **Set up a Virtual Environment (Recommended):**
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Windows: .\.venv\Scripts\activate
    ```
3.  **Install Requirements:**
    Each notebook may have slightly different dependencies. A general `requirements.txt` might be provided, or check the setup cells within each notebook. Common libraries include:
    ```bash
    pip install tensorflow scikit-learn matplotlib gensim sentence-transformers transformers torch
    # You might need to install specific versions or extras depending on your setup (e.g., for GPU)
    ```
    *(Note: Some notebooks might download pre-trained models on their first run, requiring an internet connection.)*

4.  **Run Jupyter Notebooks:**
    Launch Jupyter Lab or Jupyter Notebook and navigate to the desired notebook file (`.ipynb`).
    ```bash
    jupyter lab
    # or
    jupyter notebook
    ```

## �� Contributing

Contributions, issues, and feature requests are welcome! Feel free to check [issues page](https://github.com/ayman-m/EmbedEvolution/issues) if you want to contribute. Please adhere to this project's `CODE_OF_CONDUCT.md` (if you create one).

## 📜 License

This project is licensed under the MIT License - see the `LICENSE.md` file for details (if you create one).

