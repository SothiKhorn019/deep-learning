**Lemmatizer NoAttn**  
- **What it does:**  
  - Implements a vanilla sequence-to-sequence character-level lemmatizer.  
  - Encodes each source word with a bidirectional GRU (summing its forward/backward final states).  
  - Decodes one character at a time with another GRU, using “teacher forcing” at train time (feeding in the gold previous character) and greedy sampling at inference.  
  - Optionally ties the decoder’s input embeddings to its output‐projection weights.  

- **Key goals:**  
  1. Practice building a basic encoder–decoder pipeline in PyTorch.  
  2. Understand how to handle variable-length character sequences (packing/padding).  
  3. Learn to implement training-time vs. inference-time decoding.  
  4. Explore the effect of tying embeddings on model size and performance.  

---

**Lemmatizer Attn**  
- **What it does:**  
  - Extends the NoAttn model by (a) returning a hidden state for *every* character in the source word and (b) wrapping the decoder GRUCell in a Bahdanau‐style attention mechanism.  
  - At each decoding step, projects both the entire encoder output sequence and the current decoder state into a shared “attention space,” computes alignment scores, derives a context vector, and concatenates it with the decoder’s input embedding.  
  - Proceeds with greedy decoding as before, but now each step attends back over all source‐character representations.  

- **Key goals:**  
  1. See how attention lets the decoder dynamically focus on different parts of the input.  
  2. Reinforce the mechanics of additive (Bahdanau) attention: projection, scoring, softmax, context aggregation.  
  3. Compare performance (accuracy, convergence speed) against the no-attention baseline.  
  4. Experiment with varying RNN dimensionality to observe its impact on both models.