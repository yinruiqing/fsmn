# fsmn
Feedforward Sequential Memory Networks

PyTorch implementations for FSMN (Feedforward Sequential Memory Networks), cFSMN (Compact FSMN), DFSMN (Deep-FSMN).
Still in development (It will be implemented in two ways: linear product and conv1d.).

- FSMN: scalar FSMN
- CSFSMN: compact scalar FSMN
- VFSMN: vectorized FSMN
- VFSMNv2: vectorized FSMN implemented by conv1d
- CVFSMN: compact vectorized FSMN 
- CVFSMNv2: compact vectorized FSMN implemented by conv1d

The code is modified from [tensorflow](https://github.com/katsugeneration/tensor-fsmn) version.

See:
- Feedforward Sequential Memory Networks: A New Structure to Learn Long-term Dependency [[arXiv](https://arxiv.org/abs/1512.08301)]
- Compact Feedforward Sequential Memory Networks for Large Vocabulary Continuous Speech Recognition [[PDF](https://pdfs.semanticscholar.org/eb62/dabac5f62f267a42b9f2615e057dd21eb9d3.pdf)]
- Deep-FSMN for Large Vocabulary Continuous Speech Recognition [[arXiv](https://arxiv.org/abs/1803.05030)]