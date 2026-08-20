# Dataset card: Tiny Shakespeare

## Source

The project downloads `input.txt` from Andrej Karpathy's `char-rnn` repository:
<https://github.com/karpathy/char-rnn/tree/master/data/tinyshakespeare>.

The text is derived from works of William Shakespeare, which are in the public domain. The upstream repository should be consulted for provenance and redistribution details.

## Use

- Model: TinyGPT
- Task: next-character prediction
- Split: first 90% for training, final 10% for validation
- Quick profile: first 200,000 characters before the 90/10 split
- Full profile: complete downloaded file

## Leakage controls

Training batches never sample from the validation suffix. Because the source is one continuous compilation and repeated phrases may occur, this split does not guarantee independence by play, speaker, or passage.

## Quality concerns

Character prediction on a small literary corpus is not representative of modern LLM pretraining or general language ability. Perplexity is only comparable between runs with the same tokenizer, split, and context length.

