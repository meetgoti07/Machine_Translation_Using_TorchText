# Machine Translation Using TorchText (German to English)

## Project Overview

This project implements a sequence-to-sequence neural machine translation model using PyTorch and TorchText to translate German sentences to English. The model employs an Encoder-Decoder architecture with GRU (Gated Recurrent Units) and demonstrates the complete pipeline from data preprocessing to translation generation.

## Dataset

**Multi30k Dataset** - A multilingual image description dataset:

- **Language pair**: German (DE) → English (EN)
- **Training split**: German-English sentence pairs
- **Validation split**: Used for evaluation and translation examples
- **Source**: Modified Multi30k dataset from small_DL_repo
- **Tokenization**: Spacy tokenizers for both languages

### Dataset Characteristics:

- **Parallel corpus**: Aligned German-English sentence pairs
- **Domain**: Image descriptions and captions
- **Vocabulary size**: Dynamic based on token frequency
- **Special tokens**: `<unk>`, `<pad>`, `<bos>`, `<eos>`

## Model Architecture

### Sequence-to-Sequence with Attention

#### Encoder (German → Hidden State)

```python
Encoder:
├── Embedding(vocab_de_size, embed_size=300)
├── Dropout(0.1)
└── GRU(embed_size=300, hidden_size=512)
```

#### Decoder (Hidden State → English)

```python
Decoder:
├── Embedding(vocab_en_size, embed_size=300)
├── ReLU activation
├── GRU(embed_size=300, hidden_size=512)
├── Linear(hidden_size=512, vocab_en_size)
└── LogSoftmax(dim=-1)
```

### Architecture Details:

- **Embedding dimension**: 300
- **Hidden state size**: 512
- **Sequence processing**: Teacher forcing during training
- **Output**: Log probabilities for English vocabulary

## Implementation Details

### Text Processing Pipeline

#### 1. Tokenization

```python
# Language-specific tokenizers
de_tokenizer = get_tokenizer('spacy', language='de_core_news_sm')
en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
```

#### 2. Vocabulary Building

```python
# Builds vocabularies from tokenized text
vocab_de = build_vocab_from_iterator(yield_tokens(...), min_freq=1)
vocab_en = build_vocab_from_iterator(yield_tokens(...), min_freq=1)
```

#### 3. Sequence Processing

```python
def collate_fn(batch):
    # Tokenizes text
    # Adds special tokens (<bos>, <eos>)
    # Pads sequences to equal length
    # Returns padded tensors
```

### Training Configuration

- **Batch size**: 32
- **Epochs**: 20
- **Loss function**: NLLLoss (ignores padding tokens)
- **Optimizers**: Adam for both encoder and decoder
- **Learning rate**: 0.001
- **Weight decay**: 0.001

## Training Strategy

### Teacher Forcing Training

```python
def train_one_epoch():
    # 1. Encode source sentence
    encoder_outputs, encoder_hidden = encoder(source_ids)

    # 2. Initialize decoder with encoder's final state
    decoder_hidden = encoder_hidden

    # 3. Teacher forcing: use ground truth as input
    predictions, _ = decoder(target_ids[:, :-1], decoder_hidden)

    # 4. Compute loss against shifted targets
    loss = loss_fn(predictions, target_ids[:, 1:])
```

### Inference Strategy

```python
def eval_one_epoch():
    # 1. Encode source sentence
    # 2. Generate translation word by word
    # 3. Use previous prediction as next input
    # 4. Stop at <eos> token or max length
```

## Key Features

### 1. **Comprehensive Text Preprocessing**

- **Tokenization**: Language-specific processing
- **Vocabulary management**: Special token handling
- **Sequence alignment**: Padding and batching

### 2. **Bidirectional Training/Inference**

- **Training**: Teacher forcing for stable learning
- **Inference**: Autoregressive generation
- **Flexibility**: Different strategies for training vs. evaluation

### 3. **Translation Quality Assessment**

```python
# Real translation examples shown during final epoch:
# Source (German): Original sentence
# Ground Truth (English): Reference translation
# Predicted (English): Model-generated translation
```

### 4. **Memory-Efficient Processing**

- **Padding tokens ignored** in loss calculation
- **Batch processing** for efficiency
- **Dynamic sequence lengths**

## Project Structure

```
Machine_Translation_Using_TorchText/
├── Machine_Translation_German_To_English_TorchText.ipynb
├── .gitignore
└── README.md
```

## Usage

### Prerequisites

```bash
pip install torch==2.0.1 torchtext==0.15.2 torchdata spacy
python -m spacy download en_core_web_sm
python -m spacy download de_core_news_sm
```

### Running the Model

1. **Environment Setup**: Install dependencies and language models
2. **Data Loading**: Automatic download of Multi30k dataset
3. **Training**: Execute notebook cells for complete pipeline
4. **Translation**: View sample translations in final epoch

### Key Code Components

#### Model Training

```python
for e in range(n_epochs):
    train_loss = train_one_epoch()
    eval_loss = eval_one_epoch(e, n_epochs)
    # Displays sample translations in final epoch
```

#### Translation Generation

```python
# Autoregressive generation:
input_id = target[:, 0]  # Start with <bos>
for j in range(max_length):
    probs, hidden = decoder(input_id.unsqueeze(1), hidden)
    _, input_id = torch.topk(probs, 1, dim=-1)
    if input_id.item() == EOS_IDX:
        break
```

## Technical Highlights

### 1. **Sequence-to-Sequence Learning**

- **Variable length handling**: Dynamic sequence processing
- **Context preservation**: Hidden state transfer from encoder to decoder
- **End-to-end training**: Joint optimization of both components

### 2. **Advanced Text Processing**

```python
# Special token management:
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
```

### 3. **Robust Training Pipeline**

- **Loss masking**: Ignores padding tokens
- **Gradient optimization**: Separate optimizers for components
- **Progress monitoring**: Real-time loss tracking

### 4. **Translation Quality Evaluation**

- **Sample generation**: Shows actual translations
- **Comparison display**: Source vs. reference vs. predicted
- **Qualitative assessment**: Visual inspection of results

## Model Performance

### Training Metrics:

- **Training loss**: Decreases consistently across epochs
- **Evaluation loss**: Validation performance tracking
- **Translation quality**: Improves with training progression

### Sample Output Format:

```
Source Sentence: [German text]
GT Sentence: [English reference]
Predicted Sentence: [Model translation]
```

## Future Improvements

1. **Architecture Enhancements**:

   - Attention mechanisms (Bahdanau/Luong attention)
   - Transformer architecture
   - Bidirectional encoders

2. **Training Optimizations**:

   - Learning rate scheduling
   - Beam search decoding
   - BLEU score evaluation

3. **Data Improvements**:

   - Larger parallel corpora
   - Data augmentation techniques
   - Domain adaptation

4. **Advanced Features**:
   - Subword tokenization (BPE)
   - Copy mechanisms
   - Coverage penalties

## Dependencies

- **PyTorch 2.0.1**: Deep learning framework
- **TorchText 0.15.2**: Text processing utilities
- **Spacy**: Tokenization and NLP preprocessing
- **TorchData**: Data loading utilities

## Key Achievements

- Complete seq2seq implementation from scratch
- Bilingual text processing pipeline
- Teacher forcing training strategy
- Autoregressive inference
- Real translation examples generation
- Comprehensive vocabulary management

## Learning Outcomes

This project demonstrates:

- **Sequence-to-Sequence Learning**: Core NMT concepts
- **RNN Architectures**: GRU-based encoder-decoder
- **Text Processing**: TorchText library usage
- **Multilingual NLP**: Cross-language processing
- **Training Strategies**: Teacher forcing vs. autoregressive generation

## Translation Examples

The model learns to translate various German phrases to English:

- **Simple sentences**: Basic subject-verb-object structures
- **Descriptive text**: Image caption-style content
- **Complex grammar**: German grammatical structures to English

---
