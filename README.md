# LOL-LM: Satirical News Generation with GPT Finetuning

**Project 1: Finetuning GPT Models for Automated News Generation**

A complete pipeline for finetuning GPT models to generate satirical news articles in the style of The Onion and other satirical news sources.

## 🎯 Features

- **Easy Data Ingestion**: Simple JSON/CSV format for adding your own satirical articles
- **Flexible Training**: Support for multiple GPT model sizes (DistilGPT2, GPT-2, GPT-2 Medium)
- **Multiple Generation Modes**:
  - Headline generation
  - Full article generation
  - Text continuation
  - Batch generation
- **Comprehensive Evaluation**: Perplexity, diversity metrics, and quality analysis
- **Interactive Mode**: Generate articles interactively
- **Production Ready**: Save/load models, batch processing, configurable parameters

## 📋 Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- See `requirements.txt` for full list

## 🚀 Quick Start

### 1. Installation

```bash
# Clone or download this repository
cd LOL-LM

# Create virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare Your Data

Create a JSON file with your satirical articles:

```json
{
  "articles": [
    {
      "title": "Your Satirical Headline Here",
      "content": "The article content...",
      "category": "Technology"
    }
  ]
}
```

Or use CSV format with columns: `title`, `content`, `category`

**Where to get data:**
- The Onion (https://www.theonion.com/)
- The Babylon Bee (https://babylonbee.com/)
- Hard Times (https://thehardtimes.net/)
- Your own creative satirical writing

### 3. Validate Your Data

```python
from data_loader import validate_data

validate_data('path/to/your/articles.json')
```

### 4. Train the Model

Open `lollm.ipynb` and follow the step-by-step tutorial, or use Python:

```python
from data_loader import SatiricalNewsDataset
from model_trainer import SatiricalNewsTrainer
from transformers import AutoTokenizer

# Load data
tokenizer = AutoTokenizer.from_pretrained("gpt2")
dataset = SatiricalNewsDataset(
    'data/articles.json',
    tokenizer,
    max_length=512
)

# Train
trainer = SatiricalNewsTrainer(model_name="gpt2")
train_ds, eval_ds = trainer.prepare_datasets(dataset)
trainer.train(train_ds, eval_ds, num_epochs=5)
```

### 5. Generate Articles

```python
from text_generator import SatiricalNewsGenerator

generator = SatiricalNewsGenerator("models/satirical_news_gpt")

# Generate headlines
headlines = generator.generate_headline(
    prompt="Area Man",
    category="Technology",
    num_return_sequences=3
)

# Generate full article
article = generator.generate_article(
    headline="Scientists Discover Coffee Works Through Placebo Effect",
    category="Science"
)

print(article['content'])
```

## 📁 Project Structure

```
LOL-LM/
├── lollm.ipynb           # Main tutorial notebook
├── data_loader.py        # Data loading and preprocessing
├── model_trainer.py      # Model training pipeline
├── text_generator.py     # Text generation utilities
├── evaluation.py         # Evaluation metrics
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── data/                # Your data files go here
│   └── sample_articles.json
└── models/              # Trained models saved here
    └── satirical_news_gpt/
```

## 🎓 Tutorial Notebook

The `lollm.ipynb` notebook provides a complete walkthrough:

1. **Data Preparation**: Create and validate your dataset
2. **Model Selection**: Choose the right model size for your needs
3. **Training**: Finetune the model on your data
4. **Generation**: Generate satirical news in multiple ways
5. **Evaluation**: Assess model quality
6. **Deployment**: Save and load models for production use

## 📊 Data Format

### JSON Format
```json
{
  "articles": [
    {
      "title": "Article Headline",
      "content": "Article body text...",
      "category": "Technology"  // Optional
    }
  ]
}
```

### CSV Format
```
title,content,category
"Article Headline","Article body...","Technology"
```

## 🔧 Configuration

### Model Options
- `distilgpt2`: Fastest, smallest (82M parameters)
- `gpt2`: Balanced (124M parameters) 
- `gpt2-medium`: Better quality (355M parameters)
- `gpt2-large`: Best quality (774M parameters, requires more VRAM)

### Training Parameters
```python
trainer.train(
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    num_epochs=5,           # More epochs = better fit
    batch_size=4,           # Reduce if out of memory
    learning_rate=5e-5,     # Lower for fine-tuning
    gradient_accumulation_steps=4  # Effective larger batch
)
```

### Generation Parameters
```python
generator.generate_headline(
    temperature=0.8,  # Higher = more creative (0.7-1.0)
    top_k=50,         # Top-k sampling (40-50)
    top_p=0.95,       # Nucleus sampling (0.9-0.95)
    max_length=100    # Maximum tokens to generate
)
```

## 📈 Evaluation Metrics

The evaluation module provides:

- **Perplexity**: How well the model fits the data (lower is better)
- **Distinct-N**: Vocabulary diversity (higher is better)
- **Repetition Score**: Text repetitiveness (lower is better)
- **Readability**: Average sentence length and structure

```python
from evaluation import ModelEvaluator

evaluator = ModelEvaluator("models/satirical_news_gpt")
report = evaluator.comprehensive_evaluation(test_dataset)
```

## 💡 Tips for Better Results

### Data Quality
- **Quantity**: Aim for at least 100-500 articles for decent results
- **Quality**: Ensure articles are well-written and consistent in style
- **Diversity**: Include various topics and writing styles
- **Cleaning**: Remove HTML, ads, and non-article content

### Training
- **Start Small**: Test with DistilGPT2 first
- **Monitor Loss**: Loss should decrease steadily
- **Validation**: Use evaluation set to prevent overfitting
- **Checkpoints**: Save intermediate checkpoints
- **GPU**: Use GPU for much faster training

### Generation
- **Temperature**: Experiment with 0.7-1.0 for creativity
- **Prompts**: Good prompts lead to better outputs
- **Filtering**: Generate multiple samples and pick the best
- **Post-processing**: Clean up generated text if needed

## 🐛 Troubleshooting

### Out of Memory
```python
# Reduce batch size
trainer.train(batch_size=1, gradient_accumulation_steps=8)

# Or use a smaller model
trainer = SatiricalNewsTrainer(model_name="distilgpt2")
```

### Poor Quality Output
```python
# Try more training epochs
trainer.train(num_epochs=10)

# Or add more training data
# Or try a larger model (gpt2-medium)
```

### Repetitive Output
```python
# Increase temperature and use nucleus sampling
generator.generate_article(
    headline="...",
    temperature=0.9,
    top_p=0.95
)
```

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
- [GPT-2 Paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [Fine-tuning Guide](https://huggingface.co/docs/transformers/training)

## 🤝 Contributing

This is a course project, but feel free to:
- Add more features
- Improve evaluation metrics
- Optimize training pipeline
- Add support for more model architectures

## 📝 License

This project is for educational purposes as part of the Advanced AI course group project.

## 🎉 Examples

### Generated Headlines
- "Area Man Still Hasn't Learned Python Despite 47 'Learn Python in 24 Hours' Books"
- "Scientists Discover Coffee Works Through Placebo Effect"
- "Local Developer Achieves Inbox Zero By Declaring Email Bankruptcy"

### Usage in Production

```python
# Load trained model
generator = SatiricalNewsGenerator("models/satirical_news_gpt")

# Generate batch of articles
articles = generator.generate_batch(
    num_articles=10,
    categories=["Technology", "Politics", "Science"]
)

# Save to file
import json
with open('generated_articles.json', 'w') as f:
    json.dump(articles, f, indent=2)
```

## 🚀 Next Steps

After completing the basic tutorial:

1. **Collect More Data**: Scrape or manually collect more satirical articles
2. **Experiment with Models**: Try GPT-2 Medium or GPT-Neo
3. **Fine-tune Parameters**: Optimize temperature, top-k, top-p
4. **Add Features**: Category-specific models, style transfer, etc.
5. **Deploy**: Create a web interface or API

## 📧 Support

For questions or issues:
- Check the troubleshooting section
- Review the tutorial notebook
- Consult the code comments

---

**Happy Satirical News Generation! 📰😄**
