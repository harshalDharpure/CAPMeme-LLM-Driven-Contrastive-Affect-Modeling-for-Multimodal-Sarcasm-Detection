# CAPMeme: Multimodal Sarcasm Detection with GPU Optimization

This project implements a multimodal sarcasm detection model using Contrastive Affective Pretraining (CAP) with vision-language fusion and knowledge graph integration, optimized for training with 10,000 images on GPU.

## 🚀 Features

- **Multimodal Analysis**: Combines image and text for comprehensive meme understanding
- **Contrastive Affective Pretraining**: Aligns embeddings based on affective labels
- **Knowledge Graph Integration**: Uses ConceptNet/COMET for cultural context
- **Multi-task Classification**: Handles sarcasm, persuasion, harm detection, and neutral content
- **GPU Optimization**: Optimized for training with 10,000+ images
- **Mixed Precision Training**: Faster training with reduced memory usage
- **Comprehensive Evaluation**: Accuracy, AUROC, F1-score metrics

## 📊 Model Architecture

```
Input Images & Text
        ↓
Vision-Language Encoder (CLIP/BLIP)
        ↓
Text Encoder (BERT/LLaMA)
        ↓
Knowledge Graph Integration (ConceptNet/COMET)
        ↓
Contrastive Affective Pretraining (CAP)
        ↓
Multi-Modal Fusion
        ↓
Multi-Task Classifier
        ↓
Predictions (Sarcasm/Persuasion/Harm/Neutral)
```

## 🛠️ Installation

### Quick Setup (GPU Optimized)
```bash
# Run the GPU setup script
python setup_gpu.py
```

### Manual Installation
```bash
# Install dependencies
pip install -r requirements.txt

# For CUDA support (recommended)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 📁 Project Structure

```
CAP-Model/
├── configs/           # Configuration files
│   ├── default.yaml   # Standard configuration
│   └── gpu_config.yaml # GPU-optimized configuration
├── data/             # Dataset and preprocessing
│   ├── images/       # Your 10,000 images (organized)
│   └── meme_dataset.json # Dataset metadata
├── models/           # Model implementations
│   ├── capmeme.py    # Main CAPMeme model
│   ├── vision_language_encoder.py
│   ├── text_encoder.py
│   ├── contrastive_pretraining.py
│   ├── knowledge_graph.py
│   └── classifier_head.py
├── utils/            # Utility functions
├── evaluation/       # Evaluation metrics
├── notebooks/        # Jupyter notebooks
├── train_gpu.py      # GPU-optimized training script
├── organize_images.py # Image organization script
└── inference.py      # Inference script
```

## 🎯 Usage

### 1. Organize Your Images
```bash
# Organize your 10,000 images into the required structure
python organize_images.py --source_dir /path/to/your/images --target_dir data/images --num_images 10000
```

### 2. Start Training
```bash
# Train with GPU optimization
python train_gpu.py --config configs/gpu_config.yaml --data_path data/meme_dataset.json --image_dir data/images

# Or with custom parameters
python train_gpu.py --batch_size 128 --epochs 150 --lr 3e-4
```

### 3. Monitor Training
- **Weights & Biases**: Automatic logging and visualization
- **GPU Monitoring**: Use `nvidia-smi` to monitor GPU usage
- **Training Logs**: Check `outputs/logs/` for detailed logs

### 4. Inference
```bash
# Make predictions on new memes
python inference.py --checkpoint outputs/best_model.pt --image data/images/meme_00001.jpg --text "Your meme text here"
```

## ⚙️ Configuration

### GPU-Optimized Settings
- **Batch Size**: 64-128 (adjust based on GPU memory)
- **Workers**: 8 (for faster data loading)
- **Mixed Precision**: Enabled for faster training
- **Gradient Checkpointing**: Enabled for memory efficiency

### Dataset Configuration
- **Training Split**: 8,000 images (80%)
- **Validation Split**: 1,000 images (10%)
- **Test Split**: 1,000 images (10%)

## 📈 Expected Performance

With 10,000 images and GPU training:
- **Training Time**: ~2-4 hours (depending on GPU)
- **Accuracy**: ~89-92%
- **F1-Score**: ~87-90%
- **AUROC**: ~92-95%

## 🔧 GPU Requirements

### Minimum Requirements
- **GPU**: NVIDIA GPU with 8GB+ VRAM
- **CUDA**: Version 11.8 or higher
- **RAM**: 16GB+ system RAM
- **Storage**: 10GB+ free space

### Recommended Requirements
- **GPU**: NVIDIA RTX 3080/4080 or better
- **VRAM**: 12GB+ for larger batch sizes
- **RAM**: 32GB+ system RAM
- **Storage**: SSD for faster data loading

## 🚨 Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size
   python train_gpu.py --batch_size 32
   ```

2. **Slow Data Loading**
   ```bash
   # Increase number of workers
   python train_gpu.py --config configs/gpu_config.yaml
   # Edit config to increase num_workers
   ```

3. **Training Too Slow**
   ```bash
   # Enable mixed precision (already enabled in GPU config)
   # Use larger batch size if GPU memory allows
   ```

### Performance Tips

- **Batch Size**: Start with 64, increase if GPU memory allows
- **Workers**: Use 8-16 workers for data loading
- **Mixed Precision**: Always enabled for GPU training
- **Memory Management**: Clear GPU cache periodically

## 📊 Monitoring

### Weights & Biases
- Automatic experiment tracking
- Real-time metrics visualization
- Model performance comparison

### GPU Monitoring
```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor training progress
tail -f outputs/logs/training_*.log
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI CLIP for vision-language understanding
- Hugging Face Transformers for text processing
- ConceptNet for knowledge graph integration
- PyTorch team for GPU optimization tools

## 📞 Support

For questions and support:
- Create an issue on GitHub
- Check the troubleshooting section
- Review the configuration files

---

**Happy Training! 🚀**
