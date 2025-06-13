# MathInk-AI

![Python](https://img.shields.io/badge/Python-3.11+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-orange.svg)
![Flask](https://img.shields.io/badge/Flask-2.3+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

A production-ready handwritten mathematical equation recognition system with real-time solving capabilities.

## Features

- **Recognition**: Digits (0-9), operators (+, -, ×, =), and variables (a-z)
- **Solving Modes**: Basic arithmetic evaluation and linear equation solving
- **Performance**: 96.8% accuracy with sub-second processing time
- **Architecture**: Enhanced CNN with residual connections and attention mechanisms
- **Compatibility**: Automatic model format conversion for modern TensorFlow versions

## Quick Start

### Prerequisites
- Python 3.11+
- 4GB+ RAM
- GPU (optional, recommended)

### Installation
```bash
git clone https://github.com/Dipan2004/MathInk-AI.git
cd MathInk-AI
python setup.py
```

### Run Application
```bash
python app.py
```
Access the web interface at `http://localhost:5000`

## Usage

### Web Interface
1. Upload handwritten equation image
2. Select solving mode (Basic Arithmetic or Linear Equations)
3. Click "Analyze Equation" for results

### Python API
```python
from enhanced_cnn import EnhancedCNN
import cv2

model = EnhancedCNN()
image = cv2.imread('equation.png', cv2.IMREAD_GRAYSCALE)
equation = model.predict(image)
```

### REST API
```bash
curl -X POST -F "file=@equation.png" -F "mode=linear" http://localhost:5000/predict
```

## Model Conversion

The system includes automatic conversion from legacy Keras models (JSON + H5) to modern .keras format for compatibility with TensorFlow 2.x and Python 3.11+.

```bash
python converter.py
```

## Architecture

**Processing Pipeline:**
Input Image → Preprocessing → Character Detection → Equation Assembly → Solving → Result

**Model Architecture:**
- Dual-path CNN combining MobileNetV2 and custom ResNet branches
- Feature fusion layer
- 39-class classification head
- Post-processing and correction pipeline

## Performance

| Metric | Value |
|--------|-------|
| Overall Accuracy | 96.8% |
| Processing Speed | 0.3s |
| Model Size | 15MB |
| Memory Usage | 200MB |

## Training

Organize training data by class:
```
data/extracted_images/
├── 0/          # Digit 0 images
├── 1/          # Digit 1 images
├── plus/       # Plus sign images
└── ...
```

Run training:
```bash
python train.py
```

## Project Structure

```
MathInk-AI/
├── app.py                    # Flask web application
├── train.py                  # Training pipeline
├── converter.py              # Model format converter
├── setup.py                  # Setup script
├── models/                   # Model files
├── templates/                # HTML templates
├── static/                   # Web assets
└── data/                     # Training data
```

## Contributing

Contributions are welcome. Please:
1. Fork the repository
2. Create a feature branch
3. Submit a pull request

Priority areas:
- Support for fractions and complex expressions
- Additional mathematical symbols
- Performance optimizations
- Mobile app development

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Contact

- **Issues**: [GitHub Issues](https://github.com/Dipan2004/MathInk-AI/issues)
- **Email**: dipangiri.dev@gmail.com

---

**Author**: Dipan Giri
