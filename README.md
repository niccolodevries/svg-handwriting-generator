# SVG Handwriting Generator

A Python GUI application that generates realistic handwriting as SVG files at A4 paper size. The output uses single-stroke paths (M/L commands only), making it ideal for use with pen plotters.

Powered by a recurrent neural network trained on real handwriting samples.

## Features

- **Neural handwriting synthesis** — generates realistic, human-like handwriting using an RNN
- **13 handwriting styles** — choose from different writer personalities (styles 0-12)
- **Plotter-ready SVG** — all strokes are single continuous paths with no fills, perfect for pen plotters
- **A4 page layout** — proper margins, word wrapping, and line spacing
- **GUI with live preview** — see the rendered handwriting before exporting
- **Configurable** — adjust neatness (bias), scale, line spacing, stroke width, and ink color

## Setup

Requires Python 3.12 (for TensorFlow compatibility and tkinter support).

```bash
# Create virtual environment
python3.12 -m venv .venv

# Install dependencies
.venv/bin/pip install tensorflow tf-keras tensorflow-probability svgwrite numpy scipy scikit-learn matplotlib pandas
```

## Usage

```bash
.venv/bin/python3 main.py
```

This opens the GUI where you can:
1. Type or paste your text
2. Choose a handwriting style and adjust settings
3. Click **Generate Handwriting** to render
4. Click **Export SVG** to save the A4-sized SVG file

## Project Structure

```
main.py          — GUI application (tkinter)
engine.py        — Rendering engine: text layout, A4 page mapping, SVG generation
synth/           — Neural handwriting synthesis model (see attribution below)
  demo.py        — Model API (Hand class)
  rnn.py         — RNN model definition
  rnn_cell.py    — LSTM attention cell
  drawing.py     — Stroke processing utilities
  checkpoints/   — Pretrained model weights
  styles/        — Handwriting style priming data
```

## Acknowledgments

The neural handwriting synthesis model in the `synth/` directory is based on **[handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)** by **Sean Vasquez** ([@sjvasquez](https://github.com/sjvasquez)).

This is an implementation of the handwriting synthesis experiments described in the paper:

> **Generating Sequences with Recurrent Neural Networks**
> Alex Graves
> [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)

The original code was written for TensorFlow 1.x and has been modified here for TensorFlow 2.x compatibility (`tf.compat.v1`). The pretrained model weights and style data are from the original repository. All credit for the model architecture, training, and pretrained weights goes to Sean Vasquez and the original paper by Alex Graves.

### Changes made to the original `synth/` code

- Migrated from TensorFlow 1.x to TensorFlow 2.x using `tf.compat.v1` compatibility layer
- Replaced `tf.contrib` APIs with their TF2 equivalents (`tensorflow_probability`, `tf.compat.v1.nn.rnn_cell`)
- Replaced removed internal TensorFlow APIs in `rnn_ops.py` with compatible implementations
- Fixed NumPy deprecations (`.tostring()` to `.tobytes()`, `np.sum` with generators)
- Fixed file path handling to work when imported from a parent directory
- Added `get_stroke_data()` method to `Hand` class for programmatic access to raw stroke data
