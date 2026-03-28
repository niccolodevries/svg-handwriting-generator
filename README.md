# SVG Handwriting Generator

A Python GUI application that generates realistic handwriting as SVG files at A4 paper size. The output uses single-stroke paths (M/L commands only), making it ideal for use with pen plotters.

Powered by a recurrent neural network (Graves, 2013) trained on real handwriting samples, ported to PyTorch.

## Features

- **Neural handwriting synthesis** — generates realistic, human-like handwriting using an RNN
- **13 handwriting styles** — choose from different writer personalities (styles 0-12)
- **Plotter-ready SVG** — all strokes are single continuous paths with no fills, perfect for pen plotters
- **Direct Bambu Suite export** — export `.lac` files that open directly in Bambu Suite for the H2D pen plotter, no SVG import step needed
- **PDF export** — multi-page PDF generation with no external libraries
- **A4 page layout** — proper margins, word wrapping, and consistent text scaling
- **GUI with live preview** — see the rendered handwriting before exporting
- **Configurable** — adjust neatness (bias), scale, line spacing, stroke width, and ink color

## Setup

Requires Python 3.10+ with tkinter support.

```bash
# Create virtual environment
python3 -m venv .venv

# Install dependencies
.venv/bin/pip install torch numpy scipy svgwrite matplotlib Pillow
```

## Usage

```bash
.venv/bin/python3 main.py
```

This opens the GUI where you can:
1. Type or paste your text
2. Choose a handwriting style and adjust settings
3. Click **Generate Handwriting** to render
4. Click **Export SVG** to save the A4-sized SVG file, or **Export LAC (Bambu Suite)** to save a `.lac` project file that can be opened directly in Bambu Suite for the H2D

## Handwriting Styles

The model includes 13 handwriting styles primed from different writers in the IAM handwriting dataset. Quality varies by style — some are production-ready for full-page text while others degrade on longer documents.

| Style | Description | Quality | Notes |
|-------|-------------|---------|-------|
| 0 | Upright, well-spaced, relaxed print | Excellent | Best all-round style. Clean and highly legible |
| 1 | Slightly slanted, compact print | Good | Slightly dense but clean. Minor trailing artifacts |
| 2 | Highly cursive, connected strokes | Poor | Degrades on long text. Words merge, letters garble in second half of page |
| 3 | Upright print with loopy descenders | Excellent | Clean and consistent at any length |
| 4 | Scratchy, compact, slightly irregular | Fair | Occasional garbled/overlapping text on longer passages |
| 5 | Clean, compact, slightly slanted | Good | Well-suited for dense text. Reliable |
| 6 | Wide spacing, upright, round forms | Fair | Inconsistent sizing on long text. Some garbled words |
| 7 | Relaxed cursive-print hybrid | Very Good | Natural and legible. Clean full-page output |
| 8 | Upright, round, slightly lighter strokes | Very Good | Similar to style 7 with rounder forms |
| 9 | Compressed, hurried, smaller letters | Fair | Can compress horizontally on long lines |
| 10 | Slightly slanted, natural flow | Excellent | Best full-page style. Consistent and convincing |
| 11 | Loose, irregular, spaced-out | Poor | Multiple garbled words, inconsistent sizing on long text |
| 12 | Loopy cursive, flowing connections | Very Good | Dense but legible. Distinctive character |

**Recommended styles for production use:** 0, 3, 5, 7, 8, 10, 12

## Known Limitations

These are inherent to the pretrained Graves RNN model weights:

- **Letter "x"** is malformed in most styles (renders as "i", "r", or "o"). Only styles 0, 6, and 8 consistently produce a correct "x"
- **All-capitals text** tends to collapse into overlapping strokes. Mixed case works much better
- **Trailing artifacts** — small stray marks occasionally appear after line-ending periods
- **Styles 2 and 11** degrade significantly on full-page text and are not recommended for long documents
- **Characters Q and X** are not in the training set and are automatically replaced with "q" and "x"

## Testing

A visual test suite generates high-resolution cropped PNG images across 58 test cases:

```bash
# Install test dependencies
.venv/bin/pip install cairosvg

# Run tests (uses multiprocessing for speed)
.venv/bin/python3 test_handwriting.py --workers 4
```

This produces `test_output/report.html` with side-by-side comparisons of word rendering, line wrapping, all 13 styles, bias sweep, and a full-page mixed Dutch/English stress test.

## Project Structure

```
main.py                — GUI application (tkinter)
engine.py              — Rendering engine: text layout, A4 page mapping, SVG/PDF/LAC generation
test_handwriting.py    — Visual test suite (58 test cases, parallel execution)
synth/                 — Neural handwriting synthesis model
  model.py             — PyTorch model (3-layer LSTM + attention + GMM output)
  demo.py              — Model API (Hand class)
  drawing.py           — Stroke processing utilities (denoise, align, coordinate conversion)
  convert_weights.py   — One-time TF checkpoint to PyTorch converter (only needs TensorFlow)
  checkpoints/         — Pretrained model weights (model.pt + original TF checkpoint)
  styles/              — Handwriting style priming data (13 styles from IAM dataset)
  requirements.txt     — Python dependencies
```

## Acknowledgments

The neural handwriting synthesis model is based on **[handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis)** by **Sean Vasquez** ([@sjvasquez](https://github.com/sjvasquez)).

This is an implementation of the handwriting synthesis experiments described in the paper:

> **Generating Sequences with Recurrent Neural Networks**
> Alex Graves
> [arXiv:1308.0850](https://arxiv.org/abs/1308.0850)

The model architecture and pretrained weights originate from Sean Vasquez's original TensorFlow implementation. The model has been ported to PyTorch with identical weight mapping and behavior.
