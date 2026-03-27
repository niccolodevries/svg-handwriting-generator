"""Handwriting rendering engine using neural handwriting synthesis.

Wraps the sjvasquez/handwriting-synthesis RNN model to generate
realistic single-stroke handwriting on A4 pages.
"""

import os
import sys
import warnings

# TF setup must happen before any TF imports
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

import numpy as np

# Add synth to path
_synth_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth')
if _synth_dir not in sys.path:
    sys.path.insert(0, _synth_dir)

import drawing as synth_drawing

# A4 dimensions in mm
A4_WIDTH = 210.0
A4_HEIGHT = 297.0

DEFAULT_MARGINS = {'top': 25, 'bottom': 25, 'left': 20, 'right': 20}


class HandwritingModel:
    """Lazy-loaded singleton for the neural handwriting model."""
    _instance = None

    @classmethod
    def get(cls):
        if cls._instance is None:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                from demo import Hand
                cls._instance = Hand()
        return cls._instance


def _valid_chars():
    """Return set of valid characters for the model."""
    return set(synth_drawing.alphabet)


def sanitize_text(text):
    """Replace unsupported characters with closest alternatives."""
    valid = _valid_chars()
    replacements = {
        '\u2018': "'", '\u2019': "'", '\u201c': '"', '\u201d': '"',
        '\u2013': '-', '\u2014': '-', '\u2026': '...',
        '\t': '    ', 'Q': 'q', 'X': 'x',  # Q and X missing from training set
    }
    result = []
    for ch in text:
        if ch in valid or ch == '\n':
            result.append(ch)
        elif ch in replacements:
            result.append(replacements[ch])
        elif ch.upper() in valid:
            result.append(ch.upper())
        # Skip unsupported characters
    return ''.join(result)


def wrap_text(text, max_chars=60):
    """Word-wrap text into lines of at most max_chars, respecting newlines."""
    lines = []
    for paragraph in text.split('\n'):
        if not paragraph.strip():
            lines.append('')
            continue
        words = paragraph.split(' ')
        current = ''
        for word in words:
            if not word:
                continue
            if current and len(current) + 1 + len(word) > max_chars:
                lines.append(current)
                current = word
            else:
                current = current + ' ' + word if current else word
        if current:
            lines.append(current)
    return lines


class HandwritingRenderer:
    """Renders text as realistic handwriting using a neural network."""

    def __init__(self, style=9, bias=0.75, stroke_width=0.4,
                 color='#1a1a2e', scale=1.0, line_spacing=1.0,
                 max_chars_per_line=60, margins=None):
        self.style = style
        self.bias = bias
        self.stroke_width = stroke_width
        self.color = color
        self.scale = scale  # overall scale multiplier
        self.line_spacing = line_spacing  # extra line spacing multiplier
        self.max_chars_per_line = max_chars_per_line
        self.margins = margins or dict(DEFAULT_MARGINS)

    def render(self, text):
        """Render text and return (svg_paths, polylines, page_w, page_h).

        svg_paths: list of SVG path d-strings
        polylines: list of [(x,y), ...] for preview
        """
        text = sanitize_text(text)
        lines = wrap_text(text, self.max_chars_per_line)

        if not lines or all(l.strip() == '' for l in lines):
            return [], [], A4_WIDTH, A4_HEIGHT

        hand = HandwritingModel.get()

        # Filter out empty lines for the model, track their positions
        model_lines = []
        line_indices = []  # maps model output index to page line index
        for i, line in enumerate(lines):
            if line.strip():
                model_lines.append(line)
                line_indices.append(i)

        if not model_lines:
            return [], [], A4_WIDTH, A4_HEIGHT

        biases = [self.bias] * len(model_lines)
        styles = [self.style] * len(model_lines)

        stroke_data = hand.get_stroke_data(model_lines, biases=biases, styles=styles)

        # Convert stroke data to page coordinates
        svg_paths = []
        polylines = []

        usable_w = A4_WIDTH - self.margins['left'] - self.margins['right']
        usable_h = A4_HEIGHT - self.margins['top'] - self.margins['bottom']

        # Base line height in mm (tuned for readable handwriting)
        base_line_height = 8.0 * self.scale * self.line_spacing
        total_lines = len(lines)

        for data_idx, offsets in enumerate(stroke_data):
            page_line_idx = line_indices[data_idx]

            offsets = offsets.copy()
            offsets[:, :2] *= 1.5

            coords = synth_drawing.offsets_to_coords(offsets)
            coords = synth_drawing.denoise(coords)
            coords[:, :2] = synth_drawing.align(coords[:, :2])

            # Flip Y
            coords[:, 1] *= -1

            # Normalize to bounding box
            x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
            x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
            raw_w = x_max - x_min
            raw_h = y_max - y_min

            if raw_w < 1 or raw_h < 1:
                continue

            # Scale to fit usable width
            fit_scale = min(usable_w / raw_w, base_line_height / raw_h) * self.scale
            # Don't exceed usable width
            if raw_w * fit_scale > usable_w:
                fit_scale = usable_w / raw_w

            # Position on page
            x_offset = self.margins['left']
            y_offset = self.margins['top'] + (page_line_idx + 1) * base_line_height

            if y_offset + base_line_height > A4_HEIGHT - self.margins['bottom']:
                break  # off page

            # Build path
            path_parts = []
            flat_points = []
            prev_eos = 1.0

            for x, y, eos in zip(coords[:, 0], coords[:, 1], coords[:, 2]):
                px = x_offset + (x - x_min) * fit_scale
                py = y_offset + (y - y_min) * fit_scale

                if prev_eos == 1.0:
                    path_parts.append(f'M{px:.2f},{py:.2f}')
                    if flat_points:
                        polylines.append(flat_points)
                        flat_points = []
                else:
                    path_parts.append(f'L{px:.2f},{py:.2f}')

                flat_points.append((px, py))
                prev_eos = eos

            if flat_points:
                polylines.append(flat_points)

            svg_paths.append(' '.join(path_parts))

        return svg_paths, polylines, A4_WIDTH, A4_HEIGHT


def generate_svg(svg_paths, color='#1a1a2e', stroke_width=0.4):
    """Generate a complete SVG document string for A4."""
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="{A4_WIDTH}mm" height="{A4_HEIGHT}mm" '
        f'viewBox="0 0 {A4_WIDTH} {A4_HEIGHT}">',
        f'  <rect width="{A4_WIDTH}" height="{A4_HEIGHT}" fill="white"/>',
    ]

    for path_d in svg_paths:
        if path_d:
            lines.append(
                f'  <path d="{path_d}" fill="none" stroke="{color}" '
                f'stroke-width="{stroke_width}" stroke-linecap="round" '
                f'stroke-linejoin="round"/>'
            )

    lines.append('</svg>')
    return '\n'.join(lines)
