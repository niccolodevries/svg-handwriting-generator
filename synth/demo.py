import os
import logging

import numpy as np
import torch
import svgwrite

_synth_dir = os.path.dirname(os.path.abspath(__file__))

import drawing
from model import HandwritingSynthesisModel


class Hand(object):

    def __init__(self):
        self.device = 'cpu'
        self.model = HandwritingSynthesisModel(
            lstm_size=400,
            num_attn_mixture_components=10,
            num_output_mixture_components=20,
            vocab_size=len(drawing.alphabet),
        )
        checkpoint_path = os.path.join(_synth_dir, 'checkpoints', 'model.pt')
        state_dict = torch.load(checkpoint_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state_dict)
        self.model.eval()

    def write(self, filename, lines, biases=None, styles=None, stroke_colors=None, stroke_widths=None):
        valid_char_set = set(drawing.alphabet)
        for line_num, line in enumerate(lines):
            if len(line) > 75:
                raise ValueError(
                    "Each line must be at most 75 characters. "
                    "Line {} contains {}".format(line_num, len(line))
                )
            for char in line:
                if char not in valid_char_set:
                    raise ValueError(
                        "Invalid character {} detected in line {}. "
                        "Valid character set is {}".format(char, line_num, valid_char_set)
                    )

        strokes = self._sample(lines, biases=biases, styles=styles)
        self._draw(strokes, lines, filename, stroke_colors=stroke_colors, stroke_widths=stroke_widths)

    def _sample(self, lines, biases=None, styles=None):
        num_samples = len(lines)
        max_tsteps = 40 * max(len(i) for i in lines)
        biases = biases if biases is not None else [0.5] * num_samples

        x_prime = np.zeros([num_samples, 1200, 3])
        x_prime_len = np.zeros([num_samples])
        chars = np.zeros([num_samples, 120])
        chars_len = np.zeros([num_samples])

        if styles is not None:
            for i, (cs, style) in enumerate(zip(lines, styles)):
                x_p = np.load(os.path.join(_synth_dir, 'styles', 'style-{}-strokes.npy'.format(style)))
                c_p = np.load(os.path.join(_synth_dir, 'styles', 'style-{}-chars.npy'.format(style)))
                c_p = c_p.tobytes().decode('utf-8')

                c_p = str(c_p) + " " + cs
                c_p = drawing.encode_ascii(c_p)
                c_p = np.array(c_p)

                x_prime[i, :len(x_p), :] = x_p
                x_prime_len[i] = len(x_p)
                chars[i, :len(c_p)] = c_p
                chars_len[i] = len(c_p)
        else:
            for i in range(num_samples):
                encoded = drawing.encode_ascii(lines[i])
                chars[i, :len(encoded)] = encoded
                chars_len[i] = len(encoded)

        # Convert to tensors
        chars_t = torch.from_numpy(chars).long().to(self.device)
        chars_len_t = torch.from_numpy(chars_len).int().to(self.device)
        bias_t = torch.tensor(biases, dtype=torch.float32, device=self.device)

        prime = styles is not None
        x_prime_t = torch.from_numpy(x_prime).float().to(self.device) if prime else None
        x_prime_len_t = torch.from_numpy(x_prime_len).int().to(self.device) if prime else None

        samples = self.model.sample(
            chars_t, chars_len_t, bias_t, max_tsteps,
            prime=prime, x_prime=x_prime_t, x_prime_len=x_prime_len_t,
        )
        return samples

    def _draw(self, strokes, lines, filename, stroke_colors=None, stroke_widths=None):
        stroke_colors = stroke_colors or ['black'] * len(lines)
        stroke_widths = stroke_widths or [2] * len(lines)

        line_height = 60
        view_width = 1000
        view_height = line_height * (len(strokes) + 1)

        dwg = svgwrite.Drawing(filename=filename)
        dwg.viewbox(width=view_width, height=view_height)
        dwg.add(dwg.rect(insert=(0, 0), size=(view_width, view_height), fill='white'))

        initial_coord = np.array([0, -(3 * line_height / 4)])
        for offsets, line, color, width in zip(strokes, lines, stroke_colors, stroke_widths):
            if not line:
                initial_coord[1] -= line_height
                continue

            offsets[:, :2] *= 1.5
            stroke_coords = drawing.offsets_to_coords(offsets)
            stroke_coords = drawing.denoise(stroke_coords)
            stroke_coords[:, :2] = drawing.align(stroke_coords[:, :2])

            stroke_coords[:, 1] *= -1
            stroke_coords[:, :2] -= stroke_coords[:, :2].min() + initial_coord
            stroke_coords[:, 0] += (view_width - stroke_coords[:, 0].max()) / 2

            prev_eos = 1.0
            p = "M{},{} ".format(0, 0)
            for x, y, eos in zip(*stroke_coords.T):
                p += '{}{},{} '.format('M' if prev_eos == 1.0 else 'L', x, y)
                prev_eos = eos
            path = svgwrite.path.Path(p)
            path = path.stroke(color=color, width=width, linecap='round').fill("none")
            dwg.add(path)

            initial_coord[1] -= line_height

        dwg.save()

    def get_stroke_data(self, lines, biases=None, styles=None):
        """Return raw stroke data as list of numpy arrays (one per line).

        Each array has shape (N, 3) with columns [dx, dy, end_of_stroke].
        """
        return self._sample(lines, biases=biases, styles=styles)
