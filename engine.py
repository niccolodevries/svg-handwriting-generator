"""Handwriting rendering engine using neural handwriting synthesis.

Wraps a PyTorch RNN model (Graves architecture) to generate
realistic single-stroke handwriting on A4 pages.
"""

import os
import sys

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
        if ch in valid or ch in ('\n', '\x00'):
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

    def __init__(self, style=1, bias=0.80, stroke_width=0.25,
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
        """Render text and return (pages, page_w, page_h).

        pages: list of dicts, each with 'svg_paths' and 'polylines' keys

        Use '---' on its own line to force a page break.
        Paragraphs are kept together when possible.
        """
        # Handle page break markers before sanitization
        PAGE_BREAK = '\x00PAGEBREAK\x00'
        text = text.replace('\n---\n', f'\n{PAGE_BREAK}\n')
        # Handle --- at start/end of text too
        if text.startswith('---\n'):
            text = PAGE_BREAK + text[3:]
        if text.endswith('\n---'):
            text = text[:-3] + PAGE_BREAK

        text = sanitize_text(text)
        lines = wrap_text(text, self.max_chars_per_line)

        if not lines or all(l.strip() == '' for l in lines):
            return [{'svg_paths': [], 'polylines': []}], A4_WIDTH, A4_HEIGHT

        usable_w = A4_WIDTH - self.margins['left'] - self.margins['right']
        base_line_height = 8.0 * self.scale * self.line_spacing
        usable_h = A4_HEIGHT - self.margins['top'] - self.margins['bottom']
        lines_per_page = max(1, int(usable_h / base_line_height))

        # Assign each line a (page, line_on_page) considering:
        # - paragraphs kept together when possible
        # - '---' page break markers
        line_layout = {}  # line_index -> (page_num, line_on_page)
        cur_page = 0
        cur_line = 0

        i = 0
        while i < len(lines):
            # Page break marker
            if PAGE_BREAK in lines[i]:
                if cur_line > 0:
                    cur_page += 1
                    cur_line = 0
                i += 1
                continue

            # Blank line (paragraph separator)
            if lines[i].strip() == '':
                if cur_line > 0:
                    if cur_line >= lines_per_page:
                        cur_page += 1
                        cur_line = 0
                    else:
                        line_layout[i] = (cur_page, cur_line)
                        cur_line += 1
                i += 1
                continue

            # Collect paragraph lines (consecutive non-blank, non-pagebreak)
            para_start = i
            while (i < len(lines) and lines[i].strip() != ''
                   and PAGE_BREAK not in lines[i]):
                i += 1
            para_len = i - para_start

            # If paragraph doesn't fit on remaining page, move to next page
            # (unless we're at the top of a page already)
            if cur_line > 0 and cur_line + para_len > lines_per_page:
                cur_page += 1
                cur_line = 0

            # Place paragraph lines
            for j in range(para_start, i):
                line_layout[j] = (cur_page, cur_line)
                cur_line += 1
                if cur_line >= lines_per_page:
                    cur_page += 1
                    cur_line = 0

        # Filter renderable lines (non-empty, non-pagebreak)
        model_lines = []
        model_line_indices = []  # index into 'lines' list
        for idx, line in enumerate(lines):
            if line.strip() and PAGE_BREAK not in line and idx in line_layout:
                model_lines.append(line)
                model_line_indices.append(idx)

        if not model_lines:
            return [{'svg_paths': [], 'polylines': []}], A4_WIDTH, A4_HEIGHT

        hand = HandwritingModel.get()
        biases = [self.bias] * len(model_lines)
        styles = [self.style] * len(model_lines)
        stroke_data = hand.get_stroke_data(model_lines, biases=biases, styles=styles)

        # First pass: process all strokes and compute a consistent scale
        processed = []
        raw_heights = []
        for offsets in stroke_data:
            offsets = offsets.copy()
            offsets[:, :2] *= 1.5

            coords = synth_drawing.offsets_to_coords(offsets)
            coords = synth_drawing.denoise(coords)
            coords[:, :2] = synth_drawing.align(coords[:, :2])
            coords[:, 1] *= -1

            x_min, y_min = coords[:, 0].min(), coords[:, 1].min()
            x_max, y_max = coords[:, 0].max(), coords[:, 1].max()
            raw_w = x_max - x_min
            raw_h = y_max - y_min

            processed.append({
                'coords': coords,
                'x_min': x_min, 'y_min': y_min,
                'raw_w': raw_w, 'raw_h': raw_h,
            })
            if raw_h > 1:
                raw_heights.append(raw_h)

        if not raw_heights:
            return [{'svg_paths': [], 'polylines': []}], A4_WIDTH, A4_HEIGHT

        # Use median height for consistent character sizing across all lines
        median_h = float(np.median(raw_heights))
        uniform_scale = (base_line_height / median_h) * self.scale

        # Second pass: render with consistent scale
        pages = [{'svg_paths': [], 'polylines': []}]

        for data_idx, line_data in enumerate(processed):
            line_idx = model_line_indices[data_idx]
            page_num, line_on_page = line_layout[line_idx]

            while page_num >= len(pages):
                pages.append({'svg_paths': [], 'polylines': []})

            if line_data['raw_w'] < 1 or line_data['raw_h'] < 1:
                continue

            coords = line_data['coords']
            x_min = line_data['x_min']
            y_min = line_data['y_min']
            raw_w = line_data['raw_w']

            # Use uniform scale, but clamp if line would overflow page width
            scale = uniform_scale
            if raw_w * scale > usable_w:
                scale = usable_w / raw_w

            x_offset = self.margins['left']
            y_offset = self.margins['top'] + (line_on_page + 1) * base_line_height

            path_parts = []
            flat_points = []
            prev_eos = 1.0

            for x, y, eos in zip(coords[:, 0], coords[:, 1], coords[:, 2]):
                px = x_offset + (x - x_min) * scale
                py = y_offset + (y - y_min) * scale

                if prev_eos == 1.0:
                    path_parts.append(f'M{px:.2f},{py:.2f}')
                    if flat_points:
                        pages[page_num]['polylines'].append(flat_points)
                        flat_points = []
                else:
                    path_parts.append(f'L{px:.2f},{py:.2f}')

                flat_points.append((px, py))
                prev_eos = eos

            if flat_points:
                pages[page_num]['polylines'].append(flat_points)

            pages[page_num]['svg_paths'].append(' '.join(path_parts))

        return pages, A4_WIDTH, A4_HEIGHT


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


def generate_pdf(pages, output_path, color='#1a1a2e', stroke_width=0.4):
    """Generate a multi-page PDF file with the handwriting paths.

    pages: list of dicts with 'svg_paths' key (one per page)
    Produces a minimal valid PDF directly — no external libraries needed.
    """
    mm2pt = 72.0 / 25.4
    page_w = A4_WIDTH * mm2pt
    page_h = A4_HEIGHT * mm2pt

    c = color.lstrip('#')
    r, g, b = int(c[0:2], 16) / 255, int(c[2:4], 16) / 255, int(c[4:6], 16) / 255

    # Build content stream for each page
    page_streams = []
    for page in pages:
        stream_parts = [
            f'{r:.4f} {g:.4f} {b:.4f} RG',
            f'{stroke_width * mm2pt:.4f} w',
            '1 J', '1 j',
        ]
        for path_d in page['svg_paths']:
            if not path_d:
                continue
            for part in path_d.split():
                if part.startswith('M'):
                    coords = part[1:].split(',')
                    x = float(coords[0]) * mm2pt
                    y = page_h - float(coords[1]) * mm2pt
                    stream_parts.append(f'{x:.4f} {y:.4f} m')
                elif part.startswith('L'):
                    coords = part[1:].split(',')
                    x = float(coords[0]) * mm2pt
                    y = page_h - float(coords[1]) * mm2pt
                    stream_parts.append(f'{x:.4f} {y:.4f} l')
            stream_parts.append('S')
        page_streams.append('\n'.join(stream_parts).encode('latin-1'))

    num_pages = len(page_streams)

    # PDF objects:
    # 1 = Catalog, 2 = Pages
    # Then for each page: Page obj + Stream obj (ids 3,4 then 5,6 etc.)
    objects = []

    # Object 1: Catalog
    objects.append(b'1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj')

    # Object 2: Pages — kids are page objects at ids 3, 5, 7, ...
    kids = ' '.join(f'{3 + i * 2} 0 R' for i in range(num_pages))
    objects.append(
        f'2 0 obj\n<< /Type /Pages /Kids [{kids}] /Count {num_pages} >>\nendobj'
        .encode('latin-1')
    )

    # Page + stream object pairs
    for i, stream_bytes in enumerate(page_streams):
        page_obj_id = 3 + i * 2
        stream_obj_id = 4 + i * 2
        objects.append(
            f'{page_obj_id} 0 obj\n<< /Type /Page /Parent 2 0 R '
            f'/MediaBox [0 0 {page_w:.4f} {page_h:.4f}] '
            f'/Contents {stream_obj_id} 0 R >>\nendobj'.encode('latin-1')
        )
        objects.append(
            f'{stream_obj_id} 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n'
            .encode('latin-1')
            + stream_bytes
            + b'\nendstream\nendobj'
        )

    with open(output_path, 'wb') as f:
        f.write(b'%PDF-1.4\n')
        offsets = []
        for obj in objects:
            offsets.append(f.tell())
            f.write(obj)
            f.write(b'\n')
        xref_offset = f.tell()
        f.write(f'xref\n0 {len(objects) + 1}\n'.encode('latin-1'))
        f.write(b'0000000000 65535 f \n')
        for off in offsets:
            f.write(f'{off:010d} 00000 n \n'.encode('latin-1'))
        f.write(
            f'trailer\n<< /Size {len(objects) + 1} /Root 1 0 R >>\n'
            f'startxref\n{xref_offset}\n%%EOF\n'.encode('latin-1')
        )


def _svg_path_to_lac_path(svg_path_d):
    """Convert SVG path d-string from engine format to .lac format.

    Engine format:  M23.23,245.75 L23.23,245.64
    LAC format:     M 23.230000 245.750000 L 23.230000 245.640000
    """
    parts = svg_path_d.split()
    result = []
    for part in parts:
        if part.startswith('M'):
            coords = part[1:].split(',')
            result.append(f'M {float(coords[0]):.6f} {float(coords[1]):.6f}')
        elif part.startswith('L'):
            coords = part[1:].split(',')
            result.append(f'L {float(coords[0]):.6f} {float(coords[1]):.6f}')
        else:
            result.append(part)
    return ' '.join(result)


def generate_lac(svg_paths, output_path):
    """Generate a .lac file (Bambu Suite project) for the H2D pen plotter.

    svg_paths: list of SVG path d-strings from HandwritingRenderer.render()
    output_path: file path to write the .lac file
    """
    import io
    import json
    import struct
    import zipfile
    import zlib

    if not svg_paths:
        raise ValueError("No paths to export")

    # Object ID scheme: group_id=16, paths start at 18, increment by 3
    # Additional IDs: batch=10, plate=14
    group_id = 16
    first_path_id = 18
    path_ids = [first_path_id + i * 3 for i in range(len(svg_paths))]

    # Build 2dmodel.json
    obj_list = []
    for i, (path_d, obj_id) in enumerate(zip(svg_paths, path_ids)):
        obj_list.append({
            "color": "0 0 0 255",
            "flags": ["FreeAspectRatio"],
            "is_closed": False,
            "name": f"stroke {i + 1}",
            "obj_id": obj_id,
            "path_data": _svg_path_to_lac_path(path_d),
            "toolhead_setting": "Fine Point 0.3mm",
            "type": "PathObject",
        })

    # Group object containing all paths
    obj_list.append({
        "components": [{"obj_id": pid, "transform": "1 0 0 1 0 0"} for pid in path_ids],
        "flags": ["FreeAspectRatio"],
        "name": "handwriting",
        "obj_id": group_id,
        "type": "AttachedGroup",
    })

    model = {
        "Application": "Bambu Suite",
        "FileVersion": "01.02.01.00",
        "canvas_list": [{
            "components": [{"obj_id": group_id, "transform": "1 0 0 1 70.5 22.7"}],
            "index": 1,
            "name": "",
            "obj_list": obj_list,
            "type_count": {},
        }],
    }

    # Build project_settings.json
    object_settings = [{"obj_id": pid, "process_type": "KCPenDraw"} for pid in path_ids]

    project_settings = {
        "canvas_settings": [{
            "index": 1,
            "making_batch_list": [{
                "auto_arranged": True,
                "batch_settings": {"classVersion": 3, "processing_mode": "PLANE"},
                "making_plate_list": [{
                    "components": [{
                        "obj_id": group_id,
                        "transform": "2.22045e-16 1 -1 2.22045e-16 297.5 0.4999",
                    }],
                    "name": "",
                    "obj_id": 14,
                    "plate_mirror": False,
                    "plate_settings": {"classVersion": 3},
                }],
                "material_id": "",
                "material_name": "",
                "material_settings_name": "",
                "name": "",
                "obj_id": 10,
                "process_category": 4,
            }],
            "object_settings": object_settings,
        }],
        "project_settings": {
            "classVersion": 3,
            "machine_settings_name": "Bambu Lab H2D-10W",
            "version": None,
        },
    }

    entry = {"Application": "Bambu Suite", "FileVersion": "01.02.01.00"}

    content_types = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">\n'
        ' <Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>\n'
        ' <Default Extension="model" ContentType="application/vnd.ms-package.3dmanufacturing-3dmodel+xml"/>\n'
        ' <Default Extension="png" ContentType="image/png"/>\n'
        ' <Default Extension="gcode" ContentType="text/x.gcode"/>\n'
        '</Types>\n'
    )

    rels = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">\n'
        ' <Relationship Target="/2D/design_thumbnail.png" Id="rel-1" '
        'Type="http://schemas.openxmlformats.org/package/2006/relationships/metadata/thumbnail"/>\n'
        '</Relationships>\n'
    )

    # Minimal 1x1 white PNG for thumbnails
    def _minimal_png():
        """Create a minimal 1x1 white PNG."""
        def _chunk(chunk_type, data):
            c = chunk_type + data
            return struct.pack('>I', len(data)) + c + struct.pack('>I', zlib.crc32(c) & 0xffffffff)

        sig = b'\x89PNG\r\n\x1a\n'
        ihdr = _chunk(b'IHDR', struct.pack('>IIBBBBB', 1, 1, 8, 2, 0, 0, 0))
        raw = zlib.compress(b'\x00\xff\xff\xff')
        idat = _chunk(b'IDAT', raw)
        iend = _chunk(b'IEND', b'')
        return sig + ihdr + idat + iend

    # Machine config (H2D-10W) — load from bundled reference or use embedded
    config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'h2d_config.json')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            machine_config = f.read()
    else:
        machine_config = json.dumps(_H2D_DEFAULT_CONFIG, indent=4)

    # making_result.json — compute bounding polygon from paths
    all_points = []
    for path_d in svg_paths:
        for part in path_d.split():
            if part.startswith(('M', 'L')):
                coords = part[1:].split(',')
                all_points.append((float(coords[0]), float(coords[1])))
    if all_points:
        xs = [p[0] for p in all_points]
        ys = [p[1] for p in all_points]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        polygon = [
            {"x": int(x_min), "y": int(y_min)},
            {"x": int(x_max), "y": int(y_min)},
            {"x": int(x_max), "y": int(y_max)},
            {"x": int(x_min), "y": int(y_max)},
        ]
    else:
        polygon = []

    making_result = {
        "canvas_list": [{
            "index": 1,
            "making_batch_list": [{
                "index": 1,
                "making_plate_list": [{
                    "index": 1,
                    "polygon_points": polygon,
                    "prediction": 0,
                    "process_mode_list": [],
                    "thumbnail": "c1_b1_p1.png",
                    "toolhead_list": [],
                }],
                "material_config": ".config",
                "material_id": "",
                "material_name": "",
                "material_thumbnail": ".png",
                "process_category": 4,
                "processing_mode": 1,
            }],
        }],
    }

    thumbnail = _minimal_png()

    with zipfile.ZipFile(output_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('[Content_Types].xml', content_types)
        zf.writestr('_rels/.rels', rels)
        zf.writestr('2D/entry.json', json.dumps(entry, indent=4))
        zf.writestr('2D/2dmodel.json', json.dumps(model, indent=4))
        zf.writestr('2D/design_thumbnail.png', thumbnail)
        zf.writestr('Metadata2D/project_settings.json', json.dumps(project_settings, indent=4))
        zf.writestr('Metadata2D/Bambu Lab H2D-10W.config', machine_config)
        zf.writestr('Metadata2D/making_result.json', json.dumps(making_result, indent=4))
        zf.writestr('Metadata2D/c1_b1_p1.png', thumbnail)
        zf.writestr('Metadata2D/c1_b1_p1_small.png', thumbnail)
        zf.writestr('Metadata2D/full_c1_b1_p1.png', thumbnail)
        zf.writestr('Metadata2D/full_c1_b1_p1_small.png', thumbnail)


# Default H2D-10W machine config
_H2D_DEFAULT_CONFIG = {
    "_4axis_dir": "1x0",
    "_4axis_height": 170,
    "_4axis_max_length": 228,
    "_4axis_radius": 50,
    "_4axis_split_tolerance": 0.1,
    "_4axis_start": "113x135",
    "ac_end_gcode": "; active cut end gcode\n",
    "ac_start_gcode": "; active cut start gcode\n",
    "bed_area": ["0x0", "350x0", "350x320", "0x320"],
    "bezier_for_4_axis": True,
    "bezier_travel": False,
    "build_plate_area": ["8x-18.2", "343x-18.2", "343x328.3", "8x328.3"],
    "classVersion": 3,
    "current_area_height": 0,
    "current_area_offset_x": 0,
    "current_area_offset_y": 0,
    "current_area_width": 0,
    "cutting_high_precision": False,
    "draw_end_gcode": ";=== H2D draw end ===\nG91\nG380 S2 Z100 F1200\nG90\nG0 X200 Y260 F6000\nM480 S0\nM481 S0\n",
    "draw_start_gcode": ";=== H2D brush start_gcode ===\nM561 P0 U50\nM561 P1 U80\nM561 P2 U80\nM561 P3 U40\nM446 S0\nG90\nG0 X160 F4000\nG0 Y220 Z160 F2000\nM400\nM1006 S1\nM1006 L99 M99 N99\nM1006 A37 B10 L100 C37 D10 M40 E0 F10 N60\nM1006 A0 B10 L100 C0 D10 M40 E0 F10 N60\nM1006 A41 B10 L100 C41 D10 M40 E0 F10 N60\nM1006 W\nM400\nM1011 S[blade_idx]\nM1014 S[blade_idx]\nM400\nM1006 S1\nM1006 L99 M99 N99\nM1006 A53 B9 L99 C53 D9 M99 E53 F9 N99\nM1006 A56 B9 L99 C56 D9 M99 E56 F9 N99\nM1006 A61 B9 L99 C61 D9 M99 E61 F9 N99\nM1006 A53 B9 L99 C53 D9 M99 E53 F9 N99\nM1006 A56 B9 L99 C56 D9 M99 E56 F9 N99\nM1006 A61 B18 L99 C61 D18 M99 E61 F18 N99\nM1006 W\nM400 S1\nM400 P200\nM972 S16 P0 C0\nM480 S1\nG55\nG0 X160 Y160 Z160 F4000\nT1500\n",
    "drawing_cali_plate_area": ["-11x25", "-6x25", "-6x35", "-11x35"],
    "drawing_mat_cali_zero": False,
    "drawing_travel_speed": 100,
    "drawing_travel_z": 3,
    "electric_current": 0.4,
    "fire_power_threshold": 36,
    "fire_speed_threshold": 10,
    "from": "project",
    "global_end_gcode": ";=== H2D global end ===\nM400\nM1006 S1\nM1006 L99 M99 N99\nM1006 A53 B10 L99 C53 D10 M99 E53 F10 N99\nM1006 A57 B10 L99 C57 D10 M99 E57 F10 N99\nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0\nM1006 A53 B10 L99 C53 D10 M99 E53 F10 N99\nM1006 A57 B10 L99 C57 D10 M99 E57 F10 N99\nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0\nM1006 A48 B10 L99 C48 D10 M99 E48 F10 N99\nM1006 A0 B15 L0 C0 D15 M0 E0 F15 N0\nM1006 A60 B10 L99 C60 D10 M99 E60 F10 N99\nM1006 W\nM400\n",
    "global_start_gcode": ";=== H2D global start ===\nM483\n{if has_print_then_process}\nM482 S1\nM482 S0\n{endif}\nG91\nG380 S2 Z20 F1200\nG90\nG90\nG0 X160 F4000\nG0 Y220 F6000\nG0 X175 Y288.4 F6000\nM972 S24 P0\nM400 P200\nM1002 gcode_claim_action : 44\nM972 S9 P0 T10000\nM1002 gcode_claim_action : 52\nM400 P200\nM972 S10 P0 C1\nM1002 gcode_claim_action : 0\nM1024\n",
    "instantiation": True,
    "is_power_reduction_from_z": True,
    "is_support_arc": True,
    "is_support_bezier": True,
    "is_support_z": True,
    "kc_end_gcode": ";=== H2D cutting end_gcode ===\nG91\nG380 S2 Z250 F1200\nG90\nG0 X200 Y300 F6000\nM480 S0\nM481 S0",
    "kc_knife_types": ["Fine pointed", "Perforating", "Pen"],
    "kc_start_gcode": ";=== H2D passive cutting start_gcode ===\nM561 P0 U50\nM561 P1 U80\nM561 P2 U80\nM561 P3 U45\nM446 S0\nG90\nG0 X160 F4000\nG0 Y220 Z160 F2000\nM400\nM1006 S1\nM1006 L99 M99 N99\nM1006 A37 B10 L100 C37 D10 M40 E0 F10 N60\nM1006 A0 B10 L100 C0 D10 M40 E0 F10 N60\nM1006 A41 B10 L100 C41 D10 M40 E0 F10 N60\nM1006 W\nM400\nM1011 S[blade_idx]\nM1014 S[blade_idx]\nM400\nM1006 S1\nM1006 L99 M99 N99\nM1006 A53 B9 L99 C53 D9 M99 E53 F9 N99\nM1006 A56 B9 L99 C56 D9 M99 E56 F9 N99\nM1006 A61 B9 L99 C61 D9 M99 E61 F9 N99\nM1006 A53 B9 L99 C53 D9 M99 E53 F9 N99\nM1006 A56 B9 L99 C56 D9 M99 E56 F9 N99\nM1006 A61 B18 L99 C61 D18 M99 E61 F18 N99\nM1006 W\nM400 S1\nG90\nG0 X175 Y160 F2000\nM400 P200\nM972 S16 P0 C0\nM480 S1\nG55\nG0 X160 Y160 Z60 F4000\nT1000\nT1300\n",
    "kc_working_area": ["25.5x6.3", "325.5x6.3", "325.5x291.3", "25.5x291.3"],
    "laser_end_gcode": "",
    "laser_max_z_area": 2,
    "laser_module_mesh": [],
    "laser_power": 10,
    "laser_spot_height": 0.03,
    "laser_spot_width": 0.14,
    "laser_start_gcode": "",
    "laser_working_area": ["20.5x18.3", "330.5x18.3", "330.5x288.3", "20.5x288.3"],
    "machine_max_acceleration_u": 100,
    "machine_max_acceleration_x": 6000,
    "machine_max_acceleration_y": 6000,
    "machine_max_acceleration_z": 500,
    "machine_max_jerk_u": 0.5,
    "machine_max_jerk_x": 9,
    "machine_max_jerk_y": 9,
    "machine_max_jerk_z": 3,
    "machine_max_speed_u": 8,
    "machine_max_speed_x": 1000,
    "machine_max_speed_y": 1000,
    "machine_max_speed_z": 25,
    "machine_speed_comfortable": 60,
    "machine_speed_travel": 300,
    "measure_height_area": ["69x50", "330.5x50", "330.5x283.3", "69x283.3"],
    "min_direction_change": False,
    "name": "Bambu Lab H2D-10W",
    "optimize_tip_offset": True,
    "origin_position": "LEFT_BOTTOM",
    "pen_offset": [0, 0, 0, 30],
    "plate_turning_area": ["-16x10", "-6x10", "-6x25", "-16x25"],
    "pressure_mapping": [],
    "printer_model": "Bambu Lab H2D",
    "printer_variant": 10,
    "ptc_offset": [0, 0],
    "s_value": 1000,
    "surface_z_delayed": 0.5,
    "travel_pressure": -5,
    "travel_z": 3,
    "type": "machine",
    "us_end_gcode": "; ultra sound cut end gcode\n",
    "us_start_gcode": "; ultra sound cut start gcode\n",
    "use_sound_first_blade": True,
    "using_R_for_G2G3": False,
    "using_Z_for_pressure": True,
    "vendor": "BBL",
    "working_z_base": 0,
}
