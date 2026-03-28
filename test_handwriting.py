#!/usr/bin/env python3
"""Visual test suite for handwriting synthesis.

Generates high-resolution cropped PNG images and an HTML report to evaluate
model behavior across different texts, styles, biases, and edge cases.

Usage:
    python test_handwriting.py [--output-dir test_output] [--workers 4]
"""

import os
import sys
import argparse
import multiprocessing
from datetime import datetime
from functools import partial

# Add synth to path
_synth_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'synth')
if _synth_dir not in sys.path:
    sys.path.insert(0, _synth_dir)


def _ensure_cairo():
    """Ensure Homebrew cairo is discoverable on macOS."""
    import platform
    if platform.system() == 'Darwin':
        brew_lib = '/opt/homebrew/opt/cairo/lib'
        if os.path.isdir(brew_lib):
            os.environ.setdefault('DYLD_LIBRARY_PATH', brew_lib)


def crop_to_content(image_path, padding=40):
    """Crop a PNG to its non-white content with padding."""
    from PIL import Image, ImageChops

    img = Image.open(image_path).convert('RGB')
    bg = Image.new('RGB', img.size, (255, 255, 255))
    diff = ImageChops.difference(img, bg)
    bbox = diff.getbbox()
    if bbox:
        x0 = max(0, bbox[0] - padding)
        y0 = max(0, bbox[1] - padding)
        x1 = min(img.width, bbox[2] + padding)
        y1 = min(img.height, bbox[3] + padding)
        img.crop((x0, y0, x1, y1)).save(image_path)


def _run_single_test(test, output_dir):
    """Run a single test case in a worker process.

    Each worker lazily loads the model on first use (via the singleton).
    """
    _ensure_cairo()
    import cairosvg
    from engine import HandwritingRenderer, generate_svg

    name = test['name']
    text = test['text']
    png_path = test['png']
    kwargs = test['params']

    full_path = os.path.join(output_dir, png_path)
    renderer = HandwritingRenderer(
        style=kwargs.get('style', 1),
        bias=kwargs.get('bias', 0.8),
        stroke_width=kwargs.get('stroke_width', 0.25),
        color=kwargs.get('color', '#1a1a2e'),
        scale=kwargs.get('scale', 1.0),
        line_spacing=kwargs.get('line_spacing', 1.0),
        max_chars_per_line=kwargs.get('max_chars_per_line', 60),
    )
    pages, pw, ph = renderer.render(text)
    svg = generate_svg(pages[0]['svg_paths'], renderer.color, renderer.stroke_width)
    cairosvg.svg2png(bytestring=svg.encode('utf-8'),
                     write_to=full_path, scale=3)
    crop_to_content(full_path)
    return name


def build_test_list():
    """Build the list of all test case definitions (no execution)."""
    tests = []

    def add(name, text, png_path, group=None, **kwargs):
        tests.append({'name': name, 'text': text, 'png': png_path,
                      'group': group, 'params': kwargs})

    # ── 1. Word rendering ──
    g = 'words'
    add('Common words', 'The quick brown fox jumps over the lazy dog.',
        'words_common.png', group=g)
    add('Long words', 'Antidisestablishmentarianism and pseudopseudohypoparathyroidism',
        'words_long.png', group=g)
    add('Short words', 'I am a go to be it on up so no',
        'words_short.png', group=g)
    add('Numbers', 'There are 42 cats and 7 dogs in 3 houses.',
        'words_numbers.png', group=g)
    add('Punctuation', 'Hello! How are you? "Fine," she said. It\'s great.',
        'words_punctuation.png', group=g)
    add('Repeated letters', 'aaa bbb ccc ddd eee fff ggg hhh',
        'words_repeated.png', group=g)
    add('Capital letters', 'ABCDEFGHIJKLMNOPRSTUVWY',
        'words_capitals.png', group=g)
    add('Mixed case', 'Hello World Test Case Mixed Letters',
        'words_mixed_case.png', group=g)

    # ── 2. Line wrapping ──
    g = 'wrap'
    long_text = ('This is a longer piece of text that should wrap across '
                 'multiple lines to test how the system handles line breaking '
                 'and word wrapping at different widths.')
    add('Wrap at 40 chars', long_text, 'wrap_40.png', group=g,
        max_chars_per_line=40)
    add('Wrap at 60 chars', long_text, 'wrap_60.png', group=g,
        max_chars_per_line=60)
    add('Wrap at 75 chars', long_text, 'wrap_75.png', group=g,
        max_chars_per_line=75)
    tricky = 'We went to the supermarket yesterday afternoon and bought everything we needed.'
    add('Wrap tricky 30', tricky, 'wrap_tricky_30.png', group=g,
        max_chars_per_line=30)
    add('Wrap tricky 50', tricky, 'wrap_tricky_50.png', group=g,
        max_chars_per_line=50)

    # ── 3. Multi-line and paragraphs ──
    g = 'multiline'
    add('Two paragraphs',
        'First paragraph with some text.\n\nSecond paragraph here.',
        'multiline_paragraphs.png', group=g)
    add('Three lines',
        'Line one of the text.\nLine two continues here.\nLine three ends it.',
        'multiline_three.png', group=g)
    add('Short then long',
        'Hi.\nThis is a much longer second line that tests consistent sizing.',
        'multiline_short_long.png', group=g)
    add('Long then short',
        'This is a long first line that should be the same size as the next.\nBye.',
        'multiline_long_short.png', group=g)

    # ── 4. Edge cases ──
    g = 'edge'
    add('Single character', 'A', 'edge_single_char.png', group=g)
    add('Single word', 'Handwriting', 'edge_single_word.png', group=g)
    add('All lowercase', 'abcdefghijklmnopqrstuvwxyz',
        'edge_lowercase.png', group=g)

    # ── 5. Style comparison ──
    style_text = 'The quick brown fox jumps over the lazy dog.'
    for s in range(13):
        add(f'Style {s}', style_text, f'style_{s}.png', group='style', style=s)

    # ── 6. Bias sweep ──
    bias_text = 'Pack my box with five dozen liquor jugs.'
    for b in [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]:
        add(f'Bias {b:.2f}', bias_text, f'bias_{b:.2f}.png',
            group='bias', bias=b, style=1)

    # ── 7. Real-world text (Apple - Dutch/English mix) ──
    apple_text = (
        "Apple is vooral bekend van zijn reeks Apple "
        "Macintosh-computers, de iPod (muziekspeler), de iPhone "
        "(smartphone) en de iPad (tablet). Naast hardware ontwikkelt "
        "het zelf ook veel software voor het eigen macOS, maar ook "
        "voor Windows. Eind jaren zeventig is Apple bekend geworden "
        "met de Apple II, een van de eerste succesvolle personal "
        "computers ter wereld. De Apple II speelde een essenti\u00eble "
        "rol bij de opkomst van de pc-markt. Aan deze reeks van "
        "kleine successen kwam echter een abrupt einde door mogelijk "
        "bankroet. In 1997 stond het bedrijf aan de rand van de "
        "financi\u00eble afgrond, maar met een investering van 160 "
        "miljoen dollar (door Microsoft) is het bedrijf weer op de "
        "been geholpen. Apple's product lineup includes portable and "
        "home hardware like the iPhone, iPad, Apple Watch, Mac, "
        "Apple Vision Pro, AirPods, and Apple TV; several in-house "
        "operating systems such as iOS, iPadOS, and macOS; and "
        "various software and services including Apple Pay and "
        "iCloud, as well as multimedia streaming services like Apple "
        "Music and Apple TV. Since 2011, Apple has for the most part "
        "been the world's largest company by market capitalization, "
        "and, as of 2024, is the largest manufacturing company by "
        "revenue, the fourth-largest personal computer vendor, the "
        "largest vendor of tablet computers, and the largest vendor "
        "of mobile phones. Apple became the first publicly traded US "
        "company to be valued at over 1 trillion in 2018, and, as "
        "of October 2025, is valued at just over 4 trillion."
    )
    for s in range(13):
        add(f'Apple style {s}', apple_text, f'apple_style_{s}.png',
            group='apple', style=s, max_chars_per_line=60)

    # ── 8. Scale and spacing ──
    scale_text = 'Scale and spacing test.'
    add('Scale 0.7', scale_text, 'scale_0.7.png', group='scale', scale=0.7)
    add('Scale 1.0', scale_text, 'scale_1.0.png', group='scale', scale=1.0)
    add('Scale 1.5', scale_text, 'scale_1.5.png', group='scale', scale=1.5)
    add('Line spacing 0.8', scale_text + '\nSecond line.',
        'spacing_0.8.png', group='spacing', line_spacing=0.8)
    add('Line spacing 1.5', scale_text + '\nSecond line.',
        'spacing_1.5.png', group='spacing', line_spacing=1.5)

    return tests


def generate_html_report(results, output_dir):
    """Generate an HTML report with all test images."""
    html_path = os.path.join(output_dir, 'report.html')
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    seen = []
    groups = {}
    for r in results:
        g = r['group']
        if g not in groups:
            seen.append(g)
            groups[g] = []
        groups[g].append(r)

    group_titles = {
        'words': 'Word Rendering',
        'wrap': 'Line Wrapping',
        'multiline': 'Multi-line & Paragraphs',
        'edge': 'Edge Cases',
        'style': 'Style Comparison (0-12)',
        'bias': 'Bias Sweep (neatness)',
        'apple': 'Real-world Text (Apple - Dutch/English)',
        'scale': 'Scale',
        'spacing': 'Line Spacing',
    }

    html = [
        '<!DOCTYPE html>',
        '<html><head><meta charset="utf-8">',
        '<title>Handwriting Test Report</title>',
        '<style>',
        'body { font-family: -apple-system, sans-serif; max-width: 1400px; margin: 0 auto; padding: 20px; background: #f5f5f5; }',
        'h1 { color: #333; }',
        'h2 { color: #555; margin-top: 40px; border-bottom: 2px solid #ddd; padding-bottom: 8px; }',
        '.test-card { background: white; border-radius: 8px; margin: 12px 0; padding: 16px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }',
        '.test-card h3 { margin: 0 0 4px 0; color: #333; font-size: 14px; }',
        '.test-card .text { color: #888; font-size: 12px; margin-bottom: 8px; font-style: italic; }',
        '.test-card .params { color: #aaa; font-size: 11px; margin-bottom: 8px; }',
        '.test-card img { max-width: 100%; border: 1px solid #eee; border-radius: 4px; }',
        '.grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(500px, 1fr)); gap: 12px; }',
        '</style></head><body>',
        f'<h1>Handwriting Synthesis Test Report</h1>',
        f'<p style="color: #888;">Generated: {timestamp}</p>',
    ]

    for group_key in seen:
        items = groups[group_key]
        title = group_titles.get(group_key, group_key.title())
        html.append(f'<h2>{title}</h2>')
        html.append('<div class="grid">')
        for r in items:
            params_str = ', '.join(f'{k}={v}' for k, v in r['params'].items()) if r['params'] else 'defaults'
            html.append(f'<div class="test-card">')
            html.append(f'  <h3>{r["name"]}</h3>')
            html.append(f'  <div class="text">"{r["text"][:80]}{"..." if len(r["text"]) > 80 else ""}"</div>')
            html.append(f'  <div class="params">{params_str}</div>')
            html.append(f'  <img src="{r["png"]}" alt="{r["name"]}">')
            html.append(f'</div>')
        html.append('</div>')

    html.append('</body></html>')

    with open(html_path, 'w') as f:
        f.write('\n'.join(html))
    print(f'HTML report: {html_path}')


def main():
    parser = argparse.ArgumentParser(description='Visual handwriting test suite')
    parser.add_argument('--output-dir', default='test_output',
                        help='Output directory for test results')
    parser.add_argument('--workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')
    args = parser.parse_args()

    num_workers = args.workers or multiprocessing.cpu_count()
    tests = build_test_list()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Running {len(tests)} tests with {num_workers} workers...')
    print(f'Output: {args.output_dir}/\n')

    worker_fn = partial(_run_single_test, output_dir=args.output_dir)
    completed = 0

    with multiprocessing.Pool(num_workers) as pool:
        for name in pool.imap_unordered(worker_fn, tests):
            completed += 1
            print(f'  [{completed}/{len(tests)}] {name}')

    generate_html_report(tests, args.output_dir)
    print(f'\nDone! {len(tests)} tests completed.')
    print(f'Open {args.output_dir}/report.html to view results.')


if __name__ == '__main__':
    main()
