"""
=============================================================================
 TOBBEN FONT COMPILER PIPELINE
=============================================================================
Converts ink handwriting from `src/source/tobben.jpg` into TrueType (.ttf)
and Web Open Font Format (.woff) in the `dist/` directory.

Pipeline:
 1. Binarize high-res scan with adaptive Gaussian thresholding.
 2. Segment lines and cluster multi-stroke glyphs (!, ?, :, quotes, rings).
 3. Trace contours using 2-level hierarchy (outer loops vs internal cutouts).
 4. Enforce TrueType winding rules and snap glyphs to semantic baselines.
 5. Generate font tables (cmap, glyf, hmtx, head, hhea, OS/2, post).
 6. Export `dist/Tobben-Regular.ttf` and `dist/Tobben-Regular.woff`.
=============================================================================
"""

import os
import cv2
import numpy as np
from fontTools.fontBuilder import FontBuilder
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib import TTFont

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
SOURCE_IMG = os.path.join(BASE_DIR, 'src', 'source', 'tobben.jpg')
DIST_DIR = os.path.join(BASE_DIR, 'dist')
OUT_TTF = os.path.join(DIST_DIR, 'Tobben-Regular.ttf')
OUT_WOFF = os.path.join(DIST_DIR, 'Tobben-Regular.woff')

def polygon_signed_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += pts[i][0] * pts[j][1] - pts[j][0] * pts[i][1]
    return area / 2.0

def build():
    print(f"[1/4] Reading source image: {SOURCE_IMG}")
    if not os.path.exists(SOURCE_IMG):
        raise FileNotFoundError(f"Source image not found: {SOURCE_IMG}")

    img = cv2.imread(SOURCE_IMG)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 55, 16
    )
    kernel = np.ones((2, 2), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    # Line definitions: (top_y, bottom_y, glyph_names)
    BANDS = [
        (530, 775, list("ABCDEFGHIJKLMNOP")),
        (780, 990, ["Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "AE", "Oslash", "Aring"]),
        (995, 1215, ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "zero", "comma", "period", "hyphen", "exclam", "question"]),
        (1220, 1430, ["exclam_alt", "quotedbl", "numbersign", "percent", "ampersand", "slash", "backslash", "parenleft", "parenright", "bracketleft", "bracketright", "braceleft", "braceright", "equal", "plus"]),
        (1435, 1660, ["less", "greater", "at", "colon", "semicolon", "underscore", "dollar", "bar", "asterisk", "quotesingle", "asciitilde"])
    ]

    SCALE = 700.0 / 170.0
    SIDE_BEARING = 60

    glyf_dict = {}
    metrics_dict = {}
    glyph_names = ['.notdef', 'space']

    # .notdef fallback box
    pen = TTGlyphPen(None)
    pen.moveTo((50, 0))
    pen.lineTo((50, 700))
    pen.lineTo((450, 700))
    pen.lineTo((450, 0))
    pen.closePath()
    pen.moveTo((100, 50))
    pen.lineTo((400, 50))
    pen.lineTo((400, 650))
    pen.lineTo((100, 650))
    pen.closePath()
    glyf_dict['.notdef'] = pen.glyph()
    metrics_dict['.notdef'] = (500, 50)

    # space
    glyf_dict['space'] = TTGlyphPen(None).glyph()
    metrics_dict['space'] = (320, 0)

    print("[2/4] Vectorizing and anchoring glyphs to the baseline...")

    for line_idx, (y1, y2, expected_names) in enumerate(BANDS):
        mask = cleaned[y1:y2, 200:img.shape[1]-200]
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        comps = []
        for i in range(1, num_labels):
            x, y, w, h, area = stats[i]
            if area > 35:
                comps.append({'x1': x + 200, 'y1': y + y1, 'x2': x + 200 + w, 'y2': y + y1 + h, 'area': area})
                
        comps.sort(key=lambda c: c['x1'])
        
        merged = []
        for c in comps:
            if not merged:
                merged.append(dict(c))
                continue
            prev = merged[-1]
            x_overlap = min(prev['x2'], c['x2']) - max(prev['x1'], c['x1'])
            gap = c['x1'] - prev['x2']
            
            should_merge = False
            if x_overlap > -8 and (max(prev['x2'], c['x2']) - min(prev['x1'], c['x1']) < 140):
                should_merge = True
            elif gap < 18 and ((c['x2'] - c['x1'] < 35) or (prev['x2'] - prev['x1'] < 35)):
                should_merge = True
                
            if should_merge:
                prev['x1'] = min(prev['x1'], c['x1'])
                prev['y1'] = min(prev['y1'], c['y1'])
                prev['x2'] = max(prev['x2'], c['x2'])
                prev['y2'] = max(prev['y2'], c['y2'])
                prev['area'] += c['area']
            else:
                merged.append(dict(c))

        for j, box in enumerate(merged):
            if j >= len(expected_names):
                break
            gname = expected_names[j]
            if gname == 'exclam_alt':
                continue

            pad = 4
            crop_x1 = max(0, box['x1'] - pad)
            crop_y1 = max(0, box['y1'] - pad)
            crop_x2 = min(img.shape[1], box['x2'] + pad)
            crop_y2 = min(img.shape[0], box['y2'] + pad)
            char_crop = cleaned[crop_y1:crop_y2, crop_x1:crop_x2]
            
            contours, hierarchy = cv2.findContours(char_crop, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_TC89_KCOS)
            
            # Semantic baseline anchoring
            if gname in list("ABCDEFGHIJKLMNOPRSTUVWXYZ") + ["AE", "Oslash", "Aring"] + [
                "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine",
                "period", "exclam", "question", "colon", "dollar", "bar"
            ]:
                anchor_y_pixel = box['y2']
                target_font_y = 0
            elif gname == 'Q':
                anchor_y_pixel = box['y2']
                target_font_y = -70
            elif gname == 'comma':
                anchor_y_pixel = box['y1']
                target_font_y = 80
            elif gname == 'semicolon':
                anchor_y_pixel = box['y1']
                target_font_y = 350
            elif gname in ['quotedbl', 'quotesingle']:
                anchor_y_pixel = box['y1']
                target_font_y = 660
            elif gname == 'hyphen':
                anchor_y_pixel = (box['y1'] + box['y2']) / 2
                target_font_y = 270
            elif gname == 'underscore':
                anchor_y_pixel = box['y1']
                target_font_y = -30
            elif gname == 'asterisk':
                anchor_y_pixel = (box['y1'] + box['y2']) / 2
                target_font_y = 480
            elif gname == 'asciitilde':
                anchor_y_pixel = (box['y1'] + box['y2']) / 2
                target_font_y = 300
            elif gname in ['equal', 'plus', 'less', 'greater', 'at']:
                anchor_y_pixel = (box['y1'] + box['y2']) / 2
                target_font_y = 280
            elif gname in ['parenleft', 'parenright', 'bracketleft', 'bracketright', 'braceleft', 'braceright', 'slash', 'backslash']:
                anchor_y_pixel = (box['y1'] + box['y2']) / 2
                target_font_y = 270
            else:
                anchor_y_pixel = box['y2']
                target_font_y = 0

            pen = TTGlyphPen(None)
            char_min_x = box['x1']
            
            if hierarchy is not None:
                for ci, cnt in enumerate(contours):
                    if cv2.contourArea(cnt) < 15:
                        continue
                    
                    approx = cv2.approxPolyDP(cnt, epsilon=1.2, closed=True)
                    if len(approx) < 3:
                        continue
                    
                    pts = []
                    for pt in approx:
                        px = pt[0][0] + crop_x1
                        py = pt[0][1] + crop_y1
                        fx = int((px - char_min_x) * SCALE + SIDE_BEARING)
                        fy = int((anchor_y_pixel - py) * SCALE + target_font_y)
                        pts.append((fx, fy))
                    
                    is_hole = (hierarchy[0][ci][3] >= 0)
                    area = polygon_signed_area(pts)
                    
                    if not is_hole and area > 0:
                        pts.reverse()
                    elif is_hole and area < 0:
                        pts.reverse()
                        
                    pen.moveTo(pts[0])
                    for pt in pts[1:]:
                        pen.lineTo(pt)
                    pen.closePath()

            glyph = pen.glyph()
            glyf_dict[gname] = glyph
            glyph_names.append(gname)
            
            glyph_w = int((box['x2'] - box['x1']) * SCALE)
            advance_w = glyph_w + (SIDE_BEARING * 2)
            metrics_dict[gname] = (advance_w, SIDE_BEARING)

    print(f"[3/4] Compiling OpenType tables for {len(glyf_dict)} glyphs...")
    fb = FontBuilder(1000, isTTF=True)
    fb.setupGlyphOrder(glyph_names)

    GLYPH_TO_CHARS = {
        'A': ['A', 'a'], 'B': ['B', 'b'], 'C': ['C', 'c'], 'D': ['D', 'd'],
        'E': ['E', 'e'], 'F': ['F', 'f'], 'G': ['G', 'g'], 'H': ['H', 'h'],
        'I': ['I', 'i'], 'J': ['J', 'j'], 'K': ['K', 'k'], 'L': ['L', 'l'],
        'M': ['M', 'm'], 'N': ['N', 'n'], 'O': ['O', 'o'], 'P': ['P', 'p'],
        'Q': ['Q', 'q'], 'R': ['R', 'r'], 'S': ['S', 's'], 'T': ['T', 't'],
        'U': ['U', 'u'], 'V': ['V', 'v'], 'W': ['W', 'w'], 'X': ['X', 'x'],
        'Y': ['Y', 'y'], 'Z': ['Z', 'z'],
        'AE': ['\u00C6', '\u00E6'],
        'Oslash': ['\u00D8', '\u00F8'],
        'Aring': ['\u00C5', '\u00E5'],
        'zero': ['0'], 'one': ['1'], 'two': ['2'], 'three': ['3'], 'four': ['4'],
        'five': ['5'], 'six': ['6'], 'seven': ['7'], 'eight': ['8'], 'nine': ['9'],
        'comma': [','], 'period': ['.'], 'hyphen': ['-'],
        'exclam': ['!'], 'question': ['?'],
        'quotedbl': ['"'], 'numbersign': ['#'], 'percent': ['%'],
        'ampersand': ['&'], 'slash': ['/'], 'backslash': ['\\'],
        'parenleft': ['('], 'parenright': [')'],
        'bracketleft': ['['], 'bracketright': [']'],
        'braceleft': ['{'], 'braceright': ['}'],
        'equal': ['='], 'plus': ['+'],
        'less': ['<'], 'greater': ['>'], 'at': ['@'],
        'colon': [':'], 'semicolon': [';'], 'underscore': ['_'],
        'dollar': ['$'], 'bar': ['|'], 'asterisk': ['*'],
        'quotesingle': ["'"], 'asciitilde': ['~'],
        'space': [' ']
    }

    cmap = {}
    for gname, chars in GLYPH_TO_CHARS.items():
        if gname in glyf_dict:
            for ch in chars:
                cmap[ord(ch)] = gname

    fb.setupCharacterMap(cmap)
    fb.setupGlyf(glyf_dict)
    fb.setupHorizontalMetrics(metrics_dict)
    fb.setupHorizontalHeader(ascent=850, descent=-250)
    fb.setupNameTable({
        'familyName': 'Tobben',
        'styleName': 'Regular',
        'uniqueFontIdentifier': 'Tobben-Regular:2026:v0.2',
        'fullName': 'Tobben Regular',
        'psName': 'Tobben-Regular',
        'version': 'Version 0.200',
    })
    fb.setupOS2(
        sTypoAscender=850,
        sTypoDescender=-250,
        sTypoLineGap=100,
        usWinAscent=850,
        usWinDescent=250,
        sxHeight=500,
        sCapHeight=700
    )
    fb.setupPost()

    os.makedirs(DIST_DIR, exist_ok=True)
    fb.save(OUT_TTF)
    print(f"[4/4] Saved TTF: {OUT_TTF} ({os.path.getsize(OUT_TTF)} bytes)")

    # Export WOFF
    font = TTFont(OUT_TTF)
    font.flavor = 'woff'
    font.save(OUT_WOFF)
    print(f"      Saved WOFF: {OUT_WOFF} ({os.path.getsize(OUT_WOFF)} bytes)")
    print("\n BUILD COMPLETE! All formats generated successfully.")

if __name__ == '__main__':
    build()
