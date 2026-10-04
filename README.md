# Tobben Regular (v0.2)

A handmade, authentic display font compiled directly from physical ink-on-paper handwriting using an open-source Python computer vision and typography pipeline.

![Baseline Check](docs/baseline_check.png)

---

## 📦 What's in the Box?

- **`dist/Tobben-Regular.ttf`**: Desktop TrueType font (ready to install on Windows, macOS, and Linux).
- **`dist/Tobben-Regular.woff`**: Compressed Web Open Font Format (ready to deploy to websites).
- **`preview/index.html`**: Interactive live specimen and typewriter sandbox.
- **`src/source/tobben.jpg`**: Original high-resolution handwritten source sheet.
- **`src/build.py`**: Pure Python compiler pipeline that converts the source image into vector font tables.

---

## 🗂️ Character Coverage (71 Glyphs)

| Category | Glyphs Included |
| :--- | :--- |
| **Uppercase Letters** | `A B C D E F G H I J K L M N O P Q R S T U V W X Y Z` |
| **Scandinavian Vowels** | `Æ Ø Å` |
| **Lowercase Mapping** | `a`–`z`, `æ`, `ø`, `å` (automatically mapped to uppercase glyphs) |
| **Digits** | `0 1 2 3 4 5 6 7 8 9` |
| **Punctuation** | `, . - ! ? ' " : ; _` |
| **Enclosures & Slashes** | `( ) [ ] { } / \` |
| **Math & Symbols** | `+ = < > @ # % & $ \| * ~` |

---

## 🚀 Installation & Usage

### Desktop (Windows / macOS)
- **Windows**: Right-click [`dist/Tobben-Regular.ttf`](dist/Tobben-Regular.ttf) and select **Install** (or **Install for all users**).
- **macOS**: Double-click `dist/Tobben-Regular.ttf` and click **Install Font** in Font Book.

### Web (`@font-face`)
```css
@font-face {
  font-family: 'Tobben';
  src: url('dist/Tobben-Regular.woff') format('woff'),
       url('dist/Tobben-Regular.ttf') format('truetype');
  font-weight: normal;
  font-style: normal;
}

body {
  font-family: 'Tobben', sans-serif;
}
```

---

## 🛠️ Building From Source

### 1. Requirements
Install the required Python libraries using `pip`:
```bash
pip install -r requirements.txt
```

### 2. Compile the Font
Run the build script from the repository root:
```bash
python src/build.py
```
This reads `src/source/tobben.jpg`, extracts the contours, anchors glyphs to the baseline, and updates `dist/Tobben-Regular.ttf` and `dist/Tobben-Regular.woff`.

### 3. Test in the Browser
Open `preview/index.html` in any web browser to type test words and inspect the glyph specimen grid.

---

## 📐 Pipeline Architecture

```mermaid
flowchart TD
    A["Raw Photo<br/><b>src/source/tobben.jpg</b>"] -->|Adaptive Gaussian Thresholding| B["Binary Ink Mask<br/><i>Denoised pixel grid</i>"]
    B -->|Horizontal Projections & Connectivity| C["Line & Character Segmentation<br/><i>Multi-stroke clustering for dots & accents</i>"]
    C -->|cv2.findContours with RETR_CCOMP| D["Vector Contours<br/><i>2-level hierarchy: exterior loops vs interior holes</i>"]
    D -->|TrueType Winding & Coordinate Flip| E["Semantic Baseline Alignment<br/><i>Y=0 baseline snap, descenders, math axis centering</i>"]
    E -->|1000 Units Per Em Scaling| F["TrueType Glyph Contours<br/><i>TTGlyphPen line and curve instructions</i>"]
    F -->|FontTools FontBuilder| G["OpenType Table Assembly<br/><i>cmap, glyf, hmtx, head, hhea, OS/2, post</i>"]
    G --> H["Desktop Font<br/><b>dist/Tobben-Regular.ttf</b>"]
    G --> I["Web Font<br/><b>dist/Tobben-Regular.woff</b>"]

    classDef primary fill:#1e293b,stroke:#38bdf8,stroke-width:2px,color:#f8fafc;
    classDef output fill:#0f172a,stroke:#4ade80,stroke-width:2px,color:#f8fafc;
    class A,B,C,D,E,F,G primary;
    class H,I output;
```

![Segmentation Map](docs/segmentation_map.jpg)
