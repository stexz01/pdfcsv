<p align="center">
  <img src="assets/logo.svg" alt="PDFcsv Logo" width="120">
</p>

<h1 align="center">PDFcsv</h1>
<h3 align="center">Universal PDF to CSV Extractor</h3>

<p align="center">
  <img src="https://img.shields.io/badge/version-1.4.0-blue.svg" alt="Version">
  <img src="https://img.shields.io/badge/python-3.8+-green.svg" alt="Python">
  <img src="https://img.shields.io/badge/license-MIT-orange.svg" alt="License">
  <img src="https://img.shields.io/badge/platform-linux%20%7C%20macos%20%7C%20windows-lightgrey.svg" alt="Platform">
</p>

<p align="center">
  <b>Extract tabular data from any PDF with intelligent column detection and bank statement support</b>
</p>

<p align="center">
  <code>pdf-to-csv</code> · <code>pdf-to-excel</code> · <code>pdf-to-json</code> · <code>bank-statement-parser</code>
</p>

---

## Quick Start

```bash
pip3 install git+https://github.com/stexz01/pdfcsv.git
pdfcsv your_file.pdf
```

**Done!** All dependencies included. Works on Linux, macOS, and Windows.

---

## Demo

<p align="center">
  <img src="assets/demo.gif" alt="PDFcsv Demo" width="700">
</p>

<p align="center">
  <i>Interactive column selection with live preview</i>
</p>

---

## Installation

### Option 1: pip install (Recommended)

```bash
pip3 install git+https://github.com/stexz01/pdfcsv.git
```

### Option 2: Clone & Install

```bash
git clone https://github.com/stexz01/pdfcsv.git
cd pdfcsv
pip3 install -e .
```

### Option 3: Direct Script

```bash
curl -O https://raw.githubusercontent.com/stexz01/pdfcsv/main/pdfcsv.py
pip3 install pdfplumber openpyxl
python3 pdfcsv.py input.pdf
```

**Requirements:** Python 3.8+

---

## Usage

### Basic Commands

| Command | Description |
|---------|-------------|
| `pdfcsv file.pdf` | Interactive mode - select columns with arrow keys |
| `pdfcsv file.pdf --columns 6` | Extract rows with 6 columns |
| `pdfcsv file.pdf --analyze` | Show column structure analysis |
| `pdfcsv file.pdf -o out.csv` | Custom output filename |
| `pdfcsv file.pdf -f excel` | Export as Excel (.xlsx) |
| `pdfcsv file.pdf --silent` | Minimal output (scripting) |

### Output Formats

| Format | Flag | Extension |
|--------|------|-----------|
| CSV | `-f csv` | .csv |
| Excel | `-f excel` | .xlsx |
| JSON | `-f json` | .json |
| JSON Lines | `-f jsonl` | .jsonl |
| Markdown | `-f markdown` | .md |
| TSV | `-f tsv` | .tsv |

### Examples

```bash
# Bank statement to CSV
pdfcsv statement.pdf

# Invoice to Excel with specific columns
pdfcsv invoice.pdf --columns 5 -f excel -o invoice.xlsx

# Analyze structure first
pdfcsv report.pdf --analyze
pdfcsv report.pdf --columns 7

# Silent mode for scripts
pdfcsv data.pdf --columns 4 --silent
```

### All Options

```
pdfcsv <file.pdf> [options]

Options:
  --columns N, --column N    Extract rows with N columns
  --analyze                  Show column distribution
  --gap N                    Gap threshold in pixels (default: 5)
  -o, --output FILE          Output filename
  -f, --format FORMAT        Output format (csv/excel/json/jsonl/md/tsv)
  --silent                   Minimal output (only result or errors)
  -h, --help                 Show help
  -v, --version              Show version
```

---

## Features

| Feature | Description |
|---------|-------------|
| **Smart Column Detection** | Gap-based analysis using character X-positions |
| **Bank Statement Support** | Auto-fills empty debit/credit columns with `-` |
| **Interactive CLI** | Arrow-key navigation with live preview |
| **Auto-Select** | Single map option? Automatically selected |
| **Multi-Format Export** | CSV, Excel, JSON, Markdown, TSV |
| **15+ Languages** | Bank keyword detection in multiple languages |
| **Deduplication** | Removes duplicate rows and columns |
| **Encrypted PDF Handling** | Helpful error messages for protected files |

---

## Bank Statement Support

PDFcsv automatically detects bank statements and handles missing columns:

```
Raw PDF:                          Processed Output:
Date | Desc | 500.00 | 1000      Date | Desc | Debit | Credit | Balance
                                 01/01 | ATM  | 500   | -      | 1000
                                 01/02 | Dep  | -     | 200    | 1200
```

**Supported Languages:** English, Spanish, French, German, Portuguese, Italian, Dutch, Hindi, Arabic, Chinese, Japanese, Korean, Russian, Turkish

---

## Interactive Mode

Run without `--columns` to enter interactive selection:

```
  (3) Map found - Select the perfect one

  ──────────────────────────────────────────────────────────
  > (46 rows) 7 columns
    (12 rows) 4 columns
    (5 rows) 2 columns
  ──────────────────────────────────────────────────────────
  columns: Date | Chq No | Particulars | Debit | Credit | Balance | Init.
        1. 02-04-2023 | - | Pay/ONSPG202 | - | 9170.00 | 1266873.70 | 115
        2. 03-04-2023 | - | PAYMENTSSERV | - | 21000.00 | 1287873.70 | 248
  ──────────────────────────────────────────────────────────
  [up/down] move  [Enter] select  [q] quit
```

- **Single option?** Auto-selects without prompting
- Use **arrow keys** to navigate
- Press **Enter** to confirm
- Press **q** to cancel

---

## Troubleshooting

### Password-Protected PDF
```bash
# Remove protection first
qpdf --decrypt input.pdf unlocked.pdf
pdfcsv unlocked.pdf
```

### Wrong Column Count
```bash
# Analyze structure first
pdfcsv file.pdf --analyze

# Adjust gap threshold
pdfcsv file.pdf --gap 10  # Wider gaps
pdfcsv file.pdf --gap 3   # Tighter text
```

### No Matching Rows
```bash
# Check available column counts
pdfcsv file.pdf --analyze

# Extract specific count
pdfcsv file.pdf --columns 5
```

---

## How It Works

```
┌─────────────────────────────────────────────────────────────────────┐
│                         PDF DOCUMENT                                │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  1. CHARACTER EXTRACTION                                            │
│     └─ Extract all characters with X,Y coordinates (pdfplumber)     │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  2. LINE GROUPING                                                   │
│     └─ Group characters by Y position (same row = same Y)           │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  3. GAP-BASED COLUMN DETECTION                                      │
│     └─ Detect gaps between characters (gap > threshold = new col)   │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  4. STRUCTURE ANALYSIS                                              │
│     └─ Group rows by column count, find most common structure       │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  5. BANK STATEMENT PROCESSING (if detected)                         │
│     └─ Identify debit/credit columns, fill missing with '-'         │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│  6. OUTPUT                                                          │
│     └─ Export to CSV / Excel / JSON / Markdown / TSV                │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
pdfcsv/
├── pdfcsv.py             # Main CLI tool
├── banking_keywords.py   # Multi-language bank keywords
├── pyproject.toml        # Package configuration
├── requirements.txt      # Dependencies
├── LICENSE               # MIT License
├── README.md             # Documentation
└── assets/
    ├── logo.svg          # Project logo
    └── demo.gif          # CLI demo
```

---

## Contributing

Contributions are welcome! Here's how to get started:

### Development Setup

```bash
git clone https://github.com/stexz01/pdfcsv.git
cd pdfcsv
pip3 install -e ".[dev]"
```

### Areas for Contribution

- [ ] Additional bank statement formats
- [ ] More language keywords in `banking_keywords.py`
- [ ] Performance optimizations for large PDFs
- [ ] GUI interface (web or desktop)
- [ ] Test cases and CI/CD pipeline
- [ ] Documentation improvements

### Pull Request Process

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Test with various PDF types
5. Commit (`git commit -m 'Add amazing feature'`)
6. Push (`git push origin feature/amazing-feature`)
7. Open a Pull Request

---

## Changelog

### v1.4.0 (Current)
- Multiple output formats: CSV, TSV, JSON, JSONL, Markdown, Excel
- `--format` / `-f` flag for output type selection
- `--silent` flag for minimal output (scripting)
- `--column` alias for `--columns`
- Auto-select when only one map option exists
- All dependencies bundled (no extras needed)

### v1.3.x
- Interactive column selector with arrow keys
- Bank statement gap-filling with position analysis
- Multi-language keyword support
- Automatic deduplication
- Encrypted PDF handling

---

## License

MIT License - see [LICENSE](LICENSE) file.

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/stexz01">@stexz01</a>
</p>

<p align="center">
  <sub>
    <b>Keywords:</b> pdf to csv, pdf to excel, pdf to json, extract table from pdf,
    bank statement parser, pdf table extractor, python pdf parser, convert pdf to csv,
    pdf data extraction, tabular data extractor, financial pdf parser
  </sub>
</p>
