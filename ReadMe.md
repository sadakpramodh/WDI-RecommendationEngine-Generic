# CCM Quality Bands - Data-Driven Process Parameter Analysis

A Streamlit-based application for analyzing Centrifugal Casting Machine (CCM) manufacturing data to identify optimal quality parameter bands (green/red zones) for pipe production.

## Overview

This application helps identify process parameter ranges that minimize pipe rejection rates by:
- Analyzing historical CCM production data
- Computing statistical control limits (LCL/UCL) for each parameter
- Visualizing rejection probabilities across parameter ranges
- Supporting visual defect-specific analysis
- Exporting actionable quality bands for operators

## Features

### 📊 Core Capabilities
- **Automatic CCM Detection**: Identifies CCM machine from filename (e.g., `CCM2_export.csv`)
- **YAML-Based Column Mapping**: Automatically standardizes column names using machine-specific configs
- **Multi-Tier Analysis**: Analyzes data at Class → DN → Global levels based on sample availability
- **Visual Defect Segmentation**: Separate analysis for different defect types
- **Interactive Visualizations**: Histograms with Gaussian and KDE distribution curves
- **Excel Export**: Download cumulative results with LCL/UCL limits

### 🎯 Statistical Methods
- **IQR-Based Outlier Filtering**: Removes extreme values before analysis
- **Adaptive Binning**: Uses quantile-based (qcut) or equal-width (cut) binning
- **Rejection Probability Mapping**: Calculates rejection rates per parameter bin
- **Multi-Fallback Strategy**: Ensures limits are computed even with sparse data

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup

1. **Clone the repository**
```bash
git clone <repository-url>
cd ccm-quality-bands
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

**Required packages:**
```
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
plotly>=5.17.0
pyyaml>=6.0
openpyxl>=3.1.0
xlsxwriter>=3.1.0
```

3. **Verify configuration files**
Ensure YAML config files exist in the `configs/` directory:
```
configs/
├── ccm2.yaml
├── ccm3.yaml
├── ...
└── ccm12.yaml
```

## Usage

### Running the Application

```bash
streamlit run generic_recommendation_engine.py
```

The app will open in your default browser at `http://localhost:8501`

### Step-by-Step Workflow

#### 1. **Upload CSV Data**
- Click "Browse files" in the sidebar
- Upload CCM export CSV file
- **Naming convention**: Include CCM number in filename (e.g., `CCM2_data.csv`)

#### 2. **Required Columns** (after YAML mapping)
The application expects these columns:
- `DN` - Pipe diameter (or `dia`)
- `Pipe_Class` - Pipe classification (K7, C40, etc.)
- `Rejected_Flag` - Binary rejection indicator (0=accepted, 1=rejected)
  - Alternatively, `Status` column with values "Green" or "Rejected"

#### 3. **Select Analysis Parameters**
- **DN Filter**: Choose pipe diameter
- **Pipe Class Filter**: Select class (K7+C40 merged automatically)
- **Visual Defect**: If available, select specific defects or "All Defects"
- **Parameters**: Select process parameters to analyze (e.g., `cooling_rpm`, `Socket_rpm`)

#### 4. **Review Results**

**Cumulative Table** (top section):
| Column | Description |
|--------|-------------|
| CCM | Machine identifier |
| DN | Pipe diameter |
| Pipe_Class | Pipe classification |
| Visual_Defect | Defect type analyzed |
| Parameter | Process parameter |
| LCL | Lower Control Limit (green band start) |
| UCL | Upper Control Limit (green band end) |
| Rejection % (Defect) | % of DN+Class pipes rejected with this defect |
| Rejection % (Band) | % of pipes rejected within [LCL, UCL] |

**Charts & Statistics** (bottom section):
- Interactive histograms with color-coded bins
- Green/Amber/Red zones based on rejection probability
- Overlay distribution curves (Gaussian & KDE)
- Per-parameter metrics cards

#### 5. **Download Results**
Click "📥 Download Cumulative Results (Excel)" to export all limits

## Configuration

### YAML Structure

Each `ccmX.yaml` file defines:

```yaml
extract:
  database: "Reports"
  table: "CCM2"
  cdc: "CCM Date"
  # ... connection details

load:
  table: "t_ccm2"
  load_type: "incremental_load"

transform:
  rename: "yes"
  rename_columns:
    # Source Column: Target Column
    "Dia": "DN"
    "PipeClass": "Pipe_Class"
    "cooling_rpm": "cooling_rpm"
    # ... 100+ parameter mappings
```

### Adding New CCM Configs

1. Create `configs/ccmX.yaml`
2. Define `rename_columns` mapping from source → standardized names
3. Ensure mapping includes: `DN`, `Pipe_Class`, and rejection indicator

## Methodology

### 1. **Data Tiering**
```
IF samples(DN + Pipe_Class) >= 30:
    Use Class-specific data
ELIF samples(DN) >= 30:
    Use DN-specific data
ELSE:
    Use Global dataset
```

### 2. **Outlier Removal**
- Calculate IQR (Q1, Q3) for parameter
- Remove values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]

### 3. **Binning Strategy**
```
TRY qcut (quantile-based):
    bins = sqrt(n_samples), max 50 bins
FALLBACK TO cut (equal-width):
    bins = min(50, n_unique_values)
FINAL FALLBACK:
    Use IQR band [Q1, Q3]
```

### 4. **Green Band Identification**

**Primary Method**: Contiguous low-rejection regions
```
FOR each parameter bin:
    IF rejection_probability <= class_threshold:
        Mark as candidate "green" bin
    
SELECT longest contiguous sequence of green bins
```

**Fallback Method**: Coverage-optimized band
```
FIND bin range [i, j] WHERE:
    - Coverage >= 30% of total samples
    - Rejection probability is minimized
```

### 5. **Color Classification**
- **Green**: Bin within [LCL, UCL] limits
- **Red**: Bin outside limits AND rejection_prob > threshold
- **Amber**: Bin outside limits BUT rejection_prob ≤ threshold

## Tunables

Adjust these constants in the code:

```python
MIN_SAMPLES_CLASS = 30      # Minimum samples for class-level analysis
TARGET_COVERAGE = 0.30      # Minimum coverage for fallback bands
MAX_BINS = 50               # Maximum bins for histogram
SMALL_SAMPLE_BAND = 0.50    # IQR fraction for small samples
```

## Interpretation Guide

### Metrics Card
- **Method**: Binning strategy used (qcut/cut/IQR-fallback)
- **Coverage %**: % of pipes within [LCL, UCL]
- **Rejection % (Band)**: Actual rejection rate in green band
  - ⚠️ **Target**: < class baseline rejection rate

### Chart Colors
- **Green Overlay**: Recommended operating range [LCL, UCL]
- **Green Bars**: Accepted pipes in bin
- **Orange Bars**: Rejected pipes in bin
- **Green/Amber/Red Shading**: Bin classification

### When to Act
🔴 **High Priority**: Parameters with >20% rejection in green band  
🟡 **Monitor**: Parameters with 10-20% rejection in green band  
🟢 **Acceptable**: Parameters with <10% rejection in green band

## Troubleshooting

### "Missing required columns"
- Check YAML mapping includes `DN`, `Pipe_Class`
- Verify CSV column names match YAML source names
- Ensure `Status` column exists if no `Rejected_Flag`

### "All-NaN/non-numeric" error
- Parameter contains text values or is entirely null
- Check source data for that parameter
- Remove non-numeric parameters from selection

### Empty green bands
- Rejection rate may be uniformly high
- Try relaxing `TARGET_COVERAGE` tunable
- Consider analyzing different DN/Class combination

## File Structure

```
.
├── generic_recommendation_engine.py  # Main Streamlit app
├── configs/
│   ├── ccm2.yaml                     # CCM2 column mappings
│   ├── ccm3.yaml                     # CCM3 column mappings
│   └── ...                           # Additional CCM configs
├── config.yaml                       # Database connection config
├── requirements.txt                  # Python dependencies
└── README.md                         # This file
```

## Technical Notes

### Class Merging
- K7 and C40 classes are automatically merged as "K7+C40"
- Rationale: Similar specifications, combined analysis improves sample size

### Rejection % Calculations
- **Rejection % (Defect)**: Uses ALL pipes in DN+Class as denominator
- **Rejection % (Band)**: Uses only pipes within [LCL, UCL] as denominator

### Distribution Curves
- **Gaussian**: Normal distribution fit (mean, std)
- **KDE**: Kernel Density Estimation with Silverman's rule
- Curves are **scaled** to histogram counts for visual comparison

## Contributing

To add support for new CCM machines:
1. Export data with consistent column structure
2. Create `configs/ccmX.yaml` with column mappings
3. Test with sample data
4. Update this README with any machine-specific notes

## License

[Add your license information]

## Support

For issues or questions:
- Create an issue in the repository
- Contact: [Your contact information]

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Compatible CCMs**: 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12