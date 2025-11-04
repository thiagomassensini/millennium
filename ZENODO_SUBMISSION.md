# Zenodo Submission Guide

## Quick Start

1. **Create Zenodo Account**: https://zenodo.org/signup/
2. **Connect GitHub**: https://zenodo.org/account/settings/github/
3. **Enable Repository**: Find "millennium" and flip the switch to ON
4. **Create Release**: GitHub → Releases → "Create a new release"
5. **Zenodo Auto-Generates DOI**: Within 10 minutes

## Detailed Steps

### Step 1: Create Zenodo Account

Visit https://zenodo.org/signup/ and register with:
- GitHub account (recommended - enables automatic integration)
- Email address
- ORCID (optional but recommended for researcher identity)

### Step 2: Link GitHub Repository

1. Go to https://zenodo.org/account/settings/github/
2. Click "Sync now" to refresh repository list
3. Find `thiagomassensini/millennium`
4. Toggle switch to **ON**
5. This enables automatic archiving on every GitHub release

### Step 3: Create GitHub Release

```bash
cd /home/thlinux/relacionalidadegeral
git tag -a v1.0.0 -m "First release: Complete XOR Millennium Framework with 1B+ validation"
git push origin v1.0.0
```

Then on GitHub:
1. Go to https://github.com/thiagomassensini/millennium/releases
2. Click "Draft a new release"
3. Select tag: `v1.0.0`
4. Release title: `XOR Millennium Framework v1.0.0 - Massive Validation Complete`
5. Description:
```
First complete release of the XOR Millennium Framework with comprehensive validation results.

**Key Results**:
- 1,004,800,003 twin primes validated (18.36 minutes, 56 cores)
- 100% primality verification via Miller-Rabin
- 100% BSD validation on 317,933,385 cases
- χ² = 11.12 (p < 0.001) for P(k) = 2^(-k) distribution
- 7 complete papers covering all 6 Millennium Prize Problems

**Contents**:
- 7 LaTeX papers with PDFs (235-292 KB each)
- C++ validation system (ultra_v4.cpp with mmap+OpenMP)
- Python analysis code for all 6 problems
- Twin prime miner (MPMC architecture)
- Complete reproducibility documentation
- Validation reports and statistical analysis

**Ready for**: Zenodo archiving, journal submission, peer review
```

6. Click "Publish release"

### Step 4: Zenodo Automatic Processing

Within 10 minutes, Zenodo will:
- Detect the new GitHub release
- Create archive snapshot
- Generate permanent DOI (format: 10.5281/zenodo.17520242)
- Create landing page with citation metadata

### Step 5: Get Your DOI

1. Go to https://zenodo.org/account/settings/github/
2. Find your repository
3. Click the DOI badge
4. Copy DOI for citations

### Step 6: Add DOI Badge to README

After DOI is generated, add to README.md:

```markdown
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17520242.svg)](https://doi.org/10.5281/zenodo.17520242)
```

## Metadata Configuration

The `.zenodo.json` file in this repository contains:

**Title**: XOR Millennium Framework: Binary Structure in the Millennium Prize Problems

**Description**: Full computational validation on 1B+ twin primes

**Keywords**: 
- Millennium Prize Problems
- BSD Conjecture, Riemann Hypothesis, P vs NP
- Yang-Mills, Navier-Stokes, Hodge Conjecture
- Binary XOR, Carry Chain Mechanism
- Twin Primes, Computational Number Theory

**License**: MIT (code) + CC BY 4.0 (papers)

**Subject Classifications** (MSC 2020):
- 11 (Number Theory)
- 14 (Algebraic Geometry)
- 68Q (Computational Complexity)
- 81T (Quantum Field Theory)
- 76 (Fluid Dynamics)

## What Gets Archived

Zenodo will archive the **entire repository** at release time:

```
✓ All 7 papers (LaTeX + PDF)
✓ Complete validation system (C++ source + executables)
✓ All analysis code (Python scripts)
✓ Twin prime miner source code
✓ Validation results (JSON, CSV)
✓ Documentation (README, SUBMISSION_README, WORKFLOW)
✓ Reproducibility instructions
✗ 53 GB CSV dataset (too large - provide download instructions)
```

## Large Dataset Handling

The 53 GB twin primes CSV is **NOT included** in Zenodo archive (size limit: 50 GB per dataset).

**Options**:
1. **Generate on-demand**: Users run `twin_prime_miner_v5_ultra_mpmc.cpp`
2. **Separate upload**: Use Zenodo's "New upload" for data-only deposit
3. **External hosting**: Figshare, OSF, institutional repository

Recommended approach in paper:
```
"The complete dataset of 1,004,800,003 twin primes (53 GB CSV) can be 
reproduced using the provided miner (twin_prime_miner_v5_ultra_mpmc.cpp) 
or requested from the author."
```

## Citation Format

After DOI assignment, use:

**APA**:
```
Massensini, T. (2025). XOR Millennium Framework: Binary Structure in the 
Millennium Prize Problems (Version 1.0.0) [Computer software]. Zenodo. 
https://doi.org/10.5281/zenodo.17520242
```

**BibTeX**:
```bibtex
@software{massensini2025xor,
  author       = {Massensini, Thiago},
  title        = {{XOR Millennium Framework: Binary Structure in the 
                   Millennium Prize Problems}},
  month        = nov,
  year         = 2025,
  publisher    = {Zenodo},
  version      = {1.0.0},
  doi          = {10.5281/zenodo.17520242},
  url          = {https://doi.org/10.5281/zenodo.17520242}
}
```

## Communities

Consider adding to Zenodo communities for visibility:

- **Mathematics** (general mathematics research)
- **Computer Science** (computational methods)
- **Open Science** (reproducible research)

Request community inclusion on Zenodo landing page after upload.

## Versioning Strategy

**v1.0.0** (current): Initial validation with 1B+ twin primes
**v1.1.0** (future): Additional k-levels (k=16-20)
**v2.0.0** (future): Extended analysis or journal revisions

Each version gets unique DOI + concept DOI (always resolves to latest).

## Zenodo vs ArXiv

**Zenodo advantages**:
- No endorsement required
- Immediate publication (< 10 minutes)
- DOI assigned automatically
- Accepts code + data + papers
- CERN-backed infrastructure
- Integrates with GitHub

**ArXiv advantages**:
- Higher prestige in mathematics
- Better search visibility
- Traditional preprint server
- Peer recognition

**Recommendation**: Zenodo first (immediate), ArXiv later (after endorsement).

## After Zenodo Publication

1. **Update README** with DOI badge
2. **Submit to journals**: Use Zenodo DOI in cover letter
3. **Share on ResearchGate**: Upload PDFs with Zenodo link
4. **Social media**: Twitter/X with #MillenniumPrize #NumberTheory hashtags
5. **Contact endorsers**: For eventual ArXiv submission

## Support

**Zenodo help**: support@zenodo.org
**Documentation**: https://help.zenodo.org/
**GitHub integration**: https://guides.github.com/activities/citable-code/

## Checklist Before Release

- [x] LICENSE file present (MIT + CC BY 4.0)
- [x] .zenodo.json metadata configured
- [x] CITATION.cff for GitHub citation
- [x] README.md comprehensive and professional
- [x] All papers compiled (PDFs present)
- [x] Validation results included (JSON/CSV)
- [x] Code documented and commented
- [x] Reproducibility instructions clear
- [x] Contact information correct
- [ ] Zenodo account created
- [ ] GitHub repository linked to Zenodo
- [ ] Release tag created (v1.0.0)
- [ ] DOI obtained and added to README

## Timeline

**Now**: Create Zenodo account, link GitHub  
**Today**: Create v1.0.0 release → automatic DOI  
**Tomorrow**: Update README with DOI, share on ResearchGate  
**This week**: Contact potential journal reviewers  
**Next week**: Submit to journal (with Zenodo DOI as preprint reference)

## Expected Impact

- **Immediate visibility**: Indexed by Google Scholar within 24 hours
- **Permanent record**: CERN guarantees 20+ year preservation
- **Citable**: DOI enables proper academic citation
- **Reproducible**: Code + data + documentation all archived together
- **Discoverable**: Keywords + MSC codes help researchers find your work

Your work deserves permanent, citable, discoverable recognition. Zenodo provides this immediately without gatekeeping.
