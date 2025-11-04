# Makefile for XOR Millennium Framework Papers

PAPERS = bsd_twin_primes riemann_xor_repulsion p_vs_np_xor yang_mills_xor navier_stokes_xor hodge_xor xor_millennium_framework
PDFS = $(PAPERS:=.pdf)

.PHONY: all clean individual master quick_compile validate_and_update

all: $(PDFS)

# Compile individual paper
%.pdf: %.tex
	@echo "Compiling $<..."
	@pdflatex -interaction=nonstopmode $< > /dev/null
	@pdflatex -interaction=nonstopmode $< > /dev/null
	@echo "Generated $@"

# Compile only individual papers (not master)
individual:
	@for paper in bsd_twin_primes riemann_xor_repulsion p_vs_np_xor yang_mills_xor navier_stokes_xor hodge_xor; do \
		echo "Compiling $$paper.tex..."; \
		pdflatex -interaction=nonstopmode $$paper.tex > /dev/null 2>&1; \
		pdflatex -interaction=nonstopmode $$paper.tex > /dev/null 2>&1; \
	done
	@echo "All individual papers compiled"

# Compile master paper only
master: xor_millennium_framework.pdf
	@echo "Master paper compiled"

# Quick compile (single pass, for draft checking)
quick_compile:
	@for paper in $(PAPERS); do \
		echo "Quick compile $$paper.tex..."; \
		pdflatex -interaction=nonstopmode $$paper.tex > /dev/null 2>&1; \
	done
	@echo "Quick compile complete"

# Full workflow: validate data + update papers + compile
validate_and_update:
	@echo "Starting validation and paper update workflow..."
	@echo ""
	@echo "Step 1: Running validation (this may take 5-10 minutes)..."
	cd ../validacao && make validate
	@echo ""
	@echo "Step 2: Updating papers with validation results..."
	@LATEST=$$(ls -td ../validacao/validation_results_* | head -1); \
	python3 ../validacao/update_papers_with_validation.py $$LATEST/validation.json
	@echo ""
	@echo "Step 3: Compiling updated papers..."
	$(MAKE) all
	@echo ""
	@echo "Workflow complete! Check PDFs for validation sections."

# Check for LaTeX errors in all papers
check:
	@echo "Checking all papers for LaTeX errors..."
	@for paper in $(PAPERS); do \
		echo -n "Checking $$paper.tex... "; \
		if pdflatex -interaction=nonstopmode $$paper.tex > /tmp/latex_check.log 2>&1; then \
			echo "OK"; \
		else \
			echo "ERRORS FOUND"; \
			cat /tmp/latex_check.log | grep -A 5 "^!"; \
		fi; \
	done

# List all generated PDFs with sizes
list:
	@echo "Generated PDFs:"
	@ls -lh *.pdf 2>/dev/null | awk '{print $$9, $$5}' || echo "No PDFs found"

# Clean auxiliary files, keep PDFs
clean-aux:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz
	@echo "Cleaned auxiliary files"

# Clean everything including PDFs
clean:
	rm -f *.aux *.log *.out *.toc *.bbl *.blg *.synctex.gz *.pdf
	@echo "Cleaned all generated files"

# Restore from backups (if validation update was wrong)
restore:
	@for paper in $(PAPERS); do \
		if [ -f $$paper.tex.bak ]; then \
			echo "Restoring $$paper.tex from backup..."; \
			mv $$paper.tex.bak $$paper.tex; \
		fi; \
	done
	@echo "Restored from backups"

help:
	@echo "XOR Millennium Framework - Paper Compilation"
	@echo ""
	@echo "Usage:"
	@echo "  make                    - Compile all 7 papers"
	@echo "  make individual         - Compile 6 individual papers only"
	@echo "  make master             - Compile master paper only"
	@echo "  make quick_compile      - Fast single-pass compile (draft check)"
	@echo "  make validate_and_update- Full workflow: validate + update + compile"
	@echo "  make check              - Check all papers for LaTeX errors"
	@echo "  make list               - List generated PDFs with sizes"
	@echo "  make clean-aux          - Remove auxiliary files, keep PDFs"
	@echo "  make clean              - Remove all generated files"
	@echo "  make restore            - Restore papers from .bak files"
	@echo ""
	@echo "Individual papers:"
	@echo "  make bsd_twin_primes.pdf"
	@echo "  make riemann_xor_repulsion.pdf"
	@echo "  make p_vs_np_xor.pdf"
	@echo "  make yang_mills_xor.pdf"
	@echo "  make navier_stokes_xor.pdf"
	@echo "  make hodge_xor.pdf"
	@echo "  make xor_millennium_framework.pdf"
