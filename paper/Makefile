LATEX       = pdflatex -interaction=nonstopmode -halt-on-error
TMP_SUFFS   = .pdf .aux .bbl .blg .log .dvi .ps .eps .out .gz Notes.bib

all: shrinkingparallaxes.pdf

NAME = shrinkingparallaxes

${NAME}.pdf: ${NAME}.tex *.bib
	${LATEX} ${NAME}
	bibtex ${NAME}
	${LATEX} ${NAME}
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."
	( grep "Rerun to get" ${NAME}.log && ${LATEX} ${NAME} ) || echo "Done."

clean:
	rm -f $(foreach suff, ${TMP_SUFFS}, ${NAME}${suff})
