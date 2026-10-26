#!/usr/bin/env Rscript
# Paths: this script writes only to the output prefix given on the command line. Nothing is hardcoded.
# =============================================================================
# plot_trees.R - render ONE tree, for the worked example (R + ape)
# =============================================================================
# Draws a single Newick tree as a compact vector PDF plus a 400-dpi PNG, so a
# new user running examples/run_example.sh sees their own result rendered.
#
#   Rscript scripts/dev/plot_trees.R --single <tree.newick> <out_prefix>
#
# It writes ONLY to the prefix it is given and never into results/figures/.
#
# This script previously had a second, default "grid" mode that drew every
# benchmark genus into one grid figure under the results tree. That output was
# superseded by scripts/trees.py, which emits the tree record
# under the naming contract (FigS04a, FigS04b, FigS05a_p01..p06) with a row in
# FIGURE_MANIFEST.tsv. The grid mode has been removed; the original full version
# is kept outside the public tree for provenance.
#
# Style matches the figure suite: Helvetica/Arial family, ~size 6, thin edges,
# small panel area, vector PDF + 400-dpi PNG. Requires only 'ape'.
# =============================================================================

suppressWarnings(suppressMessages({
  if (!requireNamespace("ape", quietly = TRUE)) {
    cat("ERROR: R package 'ape' is required.  install.packages('ape')\n"); quit(status = 1)
  }
  library(ape)
}))

args <- commandArgs(trailingOnly = TRUE)
# Paths: this script writes ONLY to the output prefix given on the command line, so it needs no
# repository root. A REPO variable used to be derived here at the wrong depth and never read; it was
# removed rather than repaired, so there is no latent wrong-depth path to trip over.

BASE_CEX <- 0.45   # tip-label size -> ~6 pt at these panel dimensions
FAM      <- "sans" # Helvetica/Arial family

# PNG: R's png()/cairo device needs libtiff, which is absent on many headless HPC nodes.
# Render the vector PDF first (always works), then rasterise it with pdftoppm/gs if present.
png_from_pdf <- function(pdf_path, png_path, dpi = 400) {
  if (nzchar(Sys.which("pdftoppm"))) {
    system2("pdftoppm", c("-png", "-r", dpi, "-singlefile", shQuote(pdf_path),
                          shQuote(sub("\\.png$", "", png_path))), stdout = FALSE, stderr = FALSE)
  } else if (nzchar(Sys.which("gs"))) {
    system2("gs", c("-q", "-dNOPAUSE", "-dBATCH", "-sDEVICE=png16m", paste0("-r", dpi),
                    paste0("-sOutputFile=", shQuote(png_path)), shQuote(pdf_path)),
            stdout = FALSE, stderr = FALSE)
  }
  file.exists(png_path)
}

# --- shorten long genome labels so they fit in small panels -------------------
# e.g. "Pan_troglodytes_W_Sa23_69PR" -> "P.troglodytes 69PR"
short_labels <- function(tips) {
  vapply(tips, function(x) {
    p <- strsplit(x, "_", fixed = TRUE)[[1]]
    if (length(p) >= 2 && grepl("^[A-Z][a-z]+$", p[1])) {
      gen <- substr(p[1], 1, 1); sp <- p[2]; id <- p[length(p)]
      sprintf("%s.%s %s", gen, sp, id)
    } else if (nchar(x) > 22) paste0(substr(x, 1, 20), "..") else x
  }, character(1), USE.NAMES = FALSE)
}

draw_tree <- function(tr, title = NULL, cex = BASE_CEX, show_support = TRUE,
                      max_labelled = 24) {
  tr$tip.label <- short_labels(tr$tip.label)
  tr <- ladderize(tr)
  n  <- length(tr$tip.label)
  # Beyond ~30 tips a small panel cannot render legible labels (66 tips in ~2 in leaves ~2 pt per
  # label), so we show the topology only and say so in the title rather than print unreadable text.
  labelled <- n <= max_labelled
  lab_cex  <- if (n > 20) cex * 0.80 else cex
  par(mar = c(0.4, 0.2, 1.0, 0.2), family = FAM, xpd = NA)
  plot.phylo(tr, cex = lab_cex, edge.width = 0.35, no.margin = FALSE,
             show.tip.label = labelled, label.offset = 0.002, y.lim = NULL)
  # branch support (UFBoot) as small node dots when present and informative
  if (show_support && !is.null(tr$node.label)) {
    sup <- suppressWarnings(as.numeric(tr$node.label))
    if (any(!is.na(sup))) {
      ok <- !is.na(sup) & sup >= 95
      if (any(ok)) nodelabels(node = which(ok) + n, pch = 16, cex = 0.22, col = "#0072B2")
    }
  }
  if (!is.null(title)) title(main = title, cex.main = cex * 1.25, font.main = 1, line = 0.15)
}

# =============================== SINGLE MODE =================================
if (length(args) >= 1 && args[1] == "--single") {
  if (length(args) < 3) { cat("usage: --single <tree.newick> <out_prefix>\n"); quit(status = 2) }
  tre <- args[2]; pref <- args[3]
  if (!file.exists(tre)) { cat("ERROR: no such tree:", tre, "\n"); quit(status = 1) }
  tr <- read.tree(tre)
  ntip <- length(tr$tip.label)
  w <- 3.2; h <- max(1.6, 0.22 * ntip + 0.8)     # compact, grows a little with taxa
  pdf(paste0(pref, ".pdf"), width = w, height = h, pointsize = 6); draw_tree(tr,
      sprintf("ML tree  (n = %d tips)", ntip), cex = 0.55); invisible(dev.off())
  ok <- png_from_pdf(paste0(pref, ".pdf"), paste0(pref, ".png"))
  cat(sprintf("wrote %s.pdf%s  (%d tips)\n", basename(pref),
              if (ok) " + .png" else " (PNG skipped: no pdftoppm/gs)", ntip))
  quit(status = 0)
}


# No --single argument: this script has no other mode. Fail loudly rather than
# silently doing nothing, which is how the removed grid mode used to be reached.
cat("usage: plot_trees.R --single <tree.newick> <out_prefix>\n")
quit(status = 2)
