One-off research and provenance scripts. Each produced a measurement or figure
that is recorded in the top-level README and figures/; none is imported by the
library, the app, or the tests. They are kept runnable for reproducibility:

    python archive/exp_wht.py        (run from anywhere; each script adds the
                                      repo root to its own import path)

test_image.npy is the synthetic image some of these scripts share, generated
by make_test_image.py.
