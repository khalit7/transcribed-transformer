# Golden prompt renderings

Each file is one prompt as the code rendered it on a fixed fixture before the templates moved to
`data/labelled_data*/prompts/` (2026-10-31); `tests/test_prompts.py` re-renders every one and asserts equality, so a
template edit is a visible, deliberate change. `label_episodes_v2.txt` was re-captured after the first pilot's cut
questions left the bank (its fixture takes the bank's first three non-template questions), from the template two
independent reviewers had verified byte-identical to the previous string.
