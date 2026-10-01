# A2D episode conversion notes

`attach_episode_annotation.py` writes platform labels under
`meta/annotations/episode_labels.jsonl` and a machine-readable
`compatibility_audit.json`.  These files are metadata and do not alter the
source A2D episode.  A record is eligible for training only when its latest
annotation is approved, all retained frames are continuous, and every event
frame used by the label is present in a retained segment.
