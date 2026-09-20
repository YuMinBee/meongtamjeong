# Existing PetFinder color-text diagnostic

Declared before this diagnostic is computed. This test corpus was previously
inspected in August; this is exploratory reuse, NOT a new independent holdout.
No training, model selection, weight search, or application change.

Use the existing full-photo embeddings, dog photo 1 as query, photo 2 as gallery.
Evaluate the existing test dogs with known primary color; retain all test dogs
in identity retrieval. Korean query is only the structured primary color rendered
with the existing COLOR_KO mapping plus '털의 강아지를 찾아줘'. No name, ID, location,
or generated visual caption. Metadata is a proxy, not new human relevance labels.

Systems: CLIP image, DINO image, CLIP image+text, DINO image+CLIP text score
fusion, and DINO image+existing dog-trained Flow text. All text weights .20,
chosen in the original project; do not optimize against these results. The
original checkpoint is verified against its training report hash.
Also use cyclically shifted query texts for the CLIP composition and Flow
composition controls, in sorted pet-ID order. Some colors may coincide;
report this rate instead of asserting every shifted text is incorrect.

Identity Recall@1/5/10 and MRR use all test gallery dogs. Color nDCG@10 is binary
primary-color agreement, excludes the query dog, and excludes gallery dogs
whose color is unknown. Queries without a remaining known-color positive have
undefined nDCG and are excluded only from that metric. Group paired confidence
intervals by organization (2,000 bootstrap resamples), not by repeated descriptions.
No claims about temperament, free-form language, or actual missing-dog deployment.
