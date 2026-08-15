# CC0 dog-photo pilot

`download_cc0_dogs.py` downloads only Wikimedia Commons JPEG files whose live
`extmetadata` explicitly reports all three of the following:

- `LicenseShortName = CC0`
- a Creative Commons Zero 1.0 license URL
- Creative Commons Zero usage terms

It then keeps only images in which the existing TorchVision Faster R-CNN model
detects at least one dog occupying a useful portion of the frame, and rejects
metadata categories associated with paintings, drawings, sculptures, and other
non-photographic works. The generated files and manifest are written under the
ignored `artifacts/cc0_dog_photos_diverse/` directory. To prevent one upload
series from dominating the pilot, the default policy keeps at most five images
per credited artist and three images per normalized title series.

```powershell
conda run -n meong-contest-full python `
  -m experiments.dino_fusion.download_cc0_dogs `
  --limit 100
```

Every accepted item records its Commons file page, original and downloaded
URLs, creator/credit fields, live license metadata, hashes, retrieval time, and
dog-detection result. `gallery.html` provides a local review page whose titles
link back to the corresponding Commons file pages. Keep the manifest with any
redistributed copy.

CC0 materially reduces copyright-compliance work but is not a warranty. Review
images manually before publication because personality, privacy, trademark,
moral-rights, and jurisdiction-specific restrictions can still apply. These
images are also unsuitable as a leakage-free foundation-model benchmark because
they may have appeared during pretraining.
