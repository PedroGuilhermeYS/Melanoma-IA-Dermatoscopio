# NOTICE

This repository combines two distinct bodies of work under the single
`LICENSE` file (CC-BY-NC-4.0), which historically named only the original
author. This file exists to make the split explicit, since a paper
submission / code-availability statement needs unambiguous authorship.

## Original work (Fabian Wolz)

- The EVA-02 model architecture choice, the fine-tuning methodology, and
  the published checkpoint weights (`fawo/eva02-small-melanoma-classifier`
  on the Hugging Face Hub) are the work of **Fabian Wolz**
  (https://github.com/FaGit99/melanoma-classifier-eva02), released under
  CC-BY-NC-4.0.
- `results/threshold_key_indicators.csv` (candidate decision thresholds
  and their sensitivity/specificity trade-offs) originates from that same
  upstream evaluation.
- No training code is included in this repository (see `.gitignore`) --
  training happened upstream, in the original project.

## Derivative / new work (this repository)

- `melanoma_ia/` (the Python package: model loading, inference engine,
  Grad-CAM, the FastAPI inference server, the CLI, the camera-orchestrator
  pipeline, and the PyQt6 research GUI) is new code written for this
  low-cost dermatoscope project, consuming the checkpoint above via the
  Hugging Face Hub.
- Integration with the companion
  [`raspberry-pi-camera-web`](https://github.com/PedroGuilhermeYS/raspberry-pi-camera-web)
  repository (the `CameraClient`, the orchestrator, the HTTP contract
  between the two services) is original to this project.
