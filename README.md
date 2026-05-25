# Pioneer Analysis

Manuscript and supplemental analysis notebooks for the Pioneer study.

This is the research code accompanying our preprint:

> **An Implantable Device that Converses with Patients and Learns to Co-Manage Epilepsy.**
> medRxiv (2026). https://doi.org/10.64898/2026.01.26.26344234

See also the companion repository: [Pioneer Chat Interface](https://github.com/penn-cnt/Pioneer-Chat-Release) (patient-facing chat UI).

## Disclaimer

This software is provided for research purposes only. It is **not** a medical device and must not be used for clinical decision-making outside of an approved research protocol.

## What's here

- `manuscript_results.ipynb`: figures and statistics reported in the main manuscript (patient response rates, system response latency, message-type distribution, user response latency, data agent routing and accuracy, System Usability Scale).
- `supplemental_results.ipynb`: per-patient × per-day heatmaps of LLM engagement, survey response, and event-prompt response across the 7-day EMU stay.
- `utils/`: shared helper files used by both notebooks.

## Data availability

This repository is analysis code only. The clinical database, the patient roster, and CSV files referenced in the notebooks are **not** distributed. The notebooks are released for transparency and methodological reference.

## Citation

If you use this code or build on it, please cite the preprint:

```bibtex
@article{pioneer2026,
  title   = {An Implantable Device that Converses with Patients and Learns to Co-Manage Epilepsy},
  author  = {Goldblum, Zack and Shi, Haoer and Xu, Zhongchuan and Ojemann, William K. S. and Aguila, Carlos A. and Long, Kevin and Xie, Kevin and Nix, Kerry C. and Walsh, Katie and Chang, Ellie and Lavelle, Sarah and Bach, Brandon and Davis, Kathryn A. and Sinha, Nishant and Hammer, Lauren H. and Conrad, Erin C. and Litt, Brian},
  journal = {medRxiv},
  year    = {2026},
  doi     = {10.64898/2026.01.26.26344234},
  url     = {https://doi.org/10.64898/2026.01.26.26344234}
}
```

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
