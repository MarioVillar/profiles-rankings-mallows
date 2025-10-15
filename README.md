# Profiles of rankings by the Mallows model

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15315407.svg)](https://doi.org/10.5281/zenodo.15315407)

This repository contains the code used to generate the dataset .

This dataset contains profiles of rankings sampled using the Mallows model (Mallows, 1957), through the Repeated Insertion Model (Doignon et al, 2004). Profiles with different characteristics are included. For every combination of the following parameters, there are 1000 profiles:

- Number of alternatives: {3, 4, …, 16, 20, 25, 30}
- Number of voters: {10, 25, 75, 100, 125, …, 1000}
- Mallows model original and normalised dispersion (Boehmer et al, 2023): {0.1, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 0.99, 0.999, 1.0}

The file [data_generation.py](data_generation.py) contains the code used to generate the dataset. The file [load_data.py](load_data.py) details an example of loading the data to a Pandas DataFrame, which assumes an environment variable `MALLOWS_DATASET_PATH` with the path to the data.

Since there is a lack of agreement in the literature regarding the notation of the number of alternatives and the number of voters, we have used `n_a` and `n_v`, respectively. For an specific combination of `phi` (dispersion), `n_a` and `n_v` there are 1000 profiles sampled with the original dispersion and 1000 profiles sampled with the normalised dispersion. The structure of the loaded profiles is as follows:

| Column name    | Data type | Description                                                                                                                                           |
| -------------- | --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `phi`          | float     | Mallows model dispersion.                                                                                                                             |
| `norm_mallows` | bool      | Whether the dispersion was normalised regarding the number of alternatives (Boehmer et al, 2023) or not. True is normalised, False is not normalised. |
| `n_a`          | int       | Number of alternatives.                                                                                                                               |
| `n_v`          | int       | Number of voters.                                                                                                                                     |
| `profile_idx`  | int       | Index of the profile (as there are 1000 profiles for each parameter combination). Ranges from 0 to 999.                                               |
| `ranking`      | list[int] | Sampled ranking.                                                                                                                                      |
| `votes`        | int       | Number of votes of the ranking in the profile.                                                                                                        |

If you use this dataset, please cite the Zenodo entry:

```bibtex
@dataset{mario_villar_2025_15315407,
  author    = {Mario Villar and Rico, Noelia and Díaz, Irene},
  title     = {Profiles of rankings by the Mallows model},
  month     = may,
  year      = 2025,
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.15315407},
  url       = {https://doi.org/10.5281/zenodo.15315407},
}
```

# Acknowledgements

This research has been funded by the Government of Spain through project MCINN-23-PID2022-139886NB-I00 and by grant Project PAPI-24-TESIS-14 of the University of Oviedo.
