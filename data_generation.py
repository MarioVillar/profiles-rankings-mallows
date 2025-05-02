import numpy as np
from joblib import Parallel, delayed
import time
import pyarrow.parquet as pq
import pyarrow as pa
from pyarrow.dataset import HivePartitioning
from prefsampling.ordinal.mallows import phi_from_norm_phi
import dotenv
import os

dotenv.load_dotenv()


def get_mallows_data_schema() -> pa.Schema:
    """
    Get the schema of the Parquet database with Mallows data.

    :return: Schema of the Parquet Mallows database.
    :rtype: pa.Schema
    """
    return pa.schema(
        [
            ("ranking", pa.list_(pa.int64())),
            ("votes", pa.int64()),
            ("profile_idx", pa.int64()),
            ("phi", pa.float64()),
            ("n_a", pa.int64()),
            ("n_v", pa.int64()),
            ("norm_mallows", pa.bool_()),
        ]
    )


def get_mallows_data_partitioning() -> pa.Schema:
    """
    Get the partitioning schema for database with Mallows data.

    :return: Partitioning schema for the Mallows database.
    :rtype: pa.Schema
    """
    return HivePartitioning(pa.schema([("phi", pa.float64()), ("n_a", pa.int64())]))


def get_n_a_dtype(n_a: int, unsigned: bool = True) -> type:
    """
    Get the dtype needed for the number of alternatives `n_a`.

    :param n_a: Number of alternatives.
    :type n_a: int
    :param unsigned: Whether the dtype should be unsigned or not, defaults to True.
    :type unsigned: bool

    :return: The dtype needed for the number of alternatives `n_a`.
    :rtype: type
    """
    if unsigned:
        return np.uint8 if n_a < 256 else np.uint16
    else:
        return np.int8 if n_a < 127 else np.int16


def get_n_v_dtype(n_v: int, unsigned: bool = True) -> type:
    """
    Get the dtype needed for the number of voters `n_v`.

    :param n_v: Number of voters.
    :type n_v: int
    :param unsigned: Whether the dtype should be unsigned or not, defaults to True.
    :type unsigned: bool

    :return: The dtype needed for the number of voters `n_v`.
    :rtype: type
    """
    if unsigned:
        return np.uint8 if n_v < 256 else np.uint16
    else:
        return np.int8 if n_v < 127 else np.int16


def mallows_insert_distributions(num_alternatives: int, phi: float) -> dict:
    """
    Get the insertion probability distributions the Repeated Insertion Model of Doignon et al. (2004) for the Mallows model.

    It is the distribution used in the package [preflibtools](`<https://github.com/PrefLib/preflibtools>`_).

    :param num_alternatives: Number of alternatives.
    :type num_alternatives: int
    :param phi: Dispersion parameter of the Mallows model.
    :type phi: float

    :return: A dictionary with the distributions probabilities for alternative.
    :rtype: dict
    """
    distributions = {}

    for i in range(1, num_alternatives + 1):
        # Start with an empty distro of length i
        distribution = [0] * i

        # compute the denom = phi^0 + phi^1 + ... phi^(i-1)
        denominator = sum([pow(phi, k) for k in range(i)])

        # Fill each element of the distro with phi^(i-j) / denominator
        for j in range(1, i + 1):
            distribution[j - 1] = pow(phi, i - j) / denominator

        distributions[i] = distribution

    return distributions


def sample_mallows_profiles(
    n_v: int,
    n_a: int,
    phi: float,
    n_prf: int,
    norm_mallows: bool = False,
) -> tuple:
    """
    Generate a set of Mallows profiles with the given parameters.

    :param n_v: Number of voters in the profile.
    :type n_v: int
    :param n_a: Number of alternatives in the profile.
    :type n_a: int
    :param phi: dispersion parameter of the Mallows model.
    :type phi: float
    :param n_prf: Number of profiles to generate.
    :type n_prf: int
    :param norm_mallows: Whether to normalize phi according to the number of alternatives or not. See `Boehmer, Faliszewski and Kraiczy (2023) <https://proceedings.mlr.press/v202/boehmer23b.html>`_ for more details. Defaults to False (not normalize).
    :type norm_mallows: bool

    :returns: tuple containing:
        - *(list[np.ndarray[np.ndarray[uint]]])* - List of arrays of shape (nº of generated rankings in the profile, n_a). Each array corresponds to a profile, with each of the rankings generated. A row of the np matrix corresponds to a ranking: the i-th element of this array is the index to which the i-th alternative is mapped in the ranking. For example, if the array is [2, 0, 1] then the ranking is a_1 > a_2 > a_0.
        - *(list[np.ndarray[uint]])* - List of arrays of shape (nº of generated rankings in the profile,). Each array corresponds to a profile, with the number of votes each ranking received. Index `i` in the `j`-th array of the list corresponds to the votes of the ranking in the `i`-th row of the `j`-th array in the list `prfs_rks`.
    :rtype: tuple
    """
    references = np.arange(n_a)

    alt_idx_dtype = get_n_a_dtype(n_a=n_a)
    vot_idx_dtype = get_n_v_dtype(n_v=n_v)

    # Store each of the n_prf profiles
    # For each profile, an array with each of the rankings generated (the order of the alternatives)
    prfs_rks = []  # List of arrays of shape (nº of generated rankings in the profile, n_a)
    # For each profile, an array with the number of votes each ranking received. Index i corresponds to the votes of the ranking in the i-th position of the prfs_rks
    prfs_rks_votes = []  # List of arrays of shape (nº of generated rankings in the profile)

    # Normalize phi if requested
    phi = phi_from_norm_phi(num_candidates=n_a, norm_phi=phi) if norm_mallows else phi

    # Precompute the distros for each dispersion
    insert_distribution = mallows_insert_distributions(n_a, phi)

    # n_prf is the number of profiles to generate
    for n_i in range(n_prf):
        vote_map_ind = {}

        # Now, generate votes...
        for voter in range(n_v):
            insert_vector = [0] * n_a

            for i in range(1, len(insert_vector) + 1):
                # options are 1...max
                insert_vector[i - 1] = np.random.choice(range(1, i + 1), 1, p=insert_distribution[i])[0]

            # Generate the vote itself and add it to the vote map
            vote = []
            for i in range(len(references)):
                vote.insert(insert_vector[i] - 1, references[i])

            vote_tuple = tuple(vote)
            vote_map_ind[vote_tuple] = vote_map_ind.get(vote_tuple, 0) + 1

        prfs_rks.append(np.array(list(vote_map_ind.keys()), dtype=alt_idx_dtype))
        prfs_rks_votes.append(np.array(list(vote_map_ind.values()), dtype=vot_idx_dtype))

    return prfs_rks, prfs_rks_votes


def mallows_exp(
    n_prf: int,
    phi_list: list,
    n_a_list: list,
    n_v_list: list,
    dir_path: str,
    fix_same_seed: bool = False,
    verbose: int = 0,
) -> float:
    """
    Generate a set of `n_prf` Mallows profiles with each of the dispersion values in `phi_list`, each of the number of alternatives in `n_a_list` and each of the number of voters in `n_v_list`. `n_prf` profiles are generated without normalising `phi` and `n_prf` profiles are generated normalising it.

    :param n_prf: Number of profiles to generate for each of the number of voters.
    :type n_prf: int
    :param phi_list: Values for the dispersion parameter of the Mallows model.
    :type phi_list: list
    :param n_a_list: List of number of alternatives considered for the profiles.
    :type n_a_list: list
    :param n_v_list: List of number of voters to generate profiles for.
    :type n_v_list: list
    :param dir_path: Path to the directory where the generated profiles will be stored.
    :type dir_path: str
    :param fix_same_seed: Whether to fix the same seed before generating each dataset of profiles (one unique fix) or fixing the a different seed for each dataset generated (for each value of `phi_list` and each value of `n_a_list`). If `False`, the different seeds are computed as `n_a * (10^phi_dec_prec + 1) + 10^phi_dec_prec * phi`, defaults to False
    :type fix_same_seed: bool, optional
    :param verbose: Verbose level for messages. `0` does not show messages, `1` shows the total time taken to execute the experimentation and `2` additionaly displays messages at the beggining and the end of the generation of each Mallows dataset, defaults to 0
    :type verbose: int

    :return: Time taken to execute all the experimentation.
    :rtype: float
    """

    def max_decimal_digits(floats_list):
        max_digits = 0
        for num in floats_list:
            num_str = str(num)
            if "." in num_str:
                decimal_part = num_str.split(".")[1]
                max_digits = max(max_digits, len(decimal_part))
        return max_digits

    # Maximum number of decimal places of phi values, used for the seed when fix_same_seed = False
    phi_dec_prec = 10 ** max_decimal_digits(phi_list)
    phi_dec_prec_n_a = phi_dec_prec + 1

    def joblib_mallows_exp(phi: float, n_a: int):
        # Set seed for reproducibility
        np.random.seed(seed=0 if fix_same_seed else int(n_a * phi_dec_prec_n_a + phi_dec_prec * phi))

        if verbose > 1:
            print(f"Starting with Mallows dataset for phi={phi}, n_a={n_a}.\n")

        # Generates two Mallows datasets for each value of phi, each value of n_a and each value of n_v:
        # - One with the original phi value
        # - One with the normalized phi value
        for norm_mallows in [False, True]:
            pa_ranking_list = []
            pa_votes_list = []
            pa_profile_idx_list = []
            pa_phi_list = []
            pa_n_a_list = []
            pa_n_v_list = []
            pa_norm_mallows_list = []

            for n_v in n_v_list:
                prfs_rks, prfs_rks_votes = sample_mallows_profiles(
                    n_v=n_v, n_a=n_a, phi=phi, n_prf=n_prf, norm_mallows=norm_mallows
                )

                # Store the dataset
                profile_idx = [i for i in range(len(prfs_rks)) for _ in prfs_rks[i]]

                pa_ranking_list.extend(np.concatenate(prfs_rks).tolist())
                pa_votes_list.extend(np.concatenate(prfs_rks_votes).tolist())
                pa_profile_idx_list.extend(profile_idx)
                pa_phi_list.extend([phi] * len(profile_idx))
                pa_n_a_list.extend([n_a] * len(profile_idx))
                pa_n_v_list.extend([n_v] * len(profile_idx))
                pa_norm_mallows_list.extend([norm_mallows] * len(profile_idx))

            table = pa.Table.from_pydict(
                {
                    "ranking": pa_ranking_list,
                    "votes": pa_votes_list,
                    "profile_idx": pa_profile_idx_list,
                    "phi": pa_phi_list,
                    "n_a": pa_n_a_list,
                    "n_v": pa_n_v_list,
                    "norm_mallows": pa_norm_mallows_list,
                }
            )

            # Guardar en Parquet
            pq.write_to_dataset(
                table,
                dir_path,
                partitioning=get_mallows_data_partitioning(),
                schema=get_mallows_data_schema(),
            )

        if verbose > 1:
            print(f"Finished with Mallows dataset for phi={phi}, n_a={n_a}.\n")

    init_time = time.time()

    Parallel(n_jobs=-1)(delayed(joblib_mallows_exp)(phi, n_a) for n_a in n_a_list for phi in phi_list)

    total_time = time.time() - init_time

    if verbose > 0:
        print(f"\nTotal time taken: {total_time} seconds.")

    return total_time


if __name__ == "__main__":

    time = mallows_exp(
        n_prf=1000,
        phi_list=[0.1, 0.4, 0.5, 0.6, 0.7, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0],
        n_a_list=[i for i in range(3, 17)] + [20, 25, 30],
        n_v_list=[10] + [i for i in range(25, 1001, 25)],
        dir_path=os.getenv("MALLOWS_DATASET_PATH"),
        fix_same_seed=False,
    )

    print(f"Time taken: {time} seconds.")
