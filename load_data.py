import os
import dotenv
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pandas as pd

dotenv.load_dotenv()


from data_generation import get_mallows_data_schema


def get_pa_filters(
    n_a_list: list[int] = None, n_v_list: list[int] = None, phi_list: float = None, add_filters: list = None
) -> list:
    """
    Get the Parquet filters to apply when loading data from disk.

    :param n_a_list: List of the numbers of alternatives to load, defaults to None
    :type n_a_list: list[int], optional
    :param n_v_list: List of the numbers of voters to load, defaults to None
    :type n_v_list: list[int], optional
    :param phi_list: List of the dispersion values to load, defaults to None
    :type phi_list: float, optional
    :param add_filters: Additional filters to apply, in the format used by Parquet. Defaults to None
    :type add_filters: list, optional

    :return: List of Parquet filters to apply to the data.
    :rtype: list
    """
    # Define Parquet filters
    filters = []

    if n_a_list is not None:
        filters.append(("n_a", "in", n_a_list))

    if n_v_list is not None:
        filters.append(("n_v", "in", n_v_list))

    if phi_list is not None:
        filters.append(("phi", "in", phi_list))

    if add_filters is not None:
        filters.extend(add_filters)

    return filters if len(filters) > 0 else None


def load_exp_pa_part_dataset(
    dataset_path: str,
    schema: pa.Schema = None,
    n_a_list: list[int] = None,
    n_v_list: list[int] = None,
    phi_list: float = None,
    add_filters: list = None,
) -> pd.DataFrame:
    """
    Load the data stored in a partitioned Parquet dataset.

    :param dataset_path: Path to the Parquet dataset.
    :type dataset_path: str
    :param schema: Schema of the Parquet dataset, defaults to None (loading the schema from the dataset). The schema can only be None if no other optional parameters are specified.
    :type schema: pa.Schema
    :param n_a_list: List of the number of alternatives to load results for, defaults to None (loading all values)
    :type n_a_list: list[int]
    :param n_v_list: List of the number of voters to load results for, defaults to None (loading all values)
    :type n_v_list: list[int]
    :param phi_list: List of Mallows model dispersion values to load results for, defaults to None (loading all values)
    :type phi_list: float
    :param add_filters: List of additional filters to apply to the data. Each element of the list should be a tuple with the filter name, the filter condition and the filter value. Defaults to None (no additional filters).
    :type add_filters: list

    :raises ValueError: If the schema is None and any of the optional parameters are specified.
    :raises ValueError: If the resulting dataframe is empty.

    :return: Data.
    :rtype: pd.DataFrame
    """
    if schema is None and not all([n_a_list is None, n_v_list is None, phi_list is None, add_filters is None]):
        raise ValueError("The schema should be specified if any of the optional parameters are specified.")

    filters = get_pa_filters(n_a_list=n_a_list, n_v_list=n_v_list, phi_list=phi_list, add_filters=add_filters)

    dataset = pq.ParquetDataset(dataset_path, filters=filters, schema=schema)

    df = dataset.read().to_pandas()

    if df.empty:
        raise ValueError("No experimentation data found. Check the values of the parameters.")

    return df


def load_mallows_data(
    phi: float, n_a: int, dir_path: str, n_v: int = None, norm_mallows: bool = True, profile_idx: int = None
) -> pd.DataFrame:
    """
    Load the Mallows profiles corresponding to the specified parameter combination, retrieving the permutations of the central ranking.

    :param phi: Dispersion parameter of the Mallows model.
    :type phi: float
    :param n_a: Number of alternatives in the profile.
    :type n_a: int
    :param dir_path: Path to the directory where the dataset is stored.
    :type dir_path: str
    :param n_v: Number of voters in the profile, defaults to None (loading all values for the specified n_a and phi).
    :type n_v: int
    :param central_rk: Central ranking considered for the Mallows model, optional. By default, the lexicographical order is considered.
    :type central_rk: np.ndarray[int], optional
    :param norm_mallows: Whether to return the normalized dispersion data or non-normalized dispersion data, defaults to True.
    :type norm_mallows: bool, optional
    :param profile_idx: Index of the profile to load, defaults to None (loading all profiles with the specified parameters).
    :type profile_idx: int, optional

    :returns: Dataframe containing the Mallows profiles generated, with the permutations of the central ranking and their votes.
    :rtype: pd.DataFrame
    """
    additional_filters = [("norm_mallows", "=", norm_mallows)]

    if profile_idx is not None:
        additional_filters.append(("profile_idx", "=", profile_idx))

    return load_exp_pa_part_dataset(
        dataset_path=dir_path,
        schema=get_mallows_data_schema(),
        n_a_list=[n_a],
        n_v_list=[n_v] if n_v is not None else None,
        phi_list=[phi],
        add_filters=additional_filters,
    )


def load_mallows_profiles(
    phi: float,
    n_a: int,
    dir_path: str,
    n_v: int = None,
    central_rk: np.ndarray[int] = None,
    norm_mallows: bool = True,
    profile_idx: int = None,
) -> pd.DataFrame:
    """
    Load the Mallows profiles corresponding to the specified parameter combination.

    :param phi: Dispersion parameter of the Mallows model.
    :type phi: float
    :param n_a: Number of alternatives in the profile.
    :type n_a: int
    :param dir_path: Path to the directory where the dataset is stored.
    :type dir_path: str
    :param n_v: Number of voters in the profile, defaults to None (loading all values for the specified n_a and phi).
    :type n_v: int
    :param central_rk: Central ranking considered for the Mallows model, optional. By default, the lexicographical order is considered.
    :type central_rk: np.ndarray[int], optional
    :param norm_mallows: Whether to return the normalized dispersion data or non-normalized dispersion data, defaults to True.
    :type norm_mallows: bool, optional
    :param profile_idx: Index of the profile to load, defaults to None (loading all profiles with the specified parameters).
    :type profile_idx: int, optional

    :returns: Dataframe containing the Mallows profiles generated, with the rankings and their votes.
    :rtype: pd.DataFrame
    """
    # Load the profiles with the permutations of the central ranking
    profiles = load_mallows_data(
        phi=phi, n_a=n_a, dir_path=dir_path, n_v=n_v, norm_mallows=norm_mallows, profile_idx=profile_idx
    )

    # Convert the 'permutation' column from list to np.ndarray
    if central_rk is None or np.array_equal(central_rk, np.arange(n_a)):
        # If the central ranking is the lexicographic order, directly rename the column
        profiles.rename(columns={"permutation": "ranking"}, inplace=True)
    else:
        # Otherwise, convert map the permutation of the central ranking and rename the column
        profiles.insert(0, "ranking", profiles["permutation"].apply(lambda rk: central_rk[rk]))
        profiles.drop(columns=["permutation"], inplace=True)

    return profiles


if __name__ == "__main__":
    PHI = 0.9  # Mallows dispersion
    NORM_MALLOWS = True  # Use normalized dispersion
    N_A = 3  # Number of alternatives
    N_V = 10  # Number of voters
    central_rk = np.arange(N_A)  # Central ranking (lexicographic order)

    profiles = load_mallows_profiles(
        phi=PHI,
        norm_mallows=NORM_MALLOWS,
        n_a=N_A,
        n_v=N_V,
        central_rk=central_rk,
        profile_idx=None,  # Load all profiles with this parameter combination
        dir_path=os.getenv("MALLOWS_DATASET_PATH"),
    )
