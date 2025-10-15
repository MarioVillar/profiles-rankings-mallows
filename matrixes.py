import numpy as np


def get_outranking_matrix(rankings: np.ndarray[int, int], votes: np.ndarray[int]) -> np.ndarray[int, int]:
    """
    Computes the outranking matrix from the ranking mappings and votes.

    :param rankings: The rankings of shape (number of unique rankings in profile, n_alternatives).
    :type rankings: np.ndarray[int, int]
    :param votes: The votes for each unique ranking of the profile, of shape (number of unique rankings in profile,).
    :type votes: np.ndarray[int]

    :return: The outranking matrix of shape (n_alternatives, n_alternatives) where n_alternatives is the number of alternatives.
    :rtype: np.ndarray[int, int]
    """
    num_unique_rankings, num_alternatives = rankings.shape

    # Initialize the matrix
    outrk_matrix = np.zeros((num_alternatives, num_alternatives), dtype=int)

    # Get matrix of the index in each unique ranking of each alternative
    idx_matrix = np.zeros((num_unique_rankings, num_alternatives), dtype=int)

    # Get the index of each alternative in each unique ranking
    for altern in range(num_alternatives):
        idx_matrix[:, altern] = np.where(rankings == altern)[1]

    # Count outranking matrix
    for i in range(num_alternatives):  # alternative i
        for j in range(i + 1, num_alternatives):  # alternative j
            tie_sum = ((idx_matrix[:, i] == idx_matrix[:, j]) * votes * 0.5).sum()

            outrk_matrix[i, j] = ((idx_matrix[:, i] < idx_matrix[:, j]) * votes).sum() + tie_sum
            outrk_matrix[j, i] = ((idx_matrix[:, i] > idx_matrix[:, j]) * votes).sum() + tie_sum

    return outrk_matrix


def permute_om_from_lexicographic(
    outrk_matrix: np.ndarray[int, int], central_rk: np.ndarray[int]
) -> np.ndarray[int, int]:
    """
    Permutes the outranking matrix from the lexicographical order of alternatives to the order given by the central ranking.

    This function is useful when just the outranking matrix is needed and not the full profile of rankings. In that case, there is no need to permute the central ranking over the profile of rankings. The corresponding outranking matrix can be directly obtained by permuting the elements of the outranking matrix corresponding to the lexicographical order.

    :param outrk_matrix: The outranking matrix of shape (n_alternatives, n_alternatives) where n_alternatives is the number of alternatives. The outranking matrix is assumed to be in the lexicographical order of alternatives.
    :type outrk_matrix: np.ndarray[int, int]
    :param central_rk: The central ranking considered for the Mallows model, of shape (n_alternatives,).
    :type central_rk: np.ndarray[int]

    :return: The permuted outranking matrix of shape (n_alternatives, n_alternatives) where n_alternatives is the number of alternatives. The outranking matrix is in the order given by the central ranking.
    :rtype: np.ndarray[int, int]
    """
    # The element outrk_matrix[i,j] should be mapped to new_outrk_matrix[central_rk[i], central_rk[j]]
    new_outrk_matrix = np.zeros_like(outrk_matrix)
    idx = np.arange(len(central_rk))
    new_outrk_matrix[np.ix_(central_rk, central_rk)] = outrk_matrix[np.ix_(idx, idx)]

    return new_outrk_matrix


def get_om_from_permutations(
    permutations: np.ndarray[int, int], votes: np.ndarray[int], central_rk: np.ndarray[int]
) -> np.ndarray[int, int]:
    """
    Computes the outranking matrix of a profile of rankings. The profile is given by its unique rankings in the form of permutations of the central ranking, and the votes of each unique ranking.

    :param permutations: The rankings in the form of permutations of the central ranking, of shape (number of unique rankings in profile, n_alternatives).
    :type permutations: np.ndarray[int, int]
    :param votes: The votes for each unique ranking of the profile, of shape (number of unique rankings in profile,).
    :type votes: np.ndarray[int]
    :param central_rk: The central ranking considered for the Mallows model, of shape (n_alternatives,).
    :type central_rk: np.ndarray[int]

    :return: The outranking matrixs, of shape (n_alternatives, n_alternatives) where n_alternatives is the number of alternatives.
    :rtype: np.ndarray[int, int]
    """
    return permute_om_from_lexicographic(
        outrk_matrix=get_outranking_matrix(rankings=permutations, votes=votes), central_rk=central_rk
    )


if __name__ == "__main__":
    import os
    from load_data import load_mallows_profiles

    PHI = 0.9  # Mallows dispersion
    NORM_MALLOWS = True  # Use normalized dispersion
    N_A = 3  # Number of alternatives
    N_V = 10  # Number of voters
    PRF_IDX = 0  # Index of the profile to load

    profile = load_mallows_profiles(
        phi=PHI,
        norm_mallows=NORM_MALLOWS,
        n_a=N_A,
        n_v=N_V,
        profile_idx=PRF_IDX,
        dir_path=os.getenv("MALLOWS_DATASET_PATH"),
    )

    om_lex = get_outranking_matrix(rankings=np.vstack(profile["ranking"].to_numpy()), votes=profile["votes"].to_numpy())

    inv_lex_rk = np.arange(N_A - 1, -1, -1)

    om_inv_lex = get_om_from_permutations(
        permutations=np.vstack(profile["ranking"].to_numpy()), votes=profile["votes"].to_numpy(), central_rk=inv_lex_rk
    )

    profile_inv_lex = load_mallows_profiles(
        phi=PHI,
        norm_mallows=NORM_MALLOWS,
        n_a=N_A,
        n_v=N_V,
        central_rk=inv_lex_rk,
        profile_idx=PRF_IDX,
        dir_path=os.getenv("MALLOWS_DATASET_PATH"),
    )

    om_inv_lex_from_rks = get_outranking_matrix(
        rankings=np.vstack(profile_inv_lex["ranking"].to_numpy()), votes=profile_inv_lex["votes"].to_numpy()
    )

    print(f"Lexicographic order: {np.arange(N_A)}")

    print("\nProfile for lexicographic order:")
    print(profile)

    print("\nOutranking matrix for lexicographic order:")
    print(om_lex)

    print("\n", "*" * 80, f"Inverse of lexicographic order: {inv_lex_rk}", sep="\n")

    print("\nProfile for inverse of lexicographic order:")
    print(profile_inv_lex)

    print("\nOutranking matrix for inverse lexicographic order (from the permutations):")
    print(om_inv_lex)

    print("\nOutranking matrix for inverse lexicographic order (from the rankings; it is the same as above):")
    print(om_inv_lex_from_rks)
