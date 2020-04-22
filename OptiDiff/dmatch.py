import numpy as np
import numba as nb
from OptiDiff import dtw


def while_shorter(sample, s):
    """
    Finds the index to cut the segment.
    :param sample: array of coordinates, starting from origin.
    :param s: size of the segment
    :return: index to cut
    """
    for i in range(sample.shape[0]):
        if sample[i] < s:
            continue
        else:
            return i
    return sample.shape[0] - 1


def create_seeds(molecule, s=150):
    """
    Creates list of seeds molecule coordinates.
    :param molecule: array of label coordinates
    :param s: size of the seed segments
    :return: list of seeds
    """
    res = list()
    for label_i in range(molecule.shape[0]):
        current_frag = molecule[label_i:] - molecule[label_i]
        until = while_shorter(current_frag, s)
        if until == len(current_frag) - 1:
            continue
        if current_frag[-1] < 150:
            break
        elif len(current_frag[1:until]):
            res.append(current_frag[1:until])
        else:
            continue
    return res


@nb.njit
def create_sim_matrix(seed1, seed2):
    dm = np.zeros((seed1.shape[0], seed2.shape[0]))
    for i in range(dm.shape[0]):
        for j in range(dm.shape[1]):
            dm[i, j] = 1/(np.abs(seed1[i] - seed2[j]) + 1)
    return dm


def pad_molecule_seeds(seeds):
    res = np.zeros((len(seeds), max([len(x) for x in seeds])))
    for i in range(len(seeds)):
        res[i, :len(seeds[i])] = seeds[i]
    return res


@nb.njit
def find_first_zero(arr):
    for i in range(arr.shape[0]):
        if arr[i] == 0:
            return i
    return arr.shape[0]


@nb.njit
def dtw_molecule_vs_reference_numba(molecule_seeds,
                                    reference_seeds,
                                    gap_open_penalty=5,
                                    gap_extension_penalty=5):
    scores = np.zeros((molecule_seeds.shape[0], reference_seeds.shape[0]))
    number_of_matching_labels = np.zeros((molecule_seeds.shape[0], reference_seeds.shape[0]))
    number_of_unmatching_labels = np.zeros((molecule_seeds.shape[0], reference_seeds.shape[0]))
    for i in range(molecule_seeds.shape[0]):
        for j in range(reference_seeds.shape[0]):
            mol_seed = molecule_seeds[i][: find_first_zero(molecule_seeds[i])]
            ref_seed = reference_seeds[j][: find_first_zero(reference_seeds[j])]
            simmat = create_sim_matrix(mol_seed, ref_seed)
            index1, index2, score = dtw.dtw_align(simmat, gap_open_penalty=gap_open_penalty,
                                                  gap_extend_penalty=gap_extension_penalty)
            scores[i, j] = score
            number_of_unmatching_labels[i, j] = np.where(index1 == -1)[0].shape[0] + np.where(index2 == -1)[0].shape[0]
            number_of_matching_labels[i, j] = np.where((index1 * index2) > 0)[0].shape[0]
    return (scores,
            number_of_matching_labels * 2,
            number_of_unmatching_labels,
            scores * (number_of_matching_labels * 2 - number_of_unmatching_labels))


def dtw_molecule_vs_reference(molecule_seeds,
                              reference_seeds,
                              gap_open_penalty=5,
                              gap_extension_penalty=5):
    scores = np.zeros((len(molecule_seeds), len(reference_seeds)))
    number_of_matching_labels = np.zeros((len(molecule_seeds), len(reference_seeds)))
    number_of_unmatching_labels = np.zeros((len(molecule_seeds), len(reference_seeds)))
    for i in range(len(molecule_seeds)):
        for j in range(len(reference_seeds)):
            simmat = create_sim_matrix(molecule_seeds[i], reference_seeds[j])
            index1, index2, score = dtw.dtw_align(simmat, gap_open_penalty=gap_open_penalty,
                                                  gap_extend_penalty=gap_extension_penalty)
            scores[i, j] = score
            number_of_unmatching_labels[i, j] = np.where(index1 == -1)[0].shape[0] + np.where(index2 == -1)[0].shape[0]
            number_of_matching_labels[i, j] = np.where((index1 * index2) > 0)[0].shape[0]
    return (scores,
            number_of_matching_labels * 2,
            number_of_unmatching_labels,
            scores * (number_of_matching_labels * 2 - number_of_unmatching_labels))


def molecules_vs_reference(molecules_seeds,
                           reference_seeds,
                           gap_open_penalty=5,
                           gap_extension_penalty=5,
                           thr=4):
    ids_res = list()
    scores_res = list()
    ref_seeds = pad_molecule_seeds(reference_seeds)
    for i in range(len(molecules_seeds)):
        mol_seeds = pad_molecule_seeds(molecules_seeds[i])
        scores, _, unmatched, _ = dtw_molecule_vs_reference_numba(mol_seeds, ref_seeds,
                                                                  gap_open_penalty=gap_open_penalty,
                                                                  gap_extension_penalty=gap_extension_penalty)
        ids_res.append(np.argmax(scores - unmatched, axis=1)[np.where(np.max(scores - unmatched, axis=1) > thr)[0]])
        scores_res.append(np.max(scores - unmatched, axis=1)[np.where(np.max(scores - unmatched, axis=1) > thr)[0]])
    return ids_res, scores_res
