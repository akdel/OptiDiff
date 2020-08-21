from OptiScan import utils
from typing import List, Tuple, Dict, Generator, Any, Iterator
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from LSH import lsh
from dataclasses import dataclass
import numba as nb
from scipy import stats, signal
import itertools
import random
import fire

BNX_HEAD = "/home/biridir/PycharmProjects/OptiScan/bnx_head.txt"

@nb.njit
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count


@nb.njit
def bit_distance(b1, b2):
    return countSetBits(b1 ^ b2)


@nb.njit(parallel=True)
def comp_bins(b1, b2):
    heatmap = np.zeros((b1.shape[0], b2.shape[0]))
    for i in range(heatmap.shape[0]):
        for j in nb.prange(heatmap.shape[1]):
            for r in range(b1.shape[-1]):
                heatmap[i, j] += bit_distance(b1[i, r], b2[j, r])
    return heatmap


@dataclass
class MoleculeSeg:
    """
    Contains molecule segment information used in SV detection.
    """
    index: int
    molecule_length: float
    segments: np.ndarray
    label_densities: List[int]
    zoom_factor: int = 500
    segment_length: int = 200
    nbits: int = 64
    compressed_segments = None
    lower_bound: int = 3

    @classmethod
    def from_bnx_line(cls, bnx_array_entry: dict, reverse=False,
                      segment_length: int = 200, zoom_factor: int = 500,
                      nbits: int = 64, lower_bound: int = 3, snr: float = 3.5):
        index: int = int(bnx_array_entry["info"][1])
        length: float = float(bnx_array_entry["info"][2])
        labels: np.ndarray = (np.array(bnx_array_entry["labels"][:-1]) / zoom_factor).astype(int)[np.array(bnx_array_entry["label_snr"]) >= snr]
        sig: np.ndarray = np.zeros(int(length) // zoom_factor + 1)
        sig[labels] = 5000.
        log_sig: np.ndarray = np.log1p(ndimage.gaussian_filter1d(sig, sigma=1))
        if reverse:
            log_sig = log_sig[::-1]
            labels = np.array([log_sig.shape[0] - x for x in labels])
            index *= -1
        segments, label_density = get_segments(segment_length, log_sig, labels)
        return cls(index, length, np.array(segments), label_density,
                   zoom_factor=zoom_factor, segment_length=segment_length,
                   nbits=nbits, lower_bound=lower_bound)

    def compress(self):
        def create_randoms(nbits=self.nbits, l=self.segment_length):
            randoms = np.zeros((nbits, l))
            steps = l / nbits
            for i in range(1, nbits):
                x = np.zeros(l)
                x[int(i * steps):int(i * steps + steps)] = l
                randoms[i] = x - np.mean(x)
            return randoms

        randoms: np.ndarray = create_randoms()
        filtered_values = np.array(self.label_densities) >= self.lower_bound
        self.compressed_segments: np.ndarray = lsh.VectorsInLSH(self.nbits,
                                                                self.segments,
                                                                custom_table=randoms).search_results[filtered_values]
        self.segments = self.segments[filtered_values]# = b'\xff' * (self.nbits//8) # This gives the maximum value for that many bytes
        self.label_densities = list(np.array(self.label_densities)[filtered_values])


def get_segments(segment_length: int, sig: np.ndarray, labels: List[int]):
    segments: List[np.ndarray] = [sig[i:i + segment_length] for i in labels
                                  if sig[i:i + segment_length].shape[0] == segment_length]
    label_density: List[int] = [len([x for x in labels if i < x < i + segment_length]) for i in labels
                                if sig[i:i + segment_length].shape[0] == segment_length]
    return segments, label_density


class CmapToSignal(utils.CmapParser):
    def __init__(self, cmap_path):
        utils.CmapParser.__init__(self, cmap_path)
        self.read_and_load_cmap_file()
        self.get_position_indexes()
        self.simulated_chromosomes = dict()
        self.chromosome_lsh = None
        self.chromosome_indices = list()
        self.chr_segments = list()
        self.chromosome_segment_indices = list()
        self.chromosome_segment_density = dict()
        self.prepared = False
        self.segment_length = None
        self.zoom_factor = None
        self.nbits = None

    def prepare(self, zoom_factor=500, segment_length=200, nbits=64):
        self.nbits = nbits
        self.simulate_all_chrs(zoom_factor=zoom_factor)
        self.create_chr_segments(length=segment_length)
        self.chr_compress(length=segment_length, nbits=nbits)
        self.segment_length = segment_length
        self.zoom_factor = zoom_factor
        self.prepared = True

    def generate_segment_from_bits(self, bits_id):
        bits = self.chromosome_lsh.search_results[bits_id:bits_id + 1].view("uint8")
        return np.unpackbits(bits).flatten()

    def simulate_chr(self, chr_id, zoom_factor=500):
        self.position_index[chr_id] = np.unique((np.array(self.position_index[chr_id]) // zoom_factor).astype(int))
        indices = self.position_index[chr_id]
        arr = np.zeros(np.max(indices[-1]) + 1, dtype=float)
        arr[indices] = 5000.
        self.simulated_chromosomes[chr_id] = np.log1p(ndimage.gaussian_filter1d(arr, sigma=1))

    def simulate_all_chrs(self, zoom_factor=500):
        for chr_id in self.position_index:
            self.simulate_chr(chr_id, zoom_factor=zoom_factor)

    def create_chr_segments(self, length=100):
        self.segment_count = 0
        cumulative = 0
        for chr_id, molecule in self.simulated_chromosomes.items():
            labels = np.array(self.position_index[chr_id])
            current_segments = [molecule[i:i + length] for i in labels if
                                molecule[i:i + length].shape[0] == length]
            if not len(current_segments):
                continue
            for j in range(len(labels)):
                current_density = len([x for x in (labels[j:] - labels[j]) if x < length])
                if chr_id not in self.chromosome_segment_density:
                    self.chromosome_segment_density[chr_id] = [current_density]
                else:
                    self.chromosome_segment_density[chr_id].append(current_density)
            self.chr_segments.append([molecule[i:i + length] for i in labels if
                                      molecule[i:i + length].shape[0] == length])
            self.chromosome_segment_indices.append((cumulative, cumulative + len(self.chr_segments[-1])))
            cumulative += len(self.chr_segments[-1])
            self.segment_count += len(self.chr_segments[-1])
        self.chr_segments = np.vstack(self.chr_segments)

    def chr_compress(self, nbits=16, length=100, custom_table=np.array([])):
        def create_randoms(nbits=nbits, l=length):
            randoms = np.zeros((nbits, l))
            steps = l / nbits
            for i in range(1, nbits):
                x = np.zeros(l)
                x[int(i * steps):int(i * steps + steps)] = l
                randoms[i] = x - np.mean(x)
            return randoms

        randoms = create_randoms()
        if custom_table.shape[0]:
            self.chromosome_lsh = lsh.VectorsInLSH(nbits, self.chr_segments, custom_table=custom_table)
        else:
            self.chromosome_lsh = lsh.VectorsInLSH(nbits, self.chr_segments, custom_table=randoms)


@dataclass
class ChromosomeSeg:
    index: int
    chromosome_length: int
    label_densities: np.ndarray
    kb_indices: np.ndarray
    zoom_factor: int
    segment_length: int
    compressed_segment_graph: Dict[bytes, List[int]]
    segments: np.ndarray
    nbits: int

    @classmethod
    def from_cmap_signals(cls, cmap_signals: CmapToSignal, chromosome_id: int, density_filter: int = 40):
        if cmap_signals.prepared:
            start, end = cmap_signals.chromosome_segment_indices[
                list(cmap_signals.position_index.keys()).index(chromosome_id)]
            compressed_segments: np.ndarray = cmap_signals.chromosome_lsh.search_results[start:end]
            kb_segment_indices: np.ndarray = cmap_signals.position_index[chromosome_id] / 2 # TODO: do this using zoom factor
            segment_graph: Dict[bytes, List[int]] = dict()
            label_densities: Dict[bytes, int] = dict()
            for i, segment in enumerate(compressed_segments):
                current_density = cmap_signals.chromosome_segment_density[chromosome_id][i]
                if current_density >= density_filter:
                    segment = b'0'
                    compressed_segments[i] = b'0'
                    current_density = 1
                if segment not in segment_graph:
                    segment_graph[segment] = [i]
                    label_densities[segment] = current_density
                else:
                    segment_graph[segment].append(i)
        else:
            raise BrokenPipeError("Cmap signals not prepared")
        print(cmap_signals.chromosome_lsh.search_results.shape)
        return cls(chromosome_id, kb_segment_indices[-1] * 1000,
                   np.array(list(label_densities.values())),
                   kb_segment_indices, cmap_signals.zoom_factor, cmap_signals.segment_length, segment_graph,
                   np.array(list(segment_graph.keys()), dtype=cmap_signals.chromosome_lsh.search_results.dtype),
                   cmap_signals.nbits)

    @property
    def total_segment_count(self):
        return self.kb_indices.shape[0]

    def expose_segments(self) -> np.ndarray:
        exposed = np.zeros(self.kb_indices.shape[0], dtype=f"|S{self.nbits//8}")
        for b in self.compressed_segment_graph:
            exposed[self.compressed_segment_graph[b]] = b
        return exposed


@dataclass
class Scores:
    molecule_id: int
    chromosome_id: int
    segment_matches: Dict[int, np.ndarray]
    total_segments: int
    segments: List[bytes] # numpy bytes array

    @classmethod
    def from_molecule_and_chromosome(cls, molecule: MoleculeSeg, chromosome: ChromosomeSeg, distance_thr: float = 1.8, max_distance_thr: float = 8):
        assert molecule.nbits == chromosome.nbits
        assert molecule.segment_length == chromosome.segment_length
        if molecule.segments.shape[0] <= 1:
            return Scores(molecule.index, chromosome.index, {}, 0, [])
        if molecule.compressed_segments is None:
            molecule.compress()
        if not len(molecule.compressed_segments):
            return Scores(molecule.index, chromosome.index, dict(),
                          len(molecule.segments), list(molecule.compressed_segments))
        mol_bits: np.ndarray = molecule.compressed_segments.view("uint8").reshape(
            (molecule.compressed_segments.shape[0], -1))
        chr_bits: np.ndarray = chromosome.segments.view("uint8").reshape((chromosome.segments.shape[0], -1))
        distances: np.ndarray = comp_bins(mol_bits, chr_bits)
        segment_matches: Dict[int, np.ndarray] = dict()
        for i in range(mol_bits.shape[0]):
            densities = (chromosome.label_densities[:distances.shape[1]] + molecule.label_densities[
                i]) / 2 / distance_thr
            # print(densities)
            densities[densities > max_distance_thr] = max_distance_thr
            matching_segment_indices = list(np.where(distances[i] <= densities)[0])
            if len(matching_segment_indices):
                segment_matches[i] = np.concatenate(
                    [chromosome.compressed_segment_graph[chromosome.segments[i]] for i in matching_segment_indices])
            else:
                continue
        return Scores(molecule.index, chromosome.index, segment_matches,
                      len(molecule.segments), list(molecule.compressed_segments))

    def proceeding(self, i: int):
        assert i in self.segment_matches
        keys = list(self.segment_matches.keys())
        try:
            next_key = keys[keys.index(i) + 1]
            return next_key, self.segment_matches[next_key]
        except KeyError and IndexError:
            return -1, -1

    def get_best_path(self, optimum_path: bool = False):
        if len(self.segment_matches) <= 1:
            return []
        first_key: int = list(self.segment_matches.keys())[0]
        current: List[Tuple[int, int]] = [(first_key, i) for i in self.segment_matches[first_key]]
        visited = {(k, i): (0, [i]) for (k, i) in current}
        target_length = len(self.segment_matches)
        while len(current):
            if optimum_path:
                current.sort(key=lambda x: visited[x][0], reverse=True)
            now = current.pop()
            if len(visited[now][1]) == target_length:
                return visited[now][1]
            next_key_id, next_segment_ids = self.proceeding(now[0])
            cost = visited[now][0]
            old_path = visited[now][1]
            for next_segment_id in next_segment_ids:
                new_cost = cost + abs(now[1] - next_segment_id)
                if (next_key_id, next_segment_id) not in visited or visited[(next_key_id, next_segment_id)][
                    0] >= new_cost:
                    current.append((next_key_id, next_segment_id))
                    visited[(next_key_id, next_segment_id)] = (new_cost, old_path + [next_segment_id])
        return []


def get_matched_ratio(all_scores: List[Scores]):
    total = len(all_scores) / 2
    used = set()
    for score in all_scores:
        try:
            if np.concatenate(list(score.segment_matches.values())).shape[0]:
                used.add(abs(score.molecule_id))
        except ValueError:
            continue
    return len(used) / total


@dataclass
class MoleculeSegmentPath:
    molecule_id: int
    forward_paths: \
        Dict[int, List[int]]  # keys are chromosome ids and values are lists of paths as segment ids as nodes.
    reverse_paths: Dict[int, List[int]]
    total_segments: int
    segments: List[Tuple[int, bytes, int, bool]]

    def unpack_segment(self, segment_id, nbits=64):
        i, b, _, forward = self.segments[segment_id]
        return np.unpackbits(np.array([b], dtype=f"|S{nbits // 8}").view("uint8"))


def segment_paths_from_scores(scores: Generator[Scores, Any, None], optimum_path: bool = False) -> List[
    MoleculeSegmentPath]:
    molecules: Dict[int, MoleculeSegmentPath] = dict()
    for score in scores:
        segments: List[Tuple[int, bytes, int, bool]] = [(int(score.segment_matches[i][0]), score.segments[i],
                                                         score.chromosome_id, score.molecule_id > 0) if i in score.segment_matches else (-1, score.segments[i]) for i in score.segment_matches]
        if abs(score.molecule_id) not in molecules:
            if score.molecule_id > 0:
                molecules[abs(score.molecule_id)] = MoleculeSegmentPath(abs(score.molecule_id),
                                                                        {score.chromosome_id: score.get_best_path(
                                                                            optimum_path)},
                                                                        {}, score.total_segments, segments)
            else:
                molecules[abs(score.molecule_id)] = MoleculeSegmentPath(abs(score.molecule_id), {},
                                                                        {score.chromosome_id: score.get_best_path(
                                                                            optimum_path)}, score.total_segments,
                                                                        segments)
        else:
            if score.molecule_id > 0:
                molecules[abs(score.molecule_id)].forward_paths[score.chromosome_id] = score.get_best_path(optimum_path)
                molecules[abs(score.molecule_id)].segments += segments
            else:
                molecules[abs(score.molecule_id)].reverse_paths[score.chromosome_id] = score.get_best_path(optimum_path)
                molecules[abs(score.molecule_id)].segments += segments

    return list(molecules.values())


def molecule_is_inverted_in_chromosome(molecule: MoleculeSegmentPath, chr_id: int):
    if len(molecule.forward_paths[chr_id]) and len(molecule.reverse_paths[chr_id]):
        return True
    else:
        return False


def molecule_is_inverted_all_chromosomes(molecule: MoleculeSegmentPath):
    forward, reverse = False, False
    for chr_id in molecule.forward_paths.keys():
        if len(molecule.forward_paths[chr_id]):
            forward = True
        if len(molecule.reverse_paths[chr_id]):
            reverse = True
    if forward and reverse:
        return True
    else:
        return False


@dataclass
class MoleculesOnChromosomes:
    molecules: Dict[int, MoleculeSegmentPath]
    chromosomes: Dict[int, ChromosomeSeg]
    counts_per_segment: Dict[int, Tuple[np.ndarray, np.ndarray]]
    molecules_per_segment: Dict[int, List[List[int]]]
    distance_thr: float
    segments_per_segment: Dict[int, List[np.ndarray]]

    @classmethod
    def from_molecules_and_chromosomes(cls,
                                       molecules: List[MoleculeSeg],
                                       chromosomes: List[ChromosomeSeg],
                                       distance_thr: float = 1.8, optimum_path: bool = False,
                                       max_distance_thr: float = 8.0):
        scores: Generator[Scores, Any, None] = (Scores.from_molecule_and_chromosome(y, x, distance_thr=distance_thr,
                                                                                    max_distance_thr=max_distance_thr)
                                                for y in molecules for x in chromosomes)
        molecule_segment_paths: List[MoleculeSegmentPath] = segment_paths_from_scores(scores, optimum_path=optimum_path)
        chromosomes_dict: Dict[int, ChromosomeSeg] = {chromosome.index: chromosome for chromosome in chromosomes}
        molecules_per_segment: Dict[int, List[int]] = {x: list() for x in chromosomes_dict.keys()}
        segments_per_segment: Dict[int, List[np.ndarray]] = {x: list() for x in chromosomes_dict.keys()}
        molecule_ids_per_segment: Dict[int, List[List[int]]] = {
            x: [list() for _ in range(chromosomes_dict[x].total_segment_count)] for x in chromosomes_dict.keys()}
        for molecule_segment_path in molecule_segment_paths:
            for chromosome_id in chromosomes_dict.keys():
                all_segments = molecule_segment_path.forward_paths[chromosome_id] + \
                               molecule_segment_path.reverse_paths[chromosome_id]

                molecules_per_segment[chromosome_id] += all_segments
                for segment in np.unique(all_segments):
                    molecule_ids_per_segment[chromosome_id][segment].append(molecule_segment_path.molecule_id)
        counts_per_segment: Dict[int, Tuple[np.ndarray, np.ndarray]] = {k: np.unique(v, return_counts=True) for (k, v)
                                                                        in molecules_per_segment.items()}
        return cls({x.molecule_id: x for x in molecule_segment_paths},
                   chromosomes_dict, counts_per_segment, molecule_ids_per_segment, distance_thr, {})

    def signal_from_chromosome(self, chromosome_id: int) -> np.ndarray:
        chromosome: ChromosomeSeg = self.chromosomes[chromosome_id]
        molecule_segments = [list(itertools.chain.from_iterable([x.forward_paths[chromosome_id],
                                                                 x.reverse_paths[chromosome_id]])) for x in
                             self.molecules.values()]
        sig: np.ndarray = np.zeros(int(chromosome.kb_indices[-1] + (chromosome.segment_length / 2) + 1))
        for mol_seg in molecule_segments:
            current = np.zeros(sig.shape[0])
            for seg in mol_seg:
                start, end = int(chromosome.kb_indices[seg]), \
                             int(chromosome.kb_indices[seg] + (chromosome.segment_length / 2 )) # / self.distance_thr))
                current[start: end] = signal.windows.gaussian(end - start, (end - start) * 0.5)
            sig += current
        sig[np.where(sig < 1)[0]] = 1
        return sig

    def plot_chromosome_segment(self, segment_id: int, chromosome_id: int):
        chromosome: ChromosomeSeg = self.chromosomes[chromosome_id]
        np.unpackbits(chromosome.segments[segment_id])
        pass

    def inverted_ratio_from_chromosome(self, chromosome_id: int) -> np.ndarray:
        chromosome: ChromosomeSeg = self.chromosomes[chromosome_id]
        inverted: np.ndarray = np.zeros(int(chromosome.kb_indices[-1] + (chromosome.segment_length / 2) + 1)) + 0.001
        for molecule_id, molecule in self.molecules.items():
            current = np.zeros(inverted.shape[0])
            if len(molecule.reverse_paths[chromosome_id]) and len(molecule.forward_paths[chromosome_id]):
                for seg in molecule.forward_paths[chromosome_id] + molecule.reverse_paths[chromosome_id]:
                    start, end = int(chromosome.kb_indices[seg]), \
                                 int(chromosome.kb_indices[seg] + (chromosome.segment_length / 2 / self.distance_thr))
                    current[start: end] = signal.windows.gaussian(end - start, (end - start) * 0.5)
                inverted += current
            else:
                continue
        return inverted


@dataclass
class UnspecificSV:
    region: Tuple[int, int, int, int]  # (chrid, kb_mid, kb_start, kb_end)
    score: float
    original_match_threshold: float
    reference_molecules_left: List[MoleculeSegmentPath]
    reference_molecules_right: List[MoleculeSegmentPath]
    sv_candidate_molecules_left: List[MoleculeSegmentPath]
    sv_candidate_molecules_right: List[MoleculeSegmentPath]
    inversion_ttest: Tuple[float, float]

    def plot_molecules(self, chromosome: ChromosomeSeg, data="reference", side="left") -> None:
        assert chromosome.index == self.region[0]
        chrom_id: int = self.region[0]
        colors = ["green", "yellow", "orange", "pink", "brown", "black", "blue", "red"]
        ci: int = 0
        i: int = 0
        distances: np.ndarray = chromosome.kb_indices
        if data == "reference":
            if side == "left":
                molecules = self.reference_molecules_left
            elif side == "right":
                molecules = self.reference_molecules_right
            else:
                return None
        elif data == "sv_candidate":
            if side == "left":
                molecules = self.sv_candidate_molecules_left
            elif side == "right":
                molecules = self.sv_candidate_molecules_right
            else:
                return None
        else:
            return None
        for molecule in molecules:
            for coord in np.concatenate([molecule.reverse_paths[chrom_id], molecule.forward_paths[chrom_id]]):
                plt.plot([distances[int(coord)], distances[int(coord)] + chromosome.segment_length],
                         [i, i], color=colors[ci])
                i += 1
            i += 3
            ci = (ci + 1) % len(colors)
        plt.show()


def find_unspecific_sv_sites(reference: MoleculesOnChromosomes,
                             sv_candidate: MoleculesOnChromosomes,
                             z_thr: float = 5.) -> List[UnspecificSV]:
    result: List[UnspecificSV] = list()
    for chr_id in reference.counts_per_segment.keys():
        assert (chr_id in reference.counts_per_segment) and (chr_id in sv_candidate.counts_per_segment)
        sig = reference.signal_from_chromosome(chr_id) / sv_candidate.signal_from_chromosome(chr_id)
        peak_indices = utils.get_peaks(sig, z_thr, np.median(sig))
        plt.plot(sig)
        plt.scatter((peak_indices), sig[np.array(peak_indices).astype(int)], c="red")
        plt.show()
        for start, end, score in list({find_boundaries(sig, x) for x in peak_indices}):
            print(start, end, score)
            reference_molecules_left, reference_molecules_right, \
            sv_molecules_left, sv_molecules_right, \
            inversions = get_molecules_in_region(chr_id, start, end, reference, sv_candidate)
            result.append(UnspecificSV((chr_id, (start + end) / 2, start, end),
                                       score, reference.distance_thr, reference_molecules_left,
                                       reference_molecules_right,
                                       sv_molecules_left, sv_molecules_right, inversions))
    return result


def get_molecules_in_region(chr_id, start, end, reference: MoleculesOnChromosomes,
                            sv_candidate: MoleculesOnChromosomes) -> Tuple[List[MoleculeSegmentPath],
                                                                           List[MoleculeSegmentPath],
                                                                           List[MoleculeSegmentPath],
                                                                           List[MoleculeSegmentPath],
                                                                           Tuple[float, float]]:
    start_kb = start - (reference.chromosomes[chr_id].segment_length / 2) * 2
    end_kb = end + (reference.chromosomes[chr_id].segment_length / 2) * 2
    mid_kb = (start + end) / 2
    inversion_number = 0

    proximal_left_segment_ids = [i for (i, x) in enumerate(reference.chromosomes[chr_id].kb_indices) if
                                 start_kb < x <= mid_kb]
    proximal_right_segment_ids = [i for (i, x) in enumerate(reference.chromosomes[chr_id].kb_indices) if
                                  mid_kb < x <= end_kb]

    proximal_left_chr_molecule_ids = np.unique(
        np.concatenate([reference.molecules_per_segment[chr_id][i] for i in proximal_left_segment_ids]))
    proximal_left_sv_molecule_ids = np.unique(
        np.concatenate([sv_candidate.molecules_per_segment[chr_id][i] for i in proximal_left_segment_ids]))

    proximal_rigth_chr_molecule_ids = np.unique(
        np.concatenate([reference.molecules_per_segment[chr_id][i] for i in proximal_right_segment_ids]))
    proximal_right_sv_molecule_ids = np.unique(
        np.concatenate([sv_candidate.molecules_per_segment[chr_id][i] for i in proximal_right_segment_ids]))

    reference_molecules_left = [reference.molecules[k] for k in proximal_left_chr_molecule_ids]
    reference_molecules_right = [reference.molecules[k] for k in proximal_rigth_chr_molecule_ids]
    sv_molecules_left = [sv_candidate.molecules[k] for k in proximal_left_sv_molecule_ids]
    sv_molecules_right = [sv_candidate.molecules[k] for k in proximal_right_sv_molecule_ids]

    inverted_sv_ratios = list()
    inverted_ref_ratios = list()
    for molecule in sv_molecules_left + sv_molecules_right:
        forwards = 0.
        backwards = 0.
        for chr_id in reference.chromosomes.keys():
            forwards += len(molecule.forward_paths[chr_id])
            backwards += len(molecule.reverse_paths[chr_id])
        if forwards == 0 and backwards == 0:
            continue
        else:
            inverted_sv_ratios.append(min(forwards, backwards) / max(forwards, backwards))
    for molecule in reference_molecules_left + reference_molecules_right:
        forwards = 0.
        backwards = 0.
        for chr_id in reference.chromosomes.keys():
            forwards += len(molecule.forward_paths[chr_id])
            backwards += len(molecule.reverse_paths[chr_id])
        if forwards == 0 and backwards == 0:
            continue
        else:
            inverted_ref_ratios.append(min(forwards, backwards) / max(forwards, backwards))
    if np.mean(inverted_sv_ratios) <= np.mean(inverted_ref_ratios):
        return reference_molecules_left, reference_molecules_right, \
               sv_molecules_left, sv_molecules_right, (1, 1)
    else:
        tscore, pvalue = stats.ttest_ind(inverted_sv_ratios, inverted_ref_ratios, equal_var=False)
        return reference_molecules_left, reference_molecules_right, sv_molecules_left, sv_molecules_right, (
            tscore, pvalue)


def find_boundaries(sig, peak, snr=1.5):
    snr_thr = max(1, np.median(sig)) * snr
    start = end = peak
    for end in range(peak, len(sig) - 5):
        if np.median(sig[end:end + 3]) > snr_thr:
            continue
        else:
            break
    for start in list(range(3, peak))[::-1]:
        if np.median(sig[start - 3:start]) > snr_thr:
            continue
        else:
            break
    return int(start), int(end), int(sig[(start + end) // 2])


@dataclass
class Translocation:
    origin: Tuple[int, int, int, int]
    inserted_region: Tuple[int, int, int]
    score: float

    @property
    def line(self):
        chrid, _, origin_start, origin_end = self.origin
        target_chrid, target_start, target_end = self.inserted_region
        return f"{str(Translocation)}\t{chrid}\t{origin_start}\t{origin_end}\t{target_chrid}\t{target_start}\t{target_end}\n"


@dataclass
class Duplication:
    translocation: Translocation
    t_score: float
    p_value: float

    @property
    def line(self):
        chrid, _, origin_start, origin_end = self.translocation.origin
        target_chrid, target_start, target_end = self.translocation.inserted_region
        return f"{str(Duplication)}\t{chrid}\t{origin_start}\t{origin_end}\t{target_chrid}\t{target_start}\t{target_end}\n"

@dataclass
class Inversion:
    based_on: [Translocation, Duplication, UnspecificSV]

    @property
    def region(self):
        if type(self.based_on) == Translocation:
            return self.based_on.inserted_region
        elif type(self.based_on) == Duplication:
            return self.based_on.translocation.inserted_region
        else:
            return self.based_on.region

    @property
    def line(self):
        target_chrid, _, target_start, target_end = self.region
        if type(self.based_on) == Duplication:
            chrid, _, origin_start, origin_end = self.based_on.translocation.origin
        elif type(self.based_on) == Translocation:
            chrid, _, origin_start, origin_end = self.based_on.origin
        else:
            chrid, _, origin_start, origin_end = self.region
        return f"{str(type(self.based_on))}and{str(Inversion)}\t{chrid}\t{origin_start}\t{origin_end}\t{target_chrid}\t{target_start}\t{target_end}\n"


def check_translocation_or_inversion(unspecific_sv: UnspecificSV,
                                     chromosome: ChromosomeSeg,
                                     thr: float = 5,
                                     inversion_test: [bool, Translocation, Duplication, UnspecificSV] = False) -> [
    Translocation, UnspecificSV, Inversion]:
    chromosome_id = chromosome.index

    def filt_func(x):
        return molecule_is_inverted_in_chromosome(x, chromosome_id)

    if inversion_test:
        translocation_ratio_left = get_translocation_ratio_signal(
            list(filter(filt_func, unspecific_sv.reference_molecules_left)),
            list(filter(filt_func, unspecific_sv.sv_candidate_molecules_left)),
            chromosome, unspecific_sv.original_match_threshold)
        translocation_ratio_right = get_translocation_ratio_signal(
            list(filter(filt_func, unspecific_sv.reference_molecules_right)),
            list(filter(filt_func, unspecific_sv.sv_candidate_molecules_right)),
            chromosome, unspecific_sv.original_match_threshold)
    else:
        translocation_ratio_left = get_translocation_ratio_signal(unspecific_sv.reference_molecules_left,
                                                                  unspecific_sv.sv_candidate_molecules_left,
                                                                  chromosome, unspecific_sv.original_match_threshold)
        translocation_ratio_right = get_translocation_ratio_signal(unspecific_sv.reference_molecules_right,
                                                                   unspecific_sv.sv_candidate_molecules_right,
                                                                   chromosome, unspecific_sv.original_match_threshold)
    plt.plot(translocation_ratio_left)
    plt.plot(translocation_ratio_right)
    plt.show()
    peak_indices_left = utils.get_peaks(translocation_ratio_left, thr, max(1, np.median(translocation_ratio_left)))
    peak_indices_right = utils.get_peaks(translocation_ratio_right, thr, max(1, np.median(translocation_ratio_right)))
    if len(peak_indices_left) and len(peak_indices_right):
        top_left = list(sorted([x for x in peak_indices_left], key=lambda x: translocation_ratio_left[int(x)]))[-1]
        top_right = list(sorted([x for x in peak_indices_right], key=lambda x: translocation_ratio_right[int(x)]))[-1]
        score = (translocation_ratio_left[int(top_left)] + translocation_ratio_right[int(top_right)]) / 2
        if abs(top_left - top_right) <= chromosome.segment_length // 2:
            mid = (top_left + top_right) // 2
            top_left = mid - chromosome.segment_length // 2
            top_right = mid + chromosome.segment_length // 2
        if inversion_test:
            return Inversion(inversion_test)
        else:
            return Translocation(unspecific_sv.region, (chromosome.index, min(top_left, top_right),
                                                        max(top_left, top_right)), score)
    else:
        if inversion_test:
            translocation_ratio_all = get_translocation_ratio_signal(
                list(filter(filt_func,
                            unspecific_sv.reference_molecules_right + unspecific_sv.reference_molecules_left)),
                list(filter(filt_func,
                            unspecific_sv.sv_candidate_molecules_right + unspecific_sv.sv_candidate_molecules_left)),
                chromosome, unspecific_sv.original_match_threshold)
        else:
            translocation_ratio_all = get_translocation_ratio_signal(
                unspecific_sv.reference_molecules_right + unspecific_sv.reference_molecules_left,
                unspecific_sv.sv_candidate_molecules_right + unspecific_sv.sv_candidate_molecules_left,
                chromosome, unspecific_sv.original_match_threshold)
        peak_indices_all = utils.get_peaks(translocation_ratio_all, thr, max(1, np.median(translocation_ratio_all)))
        if len(peak_indices_all):
            top_all = list(sorted([x for x in peak_indices_all], key=lambda x: translocation_ratio_all[int(x)]))[-1]
            score = translocation_ratio_left[int(top_all)]
            if inversion_test:
                return Inversion(inversion_test)
            else:
                return Translocation(unspecific_sv.region, (chromosome.index,
                                                            max(0, top_all - chromosome.segment_length // 2),
                                                            max(0, top_all + chromosome.segment_length // 2)), score)
        else:
            return unspecific_sv


def get_translocation_ratio_signal(reference_molecules: List[MoleculeSegmentPath],
                                   sv_candidate_molecules: List[MoleculeSegmentPath],
                                   chromosome: ChromosomeSeg, distance_thr: float) -> np.ndarray:
    chromosome_id = chromosome.index
    reference_molecule_segments = [
        list(itertools.chain.from_iterable([x.forward_paths[chromosome_id], x.reverse_paths[chromosome_id]])) for x in
        reference_molecules]
    sv_candidate_molecule_segments = [
        list(itertools.chain.from_iterable([x.forward_paths[chromosome_id], x.reverse_paths[chromosome_id]])) for x in
        sv_candidate_molecules]

    sig_ref = segments_to_sig(reference_molecule_segments, chromosome.kb_indices, chromosome.segment_length,
                              distance_thr)
    sig_sv = segments_to_sig(sv_candidate_molecule_segments, chromosome.kb_indices, chromosome.segment_length,
                             distance_thr)
    return sig_sv / sig_ref


def segments_to_sig(molecule_segments: List[List[int]], kb_indices: np.ndarray, segment_length: int,
                    distance_thr: float) -> np.ndarray:
    sig = np.zeros(int(kb_indices[-1] + (segment_length / 2) + 1))
    for mol_seg in molecule_segments:
        current = np.zeros(sig.shape[0])
        for seg in mol_seg:
            start, end = int(kb_indices[seg]), \
                         int(kb_indices[seg] + (segment_length / 2 / distance_thr))
            current[start: end] = signal.windows.gaussian(end - start, (end - start) * 0.5)
        sig += current
    sig[np.where(sig < 1)[0]] = 1
    return sig


def check_duplication(translocation: Translocation,
                      reference: MoleculesOnChromosomes,
                      sv_candidate: MoleculesOnChromosomes,
                      p_value_thr: float = 0.001) -> [Duplication, Translocation]:
    chromosome_id = translocation.inserted_region[0]
    ratios = sv_candidate.signal_from_chromosome(chromosome_id) / reference.signal_from_chromosome(chromosome_id)
    start, end = translocation.inserted_region[1:]
    without = np.concatenate([ratios[:start], ratios[end:]])
    duplication_candidate = ratios[start:end]
    if np.mean(without) >= np.mean(duplication_candidate):
        return translocation
    ttest_p = stats.ttest_ind(without, duplication_candidate, equal_var=False)
    if ttest_p[1] <= p_value_thr:
        return Duplication(translocation, ttest_p[0], ttest_p[1])
    else:
        return translocation


@dataclass
class Inversion:
    based_on: [Translocation, Duplication, UnspecificSV]

    @property
    def region(self):
        if type(self.based_on) == Translocation:
            return self.based_on.inserted_region
        elif type(self.based_on) == Duplication:
            return self.based_on.translocation.inserted_region
        else:
            return self.based_on.region


def check_inversion_from_unspecific_sv(unspecific_sv: UnspecificSV, thr: float = 0.01,
                                       additional: [None, Translocation, Duplication] = None) -> [Inversion,
                                                                                                  UnspecificSV,
                                                                                                  Duplication]:
    if unspecific_sv.inversion_ttest[1] <= thr:
        if additional is not None:
            return Inversion(additional)
        else:
            return Inversion(unspecific_sv)
    else:
        if additional is not None:
            return additional
        else:
            return unspecific_sv


@dataclass
class SmallDeletionOrInsertion:
    region: Tuple[int, int, int, int]
    score: float
    potential_deletion_size: float

    @property
    def line(self):
        chrid, _, origin_start, origin_end = self.region
        target_chrid, target_start, target_end = "N/A", "N/A", "N/A"
        return f"{str(SmallDeletionOrInsertion)}\t{chrid}\t{origin_start}\t{origin_end}\t{target_chrid}\t{target_start}\t{target_end}\n"


@dataclass
class Deletion:
    region: Tuple[int, int, int, int]
    size: float

    @property
    def line(self):
        chrid, _, origin_start, origin_end = self.region
        target_chrid, target_start, target_end = "N/A", "N/A", "N/A"
        return f"{str(Deletion)}\t{chrid}\t{origin_start}\t{origin_end}\t{target_chrid}\t{target_start}\t{target_end}\n"


def check_deletion(unspecific_sv: UnspecificSV, chromosome: ChromosomeSeg) -> [SmallDeletionOrInsertion, Deletion]:
    deletion_size: int = abs(unspecific_sv.region[2] - unspecific_sv.region[3])
    if deletion_size >= (chromosome.segment_length / 2):
        return Deletion(unspecific_sv.region, deletion_size)
    else:
        return SmallDeletionOrInsertion(unspecific_sv.region, unspecific_sv.score, deletion_size)


def find_specific_sv(unspecific_sv: UnspecificSV,
                     reference: MoleculesOnChromosomes,
                     candidate: MoleculesOnChromosomes) -> List[Any]:
    chromosomes = reference.chromosomes
    first_chr_id = list(reference.chromosomes.keys())[0]
    svs = list()
    for chr_id in chromosomes:
        current_sv = check_translocation_or_inversion(unspecific_sv, chromosomes[chr_id])
        if type(current_sv) == Translocation:
            current_sv = check_duplication(current_sv, reference, candidate)
        inversion = check_translocation_or_inversion(unspecific_sv, chromosomes[chr_id], inversion_test=current_sv)
        if type(inversion) == Inversion:
            current_sv = inversion
        if type(current_sv) != UnspecificSV:
            svs.append(current_sv)
    if not len(svs):
        svs.append(check_deletion(unspecific_sv, chromosomes[first_chr_id]))
    return svs


def from_fasta(fasta_path, digestion_motif="GCTCTTC", enzyme_name="BSPQ1", channel="1") -> CmapToSignal:
    temp_name = "temp.cmap"
    fasta_to_cmap(fasta_path, temp_name, digestion_motif=digestion_motif, enzyme_name=enzyme_name, channel=channel)
    return CmapToSignal(temp_name)


def fasta_to_cmap(fasta_in_path, cmap_out_path, digestion_motif="GCTCTTC",
                  enzyme_name="BSPQ1", channel="1"):
    fasta = utils.FastaObject(fasta_in_path)
    fasta.initiate_fasta_array()
    fasta.fill_complete_fasta_array()
    fasta.write_fasta_to_cmap(digestion_sequence=digestion_motif, output_file_name=cmap_out_path,
                              enzyme_name=enzyme_name, channel=channel)


def filter_and_prepare_molecules_on_chromosomes(
        cmap_file_location: str,
        bnx_file_location: str,
        subsample_ratio: float = 1.0,
        nbits: int = 64, segment_length: int = 275,
        zoom_factor: int = 500, minimum_molecule_length: int = 150_000,
        distance_threshold: float = 1.7,
        density_filter: int = 40) -> MoleculesOnChromosomes:
    cmap: CmapToSignal = CmapToSignal(cmap_file_location)
    cmap.prepare(segment_length=segment_length, nbits=64)
    chromosomes: List[ChromosomeSeg] = [ChromosomeSeg.from_cmap_signals(cmap, chr_id, density_filter=density_filter) for chr_id in
                                        {int(x[0]) for x in cmap.cmap_lines if len(x)}]
    bnx_lines = get_subsampled_bnx_lines(bnx_file_location,
                                         minimum_molecule_length,
                                         subsample_ratio)

    molecules: Iterator[MoleculeSeg] = itertools.chain.from_iterable(
        (
            molecule_generator_from_bnx_lines(bnx_lines,
                                              False,
                                              segment_length,
                                              zoom_factor,
                                              nbits),
            molecule_generator_from_bnx_lines(bnx_lines,
                                              True,
                                              segment_length,
                                              zoom_factor,
                                              nbits),
        )
    )
    return MoleculesOnChromosomes.from_molecules_and_chromosomes(molecules, chromosomes,
                                                                 distance_thr=distance_threshold)


@dataclass
class SvResult:
    SV: [Deletion, SmallDeletionOrInsertion, Translocation, Duplication, Inversion]
    sample_bnx_file: str
    reference_bnx_file: str
    sample_subsample: float
    reference_subsample: float
    subsampled_sample_bnx_file: str
    subsampled_reference_bnx_file: str

    @property
    def line(self):
        return self.SV.line


def detect_structural_variation_for_multiple_datasets(cmap_reference_file: str,
                                                      reference_bnx_file: str,
                                                      sv_candidate_bnx_files: List[str],
                                                      sv_subsample_ratio: float = 1.0,
                                                      reference_subsample_ratio: float = 1.0,
                                                      nbits: int = 64, segment_length: int = 275,
                                                      zoom_factor: int = 500,
                                                      minimum_molecule_length: int = 150_000,
                                                      distance_threshold: float = 1.7,
                                                      unspecific_sv_threshold: float = 10.0,
                                                      density_filter: int = 40) -> List[SvResult]:
    reference_molecules_on_chromosomes: MoleculesOnChromosomes = \
        filter_and_prepare_molecules_on_chromosomes(
            cmap_reference_file,
            reference_bnx_file,
            subsample_ratio=reference_subsample_ratio,
            nbits=nbits,
            segment_length=segment_length,
            zoom_factor=zoom_factor,
            minimum_molecule_length=minimum_molecule_length,
            distance_threshold=distance_threshold,
            density_filter=density_filter
        )
    svs_found: List[SvResult] = list()
    for sv_candidate_bnx_file in sv_candidate_bnx_files:
        output_file = open(sv_candidate_bnx_file + ".SV_results.tsv", "w")
        sv_candidate_molecules_on_chromosomes: MoleculesOnChromosomes = \
            filter_and_prepare_molecules_on_chromosomes(
                cmap_reference_file,
                sv_candidate_bnx_file,
                subsample_ratio=sv_subsample_ratio,
                nbits=nbits,
                segment_length=segment_length,
                zoom_factor=zoom_factor,
                minimum_molecule_length=minimum_molecule_length,
                distance_threshold=distance_threshold,
                density_filter=density_filter
            )
        for unspecific_sv in find_unspecific_sv_sites(reference_molecules_on_chromosomes,
                                                      sv_candidate_molecules_on_chromosomes,
                                                      z_thr=unspecific_sv_threshold):
            if unspecific_sv.region:
                svs_found.append(
                    SvResult(
                        find_specific_sv(unspecific_sv,
                                         reference_molecules_on_chromosomes,
                                         sv_candidate_molecules_on_chromosomes),
                        sample_bnx_file=sv_candidate_bnx_file,
                        sample_subsample=sv_subsample_ratio,
                        reference_bnx_file=reference_bnx_file,
                        reference_subsample=reference_subsample_ratio,
                        subsampled_sample_bnx_file=f"{sv_candidate_bnx_file}.{sv_subsample_ratio}",
                        subsampled_reference_bnx_file=f"{reference_bnx_file}.{reference_subsample_ratio}"
                    )
                )
                output_file.write(svs_found[-1].line)
        output_file.close()
    return svs_found


def get_subsampled_bnx_lines(bnx_path: str,
                             minimum_molecule_length: int,
                             subsample_ratio: float):
    bnx: utils.BnxParser = utils.BnxParser(bnx_path)
    bnx.read_bnx_file()
    lines = random.choices(
        [bnx.bnx_arrays[i] for i in range(len(bnx.bnx_arrays))
         if float(bnx.bnx_arrays[i]["info"][2]) >= minimum_molecule_length],
        k=int(len(bnx.bnx_arrays) * subsample_ratio)
    )
    used_molecule_ids = [int(x["info"][1]) for x in lines]
    array_dict = get_array_dict(bnx)
    bnx.bnx_arrays = [array_dict[x] for x in used_molecule_ids]
    bnx.write_arrays_as_bnx(f"{bnx_path}.{subsample_ratio}")
    return lines


def molecule_generator_from_bnx_lines(bnx_lines, reverse: bool, segment_length: int, zoom_factor: int, nbits: int) -> \
        Iterator[MoleculeSeg]:
    for line in bnx_lines:
        yield MoleculeSeg.from_bnx_line(line, reverse=reverse, segment_length=segment_length,
                                        zoom_factor=zoom_factor, nbits=nbits)


def get_array_dict(bnx_obj: utils.BnxParser):
    return {int(x["info"][1]): x for x in bnx_obj.bnx_arrays}



if __name__ == "__main__":
    pass
