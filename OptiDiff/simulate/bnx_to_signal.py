from OptiScan import utils
from typing import List, Tuple, Dict
import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
from LSH import lsh
from dataclasses import dataclass
import numba as nb
from scipy import stats, signal


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


class BnxToLabels(utils.BnxParser):
    def __init__(self, bnx_path):
        utils.BnxParser.__init__(self, bnx_path)
        self.read_bnx_file()

    def generate_molecule_labels(self):
        for molecule_id in range(len(self.bnx_arrays)):
            yield list(self.bnx_arrays[molecule_id]["labels"])


class CmapToLabels(utils.CmapParser):
    def __init__(self, cmap_path):
        utils.CmapParser.__init__(self, cmap_path)
        self.read_and_load_cmap_file()
        self.get_position_indexes()

    def generate_cmap_labels(self):
        for chr_id in self.position_index.keys():
            yield np.unique((np.array(self.position_index[chr_id])).astype(int))


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

    @classmethod
    def from_bnx_line(cls, bnx_array_entry: dict, reverse=False,
                      segment_length: int = 200, zoom_factor: int = 500,
                      nbits=64):
        index = int(bnx_array_entry["info"][1])
        length = float(bnx_array_entry["info"][2])
        labels = (np.array(bnx_array_entry["labels"]) / zoom_factor).astype(int)
        sig = np.zeros(int(length) // zoom_factor + 1)
        sig[labels] = 5000.
        log_sig = np.log1p(ndimage.gaussian_filter1d(sig, sigma=1))
        if reverse:
            log_sig = log_sig[::-1]
            labels = np.array([log_sig.shape[0] - x for x in labels])
            index *= -1
        segments, label_density = get_segments(segment_length, log_sig, labels)
        return cls(index, length, np.array(segments), label_density,
                   zoom_factor=zoom_factor, segment_length=segment_length,
                   nbits=nbits)

    def compress(self):
        def create_randoms(nbits=self.nbits, l=self.segment_length):
            randoms = np.zeros((nbits, l))
            steps = l / nbits
            for i in range(1, nbits):
                x = np.zeros(l)
                x[int(i * steps):int(i * steps + steps)] = l
                randoms[i] = x - np.mean(x)
            return randoms

        randoms = create_randoms()
        self.compressed_segments = lsh.VectorsInLSH(self.nbits, self.segments, custom_table=randoms).search_results


def compress_molecule_segments(segmented_molecules: List[MoleculeSeg]):
    all_segments = np.vstack([x.segments for x in segmented_molecules])


def get_segments(segment_length, signal, labels):
    segments = [signal[i:i + segment_length] for i in labels
                if signal[i:i + segment_length].shape[0] == segment_length]
    label_density = [len([x for x in labels if i < x < i + segment_length]) for i in labels
                     if signal[i:i + segment_length].shape[0] == segment_length]
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
        self.nbits = 64
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
    def from_cmap_signals(cls, cmap_signals: CmapToSignal, chromosome_id: int):
        if cmap_signals.prepared:
            start, end = cmap_signals.chromosome_segment_indices[chromosome_id - 1]
            compressed_segments = cmap_signals.chromosome_lsh.search_results[start:end]
            kb_segment_indices = cmap_signals.position_index[chromosome_id] / 2
            segment_graph = dict()
            label_densities = dict()
            for i, segment in enumerate(compressed_segments):
                if segment not in segment_graph:
                    segment_graph[segment] = [i]
                    label_densities[segment] = cmap_signals.chromosome_segment_density[chromosome_id][i]
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


@dataclass
class Scores:
    molecule_id: int
    chromosome_id: int
    segment_matches: Dict[int, np.ndarray]

    @classmethod
    def from_molecule_and_chromosome(cls, molecule: MoleculeSeg, chromosome: ChromosomeSeg, distance_thr: float = 1.8):
        assert molecule.nbits == chromosome.nbits
        assert molecule.segment_length == chromosome.segment_length
        if molecule.segments.shape[0] <= 1:
            return Scores(molecule.index, chromosome.index, {})
        if molecule.compressed_segments is None:
            molecule.compress()
        mol_bits = molecule.compressed_segments.view("uint8").reshape((molecule.compressed_segments.shape[0], -1))
        chr_bits = chromosome.segments.view("uint8").reshape((chromosome.segments.shape[0], -1))
        distances = comp_bins(mol_bits, chr_bits)
        segment_matches = dict()
        for i in range(mol_bits.shape[0]):
            densities = (chromosome.label_densities[:distances.shape[1]] + molecule.label_densities[
                i]) / 2 / distance_thr
            matching_segment_indices = list(np.where(distances[i] <= densities)[0])
            if len(matching_segment_indices):
                segment_matches[i] = np.concatenate(
                    [chromosome.compressed_segment_graph[chromosome.segments[i]] for i in matching_segment_indices])
            else:
                continue
        return Scores(molecule.index, chromosome.index, segment_matches)

    def proceeding(self, i: int):
        assert i in self.segment_matches
        keys = list(self.segment_matches.keys())
        try:
            next_key = keys[keys.index(i) + 1]
            return next_key, self.segment_matches[next_key]
        except KeyError and IndexError:
            return -1, -1

    def get_best_path(self):
        if len(self.segment_matches) <= 1:
            return []
        first_key = list(self.segment_matches.keys())[0]
        current = [(first_key, i) for i in self.segment_matches[first_key]]
        visited = {(k, i): (0, [i]) for (k, i) in current}
        target_length = len(self.segment_matches)
        while len(current):
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


@dataclass
class MoleculeSegmentPath:
    molecule_id: int
    forward_paths: \
        Dict[int, List[int]]  # keys are chromosome ids and values are lists of paths as segment ids as nodes.
    reverse_paths: Dict[int, List[int]]


def segment_paths_from_scores(scores: List[Scores]) -> List[MoleculeSegmentPath]:
    molecules: Dict[int, MoleculeSegmentPath] = dict()
    for score in scores:
        if abs(score.molecule_id) not in molecules:
            if score.molecule_id > 0:
                molecules[abs(score.molecule_id)] = MoleculeSegmentPath(abs(score.molecule_id),
                                                                        {score.chromosome_id: score.get_best_path()},
                                                                        {})
            else:
                molecules[abs(score.molecule_id)] = MoleculeSegmentPath(abs(score.molecule_id), {},
                                                                        {score.chromosome_id: score.get_best_path()})
        else:
            if score.molecule_id > 0:
                molecules[abs(score.molecule_id)].forward_paths[score.chromosome_id] = score.get_best_path()
            else:
                molecules[abs(score.molecule_id)].reverse_paths[score.chromosome_id] = score.get_best_path()
    return list(molecules.values())


@dataclass
class MoleculesOnChromosomes:
    molecules: Dict[int, MoleculeSegmentPath]
    chromosomes: Dict[int, ChromosomeSeg]
    counts_per_segment: Dict[int, Tuple[np.ndarray, np.ndarray]]
    molecules_per_segment: Dict[int, List[List[int]]]

    @classmethod
    def from_molecules_and_chromosomes(cls,
                                       molecules: List[MoleculeSeg],
                                       chromosomes: List[ChromosomeSeg],
                                       distance_thr: float = 1.8):
        scores: List[Scores] = [Scores.from_molecule_and_chromosome(y, x, distance_thr=distance_thr) for y in molecules
                                for x in chromosomes]
        molecule_segment_paths: List[MoleculeSegmentPath] = segment_paths_from_scores(scores)
        chromosomes_dict: Dict[int, ChromosomeSeg] = {chromosome.index: chromosome for chromosome in chromosomes}
        molecules_per_segment: Dict[int, List[int]] = {x: list() for x in chromosomes_dict.keys()}
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
                   chromosomes_dict, counts_per_segment, molecule_ids_per_segment)


@dataclass
class UnspecificSV:
    region: Tuple[int, int, int]  # (chrid, kb_start, kb_end)
    score: float
    reference_molecules: List[MoleculeSegmentPath]
    sv_candidate_molecules: List[MoleculeSegmentPath]


def find_unspecific_sv_sites(reference: MoleculesOnChromosomes, sv_candidate: MoleculesOnChromosomes) -> List[UnspecificSV]:
    result: List[UnspecificSV] = list()
    for chr_id in reference.counts_per_segment.keys():
        assert (chr_id in reference.counts_per_segment) and (chr_id in sv_candidate.counts_per_segment)
        ratios = list()
        ref_ids, ref_counts = reference.counts_per_segment[chr_id]
        sv_ids, sv_counts = sv_candidate.counts_per_segment[chr_id]
        segments = dict()
        for i, segment_id in enumerate(ref_ids):
            segments[segment_id] = [ref_counts[i], 0]
        for i, segment_id in enumerate(sv_ids):
            try:
                segments[segment_id][1] += sv_counts[i]
            except KeyError:
                continue
        for ref_id in ref_ids:
            if segments[ref_id][1] < 1:
                ratios.append((reference.chromosomes[chr_id].kb_indices[ref_id], segments[ref_id][0]))
            else:
                ratios.append((reference.chromosomes[chr_id].kb_indices[ref_id], segments[ref_id][0]/segments[ref_id][1]))
        sig = stats.zscore([x[1] for x in sorted(ratios, key=lambda x: x[0])])
        snr_thr = max(1, np.median(sig)) * 3.5
        peak_indices = signal.find_peaks(sig)[0]
        peak_indices = peak_indices[sig[peak_indices] > snr_thr]
        for start, end in list({find_boundaries(sig, x) for x in peak_indices}):
            start_kb = reference.chromosomes[chr_id].kb_indices[start] - (reference.chromosomes[chr_id].segment_length*2)
            end_kb = reference.chromosomes[chr_id].kb_indices[end] + reference.chromosomes[chr_id].segment_length
            proximal_segment_ids = [i for (i, x) in enumerate(reference.chromosomes[chr_id].kb_indices) if start_kb < x <= end_kb]
            proximal_chr_molecule_ids = np.unique(np.concatenate([reference.molecules_per_segment[chr_id][i] for i in proximal_segment_ids]))
            proximal_sv_molecule_ids = np.unique(np.concatenate([sv_candidate.molecules_per_segment[chr_id][i] for i in proximal_segment_ids]))
            reference_molecules = [reference.molecules[k] for k in proximal_chr_molecule_ids]
            sv_molecules = [sv_candidate.molecules[k] for k in proximal_sv_molecule_ids]
            result.append(UnspecificSV((chr_id, start, end), snr_thr, reference_molecules, sv_molecules))
    return result


def find_boundaries(sig, peak, snr=1.5):
    snr_thr = max(1, np.median(sig)) * snr
    start = end = peak
    for end in range(peak, len(sig)-1):
        if sig[end] < snr_thr:
            continue
        else:
            break
    for start in list(range(0, peak))[::-1]:
        if sig[start] < snr_thr:
            continue
        else:
            break
    return start, end


@dataclass
class Translocation:
    origin: Tuple[int, int, int]
    inserted_region: Tuple[int, int, int]
    duplication_ratios: Dict[int, np.ndarray]


@dataclass
class Duplication:
    translocation: Translocation
    ttest_score: Tuple[float, float]


@dataclass
class Inversion:
    ratios: Dict[int, np.ndarray]


def find_specific_sv(unspecific_sv: UnspecificSV, reference: MoleculesOnChromosomes, candidate: MoleculesOnChromosomes):
    # First check translocation
        # then check duplication
            # translocation if false
            # duplication if true
            # Check inversions
                # if true then add inversion to translocation/duplication
    # Check inversion
        # if true then return inversion
    # if none of the above then SV < seed size or deletion > 1kb
    pass


def from_fasta(fasta_path, digestion_motif="GCTCTTC", enzyme_name="BSPQ1", channel="1"):
    temp_name = "temp.cmap"
    fasta_to_cmap(fasta_path, temp_name, digestion_motif=digestion_motif, enzyme_name=enzyme_name, channel=channel)
    cmaptosignal = CmapToSignal(temp_name)
    return cmaptosignal


def fasta_to_cmap(fasta_in_path, cmap_out_path, digestion_motif="GCTCTTC",
                  enzyme_name="BSPQ1", channel="1"):
    fasta = utils.FastaObject(fasta_in_path)
    fasta.initiate_fasta_array()
    fasta.fill_complete_fasta_array()
    fasta.write_fasta_to_cmap(digestion_sequence=digestion_motif, output_file_name=cmap_out_path,
                              enzyme_name=enzyme_name, channel=channel)


if __name__ == "__main__":
    bnx = utils.BnxParser(
        "/home/biridir/PycharmProjects/optidiff/data/SVs/temp_svd_dup1715773-2215773-1067172-dup.fasta.bnx")
    bnx.read_bnx_file()
    bnx_ref = utils.BnxParser("/home/biridir/PycharmProjects/optidiff/all_ref_5mb.fasta.bnx")
    bnx_ref.read_bnx_file()
    ms_ref = [MoleculeSeg.from_bnx_line(x, reverse=False, segment_length=200) for x in bnx_ref.bnx_arrays] + [
        MoleculeSeg.from_bnx_line(x, reverse=True, segment_length=200) for x in bnx_ref.bnx_arrays]
    ms = [MoleculeSeg.from_bnx_line(x, reverse=False, segment_length=200) for x in bnx.bnx_arrays] + [
        MoleculeSeg.from_bnx_line(x, reverse=True, segment_length=200) for x in bnx.bnx_arrays]
    cmap = CmapToSignal("/home/biridir/PycharmProjects/optidiff/temp_all_chr.cmap")
    cmap.prepare(segment_length=200)
    chrom1 = ChromosomeSeg.from_cmap_signals(cmap, 1)
    # chrom2 = ChromosomeSeg.from_cmap_signals(cmap, 1)
    # chrom2.index = 2
    chroms = [chrom1]
    res = MoleculesOnChromosomes.from_molecules_and_chromosomes(ms, chroms, distance_thr=1.8)
    res_ref = MoleculesOnChromosomes.from_molecules_and_chromosomes(ms_ref, chroms, distance_thr=1.8)
    print(find_unspecific_sv_sites(res_ref, res))
    # paths = [Scores.from_molecule_and_chromosome(m, chrom, distance_thr=1.8).get_best_path() for m in ms if
    #          len(m.segments) > 1]
    # paths = list(filter(lambda x: len(x) >= 2, paths))
    # ids_used = np.concatenate(paths)
    # ids_used, counts = np.unique(ids_used, return_counts=True)
    # print(len(paths) / (len(ms) // 2))
    # plt.scatter(x=chrom.kb_indices[ids_used], y=counts)
    # plt.show()
    # plt.hist(counts, bins=20)
    # plt.show()
