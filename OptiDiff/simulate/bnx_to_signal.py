from OptiScan import utils
from typing import List, Tuple, Dict
import numpy as np
from scipy import ndimage
# import matplotlib.pyplot as plt
from LSH import lsh
from dataclasses import dataclass


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
    molecule_length: int
    segments: np.ndarray
    label_densities: List[int]
    zoom_factor: int = 500
    segment_length: int = 100
    compressed_segments = None

    @classmethod
    def from_bnx_line(cls, bnx_array_entry: dict, reverse=False, segment_length: int = 100, zoom_factor: int = 500):
        index = int(bnx_array_entry["info"][1])
        length = int(bnx_array_entry["info"][2])
        labels = (np.array(bnx_array_entry["labels"]) / zoom_factor).astype(int)
        sig = np.zeros(length//zoom_factor)
        sig[labels] = 5000.
        log_sig = np.log1p(ndimage.gaussian_filter1d(sig, sigma=1))
        if reverse:
            log_sig = log_sig[::-1]
            labels = np.array([log_sig.shape[0]-x for x in labels])
            index *= -1
        segments, label_density = get_segments(segment_length, log_sig, labels)
        return cls(index, length, np.array(segments), label_density,
                   zoom_factor=zoom_factor, segment_length=segment_length)


def get_segments(segment_length, signal, labels):
    segments = [signal[i:i + segment_length] for i in labels
                if signal[i:i + segment_length].shape[0] == segment_length]
    label_density = [len([x for x in labels if i < x < i + segment_length]) for i in labels
                     if signal[i:i + segment_length].shape[0] == segment_length]
    return segments, label_density


class BnxToSignal(utils.BnxParser):
    def __init__(self, bnx_path, subsample=1):
        utils.BnxParser.__init__(self, bnx_path)
        self.read_bnx_file()
        self.subsample_bnx_arrays(subsample)
        self.molecule_segment_indices = list()
        self.molecule_segments = list()
        self.mol_segment_count = 0
        self.molecule_lsh = None
        self.molecule_count = len(self.bnx_arrays)
        self.segment_densities = list()
        self.filtered_molecules = list()

    def write_arrays_as_bnx(self, outname):
        """
        writes the bnx_arrays into file
        ie// this is useful if bnx_arrays are modified.
        """
        f = open(outname, "w")
        lines = [str(self.bnx_head_text)]
        i = 0
        for arr in self.bnx_arrays:
            i += 1
            lines.append("\t".join(list(map(str, arr["info"]))))
            lines.append("1\t" +  "\t".join([str(x) for x in arr["labels"]]))
            lines.append("QX11\t" +  "\t".join([str(x) for x in arr["label_snr"]]))
            lines.append("QX12\t" +  "\t".join([str(x) for x in arr["raw_intensities"]]))
        f.write("\n".join(lines) + "\n")
        f.close()

    def subsample_bnx_arrays(self, ratio):
        if ratio >= 1:
            self.bnx_arrays = self.bnx_arrays
        else:
            samples = np.random.choice(np.arange(len(self.bnx_arrays)), int(len(self.bnx_arrays) * ratio), replace=False)
            self.bnx_arrays = [self.bnx_arrays[x] for x in samples]

    def create_signal(self, molecule_id):
        molecule = self.bnx_arrays[molecule_id]
        indices = (np.array(molecule["labels"])/500.).astype(int)
        total_length = int(float(molecule["info"][2])/500.)
        sig = np.zeros(total_length+1, dtype=float)
        rev_sig = np.zeros(total_length+1, dtype=float)
        sig[indices] = 5000.
        rev_sig[np.array([rev_sig.shape[0]-x-1 for x in indices])] = 5000.
        log_sig = np.log1p(ndimage.gaussian_filter1d(sig, sigma=1))
        log_rev_sig = np.log1p(ndimage.gaussian_filter1d(rev_sig, sigma=1))
        self.bnx_arrays[molecule_id]["real_id"] = molecule_id
        self.bnx_arrays[molecule_id]["ori"] = "+"
        self.bnx_arrays[molecule_id]["signal"] = log_sig
        self.bnx_arrays[molecule_id]["labels"] = indices
        self.bnx_arrays[molecule_id]["info"][2] = total_length
        self.bnx_arrays.append({})
        self.bnx_arrays[-1]["real_id"] = molecule_id
        self.bnx_arrays[-1]["ori"] = "-"
        self.bnx_arrays[-1]["signal"] = log_rev_sig
        self.bnx_arrays[-1]["labels"] = np.array([rev_sig.shape[0]-x for x in indices])

    def simulate_log_signals(self):
        for i in range(len(self.bnx_arrays)):
            bnx_array = self.bnx_arrays[i]
            if len(bnx_array["labels"]) >=5:
                self.create_signal(i)
            else:
                continue

    def create_mol_segments(self, length=100):
        filtered_mols = list(filter(lambda x: np.sum(x["signal"]), self.bnx_arrays))
        self.mol_segment_count = 0

        cumulative = 0
        for molecule in filtered_mols:
            current_segments = [molecule["signal"][i:i+length] for i in molecule["labels"] if molecule["signal"][i:i+length].shape[0] == length]
            current_density = [len([x for x in molecule["labels"] if i < x < i + length]) for i in molecule["labels"] if molecule["signal"][i:i+length].shape[0] == length]
            if not len(current_segments):
                continue

            self.molecule_segments.append([molecule["signal"][i:i + length] for i in molecule["labels"] if molecule["signal"][i:i + length].shape[0] == length])
            self.segment_densities.append(current_density)
            self.molecule_segment_indices.append((cumulative, cumulative + len(self.molecule_segments[-1])))
            cumulative += len(self.molecule_segments[-1])
            self.mol_segment_count += len(self.molecule_segments[-1])
            self.filtered_molecules.append(molecule)
        self.molecule_segments = np.vstack(self.molecule_segments)

    def mol_compress(self, length=100, nbits=16, custom_table=np.array([])):
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

            self.molecule_lsh = lsh.VectorsInLSH(nbits, self.molecule_segments, custom_table=custom_table)
        else:
            self.molecule_lsh = lsh.VectorsInLSH(nbits, self.molecule_segments, custom_table=randoms)


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

    def prepare(self, zoom_factor=500, segment_length=200, nbits=64):
        self.simulate_all_chrs(zoom_factor=zoom_factor)
        self.create_chr_segments(length=segment_length)
        self.chr_compress(length=segment_length, nbits=nbits)
        self.segment_length = segment_length
        self.zoom_factor = zoom_factor
        self.prepared = True

    def generate_segment_from_bits(self, bits_id):
        bits = self.chromosome_lsh.search_results[bits_id:bits_id+1].view("uint8")
        return np.unpackbits(bits).flatten()

    def simulate_chr(self, chr_id, zoom_factor=500):
        self.position_index[chr_id] = np.unique((np.array(self.position_index[chr_id])//zoom_factor).astype(int))
        indices = self.position_index[chr_id]
        arr = np.zeros(np.max(indices[-1])+1, dtype=float)
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
    label_densities: List[int]
    kb_indices: np.ndarray
    zoom_factor: int
    segment_length: int
    compressed_segment_graph: Dict[bytes, List[int]]

    @classmethod
    def from_cmap_signals(cls, cmap_signals: CmapToSignal, chromosome_id: int):
        if cmap_signals.prepared:
            start, end = cmap_signals.chromosome_segment_indices[chromosome_id]
            compressed_segments = cmap_signals.chromosome_lsh[start:end]
            kb_segment_indices = cmap_signals.position_index[chromosome_id] / 2
            segment_graph = dict()
            for i, segment in enumerate(compressed_segments):
                if segment not in segment_graph:
                    segment_graph[segment] = [i]
                else:
                    segment_graph[segment].append(i)
        else:
            raise BrokenPipeError("Cmap signals not prepared")
        return cls(chromosome_id, kb_segment_indices[-1]*1000, cmap_signals.chromosome_segment_density[chromosome_id],
                   kb_segment_indices, cmap_signals.zoom_factor, cmap_signals.segment_length, segment_graph)


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
    bnx = BnxToSignal("/Users/akdel/PycharmProjects/OptiDiff/yeast_output.label_0.1.bnx")
    bnx.bnx_arrays = bnx.bnx_arrays[:100]
    bnx.simulate_log_signals()
    bnx.create_mol_segments()
    bnx.mol_compress()
    print(bnx.molecule_lsh.search_results)
    print(bnx.molecule_lsh.bin_ids_used.shape)
    # plt.plot(bnx.segments[0])
    # plt.show()
    # plt.plot(bnx.segments[-1])
    # plt.show()
    # cmap_stuff = from_fasta("/Users/akdel/PycharmProjects/OptiDiff/S288C_reference_sequence_R64-2-1_20150113.fsa")
