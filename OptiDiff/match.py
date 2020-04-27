import numpy as np
import numba as nb
from OptiDiff.simulate import bnx_to_signal as bts
from scipy import stats
from subprocess import check_call as ck


class Seeder(bts.CmapToSignal, bts.BnxToSignal):
    def __init__(self, cmap_path, bnx_path, length=100, n_bits=16, zoom_factor=500, subsample=1):
        self.n_bits = n_bits
        bts.CmapToSignal.__init__(self, cmap_path)
        bts.BnxToSignal.__init__(self, bnx_path, subsample=subsample)
        self.length = length
        self.bnx_arrays = self.bnx_arrays
        self.simulate_log_signals()
        self.simulate_all_chrs(zoom_factor=zoom_factor)
        self.create_mol_segments(length=length)
        self.create_chr_segments(length=length)
        self.mol_compress(length=length, nbits=n_bits)
        self.chr_compress(length=length, nbits=n_bits)
        self.matched_molecule_info = dict()
        self.matched_molecule_info_rev = dict()

    def get_mol_bits(self, mol_id):
        start, end = self.molecule_segment_indices[mol_id]
        return self.molecule_lsh.search_results[start:end]

    def mol_to_matched_segments(self, mol_id, thr=1):
        mol_bits = self.get_mol_bits(mol_id)
        mol_bits = mol_bits.view("uint8").reshape((mol_bits.shape[0], -1))
        chr_bits = self.chromosome_lsh.search_results.view("uint8").reshape(
            (self.chromosome_lsh.search_results.shape[0], -1))
        res = np.zeros(mol_bits.shape[0], dtype=int)
        _all = comp_bins(mol_bits, chr_bits)
        for i in range(mol_bits.shape[0]):
            if thr != 0:
                found = np.where(_all[i].flatten() <= (np.array(self.chromosome_segment_density)/thr)[:_all[i].flatten().shape[0]])[0]
            else:
                found = np.where(_all[i].flatten() == 0)[0]
            if len(found):
                res[i] = found[0]
            else:
                res[i] = -1
        return res[res >= 0]

    def mol_to_matched_segments_with_indices(self, mol_id, thr=1):
        mol_bits = self.get_mol_bits(mol_id)
        mol_bits = mol_bits.view("uint8").reshape((mol_bits.shape[0], -1))
        chr_bits = self.chromosome_lsh.search_results.view("uint8").reshape(
            (self.chromosome_lsh.search_results.shape[0], -1))
        res = np.zeros(mol_bits.shape[0], dtype=int)
        _all = comp_bins(mol_bits, chr_bits)
        for i in range(mol_bits.shape[0]):
            if thr != 0:
                found = np.where(_all[i].flatten() <= (np.array(self.chromosome_segment_density)/thr)[:_all[i].flatten().shape[0]])[0]
            else:
                found = np.where(_all[i].flatten() == 0)[0]
            if len(found):
                res[i] = found[0]
            else:
                res[i] = -1
        return res[res >= 0], np.where(res >= 0)[0]

    def mol_to_matched_segmentsv2(self, mol_id, thr=1):
        mol_bits = self.get_mol_bits(mol_id)
        mol_bits = mol_bits.view("uint8").reshape((mol_bits.shape[0], -1))
        chr_bits = self.chromosome_lsh.search_results.view("uint8").reshape(
            (self.chromosome_lsh.search_results.shape[0], -1))
        res = np.zeros(mol_bits.shape[0], dtype=int)
        _all = comp_bins(mol_bits, chr_bits)
        for i in range(mol_bits.shape[0]):
            found = np.where(_all[i].flatten() <= thr)[0]
            if len(found):
                res[i] = found[0]
            else:
                res[i] = -1
        return res[res >= 0]

    def molecule_to_edges_v2(self, mol_id, reference_mol=False, thr=1):
        matched_segments = self.mol_to_matched_segments(mol_id, thr=thr)
        if matched_segments.shape[0] < 2:
            return set()
        edges = set()
        segment_distances = [self.position_index[1][x] for x in matched_segments]
        for i in range(matched_segments.shape[0] - 1):
            if reference_mol and (0 < (segment_distances[i + 1] - segment_distances[i]) <= 50):
                edges.add((matched_segments[i],
                           matched_segments[i + 1]))
            elif not reference_mol:
                edges.add((matched_segments[i],
                           matched_segments[i + 1]))
            else:
                continue
        for n1, n2 in edges:
            n1 = self.position_index[1][n1]
            n2 = self.position_index[1][n2]
            if n1 in self.matched_molecule_info:
                self.matched_molecule_info[n1].append(mol_id)
            else:
                self.matched_molecule_info[n1] = [mol_id]
            if n2 in self.matched_molecule_info:
                self.matched_molecule_info[n2].append(mol_id)
            else:
                self.matched_molecule_info[n2] = [mol_id]
        self.matched_molecule_info_rev[mol_id] = {self.position_index[1][n1] for (n1, n2) in edges} | {self.position_index[1][n2] for (n1, n2) in edges}
        return edges

    def cut_molecules_at_loc(self, mol_id, loc_id, reference_mol=False, thr=1):
        matched_segments, idx = self.mol_to_matched_segments_with_indices(mol_id, thr=thr)
        if matched_segments.shape[0] < 2:
            return set()
        segment_pos = np.where(matched_segments == loc_id)[0]
        if segment_pos:
            mol_bits = self.get_mol_bits(mol_id)
            return (mol_bits[:segment_pos[0]], mol_bits[segment_pos[0]:])
        else:
            return ([], [])

    def molecule_to_chr(self, mol_id, chr_pos, chr_neg, thr=1, reference_mol=False):
        mol_bits = self.get_mol_bits(mol_id)
        mol_bits = mol_bits.view("uint8").reshape((mol_bits.shape[0], -1))
        chr_bits = self.chromosome_lsh.search_results.view("uint8").reshape(
            (self.chromosome_lsh.search_results.shape[0], -1))
        all_res = comp_bins(mol_bits, chr_bits)
        mol_edges = set()
        for i in range(mol_bits.shape[0] - 1):
            # current = mol_bits[i].reshape((1, -1))
            # _next = mol_bits[i+1].reshape((1, -1))
            res1 = (np.where(all_res[i].flatten() <= thr)[0])
            res2 = (np.where(all_res[i + 1].flatten() <= thr)[0] - 1)
            if (not reference_mol) and ((res1.shape[0] == 1) and (res2.shape[0] == 1)):
                mol_edges.add((res1[0], res2[0] + 1))
            elif reference_mol and ((res1.shape[0] == 1) and (res2.shape[0] == 1)) and (res1[0] <= res2[0]) and (
                    (res2[0] - res1[0]) <= 10):
                mol_edges.add((res1[0], res2[0] + 1))
            if (not res1.shape[0]) and (not res2.shape[0]):
                continue
            else:
                indices, counts = np.unique(np.sort(np.concatenate((res1, res2))), return_counts=True)
                chr_pos[indices[np.where(counts == 2)[0]]] += 2
                chr_pos[indices[np.where(counts == 2)[0]] + 1] += 2
                if np.where(counts < 2)[0].shape[0] == 1:
                    chr_neg[indices[np.where(counts < 2)[0]]] += 2
                else:
                    chr_neg[indices[np.where(counts < 2)[0]]] += np.std(indices[np.where(counts < 2)[0]])
        return chr_pos, chr_neg, mol_edges

    def molecules_to_chr(self, thr=1, reference_mols=False):
        edges = {}
        pos_res = np.zeros(self.chromosome_lsh.search_results.shape[0])
        neg_res = np.zeros(self.chromosome_lsh.search_results.shape[0])
        for i in range(len(self.molecule_segment_indices)):
            pos_res, neg_res, mol_edges = self.molecule_to_chr(i, pos_res, neg_res,
                                                               thr=thr, reference_mol=reference_mols)
            if len(mol_edges):
                edges[i] = mol_edges
        return pos_res, neg_res, edges

    def molecules_to_chr_v2(self, thr=1, reference_mols=False):
        edges = {}
        for i in range(len(self.molecule_segment_indices)):
            mol_edges = self.molecule_to_edges_v2(i, thr=thr, reference_mol=reference_mols)
            if len(mol_edges):
                edges[i] = mol_edges
            else:
                continue
        return edges


class MatchToRef:
    def __init__(self, ref_seed: Seeder, match_seed: Seeder, thr=15):
        self.reference = ref_seed
        self.seed = match_seed
        self.reference_edges = self.reference.molecules_to_chr_v2(thr=thr, reference_mols=False)
        self.match_edges = self.seed.molecules_to_chr_v2(thr=thr, reference_mols=False)
        self.mean_match_edge_freq = np.mean(list(self._get_match_edge_freq().values()))
        self.mean_ref_edge_freq = np.mean(list(self._get_ref_edge_freq().values()))

    def _get_match_edge_freq(self, normalize=True, min_coverage=4):
        all_edges = dict()
        total_edges = 0
        for edges in self.match_edges.values():
            for edge in edges:
                if edge not in all_edges:
                    all_edges[edge] = 1
                else:
                    all_edges[edge] += 1
                total_edges += 1
        # for edge in all_edges:
        #     all_edges[edge] /= total_edges
        return all_edges
        # if normalize:
        #     total = np.sum(list(all_edges.values()))
        #     new_edges = {}
        #     for edge in all_edges:
        #         if all_edges[edge] >= min_coverage:
        #             new_edges[edge] = all_edges[edge] / total
        #     return new_edges
        # else:
        #     new_edges = {}
        #     for edge in all_edges:
        #         if all_edges[edge] >= min_coverage:
        #             new_edges[edge] = all_edges[edge]
        #     return new_edges

    def _get_ref_edge_freq(self, normalize=True, min_coverage=4):
        all_edges = dict()
        total_edges = 0
        for edges in self.reference_edges.values():
            for edge in edges:
                if edge not in all_edges:
                    all_edges[edge] = 1
                else:
                    all_edges[edge] += 1
                total_edges += 1
        # for edge in all_edges:
        #     all_edges[edge] /= total_edges
        return all_edges
        # if normalize:
        #     total = np.sum(list(all_edges.values()))
        #     new_edges = {}
        #     for edge in all_edges:
        #         if all_edges[edge] >= min_coverage:
        #             new_edges[edge] = all_edges[edge] / total
        #     return new_edges
        # else:
        #     new_edges = {}
        #     for edge in all_edges:
        #         if all_edges[edge] >= min_coverage:
        #             new_edges[edge] = all_edges[edge]
        #     return new_edges

    def _get_all_ref_edges(self):
        ref_edges = set()
        for edges in self.reference_edges.values():
            ref_edges |= set(edges)
        return ref_edges

    def reference_minus_samples(self, zscore_thr=2.5):
        ref_edges = self._get_ref_edge_freq()
        match_edges = self._get_match_edge_freq()
        missing_edges = dict()
        for edge, n in ref_edges.items():
            if edge in match_edges:
                missing_edges[edge] = max(1, ref_edges[edge] / match_edges[edge]) * max(0, ref_edges[edge] - match_edges[edge])
            else:
                missing_edges[edge] = ref_edges[edge]**2
        edge_ids = list(missing_edges.keys())
        data = normalize_between(0, 200, np.array(list(missing_edges.values())))
        # data = np.array(list(missing_edges.values()))
        # data = data/10
        filtered_idx = np.where(stats.zscore(data) >= zscore_thr)[0]
        print(data[filtered_idx])
        top_edges = [edge_ids[filtered_idx[i]] for i in np.argsort(data[filtered_idx])[::-1]]
        return missing_edges, top_edges

    def reference_minus_samples_debug(self, zscore_thr=2.5):

        ref_edges = self._get_ref_edge_freq()
        match_edges = self._get_match_edge_freq()
        missing_edges = dict()

        for edge, n in ref_edges.items():
            if edge in match_edges:
                missing_edges[edge] = max(1, ref_edges[edge] / match_edges[edge]) * max(0, ref_edges[edge] - match_edges[edge])
            else:
                missing_edges[edge] = ref_edges[edge]**2
        edge_ids = list(missing_edges.keys())
        data = normalize_between(0, 200, np.array(list(missing_edges.values())))
        # data = np.array(list(missing_edges.values()))
        # data = data/10
        filtered_idx = np.where(stats.zscore(data) >= zscore_thr)[0]
        print(data[filtered_idx])
        top_edges = [edge_ids[filtered_idx[i]] for i in np.argsort(data[filtered_idx])[::-1]]
        return np.array(edge_ids), np.array(list(missing_edges.values())), data, stats.zscore(data)

    def sample_minus_reference(self, zscore_thr=2.5):
        ref_edges = self._get_ref_edge_freq()
        match_edges = self._get_match_edge_freq()
        missing_edges = dict()
        for edge, n in match_edges.items():
            if edge in ref_edges:
                missing_edges[edge] = max(0, match_edges[edge] - ref_edges[edge])
            else:
                missing_edges[edge] = match_edges[edge]
        edge_ids = list(missing_edges.keys())
        top_edges = [edge_ids[i] for i in np.where(stats.zscore(list(missing_edges.values())) >= zscore_thr)[0]]
        return missing_edges, top_edges

    def get_insertion_edges(self):
        ref_edges = self._get_all_ref_edges()
        match_edges = self._get_match_edge_freq()
        return {x: ref_edges[x] for x in set(list(match_edges.keys())) - ref_edges}

    def get_top_deletion_index(self, top=3):
        del_indices = list(sorted(self.get_deletion_edges().items(), key=lambda x: x[1], reverse=True))[:top]
        return [(self.reference.position_index[1][i1] / 2,
                 self.reference.position_index[1][i2] / 2) for (i1, i2), n in del_indices]

    def find_duplicated(self, zscore_thr=3.5, perc=40, max_jump_distance=5):
        ref_freq, match_freq = self._get_ref_edge_freq(normalize=False), self._get_match_edge_freq(normalize=False)
        dups = list()
        median = np.percentile(list(ref_freq.values()), perc)
        for edge in ref_freq:
            if edge in match_freq:
                if (ref_freq[edge] >= median) and (abs(edge[0] - edge[1]) <= max_jump_distance):
                    dups.append((self.reference.position_index[1][edge[0]] * 500,
                                 self.reference.position_index[1][edge[1]] * 500 + self.reference.length,
                                 match_freq[edge] - ref_freq[edge]))
        data = normalize_between(0, 100, [x[-1] for x in dups])
        zscores = stats.zscore(data)
        argsorted_zscores = np.argsort(zscores)[::-1]
        print(zscores[argsorted_zscores])
        return [dups[i] for i in argsorted_zscores if zscores[i] >= zscore_thr]

    def test_deletion2(self, deletion_range: tuple, zscore_thr=2.5):
        del_start, del_end = max(0, deletion_range[0] - self.seed.length), deletion_range[1]
        missing_edges, top_edges = self.reference_minus_samples(zscore_thr=zscore_thr)
        tp_hits = 0
        # print(missing_edges, top_edges)
        if len(top_edges) <= 0:
            return "FN"
        for _from, _to in top_edges:
            start = int(self.reference.position_index[1][_from])
            end = int(self.reference.position_index[1][_to])
            # print(start*500, end*500)
            if len(set(range(start, end, 1)).intersection(set(range(del_start, del_end, 1)))):
                # print(True)
                tp_hits += 1
            else:
                print(False)
        if tp_hits:
            print(tp_hits, len(top_edges))
            return "TP"
        else:
            print(len(top_edges))
            return "FP"

    def test_deletion(self, deletion_range: tuple, zscore_thr=2.5):
        del_start, del_end = max(0, deletion_range[0] - self.seed.length), deletion_range[1]
        missing_edges, top_edges = self.reference_minus_samples(zscore_thr=zscore_thr)
        print(len(missing_edges), len(top_edges))
        if len(top_edges) <= 0:
            return "FN"
        for _from, _to in top_edges:
            start = int(self.reference.position_index[1][_from])
            end = int(self.reference.position_index[1][_to])
            if len(set(range(start, end, 1)).intersection(set(range(del_start, del_end, 1)))):
                return "TP"
        else:
            return "FP"

    def test_negative(self, zscore_thr=2.5):
        missing_edges, top_edges = self.reference_minus_samples(zscore_thr=zscore_thr)
        print(len(missing_edges), len(top_edges))
        if len(top_edges):
            return "FP"
        else:
            return "TN"


def normalize_between(new_min, new_max, data):
    old_min, old_max = np.min(data), np.max(data)
    return np.array([(x-old_min) * ((new_max - new_min)/(old_max - old_min)) for x in data])


def test_deletions_from_bnx_files(reference_seed: Seeder, list_of_bnx_files, zscore_thr=2.5, match_score_thr=9):
    test_results = list()
    for f in list_of_bnx_files:
        _f = f.split("temp_svd_del")[-1]
        try:
            _f = [int(x) // 500 for x in _f.split("-del.fasta.bnx")[0].split("-")]
        except ValueError:
            _f = [int(x) // 500 for x in _f.split("-retry-del.fasta.bnx")[0].split("-")]
        _f = (_f[0] - reference_seed.length, _f[1])
        svd_seed = Seeder(reference_seed.cmap_file_location, f,
                          length=reference_seed.length,
                          n_bits=reference_seed.n_bits)
        m = MatchToRef(reference_seed, svd_seed,
                       thr=match_score_thr)
        if (_f[1] * 500) > 4600000:
            print(_f[1] * 500)
            print(False)
            continue
        else:
            res = m.test_deletion2(_f, zscore_thr=zscore_thr)
            test_results.append((res, abs((_f[0] - _f[1]) * 500), f))
            print((res, abs((_f[0] - _f[1]) * 500), f))
    return test_results

def negatives(reference_seed: Seeder, list_of_bnx_files, zscore_thr=2.5, match_score_thr=9):
    test_results = list()
    for f in list_of_bnx_files:
        svd_seed = Seeder(reference_seed.cmap_file_location, f,
                          length=reference_seed.length,
                          n_bits=reference_seed.n_bits)
        m = MatchToRef(reference_seed, svd_seed,
                       thr=match_score_thr)

        res = m.test_negative(zscore_thr=zscore_thr)
        test_results.append((res, f))
    return test_results

class BngBench:
    def __init__(self, refaligner_path, pipeline_path, reference_cmap,
                 list_of_svd_bnx_files, temp_folder_path, pipeline_arguments_path):
        self.refaligner_path = refaligner_path
        self.pipeline_path = pipeline_path
        self.ref_cmap = reference_cmap
        self.svd_bnx_files = list_of_svd_bnx_files
        self.temp_path = temp_folder_path
        self.args_xml = pipeline_arguments_path
        self.smaps = list()

    def run_bng(self, bnx_filename):
        command = f"python2 {self.pipeline_path} -T 15 -j 15 -l {self.temp_path} -t {self.refaligner_path} -b" \
                  f" {bnx_filename} -r {self.ref_cmap} -y -i 1 -a {self.args_xml}"
        ck(command, shell=True)
        # /home/bionanosolve/tools/pipeline/1.0/Pipeline/1.0/pipelineCL.py

    def collect_smap(self):
        smap_path = f"{self.temp_path}/contigs/exp_refineFinal1_sv/EXP_REFINEFINAL1.smap"
        smap = open(smap_path, "r")
        self.smaps.append(smap.read())
        smap.close()

    def remove_contents(self):
        ck(f"rm -r {self.temp_path}", shell=True)
        ck(f"mkdir {self.temp_path}", shell=True)

    def collect_all_smap(self):
        for f in self.svd_bnx_files:
            self.run_bng(f)
            self.collect_smap()
            self.remove_contents()


# def remove_false_match_segments(mol_segments):
#     for i in range(mol_segments.shape[0] - 1):
#         if mol_segments[i] > mol_segments[i + 1]:
#             mol_segments = np.concatenate((mol_segments[:i], mol_segments[i + 1:]))
#             return remove_false_match_segments(mol_segments)
#     return mol_segments


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


@nb.njit
def countSetBits(n):
    count = 0
    while (n):
        count += n & 1
        n >>= 1
    return count


if __name__ == "__main__":
    refaligner_path = "/mnt/LTR_userdata/akdel001/phd/Tools/bionano_solve_2019/Solve3.4_06042019a/RefAligner/8949.9232rel"
    pipeline_path = "/mnt/LTR_userdata/akdel001/phd/Tools/bionano_solve_2019/Solve3.4_06042019a/Pipeline/06042019/pipelineCL.py"
    reference_cmap = "/home/akdel001/svd_test_bnx/temp_chr4.cmap"
    from os import listdir as ls
    list_of_svd_bnx_files = ["/home/akdel001/svd_test_bnx/" + x for x in ls("/home/akdel001/svd_test_bnx/") if x.endswith(".bnx")]
    temp_folder_path = "/home/akdel001/svd_test_bnx/temp_folder/"
    pipeline_arguments_path = "/home/akdel001/svd_test_bnx/exp_optArguments.xml"
    bng = BngBench(refaligner_path, pipeline_path, reference_cmap,
                   list_of_svd_bnx_files, temp_folder_path, pipeline_arguments_path)
    bng.collect_all_smap()
    pass


class Nodes:
    def __init__(self, node_properties):
        self.nodes = node_properties

    @classmethod
    def from_distances(cls, nodes, distances):
        return dict.fromkeys(nodes, distances)


class SpatialGraph:
    def __init__(self, edges: tuple, node_properties: Nodes):
        self.node_properties = node_properties
        self.edges = edges

