from OptiScan import utils
import numpy as np
import numba as nb
from subprocess import check_call as ck


class Fasta(utils.FastaObject):
    def __init__(self, fasta_path, digestion_sequence="GCTCTTC"):
        utils.FastaObject.__init__(self, fasta_path)
        self.initiate_fasta_array()
        self.fill_complete_fasta_array()
        self.indices = list()
        self.get_indices_from_lengths()
        self.digest_fasta_array(digestion_sequence)

    def get_indices_from_lengths(self):
        total = 0
        for i in range(len(self.lengths)):
            self.indices.append((total, self.lengths[i]+total))
            total += self.lengths[i]

    def write_single_chr(self, chr_id, fname="temp.fasta"):
        start, end = self.indices[chr_id]
        header = f"> {chr_id}\n"
        fasta_text = "".join(list(self.fasta_array[0][start:end].view("S1").astype("U1"))) + "\n"
        f = open(fname, "w")
        f.write(header)
        f.write(fasta_text)
        f.close()

    def write_all_chr(self,  fname="temp.fasta", lim=5000000):
        header = f"> all_chr\n"
        fasta_text = "".join(list(self.fasta_array[0].view("S1").astype("U1")))[:lim] + "\n"
        f = open(fname, "w")
        f.write(header)
        f.write(fasta_text)
        f.close()

    def write_fasta_to_cmap(self, digestion_sequence: str, output_file_name: str, enzyme_name="BSP1Q", channel=1):
        """
        Writes the sequences into a CMAP file with an in silico digestion by the given digestion sequence.
        """
        from OptiScan.utils import CMAP_HEADER

        def gen_line():
            return '%s\t%s\t%s\t%s\t%s\t%s\t1.0\t1\t1\n'

        digested_array = np.sum(self.fasta_digestion_array, axis=0)

        f = open(output_file_name, 'w')
        f.write(CMAP_HEADER % (digestion_sequence, enzyme_name))

        for i in range(len(self.lengths)):
            start = sum(self.lengths[:i])
            end = sum(self.lengths[:i + 1])
            _id = i + 1
            nicking_sites = np.where(digested_array[start:end] > 0.0)[0]
            if nicking_sites.shape[0] == 0:
                continue
            else:
                length = end - start
                for j in range(nicking_sites.shape[0]):
                    line = gen_line() % (_id, length, nicking_sites.shape[0], j + 1, channel, nicking_sites[j])
                    f.write(line)
                line = gen_line()
                line = line % (_id, length, nicking_sites.shape[0], nicking_sites.shape[0] + 1, 0, length)
                f.write(line)
        f.close()


class SvdFromFastaArray:
    def __init__(self, fasta_array, digestion_array,
                 omsim_exec_path, omsim_enzyme_path,
                 omsim_template_path, cov=50):
        self.omsim = OmsimWrapper(omsim_exec_path, omsim_enzyme_path, omsim_template_path, cov=cov)
        digestion_array = np.sum(digestion_array, axis=0)
        self.digestion_indices = np.where(digestion_array > 0)[0]
        self.fasta_array = fasta_array[0]
        self.tracked_changes = list()

    def introduce_deletion(self, start_dist=50000, var=10000):
        locus = np.random.randint(0, self.fasta_array.shape[0]-start_dist-var)
        scale = int(np.random.normal(start_dist, var//2, 1))
        self.fasta_array = np.concatenate((self.fasta_array[:locus], self.fasta_array[locus+scale:]))
        self.tracked_changes.append((locus, scale, "del"))

    def introduce_deletion_exact(self, start=50000, end=100000):
        self.fasta_array = np.concatenate((self.fasta_array[:start], self.fasta_array[end:]))
        self.tracked_changes.append((start, end, "del"))

    def introduce_deletion_label_based(self, number_of_labels=5):
        if number_of_labels == -1:
            _from = np.random.randint(0, self.digestion_indices.shape[0]-1)
            _to = _from + 1
            _from = self.digestion_indices[_from]
            _to = _to + self.digestion_indices[_to]
            _to - np.random.choice(np.arange(_from, _to))
        else:
            number_of_labels += 1
            _from = np.random.randint(0, self.digestion_indices.shape[0] - number_of_labels)
            _to = _from + number_of_labels
            _from = self.digestion_indices[_from]
            _to = _to + self.digestion_indices[_to]
        self.fasta_array = np.concatenate((self.fasta_array[:_from], self.fasta_array[_to:]))
        self.tracked_changes.append((_from, _to, "del"))

    def copy_paste(self, length=10000):
        start = np.random.randint(0, self.fasta_array.shape[0] - length)
        end = start + length
        _to = np.random.randint(0, self.fasta_array.shape[0] - length)
        self.fasta_array = np.concatenate((self.fasta_array[:_to], self.fasta_array[start:end], self.fasta_array[_to:]))
        self.tracked_changes.append((start, end, _to, "dup"))

    def write_ref(self, fname="temp_ref.fasta"):
        header = "> ref\n"
        fasta_text = "".join(list(self.fasta_array.view("S1").astype("U1"))) + "\n"
        f = open(fname, "w")
        f.write(header)
        f.write(fasta_text)
        f.close()
        self.omsim.run_simulation_from_fasta(fname)

    def write(self, fname="temp_svd.fasta"):
        header = "> svd\n"
        fasta_text = "".join(list(self.fasta_array.view("S1").astype("U1"))) + "\n"
        f = open(fname, "w")
        f.write(header)
        f.write(fasta_text)
        f.close()
        self.omsim.run_simulation_from_fasta(fname)


class OmsimWrapper:
    def __init__(self, executable_path, enzymes_path, param_template_path, cov=50):
        self.cov = cov
        self.exec = executable_path
        self.enzymes = enzymes_path
        self.template = param_template_path
        self.params = "params.xml"

    def run_simulation_from_fasta(self, fasta_path):
        template = open(self.template, "r").read()
        print(template)
        template = template.replace("{file1}", fasta_path)
        template = template.replace("{file2}", self.enzymes)
        template = template.replace("{cov}", str(self.cov))
        f = open(self.params, "w")
        f.write(template)
        f.close()
        ck(f"python2 {self.exec} {self.params}", shell=True)
        ck(f"mv yeast_output.label_0.1.bnx {fasta_path}.bnx", shell=True)


if __name__ == "__main__":
    import json
    config_path = "config.json"
    parameters = json.loads(config_path)
    omsim_exec_path = parameters["omsim_exec_path"]
    omsim_param_template_path = parameters["omsim_template_path"]
    omsim_enzyme_path = parameters["omsim_enzyme_path"]

    fasta = Fasta("/Users/akdel/PycharmProjects/OptiDiff/S288C_reference_sequence_R64-2-1_20150113.fsa")
    print(fasta.fasta_array.shape)
    fasta.write_all_chr(fname="temp.fasta", lim=5000000) # concats chromosomes into a single fasta entry
    fasta = Fasta("temp.fasta") # reloads fasta
    print(fasta.fasta_array.shape)
    fasta.write_fasta_to_cmap(digestion_sequence="GCTCTTC", output_file_name="temp_all_chr.cmap",
                              enzyme_name="BSPQ1", channel=1) # writes fasta cmap and digests the sequence
    # for _ in range(50):
    #     svd_fasta = SvdFromFastaArray(fasta.fasta_array, fasta.fasta_digestion_array, omsim_exec_path, omsim_enzyme_path,
    #                                   omsim_param_template_path, cov=75)
    #     svd_fasta.write(fname=f"negatives/{np.random.randint(0, 1000000, 1)}.fasta")
    # exit()

    # svd_fasta = SvdFromFastaArray(fasta.fasta_array , fasta.fasta_digestion_array, omsim_exec_path, omsim_enzyme_path,
    #                               omsim_param_template_path, cov=700) # loads the fasta array and digested sequence
    # svd_fasta.write_ref(fname="all_ref6.fasta")
    #
    # fasta = Fasta("/Users/akdel/PycharmProjects/OptiDiff/OptiDiff/simulate/all_ref6.fasta")
    # print(fasta.fasta_array.shape)
    # fasta.write_all_chr(fname="temp.fasta")  # concats chromosomes into a single fasta entry
    fasta = Fasta("temp.fasta")  # reloads fasta
    print(fasta.fasta_array.shape)
    fasta.write_fasta_to_cmap(digestion_sequence="GCTCTTC", output_file_name="temp_all_chr.cmap",
                              enzyme_name="BSPQ1", channel=1)  # writes fasta cmap and digests the sequence
    print(fasta.fasta_array.shape)
    svd_fasta = SvdFromFastaArray(fasta.fasta_array, fasta.fasta_digestion_array, omsim_exec_path, omsim_enzyme_path,
                                  omsim_param_template_path, cov=1800)
    svd_fasta.write_ref(fname="all_ref_5mb.fasta")
    # exit()
    # print(fasta.fasta_array.shape)
    # svd_fasta = SvdFromFastaArray(fasta.fasta_array, fasta.fasta_digestion_array, omsim_exec_path, omsim_enzyme_path,
    #                               omsim_param_template_path, cov=600)
    # svd_fasta.introduce_deletion_exact(1309859, 1317596)
    # svd_fasta.write_ref(fname="all_ref_fp.fasta")
    # exit()
    # svd_fasta = SvdFromFastaArray(fasta.fasta_array, fasta.fasta_digestion_array, omsim_exec_path, omsim_enzyme_path,
    #                               omsim_param_template_path, cov=600)
    # print(svd_fasta.fasta_array.shape)
    # svd_fasta.introduce_deletion_exact(1309859, 1317596)
    # svd_fasta.write(
    #     fname=f"deletion_svd_low/temp_svd_del{svd_fasta.tracked_changes[0][0]}-{svd_fasta.tracked_changes[0][1]}-retry-del.fasta")
    # exit()
    for _ in range(5):
        svd_fasta = SvdFromFastaArray(fasta.fasta_array, fasta.fasta_digestion_array, omsim_exec_path, omsim_enzyme_path,
                                      omsim_param_template_path, cov=400)
        print(svd_fasta.fasta_array.shape)
        svd_fasta.copy_paste(length=500000)
        # svd_fasta.introduce_deletion_label_based(np.random.choice([-1, 0, 1, 2, 3]))
        # svd_fasta.copy_paste(length=200000)
        print(svd_fasta.fasta_array.shape)
        print(svd_fasta.tracked_changes)
        # svd_fasta.write(fname=f"simulated_svd2/temp_svd_del{svd_fasta.tracked_changes[0][0]}-{svd_fasta.tracked_changes[0][1]}-{svd_fasta.tracked_changes[0][-1]}.fasta")
        svd_fasta.write(fname=f"simulated_svd/temp_svd_dup{svd_fasta.tracked_changes[0][0]}-{svd_fasta.tracked_changes[0][1]}-{svd_fasta.tracked_changes[0][2]}-{svd_fasta.tracked_changes[0][-1]}.fasta")
