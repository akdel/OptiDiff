import numpy as np
import glob
from dataclasses import dataclass
import itertools
import typing as ty
from enum import Enum


class SVType(Enum):
    Deletion = 0
    Translocation = 1
    Duplication = 2
    InplaceInversion = 3
    InvertedTranslocation = 4
    InvertedDuplication = 5
    InsertionOrDeletion = 6
    Unspecific = 7


@dataclass
class Result:
    sv_type: SVType
    chrid: ty.Union[int, None]
    origin_start: ty.Union[int, None]
    origin_end: ty.Union[int, None]
    target_chrid: ty.Union[int, None]
    target_start: ty.Union[int, None]
    target_end: ty.Union[int, None]

    @classmethod
    def from_result_line(cls, line: str) -> "Result":
        type_string, chr_id, origin_start, origin_end, target_chrid, target_start, target_end = line.split("\t")

        def check_for_none(input: str) -> ty.Union[int, None]:
            if input.strip() == "N/A":
                return None
            else:
                return int(input.strip())

        def type_string_to_sv_type(type_string: str) -> SVType:
            if "Insertion" in type_string:
                return SVType.InsertionOrDeletion
            elif "Deletion" in type_string:
                return SVType.Deletion
            elif "Inversion" in type_string:
                if "Translocation" in type_string:
                    return SVType.InvertedTranslocation
                elif "Duplication" in type_string:
                    return SVType.InvertedDuplication
                else:
                    return SVType.InplaceInversion
            elif "Translocation" in type_string:
                return SVType.Translocation
            elif "Duplication" in type_string:
                return SVType.Duplication
            else:
                return SVType.Unspecific

        sv_type = type_string_to_sv_type(type_string)
        chr_id, origin_start, origin_end, target_chrid, target_start, target_end = [check_for_none(x) for x in
                                                                                    [chr_id, origin_start, origin_end,
                                                                                     target_chrid, target_start,
                                                                                     target_end]]
        return cls(sv_type,
                   chr_id,
                   origin_start,
                   origin_end,
                   target_chrid,
                   target_start,
                   target_end)

    def equal_to(self, other: "Result", intersection_tolerance: int = 10) -> bool:
        def check_intersection(first_line: ty.Tuple[ty.Union[int, None], ty.Union[int, None]],
                               second_line: ty.Tuple[ty.Union[int, None], ty.Union[int, None]],
                               tolerance: int = intersection_tolerance) -> bool:
            p11, p12 = first_line
            p21, p22 = second_line
            if (p11 is None) or (p12 is None) or (p21 is None) or (p22 is None):
                return False
            if (p21 - tolerance) < p11 < (p22 + tolerance):
                return True
            elif (p21 - tolerance) < p12 < (p22 + tolerance):
                return True
            elif (p11 - tolerance) < p21 < (p12 + tolerance):
                return True
            elif (p11 - tolerance) < p22 < (p12 + tolerance):
                return True
            else:
                return False

        if self.sv_type != other.sv_type:
            return False
        elif check_intersection((self.target_start, self.target_end),
                                (other.target_start, other.target_end)):
            return True
        elif check_intersection((self.target_start, self.target_end),
                                (other.origin_start, other.origin_end)):
            return True
        elif check_intersection((self.origin_start, self.origin_end),
                                (other.origin_start, other.origin_end)):
            return True
        else:
            return False


@dataclass
class ResultsForRun:
    results: ty.List[Result]
    filename: str
    coverage: int

    @classmethod
    def from_result_file(cls, filename: str, coverage: int) -> "ResultsForRun":
        results: ty.List[Result] = list()
        for line in iter(open(filename, "r")):
            if not line.startswith("#"):
                results.append(Result.from_result_line(line))
        return cls(results, filename, coverage)

    def compare_to(self, other: "ResultsForRun") -> ty.List[bool]:
        comparisons: ty.List[bool] = [False for _ in range(len(self.results))]
        for i, this_result in enumerate(self.results):
            for other_result in other.results:
                if this_result.equal_to(other_result):
                    comparisons[i] = True
        return comparisons

    def contained_ratio(self, other: "ResultsForRun") -> float:
        compared: ty.List[bool] = self.compare_to(other)
        return len([x for x in compared if x is True]) / len(compared)


Coverage = int

@dataclass
class SVDetectionParameters:
    cmap_reference_file: str
    reference_bnx_file: str
    subsample_range: ty.Tuple[float, float, int] = (0.1, 1., 3)
    reference_subsample_ratio: float = 1.0
    nbits: int = 64
    segment_length: int = 275
    zoom_factor: int = 500
    minimum_molecule_length: int = 150_000
    distance_threshold: float = 1.7
    unspecific_sv_threshold: ty.Union[float, str] = 10.0
    density_filter: int = 40
    translocation_threshold: int = 10


@dataclass
class Performance:
    simulated_filenames: ty.Dict[Coverage, str]
    ground_truth: ty.Dict[Coverage, ty.List[ResultsForRun]]
    detected: ty.Dict[Coverage, ty.List[ResultsForRun]]
    coverages: ty.Set[int]

    @classmethod
    def from_folder(cls, path: str, ground_truth_function: ty.Callable,
                    tsv_parser_function: ty.Callable,
                    with_run: ty.Union[None, SVDetectionParameters] = None) -> "Performance":
        bnx_files = glob.glob(path + "*.bnx")
        ground_truth = [ground_truth_function(x) for x in bnx_files]
        ground_truth = {result.filename: result for result in ground_truth}

        if with_run is not None:
            start, end, step = with_run.subsample_range
            from OptiDiff.sv_detect import detect_structural_variation_for_multiple_datasets
            for i in np.linspace(start, end, step):
                _ = detect_structural_variation_for_multiple_datasets(cmap_reference_file=with_run.cmap_reference_file,
                                                                      reference_bnx_file=with_run.reference_bnx_file,
                                                                      sv_candidate_bnx_files=bnx_files,
                                                                      sv_subsample_ratio=i,
                                                                      reference_subsample_ratio=with_run.reference_subsample_ratio,
                                                                      nbits=with_run.nbits,
                                                                      segment_length=with_run.segment_length,
                                                                      zoom_factor=with_run.zoom_factor,
                                                                      minimum_molecule_length=with_run.minimum_molecule_length,
                                                                      distance_threshold=with_run.distance_threshold,
                                                                      unspecific_sv_threshold=with_run.unspecific_sv_threshold,
                                                                      density_filter=with_run.density_filter,
                                                                      translocation_threshold=with_run.translocation_threshold)
        detected = [tsv_parser_function(x) for x in glob.glob(path + "*.tsv")]
        detected = {results.filename: results for results in detected}
        coverages = {results.coverage for results in detected.values()}
        return cls(bnx_files, ground_truth, detected, coverages)

    def correct_detections(self):
        pass

    def correct_classification(self):
        pass

    def incorrect_classification(self):
        # False positive categories
        pass


def tsv_filename_to_result(filename: str) -> ResultsForRun:

    # tsv filename -> 5949562-104552-del_120x.fasta.bnx_0.1_SV_results.tsv
    path = "/".join(filename.split("/")[:-1])
    tsv = filename.split("/")[-1]
    left = filename.split(".")[0]
    coverage: int = int(float(filename.split("_")[4]) * 120)
    bnx_name = f"{path}/{left}.fasta.bnx"
    results: ResultsForRun = ResultsForRun.from_result_file(filename, coverage)
    results.filename = bnx_name
    return results


def bnx_deletion_filename_to_result(filename: str) -> ResultsForRun:
    start, end, _ = filename.split("/")[-1].split("-")
    sv_type: SVType = SVType.Deletion
    chr_id: int = 1
    origin_start: float = int(start)/1000
    origin_end: float = int(end)/1000
    target_chrid = None
    target_start = None
    target_end = None
    results: ty.List[Result] = [Result(sv_type,
                                      chr_id,
                                      int(origin_start), int(origin_end),
                                      target_chrid,
                                      target_start, target_end)]
    coverage: int = 120
    return ResultsForRun(results, filename, coverage)

