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
    def from_bng_line(cls, line):
        # SmapEntryID', 'QryContigID', 'RefcontigID1', 'RefcontigID2', 'QryStartPos', 'QryEndPos', 'RefStartPos', 'RefEndPos', 'Confidence', 'Type', 'XmapID1', 'XmapID2', 'LinkID', 'QryStartIdx', 'QryEndIdx', 'RefStartIdx', 'RefEndIdx', 'Zygosity', 'Genotype', 'GenotypeGroup', 'RawConfidence', 'RawConfidenceLeft', 'RawConfidenceRight', 'RawConfidenceCenter'
        _, _, chr_id, target_chrid, _, _, origin_start, origin_end, _, type_string, _, _, _, _, _, _, _, _, _, _, _, _, _, _ = line.split("\t")
        type_string: str = type_string.lower()

        def type_string_to_sv_type(type_string: str) -> SVType:
            # TODO: This should be adapted into bng result types after the results are out.
            if "insertion" in type_string:
                return SVType.InsertionOrDeletion
            elif "deletion" in type_string:
                return SVType.Deletion
            elif "inversion" in type_string:
                if "translocation" in type_string:
                    return SVType.InvertedTranslocation
                elif "duplication" in type_string:
                    return SVType.InvertedDuplication
                else:
                    return SVType.InplaceInversion
            elif "translocation" in type_string:
                return SVType.Translocation
            elif "duplication" in type_string:
                return SVType.Duplication
            else:
                return SVType.Unspecific

        sv_type = type_string_to_sv_type(type_string)
        return cls(sv_type,
                   int(chr_id),
                   int(float(origin_start)/1000),
                   int(float(origin_end)/1000),
                   int(target_chrid),
                   None, None)

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

    def check_intersection(self,
                           first_line: ty.Tuple[ty.Union[int, None], ty.Union[int, None]],
                           second_line: ty.Tuple[ty.Union[int, None], ty.Union[int, None]],
                           tolerance: int = 10) -> bool:
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

    def equal_to(self, other: "Result", intersection_tolerance: int = 10) -> bool:
        if self.sv_type != other.sv_type:
            return False
        else:
            return self.overlaps_with(other,
                                      intersection_tolerance=intersection_tolerance)

    def overlaps_with(self, other: "Result", intersection_tolerance: int = 10) -> bool:
        if self.check_intersection((self.target_start, self.target_end),
                                   (other.target_start, other.target_end),
                                   tolerance=intersection_tolerance):
            return True
        elif self.check_intersection((self.target_start, self.target_end),
                                     (other.origin_start, other.origin_end),
                                     tolerance=intersection_tolerance):
            return True
        elif self.check_intersection((self.origin_start, self.origin_end),
                                     (other.origin_start, other.origin_end),
                                     tolerance=intersection_tolerance):
            return True
        else:
            return False


@dataclass
class ResultsForRun:
    results: ty.List[Result]
    filename: str
    coverage: int

    class ResultFileType(Enum):
        OptiTools = 1
        BNG = 2

    @classmethod
    def from_result_file(cls, filename: str, coverage: int,
                         file_type: "ResultsForRun.ResultFileType" = ResultFileType.OptiTools) -> "ResultsForRun":
        results: ty.List[Result] = list()
        for line in iter(open(filename, "r")):
            if not line.startswith("#"):
                if file_type == ResultsForRun.ResultFileType.OptiTools:
                    results.append(Result.from_result_line(line))
                else:
                    results.append(Result.from_bng_line(line))
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
    ground_truth: ty.Dict[str, ResultsForRun]
    detected: ty.Dict[Coverage, ty.Dict[str, ResultsForRun]]
    coverages: ty.Set[int]

    @classmethod
    def from_folder(cls, path: str, ground_truth_function: ty.Callable,
                    tsv_parser_function: ty.Callable,
                    with_run: ty.Union[None, SVDetectionParameters] = None,
                    result_type: ResultsForRun.ResultFileType = ResultsForRun.ResultFileType.OptiTools) -> "Performance":
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
        detected: ty.Dict[Coverage, ty.Dict[str, ResultsForRun]] = dict()
        used_coverages: ty.Set[int] = set()
        if result_type == ResultsForRun.ResultFileType.OptiTools:
            files: ty.List[str] = glob.glob(path + "*.tsv")
        else:
            files: ty.List[str] = glob.glob(path + "*.smap")
        for x in files:
            current_detected = tsv_parser_function(x)
            if current_detected.coverage not in detected:
                detected[current_detected.coverage] = {}
            detected[current_detected.coverage][current_detected.filename] = current_detected
            used_coverages.add(current_detected.coverage)
        return cls(bnx_files, ground_truth, detected, used_coverages)

    def correct_detections(self, coverage: int):
        tp_detections: ty.Dict[str, ty.Tuple[ty.List[bool], "Performance.SVTypeCounts"]] = dict()
        for bnx_file, results in self.detected[coverage].items():

            results: ResultsForRun
            bnx_file: str
            tp_detections[bnx_file] = ([], self.SVTypeCounts())
            truth: ResultsForRun = self.ground_truth[bnx_file]
            for result in results.results:
                if truth.results[0].overlaps_with(result):
                    tp_detections[bnx_file][0].append(True)
                else:
                    tp_detections[bnx_file][0].append(False)
                    tp_detections[bnx_file][1].add_sv(result.sv_type)
        return tp_detections

    def correct_classifications(self, coverage: int):
        tp_classifications: ty.Dict[str, ty.Tuple[ty.List[bool], "Performance.SVTypeCounts"]] = dict()
        for bnx_file, results in self.detected[coverage].items():
            results: ResultsForRun
            bnx_file: str
            tp_classifications[bnx_file] = ([], self.SVTypeCounts())
            truth: ResultsForRun = self.ground_truth[bnx_file]
            for result in results.results:
                if truth.results[0].equal_to(result):
                    tp_classifications[bnx_file][0].append(True)
                else:
                    tp_classifications[bnx_file][0].append(False)
                    tp_classifications[bnx_file][1].add_sv(result.sv_type)
        return tp_classifications

    @dataclass
    class SVTypeCounts:
        deletion: int = 0
        insertion_or_small_deletion: int = 0
        translocation: int = 0
        duplication: int = 0
        inplace_inversion: int = 0
        inverted_translocation: int = 0
        inverted_duplication: int = 0

        def add_sv(self, sv_type: SVType) -> None:
            if sv_type == SVType.Deletion:
                self.deletion += 1
            elif sv_type == SVType.Translocation:
                self.translocation += 1
            elif sv_type == SVType.Duplication:
                self.duplication += 1
            elif sv_type == SVType.InsertionOrDeletion:
                self.insertion_or_small_deletion += 1
            elif sv_type == SVType.InplaceInversion:
                self.inplace_inversion += 1
            elif sv_type == SVType.InvertedDuplication:
                self.inverted_duplication += 1
            elif sv_type == SVType.InvertedTranslocation:
                self.inverted_translocation += 1

        def __add__(self, other: "Performance.SVTypeCounts") -> "Performance.SVTypeCounts":
            return Performance.SVTypeCounts(
                deletion=self.deletion + other.deletion,
                translocation=self.translocation + other.translocation,
                duplication=self.duplication + other.duplication,
                insertion_or_small_deletion=self.insertion_or_small_deletion + other.insertion_or_small_deletion,
                inplace_inversion=self.inplace_inversion + other.inplace_inversion,
                inverted_translocation=self.inverted_translocation + other.inverted_translocation,
                inverted_duplication=self.inverted_duplication + other.inverted_duplication,
            )

        def __sub__(self, other: "Performance.SVTypeCounts") -> "Performance.SVTypeCounts":
            return Performance.SVTypeCounts(
                deletion=self.deletion - other.deletion,
                translocation=self.translocation - other.translocation,
                duplication=self.duplication - other.duplication,
                insertion_or_small_deletion=self.insertion_or_small_deletion - other.insertion_or_small_deletion,
                inplace_inversion=self.inplace_inversion - other.inplace_inversion,
                inverted_translocation=self.inverted_translocation - other.inverted_translocation,
                inverted_duplication=self.inverted_duplication - other.inverted_duplication,
            )

    @dataclass
    class Evaluation:
        total_real_sites: int = 0
        true_classification_count: int = 0
        true_detection_count: int = 0
        false_detection_count: int = 0
        false_classification_details: ty.Union["Performance.SVTypeCounts", None] = None
        false_detection_details: ty.Union["Performance.SVTypeCounts", None] = None

    def evaluate(self, coverage: int):
        evaluation_results: "Performance.Evaluation" = self.Evaluation()
        detections = self.correct_detections(coverage)
        classifications = self.correct_classifications(coverage)
        evaluation_results.total_real_sites = len(self.detected[coverage])
        evaluation_results.true_detection_count = len([True for x in detections.values() if True in x[0]])
        evaluation_results.true_classification_count = len([True for x in classifications.values() if True in x[0]])
        evaluation_results.false_classification_details = self.SVTypeCounts()
        evaluation_results.false_detection_details = self.SVTypeCounts()
        for bnx_name in detections:
            evaluation_results.false_detection_details += detections[bnx_name][1]
            evaluation_results.false_classification_details += classifications[bnx_name][1]
        evaluation_results.false_classification_details -= evaluation_results.false_detection_details
        for values in detections.values():
            for value in values[0]:
                if not value:
                    evaluation_results.false_detection_count += 1
        return evaluation_results


def tsv_filename_to_result(filename: str) -> ResultsForRun:
    # tsv filename -> 5949562-104552-del_120x.fasta.bnx_0.1_SV_results.tsv
    assert filename.endswith("SV_results.tsv")
    left = filename.split(".")[0]
    coverage: int = int(float(filename.split("_")[4]) * 120)
    bnx_name = f"{left}.fasta.bnx"
    results: ResultsForRun = ResultsForRun.from_result_file(filename, coverage)
    results.filename = bnx_name
    return results


def smap_filename_to_result(filename: str) -> ResultsForRun:
    # tsv filename -> 5949562-104552-del_120x.fasta.bnx.0.1.smap
    assert filename.endswith(".smap")
    left = filename.split(".")[0]
    coverage: int = int(float(".".join(filename.split(".")[3:5])) * 120)
    bnx_name = f"{left}.fasta.bnx"
    results: ResultsForRun = ResultsForRun.from_result_file(filename, coverage,
                                                            file_type=ResultsForRun.ResultFileType.BNG)
    results.filename = bnx_name
    return results


def bnx_deletion_filename_to_result(filename: str) -> ResultsForRun:
    # bnx filename -> 5949562-104552-del_120x.fasta.bnx
    assert filename.endswith(".fasta.bnx")
    start, end, _ = filename.split("/")[-1].split("-")
    sv_type: SVType = SVType.Deletion
    chr_id: int = 1
    origin_start: float = int(start) / 1000
    origin_end: float = origin_start + (int(end) / 1000)
    results: ty.List[Result] = [Result(sv_type,
                                       chr_id,
                                       int(origin_start), int(origin_end),
                                       None, None, None)]
    coverage: int = 120
    return ResultsForRun(results, filename, coverage)
