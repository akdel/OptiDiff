xml = """<?xml version="1.0" encoding="UTF-8"?>
<!--Root containing all simulations to run-->
<simulation>
        <!--input defining a single simulation-->
        <input>
                <!--REQUIRED PARAMETERS-->
                <!--Name of the data set-->
                <name>yeast</name>
                <!--Locations of all input files (1 or more)-->
                <files>
                        <!--Location of the data set, absolute, or relative to the current directory-->
                        <file>%s</file>
                </files>
                <!--Add this tag ONLY if the sequences in the data set are circular, leave it out if they are not-->
                <!--The enzymes and their labels to use-->
                <enzymes>
                        <!--Location of the enzymes file-->
                        <file>/Users/akdel/PycharmProjects/omsim/test/ecoli/enzymes.xml</file>
                        <!--Specification of one enzyme and its label-->
                        <enzyme>
                                <!--id of the enzyme-->
                                <id>BspQI</id>
                                <!--label used for this enzyme, different labels will result in different output files-->
                                <label>label_0</label>
                        </enzyme>
                        <!--Multiple enzymes can be specified, possibly with different labels-->

                </enzymes>

                <!--OPTIONAL PARAMETERS-->
                <!--Prefix of the output, relative to the current directory-->
                <prefix>yeast_output</prefix>
                <!--Minimal desired coverage-->
                <coverage>1</coverage>
                <!--Number of chips that will be used, this takes precedence over coverage-->
                <chips>1</chips>
                <!--Number of scans taken per chip-->
                <scans_per_chip>1</scans_per_chip>
                <!--Scan size in Mbp-->
                <scan_size>100</scan_size>
                <!--Average molecule length-->
                <avg_mol_len>200000</avg_mol_len>
                <!--Standard deviation of the molecule length-->
                <sd_mol_len>150000</sd_mol_len>
                <!--Minimal molecule length-->
                <min_mol_len>20000</min_mol_len>
                <!--Maximal molecule length-->
                <max_mol_len>2500000</max_mol_len>
                <!--Average stretch factor-->
                <stretch_factor>0.85</stretch_factor>
                <!--Standard deviation of stretch factor per chip-->
                <stretch_chip_sd>0.01</stretch_chip_sd>
                <!--Standard deviation of stretch factor per scan-->
                <stretch_scan_sd>0.001</stretch_scan_sd>
                <!--Minimal number of labels required per read-->
                <min_nicks>1</min_nicks>
                <!--Additional normal noise-->
                <nick_sd>50</nick_sd>
                <!--Distance at which fragile sites on one strand have 50% chance to break-->
                <fragile_same>50</fragile_same>
                <!--Distance at which fragile sites on opposite strands have 50% chance to break-->
                <fragile_opposite>150</fragile_opposite>
                <!--If the distance is further away than treshold from the 50% point, then break chance is 100% (left) or 0% (right)-->
                <fragile_treshold>25</fragile_treshold>
                <!--Factor determining the steepness of the fragile location cutoff when the distance is less than fragile_treshold away from the 50% point-->
                <fragile_factor>3</fragile_factor>
                <!--Distance between labels where there is 50% chance of collapsing them unto one label-->
                <label_mu>1500</label_mu>
                <!--Similar to fragile_treshold, but now for label collapsing-->
                <label_treshold>500</label_treshold>
                <!--Similar to fragile_factor, but now for label collapsing-->
                <label_factor>100</label_factor>
                <!--Percent of the reads that will be extend as a chimera-->
                <chimera_rate>0.01</chimera_rate>
                <!--Mean of the normal distribution determining the distance added between two parts of a chimeric read-->
                <chimera_mu>1500</chimera_mu>
                <!--SD of the normal distribution determining the distance added between two parts of a chimeric read-->
                <chimera_sigma>500</chimera_sigma>
                <!--The seed used for all pseudo random number generators-->
                <seed>0</seed>
        </input>

        <!--More simulations can be added here-->

      </simulation>
"""