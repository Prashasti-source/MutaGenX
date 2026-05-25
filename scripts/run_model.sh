#!/bin/bash
python3 /home/csb/webserver/applications/Evoscope/executables/platform_final_1.py wildtype_1.fasta mutated_1.fasta
zip -rq results.zip ./ -x "*.json"
