#!/bin/bash
cd /home/andrea/arxiv-scanner
source venv/bin/activate
/home/andrea/arxiv-scanner/venv/bin/python main.py >> /home/andrea/arxiv-scanner/cron_log.txt 2>&1
