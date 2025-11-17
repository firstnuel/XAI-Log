""" 
    Log Parser Module 
    Parse BGL (and other datasets) into consistent format compatible with HDFS.
    Outputs: Parsed logs with EventIds and templates.
"""

import re
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogParser:
    """
    Lightweight log parser using Drain algorithm.

    Purpose:
        - Extract templates from raw logs
        - Assign EventIds to log entries
        - Prepare logs for sequence building
    """

    def __init__(self, depth=4, sim_th=0.4, max_children=100):
        """
        Initialize Drain parser.

        Args:
            depth: Drain tree depth (default: 4)
            sim_th: Similarity threshold 0-1 (default: 0.4 = 40%)
            max_children: Max children per node (default: 100)
        """
        # Configure Drain3
        config = TemplateMinerConfig()
        config.drain_depth = depth
        config.drain_sim_th = sim_th
        config.drain_max_children = max_children
        config.drain_max_clusters = 1000  # Reasonable upper limit

        self.template_miner = TemplateMiner(config=config)
        self.templates = {}  # cluster_id -> template mapping
        self.template_to_id = {}  # template -> EventId mapping
        self.event_count = 0

        logger.info(f"LogParser initialized: depth={depth}, sim_th={sim_th}")

    def extract_content(self, log_line, regex_pattern=None):
        """
        Extract log content using regex pattern.

        Args:
            log_line: Raw log line
            regex_pattern: Regex to extract content (if None, use whole line)

        Returns:
            Extracted content string
        """
        if regex_pattern is None:
            return log_line.strip()

        match = re.match(regex_pattern, log_line)
        if match:
            # Assume last group is content
            return match.groups()[-1] if match.groups() else log_line.strip()

        # Fallback: return whole line
        return log_line.strip()

    def parse_line(self, log_line, regex_pattern=None):
        """
        Parse a single log line.

        Args:
            log_line: Raw log line
            regex_pattern: Optional regex for content extraction

        Returns:
            dict with EventId and EventTemplate
        """
        # Extract content
        content = self.extract_content(log_line, regex_pattern)

        # Run Drain
        result = self.template_miner.add_log_message(content)
        cluster_id = result['cluster_id']
        template = result['template_mined']

        # Map cluster_id to EventId
        if cluster_id not in self.templates:
            self.event_count += 1
            event_id = f"E{self.event_count}"
            self.templates[cluster_id] = {
                'EventId': event_id,
                'EventTemplate': template,
                'Count': 0
            }
            self.template_to_id[template] = event_id

        # Increment count
        self.templates[cluster_id]['Count'] += 1
        event_id = self.templates[cluster_id]['EventId']

        return {
            'Content': content,
            'EventId': event_id,
            'EventTemplate': template
        }

    def parse_file(self, log_file, regex_pattern=None, sample_size=None):
        """
        Parse entire log file.

        Args:
            log_file: Path to log file
            regex_pattern: Regex for content extraction
            sample_size: If set, only parse first N lines (for testing)

        Returns:
            DataFrame with parsed logs
        """
        logger.info(f"Parsing log file: {log_file}")

        parsed_data = []

        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()

            if sample_size:
                lines = lines[:sample_size]
                logger.info(f"Sampling first {sample_size} lines")

            for idx, line in enumerate(tqdm(lines, desc="Parsing logs")):
                if not line.strip():
                    continue

                try:
                    parsed = self.parse_line(line, regex_pattern)
                    parsed['LineId'] = idx + 1
                    parsed['RawLog'] = line.strip()
                    parsed_data.append(parsed)
                except Exception as e:
                    logger.warning(f"Failed to parse line {idx + 1}: {str(e)}")
                    continue

        df = pd.DataFrame(parsed_data)

        logger.info(f"Parsed {len(df)} logs")
        logger.info(f"Discovered {len(self.templates)} unique templates")

        return df


    def get_templates(self):
        """
        Get all discovered templates.

        Returns:
            DataFrame with EventId, EventTemplate, Count
        """
        templates_list = []
        for cluster_id, info in self.templates.items():
            templates_list.append({
                'EventId': info['EventId'],
                'EventTemplate': info['EventTemplate'],
                'OccurrenceCount': info['Count']
            })

        df = pd.DataFrame(templates_list)
        df = df.sort_values('OccurrenceCount', ascending=False)

        return df

    def save_templates(self, output_file):
        """
        Save templates to CSV.

        Args:
            output_file: Path to save templates CSV
        """
        df_templates = self.get_templates()
        df_templates.to_csv(output_file, index=False)
        logger.info(f"Saved {len(df_templates)} templates to {output_file}")

        return df_templates

