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
import tempfile
import csv
import os
from typing import Dict, Optional, Any, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogParser:
    """
    Memory-safe, streaming log parser using the Drain algorithm.
    Supports pre-masking of variables (IPs, Hex, etc.) to improve template quality.
    """

    def __init__(
        self,
        depth: int = 4,
        sim_th: float = 0.4,
        max_children: int = 100,
        masking_patterns: Optional[List[Dict[str, str]]] = None,
        label_extractor = None
    ) -> None:
        """
        Initialize Drain parser.

        Args:
            depth: Drain tree depth (default: 4)
            sim_th: Similarity threshold 0-1 (default: 0.4)
            max_children: Max children per node (default: 100)
            masking_patterns: List of dicts e.g. [{'pattern': r'\d+', 'mask': '<NUM>'}]
            label_extractor: Optional function to extract labels from log lines
        """
        # Configure Drain3
        config = TemplateMinerConfig()
        config.drain_depth = depth
        config.drain_sim_th = sim_th
        config.drain_max_children = max_children
        config.drain_max_clusters = 1000  # Prevent runaway template explosion

        # Using no persistence (can be added)
        self.template_miner: TemplateMiner = TemplateMiner(config=config)

        # Template/event bookkeeping
        self.templates: Dict[int, Dict[str, Any]] = {}        # cluster_id -> {EventId, Template, Count}
        self.template_to_id: Dict[str, str] = {}   # template -> EventId
        self.event_count: int = 0

        # Pre-compile masking regexes for performance
        self.masking_patterns = []
        if masking_patterns:
            for mp in masking_patterns:
                try:
                    # Store as tuple (compiled_regex, mask_string)
                    self.masking_patterns.append((re.compile(mp['pattern']), mp['mask']))
                except re.error as e:
                    logger.error(f"Invalid regex pattern '{mp['pattern']}': {e}")

        self.label_extractor = label_extractor
        logger.info(f"LogParser initialized: depth={depth}, sim_th={sim_th}, masking_rules={len(self.masking_patterns)}")


    def extract_content(self, log_line: str, regex_pattern: Optional[str] = None) -> str:
        """
        Extract log content via regex if provided.
        
        Args:
            log_line: Raw log line
            regex_pattern: Optional regex pattern for extraction
            
        Returns:
            Extracted content string
        """
        if regex_pattern is None:
            return log_line.strip()

        match = re.match(regex_pattern, log_line.strip())
        if match:
            return match.groups()[-1] if match.groups() else log_line.strip()

        return log_line.strip()


    def apply_masking(self, content: str) -> str:
        """
        Apply regex masking rules to content string.
        """
        for regex, mask in self.masking_patterns:
            content = regex.sub(mask, content)
        return content


    def parse_line(self, log_line: str, regex_pattern: Optional[str] = None) -> Dict[str, str]:
        """
        Parse a single line using Drain.
        
        Args:
            log_line: Raw log line
            regex_pattern: Optional regex pattern for extraction
            
        Returns:
            Dictionary with Content, EventId, EventTemplate
        """
        # 1. Extract the raw message (remove timestamps, nodes, etc.)
        raw_content = self.extract_content(log_line, regex_pattern)

        # 2. Apply masking (clean hex, IPs, numbers)
        masked_content = self.apply_masking(raw_content)

        # 3. Pass MASKED content to Drain
        result = self.template_miner.add_log_message(masked_content)
        cluster_id = result["cluster_id"]
        template = result["template_mined"]

        # Assign EventId to new templates
        if cluster_id not in self.templates:
            self.event_count += 1
            event_id = f"E{self.event_count}"

            self.templates[cluster_id] = {
                "EventId": event_id,
                "EventTemplate": template,
                "Count": 0
            }
        else:
            # Update template in case Drain refined it with wildcards
            self.templates[cluster_id]["EventTemplate"] = template

        # Count occurrence
        self.templates[cluster_id]["Count"] += 1
        event_id = self.templates[cluster_id]["EventId"]

        return {
            "Content": raw_content,      # Keep original content for reference
            "MaskedContent": masked_content, # What was actually clustered
            "EventId": event_id,
            "EventTemplate": template
        }


    def parse_file(
        self, 
        log_file: str, 
        regex_pattern: Optional[str] = None, 
        sample_size: Optional[int] = None
    ) -> pd.DataFrame:
        """
        Memory-safe streaming log file parsing.

        Args:
            log_file: Path to log file
            regex_pattern: Optional regex pattern for extraction
            sample_size: If set, only parse first N lines
            
        Returns:
            DataFrame with parsed logs
        """

        logger.info(f"Parsing log file: {log_file}")

        # Create temporary CSV file for streaming output
        tmp_fd, tmp_path = tempfile.mkstemp(suffix=".parsed.csv")
        os.close(tmp_fd)  # We will reopen using csv.writer

        try:
            with open(tmp_path, "w", newline="", encoding="utf-8") as out_csv:
                writer = csv.writer(out_csv)
                # Added Label column
                writer.writerow(["LineId", "EventId", "EventTemplate", "Content", "MaskedContent", "Label", "RawLog"])

                parsed_count = 0

                with open(log_file, "r", encoding="utf-8", errors="ignore") as f:
                    for idx, line in enumerate(tqdm(f, desc="Parsing logs")):
                        if sample_size and parsed_count >= sample_size:
                            break

                        stripped = line.strip()
                        if not stripped:
                            continue

                        try:
                            parsed = self.parse_line(stripped, regex_pattern)
                            parsed_count += 1

                            # Extract label if label_extractor is provided
                            label = 0  # Default to normal
                            if self.label_extractor:
                                label = self.label_extractor(stripped)

                            # Stream output row to disk (no RAM)
                            writer.writerow([
                                idx + 1,
                                parsed["EventId"],
                                parsed["EventTemplate"],
                                parsed["Content"],
                                parsed["MaskedContent"],
                                label,
                                stripped
                            ])

                        except Exception as e:
                            logger.warning(f"Failed to parse line {idx + 1}: {str(e)}")
                            continue

            logger.info(f"Parsed {parsed_count} logs")
            logger.info(f"Discovered {len(self.templates)} unique templates")

            # Load DataFrame only once at the end
            df = pd.read_csv(tmp_path)

            return df
        
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.remove(tmp_path)


    def get_templates(self) -> pd.DataFrame:
        """
        Returns DataFrame of all discovered templates.
        
        Returns:
            DataFrame with EventId, EventTemplate, OccurrenceCount
        """

        templates_list = [
            {
                "EventId": info["EventId"],
                "EventTemplate": info["EventTemplate"],
                "OccurrenceCount": info["Count"]
            }
            for info in self.templates.values()
        ]

        df = pd.DataFrame(templates_list)
        df = df.sort_values("OccurrenceCount", ascending=False)

        return df


    def filter_templates(self, min_occurrence: int = 1, remove_noise_patterns: bool = True) -> pd.DataFrame:
        """
        Filter templates by minimum occurrence count and remove noise patterns.
        Returns DataFrame with filtered templates and mapping for rare events.

        Args:
            min_occurrence: Minimum occurrence count to keep template
            remove_noise_patterns: Remove register dumps, hex-only patterns, etc.

        Returns:
            DataFrame of filtered templates
        """
        df_templates = self.get_templates()

        if min_occurrence <= 1 and not remove_noise_patterns:
            logger.info("No template filtering applied")
            return df_templates

        # Step 1: Filter by occurrence
        if min_occurrence > 1:
            frequent = df_templates[df_templates['OccurrenceCount'] >= min_occurrence]
            rare = df_templates[df_templates['OccurrenceCount'] < min_occurrence]

            logger.info(f"Frequency filtering: min_occurrence={min_occurrence}")
            logger.info(f"  Keeping {len(frequent)} templates ({frequent['OccurrenceCount'].sum():,} logs)")
            logger.info(f"  Removing {len(rare)} rare templates ({rare['OccurrenceCount'].sum():,} logs)")
        else:
            frequent = df_templates

        # Step 2: Remove noise patterns
        if remove_noise_patterns:
            # Patterns that indicate noise (register dumps, pure hex, etc.)
            noise_patterns = [
                r'^r\d+=0x[0-9a-fA-F]+',  # Register dumps: r00=0x...
                r'^fpr\d+=0x[0-9a-fA-F]+', # FP register dumps: fpr10=0x...
                r'^esr=0x[0-9a-fA-F]+',    # ESR register: esr=0x...
                r'^0x[0-9a-fA-F\s]+$',     # Pure hex values
                r'^[\d\s\.]+$',            # Pure numbers
            ]

            # Combine patterns
            combined_pattern = '|'.join(noise_patterns)
            noise_mask = frequent['EventTemplate'].str.contains(combined_pattern, regex=True, na=False)

            noise_templates = frequent[noise_mask]
            clean_templates = frequent[~noise_mask]

            logger.info(f"Noise pattern filtering:")
            logger.info(f"  Removing {len(noise_templates)} noise templates ({noise_templates['OccurrenceCount'].sum():,} logs)")
            logger.info(f"  Keeping {len(clean_templates)} clean templates ({clean_templates['OccurrenceCount'].sum():,} logs)")

            frequent = clean_templates

        logger.info(f"Final coverage: {frequent['OccurrenceCount'].sum() / df_templates['OccurrenceCount'].sum() * 100:.2f}%")

        return frequent


    def save_templates(self, output_file: str, min_occurrence: int = 1) -> pd.DataFrame:
        """
        Save templates to CSV with optional filtering.

        Args:
            output_file: Path to save templates CSV
            min_occurrence: Minimum occurrence count to keep template

        Returns:
            DataFrame of saved templates
        """

        df_templates = self.filter_templates(min_occurrence)
        df_templates.to_csv(output_file, index=False)

        logger.info(f"Saved {len(df_templates)} templates to {output_file}")

        return df_templates