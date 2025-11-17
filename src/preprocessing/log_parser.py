"""
Log Parser Module

Parses raw log files into structured templates using the Drain algorithm.
Supports configurable regex patterns for different log formats.
"""

import re
import pandas as pd
from drain3 import TemplateMiner
from drain3.template_miner_config import TemplateMinerConfig
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple


class LogParser:
    """
    Parse raw logs into structured templates using Drain algorithm.

    The Drain algorithm groups similar log messages by creating a parse tree
    based on log message structure, extracting templates with wildcards for
    variable parts.

    Example:
        >>> parser = LogParser(depth=4, sim_th=0.4)
        >>> df = parser.parse_log_file('logs.txt', regex_patterns={...})
        >>> templates = parser.get_templates()
    """

    def __init__(
        self,
        depth: int = 4,
        sim_th: float = 0.4,
        max_children: int = 100,
        max_clusters: Optional[int] = None
    ):
        """
        Initialize Log Parser with Drain configuration.

        Args:
            depth: Depth of the parse tree (default: 4)
                Higher depth = more specific grouping
            sim_th: Similarity threshold (0-1, default: 0.4)
                Higher threshold = stricter matching
            max_children: Max children per node (default: 100)
            max_clusters: Max number of clusters/templates (default: None = unlimited)
        """
        self.depth = depth
        self.sim_th = sim_th
        self.max_children = max_children

        # Configure Drain
        config = TemplateMinerConfig()
        config.load({
            'drain_depth': depth,
            'drain_sim_th': sim_th,
            'drain_max_children': max_children,
            'drain_max_clusters': max_clusters
        })

        self.template_miner = TemplateMiner(config=config)
        self.template_map = {}  # cluster_id -> template
        self.cluster_id_map = {}  # cluster_id -> EventId (E1, E2, etc.)
        self.event_counter = 1  # For generating E1, E2, E3, ...

    def extract_fields(
        self,
        log_line: str,
        regex_patterns: Dict[str, str]
    ) -> Dict[str, str]:
        """
        Extract structured fields from log line using regex patterns.

        Args:
            log_line: Raw log line string
            regex_patterns: Dict of field_name -> regex_pattern

        Returns:
            Dict of extracted fields and remaining content

        Example:
            >>> patterns = {'date': r'\\d{6}', 'time': r'\\d{6}'}
            >>> fields = parser.extract_fields(log_line, patterns)
            >>> # {'date': '081109', 'time': '203518', 'Content': '...'}
        """
        fields = {}
        remaining = log_line.strip()

        # Try to extract each pattern in order
        for field_name, pattern in regex_patterns.items():
            match = re.search(pattern, remaining)
            if match:
                fields[field_name] = match.group(0)
                # Remove matched part from remaining string
                remaining = remaining[:match.start()] + remaining[match.end():]
                remaining = remaining.strip()

        # Whatever is left is the content to parse
        fields['Content'] = remaining

        return fields

    def parse_single_log(
        self,
        log_line: str,
        regex_patterns: Optional[Dict[str, str]] = None
    ) -> Dict[str, str]:
        """
        Parse a single log line.

        Args:
            log_line: Raw log line
            regex_patterns: Optional patterns to extract structured fields

        Returns:
            Dict with EventId, EventTemplate, and extracted fields
        """
        result = {}

        # Extract structured fields if patterns provided
        if regex_patterns:
            fields = self.extract_fields(log_line, regex_patterns)
            result.update(fields)
            content = fields['Content']
        else:
            content = log_line.strip()
            result['Content'] = content

        # Parse content with Drain
        drain_result = self.template_miner.add_log_message(content)
        cluster_id = drain_result['cluster_id']
        template = drain_result['template_mined']

        # Map cluster_id to EventId (E1, E2, etc.)
        if cluster_id not in self.cluster_id_map:
            event_id = f'E{self.event_counter}'
            self.cluster_id_map[cluster_id] = event_id
            self.template_map[event_id] = template
            self.event_counter += 1
        else:
            event_id = self.cluster_id_map[cluster_id]

        result['EventId'] = event_id
        result['EventTemplate'] = template

        return result

    def parse_log_file(
        self,
        log_file: str,
        regex_patterns: Optional[Dict[str, str]] = None,
        content_column: str = 'Content',
        max_lines: Optional[int] = None,
        show_progress: bool = True
    ) -> pd.DataFrame:
        """
        Parse an entire log file.

        Args:
            log_file: Path to log file
            regex_patterns: Optional regex patterns for field extraction
            content_column: Name of content column (default: 'Content')
            max_lines: Optional limit on number of lines to process
            show_progress: Show progress bar (default: True)

        Returns:
            DataFrame with parsed logs

        Example:
            >>> patterns = {
            ...     'date': r'\\d{6}',
            ...     'time': r'\\d{6}',
            ...     'pid': r'\\d+',
            ...     'level': r'INFO|WARN|ERROR'
            ... }
            >>> df = parser.parse_log_file('hdfs.log', patterns)
        """
        parsed_logs = []

        # Count lines for progress bar
        if show_progress:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                total_lines = sum(1 for _ in f)
            if max_lines:
                total_lines = min(total_lines, max_lines)

        # Parse logs
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            iterator = enumerate(f, 1)
            if show_progress:
                iterator = tqdm(iterator, total=total_lines, desc="Parsing logs")

            for line_num, line in iterator:
                if max_lines and line_num > max_lines:
                    break

                if not line.strip():
                    continue  # Skip empty lines

                try:
                    result = self.parse_single_log(line, regex_patterns)
                    result['LineId'] = line_num
                    parsed_logs.append(result)
                except Exception as e:
                    # Log parsing error but continue
                    if show_progress:
                        tqdm.write(f"Warning: Error parsing line {line_num}: {str(e)}")
                    continue

        df = pd.DataFrame(parsed_logs)

        # Reorder columns to put EventId and EventTemplate first
        cols = ['LineId', 'EventId', 'EventTemplate']
        other_cols = [c for c in df.columns if c not in cols]
        df = df[cols + other_cols]

        return df

    def get_templates(self) -> pd.DataFrame:
        """
        Get all discovered templates.

        Returns:
            DataFrame with EventId and EventTemplate columns
        """
        templates = [
            {'EventId': event_id, 'EventTemplate': template}
            for event_id, template in sorted(
                self.template_map.items(),
                key=lambda x: int(x[0][1:])  # Sort by numeric part of E1, E2, etc.
            )
        ]
        return pd.DataFrame(templates)

    def save_templates(self, output_file: str):
        """
        Save templates to CSV file.

        Args:
            output_file: Path to output CSV file
        """
        templates_df = self.get_templates()
        templates_df.to_csv(output_file, index=False)
        print(f"✓ Saved {len(templates_df)} templates to {output_file}")

    def get_statistics(self) -> Dict:
        """
        Get parsing statistics.

        Returns:
            Dict with statistics about parsed logs and templates
        """
        return {
            'total_templates': len(self.template_map),
            'total_clusters': len(self.cluster_id_map),
            'depth': self.depth,
            'sim_threshold': self.sim_th,
            'max_children': self.max_children
        }


# Example usage and testing
if __name__ == "__main__":
    import os

    # Test on HDFS logs (small sample)
    print("=" * 80)
    print("LOG PARSER TEST - HDFS Dataset")
    print("=" * 80)

    hdfs_log_file = "data/hdfs/HDFS.log"

    if os.path.exists(hdfs_log_file):
        # HDFS log format: <Date> <Time> <Pid> <Level> <Component>: <Content>
        hdfs_patterns = {
            'Date': r'\d{6}',
            'Time': r'\d{6}',
            'Pid': r'\d+',
            'Level': r'INFO|WARN|ERROR|FATAL',
            'Component': r'[\w\$\.]+'
        }

        # Parse first 1000 lines
        parser = LogParser(depth=4, sim_th=0.4)

        print(f"\nParsing first 1000 lines from {hdfs_log_file}...")
        df = parser.parse_log_file(
            hdfs_log_file,
            regex_patterns=hdfs_patterns,
            max_lines=1000,
            show_progress=True
        )

        print(f"\n✓ Parsed {len(df)} log lines")
        print(f"✓ Discovered {len(parser.get_templates())} unique templates")

        # Show sample
        print("\nSample parsed logs:")
        print(df[['LineId', 'EventId', 'Level', 'Component']].head(10))

        # Show templates
        print("\nDiscovered templates:")
        templates = parser.get_templates()
        print(templates.to_string(index=False))

        # Show statistics
        print("\nParsing statistics:")
        stats = parser.get_statistics()
        for key, value in stats.items():
            print(f"  {key}: {value}")

        # Save templates
        os.makedirs('test_output', exist_ok=True)
        parser.save_templates('test_output/test_templates.csv')

    else:
        print(f"HDFS log file not found at {hdfs_log_file}")
        print("Please ensure HDFS data is downloaded first.")
