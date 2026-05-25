"""
Databricks environment configuration module.
Manages switching between dev and prod Databricks environments via environment variables.
"""

import os
from typing import Dict


class DatabricksEnvironmentConfig:
    """Configuration class for managing Databricks environments."""

    def __init__(self):
        """Initialize the configuration with environment variables."""
        self.environment = os.getenv("DATABRICKS_ENVIRONMENT", "dev").lower()
        self._validate_environment()

    def _validate_environment(self):
        """Validate that the environment is either 'dev' or 'prod'."""
        if self.environment not in ["dev", "prod"]:
            print(
                f"Invalid DATABRICKS_ENVIRONMENT '{self.environment}'. "
                "Defaulting to 'dev'. Valid values are: 'dev', 'prod'"
            )
            self.environment = "dev"

    @property
    def schema_name(self) -> str:
        """Get the schema name for the current environment."""
        schema_mapping = {
            "dev": "neurology_analytics.pioneer_curated_dev",
            "prod": "neurology_analytics.pioneer_curated",
        }
        return schema_mapping[self.environment]

    @property
    def raw_schema_name(self) -> str:
        """Get the raw schema name for the current environment."""
        if self.environment == "dev":
            return "neurology_analytics.pioneer_raw_dev"
        return "neurology_analytics.pioneer_raw"

    @property
    def seizure_events_table(self) -> str:
        """Get the seizure events table name for the current environment."""
        return f"{self.schema_name}.events_seizure"

    @property
    def seizure_annotations_table(self) -> str:
        """Get the annotations table name for the current environment."""
        return f"{self.schema_name}.seizure_annotations"

    @property
    def spike_events_table(self) -> str:
        """Get the spike rate events table name for the current environment."""
        return f"{self.schema_name}.events_spike_rate"

    @property
    def spike_annotations_table(self) -> str:
        """Get the spike annotations table name for the current environment."""
        return f"{self.schema_name}.spike_annotations"

    @property
    def sparcnet_annotations_table(self) -> str:
        """Get the sparcnet annotations table name for the current environment."""
        return f"{self.schema_name}.sparcnet_annotations"

    @property
    def eeg_info_table(self) -> str:
        """Get the eeg_info table name."""
        return f"{self.raw_schema_name}.eeg_info"

    @property
    def sleep_reports_table(self) -> str:
        """Get the sleep reports table name for the current environment."""
        return f"{self.schema_name}.events_sleep_report"

    @property
    def patient_info_table(self) -> str:
        """Get the patient_info table name for the current environment."""
        return f"{self.raw_schema_name}.patient_info"

    @property
    def volume_path(self) -> str:
        """Get the volume path for the current environment."""
        volume_mapping = {
            "dev": "Volumes/neurology_analytics/pioneer_curated_dev/figures",
            "prod": "Volumes/neurology_analytics/pioneer_curated/figures",
        }
        return volume_mapping[self.environment]

    def get_category_path(self, category_name: str) -> str:
        """
        Get the correct category path for the current environment.

        Args:
            category_name: The category name (e.g., 'sleep_polar', 'sleep_staging', 'seizure_plots', 'figures')

        Returns:
            The correct path for the category in the current environment.
            Note: Both dev and prod now use the same directory structure:
            - sleep_polar -> sleep/polar
            - sleep_staging -> sleep/staging
            - seizure_plots -> seizure/heatmap
        """
        # Both dev and prod now use the same directory structure
        category_mapping = {
            "sleep_polar": "sleep/polar",
            "sleep_staging": "sleep/staging",
            "seizure_plots": "seizure/heatmap",
            "figures": "",
        }
        return category_mapping.get(category_name, category_name)

    def get_connection_params(self) -> Dict[str, any]:
        """Get connection parameters for Databricks SQL Connector."""
        tls_no_verify = os.getenv("DATABRICKS_TLS_NO_VERIFY", "false").lower() == "true"
        return {
            "server_hostname": os.getenv("DATABRICKS_SERVER_HOSTNAME"),
            "http_path": os.getenv("DATABRICKS_HTTP_PATH"),
            "access_token": os.getenv("DATABRICKS_ACCESS_TOKEN"),
            "_tls_no_verify": tls_no_verify,
        }

    def get_table_names(self) -> Dict[str, str]:
        """Get all table names for the current environment."""
        return {
            "seizure_events": self.seizure_events_table,
            "seizure_annotations": self.seizure_annotations_table,
            "spike_events": self.spike_events_table,
            "spike_annotations": self.spike_annotations_table,
            "sleep_reports": self.sleep_reports_table,
            "patient_info": self.patient_info_table,
            "eeg_info": self.eeg_info_table,
            "schema": self.schema_name,
        }

    def log_environment_info(self):
        """Log the current environment configuration."""
        print(f"Databricks environment: {self.environment}")
        print(f"Schema: {self.schema_name}")
        print(f"Seizure events table: {self.seizure_events_table}")
        print(f"Seizure annotations table: {self.seizure_annotations_table}")
        print(f"Spike events table: {self.spike_events_table}")
        print(f"Spike annotations table: {self.spike_annotations_table}")
        print(f"Sleep reports table: {self.sleep_reports_table}")
        print(f"EEG info table: {self.eeg_info_table}")
        print(f"Patient info table: {self.patient_info_table}")
        print(f"Volume path: {self.volume_path}")

    def get_db_uri(self) -> str:
        """Construct the full Databricks URI for the current environment."""
        api_token = os.getenv("DATABRICKS_ACCESS_TOKEN")
        host = os.getenv("DATABRICKS_SERVER_HOSTNAME")
        http_path = os.getenv("DATABRICKS_HTTP_PATH")

        if not all([api_token, host, http_path]):
            raise ValueError("Databricks connection details are not fully configured.")

        catalog, schema = self.schema_name.split(".")

        uri = (
            f"databricks://token:{api_token}@{host}?"
            f"http_path={http_path}&catalog={catalog}&schema={schema}"
        )
        return uri


# Global configuration instance
databricks_config = DatabricksEnvironmentConfig()


def get_databricks_connection_params() -> Dict[str, any]:
    """Get connection parameters for Databricks SQL Connector."""
    return databricks_config.get_connection_params()


def get_databricks_db_uri() -> str:
    """Get the full Databricks URI for the current environment."""
    return databricks_config.get_db_uri()


def get_databricks_table_names() -> Dict[str, str]:
    """
    Get Databricks table names for the current environment.

    Returns:
        Dictionary containing table names for the current environment
    """
    return databricks_config.get_table_names()


def get_databricks_environment() -> str:
    """
    Get the current Databricks environment.

    Returns:
        The current environment name ('dev' or 'prod')
    """
    return databricks_config.environment


def get_databricks_volume_path() -> str:
    """
    Get the Databricks volume path for the current environment.

    Returns:
        The volume path for the current environment
    """
    return databricks_config.volume_path


def get_databricks_category_path(category_name: str) -> str:
    """
    Get the correct category path for the current environment.

    Args:
        category_name: The category name (e.g., 'sleep_polar', 'sleep_staging', 'seizure_plots')

    Returns:
        The correct path for the category in the current environment.
        Note: Both dev and prod now use the same directory structure:
        - sleep_polar -> sleep/polar
        - sleep_staging -> sleep/staging
        - seizure_plots -> seizure/heatmap
    """
    return databricks_config.get_category_path(category_name)


def get_databricks_sleep_reports_table() -> str:
    """
    Get the sleep reports table name for the current environment.

    Returns:
        The sleep reports table name for the current environment
    """
    return databricks_config.sleep_reports_table


def get_databricks_seizure_events_table() -> str:
    """
    Get the seizure events table name for the current environment.

    Returns:
        The seizure events table name for the current environment
    """
    return databricks_config.seizure_events_table


def get_databricks_seizure_annotations_table() -> str:
    """
    Get the seizure annotations table name for the current environment.

    Returns:
        The seizure annotations table name for the current environment
    """
    return databricks_config.seizure_annotations_table


def get_databricks_eeg_info_table() -> str:
    """
    Get the eeg_info table name for the current environment.

    Returns:
        The eeg_info table name for the current environment
    """
    return databricks_config.eeg_info_table


def get_databricks_spike_events_table() -> str:
    """
    Get the spike events table name for the current environment.

    Returns:
        The spike events table name for the current environment
    """
    return databricks_config.spike_events_table


def get_databricks_spike_annotations_table() -> str:
    """
    Get the spike annotations table name for the current environment.

    Returns:
        The spike annotations table name for the current environment
    """
    return databricks_config.spike_annotations_table


def get_databricks_sparcnet_annotations_table() -> str:
    """
    Get the sparcnet annotations table name for the current environment.

    Returns:
        The sparcnet annotations table name for the current environment
    """
    return databricks_config.sparcnet_annotations_table


def get_databricks_patient_info_table() -> str:
    """
    Get the patient info table name for the current environment.
    Returns:
        The patient info table name for the current environment
    """
    return databricks_config.patient_info_table


def log_databricks_environment():
    """Log the current Databricks environment configuration."""
    databricks_config.log_environment_info()
