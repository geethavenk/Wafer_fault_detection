from datetime import datetime
from dataclasses import dataclass

@dataclass
class Logger:
    """
    A simple logger class for writing log messages with timestamps to a file.
    """
    log_file: str = 'logfile.txt'
    timestamp_format: str = '%Y-%m-%d %H:%M:%S'

    def log(self, log_message, log_level='INFO'):
        """
        Logs a message to the specified log file.

        Args:
        log_message: str
            The log message to be written to the log file.
        log_level: str, optional (default = 'INFO')
            The level of the log message e.g., 'INFO', 'WARNING', 'ERROR'    
        """

        try:
            with open(self.log_file, 'a') as file_object:
                now = datetime.now()
                formatted_time = now.strftime(self.timestamp_format)
                file_object.write(f'{formatted_time}\t\t{log_level}\t\t{log_message}\n')
         
        except Exception as e:
            print(f'Error logging message: {str(e)}')    
