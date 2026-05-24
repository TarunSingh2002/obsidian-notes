## Basics
- Logging = your program writing a **diary** of everything that happens while it runs
- Timestamps on every message
- 5 Levels: DEBUG, INFO, WARNING, ERROR, CRITICAL
- When you set a level, you only see that level **and above**. Below it gets ignored. For example, Debug level will show all 5 messages but WARNING level will only show WARNING, ERROR, CRITICAL logs not the DEBUG, INFO logs
- In development → set DEBUG (see everything). In production → set WARNING (only see problems).
- Save to file AND terminal together
- Turn off debug logs with one line
- Shows which file/function logged it
```bash
# What a log message looks like
# Format: timestamp - logger_name - level - your message
2024-01-15 10:23:45 - data_ingestion - DEBUG - Loading file train.csv
2024-01-15 10:23:46 - data_ingestion - INFO  - File loaded. 5000 rows found
2024-01-15 10:23:47 - data_ingestion - ERROR - Column 'age' is missing!
```
- 3 Components
	- Logger = We talk to logger
	- Handler = Decided where to send logs terminal/file ( StreamHandler / FileHandler )
	- Formatter = Decides how it looks (timestamp - logger_name - level - your message)

## Complete cheat sheet

```Python
import logging, os

## SETUP (do this once at the top of your project)
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger('data_ingestion')   # create logger
logger.setLevel('DEBUG')                   # min level for logger

# Console handler
ch = logging.StreamHandler()
ch.setLevel('DEBUG')

# File handler
log_file_path = os.path.join(log_dir, 'data_ingestion.log')
fh = logging.FileHandler(log_file_path)
fh.setLevel('DEBUG')

# Formatter
fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(fmt)
fh.setFormatter(fmt)

# Add handlers
logger.addHandler(ch)
logger.addHandler(fh)

## USAGE (use anywhere after setup)
logger.debug("detailed trace info")
logger.info("normal update")
logger.warning("something odd")
logger.error("something broke")
logger.critical("system dying")
logger.exception("error with traceback")  # inside except block
```

