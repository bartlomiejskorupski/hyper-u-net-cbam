import logging
from datetime import datetime

def setup_logging(name: str, filename: str) -> logging.Logger:
  logger = logging.getLogger(name)
  timestamp = datetime.today().strftime('%Y%m%d%H%M%S')
  formatter = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S")
  fh = logging.FileHandler(f'./logs/{filename}_{timestamp}.log', encoding='utf-8')
  sh = logging.StreamHandler()
  fh.setFormatter(formatter)
  sh.setFormatter(formatter)
  logger.addHandler(fh)
  logger.addHandler(sh)
  logger.setLevel(logging.INFO)
  return logger