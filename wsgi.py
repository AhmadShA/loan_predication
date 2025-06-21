import os
from eda import eda

if __name__ == "__main__":
  port = int(os.environ.get("PORT", 4000))
  eda.run(host="0.0.0.0", port=port)
